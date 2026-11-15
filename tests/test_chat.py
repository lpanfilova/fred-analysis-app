import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from openai import APIConnectionError
from pydantic import BaseModel

from chat import ChatError, EconomicChat, load_chat_settings, run_terminal_chat
from chat_tools import EconomicTools, tool_definitions


class Item(BaseModel):
    type: str
    name: str | None = None
    arguments: str | None = None
    call_id: str | None = None
    role: str | None = None
    content: list | None = None


def function_call(name="analyze_economic_series", arguments=None, call_id="call-1"):
    if arguments is None:
        arguments = {"series_id": "GDP", "start": None, "end": None, "years": None, "include_history": False}
    return Item(type="function_call", name=name, arguments=json.dumps(arguments), call_id=call_id)


def response(*calls, text="", status="completed"):
    output = list(calls) if calls else [Item(type="message", role="assistant", content=[{"type": "output_text", "text": text}])]
    return SimpleNamespace(output=output, output_text=text, status=status)


class ChatTests(unittest.IsolatedAsyncioTestCase):
    def make_chat(self, responses, **kwargs):
        client = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(side_effect=responses)))
        tools = SimpleNamespace(execute=AsyncMock(return_value={"latest": {"level": 100}, "source_url": "https://fred.stlouisfed.org/series/GDP"}))
        return EconomicChat(client, tools, **kwargs), client, tools

    async def test_function_call_receives_result_and_final_answer(self):
        chat, client, tools = self.make_chat([response(function_call()), response(text="Latest GDP is 100.")])
        snapshots = []
        responses = iter([response(function_call()), response(text="Latest GDP is 100.")])
        async def create(**kwargs):
            snapshots.append(copy.deepcopy(kwargs))
            return next(responses)
        client.responses.create.side_effect = create
        self.assertEqual(await chat.ask("What is the latest GDP?"), "Latest GDP is 100.")
        tools.execute.assert_awaited_once()
        self.assertEqual(snapshots[1]["input"][-1]["type"], "function_call_output")
        self.assertEqual(snapshots[1]["input"][-1]["call_id"], "call-1")
        self.assertEqual(json.loads(snapshots[1]["input"][-1]["output"])["latest"]["level"], 100)
        self.assertFalse(snapshots[0]["store"])

    async def test_follow_up_includes_prior_tool_and_assistant_messages(self):
        chat, client, tools = self.make_chat([
            response(function_call()), response(text="GDP answer"),
            response(function_call(arguments={"series_id": "GDP", "years": 5})), response(text="Five-year answer"),
        ])
        await chat.ask("Show GDP over ten years")
        prior = copy.deepcopy(chat.history)
        await chat.ask("Now only five years")
        self.assertEqual(chat.history[:len(prior)], prior)
        self.assertIn({"role": "user", "content": "Now only five years"}, chat.history)
        self.assertEqual(tools.execute.await_args_list[-1].args[1]["years"], 5)

    async def test_clarification_can_finish_without_any_tool_call(self):
        chat, _, tools = self.make_chat([response(text="Which series and time period?")])
        self.assertIn("Which", await chat.ask("How much did it change?"))
        tools.execute.assert_not_awaited()

    async def test_multiple_calls_each_receive_the_matching_output(self):
        chat, _, tools = self.make_chat([
            response(function_call(call_id="a"), function_call(call_id="b")), response(text="Both trends"),
        ])
        await chat.ask("Summarize both datasets")
        self.assertEqual(tools.execute.await_count, 2)
        outputs = [item for item in chat.history if item.get("type") == "function_call_output"]
        self.assertEqual([item["call_id"] for item in outputs], ["a", "b"])

    async def test_malformed_json_is_returned_as_a_tool_error(self):
        bad_call = function_call()
        bad_call.arguments = "{invalid"
        chat, _, tools = self.make_chat([response(bad_call), response(text="Please specify a valid period.")])
        await chat.ask("Show GDP")
        tools.execute.assert_not_awaited()
        self.assertIn("error", chat.last_tool_results[0]["result"])

    async def test_api_error_does_not_expose_exception_details_or_commit_partial_turn(self):
        chat, _, _ = self.make_chat([APIConnectionError(request=SimpleNamespace(), message="secret-token")])
        with self.assertRaises(ChatError) as raised:
            await chat.ask("GDP?")
        self.assertNotIn("secret-token", str(raised.exception))
        self.assertEqual(chat.history, [])

    async def test_incomplete_and_empty_answers_do_not_commit_history(self):
        for result in (response(text="partial", status="incomplete"), response(text="")):
            with self.subTest(result=result):
                chat, _, _ = self.make_chat([result])
                with self.assertRaises(ChatError):
                    await chat.ask("GDP?")
                self.assertEqual(chat.history, [])

    async def test_tool_loop_is_bounded(self):
        chat, client, tools = self.make_chat([response(function_call()), response(function_call())], max_tool_rounds=1)
        with self.assertRaisesRegex(ChatError, "limit"):
            await chat.ask("GDP?")
        self.assertEqual(tools.execute.await_count, 1)
        self.assertEqual(client.responses.create.await_count, 2)
        self.assertEqual(client.responses.create.await_args.kwargs["tool_choice"], "none")
        self.assertEqual(chat.history, [])

    async def test_reset_clears_follow_up_context(self):
        chat, _, _ = self.make_chat([response(text="Which series?")])
        await chat.ask("Hi")
        chat.reset()
        self.assertEqual(chat.history, [])
        self.assertEqual(chat.last_tool_results, [])

    async def test_terminal_accepts_reset_and_exit(self):
        api = AsyncMock()
        api.__aenter__.return_value = SimpleNamespace()
        with patch("chat.AsyncOpenAI", return_value=api), patch("builtins.input", side_effect=["/reset", "/exit"]), patch("builtins.print") as output:
            await run_terminal_chat(object(), "test-key", "test-model")
        self.assertTrue(any("cleared" in str(call) for call in output.call_args_list))


class ToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_series_tool_forwards_validated_bounds_and_omits_unrequested_history(self):
        tools = EconomicTools(object())
        with patch("chat_tools.get_series_analysis", new_callable=AsyncMock, return_value={"latest": 1, "history": [1, 2]}) as analyze:
            report = await tools.execute("analyze_economic_series", {"series_id": "GDP", "years": 10})
        self.assertNotIn("history", report)
        analyze.assert_awaited_once_with(tools.fred_client, "GDP", start=None, end=None, years=10)

    async def test_history_is_returned_when_requested(self):
        tools = EconomicTools(object())
        with patch("chat_tools.get_series_analysis", new_callable=AsyncMock, return_value={"history": [1, 2]}):
            report = await tools.execute("analyze_economic_series", {"series_id": "CPILFESL", "years": 5, "include_history": True})
        self.assertEqual(report["history"], [1, 2])

    async def test_invalid_arguments_do_not_call_fred(self):
        invalid = [{"series_id": "UNKNOWN"}, {"series_id": "GDP", "years": "5"},
                   {"series_id": "GDP", "years": True}, {"series_id": "GDP", "years": 0},
                   {"series_id": "GDP", "start": "bad"},
                   {"series_id": "GDP", "start": "2020-01-01", "years": 5},
                   {"series_id": "GDP", "end": "2020-01-01", "start": "2021-01-01"},
                   {"series_id": "GDP", "path": "/tmp/unwanted"}, [], None]
        tools = EconomicTools(object())
        with patch("chat_tools.get_series_analysis", new_callable=AsyncMock) as analyze:
            for arguments in invalid:
                with self.subTest(arguments=arguments):
                    self.assertIn("error", await tools.execute("analyze_economic_series", arguments))
            analyze.assert_not_awaited()

    async def test_unknown_tool_cannot_execute_arbitrary_code(self):
        self.assertIn("error", await EconomicTools(object()).execute("eval", {"code": "bad"}))

    async def test_transport_errors_are_sanitized(self):
        with patch("chat_tools.get_comparison_analysis", new_callable=AsyncMock, side_effect=RuntimeError("secret-token")):
            result = await EconomicTools(object()).execute("compare_economic_series", {})
        self.assertIn("error", result)
        self.assertNotIn("secret-token", json.dumps(result))

    async def test_chart_uses_generated_path_under_outputs(self):
        tools = EconomicTools(object(), output_dir="outputs")
        report = {"history": [1], "summary": {}}
        with patch("chat_tools.get_comparison_analysis", new_callable=AsyncMock, return_value=report), patch("visualization.save_comparison_chart", return_value=Path("outputs/chart.svg")) as save:
            result = await tools.execute("create_comparison_chart", {"years": 5, "format": "svg"})
        destination = save.call_args.args[1]
        self.assertEqual(destination.parent, Path("outputs"))
        self.assertEqual(destination.suffix, ".svg")
        self.assertIn("chart_path", result)
        self.assertNotIn("history", result)


class ConfigurationTests(unittest.TestCase):
    def test_user_key_spelling_and_model_override(self):
        with patch("chat.load_dotenv"), patch.dict(os.environ, {"OPEN_AI_API_KEY": "test-key", "OPENAI_MODEL": "configured-model"}, clear=True):
            self.assertEqual(load_chat_settings(), ("test-key", "configured-model"))
            self.assertEqual(load_chat_settings("explicit-model"), ("test-key", "explicit-model"))

    def test_missing_key_has_actionable_message(self):
        with patch("chat.load_dotenv"), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ChatError, "OPEN_AI_API_KEY"):
                load_chat_settings()

    def test_tool_schemas_use_strict_objects_with_all_fields_required(self):
        for tool in tool_definitions():
            schema = tool["parameters"]
            self.assertTrue(tool["strict"])
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(set(schema["required"]), set(schema["properties"]))

    def test_chat_cli_is_separate_from_batch_flags(self):
        from mcp_client import parse_args
        with patch("sys.argv", ["mcp_client.py", "--chat", "--model", "test-model"]):
            args = parse_args()
        self.assertTrue(args.chat)
        self.assertEqual(args.model, "test-model")
        for flags in (["--chat", "--compare"], ["--chat", "--history"], ["--model", "test-model"]):
            with self.subTest(flags=flags), patch("sys.argv", ["mcp_client.py", *flags]), patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    parse_args()


if __name__ == "__main__":
    unittest.main()
