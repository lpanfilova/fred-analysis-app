from contextlib import asynccontextmanager
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from streamlit.testing.v1 import AppTest

from chat import ChatError
from web_backend import WebTurn, answer_question, chart_files, data_notes, display_answer


APP = Path(__file__).resolve().parents[1] / "app.py"


class WebBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_turn_gets_fresh_async_clients_and_preserves_history(self):
        clients = []
        @asynccontextmanager
        async def session(**kwargs):
            client = object()
            clients.append(client)
            yield client

        api = AsyncMock()
        api.__aenter__.return_value = object()
        agent = SimpleNamespace(history=[], last_tool_results=[], ask=AsyncMock(return_value="An answer"))
        history = [{"role": "user", "content": "GDP?"}]
        with patch("web_backend.load_chat_settings", return_value=("test-key", "test-model")), \
             patch("web_backend.AsyncOpenAI", return_value=api) as factory, \
             patch("web_backend.fred_session", session), \
             patch("web_backend.EconomicChat", return_value=agent):
            first = await answer_question("What about inflation?", history)
            second = await answer_question("And five years?", first.history)
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(len(clients), 2)
        self.assertIsNot(clients[0], clients[1])
        self.assertIsNot(agent.history, history)
        self.assertEqual(second.answer, "An answer")
        self.assertEqual(history, [{"role": "user", "content": "GDP?"}])

    async def test_nested_model_error_has_readable_message_without_transport_details(self):
        api = AsyncMock()
        api.__aenter__.side_effect = ExceptionGroup("transport-secret", [ChatError("Check API credits.")])
        with patch("web_backend.load_chat_settings", return_value=("test-key", "test-model")), patch("web_backend.AsyncOpenAI", return_value=api):
            with self.assertRaises(ChatError) as raised:
                await answer_question("GDP?", [])
        self.assertEqual(str(raised.exception), "Check API credits.")

    async def test_unknown_connection_errors_are_sanitized(self):
        api = AsyncMock()
        api.__aenter__.side_effect = RuntimeError("secret-token")
        with patch("web_backend.load_chat_settings", return_value=("test-key", "test-model")), patch("web_backend.AsyncOpenAI", return_value=api):
            with self.assertRaises(ChatError) as raised:
                await answer_question("GDP?", [])
        self.assertNotIn("secret-token", str(raised.exception))


class WebPresentationTests(unittest.TestCase):
    def test_chart_files_allow_outputs_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs"
            output.mkdir()
            chart = output / "chart.png"
            chart.write_bytes(b"image")
            private_file = root / "private.txt"
            private_file.write_text("private")
            with patch("web_backend.OUTPUT_DIR", output):
                results = [{"result": {"chart_path": str(path)}} for path in (private_file, chart, chart)]
                self.assertEqual(chart_files(results), [chart.resolve()])

    def test_local_links_are_replaced_and_notes_keep_sources_and_gaps(self):
        path = "/local/outputs/chart.png"
        results = [{"result": {
            "chart_path": path, "sources": ["https://fred.stlouisfed.org/series/GDP"],
            "warnings": ["One quarter unavailable."], "latest_common": {"quarter": "2025Q3"},
        }}]
        self.assertNotIn(path, display_answer(f"[Chart]({path})", results))
        notes = data_notes(results)
        self.assertEqual(notes["periods"], ["2025Q3"])
        self.assertEqual(notes["warnings"], ["One quarter unavailable."])
        self.assertEqual(len(notes["sources"]), 1)


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.platform = patch("platform.system", return_value="Darwin").start()
        self.settings = patch("chat.load_chat_settings", return_value=("test-key", "test-model")).start()
        self.answer = patch("web_backend.answer_question", new_callable=AsyncMock).start()
        self.answer.return_value = WebTurn(
            "GDP answer with a source.", [{"role": "assistant", "content": "GDP answer"}],
            [{"result": {"source_url": "https://fred.stlouisfed.org/series/GDP",
                         "latest": {"observation_period": "2026Q3"}, "warnings": []}}],
        )
        self.addCleanup(patch.stopall)

    def create_app(self):
        app = AppTest.from_file(str(APP), default_timeout=10).run()
        self.assertFalse(app.exception)
        return app

    def test_initial_view_has_starters_and_enabled_chat(self):
        app = self.create_app()
        self.assertEqual(app.title[0].value, "Economic Explorer")
        self.assertFalse(app.chat_input[0].disabled)
        self.assertIn("Compare & chart", [button.label for button in app.button])
        self.answer.assert_not_awaited()

    def test_missing_key_disables_chat_without_calling_services(self):
        self.settings.side_effect = ChatError("Set OPEN_AI_API_KEY in .env.")
        app = self.create_app()
        self.assertTrue(app.chat_input[0].disabled)
        self.assertIn("OPEN_AI_API_KEY", app.warning[0].value)
        self.answer.assert_not_awaited()

    def test_non_macos_shows_scope_requirement(self):
        self.platform.return_value = "Linux"
        app = self.create_app()
        self.assertTrue(app.chat_input[0].disabled)
        self.assertIn("macOS", app.warning[0].value)

    def test_chat_follow_up_preserves_history_and_new_conversation_clears_it(self):
        app = self.create_app()
        app.chat_input[0].set_value("What is the latest GDP?").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.chat_message), 2)
        self.assertEqual(len(app.expander), 1)
        previous = list(app.session_state.conversation)
        app.chat_input[0].set_value("Now show five years").run()
        self.assertFalse(app.exception)
        self.assertEqual(self.answer.await_args.args[1], previous)
        self.assertEqual(len(app.session_state.turns), 2)
        next(button for button in app.button if button.label == "New conversation").click().run()
        self.assertEqual(app.session_state.conversation, [])
        self.assertEqual(app.session_state.turns, [])

    def test_failed_request_can_retry_without_duplicate_user_turn_or_lost_context(self):
        self.answer.side_effect = [ChatError("Connection failed."), self.answer.return_value]
        app = self.create_app()
        app.chat_input[0].set_value("GDP?").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.error[0].value, "Connection failed.")
        self.assertEqual(app.session_state.conversation, [])
        next(button for button in app.button if button.label == "Retry request").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state.turns), 1)
        self.assertNotIn("error", app.session_state.turns[0])

    def test_starter_submits_a_real_question(self):
        app = self.create_app()
        next(button for button in app.button if button.label == "Latest GDP").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(self.answer.await_args.args[0], "What is the latest reported U.S. GDP?")

    def test_interrupted_request_can_be_retried(self):
        app = self.create_app()
        app.session_state.turns = [{"question": "GDP?"}]
        app.run()
        self.assertFalse(app.exception)
        self.assertIn("interrupted", app.error[0].value)
        next(button for button in app.button if button.label == "Retry request").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state.turns), 1)
        self.assertIn("answer", app.session_state.turns[0])

    def test_chart_is_inline_and_downloadable(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chart = root / "chart.png"
            Image.new("RGB", (20, 20), "white").save(chart)
            self.answer.return_value = WebTurn(
                f"[Chart]({chart})", [], [{"result": {"chart_path": str(chart)}}]
            )
            with patch("web_backend.OUTPUT_DIR", root):
                app = self.create_app()
                app.chat_input[0].set_value("Make a comparison chart").run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.get("image")), 1)
                self.assertEqual(len(app.download_button), 1)


if __name__ == "__main__":
    unittest.main()
