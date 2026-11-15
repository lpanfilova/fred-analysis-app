import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from fred_service import get_metadata, get_observations, get_series_analysis


def response(payload=None, *, error=False, text=None):
    return SimpleNamespace(
        is_error=error, structured_content=payload,
        content=[SimpleNamespace(type="text", text=text)] if text is not None else [],
    )


class FredServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_official_mcp_pairs_are_normalized_with_metadata_preserved(self):
        # The live official connector returns newest-first [date, value] pairs.
        payload = {
            "series_id": "GDP", "raw_series_frequency": "Quarterly",
            "realtime_start": "2026-11-14", "realtime_end": "2026-11-14",
            "observations": [["2021-04-01", "115"], ["2021-01-01", "110"],
                             ["2020-04-01", "."], ["2020-01-01", "100"]],
        }
        for result in (response(payload), response(text=json.dumps(payload))):
            with self.subTest(result=result):
                client = SimpleNamespace(call_tool=AsyncMock(return_value=result))
                normalized = await get_observations(client, "GDP")
                self.assertEqual(normalized["observations"][0], {"date": "2021-04-01", "value": "115"})
                self.assertEqual(normalized["realtime_start"], "2026-11-14")
                self.assertIsInstance(payload["observations"][0], list)
                report = await get_series_analysis(client, "GDP", start="2021-01-01", end="2021-04-01")
                self.assertAlmostEqual(report["history"][0]["yoy_percent"], 10)
                self.assertIsNone(report["history"][1]["yoy_percent"])
                self.assertEqual(report["latest"]["level"], 115)

    async def test_official_mcp_monthly_pairs_calculate_core_inflation(self):
        payload = {"series_id": "CPILFESL", "observations": [
            ["2021-01-01", "103"], ["2020-01-01", "100"],
        ]}
        client = SimpleNamespace(call_tool=AsyncMock(return_value=response(payload)))
        report = await get_series_analysis(client, "CPILFESL", start="2021-01-01")
        self.assertAlmostEqual(report["latest"]["yoy_percent"], 3)

    async def test_structured_and_text_observations(self):
        payload = {"observations": [{"date": "2020-01-01", "value": "100"}], "count": "1", "offset": "0"}
        for result in (response(payload), response(text=json.dumps(payload))):
            with self.subTest(result=result):
                client = SimpleNamespace(call_tool=AsyncMock(return_value=result))
                self.assertEqual(await get_observations(client, "GDP"), payload)
                client.call_tool.assert_awaited_once_with("fred-get-series-observations", arguments={"series_id": "GDP"})

    async def test_tool_and_payload_errors_are_reported(self):
        for result in (response(error=True), response({"error_code": 400}), response({"error": "failed"})):
            client = SimpleNamespace(call_tool=AsyncMock(return_value=result))
            with self.subTest(result=result), self.assertRaises(RuntimeError):
                await get_observations(client, "GDP")

    async def test_malformed_or_partial_results_are_rejected(self):
        bad_results = [
            response(text="not JSON"), response([]), response({}),
            response({"observations": "wrong"}), response({"observations": [{}]}),
            response({"observations": [["2020-01-01"]]}),
            response({"observations": [["2020-01-01", "100", "extra"]]}),
            response({"observations": ["2020-01-01"]}),
            response({"observations": [], "count": 10}),
            response({"observations": [], "offset": 1}),
            response({"observations": [], "count": "invalid"}),
        ]
        for result in bad_results:
            client = SimpleNamespace(call_tool=AsyncMock(return_value=result))
            with self.subTest(result=result), self.assertRaises(ValueError):
                await get_observations(client, "GDP")

    async def test_metadata_uses_shared_response_validation(self):
        payload = {"seriess": [{"id": "GDP", "frequency": "Quarterly"}]}
        client = SimpleNamespace(call_tool=AsyncMock(return_value=response(payload)))
        self.assertEqual(await get_metadata(client, "GDP"), payload)
        client.call_tool.assert_awaited_once_with("fred-get-series-metadata", arguments={"series_id": "GDP"})

    async def test_fetch_and_analyze_retains_warmup_history(self):
        payload = {"observations": [
            {"date": "2020-01-01", "value": "100"},
            {"date": "2021-01-01", "value": "110"},
        ]}
        client = SimpleNamespace(call_tool=AsyncMock(return_value=response(payload)))
        report = await get_series_analysis(client, "GDP", start="2021-01-01", end="2021-01-01")
        self.assertEqual(len(report["history"]), 1)
        self.assertAlmostEqual(report["latest"]["yoy_percent"], 10)

    async def test_unknown_series_fails_before_request(self):
        client = SimpleNamespace(call_tool=AsyncMock())
        with self.assertRaises(ValueError):
            await get_series_analysis(client, "UNKNOWN")
        client.call_tool.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
