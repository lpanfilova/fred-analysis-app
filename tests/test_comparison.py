import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import pandas as pd

from comparison import CPI_RATE, GDP_RATE, align_quarterly, compare_series
from fred_service import get_comparison_analysis


def histories():
    # Identical quarterly paths produce identical YoY rates and correlation 1.
    gdp, cpi = [], []
    for i, start in enumerate(pd.date_range("2018-01-01", "2024-10-01", freq="QS-JAN")):
        value = 100 + i * i + i
        gdp.append({"date": start.date().isoformat(), "value": str(value)})
        for month in pd.date_range(start, periods=3, freq="MS"):
            cpi.append({"date": month.date().isoformat(), "value": str(value)})
    return {"observations": gdp}, {"observations": cpi}


class ComparisonTests(unittest.TestCase):
    def test_aggregation_uses_index_mean_before_calculating_growth(self):
        gdp = {"observations": [{"date": "2020-01-01", "value": "100"},
                                 {"date": "2021-01-01", "value": "110"}]}
        cpi = {"observations": [
            {"date": f"{year}-{month:02d}-01", "value": str(value)}
            for year, values in ((2020, [100, 200, 300]), (2021, [110, 240, 390]))
            for month, value in enumerate(values, 1)
        ]}
        frame = align_quarterly(gdp, cpi)
        self.assertAlmostEqual(frame.loc["2021-01-01", CPI_RATE], ((740 / 3) / 200 - 1) * 100)
        self.assertNotAlmostEqual(frame.loc["2021-01-01", CPI_RATE], 20)

    def test_missing_month_invalidates_current_and_year_later_quarters(self):
        gdp, cpi = histories()
        cpi["observations"] = [r for r in cpi["observations"] if r["date"] != "2020-02-01"]
        frame = align_quarterly(gdp, cpi)
        self.assertEqual(frame.loc["2020-01-01", "core_cpi_months"], 2)
        self.assertTrue(pd.isna(frame.loc["2020-01-01", CPI_RATE]))
        self.assertTrue(pd.isna(frame.loc["2021-01-01", CPI_RATE]))
        self.assertTrue(pd.notna(frame.loc["2020-04-01", CPI_RATE]))

    def test_incomplete_latest_quarter_is_not_treated_as_complete(self):
        gdp, cpi = histories()
        cpi["observations"] = cpi["observations"][:-1]
        report = compare_series(gdp, cpi, start="2024-01-01", end="2024-12-31")
        self.assertEqual(report["latest_common"]["quarter"], "2024Q3")
        self.assertIsNone(report["history"][-1][CPI_RATE])
        self.assertEqual(report["relationship"]["excluded_quarters"], 1)

    def test_gdp_gap_and_missing_cpi_baseline_are_not_filled(self):
        gdp, cpi = histories()
        gdp["observations"] = [r for r in gdp["observations"] if r["date"] != "2020-01-01"]
        frame = align_quarterly(gdp, cpi)
        self.assertTrue(pd.isna(frame.loc["2020-01-01", GDP_RATE]))
        self.assertTrue(pd.isna(frame.loc["2021-01-01", GDP_RATE]))

    def test_date_filter_preserves_warmup_and_returns_known_correlation(self):
        report = compare_series(*histories(), start="2020-01-01", end="2022-12-31")
        self.assertEqual(len(report["history"]), 12)
        self.assertIsNotNone(report["history"][0][CPI_RATE])
        self.assertEqual(report["relationship"]["paired_quarters"], 12)
        self.assertAlmostEqual(report["relationship"]["pearson_correlation"], 1)
        self.assertEqual(report["relationship"]["first_paired_quarter"], "2020Q1")
        self.assertEqual(report["relationship"]["last_paired_quarter"], "2022Q4")
        json.dumps(report, allow_nan=False)

    def test_bounds_select_only_whole_quarters(self):
        report = compare_series(*histories(), start="2020-02-01", end="2020-11-01")
        self.assertEqual([row["quarter"] for row in report["history"]], ["2020Q2", "2020Q3"])
        self.assertIsNone(report["relationship"]["pearson_correlation"])

    def test_rolling_window_ends_at_latest_pair_and_has_exact_quarter_count(self):
        gdp, cpi = histories()
        cpi["observations"] = cpi["observations"][:-1]
        report = compare_series(gdp, cpi, years=3)
        self.assertEqual(len(report["history"]), 12)
        self.assertEqual(report["history"][-1]["quarter"], "2024Q3")
        self.assertEqual(report["history"][0]["quarter"], "2021Q4")

    def test_constant_series_correlation_is_null(self):
        gdp, cpi = histories()
        for row in cpi["observations"]:
            row["value"] = "100"
        report = compare_series(gdp, cpi, start="2020-01-01", end="2022-12-31")
        self.assertIsNone(report["relationship"]["pearson_correlation"])
        self.assertTrue(any("must vary" in w for w in report["warnings"]))
        json.dumps(report, allow_nan=False)

    def test_sparse_pairs_are_excluded_from_correlation_but_kept_as_gaps(self):
        gdp, cpi = histories()
        cpi["observations"][24]["value"] = "."  # 2020 Q1 and 2021 Q1 unavailable.
        report = compare_series(gdp, cpi, start="2020-01-01", end="2022-12-31")
        self.assertEqual(len(report["history"]), 12)
        self.assertEqual(report["relationship"]["paired_quarters"], 10)
        self.assertEqual(report["relationship"]["excluded_quarters"], 2)
        self.assertAlmostEqual(report["relationship"]["pearson_correlation"], 1)

    def test_invalid_ranges_and_insufficient_history_fail_clearly(self):
        for bounds in ({"years": 0}, {"years": True}, {"years": 1, "start": "2020-01-01"},
                       {"start": "2021-01-01", "end": "2020-01-01"},
                       {"start": "2030-01-01"}, {"end": "2018-03-31"},
                       {"start": "2020-02-01", "end": "2020-03-31"}):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                compare_series(*histories(), **bounds)
        with self.assertRaisesRegex(ValueError, "usable"):
            compare_series({"observations": []}, histories()[1])


class ComparisonServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_mcp_pair_shape_passes_through_full_comparison_pipeline(self):
        payloads = histories()
        for payload in payloads:
            payload["observations"] = [[row["date"], row["value"]] for row in reversed(payload["observations"])]
        client = SimpleNamespace(call_tool=AsyncMock(side_effect=[
            SimpleNamespace(is_error=False, structured_content=payload) for payload in payloads
        ]))
        report = await get_comparison_analysis(client, start="2020-01-01", end="2022-12-31")
        self.assertAlmostEqual(report["relationship"]["pearson_correlation"], 1)
        self.assertEqual([call.kwargs["arguments"]["series_id"] for call in client.call_tool.await_args_list], ["GDP", "CPILFESL"])


class VisualizationTests(unittest.TestCase):
    def test_chart_preserves_gaps_and_can_export_png_svg_and_pdf(self):
        from visualization import create_comparison_figure, save_comparison_chart
        gdp, cpi = histories()
        cpi["observations"][24]["value"] = "."
        report = compare_series(gdp, cpi, start="2020-01-01", end="2022-12-31")
        figure = create_comparison_figure(report)
        self.assertTrue(pd.isna(figure.axes[0].lines[1].get_ydata()[0]))
        self.assertEqual(len(figure.axes[1].collections[0].get_offsets()), 10)
        figure.clear()
        with tempfile.TemporaryDirectory() as directory:
            for extension, signature in (("png", b"\x89PNG"), ("svg", b"<?xml"), ("pdf", b"%PDF")):
                destination = Path(directory) / "outputs" / extension / f"comparison.{extension}"
                self.assertEqual(save_comparison_chart(report, destination), destination.resolve())
                self.assertTrue(destination.read_bytes().startswith(signature))
                self.assertGreater(destination.stat().st_size, 1000)

    def test_unknown_export_format_is_rejected(self):
        from visualization import save_comparison_chart
        with self.assertRaisesRegex(ValueError, "extension"):
            save_comparison_chart({}, "comparison.html")


class ComparisonCliTests(unittest.TestCase):
    def test_chart_implies_comparison_and_accepts_custom_window(self):
        from mcp_client import parse_args
        with patch("sys.argv", ["mcp_client.py", "--chart", "output.svg", "--years", "3"]):
            args = parse_args()
        self.assertEqual(args.years, 3)
        self.assertEqual(args.chart, Path("output.svg"))

    def test_invalid_cli_combinations_fail_before_authentication(self):
        from mcp_client import parse_args
        for flags in (["--years", "3"], ["--compare", "--years", "0"],
                      ["--compare", "--years", "3", "--start", "2020-01-01"],
                      ["--chart", "output.html"]):
            with self.subTest(flags=flags), patch("sys.argv", ["mcp_client.py", *flags]), patch("sys.stderr"):
                with self.assertRaises(SystemExit) as raised:
                    parse_args()
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
