import json
import unittest

import pandas as pd

from analytics import analyze_series, calculate_gdp_qoq_annualized, calculate_yoy, prepare_data


def observations(start, end, frequency, base=100):
    return {"observations": [
        {"date": d.date().isoformat(), "value": str(base + i)}
        for i, d in enumerate(pd.date_range(start, end, freq=frequency))
    ]}


class AnalyticsTests(unittest.TestCase):
    def test_quarterly_gap_does_not_shift_year_over_year_baseline(self):
        data = observations("2020-01-01", "2021-04-01", "QS-JAN")
        del data["observations"][1]  # Missing 2020 Q2.
        levels = prepare_data(data, "QS-JAN")
        growth = calculate_yoy(levels, 4)
        self.assertEqual(len(levels), 6)
        self.assertTrue(pd.isna(levels.loc["2020-04-01"]))
        self.assertAlmostEqual(growth.loc["2021-01-01"], 4)
        self.assertTrue(pd.isna(growth.loc["2021-04-01"]))

    def test_monthly_gap_does_not_shift_inflation_baseline(self):
        data = observations("2020-01-01", "2021-02-01", "MS")
        del data["observations"][1]
        levels = prepare_data(data, "MS")
        growth = calculate_yoy(levels, 12)
        self.assertEqual(len(levels), 14)
        self.assertAlmostEqual(growth.loc["2021-01-01"], 12)
        self.assertTrue(pd.isna(growth.loc["2021-02-01"]))

    def test_explicit_missing_values_are_not_dropped_or_filled(self):
        data = {"observations": [
            {"date": "2020-01-01", "value": "100"},
            {"date": "2020-02-01", "value": "."},
            {"date": "2020-03-01", "value": None},
        ]}
        series = prepare_data(data, "MS")
        self.assertEqual(len(series), 3)
        self.assertEqual(series.isna().tolist(), [False, True, True])

    def test_invalid_observations_fail_explicitly(self):
        bad_rows = [
            {"date": "not-a-date", "value": "100"},
            {"date": "2020-01-01", "value": "bad"},
            {"date": "2020-01-01", "value": "inf"},
            {"date": "2020-01-01", "value": "nan"},
            {"date": "2020-01-01", "value": "0"},
            {"date": "2020-01-01", "value": -1},
            {"date": "2020-01-01", "value": True},
            {"date": "20200101", "value": "100"},
            {"date": "2020-01-01T12:00:00", "value": "100"},
            {"date": "2020-01-02", "value": "100"},
            {"date": "2020-01-01"},
        ]
        for row in bad_rows:
            with self.subTest(row=row), self.assertRaises(ValueError):
                prepare_data({"observations": [row]}, "MS")
        with self.assertRaisesRegex(ValueError, "frequency"):
            prepare_data({"observations": [{"date": "2020-02-01", "value": "100"}]}, "QS-JAN")

    def test_duplicate_dates_are_rejected_and_unsorted_dates_are_sorted(self):
        data = observations("2020-01-01", "2020-03-01", "MS")
        data["observations"].reverse()
        self.assertEqual(prepare_data(data, "MS").iloc[0], 100)
        data["observations"].append(data["observations"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prepare_data(data, "MS")

    def test_empty_or_all_missing_data_cannot_produce_a_latest_value(self):
        for rows in ([], [{"date": "2020-01-01", "value": "."}]):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "No usable"):
                analyze_series({"observations": rows}, "GDP")

    def test_latest_gdp_level_is_labeled_and_distinct_from_growth(self):
        data = observations("2020-01-01", "2021-01-01", "QS-JAN")
        report = analyze_series(data, "GDP")
        self.assertEqual(report["latest"]["observation_date"], "2021-01-01")
        self.assertEqual(report["latest"]["level"], 104)
        self.assertAlmostEqual(report["latest"]["yoy_percent"], 4)
        self.assertAlmostEqual(report["latest"]["qoq_annualized_percent"], ((104 / 103) ** 4 - 1) * 100)
        self.assertIn("nominal", report["title"])
        self.assertIn("Billions", report["units"])
        self.assertIn("t-4", report["definitions"]["yoy_percent"])

    def test_first_rate_in_filtered_window_uses_earlier_history(self):
        for series_id, freq, end, start, expected in (
            ("GDP", "QS-JAN", "2021-04-01", "2021-01-01", 4),
            ("CPILFESL", "MS", "2021-02-01", "2021-01-01", 12),
        ):
            with self.subTest(series_id=series_id):
                report = analyze_series(observations("2020-01-01", end, freq), series_id, start=start, end=end)
                self.assertEqual(len(report["history"]), 2)
                self.assertAlmostEqual(report["history"][0]["yoy_percent"], expected)

    def test_latest_inflation_is_unavailable_when_its_baseline_is_missing(self):
        data = observations("2020-01-01", "2021-02-01", "MS")
        data["observations"][1]["value"] = "."
        report = analyze_series(data, "CPILFESL")
        self.assertEqual(report["latest"]["observation_date"], "2021-02-01")
        self.assertIsNone(report["latest"]["yoy_percent"])
        self.assertTrue(report["warnings"])
        json.dumps(report, allow_nan=False)

    def test_trailing_missing_level_does_not_masquerade_as_latest_reported_value(self):
        data = observations("2020-01-01", "2021-02-01", "MS")
        data["observations"][-1]["value"] = "."
        report = analyze_series(data, "CPILFESL")
        self.assertEqual(report["latest"]["observation_date"], "2021-01-01")
        self.assertIsNone(report["history"][-1]["level"])
        self.assertTrue(any("Latest usable" in text for text in report["warnings"]))

    def test_rolling_ten_and_five_year_windows_include_initial_growth(self):
        for series_id, frequency, years, count in (("GDP", "QS-JAN", 10, 41), ("CPILFESL", "MS", 5, 61)):
            with self.subTest(series_id=series_id):
                data = observations("2009-01-01", "2021-01-01", frequency)
                report = analyze_series(data, series_id, years=years)
                self.assertEqual(len(report["history"]), count)
                self.assertIsNotNone(report["history"][0]["yoy_percent"])
                self.assertEqual(report["latest"]["observation_date"], "2021-01-01")

    def test_range_validation_and_partial_coverage(self):
        data = observations("2020-01-01", "2021-01-01", "MS")
        for kwargs in ({"start": "2021-01-01", "end": "2020-01-01"},
                       {"years": 0}, {"years": True}, {"years": 1, "start": "2020-01-01"},
                       {"start": "2020-02-30"}, {"start": "2030-01-01"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                analyze_series(data, "CPILFESL", **kwargs)
        report = analyze_series(data, "CPILFESL", start="2010-01-01", end="2025-01-01")
        self.assertTrue(any("Requested start" in w for w in report["warnings"]))
        self.assertTrue(any("Requested end" in w for w in report["warnings"]))

    def test_inflation_change_uses_percentage_points(self):
        data = observations("2019-01-01", "2021-01-01", "MS")
        report = analyze_series(data, "CPILFESL", start="2020-01-01")
        summary = report["summary"]
        self.assertAlmostEqual(summary["first_yoy_percent"], 12)
        self.assertAlmostEqual(summary["last_yoy_percent"], (124 / 112 - 1) * 100)
        self.assertAlmostEqual(summary["yoy_change_percentage_points"], (124 / 112 - 1) * 100 - 12)

    def test_growth_functions_preserve_missing_previous_quarter(self):
        data = observations("2020-01-01", "2021-01-01", "QS-JAN")
        del data["observations"][-2]
        growth = calculate_gdp_qoq_annualized(prepare_data(data, "QS-JAN"))
        self.assertTrue(pd.isna(growth.iloc[-1]))
        with self.assertRaises(ValueError):
            calculate_yoy(prepare_data(data, "QS-JAN"), 1)


if __name__ == "__main__":
    unittest.main()
