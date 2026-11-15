"""Calendar-aware analysis of nominal GDP and seasonally adjusted Core CPI."""

from datetime import date
import math

import pandas as pd


SERIES = {
    "GDP": {
        "title": "U.S. nominal GDP (current dollars)",
        "frequency": "QS-JAN", "periods_per_year": 4,
        "units": "Billions of U.S. dollars, seasonally adjusted annual rate",
        "source_url": "https://fred.stlouisfed.org/series/GDP",
    },
    "CPILFESL": {
        "title": "U.S. Core CPI (all items less food and energy)",
        "frequency": "MS", "periods_per_year": 12,
        "units": "Index 1982–1984=100, seasonally adjusted",
        "source_url": "https://fred.stlouisfed.org/series/CPILFESL",
    },
}


def parse_date(value):
    """Accept an ISO observation date, without time or timezone components."""
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError
        return pd.Timestamp(parsed)
    except (TypeError, ValueError):
        raise ValueError("Dates must use YYYY-MM-DD format.") from None


def _regularize(series, frequency):
    if frequency not in {"MS", "QS-JAN"}:
        raise ValueError("Frequency must be MS (monthly) or QS-JAN (quarterly).")
    if not isinstance(series.index, pd.DatetimeIndex) or series.index.hasnans:
        raise ValueError("Observations must have valid dates.")
    if series.index.tz is not None or not series.index.equals(series.index.normalize()):
        raise ValueError("Observation dates must not contain times or timezones.")
    if series.index.has_duplicates:
        raise ValueError("Duplicate observation dates are not supported.")
    series = series.sort_index()
    if series.empty:
        return series
    if not all(pd.tseries.frequencies.to_offset(frequency).is_on_offset(d) for d in series.index):
        raise ValueError("Observation dates do not match the expected frequency.")
    return series.reindex(pd.date_range(series.index.min(), series.index.max(), freq=frequency))


def prepare_data(data, frequency):
    """Keep FRED '.'/null values and absent periods as NaN; reject invalid data."""
    if not isinstance(data, dict) or not isinstance(data.get("observations"), list):
        raise ValueError("Expected a FRED observations list.")
    rows = data["observations"]
    dates, values = [], []
    for row in rows:
        if not isinstance(row, dict) or "date" not in row or "value" not in row:
            raise ValueError("Each observation must contain date and value.")
        dates.append(parse_date(row["date"]))
        raw = row["value"]
        if raw is None or raw == ".":
            values.append(float("nan"))
            continue
        try:
            value = float(raw)
        except (ValueError, TypeError):
            raise ValueError("Observation values must be numeric, '.' or null.") from None
        if isinstance(raw, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("GDP and Core CPI levels must be finite positive numbers.")
        values.append(value)
    series = pd.Series(values, index=pd.DatetimeIndex(dates), dtype=float, name="value")
    return _regularize(series, frequency)


def calculate_yoy(series, periods):
    """Percent change against the same quarter/month of the preceding year."""
    if periods not in (4, 12):
        raise ValueError("Year-over-year calculations require 4 quarters or 12 months.")
    regular = _regularize(series, "QS-JAN" if periods == 4 else "MS")
    baseline = regular.shift(periods)
    return (regular / baseline.where(baseline > 0) - 1) * 100


def calculate_gdp_qoq_annualized(series):
    """Annualized nominal quarterly growth: ((GDP[t]/GDP[t-1])**4 - 1)*100."""
    regular = _regularize(series, "QS-JAN")
    baseline = regular.shift(1)
    return ((regular / baseline.where(baseline > 0)) ** 4 - 1) * 100


def _number(value):
    return float(value) if pd.notna(value) and math.isfinite(value) else None


def analyze_series(data, series_id, *, start=None, end=None, years=None):
    """Return JSON-compatible latest values, history, definitions, and changes.

    Date bounds are inclusive observation dates (quarter/month starts), not
    release dates. All supplied history is used for growth calculations BEFORE
    filtering. A rolling window is anchored to the latest nonmissing level.
    Latest values refer to the selected window; unavailable rates remain null.
    """
    if series_id not in SERIES:
        raise ValueError("Supported series are GDP and CPILFESL.")
    if years is not None and (type(years) is not int or years <= 0 or start is not None or end is not None):
        raise ValueError("years must be a positive integer without explicit date bounds.")
    start_date = parse_date(start) if start is not None else None
    end_date = parse_date(end) if end is not None else None
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start must be on or before end.")
    spec = SERIES[series_id]
    levels = prepare_data(data, spec["frequency"])
    if levels.dropna().empty:
        raise ValueError(f"No usable observations for {series_id}.")
    if years is not None:
        end_date = levels.last_valid_index()
        start_date = end_date - pd.DateOffset(years=years)
    frame = pd.DataFrame({
        "level": levels,
        "yoy_percent": calculate_yoy(levels, spec["periods_per_year"]),
    })
    definitions = {
        "level": spec["units"],
        "yoy_percent": f"(level[t] / level[t-{spec['periods_per_year']}] - 1) * 100; percent year over year",
    }
    if series_id == "GDP":
        frame["qoq_annualized_percent"] = calculate_gdp_qoq_annualized(levels)
        definitions["qoq_annualized_percent"] = (
            "((GDP[t] / GDP[t-1]) ** 4 - 1) * 100; percent quarter over quarter, annualized; nominal GDP"
        )
    selected = frame.loc[start_date:end_date]
    valid = selected.dropna(subset=["level"])
    if valid.empty:
        raise ValueError(f"No usable {series_id} observations in the requested date range.")

    def point(timestamp, row):
        return {"observation_date": timestamp.date().isoformat(),
                **{key: _number(value) for key, value in row.items()}}

    first, last = valid.iloc[0], valid.iloc[-1]
    summary = {
        "first_observation_date": valid.index[0].date().isoformat(),
        "last_observation_date": valid.index[-1].date().isoformat(),
        "first_level": float(first["level"]), "last_level": float(last["level"]),
        "level_change": float(last["level"] - first["level"]),
        "level_change_percent": float((last["level"] / first["level"] - 1) * 100),
        "first_yoy_percent": _number(first["yoy_percent"]),
        "last_yoy_percent": _number(last["yoy_percent"]),
        "yoy_change_percentage_points": _number(last["yoy_percent"] - first["yoy_percent"]),
    }
    warnings = []
    if start_date is not None and start_date < levels.index.min():
        warnings.append("Requested start precedes available history; results cover available observations only.")
    if end_date is not None and end_date > levels.index.max():
        warnings.append("Requested end follows available history; results cover available observations only.")
    if selected["level"].isna().any():
        warnings.append("Missing levels are preserved; no values are interpolated or carried forward.")
    if selected["yoy_percent"].isna().any():
        warnings.append("Some year-over-year rates are unavailable because a current or year-earlier level is missing.")
    if valid.index[-1] != selected.index[-1]:
        warnings.append("Latest usable level precedes the last period in the selected range.")
    return {
        "series_id": series_id, "title": spec["title"], "units": spec["units"],
        "frequency": "Quarterly" if series_id == "GDP" else "Monthly",
        "yoy_label": "Nominal GDP growth, year over year" if series_id == "GDP" else "Core CPI inflation, year over year",
        "source_url": spec["source_url"], "definitions": definitions,
        "requested_range": {"start": start_date.date().isoformat() if start_date is not None else None,
                            "end": end_date.date().isoformat() if end_date is not None else None},
        "latest": point(valid.index[-1], last), "summary": summary,
        "history": [point(timestamp, row) for timestamp, row in selected.iterrows()],
        "warnings": warnings,
    }
