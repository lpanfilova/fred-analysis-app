"""Align and compare nominal GDP growth with quarterly Core CPI inflation."""

import pandas as pd

from analytics import SERIES, _number, calculate_yoy, parse_date, prepare_data


GDP_RATE = "gdp_yoy_percent"
CPI_RATE = "core_cpi_yoy_percent"


def align_quarterly(gdp_data, cpi_data):
    """Compute rates from full history; require three CPI months per quarter.

    CPI inflation is the year-over-year change in quarterly average index
    levels, rather than the average of monthly inflation rates. Missing current
    or year-earlier quarters stay missing. GDP is nominal, not inflation-adjusted.
    """
    gdp = prepare_data(gdp_data, "QS-JAN")
    cpi = prepare_data(cpi_data, "MS")
    if gdp.dropna().empty or cpi.dropna().empty:
        raise ValueError("Both GDP and Core CPI need usable observations.")
    monthly_count = cpi.resample("QS-JAN").count()
    quarterly_cpi = cpi.resample("QS-JAN").mean().where(monthly_count == 3)
    return pd.DataFrame({
        "gdp_level": gdp,
        "core_cpi_quarterly_index": quarterly_cpi,
        "core_cpi_months": monthly_count,
        GDP_RATE: calculate_yoy(gdp, 4),
        CPI_RATE: calculate_yoy(quarterly_cpi, 4),
    }).sort_index()


def compare_series(gdp_data, cpi_data, *, start=None, end=None, years=None):
    """Return a JSON-compatible comparison for any selected date window.

    Bounds select quarters wholly contained in the requested dates. Rates use
    earlier history before filtering. Rolling windows end at the most recent
    quarter with both rates available and contain `years * 4` calendar quarters.
    Pearson correlation uses only contemporaneous complete pairs, with at least
    three pairs and variation in both rates. No interpolation is performed.
    """
    if years is not None and (
        type(years) is not int or years <= 0 or start is not None or end is not None
    ):
        raise ValueError("years must be a positive integer without explicit date bounds.")
    lower = parse_date(start) if start is not None else None
    upper = parse_date(end) if end is not None else None
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("start must be on or before end.")
    frame = align_quarterly(gdp_data, cpi_data)
    all_pairs = frame.dropna(subset=[GDP_RATE, CPI_RATE])
    if all_pairs.empty:
        raise ValueError("No overlapping quarters with both year-over-year rates; at least one year of baseline data is required.")
    if years is not None:
        last_period = all_pairs.index[-1].to_period("Q")
        upper = last_period.end_time.normalize()
        lower = (last_period - (years * 4 - 1)).start_time
    quarter_ends = frame.index.to_period("Q").end_time.normalize()
    mask = pd.Series(True, index=frame.index)
    if lower is not None:
        mask &= frame.index >= lower
    if upper is not None:
        mask &= quarter_ends <= upper
    selected = frame.loc[mask]
    pairs = selected.dropna(subset=[GDP_RATE, CPI_RATE])
    if pairs.empty:
        raise ValueError("No complete comparison pairs in the requested range. Bounds must include a full quarter with both rates available.")

    correlation = None
    warnings = []
    if len(pairs) < 3:
        warnings.append("Correlation is unavailable: at least three paired quarters are required.")
    elif any(pairs[column].nunique() < 2 for column in (GDP_RATE, CPI_RATE)):
        warnings.append("Correlation is unavailable: both series must vary within the selected range.")
    else:
        correlation = _number(pairs[GDP_RATE].corr(pairs[CPI_RATE]))
        if len(pairs) < 8:
            warnings.append("Correlation is based on fewer than eight paired quarters; the sample is small.")
    missing = len(selected) - len(pairs)
    if missing:
        warnings.append(f"{missing} quarter(s) lack one or both rates and are excluded from correlation; gaps remain in history.")
    if lower is not None and lower < all_pairs.index.min():
        warnings.append("Requested start precedes the first available comparison pair; coverage is limited by available baseline data.")
    last_common_end = all_pairs.index[-1].to_period("Q").end_time.normalize()
    if upper is not None and upper > last_common_end:
        warnings.append("Requested end extends beyond the latest complete comparison pair.")

    def point(timestamp, row):
        period = timestamp.to_period("Q")
        return {
            "quarter": str(period),
            "quarter_start": period.start_time.date().isoformat(),
            "quarter_end": period.end_time.date().isoformat(),
            **{key: _number(value) for key, value in row.items()},
        }

    def summary(column):
        available = selected[column].dropna()
        return {
            "first_quarter": str(available.index[0].to_period("Q")),
            "last_quarter": str(available.index[-1].to_period("Q")),
            "first_percent": float(available.iloc[0]),
            "last_percent": float(available.iloc[-1]),
            "change_percentage_points": float(available.iloc[-1] - available.iloc[0]),
            "mean_percent": float(available.mean()),
            "min_percent": float(available.min()), "max_percent": float(available.max()),
        }

    return {
        "title": "Nominal GDP growth and Core CPI inflation",
        "frequency": "Quarterly", "units": "Percent change from the same quarter one year earlier",
        "requested_range": {"start": lower.date().isoformat() if lower is not None else None,
                            "end": upper.date().isoformat() if upper is not None else None},
        "method": {
            GDP_RATE: "(nominal GDP[t] / nominal GDP[t-4] - 1) * 100; current-dollar, seasonally adjusted GDP",
            CPI_RATE: "(quarterly average Core CPI[t] / quarterly average Core CPI[t-4] - 1) * 100; seasonally adjusted index",
            "cpi_aggregation": "Arithmetic mean of all three monthly index levels; incomplete quarters are unavailable.",
            "date_bounds": "Only quarters wholly contained in the requested dates are selected.",
            "correlation": "Pearson correlation of same-quarter year-over-year rates; complete pairs only; minimum three pairs.",
        },
        "relationship": {
            "pearson_correlation": correlation, "paired_quarters": len(pairs),
            "excluded_quarters": missing,
            "first_paired_quarter": str(pairs.index[0].to_period("Q")),
            "last_paired_quarter": str(pairs.index[-1].to_period("Q")),
            "interpretation": "Descriptive association over this period; correlation does not establish causation. GDP growth is nominal and includes price changes.",
        },
        "latest_common": point(pairs.index[-1], pairs.iloc[-1]),
        "summary": {GDP_RATE: summary(GDP_RATE), CPI_RATE: summary(CPI_RATE)},
        "history": [point(timestamp, row) for timestamp, row in selected.iterrows()],
        "sources": [SERIES[key]["source_url"] for key in ("GDP", "CPILFESL")],
        "data_vintage": {
            key: {field: payload.get(field) for field in ("realtime_start", "realtime_end", "last_updated")}
            for key, payload in (("GDP", gdp_data), ("CPILFESL", cpi_data))
        },
        "warnings": warnings,
    }
