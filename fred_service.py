"""Validate MCP responses before passing FRED data to analytics."""

import json

from analytics import SERIES, analyze_series
from comparison import compare_series


async def _call(client, tool, series_id):
    response = await client.call_tool(tool, arguments={"series_id": series_id})
    if response.is_error:
        raise RuntimeError(f"FRED request failed for {series_id} ({tool}).")
    payload = response.structured_content
    if payload is None:
        # MCP servers may return JSON as text rather than structured content.
        blocks = [block.text for block in response.content if block.type == "text"]
        try:
            payload = json.loads("\n".join(blocks))
        except (ValueError, TypeError):
            raise ValueError(f"FRED returned invalid JSON for {series_id}.") from None
    if not isinstance(payload, dict):
        raise ValueError(f"FRED returned an invalid response for {series_id}.")
    if "error_code" in payload or "error" in payload:
        raise RuntimeError(f"FRED returned an error for {series_id}.")
    return payload


async def get_metadata(client, series_id):
    return await _call(client, "fred-get-series-metadata", series_id)


async def get_observations(client, series_id):
    """Fetch full available history so range analysis includes prior-year levels.

    Reject declared partial responses rather than treating a page as the full
    series. Date filtering belongs after growth calculation in analytics.
    """
    payload = await _call(client, "fred-get-series-observations", series_id)
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise ValueError(f"FRED response for {series_id} has no observations list.")
    normalized = []
    for row in observations:
        if isinstance(row, (list, tuple)) and len(row) == 2:
            # The official MCP connector returns [date, value] pairs.
            normalized.append({"date": row[0], "value": row[1]})
        elif isinstance(row, dict) and "date" in row and "value" in row:
            normalized.append(row)
        else:
            raise ValueError(f"FRED response for {series_id} contains an invalid observation.")
    try:
        count = int(payload.get("count", len(observations)))
        offset = int(payload.get("offset", 0))
    except (ValueError, TypeError):
        raise ValueError(f"FRED returned invalid pagination for {series_id}.") from None
    if offset != 0 or count != len(observations):
        raise ValueError(f"FRED returned incomplete history for {series_id}; all observations are required.")
    return {**payload, "observations": normalized}


async def get_series_analysis(client, series_id, *, start=None, end=None, years=None):
    """Retrieve observations and answer latest/historical questions for one series."""
    if series_id not in SERIES:
        raise ValueError("Supported series are GDP and CPILFESL.")
    data = await get_observations(client, series_id)
    return analyze_series(data, series_id, start=start, end=end, years=years)


async def get_comparison_analysis(client, *, start=None, end=None, years=None):
    """Fetch both complete histories and compare their quarterly growth rates."""
    gdp = await get_observations(client, "GDP")
    cpi = await get_observations(client, "CPILFESL")
    return compare_series(gdp, cpi, start=start, end=end, years=years)
