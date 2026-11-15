async def get_metadata(client, series_id):
    response = await client.call_tool(
        "fred-get-series-metadata",
        arguments={"series_id": series_id}
    )
    if response.is_error:
        raise RuntimeError("FRED metadata request failed")
    return response.structured_content


async def get_observations(client, series_id):
    response = await client.call_tool(
        "fred-get-series-observations",
        arguments={"series_id": series_id}
    )
    if response.is_error:
        raise RuntimeError("FRED observations request failed")
    return response.structured_content