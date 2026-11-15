import asyncio
import argparse
import json
from mcp import Client
from oauth_login import BrowserLogin
from oauth_storage import KeychainTokenStorage, PersistentOAuthClientProvider
from fred_service import get_series_analysis
from analytics import parse_date

import httpx2
from pydantic import AnyUrl
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientMetadata

FRED_MCP_URL = "https://mcp.stlouisfed.org"

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze nominal GDP and Core CPI through FRED MCP.")
    parser.add_argument("--start", help="Inclusive observation start date (YYYY-MM-DD).")
    parser.add_argument("--end", help="Inclusive observation end date (YYYY-MM-DD).")
    parser.add_argument("--history", action="store_true", help="Include individual historical observations.")
    args = parser.parse_args()
    try:
        start = parse_date(args.start) if args.start is not None else None
        end = parse_date(args.end) if args.end is not None else None
        if start is not None and end is not None and start > end:
            raise ValueError("start must be on or before end.")
    except ValueError as exc:
        parser.error(str(exc))
    return args


async def main(args):
    async with BrowserLogin() as login:
        await run_client(login, args)


async def run_client(login, args):
    oauth = PersistentOAuthClientProvider(
        server_url=FRED_MCP_URL,
        client_metadata=OAuthClientMetadata(
            client_name="FRED Analysis App",
            redirect_uris=[AnyUrl(login.redirect_uri)],
            grant_types=["authorization_code", "refresh_token"],
        ),
        storage=KeychainTokenStorage(FRED_MCP_URL),
        redirect_handler=login.handle_redirect,
        callback_handler=login.handle_callback
    )

    async with httpx2.AsyncClient(
        auth = oauth,
        timeout = httpx2.Timeout(30.0, read=300.0)
    ) as http_client:

        transport = streamable_http_client(
            FRED_MCP_URL,
            http_client=http_client
        )

        async with Client(transport) as client:

            for series_id, default_years in (("GDP", 10), ("CPILFESL", 5)):
                bounds = (
                    {"start": args.start, "end": args.end}
                    if args.start is not None or args.end is not None
                    else {"years": default_years}
                )
                report = await get_series_analysis(client, series_id, **bounds)
                if not args.history:
                    report.pop("history")
                print(json.dumps(report, indent=2, allow_nan=False))

if __name__ == "__main__":
    asyncio.run(main(parse_args()))
