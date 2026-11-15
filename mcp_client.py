import asyncio
import argparse
import json
from pathlib import Path
from mcp import Client
from oauth_login import BrowserLogin
from oauth_storage import KeychainTokenStorage, PersistentOAuthClientProvider
from fred_service import get_comparison_analysis, get_series_analysis
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
    parser.add_argument("--compare", action="store_true", help="Compare quarterly GDP growth and Core CPI inflation; bounds include full quarters only.")
    parser.add_argument("--years", type=int, help="Rolling comparison window in years (default: 10); cannot combine with date bounds.")
    parser.add_argument("--chart", type=Path, help="Export a comparison chart, e.g. outputs/comparison.png (PNG, SVG, or PDF); implies --compare.")
    args = parser.parse_args()
    try:
        start = parse_date(args.start) if args.start is not None else None
        end = parse_date(args.end) if args.end is not None else None
        if start is not None and end is not None and start > end:
            raise ValueError("start must be on or before end.")
        if args.years is not None:
            if not (args.compare or args.chart):
                raise ValueError("--years requires --compare or --chart.")
            if args.years <= 0 or start is not None or end is not None:
                raise ValueError("--years must be positive and cannot be combined with --start or --end.")
        if args.chart is not None and args.chart.suffix.lower() not in {".png", ".svg", ".pdf"}:
            raise ValueError("--chart must use a .png, .svg, or .pdf extension.")
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
            if args.compare or args.chart:
                bounds = (
                    {"start": args.start, "end": args.end}
                    if args.start is not None or args.end is not None
                    else {"years": args.years if args.years is not None else 10}
                )
                report = await get_comparison_analysis(client, **bounds)
                if args.chart:
                    from visualization import save_comparison_chart
                    report["chart_path"] = str(save_comparison_chart(report, args.chart))
                if not args.history:
                    report.pop("history")
                print(json.dumps(report, indent=2, allow_nan=False))
                return
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
