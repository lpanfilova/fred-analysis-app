import asyncio
from mcp import Client
from oauth_login import BrowserLogin
from oauth_storage import KeychainTokenStorage, PersistentOAuthClientProvider
from fred_service import get_observations
from analytics import prepare_data, calculate_yoy

import httpx2
from pydantic import AnyUrl
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientMetadata

FRED_MCP_URL = "https://mcp.stlouisfed.org"

async def main():
    async with BrowserLogin() as login:
        await run_client(login)


async def run_client(login):
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

            gdp = await get_observations(client, "GDP")
            cpi = await get_observations(client, "CPILFESL")

            gdp_series = prepare_data(gdp)
            cpi_series = prepare_data(cpi)

            gdp_growth = calculate_yoy(gdp_series, 4)
            cpi_inflation = calculate_yoy(cpi_series, 12)

            print("Latest GDP growth:")
            print(gdp_growth.dropna().tail(1))

            print("Latest Core CPI inflation:")
            print(cpi_inflation.dropna().tail(1))

            print("GDP:", len(gdp["observations"]), "observations")
            print("CPI:", len(cpi["observations"]), "observations")

if __name__ == "__main__":
    asyncio.run(main())
