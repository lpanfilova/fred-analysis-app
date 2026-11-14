import asyncio
import json
from mcp import Client
from oauth_storage import InMemoryTokenStorage

import httpx2
from pydantic import AnyUrl
from urllib.parse import urlparse, parse_qs

from mcp.client.auth import OAuthClientProvider, AuthorizationCodeResult
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientMetadata

FRED_MCP_URL = "https://mcp.stlouisfed.org"

async def handle_redirect(auth_url: str):
    print(f"Open this URL in your browser:\n{auth_url}")

async def handle_callback() -> AuthorizationCodeResult:
    callback_url = input("Paste the redirected URL: ")

    params = parse_qs(urlparse(callback_url).query)

    return AuthorizationCodeResult(
        code=params["code"][0],
        state=params["state"][0],
        iss=params.get("iss", [None])[0]
    )

async def main():
    oauth = OAuthClientProvider(
        server_url=FRED_MCP_URL,
        client_metadata=OAuthClientMetadata(
            client_name="FRED Analysis App",
            redirect_uris=[AnyUrl("http://localhost:3030/callback")]
        ),
        storage=InMemoryTokenStorage(),
        redirect_handler=handle_redirect,
        callback_handler=handle_callback
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
            result = await client.list_tools()

            for tool in result.tools:
                print(tool.name)

if __name__ == "__main__":
    asyncio.run(main())