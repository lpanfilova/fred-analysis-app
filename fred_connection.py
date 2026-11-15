"""A FRED MCP connection whose async resources share one task and event loop."""

from contextlib import asynccontextmanager

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientMetadata
from pydantic import AnyUrl

from oauth_login import BrowserLogin
from oauth_storage import KeychainTokenStorage, PersistentOAuthClientProvider


FRED_MCP_URL = "https://mcp.stlouisfed.org"


@asynccontextmanager
async def connect_fred(login, on_authorize=None):
    async def redirect(auth_url):
        await login.handle_redirect(auth_url)
        if on_authorize is not None:
            on_authorize(auth_url)

    oauth = PersistentOAuthClientProvider(
        server_url=FRED_MCP_URL,
        client_metadata=OAuthClientMetadata(
            client_name="FRED Analysis App",
            redirect_uris=[AnyUrl(login.redirect_uri)],
            grant_types=["authorization_code", "refresh_token"],
        ),
        storage=KeychainTokenStorage(FRED_MCP_URL),
        redirect_handler=redirect,
        callback_handler=login.handle_callback,
    )
    async with httpx2.AsyncClient(
        auth=oauth, timeout=httpx2.Timeout(30.0, read=300.0)
    ) as http:
        async with Client(streamable_http_client(FRED_MCP_URL, http_client=http)) as client:
            yield client


@asynccontextmanager
async def fred_session(on_authorize=None):
    async with BrowserLogin() as login:
        async with connect_fred(login, on_authorize) as client:
            yield client
