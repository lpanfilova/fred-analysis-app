import asyncio
import time
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs

import httpx2
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthMetadata, OAuthToken
from pydantic import AnyUrl

from oauth_login import BrowserLogin
from oauth_storage import KeychainTokenStorage, PersistentOAuthClientProvider


SERVER = "https://mcp.example.com"
ISSUER = "https://login.example.com"


class MemoryKeychain:
    def __init__(self):
        self.values = {}

    def get_password(self, service, account):
        return self.values.get((service, account))

    def set_password(self, service, account, value):
        self.values[service, account] = value


class StorageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.backend = MemoryKeychain()
        self.storage = KeychainTokenStorage(SERVER, self.backend)
        self.metadata = OAuthMetadata(
            issuer=ISSUER, authorization_endpoint=f"{ISSUER}/authorize",
            token_endpoint=f"{ISSUER}/oauth/token",
            response_types_supported=["code"],
        )
        self.client = OAuthClientInformationFull(client_id="test-client", issuer=ISSUER)

    async def provider(self, storage):
        return PersistentOAuthClientProvider(
            server_url=SERVER,
            client_metadata=OAuthClientMetadata(
                redirect_uris=[AnyUrl("http://localhost:3030/callback")]
            ),
            storage=storage,
        )

    async def test_empty_keychain(self):
        self.assertIsNone(await self.storage.get_tokens())
        self.assertIsNone(await self.storage.get_client_info())

    async def test_restart_restores_registration_metadata_and_absolute_expiry(self):
        self.storage.metadata = self.metadata
        await self.storage.set_client_info(self.client)
        await self.storage.set_tokens(OAuthToken(
            access_token="old", token_type="Bearer", refresh_token="refresh", expires_in=3600
        ))
        expiry = self.storage.expires_at
        restarted = KeychainTokenStorage(SERVER, self.backend)
        provider = await self.provider(restarted)
        await provider._initialize()
        self.assertEqual(provider.context.token_expiry_time, expiry)
        self.assertEqual(provider.context.client_info.client_id, "test-client")
        self.assertEqual(str(provider.context.oauth_metadata.token_endpoint), f"{ISSUER}/oauth/token")
        self.assertTrue(provider.context.is_token_valid())

    async def test_expired_token_refreshes_after_restart_without_browser(self):
        self.storage.metadata = self.metadata
        await self.storage.set_client_info(self.client)
        await self.storage.set_tokens(OAuthToken(
            access_token="expired", token_type="Bearer", refresh_token="original-refresh",
            expires_in=0, scope="read",
        ))
        provider = await self.provider(KeychainTokenStorage(SERVER, self.backend))
        flow = provider._auth_flow(httpx2.Request("GET", SERVER))
        refresh_request = await anext(flow)
        self.assertEqual(str(refresh_request.url), f"{ISSUER}/oauth/token")
        self.assertEqual(parse_qs(refresh_request.content.decode())["refresh_token"], ["original-refresh"])
        request = await flow.asend(httpx2.Response(200, json={
            "access_token": "renewed", "token_type": "Bearer", "expires_in": 3600,
            "refresh_token": "rotated-refresh",
        }))
        self.assertEqual(request.headers["Authorization"], "Bearer renewed")
        with self.assertRaises(StopAsyncIteration):
            await flow.asend(httpx2.Response(200))
        saved = await KeychainTokenStorage(SERVER, self.backend).get_tokens()
        self.assertEqual(saved.refresh_token, "rotated-refresh")
        self.assertEqual(saved.scope, "read")

    async def test_refresh_preserves_unrotated_refresh_token(self):
        provider = await self.provider(self.storage)
        provider.context.oauth_metadata = self.metadata
        provider.context.current_tokens = OAuthToken(
            access_token="old", token_type="Bearer", refresh_token="keep-me"
        )
        self.assertTrue(await provider._handle_refresh_response(httpx2.Response(200, json={
            "access_token": "new", "token_type": "Bearer", "expires_in": 3600,
        })))
        self.assertEqual((await self.storage.get_tokens()).refresh_token, "keep-me")

    async def test_initial_exchange_persists_refresh_metadata(self):
        provider = await self.provider(self.storage)
        provider.context.oauth_metadata = self.metadata
        await provider._handle_token_response(httpx2.Response(200, json={
            "access_token": "initial", "token_type": "Bearer", "expires_in": 3600,
        }))
        restarted = KeychainTokenStorage(SERVER, self.backend)
        await restarted.get_tokens()
        self.assertEqual(restarted.metadata, self.metadata)
        self.assertGreater(restarted.expires_at, time.time())

    async def test_keychain_failure_is_not_silently_ignored(self):
        with patch.object(self.backend, "set_password", side_effect=RuntimeError("locked")):
            with self.assertRaisesRegex(RuntimeError, "locked"):
                await self.storage.set_client_info(self.client)

    async def test_failed_refresh_returns_to_sdk_authorization_flow(self):
        self.storage.metadata = self.metadata
        await self.storage.set_client_info(self.client)
        await self.storage.set_tokens(OAuthToken(
            access_token="expired", token_type="Bearer", refresh_token="revoked", expires_in=0
        ))
        provider = await self.provider(KeychainTokenStorage(SERVER, self.backend))
        flow = provider._auth_flow(httpx2.Request("GET", SERVER))
        await anext(flow)
        with self.assertLogs("mcp.client.auth.oauth2", level="WARNING"):
            request = await flow.asend(httpx2.Response(400))
        self.assertNotIn("Authorization", request.headers)
        discovery = await flow.asend(httpx2.Response(401, headers={
            "WWW-Authenticate": f'Bearer resource_metadata="{SERVER}/.well-known/oauth-protected-resource"'
        }))
        self.assertIn(".well-known/oauth-protected-resource", str(discovery.url))
        await flow.aclose()


class BrowserTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.browser = patch("oauth_login.webbrowser.open", return_value=True).start()
        self.addCleanup(patch.stopall)
        self.login = BrowserLogin(port=0, timeout=0.1)
        await self.login.handle_redirect("https://login.example.com/authorize?state=expected")
        self.socket = self.login._server.sockets[0]
        self.address = self.socket.getsockname()[:2]

    async def asyncTearDown(self):
        await self.login._close()

    async def request(self, target):
        reader, writer = await asyncio.open_connection(*self.address)
        writer.write(f"GET {target} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode())
        await writer.drain()
        response = await reader.read()
        writer.close()
        await writer.wait_closed()
        return response

    async def test_valid_callback_passes_issuer_and_closes_listener(self):
        response = await self.request("/callback?code=code&state=expected&iss=https%3A%2F%2Flogin.example.com")
        self.assertIn(b"200 OK", response)
        result = await self.login.handle_callback()
        self.assertEqual(result.code, "code")
        self.assertEqual(result.iss, ISSUER)
        self.assertIsNone(self.login._server)
        self.browser.assert_called_once()

    async def test_wrong_state_does_not_consume_callback(self):
        response = await self.request("/callback?code=wrong&state=attacker")
        self.assertIn(b"400 Bad Request", response)
        self.assertFalse(self.login._result.done())
        await self.request("/callback?code=correct&state=expected")
        self.assertEqual((await self.login.handle_callback()).code, "correct")

    async def test_unrelated_request_does_not_consume_callback(self):
        self.assertIn(b"404 Not Found", await self.request("/favicon.ico"))
        self.assertFalse(self.login._result.done())

    async def test_denial_reports_error_and_closes_listener(self):
        await self.request("/callback?error=access_denied&state=expected")
        with self.assertRaisesRegex(RuntimeError, "denied"):
            await self.login.handle_callback()
        self.assertIsNone(self.login._server)

    async def test_timeout_closes_listener(self):
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            await self.login.handle_callback()
        self.assertIsNone(self.login._server)

    async def test_cancel_closes_listener(self):
        task = asyncio.create_task(self.login.handle_callback())
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIsNone(self.login._server)

    async def test_browser_failure_prints_manual_link(self):
        self.browser.return_value = False
        with patch("builtins.print") as output:
            await self.login.handle_redirect("https://login.example.com/authorize?state=new")
        self.assertTrue(any("Open this URL" in str(call) for call in output.call_args_list))

    async def test_busy_port_fails_before_browser_opens(self):
        self.browser.reset_mock()
        with patch("oauth_login.asyncio.start_server", side_effect=OSError("busy")):
            with self.assertRaisesRegex(RuntimeError, "Cannot listen"):
                await self.login.handle_redirect("https://login.example.com/authorize?state=new")
        self.browser.assert_not_called()


if __name__ == "__main__":
    unittest.main()
