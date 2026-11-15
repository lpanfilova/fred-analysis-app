"""Automatic browser login with a short-lived loopback callback server."""

import asyncio
import secrets
import webbrowser
from urllib.parse import parse_qs, urlsplit

from mcp.client.auth import AuthorizationCodeResult


class BrowserLogin:
    def __init__(self, port=3030, timeout=300):
        self.port = port
        self.timeout = timeout
        self.redirect_uri = f"http://localhost:{port}/callback"
        self._server = None
        self._result = None
        self._state = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self._close()

    async def _close(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def handle_redirect(self, auth_url):
        await self._close()
        self._state = parse_qs(urlsplit(auth_url).query)["state"][0]
        self._result = asyncio.get_running_loop().create_future()
        try:
            self._server = await asyncio.start_server(
                self._handle_request, host="localhost", port=self.port, limit=8192
            )
        except OSError as exc:
            raise RuntimeError(
                f"Cannot listen on localhost:{self.port}. "
                "Close any other app using that port and try again."
            ) from exc
        print("Opening your browser to authorize FRED access...")
        try:
            opened = await asyncio.to_thread(webbrowser.open, auth_url)
        except webbrowser.Error:
            opened = False
        if not opened:
            print(f"Open this URL to continue:\n{auth_url}")

    async def handle_callback(self):
        try:
            return await asyncio.wait_for(self._result, timeout=self.timeout)
        except TimeoutError:
            raise RuntimeError(
                "FRED sign-in timed out. Run the program again."
            ) from None
        finally:
            await self._close()

    async def _handle_request(self, reader, writer):
        status, message = "400 Bad Request", "Invalid authorization callback."
        try:
            headers = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
            method, target, _ = headers.split(b"\r\n", 1)[0].decode("ascii").split()
            url = urlsplit(target)
            params = parse_qs(url.query)
            state = params.get("state", [""])[0]
            if method != "GET" or url.path != "/callback":
                status, message = "404 Not Found", "Not found."
            elif not state or not secrets.compare_digest(state, self._state):
                message = "Authorization state did not match. Continue in the original browser tab."
            elif self._result.done():
                message = "Authorization callback already received."
            elif "error" in params:
                self._result.set_exception(RuntimeError(
                    "FRED authorization was denied or failed. Run the program to try again."
                ))
                message = "Authorization failed. You can close this tab."
            elif params.get("code"):
                self._result.set_result(AuthorizationCodeResult(
                    code=params["code"][0], state=state,
                    iss=params.get("iss", [None])[0],
                ))
                status = "200 OK"
                message = "Authorization response received. You can close this tab and return to the app."
        except (ValueError, UnicodeError, asyncio.IncompleteReadError,
                asyncio.LimitOverrunError, TimeoutError):
            pass
        finally:
            body = message.encode("utf-8")
            writer.write((
                f"HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\n"
                f"Content-Length: {len(body)}\r\nCache-Control: no-store\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii") + body)
            try:
                await writer.drain()
            except ConnectionError:
                pass
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
