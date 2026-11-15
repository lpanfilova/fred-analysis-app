"""Persist FRED OAuth credentials"""

import asyncio
import json
import time

from keyring.backends.macOS import Keyring
from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthClientInformationFull, OAuthMetadata, OAuthToken


class KeychainTokenStorage:
    def __init__(self, server_url, backend=None):
        self.backend = backend if backend is not None else Keyring()
        self.service = f"fred-analysis-app:{server_url}"
        self.expires_at = None
        self.metadata = None

    async def _read(self, account):
        value = await asyncio.to_thread(self.backend.get_password, self.service, account)
        return json.loads(value) if value is not None else None

    async def _write(self, account, value):
        await asyncio.to_thread(
            self.backend.set_password, self.service, account, json.dumps(value)
        )

    async def get_tokens(self):
        saved = await self._read("tokens")
        if saved is None:
            return None
        self.expires_at = saved["expires_at"]
        self.metadata = (
            OAuthMetadata.model_validate(saved["metadata"])
            if saved["metadata"] is not None else None
        )
        return OAuthToken.model_validate(saved["tokens"])

    async def set_tokens(self, tokens):
        self.expires_at = (
            time.time() + tokens.expires_in if tokens.expires_in is not None else None
        )
        await self._write("tokens", {
            "tokens": tokens.model_dump(mode="json"),
            "expires_at": self.expires_at,
            "metadata": (
                self.metadata.model_dump(mode="json")
                if self.metadata is not None else None
            ),
        })

    async def get_client_info(self):
        saved = await self._read("client")
        return OAuthClientInformationFull.model_validate(saved) if saved else None

    async def set_client_info(self, client_info):
        await self._write("client", client_info.model_dump(mode="json"))


class PersistentOAuthClientProvider(OAuthClientProvider):
    """Restore token expiry and authorization metadata across application restarts."""

    async def _initialize(self):
        await super()._initialize()
        self.context.token_expiry_time = self.context.storage.expires_at
        self.context.oauth_metadata = self.context.storage.metadata

    async def _handle_token_response(self, response):
        self.context.storage.metadata = self.context.oauth_metadata
        await super()._handle_token_response(response)

    async def _handle_refresh_response(self, response):
        self.context.storage.metadata = self.context.oauth_metadata
        return await super()._handle_refresh_response(response)
