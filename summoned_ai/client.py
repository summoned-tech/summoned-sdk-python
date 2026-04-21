"""
Summoned AI Gateway — Python SDK

Thin HTTP client that speaks the OpenAI API format and sends
requests to the Summoned gateway. All provider routing, retries,
caching, and guardrails happen server-side.

Usage:
    from summoned_ai import Summoned

    client = Summoned(api_key="sk-smnd-...")
    response = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(response["choices"][0]["message"]["content"])
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import httpx

__all__ = [
    "Summoned",
    "AsyncSummoned",
    "SummonedError",
    "create_headers",
]

DEFAULT_BASE_URL = "http://localhost:4000"
SDK_VERSION = "0.2.0"

logger = logging.getLogger("summoned_ai")


# ---------------------------------------------------------------------------
# create_headers — for users who want to use OpenAI's SDK via our gateway
# ---------------------------------------------------------------------------

def create_headers(
    *,
    config: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Generate HTTP headers to route an OpenAI SDK request through the Summoned gateway.

    Usage with the official OpenAI Python SDK:

        from openai import OpenAI
        from summoned_ai import create_headers

        client = OpenAI(
            base_url="http://localhost:4000/v1",
            api_key="sk-smnd-...",
            default_headers=create_headers(config={"cache": True}),
        )
    """
    headers: Dict[str, str] = {"x-summoned-sdk": f"py-{SDK_VERSION}"}
    merged: Dict[str, Any] = {**(config or {})}
    if trace_id:
        merged["traceId"] = trace_id
    if metadata:
        merged["metadata"] = metadata
    if merged:
        headers["x-summoned-config"] = base64.b64encode(json.dumps(merged).encode()).decode()
    return headers


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class SummonedError(Exception):
    """Error returned by the Summoned gateway."""

    def __init__(
        self,
        message: str,
        status_code: int = 0,
        code: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.headers = headers or {}


# ---------------------------------------------------------------------------
# Response header parsing
# ---------------------------------------------------------------------------

@dataclass
class ResponseHeaders:
    provider: Optional[str] = None
    served_by: Optional[str] = None
    cost_usd: Optional[str] = None
    latency_ms: Optional[str] = None
    trace_id: Optional[str] = None
    cache: Optional[str] = None
    rate_limit_limit: Optional[str] = None
    rate_limit_remaining: Optional[str] = None


def _parse_response_headers(headers: httpx.Headers) -> ResponseHeaders:
    return ResponseHeaders(
        provider=headers.get("x-summoned-provider"),
        served_by=headers.get("x-summoned-served-by"),
        cost_usd=headers.get("x-summoned-cost-usd"),
        latency_ms=headers.get("x-summoned-latency-ms"),
        trace_id=headers.get("x-summoned-trace-id"),
        cache=headers.get("x-summoned-cache"),
        rate_limit_limit=headers.get("x-ratelimit-limit"),
        rate_limit_remaining=headers.get("x-ratelimit-remaining"),
    )


def _encode_config(config: Dict[str, Any]) -> str:
    return base64.b64encode(json.dumps(config).encode()).decode()


# ---------------------------------------------------------------------------
# Base client
# ---------------------------------------------------------------------------

class _BaseClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        admin_key: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        debug: bool = False,
        default_config: Optional[Dict[str, Any]] = None,
    ):
        self.api_key = api_key or os.environ.get("SUMMONED_API_KEY", "")
        if not self.api_key:
            raise ValueError("api_key is required (or set SUMMONED_API_KEY env var)")
        self.base_url = (base_url or os.environ.get("SUMMONED_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.admin_key = admin_key or os.environ.get("SUMMONED_ADMIN_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.debug = debug
        self.default_config = default_config
        self.last_response_headers = ResponseHeaders()

    def _build_headers(
        self,
        config: Optional[Dict[str, Any]] = None,
        use_admin_auth: bool = False,
    ) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "x-summoned-sdk": f"py-{SDK_VERSION}",
        }
        if use_admin_auth:
            if not self.admin_key:
                raise ValueError("admin_key is required for admin operations")
            headers["x-admin-key"] = self.admin_key
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        merged = {**(self.default_config or {}), **(config or {})}
        if merged:
            headers["x-summoned-config"] = _encode_config(merged)
        return headers

    def with_config(self, config: Dict[str, Any]) -> "Summoned":
        """Return a new client with merged config for subsequent requests."""
        merged = {**(self.default_config or {}), **config}
        return Summoned(
            api_key=self.api_key,
            base_url=self.base_url,
            admin_key=self.admin_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
            debug=self.debug,
            default_config=merged,
        )


# ---------------------------------------------------------------------------
# Sync chat completions
# ---------------------------------------------------------------------------

def _merge_prompt_into_config(
    config: Optional[Dict[str, Any]],
    prompt_id: Optional[str],
    prompt_variables: Optional[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    """Merge ``prompt_id`` / ``prompt_variables`` kwargs into the config dict.

    Uses camelCase wire keys (``promptId``, ``promptVariables``) so the JSON
    payload is identical to what the TypeScript SDK sends.
    """
    if prompt_id is None and prompt_variables is None:
        return config
    merged = dict(config or {})
    if prompt_id is not None:
        merged["promptId"] = prompt_id
    if prompt_variables is not None:
        merged["promptVariables"] = prompt_variables
    return merged


class _ChatCompletions:
    def __init__(self, client: "Summoned"):
        self._client = client

    def create(
        self,
        *,
        model: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        fallback_models: Optional[List[str]] = None,
        prompt_id: Optional[str] = None,
        prompt_variables: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """Create a chat completion.

        When ``prompt_id`` is set, the server-side prompt template is
        interpolated with ``prompt_variables`` and prepended to ``messages``.
        If the prompt row has a ``defaultModel``, ``model`` may be omitted
        (pass ``""`` or leave unset).
        """
        body: Dict[str, Any] = {"model": model, "messages": messages or []}
        if stream:
            body["stream"] = True
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if tools:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if top_p is not None:
            body["top_p"] = top_p
        if stop is not None:
            body["stop"] = stop
        if fallback_models:
            body["fallback_models"] = fallback_models
        body.update(kwargs)

        config = _merge_prompt_into_config(config, prompt_id, prompt_variables)

        if stream:
            return self._client._stream("/v1/chat/completions", body, config)
        return self._client._request("POST", "/v1/chat/completions", body, config=config)


class _Chat:
    def __init__(self, client: "Summoned"):
        self.completions = _ChatCompletions(client)


class _Embeddings:
    def __init__(self, client: "Summoned"):
        self._client = client

    def create(self, *, model: str, input: Union[str, List[str]], encoding_format: str = "float") -> Dict[str, Any]:
        return self._client._request("POST", "/v1/embeddings", {"model": model, "input": input, "encoding_format": encoding_format})


class _Models:
    def __init__(self, client: "Summoned"):
        self._client = client

    def list(self) -> Dict[str, Any]:
        return self._client._request("GET", "/v1/models")


# ---------------------------------------------------------------------------
# Admin APIs
# ---------------------------------------------------------------------------

class _AdminKeys:
    def __init__(self, client: "Summoned"):
        self._client = client

    def create(self, *, name: str, tenant_id: str, rate_limit_rpm: int = 60, rate_limit_tpd: int = 1_000_000) -> Dict[str, Any]:
        return self._client._request("POST", "/v1/keys", {"name": name, "tenantId": tenant_id, "rateLimitRpm": rate_limit_rpm, "rateLimitTpd": rate_limit_tpd}, use_admin_auth=True)

    def list(self, tenant_id: str) -> Dict[str, Any]:
        return self._client._request("GET", f"/v1/keys?tenantId={tenant_id}", use_admin_auth=True)

    def revoke(self, key_id: str) -> Dict[str, Any]:
        return self._client._request("DELETE", f"/v1/keys/{key_id}", use_admin_auth=True)


class _AdminVirtualKeys:
    def __init__(self, client: "Summoned"):
        self._client = client

    def create(self, *, name: str, tenant_id: str, provider_id: str, api_key: str, provider_config: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"name": name, "tenantId": tenant_id, "providerId": provider_id, "apiKey": api_key}
        if provider_config:
            body["providerConfig"] = provider_config
        return self._client._request("POST", "/admin/virtual-keys", body, use_admin_auth=True)

    def list(self, tenant_id: str) -> Dict[str, Any]:
        return self._client._request("GET", f"/admin/virtual-keys?tenantId={tenant_id}", use_admin_auth=True)

    def revoke(self, vk_id: str) -> Dict[str, Any]:
        return self._client._request("DELETE", f"/admin/virtual-keys/{vk_id}", use_admin_auth=True)


class _AdminLogs:
    def __init__(self, client: "Summoned"):
        self._client = client

    def list(self, *, limit: int = 100, source: str = "buffer", tenant_id: Optional[str] = None) -> Dict[str, Any]:
        params = f"?limit={limit}&source={source}"
        if tenant_id:
            params += f"&tenantId={tenant_id}"
        return self._client._request("GET", f"/admin/logs{params}", use_admin_auth=True)


class _AdminStats:
    def __init__(self, client: "Summoned"):
        self._client = client

    def get(self, period: str = "24h") -> Dict[str, Any]:
        return self._client._request("GET", f"/admin/stats?period={period}", use_admin_auth=True)


class _AdminProviders:
    def __init__(self, client: "Summoned"):
        self._client = client

    def list(self) -> Dict[str, Any]:
        return self._client._request("GET", "/admin/providers", use_admin_auth=True)


class _AdminPrompts:
    """Versioned prompt templates. See rfcs/0001-prompt-management.md in the
    gateway repo for the underlying model."""

    def __init__(self, client: "Summoned"):
        self._client = client

    def create(
        self,
        *,
        slug: str,
        tenant_id: str,
        template: List[Dict[str, Any]],
        variables: Optional[Dict[str, str]] = None,
        default_model: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new prompt version. If a prompt with the same (tenant, slug)
        exists, the version auto-increments and the new row becomes latest."""
        body: Dict[str, Any] = {"slug": slug, "tenantId": tenant_id, "template": template}
        if variables is not None:
            body["variables"] = variables
        if default_model is not None:
            body["defaultModel"] = default_model
        if description is not None:
            body["description"] = description
        return self._client._request("POST", "/admin/prompts", body, use_admin_auth=True)

    def list(self, tenant_id: str) -> Dict[str, Any]:
        """List the latest version of every prompt for a tenant."""
        return self._client._request(
            "GET", f"/admin/prompts?tenantId={tenant_id}", use_admin_auth=True,
        )

    def get(self, ref: str, *, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Fetch a prompt by primary key (``prm_...``) or by slug.

        When ``ref`` is a slug, ``tenant_id`` is required and the latest
        version is returned.
        """
        if ref.startswith("prm_"):
            return self._client._request("GET", f"/admin/prompts/{ref}", use_admin_auth=True)
        if not tenant_id:
            raise ValueError("tenant_id is required when fetching by slug")
        return self._client._request(
            "GET", f"/admin/prompts/by-slug/{ref}?tenantId={tenant_id}", use_admin_auth=True,
        )

    def versions(self, slug: str, *, tenant_id: str) -> Dict[str, Any]:
        """Version history for a slug, newest first."""
        return self._client._request(
            "GET", f"/admin/prompts/{slug}/versions?tenantId={tenant_id}", use_admin_auth=True,
        )

    def delete(self, prompt_id: str) -> Dict[str, Any]:
        """Soft-delete a prompt version. If it was the latest, the next-highest
        active version is promoted to latest."""
        return self._client._request("DELETE", f"/admin/prompts/{prompt_id}", use_admin_auth=True)


class _Admin:
    def __init__(self, client: "Summoned"):
        self.keys = _AdminKeys(client)
        self.virtual_keys = _AdminVirtualKeys(client)
        self.prompts = _AdminPrompts(client)
        self.logs = _AdminLogs(client)
        self.stats = _AdminStats(client)
        self.providers = _AdminProviders(client)


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------

class Summoned(_BaseClient):
    """
    Synchronous Summoned AI Gateway client.

    Usage::

        client = Summoned(api_key="sk-smnd-...")
        res = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._http = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        self.chat = _Chat(self)
        self.embeddings = _Embeddings(self)
        self.models = _Models(self)
        self.admin = _Admin(self)

    def _request(
        self,
        method: str,
        path: str,
        body: Any = None,
        config: Optional[Dict[str, Any]] = None,
        use_admin_auth: bool = False,
    ) -> Any:
        headers = self._build_headers(config, use_admin_auth)
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                delay = min(0.5 * (2 ** (attempt - 1)), 5.0)
                if self.debug:
                    logger.debug("retry %d/%d after %.1fs", attempt, self.max_retries, delay)
                time.sleep(delay)

            try:
                if self.debug:
                    logger.debug("%s %s%s %s", method, self.base_url, path, json.dumps(body)[:200] if body else "")

                resp = self._http.request(method, path, json=body, headers=headers)
                self.last_response_headers = _parse_response_headers(resp.headers)

                if self.debug:
                    h = self.last_response_headers
                    logger.debug("%d | provider=%s cache=%s latency=%sms", resp.status_code, h.provider, h.cache, h.latency_ms)

                if resp.status_code >= 500 and attempt < self.max_retries:
                    last_error = SummonedError(resp.text, resp.status_code)
                    continue

                if resp.status_code >= 400:
                    ct = resp.headers.get("content-type", "")
                    err_body = resp.json() if "json" in ct else {}
                    raise SummonedError(
                        err_body.get("error", {}).get("message", resp.text),
                        status_code=resp.status_code,
                        code=err_body.get("error", {}).get("code"),
                        headers=vars(self.last_response_headers),
                    )

                return resp.json()

            except SummonedError:
                raise
            except Exception as e:
                last_error = e
                if attempt == self.max_retries:
                    break

        raise last_error or SummonedError("Request failed after retries")

    def _stream(self, path: str, body: Any, config: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
        headers = self._build_headers(config)
        with self._http.stream("POST", path, json=body, headers=headers) as resp:
            self.last_response_headers = _parse_response_headers(resp.headers)
            if resp.status_code >= 400:
                resp.read()
                raise SummonedError(resp.text, status_code=resp.status_code)
            for line in resp.iter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    return
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Summoned":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Async chat completions
# ---------------------------------------------------------------------------

class _AsyncChatCompletions:
    def __init__(self, client: "AsyncSummoned"):
        self._client = client

    async def create(
        self,
        *,
        model: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        fallback_models: Optional[List[str]] = None,
        prompt_id: Optional[str] = None,
        prompt_variables: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        body: Dict[str, Any] = {"model": model, "messages": messages or []}
        if stream:
            body["stream"] = True
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if tools:
            body["tools"] = tools
        if fallback_models:
            body["fallback_models"] = fallback_models
        body.update(kwargs)

        config = _merge_prompt_into_config(config, prompt_id, prompt_variables)

        if stream:
            return self._client._stream("/v1/chat/completions", body, config)
        return await self._client._request("POST", "/v1/chat/completions", body, config=config)


class _AsyncChat:
    def __init__(self, client: "AsyncSummoned"):
        self.completions = _AsyncChatCompletions(client)


class _AsyncEmbeddings:
    def __init__(self, client: "AsyncSummoned"):
        self._client = client

    async def create(self, *, model: str, input: Union[str, List[str]], encoding_format: str = "float") -> Dict[str, Any]:
        return await self._client._request("POST", "/v1/embeddings", {"model": model, "input": input, "encoding_format": encoding_format})


class _AsyncModels:
    def __init__(self, client: "AsyncSummoned"):
        self._client = client

    async def list(self) -> Dict[str, Any]:
        return await self._client._request("GET", "/v1/models")


class _AsyncAdmin:
    def __init__(self, client: "AsyncSummoned"):
        self.keys = _AsyncAdminKeys(client)
        self.virtual_keys = _AsyncAdminVirtualKeys(client)
        self.prompts = _AsyncAdminPrompts(client)
        self.logs = _AsyncAdminLogs(client)
        self.stats = _AsyncAdminStats(client)
        self.providers = _AsyncAdminProviders(client)


class _AsyncAdminKeys:
    def __init__(self, client: "AsyncSummoned"):
        self._client = client

    async def create(self, *, name: str, tenant_id: str, rate_limit_rpm: int = 60, rate_limit_tpd: int = 1_000_000) -> Dict[str, Any]:
        return await self._client._request("POST", "/v1/keys", {"name": name, "tenantId": tenant_id, "rateLimitRpm": rate_limit_rpm, "rateLimitTpd": rate_limit_tpd}, use_admin_auth=True)

    async def list(self, tenant_id: str) -> Dict[str, Any]:
        return await self._client._request("GET", f"/v1/keys?tenantId={tenant_id}", use_admin_auth=True)

    async def revoke(self, key_id: str) -> Dict[str, Any]:
        return await self._client._request("DELETE", f"/v1/keys/{key_id}", use_admin_auth=True)


class _AsyncAdminVirtualKeys:
    def __init__(self, client: "AsyncSummoned"):
        self._client = client

    async def create(self, *, name: str, tenant_id: str, provider_id: str, api_key: str, provider_config: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"name": name, "tenantId": tenant_id, "providerId": provider_id, "apiKey": api_key}
        if provider_config:
            body["providerConfig"] = provider_config
        return await self._client._request("POST", "/admin/virtual-keys", body, use_admin_auth=True)

    async def list(self, tenant_id: str) -> Dict[str, Any]:
        return await self._client._request("GET", f"/admin/virtual-keys?tenantId={tenant_id}", use_admin_auth=True)

    async def revoke(self, vk_id: str) -> Dict[str, Any]:
        return await self._client._request("DELETE", f"/admin/virtual-keys/{vk_id}", use_admin_auth=True)


class _AsyncAdminLogs:
    def __init__(self, client: "AsyncSummoned"):
        self._client = client

    async def list(self, *, limit: int = 100, source: str = "buffer", tenant_id: Optional[str] = None) -> Dict[str, Any]:
        params = f"?limit={limit}&source={source}"
        if tenant_id:
            params += f"&tenantId={tenant_id}"
        return await self._client._request("GET", f"/admin/logs{params}", use_admin_auth=True)


class _AsyncAdminStats:
    def __init__(self, client: "AsyncSummoned"):
        self._client = client

    async def get(self, period: str = "24h") -> Dict[str, Any]:
        return await self._client._request("GET", f"/admin/stats?period={period}", use_admin_auth=True)


class _AsyncAdminProviders:
    def __init__(self, client: "AsyncSummoned"):
        self._client = client

    async def list(self) -> Dict[str, Any]:
        return await self._client._request("GET", "/admin/providers", use_admin_auth=True)


class _AsyncAdminPrompts:
    """Async mirror of _AdminPrompts."""

    def __init__(self, client: "AsyncSummoned"):
        self._client = client

    async def create(
        self,
        *,
        slug: str,
        tenant_id: str,
        template: List[Dict[str, Any]],
        variables: Optional[Dict[str, str]] = None,
        default_model: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"slug": slug, "tenantId": tenant_id, "template": template}
        if variables is not None:
            body["variables"] = variables
        if default_model is not None:
            body["defaultModel"] = default_model
        if description is not None:
            body["description"] = description
        return await self._client._request("POST", "/admin/prompts", body, use_admin_auth=True)

    async def list(self, tenant_id: str) -> Dict[str, Any]:
        return await self._client._request(
            "GET", f"/admin/prompts?tenantId={tenant_id}", use_admin_auth=True,
        )

    async def get(self, ref: str, *, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        if ref.startswith("prm_"):
            return await self._client._request("GET", f"/admin/prompts/{ref}", use_admin_auth=True)
        if not tenant_id:
            raise ValueError("tenant_id is required when fetching by slug")
        return await self._client._request(
            "GET", f"/admin/prompts/by-slug/{ref}?tenantId={tenant_id}", use_admin_auth=True,
        )

    async def versions(self, slug: str, *, tenant_id: str) -> Dict[str, Any]:
        return await self._client._request(
            "GET", f"/admin/prompts/{slug}/versions?tenantId={tenant_id}", use_admin_auth=True,
        )

    async def delete(self, prompt_id: str) -> Dict[str, Any]:
        return await self._client._request(
            "DELETE", f"/admin/prompts/{prompt_id}", use_admin_auth=True,
        )


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------

class AsyncSummoned(_BaseClient):
    """
    Asynchronous Summoned AI Gateway client.

    Usage::

        async with AsyncSummoned(api_key="sk-smnd-...") as client:
            res = await client.chat.completions.create(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        self.chat = _AsyncChat(self)
        self.embeddings = _AsyncEmbeddings(self)
        self.models = _AsyncModels(self)
        self.admin = _AsyncAdmin(self)

    async def _request(
        self,
        method: str,
        path: str,
        body: Any = None,
        config: Optional[Dict[str, Any]] = None,
        use_admin_auth: bool = False,
    ) -> Any:
        headers = self._build_headers(config, use_admin_auth)
        last_error: Optional[Exception] = None
        import asyncio

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                delay = min(0.5 * (2 ** (attempt - 1)), 5.0)
                await asyncio.sleep(delay)

            try:
                resp = await self._http.request(method, path, json=body, headers=headers)
                self.last_response_headers = _parse_response_headers(resp.headers)

                if resp.status_code >= 500 and attempt < self.max_retries:
                    last_error = SummonedError(resp.text, resp.status_code)
                    continue

                if resp.status_code >= 400:
                    ct = resp.headers.get("content-type", "")
                    err_body = resp.json() if "json" in ct else {}
                    raise SummonedError(
                        err_body.get("error", {}).get("message", resp.text),
                        status_code=resp.status_code,
                        code=err_body.get("error", {}).get("code"),
                        headers=vars(self.last_response_headers),
                    )

                return resp.json()

            except SummonedError:
                raise
            except Exception as e:
                last_error = e
                if attempt == self.max_retries:
                    break

        raise last_error or SummonedError("Request failed after retries")

    async def _stream(self, path: str, body: Any, config: Optional[Dict[str, Any]] = None) -> AsyncIterator[Dict[str, Any]]:
        headers = self._build_headers(config)
        async with self._http.stream("POST", path, json=body, headers=headers) as resp:
            self.last_response_headers = _parse_response_headers(resp.headers)
            if resp.status_code >= 400:
                await resp.aread()
                raise SummonedError(resp.text, status_code=resp.status_code)
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    return
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "AsyncSummoned":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
