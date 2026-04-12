# summoned-ai

Python SDK for the [Summoned AI Gateway](https://github.com/summoned-tech/summoned-ai-gateway) — OpenAI-compatible client with multi-provider routing, caching, guardrails, and more.

## Install

```bash
pip install summoned-ai
```

## Quick Start

```python
from summoned_ai import Summoned

client = Summoned(api_key="sk-smnd-...", base_url="http://localhost:4000")

response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)

print(response["choices"][0]["message"]["content"])
print(response["summoned"])  # provider, cost, latency_ms, ...
```

## Streaming

```python
for chunk in client.chat.completions.create(
    model="anthropic/claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True,
):
    print(chunk["choices"][0]["delta"].get("content", ""), end="")
```

## Config — Retries, Fallbacks, Caching, Guardrails

```python
response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    config={
        "retry": {"attempts": 3, "backoff": "exponential"},
        "fallback": ["anthropic/claude-sonnet-4-20250514", "groq/llama-3.3-70b-versatile"],
        "timeout": 30000,
        "cache": True,
        "guardrails": {
            "input": [{"type": "pii", "deny": True}],
            "output": [{"type": "contains", "params": {"operator": "none", "words": ["confidential"]}, "deny": True}],
        },
    },
)
```

## `with_config` — Reusable Client Configuration

```python
cached_client = client.with_config({"cache": True, "cacheTtl": 3600})

# All requests through cached_client use caching
cached_client.chat.completions.create(model="openai/gpt-4o", messages=[...])
```

## Use with OpenAI's SDK

```python
from openai import OpenAI
from summoned_ai import create_headers

client = OpenAI(
    base_url="http://localhost:4000/v1",
    api_key="sk-smnd-...",
    default_headers=create_headers(config={"cache": True}),
)

# Routes through the Summoned gateway with all features
res = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Async Client

```python
from summoned_ai import AsyncSummoned

async with AsyncSummoned(api_key="sk-smnd-...") as client:
    response = await client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )
```

## Embeddings

```python
result = client.embeddings.create(
    model="openai/text-embedding-3-small",
    input="The quick brown fox",
)
```

## Admin API

```python
client = Summoned(api_key="sk-smnd-...", admin_key="your-admin-key")

# API keys
key = client.admin.keys.create(name="production", tenant_id="tenant_1")
keys = client.admin.keys.list("tenant_1")
client.admin.keys.revoke("key_abc")

# Virtual keys (encrypted provider credentials)
vk = client.admin.virtual_keys.create(
    name="my-openai-key",
    tenant_id="tenant_1",
    provider_id="openai",
    api_key="sk-real-openai-key-...",
)

# Logs & stats
logs = client.admin.logs.list(limit=50)
stats = client.admin.stats.get("24h")
providers = client.admin.providers.list()
```

## Debug Mode

```python
client = Summoned(api_key="sk-smnd-...", debug=True)
# Logs request/response details via Python's logging module
```

## Response Headers

```python
client.chat.completions.create(model="openai/gpt-4o", messages=[...])

print(client.last_response_headers)
# ResponseHeaders(provider='openai', cache='MISS', latency_ms='432', ...)
```

## Error Handling

```python
from summoned_ai import SummonedError

try:
    client.chat.completions.create(model="openai/gpt-4o", messages=[...])
except SummonedError as e:
    print(e.status_code)  # 429
    print(e.code)         # "RATE_LIMITED"
    print(e.headers)      # response headers
```
