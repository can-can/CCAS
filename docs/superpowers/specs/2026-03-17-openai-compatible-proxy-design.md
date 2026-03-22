# OpenAI-Compatible API Proxy via Claude Agent SDK

## Overview

A FastAPI HTTP server (port 11435) that accepts OpenAI-format `/v1/chat/completions` requests and proxies them through the Claude Agent SDK, authenticating via the user's Claude Max subscription. No API key required.

## Endpoints

### POST `/v1/chat/completions`

Accepts OpenAI-format request body:
- `model` — passed through to Agent SDK (also maps `gpt-4`/`gpt-4o` → `claude-sonnet-4-6`)
- `messages` — array of `{role, content}` objects; content can be string or array with text/image blocks
- `stream` — boolean; when true, returns SSE
- `max_tokens` — optional, forwarded
- `temperature` — optional, forwarded

### GET `/v1/models`

Returns hardcoded list of available Claude models in OpenAI format.

## Message Conversion

### System messages
Extracted from the messages array and passed as `system_prompt` in `ClaudeAgentOptions`.

### Text messages
Converted from OpenAI `{role, content: "string"}` to Anthropic `{role, content: "string"}`.

### Multi-turn
All messages (except system) are sent as a sequence via `AsyncIterable[dict]` to the Agent SDK's `query()` function using `ClaudeSDKClient` for multi-message support.

### Image messages
OpenAI format:
```json
{"type": "image_url", "image_url": {"url": "data:image/png;base64,DATA"}}
```
Converted to Anthropic format:
```json
{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "DATA"}}
```
HTTP(S) URLs converted to:
```json
{"type": "image", "source": {"type": "url", "url": "https://..."}}
```

## Streaming

When `stream: true`:
- Returns `text/event-stream` with SSE chunks matching OpenAI format
- Each chunk: `data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":"..."}}]}`
- Terminates with `data: [DONE]`

When `stream: false`:
- Collects full response and returns single JSON object matching OpenAI response schema

## Agent SDK Configuration

```python
ClaudeAgentOptions(
    setting_sources=[],      # No CLAUDE.md or settings loading
    allowed_tools=[],        # No tools — pure chat proxy
    model=<from request>,
    system_prompt=<from system message>,
    permission_mode="bypassPermissions",
)
```

## Dependencies

- `claude-agent-sdk` — Claude Agent SDK (uses Claude Code CLI)
- `fastapi` — HTTP framework
- `uvicorn` — ASGI server

## File Structure

```
CCAS/
├── server.py          # The server
├── requirements.txt   # Dependencies
└── .venv/             # Virtual environment
```
