# CCAS — Claude Code API Server

An OpenAI-compatible HTTP proxy that routes requests through the Claude Agent SDK, using your Claude Max subscription. No API key required.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions (streaming + non-streaming) |
| `GET` | `/v1/models` | List available Claude models |
| `POST` | `/api/chat` | Ollama-compatible chat endpoint |
| `GET` | `/api/tags` | Ollama-compatible model list |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python server.py
```

Server starts on `http://localhost:11435`.

## Usage

Point any OpenAI-compatible client at `http://localhost:11435/v1`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11435/v1", api_key="unused")
response = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

## Model Aliases

Common OpenAI model names are mapped to Claude equivalents:

| OpenAI | Claude |
|--------|--------|
| `gpt-4`, `gpt-4o`, `gpt-4-turbo` | `claude-sonnet-4-6` |
| `gpt-3.5-turbo` | `claude-haiku-4-5` |

## Dependencies

- [`claude-agent-sdk`](https://pypi.org/project/claude-agent-sdk/) — Claude Code Agent SDK
- `fastapi` — HTTP framework
- `uvicorn` — ASGI server
