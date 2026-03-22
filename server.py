import json
import re
import time
import traceback
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from claude_agent_sdk import (
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)

app = FastAPI(title="CCAS - Claude Code API Server")

# Model aliases: map common OpenAI model names to Claude models
MODEL_MAP = {
    "gpt-4": "claude-sonnet-4-6",
    "gpt-4o": "claude-sonnet-4-6",
    "gpt-4o-mini": "claude-haiku-4-5",
    "gpt-4-turbo": "claude-sonnet-4-6",
    "gpt-3.5-turbo": "claude-haiku-4-5",
}

AVAILABLE_MODELS = [
    {"id": "claude-opus-4-6", "object": "model", "created": 1700000000, "owned_by": "anthropic"},
    {"id": "claude-sonnet-4-6", "object": "model", "created": 1700000000, "owned_by": "anthropic"},
    {"id": "claude-haiku-4-5", "object": "model", "created": 1700000000, "owned_by": "anthropic"},
]


# --- Request/Response Models ---


class ChatMessage(BaseModel):
    role: str
    content: Any  # string or list of content blocks


class ChatCompletionRequest(BaseModel):
    model: str = "claude-sonnet-4-6"
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None


# --- Conversion Helpers ---


def resolve_model(model: str) -> str:
    return MODEL_MAP.get(model, model)


def convert_image_block(block: dict) -> dict:
    """Convert OpenAI image_url block to Anthropic image block."""
    url = block["image_url"]["url"]
    if url.startswith("data:"):
        header, data = url.split(",", 1)
        media_type = header.split(":")[1].split(";")[0]
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    else:
        return {
            "type": "image",
            "source": {"type": "url", "url": url},
        }


def convert_content_blocks(content: Any) -> list[dict]:
    """Convert OpenAI content to Anthropic content block list."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        result = []
        for block in content:
            if block["type"] == "text":
                result.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image_url":
                result.append(convert_image_block(block))
        return result
    return [{"type": "text", "text": str(content)}]


def get_text_from_content(content: Any) -> str:
    """Extract plain text from content (string or block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if b.get("type") == "text")
    return str(content)


def has_non_text_blocks(content: Any) -> bool:
    """Check if content contains non-text blocks (images, etc)."""
    if isinstance(content, str):
        return False
    if isinstance(content, list):
        return any(b.get("type") != "text" for b in content)
    return False


def build_prompt(messages: list[ChatMessage]) -> tuple[str | None, Any]:
    """
    Build system prompt and a single user prompt from the message list.

    For multi-turn conversations, prior messages are prepended as context text.
    The last user message's content (including images) is preserved as-is.

    Returns (system_prompt, prompt) where prompt is either a string or
    an AsyncIterable for content with images.
    """
    system_prompt = None
    non_system: list[ChatMessage] = []

    for msg in messages:
        if msg.role == "system":
            system_prompt = get_text_from_content(msg.content)
        else:
            non_system.append(msg)

    if not non_system:
        return system_prompt, ""

    # Single user message, text only → simple string
    if len(non_system) == 1 and non_system[0].role == "user" and not has_non_text_blocks(non_system[0].content):
        return system_prompt, get_text_from_content(non_system[0].content)

    # Build context from prior messages, keep last user message's full content
    last_msg = non_system[-1]
    prior = non_system[:-1]

    context_lines = []
    for msg in prior:
        label = "Human" if msg.role == "user" else "Assistant"
        context_lines.append(f"{label}: {get_text_from_content(msg.content)}")

    last_content_blocks = convert_content_blocks(last_msg.content)
    has_images = any(b.get("type") == "image" for b in last_content_blocks)

    if context_lines:
        context_text = "\n\n".join(context_lines)
        prefix = f"<conversation_history>\n{context_text}\n</conversation_history>\n\n"
    else:
        prefix = ""

    if has_images:
        # Build content block array: context as text block + last message's blocks
        content = []
        if prefix:
            content.append({"type": "text", "text": prefix})
        content.extend(last_content_blocks)
        return system_prompt, content  # list of content blocks
    else:
        # All text, build a single string
        last_text = get_text_from_content(last_msg.content)
        return system_prompt, prefix + last_text


def make_options(
    model: str,
    system_prompt: str | None,
    streaming: bool = False,
    disable_builtin_tools: bool = False,
) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        setting_sources=[],
        allowed_tools=[],
        # When caller injects tool descriptions via system prompt, disable all
        # built-in CLI tools so the model is forced to respond with plain text
        # instead of native ToolUseBlocks that the CLI would try to execute.
        tools=[] if disable_builtin_tools else None,
        model=model,
        system_prompt=system_prompt,
        permission_mode="bypassPermissions",
        max_turns=1,
        include_partial_messages=streaming,
    )


def make_completion_response(text: str, model: str, completion_id: str) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def make_chunk(delta: dict, model: str, completion_id: str, finish_reason: str | None = None) -> str:
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


# --- Core: query via Agent SDK ---


async def send_prompt(prompt: Any, options: ClaudeAgentOptions):
    """Send a prompt (string or content blocks) via the Agent SDK."""
    if isinstance(prompt, str):
        async for message in query(prompt=prompt, options=options):
            yield message
    else:
        # Content blocks (with images) — use AsyncIterable to send as stream-json
        async def message_stream():
            yield {
                "type": "user",
                "message": {"role": "user", "content": prompt},
                "parent_tool_use_id": None,
            }

        async for message in query(prompt=message_stream(), options=options):
            yield message


async def run_query(
    model: str,
    system_prompt: str | None,
    prompt: Any,
    *,
    has_tools: bool = False,
    disable_builtin_tools: bool = False,
) -> str:
    """Run a non-streaming query and return the full text response."""
    options = make_options(model, system_prompt, disable_builtin_tools=disable_builtin_tools or has_tools)
    result_text = ""
    assistant_text = ""
    tool_use_blocks: list[ToolUseBlock] = []
    async for message in send_prompt(prompt, options):
        if isinstance(message, ResultMessage):
            result_text = message.result or ""
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock) and block.text:
                    assistant_text += block.text
                elif isinstance(block, ToolUseBlock):
                    tool_use_blocks.append(block)

    # If the model responded with native ToolUseBlocks (instead of text-based
    # tool call JSON), synthesize the text representation so downstream
    # parse_tool_calls_from_text() can process it.  Only do this when the
    # caller actually wants tool-calling (has_tools=True); otherwise the
    # ToolUseBlocks are for built-in CLI tools and should be ignored.
    if has_tools and tool_use_blocks:
        tool_calls_json = {
            "tool_calls": [
                {"name": tb.name, "arguments": tb.input}
                for tb in tool_use_blocks
            ]
        }
        # Append tool call JSON after any text the model may have emitted
        assistant_text = (assistant_text + "\n" + json.dumps(tool_calls_json)).strip()

    # Prefer ResultMessage, fall back to accumulated AssistantMessage text
    return result_text or assistant_text


async def stream_query(
    model: str,
    system_prompt: str | None,
    prompt: Any,
    completion_id: str,
):
    """Stream a query and yield SSE chunks."""
    options = make_options(model, system_prompt, streaming=True)

    yield make_chunk({"role": "assistant"}, model, completion_id)

    try:
        async for message in send_prompt(prompt, options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text:
                        yield make_chunk({"content": block.text}, model, completion_id)
                    elif isinstance(block, ToolUseBlock):
                        # Synthesize text for native tool use blocks
                        tc_json = json.dumps({"tool_calls": [{"name": block.name, "arguments": block.input}]})
                        yield make_chunk({"content": tc_json}, model, completion_id)
    except Exception as e:
        error_text = f"Error: {e}"
        yield make_chunk({"content": error_text}, model, completion_id)

    yield make_chunk({}, model, completion_id, finish_reason="stop")
    yield "data: [DONE]\n\n"


# --- Ollama-compatible models ---


class OllamaChatMessage(BaseModel):
    role: str
    content: str
    images: list[str] | None = None  # base64-encoded image strings


class OllamaToolFunction(BaseModel):
    name: str
    description: str = ""
    parameters: dict | None = None


class OllamaTool(BaseModel):
    type: str = "function"
    function: OllamaToolFunction


class OllamaChatRequest(BaseModel):
    model: str = "claude-sonnet-4-6"
    messages: list[OllamaChatMessage]
    stream: bool = False
    options: dict | None = None  # temperature, etc.
    tools: list[OllamaTool] | None = None
    # rig also sends these fields — accept but ignore
    think: bool = False
    max_tokens: int | None = None
    keep_alive: str | None = None
    format: Any = None


TOOL_CALL_PROTOCOL = """

<tool_calling>
You have access to these tools. To call a tool, respond with ONLY a JSON object (no markdown fences, no extra text) in this exact format:
{{"tool_calls": [{{"name": "TOOL_NAME", "arguments": {{ARGS}}}}]}}

IMPORTANT: When your task requires calling a tool, output ONLY the JSON tool call — no explanation, no markdown tables, no summaries before or after.
After a tool has been executed and you receive its result, you may call another tool OR respond with plain text.
CRITICAL: After calling the 'done' tool and receiving its result, you MUST respond with plain text only (e.g., "Done."). Do NOT call any more tools after 'done'.

Available tools:
{tool_descriptions}
</tool_calling>"""

# Browser/action tools that require explicit tool-call protocol injection.
# When only "meta" tools (done, ask_user) are present, the model should respond
# with plain text — the caller parses the text response directly.
ACTION_TOOLS = {
    "snapshot", "fill", "select_option", "click", "press", "scroll",
    "screenshot", "upload", "type", "check",
}


def build_tool_system_supplement(tools: list[OllamaTool]) -> str:
    """Build a system prompt supplement describing available tools."""
    descriptions = []
    for tool in tools:
        fn = tool.function
        desc = f"- {fn.name}: {fn.description}"
        if fn.parameters:
            desc += f"\n  Parameters: {json.dumps(fn.parameters)}"
        descriptions.append(desc)
    return TOOL_CALL_PROTOCOL.format(tool_descriptions="\n".join(descriptions))


def parse_tool_calls_from_text(text: str) -> tuple[str, list[dict]]:
    """Parse tool calls from Claude's text response.

    Returns (remaining_text, tool_calls) where tool_calls is a list of
    Ollama-format tool call dicts.
    """
    # Strip markdown fences if present
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
        stripped = stripped.strip()

    # Try to parse the entire response as a tool_calls JSON
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and "tool_calls" in parsed:
            ollama_calls = []
            for tc in parsed["tool_calls"]:
                args = tc.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                ollama_calls.append({
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": args,
                    },
                })
            return "", ollama_calls
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Try to find a {"tool_calls": ...} JSON object embedded in text
    idx = stripped.find('"tool_calls"')
    if idx > 0:
        brace_idx = stripped.rfind('{', 0, idx)
        if brace_idx >= 0:
            try:
                decoder = json.JSONDecoder()
                parsed, end = decoder.raw_decode(stripped, brace_idx)
                if isinstance(parsed, dict) and "tool_calls" in parsed:
                    ollama_calls = []
                    for tc in parsed["tool_calls"]:
                        args = tc.get("arguments", {})
                        if isinstance(args, str):
                            args = json.loads(args)
                        ollama_calls.append({
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": args,
                            },
                        })
                    remaining = (stripped[:brace_idx] + stripped[end:]).strip()
                    return remaining, ollama_calls
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

    return text, []


def make_ollama_tool_response(text: str, tool_calls: list[dict], model: str) -> dict:
    """Build an Ollama response with tool_calls."""
    return {
        "model": model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        "message": {
            "role": "assistant",
            "content": text,
            "tool_calls": tool_calls,
        },
        "done": True,
        "total_duration": 0,
        "prompt_eval_count": 0,
        "eval_count": 0,
    }


def ollama_messages_to_chat_messages(messages: list[OllamaChatMessage]) -> list[ChatMessage]:
    """Convert Ollama messages (with images array) to ChatMessage format."""
    result = []
    for msg in messages:
        if msg.images:
            content = []
            for img_b64 in msg.images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            result.append(ChatMessage(role=msg.role, content=content))
        else:
            result.append(ChatMessage(role=msg.role, content=msg.content))
    return result


def make_ollama_response(text: str, model: str, done: bool = True) -> dict:
    return {
        "model": model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        "message": {"role": "assistant", "content": text},
        "done": done,
        "total_duration": 0,
        "prompt_eval_count": 0,
        "eval_count": 0,
    }


async def stream_ollama_query(
    model: str,
    system_prompt: str | None,
    prompt: Any,
    *,
    has_tools: bool = False,
):
    """Stream a query and yield newline-delimited JSON (Ollama format)."""
    options = make_options(model, system_prompt, streaming=True, disable_builtin_tools=has_tools)

    try:
        async for message in send_prompt(prompt, options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text:
                        chunk = make_ollama_response(block.text, model, done=False)
                        yield json.dumps(chunk) + "\n"
                    elif isinstance(block, ToolUseBlock):
                        tc_json = json.dumps({"tool_calls": [{"name": block.name, "arguments": block.input}]})
                        chunk = make_ollama_response(tc_json, model, done=False)
                        yield json.dumps(chunk) + "\n"
    except Exception as e:
        error_chunk = make_ollama_response(f"Error: {e}", model, done=False)
        yield json.dumps(error_chunk) + "\n"

    done_chunk = make_ollama_response("", model, done=True)
    yield json.dumps(done_chunk) + "\n"


# --- Endpoints ---


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": AVAILABLE_MODELS}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    model = resolve_model(request.model)
    system_prompt, prompt = build_prompt(request.messages)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    try:
        if request.stream:
            return StreamingResponse(
                stream_query(model, system_prompt, prompt, completion_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            text = await run_query(model, system_prompt, prompt)
            return JSONResponse(make_completion_response(text, model, completion_id))
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            {"error": {"message": str(e), "type": "server_error"}},
            status_code=500,
        )


@app.get("/api/tags")
async def ollama_tags():
    return {
        "models": [
            {"name": m["id"], "model": m["id"], "size": 0, "digest": "", "details": {}}
            for m in AVAILABLE_MODELS
        ]
    }


@app.post("/api/chat")
async def ollama_chat(request: OllamaChatRequest):
    model = resolve_model(request.model)
    chat_messages = ollama_messages_to_chat_messages(request.messages)
    system_prompt, prompt = build_prompt(chat_messages)

    # Disable built-in CLI tools whenever the caller provides their own tools.
    # This prevents the model from responding with ToolUseBlocks for built-in
    # tools (Bash, Read) instead of plain text — which causes "No content" errors.
    caller_has_tools = bool(request.tools)

    # Only inject tool-calling protocol when browser/action tools are present.
    # For "meta-only" tool sets (done, ask_user), the model responds with plain
    # text and the caller parses it — injecting the protocol causes infinite
    # done() loops because rig keeps re-prompting.
    has_action_tools = False
    if request.tools:
        tool_names = {t.function.name for t in request.tools}
        has_action_tools = bool(tool_names & ACTION_TOOLS)

    # Even with action tools, suppress the protocol once done() has already
    # been called in this conversation.  After done(), the model should
    # respond with plain text so rig's agent loop terminates.
    if has_action_tools:
        has_tool_results = any(msg.role == "tool" for msg in request.messages)
        has_done_call = any(
            msg.role == "assistant" and '"name": "done"' in msg.content
            for msg in request.messages
        )
        if has_tool_results and has_done_call:
            has_action_tools = False

    if has_action_tools:
        supplement = build_tool_system_supplement(request.tools)
        system_prompt = (system_prompt or "") + supplement

    try:
        if request.stream:
            return StreamingResponse(
                stream_ollama_query(model, system_prompt, prompt, has_tools=caller_has_tools),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            text = await run_query(model, system_prompt, prompt, has_tools=has_action_tools, disable_builtin_tools=caller_has_tools)
            if has_action_tools:
                remaining, tool_calls = parse_tool_calls_from_text(text)
                if tool_calls:
                    return JSONResponse(make_ollama_tool_response(remaining, tool_calls, model))
            return JSONResponse(make_ollama_response(text, model))
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            {"error": str(e)},
            status_code=500,
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11435)
