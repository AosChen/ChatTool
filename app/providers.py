from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from datetime import datetime, timezone

import httpx
from fastapi import HTTPException

from app.config import settings
from app.mcp_client import get_registry
from app.models import ChatMessage, ModelInfo

logger = logging.getLogger(__name__)

_MODELS_CACHE: list[dict] = []
_MODELS_CACHE_AT = 0.0
_MODELS_CACHE_UPSTREAM = ""


_LOCAL_TOOL_DEFINITIONS = [
    {
        "name": "get_current_time",
        "description": (
            "Returns the current server date and time in ISO 8601 format (UTC). "
            "Use this whenever the user asks about the current time, today's date, "
            "or anything that depends on knowing 'now'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }
]


def _execute_local_tool(name: str, tool_input: dict) -> str:
    if name == "get_current_time":
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    return f"[Tool '{name}' is not implemented in ChatTool]"


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if settings.proxy_api_key:
        headers["Authorization"] = f"Bearer {settings.proxy_api_key}"
        headers["x-api-key"] = settings.proxy_api_key
    if extra:
        headers.update(extra)
    return headers


async def _raise_for_status(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip() or str(exc)
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Upstream proxy error: {detail}",
        ) from exc


def _request_error_detail(exc: httpx.RequestError) -> str:
    return f"{exc.__class__.__name__}: {str(exc).strip() or 'request failed'}"


def _normalize_endpoints(supported_endpoints: Sequence[str] | None) -> set[str]:
    endpoints = set()
    for endpoint in supported_endpoints or []:
        normalized = endpoint.strip().lower()
        if not normalized:
            continue
        endpoints.add(normalized)
        if not normalized.startswith("/"):
            endpoints.add(f"/{normalized}")
        if not normalized.startswith("/v1") and normalized.startswith("/"):
            endpoints.add(f"/v1{normalized}")
    return endpoints


def _select_default_endpoint(model_id: str, supported_endpoints: Sequence[str] | None = None) -> str:
    model = model_id.strip().lower()
    endpoints = _normalize_endpoints(supported_endpoints)

    if model.startswith("claude") and "/v1/messages" in endpoints:
        return "/v1/messages"
    if "/v1/responses" in endpoints or "/responses" in endpoints:
        return "/v1/responses"
    if "/v1/chat/completions" in endpoints or "/chat/completions" in endpoints:
        return "/v1/chat/completions"
    if "/v1/messages" in endpoints or "/messages" in endpoints:
        return "/v1/messages"

    if model.startswith("claude"):
        return "/v1/messages"
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        return "/v1/responses"
    return "/v1/chat/completions"


async def list_models() -> list[ModelInfo]:
    raw_models, _ = await _fetch_models()
    model_infos = [
        ModelInfo(
            id=model.get("id", "unknown"),
            display_name=model.get("name") or model.get("display_name") or model.get("id", "unknown"),
            vendor=model.get("vendor"),
            supported_endpoints=model.get("supported_endpoints") or [],
            default_endpoint=_select_default_endpoint(model.get("id", "unknown"), model.get("supported_endpoints")),
        )
        for model in raw_models
        if model.get("id")
    ]
    model_infos.sort(key=lambda item: ((item.vendor or "zzz").lower(), item.id.lower()))
    return model_infos


async def _request_with_fallback(
    client: httpx.AsyncClient,
    method: str,
    base_urls: Sequence[str],
    path: str,
    *,
    headers: dict[str, str],
    json: dict | None = None,
) -> tuple[httpx.Response, str]:
    errors: list[str] = []
    for base_url in base_urls:
        try:
            response = await client.request(
                method,
                f"{base_url.rstrip('/')}{path}",
                headers=headers,
                json=json,
            )
        except httpx.RequestError as exc:
            errors.append(f"{base_url}: {_request_error_detail(exc)}")
            continue
        return response, base_url.rstrip("/")

    detail = "; ".join(errors) if errors else "no upstream proxy candidates configured"
    raise HTTPException(status_code=502, detail=f"Unable to reach upstream proxy: {detail}")


async def _fetch_models() -> tuple[list[dict], str]:
    global _MODELS_CACHE, _MODELS_CACHE_AT, _MODELS_CACHE_UPSTREAM

    now = time.time()
    if _MODELS_CACHE and now - _MODELS_CACHE_AT < settings.models_cache_seconds:
        return _MODELS_CACHE, _MODELS_CACHE_UPSTREAM

    timeout = httpx.Timeout(settings.request_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response, upstream = await _request_with_fallback(
            client,
            "GET",
            settings.endpoint_base_url_candidates("openai"),
            "/v1/models/full/",
            headers=_headers(),
        )
        await _raise_for_status(response)
        payload = response.json()

    models = payload.get("data")
    if not isinstance(models, list):
        raise HTTPException(status_code=502, detail="Invalid models response from proxy")

    _MODELS_CACHE = models
    _MODELS_CACHE_AT = now
    _MODELS_CACHE_UPSTREAM = upstream
    return models, upstream


async def send_chat(
    model: str,
    messages: Sequence[ChatMessage],
    tools_enabled: bool = True,
) -> tuple[str, str, str, dict[str, int]]:
    raw_models, _ = await _fetch_models()
    selected = next((item for item in raw_models if item.get("id") == model), None)
    endpoint = _select_default_endpoint(model, selected.get("supported_endpoints") if selected else None)

    timeout = httpx.Timeout(settings.request_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        if endpoint == "/v1/responses":
            upstream, reply, usage = await _send_openai_responses(client, model, messages)
        elif endpoint == "/v1/messages":
            upstream, reply, usage = await _send_anthropic(client, model, messages, tools_enabled=tools_enabled)
        else:
            endpoint = "/v1/chat/completions"
            upstream, reply, usage = await _send_openai_chat_completions(client, model, messages)
    return endpoint, upstream, reply, usage


async def _send_openai_chat_completions(
    client: httpx.AsyncClient,
    model: str,
    messages: Sequence[ChatMessage],
) -> tuple[str, str, dict[str, int]]:
    payload_messages = [{"role": message.role, "content": message.content} for message in messages]
    response, upstream = await _request_with_fallback(
        client,
        "POST",
        settings.endpoint_base_url_candidates("openai"),
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": model,
            "messages": payload_messages,
        },
    )
    await _raise_for_status(response)
    data = response.json()
    raw_usage = data.get("usage") or {}
    usage = {
        "input_tokens": int(raw_usage.get("prompt_tokens") or 0),
        "output_tokens": int(raw_usage.get("completion_tokens") or 0),
    }
    try:
        return upstream, data["choices"][0]["message"]["content"], usage
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="Invalid OpenAI chat-completions response from proxy") from exc


async def _send_openai_responses(
    client: httpx.AsyncClient,
    model: str,
    messages: Sequence[ChatMessage],
) -> tuple[str, str, dict[str, int]]:
    input_items = [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in messages
    ]
    payload: dict = {
        "model": model,
        "input": input_items,
    }
    if settings.enable_web_search:
        payload["tools"] = [{"type": "web_search"}]
    response, upstream = await _request_with_fallback(
        client,
        "POST",
        settings.endpoint_base_url_candidates("openai"),
        "/v1/responses",
        headers=_headers(),
        json=payload,
    )
    await _raise_for_status(response)
    data = response.json()
    raw_usage = data.get("usage") or {}
    usage = {
        "input_tokens": int(raw_usage.get("input_tokens") or raw_usage.get("prompt_tokens") or 0),
        "output_tokens": int(raw_usage.get("output_tokens") or raw_usage.get("completion_tokens") or 0),
    }
    text = _extract_responses_text(data)
    if text:
        return upstream, text, usage
    raise HTTPException(status_code=502, detail="Invalid OpenAI responses payload from proxy")


def _extract_responses_text(data: dict) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    collected: list[str] = []
    for item in data.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text", "")
                if text:
                    collected.append(text)
    return "".join(collected).strip()


async def _send_anthropic(
    client: httpx.AsyncClient,
    model: str,
    messages: Sequence[ChatMessage],
    tools_enabled: bool = True,
) -> tuple[str, str, dict[str, int]]:
    anthropic_messages: list[dict] = [
        {"role": message.role, "content": message.content}
        for message in messages
    ]

    tools: list[dict] = []
    mcp_registry = get_registry()
    if tools_enabled:
        if settings.enable_local_tools:
            tools.extend(_LOCAL_TOOL_DEFINITIONS)
        tools.extend(mcp_registry.anthropic_tools())

    upstream_used = ""
    last_input_tokens = 0
    cumulative_output_tokens = 0
    for iteration in range(max(1, settings.tool_loop_max_iterations)):
        payload: dict = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": 4096,
        }
        if tools:
            payload["tools"] = tools

        response, upstream = await _request_with_fallback(
            client,
            "POST",
            settings.endpoint_base_url_candidates("anthropic"),
            "/v1/messages",
            headers=_headers({"anthropic-version": "2023-06-01"}),
            json=payload,
        )
        upstream_used = upstream
        await _raise_for_status(response)
        data = response.json()

        raw_usage = data.get("usage") or {}
        last_input_tokens = int(raw_usage.get("input_tokens") or 0)
        cumulative_output_tokens += int(raw_usage.get("output_tokens") or 0)

        try:
            content_blocks = data["content"]
        except (KeyError, TypeError) as exc:
            raise HTTPException(status_code=502, detail="Invalid Anthropic response from proxy") from exc

        stop_reason = data.get("stop_reason")
        tool_use_blocks = [block for block in content_blocks if block.get("type") == "tool_use"]

        if stop_reason == "tool_use" and tool_use_blocks:
            anthropic_messages.append({"role": "assistant", "content": content_blocks})

            tool_results: list[dict] = []
            for block in tool_use_blocks:
                tool_name = block.get("name", "")
                tool_input = block.get("input") or {}
                logger.info("Tool call from model: %s input=%s", tool_name, tool_input)
                if mcp_registry.has_tool(tool_name):
                    result_text = await mcp_registry.call_tool(tool_name, tool_input)
                else:
                    result_text = _execute_local_tool(tool_name, tool_input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.get("id"),
                        "content": result_text,
                    }
                )
            anthropic_messages.append({"role": "user", "content": tool_results})
            continue

        text_parts = [block["text"] for block in content_blocks if block.get("type") == "text"]
        usage = {"input_tokens": last_input_tokens, "output_tokens": cumulative_output_tokens}
        return upstream_used, "".join(text_parts).strip(), usage

    raise HTTPException(
        status_code=502,
        detail=f"Tool-use loop did not converge within {settings.tool_loop_max_iterations} iterations",
    )
