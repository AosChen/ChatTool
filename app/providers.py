from __future__ import annotations

import time
from collections.abc import Sequence

import httpx
from fastapi import HTTPException

from app.config import settings
from app.models import ChatMessage, ModelInfo

_MODELS_CACHE: list[dict] = []
_MODELS_CACHE_AT = 0.0
_MODELS_CACHE_UPSTREAM = ""


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
) -> tuple[str, str, str]:
    raw_models, _ = await _fetch_models()
    selected = next((item for item in raw_models if item.get("id") == model), None)
    endpoint = _select_default_endpoint(model, selected.get("supported_endpoints") if selected else None)

    timeout = httpx.Timeout(settings.request_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        if endpoint == "/v1/responses":
            upstream, reply = await _send_openai_responses(client, model, messages)
        elif endpoint == "/v1/messages":
            upstream, reply = await _send_anthropic(client, model, messages)
        else:
            endpoint = "/v1/chat/completions"
            upstream, reply = await _send_openai_chat_completions(client, model, messages)
    return endpoint, upstream, reply


async def _send_openai_chat_completions(
    client: httpx.AsyncClient,
    model: str,
    messages: Sequence[ChatMessage],
) -> tuple[str, str]:
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
    try:
        return upstream, data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="Invalid OpenAI chat-completions response from proxy") from exc


async def _send_openai_responses(
    client: httpx.AsyncClient,
    model: str,
    messages: Sequence[ChatMessage],
) -> tuple[str, str]:
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
    text = _extract_responses_text(data)
    if text:
        return upstream, text
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
) -> tuple[str, str]:
    anthropic_messages = [
        {"role": message.role, "content": message.content}
        for message in messages
    ]
    payload: dict = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": 4096,
    }
    if settings.enable_web_search:
        payload["tools"] = [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": settings.web_search_max_uses,
            }
        ]
    response, upstream = await _request_with_fallback(
        client,
        "POST",
        settings.endpoint_base_url_candidates("anthropic"),
        "/v1/messages",
        headers=_headers({"anthropic-version": "2023-06-01"}),
        json=payload,
    )
    await _raise_for_status(response)
    data = response.json()
    try:
        parts = data["content"]
        text_parts = [part["text"] for part in parts if part.get("type") == "text"]
        return upstream, "".join(text_parts).strip()
    except (KeyError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="Invalid Anthropic response from proxy") from exc
