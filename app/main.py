from pathlib import Path
import logging
import os

from fastapi import Cookie, Depends, FastAPI, HTTPException, Response, status
from fastapi.responses import FileResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from app.config import PublicConfig, settings
from app.models import (
    AuthRequest,
    AuthResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    CompactCreateResponse,
    CompactLoadResponse,
    CompactsResponse,
    CreateSessionRequest,
    MessagesResponse,
    ModelsResponse,
    PersistedSession,
    SessionsResponse,
    TokenUsage,
    UpdateSessionRequest,
    UserPublic,
)
from app.providers import list_models, send_chat
from app.storage import ChatStorage, get_storage
from app.mcp_client import get_registry

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title=settings.app_name)


def storage_dep() -> ChatStorage:
    return get_storage(settings.database_path, settings.message_encryption_key, settings.compacts_dir)


def set_auth_cookie(response: Response, auth_session_id: str) -> None:
    response.set_cookie(
        key=settings.auth_cookie_name,
        value=auth_session_id,
        max_age=settings.auth_session_days * 24 * 60 * 60,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite="lax",
        path="/",
    )


def clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(
        key=settings.auth_cookie_name,
        path="/",
        samesite="lax",
    )


def current_user_optional(
    auth_session_id: str | None = Cookie(default=None, alias=settings.auth_cookie_name),
    storage: ChatStorage = Depends(storage_dep),
) -> UserPublic | None:
    if not auth_session_id:
        return None
    return storage.get_user_by_auth_session(auth_session_id)


def require_current_user(user: UserPublic | None = Depends(current_user_optional)) -> UserPublic:
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


@app.on_event("startup")
async def startup() -> None:
    get_storage(settings.database_path, settings.message_encryption_key, settings.compacts_dir)
    if settings.brave_api_key:
        os.environ.setdefault("BRAVE_API_KEY", settings.brave_api_key)
    if settings.tavily_api_key:
        os.environ.setdefault("TAVILY_API_KEY", settings.tavily_api_key)
    await get_registry().startup(settings.mcp_servers_config_path)


@app.on_event("shutdown")
async def shutdown() -> None:
    await get_registry().shutdown()


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config", response_model=PublicConfig)
async def get_config() -> PublicConfig:
    return PublicConfig(
        app_name=settings.app_name,
        default_model=settings.default_model,
        proxy_target=settings.proxy_target,
        proxy_base_url_candidates=settings.proxy_base_url_candidates(),
        enable_registration=settings.enable_registration,
    )


@app.post("/api/auth/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: AuthRequest,
    response: Response,
    storage: ChatStorage = Depends(storage_dep),
) -> AuthResponse:
    if not settings.enable_registration:
        raise HTTPException(status_code=403, detail="Registration is disabled")
    user = storage.create_user(request.username, request.password)
    auth_session_id = storage.create_auth_session(user.id, settings.auth_session_days)
    set_auth_cookie(response, auth_session_id)
    return AuthResponse(user=user)


@app.post("/api/auth/login", response_model=AuthResponse)
async def login(
    request: AuthRequest,
    response: Response,
    storage: ChatStorage = Depends(storage_dep),
) -> AuthResponse:
    user, auth_error = storage.authenticate_user(request.username, request.password)
    if user is None:
        if auth_error == "user_not_found":
            raise HTTPException(status_code=401, detail="用户名不存在")
        if auth_error == "wrong_password":
            raise HTTPException(status_code=401, detail="密码错误")
        raise HTTPException(status_code=401, detail="登录失败")
    auth_session_id = storage.create_auth_session(user.id, settings.auth_session_days)
    set_auth_cookie(response, auth_session_id)
    return AuthResponse(user=user)


@app.post("/api/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    response: Response,
    auth_session_id: str | None = Cookie(default=None, alias=settings.auth_cookie_name),
    storage: ChatStorage = Depends(storage_dep),
) -> Response:
    if auth_session_id:
        storage.delete_auth_session(auth_session_id)
    clear_auth_cookie(response)
    return response


@app.get("/api/auth/me", response_model=AuthResponse)
async def me(user: UserPublic = Depends(require_current_user)) -> AuthResponse:
    return AuthResponse(user=user)


@app.get("/api/models", response_model=ModelsResponse)
async def get_models(user: UserPublic = Depends(require_current_user)) -> ModelsResponse:
    _ = user
    return ModelsResponse(data=await list_models())


@app.get("/api/sessions", response_model=SessionsResponse)
async def get_sessions(
    user: UserPublic = Depends(require_current_user),
    storage: ChatStorage = Depends(storage_dep),
) -> SessionsResponse:
    return SessionsResponse(data=storage.list_sessions(user.id))


@app.get("/api/sessions/{session_id}/messages", response_model=MessagesResponse)
async def get_session_messages(
    session_id: str,
    user: UserPublic = Depends(require_current_user),
    storage: ChatStorage = Depends(storage_dep),
) -> MessagesResponse:
    return MessagesResponse(data=storage.list_messages(user.id, session_id))


@app.post("/api/sessions", response_model=PersistedSession, status_code=status.HTTP_201_CREATED)
async def create_session(
    request: CreateSessionRequest,
    user: UserPublic = Depends(require_current_user),
    storage: ChatStorage = Depends(storage_dep),
) -> PersistedSession:
    title = (request.title or "新会话").strip() or "新会话"
    model = (request.model or settings.default_model).strip() or settings.default_model
    tools_enabled = True if request.tools_enabled is None else request.tools_enabled
    return storage.create_session(user.id, title, model, tools_enabled=tools_enabled)


@app.patch("/api/sessions/{session_id}", response_model=PersistedSession)
async def update_session(
    session_id: str,
    request: UpdateSessionRequest,
    user: UserPublic = Depends(require_current_user),
    storage: ChatStorage = Depends(storage_dep),
) -> PersistedSession:
    title = request.title.strip() if request.title is not None else None
    model = request.model.strip() if request.model is not None else None
    if title == "":
        title = "新会话"
    if model == "":
        model = settings.default_model
    return storage.update_session(
        user.id,
        session_id,
        title=title,
        model=model,
        tools_enabled=request.tools_enabled,
    )


@app.delete("/api/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    user: UserPublic = Depends(require_current_user),
    storage: ChatStorage = Depends(storage_dep),
) -> Response:
    storage.delete_session(user.id, session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user: UserPublic = Depends(require_current_user),
    storage: ChatStorage = Depends(storage_dep),
) -> ChatResponse:
    session = storage.get_session(user.id, request.session_id)
    session = storage.append_message(user.id, session.id, "user", request.content.strip())

    model = session.model or settings.default_model
    endpoint, upstream, reply, usage = await send_chat(
        model=model,
        messages=session.messages,
        tools_enabled=session.tools_enabled,
    )
    session = storage.append_message(user.id, session.id, "assistant", reply)
    return ChatResponse(
        model=model,
        endpoint=endpoint,
        upstream=upstream,
        reply=reply,
        session=session,
        usage=TokenUsage(**usage),
    )


COMPACT_SUMMARY_PROMPT = """请将我们到目前为止的对话压缩成一份「会话快照」，目的是让另一个新会话能够基于这份快照无缝继续工作。按以下结构输出 Markdown：

## 背景与目标
（用户在做什么、为什么）

## 关键决策
- 决策内容｜原因｜对后续的影响

## 涉及文件 / 代码
- 路径：作用；必要的关键片段原样保留（不要复述含义）

## 当前状态
（已完成 / 进行中 / 待办）

## 未解决的问题
（需要后续会话回答或确认的点）

## 用户偏好与约束
（语气、技术栈选择、需要避免的做法）

要求：
- 信息保真优先于篇幅，但不要复述对话原文
- 用户原话里关键的需求/约束尽量原样引用
- 不要包含寒暄、过程性表达
- 直接输出快照内容，不要加额外的开场或结尾说明"""


COMPACT_LOAD_TEMPLATE = """以下是从之前会话加载的快照，请阅读并在脑海里建立完整的上下文，然后用一两句确认你理解了主线和待办，等待我的下一步指令。不要复述快照内容。

---快照开始---
{content}
---快照结束---"""


@app.post("/api/sessions/{session_id}/compact", response_model=CompactCreateResponse)
async def create_compact(
    session_id: str,
    user: UserPublic = Depends(require_current_user),
    storage: ChatStorage = Depends(storage_dep),
) -> CompactCreateResponse:
    session = storage.get_session(user.id, session_id)
    if not session.messages:
        raise HTTPException(status_code=400, detail="会话为空，无法压缩")

    summary_messages = list(session.messages) + [
        ChatMessage(role="user", content=COMPACT_SUMMARY_PROMPT)
    ]
    model = session.model or settings.default_model
    _, _, reply, _ = await send_chat(
        model=model,
        messages=summary_messages,
        tools_enabled=False,
    )
    summary = reply.strip()
    if not summary:
        raise HTTPException(status_code=502, detail="上游未返回总结内容")

    title = session.title or "未命名快照"
    compact = storage.create_compact(
        user_id=user.id,
        title=title,
        plaintext=summary,
        source_session_id=session.id,
    )
    preview = summary[:280]
    return CompactCreateResponse(compact=compact, summary_preview=preview)


@app.get("/api/compacts", response_model=CompactsResponse)
async def list_compacts(
    user: UserPublic = Depends(require_current_user),
    storage: ChatStorage = Depends(storage_dep),
) -> CompactsResponse:
    return CompactsResponse(data=storage.list_compacts(user.id))


@app.post("/api/compacts/{compact_id}/load", response_model=CompactLoadResponse)
async def load_compact(
    compact_id: str,
    user: UserPublic = Depends(require_current_user),
    storage: ChatStorage = Depends(storage_dep),
) -> CompactLoadResponse:
    meta, plaintext = storage.load_compact_plaintext(user.id, compact_id)
    title = f"[继承] {meta.title}"
    new_session = storage.create_session(
        user.id,
        title=title[:120],
        model=settings.default_model,
        tools_enabled=True,
    )
    bootstrap = COMPACT_LOAD_TEMPLATE.format(content=plaintext)
    new_session = storage.append_message(user.id, new_session.id, "user", bootstrap)
    try:
        _, _, reply, _ = await send_chat(
            model=new_session.model or settings.default_model,
            messages=new_session.messages,
            tools_enabled=False,
        )
        confirmation = (reply or "").strip() or "已加载快照，请告诉我下一步。"
    except Exception:  # noqa: BLE001
        confirmation = "已加载快照，请告诉我下一步。"
    new_session = storage.append_message(user.id, new_session.id, "assistant", confirmation)
    return CompactLoadResponse(session=new_session, compact=meta)


@app.delete("/api/compacts/{compact_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_compact(
    compact_id: str,
    user: UserPublic = Depends(require_current_user),
    storage: ChatStorage = Depends(storage_dep),
) -> Response:
    storage.delete_compact(user.id, compact_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
