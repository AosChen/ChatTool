from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.config import PublicConfig, settings
from app.models import ChatRequest, ChatResponse, ModelsResponse
from app.providers import list_models, send_chat

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title=settings.app_name)


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
    )


@app.get("/api/models", response_model=ModelsResponse)
async def get_models() -> ModelsResponse:
    return ModelsResponse(data=await list_models())


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    model = request.model or settings.default_model
    endpoint, upstream, reply = await send_chat(
        model=model,
        messages=request.messages,
    )
    return ChatResponse(model=model, endpoint=endpoint, upstream=upstream, reply=reply)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
