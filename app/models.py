from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)


class ChatResponse(BaseModel):
    model: str
    endpoint: str
    upstream: str
    reply: str


class ModelInfo(BaseModel):
    id: str
    display_name: str
    vendor: str | None = None
    supported_endpoints: list[str] = Field(default_factory=list)
    default_endpoint: str


class ModelsResponse(BaseModel):
    data: list[ModelInfo]
