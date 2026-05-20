from pydantic import BaseModel, Field


class UserPublic(BaseModel):
    id: str
    username: str


class AuthRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=4, max_length=256)


class AuthResponse(BaseModel):
    user: UserPublic


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1)


class SessionMeta(BaseModel):
    id: str
    title: str
    model: str
    created_at: str
    updated_at: str
    tools_enabled: bool = True


class PersistedSession(SessionMeta):
    messages: list[ChatMessage] = Field(default_factory=list)


class SessionsResponse(BaseModel):
    data: list[SessionMeta]


class MessagesResponse(BaseModel):
    data: list[ChatMessage]


class CreateSessionRequest(BaseModel):
    title: str | None = Field(default=None, max_length=120)
    model: str | None = Field(default=None, max_length=120)
    tools_enabled: bool | None = None


class UpdateSessionRequest(BaseModel):
    title: str | None = Field(default=None, max_length=120)
    model: str | None = Field(default=None, max_length=120)
    tools_enabled: bool | None = None


class ChatRequest(BaseModel):
    session_id: str
    content: str = Field(min_length=1, max_length=40000)


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class ChatResponse(BaseModel):
    model: str
    endpoint: str
    upstream: str
    reply: str
    session: PersistedSession
    usage: TokenUsage = Field(default_factory=TokenUsage)


class ModelInfo(BaseModel):
    id: str
    display_name: str
    vendor: str | None = None
    supported_endpoints: list[str] = Field(default_factory=list)
    default_endpoint: str


class ModelsResponse(BaseModel):
    data: list[ModelInfo]


class CompactMeta(BaseModel):
    id: str
    title: str
    source_session_id: str | None = None
    created_at: str
    byte_size: int


class CompactsResponse(BaseModel):
    data: list[CompactMeta]


class CompactCreateResponse(BaseModel):
    compact: CompactMeta
    summary_preview: str


class CompactLoadResponse(BaseModel):
    session: "PersistedSession"
    compact: CompactMeta
