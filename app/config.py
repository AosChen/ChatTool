from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ChatTool"
    app_host: str = "0.0.0.0"
    app_port: int = 13579

    default_model: str = "gpt-5.4"

    proxy_api_key: str | None = None
    openai_base_url: str = "http://127.0.0.1:8313"
    anthropic_base_url: str = "http://127.0.0.1:8313"
    request_timeout_seconds: float = 120.0
    models_cache_seconds: float = 30.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class PublicConfig(BaseModel):
    app_name: str
    default_model: str


settings = Settings()
