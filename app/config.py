from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ChatTool"
    app_host: str = "0.0.0.0"
    app_port: int = 13579

    default_model: str = "gpt-5.4"

    proxy_api_key: str | None = None
    proxy_target: Literal["auto", "local", "tailscale"] = "auto"
    local_proxy_base_url: str = "http://127.0.0.1:8313"
    tailscale_proxy_base_url: str = "https://aoschenlinux.tailfa309c.ts.net"
    openai_base_url: str | None = None
    anthropic_base_url: str | None = None
    request_timeout_seconds: float = 120.0
    models_cache_seconds: float = 30.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def _dedupe_urls(self, urls: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for url in urls:
            normalized = url.strip().rstrip("/")
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result

    def proxy_base_url_candidates(self) -> list[str]:
        if self.proxy_target == "local":
            return self._dedupe_urls([self.local_proxy_base_url])
        if self.proxy_target == "tailscale":
            return self._dedupe_urls([self.tailscale_proxy_base_url])
        return self._dedupe_urls([self.local_proxy_base_url, self.tailscale_proxy_base_url])

    def endpoint_base_url_candidates(self, endpoint_family: Literal["openai", "anthropic"]) -> list[str]:
        explicit = self.openai_base_url if endpoint_family == "openai" else self.anthropic_base_url
        if explicit:
            return self._dedupe_urls([explicit])
        return self.proxy_base_url_candidates()


class PublicConfig(BaseModel):
    app_name: str
    default_model: str
    proxy_target: Literal["auto", "local", "tailscale"]
    proxy_base_url_candidates: list[str]


settings = Settings()
