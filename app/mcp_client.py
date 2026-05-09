from __future__ import annotations

import json
import logging
import os
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), "")

    return _ENV_VAR_PATTERN.sub(replace, value)


def _expand_env_dict(env: dict[str, str] | None) -> dict[str, str]:
    if not env:
        return {}
    expanded: dict[str, str] = {}
    for key, value in env.items():
        if not isinstance(value, str):
            continue
        expanded[key] = _expand_env(value)
    return expanded


def _has_unresolved_placeholder(env: dict[str, str]) -> str | None:
    for key, value in env.items():
        match = _ENV_VAR_PATTERN.search(value)
        if match:
            return f"{key}=${{{match.group(1)}}}"
        if value == "":
            return f"{key} is empty"
    return None


def _flatten_tool_result_content(content_blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in content_blocks or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
            continue
        block_type = getattr(block, "type", None)
        if block_type == "image":
            parts.append("[image content omitted]")
        else:
            parts.append(str(block))
    return "\n".join(p for p in parts if p).strip()


class McpRegistry:
    def __init__(self) -> None:
        self._stack: AsyncExitStack | None = None
        self._sessions: dict[str, Any] = {}
        self._tools: dict[str, tuple[str, dict]] = {}

    async def startup(self, config_path: str) -> None:
        path = Path(config_path)
        if not path.exists():
            logger.info("No MCP server config at %s; skipping MCP startup", path)
            return

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON in %s: %s", path, exc)
            return

        servers = raw.get("mcpServers") or {}
        if not servers:
            logger.info("MCP config %s has no mcpServers entries", path)
            return

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        stack = AsyncExitStack()
        await stack.__aenter__()
        self._stack = stack

        for name, spec in servers.items():
            command = spec.get("command")
            args = spec.get("args") or []
            env = _expand_env_dict(spec.get("env"))

            if not command:
                logger.warning("MCP server '%s' missing 'command'; skipping", name)
                continue

            unresolved = _has_unresolved_placeholder(env)
            if unresolved:
                logger.warning(
                    "MCP server '%s' skipped: env not resolved (%s). "
                    "Set the corresponding variable in .env.",
                    name,
                    unresolved,
                )
                continue

            params = StdioServerParameters(command=command, args=args, env=env or None)
            try:
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                listed = await session.list_tools()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to start MCP server '%s': %s", name, exc)
                continue

            self._sessions[name] = session

            tool_count = 0
            for tool in listed.tools:
                if tool.name in self._tools:
                    logger.warning(
                        "Tool name collision for '%s' (already provided by '%s'); "
                        "skipping the one from '%s'",
                        tool.name,
                        self._tools[tool.name][0],
                        name,
                    )
                    continue
                anthropic_tool = {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema or {"type": "object", "properties": {}},
                }
                self._tools[tool.name] = (name, anthropic_tool)
                tool_count += 1

            logger.info("MCP server '%s' ready; registered %d tool(s)", name, tool_count)

    async def shutdown(self) -> None:
        if self._stack is None:
            return
        try:
            await self._stack.aclose()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error closing MCP sessions: %s", exc)
        finally:
            self._stack = None
            self._sessions.clear()
            self._tools.clear()

    def anthropic_tools(self) -> list[dict]:
        return [tool_def for _, tool_def in self._tools.values()]

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    async def call_tool(self, name: str, arguments: dict) -> str:
        if name not in self._tools:
            return f"[MCP tool '{name}' is not registered]"
        server_name, _ = self._tools[name]
        session = self._sessions[server_name]
        try:
            result = await session.call_tool(name, arguments=arguments or {})
        except Exception as exc:  # noqa: BLE001
            logger.exception("MCP tool '%s' raised: %s", name, exc)
            return f"[MCP tool '{name}' error: {exc}]"

        text = _flatten_tool_result_content(result.content)
        if getattr(result, "isError", False):
            return f"[MCP tool '{name}' returned error]\n{text}"
        return text or "[MCP tool returned empty content]"


_registry = McpRegistry()


def get_registry() -> McpRegistry:
    return _registry
