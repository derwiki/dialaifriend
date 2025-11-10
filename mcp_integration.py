import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)


class MCPServerConnection:
    """
    Manages a single MCP server process and session over stdio.

    Notes:
    - Imports the `mcp` client lazily to avoid hard dependency at import time.
    - Keeps a persistent connection open for reuse.
    """

    def __init__(self, config: MCPServerConfig, client_name: str, client_version: str) -> None:
        self.config = config
        self.client_name = client_name
        self.client_version = client_version
        self._session = None  # type: ignore[var-annotated]
        self._conn_cm = None  # type: ignore[var-annotated]
        self._connected_lock = asyncio.Lock()

    async def connect(self) -> None:
        if self._session is not None:
            return
        async with self._connected_lock:
            if self._session is not None:
                return
            try:
                # Lazy import to avoid ImportError unless used
                from mcp.client.session import ClientSession  # type: ignore
                from mcp.client.transport import StdioServerParameters, connect  # type: ignore
            except Exception as exc:  # pragma: no cover - import-time guard
                raise RuntimeError(
                    "The 'mcp' package is required for MCP integration. Install with 'pip install mcp'."
                ) from exc

            params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args,
                env={**os.environ, **self.config.env},
            )

            # Establish the stdio transport and session
            self._conn_cm = connect(params)
            reader, writer = await self._conn_cm.__aenter__()
            session = ClientSession(reader, writer)

            # Initialize the session with client info
            await session.initialize(
                {
                    "clientInfo": {"name": self.client_name, "version": self.client_version},
                }
            )
            self._session = session

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    async def close(self) -> None:
        # Best-effort close; tolerate missing methods depending on SDK version
        try:
            if self._session is not None and hasattr(self._session, "close"):
                await self._session.close()  # type: ignore[attr-defined]
        finally:
            if self._conn_cm is not None:
                await self._conn_cm.__aexit__(None, None, None)  # type: ignore[func-returns-value]
        self._session = None
        self._conn_cm = None

    async def list_tools(self) -> List[str]:
        await self.connect()
        assert self._session is not None
        result = await self._session.list_tools()  # type: ignore[attr-defined]
        # Normalize to plain list of names
        tools = getattr(result, "tools", result)
        names: List[str] = []
        for t in tools:
            name = getattr(t, "name", None)
            if name is None and isinstance(t, dict):
                name = t.get("name")
            if name:
                names.append(name)
        return names

    async def list_resources(self) -> List[Dict[str, Any]]:
        await self.connect()
        assert self._session is not None
        result = await self._session.list_resources()  # type: ignore[attr-defined]
        resources = getattr(result, "resources", result)
        normalized: List[Dict[str, Any]] = []
        for r in resources:
            if isinstance(r, dict):
                normalized.append(r)
            else:
                normalized.append({
                    "uri": getattr(r, "uri", None),
                    "name": getattr(r, "name", None),
                    "description": getattr(r, "description", None),
                    "mimeType": getattr(r, "mimeType", None),
                })
        return normalized

    async def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        await self.connect()
        assert self._session is not None
        arguments = arguments or {}
        result = await self._session.call_tool(tool_name, arguments=arguments)  # type: ignore[attr-defined]
        # Normalize outputs to a simple JSON-serializable dict
        content = getattr(result, "content", result)
        normalized_items: List[Dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict):
                normalized_items.append(item)
                continue
            item_type = getattr(item, "type", None)
            if item_type == "text":
                normalized_items.append({
                    "type": "text",
                    "text": getattr(item, "text", None),
                })
            elif item_type == "image":
                normalized_items.append({
                    "type": "image",
                    "data": getattr(item, "data", None),
                    "mimeType": getattr(item, "mimeType", None),
                })
            else:
                # Fallback to repr for unknown items
                normalized_items.append({"type": str(item_type or "unknown"), "value": repr(item)})
        return {"content": normalized_items}


class MCPManager:
    """Holds and reuses connections to multiple MCP servers."""

    def __init__(self, servers: List[MCPServerConfig], client_name: str = "dial-aifriend", client_version: str = "0.1.0") -> None:
        self._client_name = client_name
        self._client_version = client_version
        self._servers: Dict[str, MCPServerConnection] = {
            s.name: MCPServerConnection(s, client_name, client_version) for s in servers
        }

    @classmethod
    def from_env(cls) -> "MCPManager":
        """
        Build from environment variable MCP_SERVERS_JSON containing a JSON array of server configs:
        [
          {"name":"slack","command":"npx","args":["-y","@modelcontextprotocol/server-slack"],"env":{"SLACK_BOT_TOKEN":"xoxb-…"}}
        ]
        """
        raw = os.getenv("MCP_SERVERS_JSON", "[]")
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            items = []
        servers: List[MCPServerConfig] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            name = it.get("name")
            command = it.get("command")
            if not name or not command:
                continue
            args = it.get("args") or []
            env = it.get("env") or {}
            servers.append(MCPServerConfig(name=name, command=command, args=list(args), env=dict(env)))
        return cls(servers)

    def list_servers(self) -> List[str]:
        return list(self._servers.keys())

    def get(self, name: str) -> MCPServerConnection:
        if name not in self._servers:
            raise KeyError(f"Unknown MCP server: {name}")
        return self._servers[name]


# Singleton accessor for app-level usage
_mcp_manager: Optional[MCPManager] = None


def get_mcp_manager() -> MCPManager:
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPManager.from_env()
    return _mcp_manager
