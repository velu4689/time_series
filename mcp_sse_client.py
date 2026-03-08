"""
mcp_sse_client.py
Handles the full MCP-over-SSE handshake and tool invocation lifecycle.
Based on confirmed working sequence:
  1. GET  /sse              → get session_id
  2. POST initialize        → negotiate protocol
  3. POST notifications/initialized → confirm
  4. POST tools/list        → discover tools
  5. POST tools/call        → invoke tools
"""

import httpx
import json
import threading
import queue
import time
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class MCPServer:
    name: str
    base_url: str           # e.g. http://atlassian:9000
    sse_path: str = "/sse"  # SSE handshake endpoint
    protocol_version: str = "2024-11-05"

    # Populated after handshake
    session_url: str = ""
    tools: list = field(default_factory=list)
    _initialized: bool = False


class MCPSSEClient:
    """
    Full MCP-over-SSE client that correctly handles:
    - SSE endpoint handshake to get session URL
    - initialize + notifications/initialized sequence
    - tools/list discovery
    - tools/call invocation
    """

    def __init__(self, server: MCPServer, timeout: int = 30):
        self.server = server
        self.timeout = timeout
        self._req_id = 0

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    # ── Step 1: SSE Handshake ─────────────────────────────────────────────────
    def get_session_url(self) -> str:
        """
        Opens SSE connection and reads the session endpoint.
        Returns the full messages URL with session_id.
        """
        sse_url = f"{self.server.base_url}{self.server.sse_path}"
        session_url = None
        event_queue = queue.Queue()

        def _stream():
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    with client.stream("GET", sse_url) as resp:
                        resp.raise_for_status()
                        current_event = None
                        for line in resp.iter_lines():
                            if line.startswith("event:"):
                                current_event = line.split(":", 1)[1].strip()
                            elif line.startswith("data:"):
                                data = line.split(":", 1)[1].strip()
                                event_queue.put((current_event, data))
                                if current_event == "endpoint":
                                    return  # got what we need
            except Exception as e:
                event_queue.put(("error", str(e)))

        t = threading.Thread(target=_stream, daemon=True)
        t.start()

        try:
            event, data = event_queue.get(timeout=self.timeout)
        except queue.Empty:
            raise TimeoutError(f"No SSE event received from {sse_url} within {self.timeout}s")

        if event == "error":
            raise ConnectionError(f"SSE connection failed: {data}")
        if event != "endpoint":
            raise ValueError(f"Expected 'endpoint' event, got '{event}'")

        # data is like: /messages/?session_id=abc123
        # build full URL
        if data.startswith("http"):
            session_url = data
        else:
            session_url = f"{self.server.base_url}{data}"

        self.server.session_url = session_url
        return session_url

    # ── Step 2 & 3: Initialize ────────────────────────────────────────────────
    def initialize(self) -> dict:
        """Sends initialize + notifications/initialized to the session endpoint."""
        if not self.server.session_url:
            self.get_session_url()

        with httpx.Client(timeout=self.timeout) as client:
            # initialize
            init_resp = client.post(
                self.server.session_url,
                json={
                    "jsonrpc": "2.0",
                    "id": self._next_id(),
                    "method": "initialize",
                    "params": {
                        "protocolVersion": self.server.protocol_version,
                        "capabilities": {},
                        "clientInfo": {
                            "name": "langflow-mcp-client",
                            "version": "1.0.0"
                        }
                    }
                }
            )
            init_resp.raise_for_status()
            init_data = init_resp.json()

            if "error" in init_data:
                raise RuntimeError(f"initialize failed: {init_data['error']}")

            # notifications/initialized (no id — it's a notification)
            client.post(
                self.server.session_url,
                json={
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {}
                }
            )

        self.server._initialized = True
        return init_data.get("result", {})

    # ── Step 4: List Tools ────────────────────────────────────────────────────
    def list_tools(self) -> list[dict]:
        """Discovers all tools exposed by this MCP server."""
        if not self.server._initialized:
            self.initialize()

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                self.server.session_url,
                json={
                    "jsonrpc": "2.0",
                    "id": self._next_id(),
                    "method": "tools/list"
                }
            )
            resp.raise_for_status()
            data = resp.json()

        if "error" in data:
            raise RuntimeError(f"tools/list failed: {data['error']}")

        tools = data.get("result", {}).get("tools", [])
        self.server.tools = tools
        return tools

    # ── Step 5: Call Tool ─────────────────────────────────────────────────────
    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Invokes a specific MCP tool and returns the text result."""
        if not self.server._initialized:
            self.initialize()

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                self.server.session_url,
                json={
                    "jsonrpc": "2.0",
                    "id": self._next_id(),
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": arguments
                    }
                }
            )
            resp.raise_for_status()
            data = resp.json()

        if "error" in data:
            raise RuntimeError(f"tools/call failed: {data['error']}")

        result = data.get("result", {})

        # MCP spec: result.content is a list of content blocks
        content = result.get("content", [])
        if isinstance(content, list):
            texts = [
                block.get("text", "")
                for block in content
                if block.get("type") == "text"
            ]
            return "\n".join(texts)

        return str(result)

    # ── Full connect helper ───────────────────────────────────────────────────
    def connect(self) -> list[dict]:
        """One-call setup: handshake + initialize + list tools."""
        print(f"[MCP] Connecting to {self.server.name} @ {self.server.base_url}{self.server.sse_path}")
        self.get_session_url()
        print(f"[MCP] Session URL: {self.server.session_url}")
        self.initialize()
        print(f"[MCP] Initialized (protocol: {self.server.protocol_version})")
        tools = self.list_tools()
        print(f"[MCP] Discovered {len(tools)} tools: {[t['name'] for t in tools]}")
        return tools

    # ── Convert MCP tools → OpenAI tool schema ────────────────────────────────
    @staticmethod
    def to_openai_tools(tools: list[dict]) -> list[dict]:
        """Converts MCP tool definitions to OpenAI function-calling schema."""
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {
                        "type": "object",
                        "properties": {}
                    })
                }
            })
        return openai_tools
