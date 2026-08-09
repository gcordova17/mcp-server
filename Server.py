"""
Minimal MCP server exposed over Streamable HTTP.

Run directly:
    python server.py

Run in Docker:
    see Dockerfile / docker-compose.yml
"""

import os
from datetime import datetime, timezone

from mcp.server import MCPServer

mcp = MCPServer(name="example-server", version="0.1.0")


@mcp.tool()
def echo(text: str) -> str:
    """Echo back the provided text. Useful for verifying the server is reachable."""
    return text


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together and return the result."""
    return a + b


@mcp.resource("time://now")
def current_time() -> str:
    """Return the current server time in ISO 8601 (UTC)."""
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    # 0.0.0.0 so the server is reachable from outside its container.
    # Configurable via env vars so the same image works locally and in Compose.
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", "8000"))

    mcp.run(transport="streamable-http", host=host, port=port)
