# Example MCP Server (Streamable HTTP)
 
A minimal MCP server built on the official Python SDK (`mcp==2.0.0`), running
over Streamable HTTP so it can live in a container and be reached over the
network.
 
It ships with two demo tools (`echo`, `add`) and one resource (`time://now`)
so you have something to test against immediately — replace them with your
own.
 
## Run it
 
```bash
docker compose up -d --build
docker compose logs -f mcp-server
```
 
The server listens on `http://localhost:8000/mcp`.
 
## Test it without a client
 
```bash
curl -i -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"0.1"}}}'
```
 
The response headers include `mcp-session-id` — reuse it in an
`Mcp-Session-Id` header on subsequent calls (`tools/list`, `tools/call`,
`resources/read`, etc.) once you've sent a `notifications/initialized`
message.
 
## Point a client at it
 
Any MCP client that supports Streamable HTTP just needs the URL:
`http://localhost:8000/mcp` (or your host's address/port if not running
locally).
 
## Add your own tools
 
```python
@mcp.tool()
def my_tool(arg: str) -> str:
    """This docstring becomes the tool description the model sees."""
    return do_something(arg)
```
 
Resources (`@mcp.resource("scheme://path")`) and prompts (`@mcp.prompt()`)
follow the same pattern. Restart the container after changes:
 
```bash
docker compose up -d --build
```
 
## Notes
 
- `MCP_HOST` / `MCP_PORT` env vars control what the server binds to inside
  the container — leave `MCP_HOST=0.0.0.0` or it won't be reachable from
  outside the container.
- This has no authentication. Fine for local/dev use behind
  `localhost:8000`; if you expose it beyond your machine, put it behind a
  reverse proxy or gateway that handles auth (or use the SDK's built-in
  OAuth support via `token_verifier`/`auth_server_provider`).
- The old `from mcp.server.fastmcp import FastMCP` import path some
  tutorials still show is gone as of `mcp==2.0.0` — it's now
  `from mcp.server import MCPServer`.
