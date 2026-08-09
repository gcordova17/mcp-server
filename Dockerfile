FROM python:3.12-slim

# Run as a non-root user
RUN useradd --create-home --uid 1000 mcp
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

USER mcp

ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8000

EXPOSE 8000

CMD ["python", "server.py"]
