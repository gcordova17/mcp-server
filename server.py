"""
MCP server exposing the TickerNN Trading API over Streamable HTTP.

Run directly:
    python server.py

Run in Docker:
    see Dockerfile / docker-compose.yml
"""

import os
from typing import Optional

import httpx
from mcp.server import MCPServer

mcp = MCPServer(name="tickernn-mcp", version="0.1.0")

API_BASE = os.environ.get("TICKERNN_API_URL", "http://host.docker.internal:8001")


@mcp.tool()
async def get_health() -> dict:
    """Check if the TickerNN Trading API is healthy and reachable."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{API_BASE}/health")
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def get_available_tickers(
    user_name: Optional[str] = None,
    list_files_name: Optional[bool] = None,
) -> dict:
    """Return the list of cryptocurrency tickers that have trained models available."""
    params = {}
    if user_name is not None:
        params["user_name"] = user_name
    if list_files_name is not None:
        params["list_files_name"] = list_files_name
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{API_BASE}/api/v1/models/tickers_available", params=params)
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def get_training_logs(
    replica_id: Optional[str] = None,
    user_name: Optional[str] = "default",
) -> dict:
    """Retrieve PPO training logs for a model. Optionally filter by replica_id or user_name."""
    params = {}
    if replica_id is not None:
        params["replica_id"] = replica_id
    if user_name is not None:
        params["user_name"] = user_name
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{API_BASE}/api/v1/models/train/logs", params=params)
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def train_model(
    ticker: str,
    n_epoch: Optional[int] = None,
    learning_rate: Optional[float] = None,
    gamma: Optional[float] = None,
    train_timesteps: Optional[int] = None,
    new_model: Optional[bool] = None,
    user_name: Optional[str] = None,
) -> dict:
    """
    Start or continue PPO reinforcement learning training for a cryptocurrency ticker.
    Use new_model=True to train from scratch, False to continue an existing model.
    Returns a replica_id that can be used to fetch training logs.
    """
    body: dict = {"ticker": ticker}
    if n_epoch is not None:
        body["n_epoch"] = n_epoch
    if learning_rate is not None:
        body["learning_rate"] = learning_rate
    if gamma is not None:
        body["gamma"] = gamma
    if train_timesteps is not None:
        body["train_timesteps"] = train_timesteps
    if new_model is not None:
        body["new_model"] = new_model

    params = {}
    if user_name is not None:
        params["user_name"] = user_name

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(f"{API_BASE}/api/v1/models/train", json=body, params=params)
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def get_historical_chart(
    ticker: str,
    days_to_render: Optional[int] = None,
    user_name: Optional[str] = None,
) -> dict:
    """
    Generate a historical price chart for a cryptocurrency ticker.
    Returns chart data for the specified number of past days.
    """
    body: dict = {"ticker": ticker}
    if days_to_render is not None:
        body["days_to_render"] = days_to_render

    params = {}
    if user_name is not None:
        params["user_name"] = user_name

    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{API_BASE}/api/v1/charts/historical", json=body, params=params)
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def get_realtime_chart(
    ticker: str,
    since: str,
    real_time: Optional[bool] = True,
    user_name: Optional[str] = None,
) -> dict:
    """
    Generate a real-time price chart for a cryptocurrency ticker starting from a given timestamp.
    'since' should be an ISO 8601 datetime string (e.g. '2026-01-01T00:00:00').
    """
    body: dict = {"ticker": ticker, "since": since}
    if real_time is not None:
        body["real_time"] = real_time

    params = {}
    if user_name is not None:
        params["user_name"] = user_name

    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{API_BASE}/api/v1/charts/realtime", json=body, params=params)
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def get_batch_chart(
    since: str,
    real_time: Optional[bool] = True,
    user_name: Optional[str] = None,
) -> dict:
    """
    Generate batch chart data for all available tickers starting from a given timestamp.
    'since' should be an ISO 8601 datetime string (e.g. '2026-01-01T00:00:00').
    """
    body: dict = {"since": since}
    if real_time is not None:
        body["real_time"] = real_time

    params = {}
    if user_name is not None:
        params["user_name"] = user_name

    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{API_BASE}/api/v1/charts/batch", json=body, params=params)
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def test_db_connection(
    ticker: str,
    since: str,
    real_time: Optional[bool] = True,
    user_name: Optional[str] = None,
) -> dict:
    """
    Test the database connection by fetching data for a ticker starting from a given timestamp.
    'since' should be an ISO 8601 datetime string (e.g. '2026-01-01T00:00:00').
    """
    body: dict = {"ticker": ticker, "since": since}
    if real_time is not None:
        body["real_time"] = real_time

    params = {}
    if user_name is not None:
        params["user_name"] = user_name

    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{API_BASE}/api/v1/test/db_connection", json=body, params=params)
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def launch_optimizer_job(
    ticker: str,
    max_rounds: int,
    n_epoch: Optional[int] = None,
    learning_rate: Optional[float] = None,
    gamma: Optional[float] = None,
    train_timesteps: Optional[int] = None,
    new_model: Optional[bool] = None,
    poll_interval_seconds: Optional[int] = None,
    round_timeout_minutes: Optional[int] = None,
    user_name: Optional[str] = None,
) -> dict:
    """
    Launch a multi-round hyperparameter optimizer job for a cryptocurrency ticker.
    Runs in the background for up to max_rounds rounds, retraining and evaluating
    the model each round. Returns a status message, not a run_id — use
    list_optimizer_jobs to find the resulting job.
    """
    body: dict = {"ticker": ticker, "max_rounds": max_rounds}
    if n_epoch is not None:
        body["n_epoch"] = n_epoch
    if learning_rate is not None:
        body["learning_rate"] = learning_rate
    if gamma is not None:
        body["gamma"] = gamma
    if train_timesteps is not None:
        body["train_timesteps"] = train_timesteps
    if new_model is not None:
        body["new_model"] = new_model
    if poll_interval_seconds is not None:
        body["poll_interval_seconds"] = poll_interval_seconds
    if round_timeout_minutes is not None:
        body["round_timeout_minutes"] = round_timeout_minutes

    params = {}
    if user_name is not None:
        params["user_name"] = user_name

    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{API_BASE}/api/v1/optimizer/jobs", json=body, params=params)
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def list_optimizer_jobs(user_name: Optional[str] = None) -> dict:
    """List optimizer jobs, optionally filtered by user_name. Returns each job's
    run_id, ticker, status, current_round, and timestamps."""
    params = {}
    if user_name is not None:
        params["user_name"] = user_name
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{API_BASE}/api/v1/optimizer/jobs", params=params)
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def get_optimizer_job(run_id: str) -> dict:
    """Get details for a single optimizer job by run_id, including per-round
    training results and the optimizer's decision/rationale for each round."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{API_BASE}/api/v1/optimizer/jobs/{run_id}")
        resp.raise_for_status()
        return resp.json()


if __name__ == "__main__":
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", "8000"))

    mcp.run(transport="streamable-http", host=host, port=port)
