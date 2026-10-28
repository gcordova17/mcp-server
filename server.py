"""
MCP server exposing the TickerNN Trading API over Streamable HTTP -- the AI
agent gateway of the Workflow Studio framework (tickerNN's bef/).

Agents authenticate with a bearer token minted here (see `token` below);
each token names an agent and the tickerNN user it acts for. This server
calls tickerNN with its own AGENT_API_KEY and always as the token's user --
no tool takes a user_name. tickerNN knows every such call is an agent's: it
records them as agent actions and refuses, server-side, anything that changes
production (approve, promote, reject, roll back, model promotion). Agents can
design workflows, run them in the sandbox and request promotion; a person
approves.

Run directly:
    python server.py                         # serve on MCP_HOST:MCP_PORT
    python server.py token create --agent claude-desktop --user gcordova
    python server.py token list
    python server.py token revoke --agent claude-desktop

Run in Docker: see Dockerfile / docker-compose.yml.
"""

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
from typing import Any, Optional

import httpx
from mcp.server import MCPServer
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

API_BASE = os.environ.get("TICKERNN_API_URL", "http://host.docker.internal:8001")
AGENT_KEY = os.environ.get("TICKERNN_AGENT_KEY", "")
TOKENS_PATH = os.environ.get("MCP_AGENT_TOKENS", "/app/secrets/agents.json")
PUBLIC_URL = os.environ.get("MCP_PUBLIC_URL", "http://localhost:8000")
WF = "/api/v1/workflow"

READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)


# ------------------------------------------------------------------ tokens

def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def load_tokens(path: str = TOKENS_PATH) -> list:
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return []


def save_tokens(rows: list, path: str = TOKENS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(rows, fh, indent=1)
    os.replace(tmp, path)


class AgentTokenVerifier:
    """Bearer tokens from TOKENS_PATH, which holds only their SHA-256. Re-read
    when the file changes, so minting or revoking needs no restart."""

    def __init__(self, path: str = TOKENS_PATH):
        self.path, self._mtime, self._rows = path, None, []

    def _current(self):
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            return []
        if mtime != self._mtime:
            self._rows, self._mtime = load_tokens(self.path), mtime
        return self._rows

    async def verify_token(self, token: str) -> AccessToken | None:
        digest = _hash(token)
        for row in self._current():
            if not row.get("revoked") and hmac.compare_digest(row["token_sha256"], digest):
                return AccessToken(token=token, client_id=row["agent"], scopes=["workflows"],
                                   subject=row["user_name"])
        return None


mcp = MCPServer(
    name="tickernn-mcp",
    version="0.2.0",
    instructions=(
        "TickerNN model pipeline (HMM / MISA -> WMM / QWMM forecasters -> WPPO trading policy) as governed "
        "workflows. Typical loop: list_workflows -> get_version (production is the active version) -> "
        "fork_draft or configure_<capability> to change a copy -> run_sandbox -> get_execution until COMPLETED "
        "-> compare_execution / get_gates -> request_promotion if the required gates pass. You act as the user "
        "your token names. You cannot approve, promote, reject or roll back: a person does that in Workflow "
        "Studio. Sandbox runs train on the GPU and can take an hour at production settings; one runs at a time. "
        "Risk Intelligence (risk_options, risk_analyze, risk_watchlist, risk_alerts, risk_status) is read-only: "
        "explainable, forward-looking risk per asset, decision support rather than trade instructions."),
    token_verifier=AgentTokenVerifier(),
    auth=AuthSettings(issuer_url=PUBLIC_URL, resource_server_url=None),
)


# ------------------------------------------------------------- API calls

def _identity():
    tok = get_access_token()
    if tok is None or not tok.subject:
        raise ToolError("unauthenticated")
    return tok.client_id, tok.subject


def _detail(resp) -> str:
    try:
        d = resp.json().get("detail")
    except Exception:
        return resp.text[:500]
    if isinstance(d, dict):
        msg = d.get("message") or json.dumps(d)
        if d.get("errors"):
            msg += ": " + "; ".join(map(str, d["errors"]))
        return msg
    if isinstance(d, list):
        return "; ".join(f"{'.'.join(map(str, e.get('loc', [])[1:]))}: {e.get('msg')}" for e in d)
    return str(d)


async def _api(method: str, path: str, *, body: Any = None, params: Optional[dict] = None,
               timeout: float = 120) -> Any:
    agent, user = _identity()
    q = {k: v for k, v in (params or {}).items() if v is not None}
    q["user_name"] = user  # always the token's user, whatever a caller passes
    headers = {"X-API-Key": AGENT_KEY, "X-Agent-Name": agent}
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.request(method, f"{API_BASE}{path}", json=body, params=q, headers=headers)
    if resp.status_code >= 400:
        raise ToolError(f"tickerNN {resp.status_code}: {_detail(resp)}")
    return resp.json()


# ------------------------------------------------------- workflow tools

@mcp.tool(annotations=READ)
async def list_capabilities() -> dict:
    """The node types a workflow can use (hmm, misa, wmm, qwmm, wppo): ports, the contract each port
    carries, and the names of their parameters. Use get_capability for one node's full parameter schema."""
    caps = (await _api("GET", f"{WF}/capabilities"))["capabilities"]
    return {"capabilities": [{
        "id": c["id"], "title": c["title"], "category": c.get("category"), "runtime": c.get("runtime"),
        "inputs": c["inputs"], "outputs": c["outputs"],
        "params": sorted(((c.get("params") or {}).get("properties") or {}).keys()),
        "description": c.get("description")} for c in caps]}


@mcp.tool(annotations=READ)
async def get_capability(capability_id: str) -> dict:
    """One capability's full contract: ports, parameter JSON schema with defaults and bounds, presets
    (MISA), and requirements (WPPO's data-length rules)."""
    return await _api("GET", f"{WF}/capabilities/{capability_id}")


@mcp.tool(annotations=READ)
async def get_policy() -> dict:
    """The promotion policy: which gates are required vs advisory, who may approve, and which workflow owns
    production. is_approver is always false for an agent."""
    return await _api("GET", f"{WF}/policy")


@mcp.tool(annotations=READ)
async def list_workflows() -> dict:
    """Every workflow, with its active (production) version."""
    return await _api("GET", f"{WF}/workflows")


@mcp.tool(annotations=READ)
async def get_workflow(workflow_id: str) -> dict:
    """A workflow and the summary of all its versions (state, base version, note, author)."""
    return await _api("GET", f"{WF}/workflows/{workflow_id}")


@mcp.tool(annotations=READ)
async def get_version(workflow_id: str, version: int) -> dict:
    """One version: its full definition (nodes, edges, settings), validation result (errors, warnings,
    sandbox stages, runnable) and overlay (what it changes relative to the version it was forked from)."""
    return await _api("GET", f"{WF}/workflows/{workflow_id}/versions/{version}")


@mcp.tool(annotations=READ)
async def validate_workflow(definition: dict, base_definition: Optional[dict] = None) -> dict:
    """Check a definition without saving it: contracts, ports, parameters (MISA's checked by MISA itself),
    WPPO data rules, the sandbox stages it would run, and the overlay against base_definition if given."""
    return await _api("POST", f"{WF}/validate", body={"definition": definition, "base": base_definition})


@mcp.tool(annotations=WRITE)
async def create_draft(workflow_id: str, definition: dict, base_version: Optional[int] = None,
                       note: Optional[str] = None) -> dict:
    """Save a full definition as a new DRAFT version (it may be invalid; it can't run until it validates).
    Prefer fork_draft or configure_<capability> for changes to an existing version."""
    return await _api("POST", f"{WF}/workflows/{workflow_id}/versions",
                      body={"definition": definition, "base_version": base_version, "note": note})


@mcp.tool(annotations=WRITE)
async def update_draft(workflow_id: str, version: int, definition: dict, note: Optional[str] = None) -> dict:
    """Replace a DRAFT version's definition. Only your own drafts; frozen versions (SANDBOX onwards) refuse."""
    return await _api("PUT", f"{WF}/workflows/{workflow_id}/versions/{version}",
                      body={"definition": definition, "note": note})


async def _fork(workflow_id: str, from_version: int, node_params: Optional[dict], settings: Optional[dict],
                note: Optional[str], capability: Optional[str] = None) -> dict:
    src = await _api("GET", f"{WF}/workflows/{workflow_id}/versions/{from_version}")
    defn = src["definition"]
    nodes = {n["id"]: n for n in defn["nodes"]}
    for node_id, params in (node_params or {}).items():
        if node_id not in nodes:
            raise ToolError(f"v{from_version} has no node {node_id!r}; nodes: {sorted(nodes)}")
        if capability and nodes[node_id]["capability"] != capability:
            raise ToolError(f"node {node_id!r} is a {nodes[node_id]['capability']} node, not {capability}")
        nodes[node_id]["params"] = params
    if settings:
        defn["settings"] = {**(defn.get("settings") or {}), **settings}
    return await _api("POST", f"{WF}/workflows/{workflow_id}/versions",
                      body={"definition": defn, "base_version": from_version,
                            "note": note or f"forked from v{from_version} by an agent"})


@mcp.tool(annotations=WRITE)
async def fork_draft(workflow_id: str, from_version: int, node_params: Optional[dict] = None,
                     settings: Optional[dict] = None, note: Optional[str] = None) -> dict:
    """Copy a version into a new DRAFT with changes. node_params maps node id -> that node's complete new
    params (they replace the old params; omitted keys fall back to the capability's defaults, so start from
    get_version's params). settings keys (assets, val_rows, test_rows) are merged into the copy's settings.
    Returns the draft with its validation. To change the graph itself, use create_draft."""
    return await _fork(workflow_id, from_version, node_params, settings, note)


@mcp.tool(annotations=WRITE)
async def run_sandbox(workflow_id: str, version: int, note: Optional[str] = None) -> dict:
    """Run a DRAFT (freezing it as SANDBOX) or re-run a SANDBOX version. Every output stays in the run's own
    directory; production is never touched. Trains on the GPU (about an hour at production settings); only
    one run at a time across all users. Poll get_execution until COMPLETED or FAILED."""
    return await _api("POST", f"{WF}/workflows/{workflow_id}/versions/{version}/run",
                      body={"mode": "SANDBOX", "note": note})


def _compact_execution(ex: dict) -> dict:
    return {**{k: ex.get(k) for k in ("execution_id", "workflow_id", "version", "mode", "state", "requested_by",
                                       "created_at", "finished_at", "result")},
            "stages": (ex.get("context") or {}).get("stages"),
            "nodes": [{k: n.get(k) for k in ("node_id", "capability_id", "state", "metrics", "error")}
                      for n in ex.get("nodes", [])]}


@mcp.tool(annotations=READ)
async def list_executions(workflow_id: Optional[str] = None) -> dict:
    """Recent sandbox runs, newest first, optionally for one workflow."""
    return await _api("GET", f"{WF}/executions", params={"workflow_id": workflow_id})


@mcp.tool(annotations=READ)
async def get_execution(execution_id: str) -> dict:
    """A sandbox run's state, stages, and per-node state, progress, metrics (forecaster test loss, WPPO
    verdict) and errors. Timestamps are UTC."""
    return _compact_execution(await _api("GET", f"{WF}/executions/{execution_id}"))


@mcp.tool(annotations=READ)
async def get_execution_log(execution_id: str, stage: str, tail: int = 100) -> dict:
    """The last `tail` lines of one stage's log (stage ids are in get_execution's stages)."""
    return await _api("GET", f"{WF}/executions/{execution_id}/log", params={"stage": stage, "tail": tail})


@mcp.tool(annotations=WRITE)
async def cancel_execution(execution_id: str) -> dict:
    """Stop a queued or running sandbox run. Its partial output is kept for inspection."""
    return _compact_execution(await _api("POST", f"{WF}/executions/{execution_id}/cancel"))


@mcp.tool(annotations=READ)
async def compare_execution(execution_id: str) -> dict:
    """A completed run's WPPO test-window verdict next to the run production serves, with caveats when the
    ticker sets or test windows differ."""
    return await _api("GET", f"{WF}/executions/{execution_id}/compare")


@mcp.tool(annotations=READ)
async def get_gates(workflow_id: str, version: int) -> dict:
    """The promotion gates for a version's latest completed sandbox run, evaluated now against what
    production serves: each gate's level (required/advisory), result and detail."""
    return await _api("GET", f"{WF}/workflows/{workflow_id}/versions/{version}/gates")


@mcp.tool(annotations=WRITE)
async def request_promotion(workflow_id: str, version: int, note: Optional[str] = None) -> dict:
    """Ask for a SANDBOX version to be promoted: if every required gate passes it becomes VALIDATED and waits
    for a person to approve it in Workflow Studio. Explain your evidence in `note`. You cannot approve it."""
    return await _api("POST", f"{WF}/workflows/{workflow_id}/versions/{version}/validate", body={"note": note})


@mcp.tool(annotations=READ)
async def get_promotion_history(workflow_id: str) -> dict:
    """Every lifecycle transition of a workflow's versions (who, human or agent, when, with which gate
    results) and the production pointer: which WPPO run /wppo/decide serves."""
    return await _api("GET", f"{WF}/workflows/{workflow_id}/promotions")


@mcp.tool(annotations=READ)
async def get_lineage(workflow_id: Optional[str] = None, version: Optional[int] = None,
                      execution_id: Optional[str] = None, run_id: Optional[str] = None) -> dict:
    """Where something came from: version ancestry (what each fork changed), sandbox runs, the data cut
    (tickers, train boundaries, test windows), forecaster checkpoints checked against the fingerprints WPPO
    recorded, promotion records (who, human or agent) and the serving pointer. Pass workflow_id (+ version,
    else the active one), an execution_id, a WPPO run_id, or nothing for what production serves now."""
    g = await _api("GET", f"{WF}/lineage", params={"workflow_id": workflow_id, "version": version,
                                                    "execution_id": execution_id, "run_id": run_id})
    return {**{k: g.get(k) for k in ("focus", "workflow_id", "version", "serving_run_id", "notes")},
            "nodes": [{**{k: n.get(k) for k in ("id", "kind", "title", "subtitle", "state", "badges", "at")},
                       "detail": {d["k"]: d["v"] for d in n.get("detail") or []}} for n in g.get("nodes", [])],
            "edges": [[e["from"], e["to"], e.get("label")] for e in g.get("edges", [])]}


@mcp.tool(annotations=READ)
async def diff_versions(workflow_id: str, a: int, b: int) -> dict:
    """Parameter-level changes from version a to version b (effective values: an omitted param counts as
    its capability default), plus added/removed nodes, connections and settings changes."""
    return await _api("GET", f"{WF}/workflows/{workflow_id}/diff", params={"a": a, "b": b})


SEM = "/api/v1/semantic"


@mcp.tool(annotations=READ)
async def semantic_find(query: str, types: Optional[str] = None, k: int = 8) -> dict:
    """Search TickerNN's semantic layer -- a typed map of the system derived from its sources: the site's pages and
    the views on them, API reads, assistant tools and sections, workflow kinds, capabilities (node types), contracts,
    workflows, versions with their last run, serving pointers, gate sets, blueprints and data -- for what a request
    is about. Results are cut to the signed-in user's entitlements. types: comma list of page, view, query, tool,
    section, kind, stage, capability, contract, workflow, version, blueprint, serving, gates, concept, action,
    dataset, series, pit_rule."""
    return await _api("GET", f"{SEM}/find", params={"q": query, "types": types, "k": k})


@mcp.tool(annotations=READ)
async def semantic_describe(entity_id: str) -> dict:
    """One semantic-layer entity (e.g. capability:sensitivity, workflow:wppo-portfolio, kind:forecast) with its
    facts and relations: what it consumes, produces, uses, serves or governs, and what points at it."""
    return await _api("GET", f"{SEM}/describe/{entity_id}")


@mcp.tool(annotations=READ)
async def semantic_brief() -> dict:
    """The system description generated from the semantic layer: every workflow kind with its decision node and
    required gates, every capability with its contracts, the workflows and what production serves. Read it first."""
    return await _api("GET", f"{SEM}/brief")


@mcp.tool(annotations=READ)
async def semantic_impact(entity_id: str) -> dict:
    """What depends on an entity, transitively: for a data series (series:price:btc-usd, series:context:CPI), a
    capability or a contract, the versions, workflows and production serving pointers it reaches, and readers of
    its whole dataset that may include it."""
    return await _api("GET", f"{SEM}/impact/{entity_id}")


@mcp.tool(annotations=READ)
async def semantic_sources(entity_id: str) -> dict:
    """Where an entity comes from: everything it depends on, transitively (semantic_impact read the other way). For
    a page's view (view:risk.stop_loss, view:portfolio.tracker; semantic_find types=view) the production workflow
    version behind it, its capabilities, and the datasets and series it reads, each series with its freshness. A
    workflow counts through its production version, not its drafts."""
    return await _api("GET", f"{SEM}/sources/{entity_id}")


@mcp.tool(annotations=READ)
async def semantic_freshness() -> dict:
    """Data series that aren't fresh (lagging, stale, delisted, missing), with their last stored date."""
    return await _api("GET", f"{SEM}/freshness")


@mcp.tool(annotations=READ)
async def check_action(action: str, params: dict) -> dict:
    """Whether an action would be allowed for your user, and why not, before calling it: the gateway (agents
    never change production), the entitlement it needs (workflow.run to run or cancel sandboxes) and its
    preconditions against live state. action: an action entity id from semantic_find(types="action"), e.g.
    "action:POST /api/v1/workflow/workflows/{workflow_id}/versions/{version}/run"; params: its path parameters."""
    return await _api("POST", f"{SEM}/check", body={"action": action, "params": params})


@mcp.tool(annotations=WRITE)
async def act(action: str, params: dict, body: Optional[dict] = None) -> dict:
    """Check an action (as check_action), then call its route as your user. Refused, with the report, when
    anything blocks it; the route applies its own rules too. Production changes are always refused for agents."""
    return await _api("POST", f"{SEM}/act", body={"action": action, "params": params, "body": body})


@mcp.tool(annotations=WRITE)
async def scaffold_capability(spec: dict, dry_run: bool = True) -> dict:
    """A new capability (node type) as a branch capability/<id>: YAML, stage script, plugin and test as skeletons
    for a person to complete, review and merge -- nothing runs before that. spec: {id, template (evaluation: after
    a WPPO-Portfolio rebalancer | family: a new research workflow kind), title, description, output_contract,
    output_about (for a new contract), tags, examples, params ({name: {type, default, ...}})}. dry_run (default)
    returns the files; false creates the branch -- only after your user confirms. Needs their "Create
    capabilities" entitlement. Start from plan_workflow's code option, which carries a prefilled spec."""
    return await _api("POST", f"{WF}/scaffolds", body={"spec": spec, "dry_run": dry_run})


@mcp.tool(annotations=READ)
async def list_scaffolds() -> dict:
    """Capability scaffold branches waiting for a person to complete and merge, newest first."""
    return await _api("GET", f"{WF}/scaffolds")


@mcp.tool(annotations=READ)
async def plan_workflow(spec: dict) -> dict:
    """Reuse-first options for building what is asked, each validated and costed, cheapest first: reuse (a
    completed run already has it), extend (add an existing capability to a workflow, ports wired), fork (a
    workflow with changed params), compose (a new workflow from its kind's blueprint), code (nothing expresses it;
    a person writes it). spec: {intent, goal: forecast|risk|portfolio|pipeline|benchmark|allocator, kind?,
    workflow_id?, add? (capability id), params? (the decision node's), node_params? ({node or capability id:
    params}, nested dicts merge), settings?, model? (a learner name)}. Saves nothing: save the chosen option's
    definition with create_draft."""
    return await _api("POST", f"{SEM}/plan", body={"spec": spec})


KB = "/api/v1/knowledge"


@mcp.tool(annotations=READ)
async def search_knowledge(query: str, k: int = 6, repo: Optional[str] = None, kind: Optional[str] = None) -> dict:
    """Hybrid (semantic + keyword) search of TickerNN's knowledge base: the READMEs, CLAUDE.md, grooming backlogs,
    plans, FRDs (historical), Python docstrings, capability/policy YAML, SQL migrations and the generated wiki.
    Hits cite `doc_id:Lx-Ly`. repo: tickerNN, apache-docker, mcp-server or wiki; kind: readme, guide, backlog,
    plan, frd, doc, docstring, capability, policy, migration, notebook, wiki (comma list)."""
    return await _api("GET", f"{KB}/search", params={"q": query, "k": k, "repo": repo, "kind": kind})


@mcp.tool(annotations=READ)
async def read_knowledge_source(doc_id: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> dict:
    """Read an indexed document (doc_id from a search hit), optionally a line range. Only indexed text."""
    return await _api("GET", f"{KB}/documents/{doc_id}", params={"start": start_line, "end": end_line})


@mcp.tool(annotations=READ)
async def list_wiki_pages() -> dict:
    """The generated wiki's pages (overview, one per subsystem, known issues, glossary) and whether each is
    generated or stale."""
    return await _api("GET", f"{KB}/wiki")


@mcp.tool(annotations=READ)
async def get_wiki_page(slug: str) -> dict:
    """One generated wiki page as markdown; its citations are doc_id:Lx-Ly."""
    return await _api("GET", f"{KB}/wiki/{slug}")


def register_capability_tools(caps: list):
    """One configure_<capability> tool per registered capability, so each node type's parameter schema is
    in front of the model as part of the tool's description."""
    for cap in caps:
        props = (cap.get("params") or {}).get("properties") or {}
        if not props:
            continue
        cid = cap["id"]

        def make(cid):
            async def configure(workflow_id: str, from_version: int, node_id: str, params: dict,
                                note: Optional[str] = None) -> dict:
                return await _fork(workflow_id, from_version, {node_id: params}, None, note, capability=cid)
            return configure

        presets = f" Presets (use as params): {json.dumps(cap['presets'])}." if cap.get("presets") else ""
        mcp.add_tool(
            make(cid), name=f"configure_{cid}", annotations=WRITE,
            description=(f"Copy a version into a new DRAFT with one {cap['title']} node's params replaced by "
                         f"`params` (omitted keys use the defaults below). {cap.get('description', '').strip()}"
                         f"{presets} Params JSON schema: {json.dumps(cap['params'], separators=(',', ':'))}"))


# ------------------------------------------------- production (read-only)

@mcp.tool(annotations=READ)
async def wppo_decide(ticker: str, position: str = "flat", bars_since_trade: Optional[int] = None) -> dict:
    """Production WPPO's buy/hold/sell for a ticker given its current position (flat/long/short), with each
    forecaster's summary and the run's own verdict. Read-only."""
    return await _api("POST", "/api/v1/wppo/decide",
                      body={"ticker": ticker, "position": position, "bars_since_trade": bars_since_trade})


@mcp.tool(annotations=READ)
async def wppo_runs() -> dict:
    """Every WPPO training run with its verdict, and which one production serves."""
    return await _api("GET", "/api/v1/wppo/runs")


# ------------------------------------------------- risk intelligence (read-only)

RI = "/api/v1/risk-intelligence"


def _compact_analysis(r: dict, keep_series: bool) -> dict:
    """The /analyze response without the drill-down series and the scenarios' day-by-day shock paths
    (thousands of numbers) unless asked for."""
    if not keep_series:
        r = {k: v for k, v in r.items() if k != "series"}
        for group in ("standard", "custom", "crisis_replays"):         # day-by-day shock paths
            for x in (r.get("scenario_sensitivity") or {}).get(group) or []:
                (x.get("shocks") or {}).pop("return_path", None)
    return r


@mcp.tool(annotations=READ)
async def risk_options() -> dict:
    """Risk Intelligence: the analyzable assets by category (equity, fixed income, FX, commodities, energy,
    crypto), the volatility models, horizons (trading days) and periods, and which risk methodology version is
    served (validated, or the built-in uncalibrated defaults)."""
    return await _api("GET", f"{RI}/options")


@mcp.tool(annotations=READ)
async def risk_analyze(asset: str, horizon: int = 20, model: Optional[str] = None, period: str = "1Y",
                       include: Optional[list[str]] = None, include_series: bool = False) -> dict:
    """Forward-looking, explainable risk for one asset (the Risk Intelligence dashboard's analysis): top risk
    drivers and composite score, cross-market stress, regime fragility and transition probabilities, risk
    momentum, liquidity, volatility surprise, hedging efficiency, factor dominance, horizon map, decision posture
    (with its rule and change triggers), scenario sensitivity, risk controls and signal confidence. Every number
    carries unit, horizon, method and quality; unavailable ones say why. model: Ensemble (default), GARCH,
    GARCH-t, EGARCH or GJR-GARCH. include: a subset of sections. A new asset takes ~5-9 s; repeats are cached.
    Decision support, not trade instructions."""
    r = await _api("POST", f"{RI}/analyze", body={"asset": asset, "horizon": horizon, "model": model,
                                                   "period": period, "include": include}, timeout=180)
    return _compact_analysis(r, include_series)


@mcp.tool(annotations=READ)
async def risk_watchlist() -> dict:
    """The user's risk watchlist: watched assets with their alert rules and the last evaluated snapshot
    (posture, composite, probabilities), plus the count of unacknowledged alerts."""
    return await _api("GET", f"{RI}/watchlist")


@mcp.tool(annotations=READ)
async def risk_alerts(unacknowledged_only: bool = True, limit: int = 50) -> dict:
    """Alerts fired by the user's watchlist rules (posture changes, thresholds crossed), newest first."""
    return await _api("GET", f"{RI}/alerts", params={"unacknowledged": unacknowledged_only, "limit": limit})


@mcp.tool(annotations=READ)
async def risk_status() -> dict:
    """Risk Intelligence service health: latency percentiles, cache hit rate, errors, the alert evaluator, and
    the served methodology with its backtest's model metrics."""
    return await _api("GET", f"{RI}/status")


# ------------------------------------------- per-ticker PPO (pre-workflow)

@mcp.tool(annotations=READ)
async def get_health() -> dict:
    """Check if the TickerNN Trading API is healthy and reachable."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{API_BASE}/health")
        resp.raise_for_status()
        return resp.json()


@mcp.tool(annotations=READ)
async def get_available_tickers(list_files_name: Optional[bool] = None) -> dict:
    """Tickers that have trained per-ticker PPO models."""
    return await _api("GET", "/api/v1/models/tickers_available", params={"list_files_name": list_files_name})


@mcp.tool(annotations=READ)
async def get_training_logs(replica_id: Optional[str] = None) -> dict:
    """Per-ticker PPO training logs, optionally for one replica_id."""
    return await _api("GET", "/api/v1/models/train/logs", params={"replica_id": replica_id})


@mcp.tool(annotations=WRITE)
async def train_model(ticker: str, n_epoch: Optional[int] = None, learning_rate: Optional[float] = None,
                      gamma: Optional[float] = None, train_timesteps: Optional[int] = None,
                      new_model: Optional[bool] = None) -> dict:
    """Start or continue per-ticker PPO training (new_model=True trains from scratch). The run is saved as a
    training version; promoting it to that ticker's production tier is for a person. Returns a replica_id
    for get_training_logs."""
    body = {k: v for k, v in {"ticker": ticker, "n_epoch": n_epoch, "learning_rate": learning_rate,
                              "gamma": gamma, "train_timesteps": train_timesteps, "new_model": new_model}.items()
            if v is not None}
    return await _api("POST", "/api/v1/models/train", body=body, timeout=300)


@mcp.tool(annotations=READ)
async def get_historical_chart(ticker: str, days_to_render: Optional[int] = None) -> dict:
    """Historical price chart data for a ticker over the past days_to_render days."""
    body = {"ticker": ticker, **({"days_to_render": days_to_render} if days_to_render is not None else {})}
    return await _api("POST", "/api/v1/charts/historical", body=body)


@mcp.tool(annotations=READ)
async def get_realtime_chart(ticker: str, since: str, real_time: Optional[bool] = True) -> dict:
    """Real-time chart data for a ticker from `since` (ISO 8601, e.g. 2026-01-01T00:00:00)."""
    return await _api("POST", "/api/v1/charts/realtime",
                      body={"ticker": ticker, "since": since, "real_time": real_time})


@mcp.tool(annotations=READ)
async def get_batch_chart(since: str, real_time: Optional[bool] = True) -> dict:
    """Chart data for every available ticker from `since` (ISO 8601)."""
    return await _api("POST", "/api/v1/charts/batch", body={"since": since, "real_time": real_time})


@mcp.tool(annotations=WRITE)
async def launch_optimizer_job(ticker: str, max_rounds: int, n_epoch: Optional[int] = None,
                               learning_rate: Optional[float] = None, gamma: Optional[float] = None,
                               train_timesteps: Optional[int] = None, new_model: Optional[bool] = None,
                               poll_interval_seconds: Optional[int] = None,
                               round_timeout_minutes: Optional[int] = None) -> dict:
    """Launch a multi-round per-ticker hyperparameter optimizer job in the background. Use
    list_optimizer_jobs to find it."""
    body = {k: v for k, v in {"ticker": ticker, "max_rounds": max_rounds, "n_epoch": n_epoch,
                              "learning_rate": learning_rate, "gamma": gamma, "train_timesteps": train_timesteps,
                              "new_model": new_model, "poll_interval_seconds": poll_interval_seconds,
                              "round_timeout_minutes": round_timeout_minutes}.items() if v is not None}
    return await _api("POST", "/api/v1/optimizer/jobs", body=body)


@mcp.tool(annotations=READ)
async def list_optimizer_jobs() -> dict:
    """Your optimizer jobs: run_id, ticker, status, current_round and timestamps."""
    return await _api("GET", "/api/v1/optimizer/jobs")


@mcp.tool(annotations=READ)
async def get_optimizer_job(run_id: str) -> dict:
    """One optimizer job with per-round results and the optimizer's rationale."""
    return await _api("GET", f"/api/v1/optimizer/jobs/{run_id}")


# ------------------------------------------------------------------- main

async def _fetch_capabilities(attempts: int = 20, delay: float = 3.0) -> list:
    """Capabilities for the configure_* tools, read once at startup with the
    agent key (tickerNN may still be starting when this container does)."""
    for _ in range(attempts):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"{API_BASE}{WF}/capabilities", headers={"X-API-Key": AGENT_KEY},
                                     params={"user_name": "default"})
            if r.status_code == 200:
                return r.json()["capabilities"]
            print(f"[mcp] capabilities: HTTP {r.status_code}: {_detail(r)}", flush=True)
        except Exception as e:
            print(f"[mcp] capabilities: {type(e).__name__}: {e}", flush=True)
        await asyncio.sleep(delay)
    return []


def token_cli(argv):
    p = argparse.ArgumentParser(prog="server.py token")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--agent", required=True, help="a name for the client, e.g. claude-desktop")
    c.add_argument("--user", required=True, help="the tickerNN user the agent acts for")
    sub.add_parser("list")
    r = sub.add_parser("revoke")
    r.add_argument("--agent", required=True)
    args = p.parse_args(argv)
    rows = load_tokens()
    if args.cmd == "create":
        if any(x["agent"] == args.agent and not x.get("revoked") for x in rows):
            raise SystemExit(f"agent {args.agent!r} already has a live token; revoke it first")
        token = "tnn_" + secrets.token_urlsafe(32)
        rows.append({"agent": args.agent, "user_name": args.user, "token_sha256": _hash(token),
                     "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "revoked": False})
        save_tokens(rows)
        print(f"token for {args.agent} (acts as {args.user}); shown once, only its hash is stored:\n{token}")
    elif args.cmd == "list":
        for x in rows:
            print(f"{x['agent']:24} user={x['user_name']:16} created={x['created_at']} "
                  f"{'REVOKED ' + x.get('revoked_at', '') if x.get('revoked') else 'active'}")
    else:
        hit = [x for x in rows if x["agent"] == args.agent and not x.get("revoked")]
        if not hit:
            raise SystemExit(f"no live token for {args.agent!r}")
        for x in hit:
            x["revoked"], x["revoked_at"] = True, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save_tokens(rows)
        print(f"revoked {args.agent}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "token":
        token_cli(sys.argv[2:])
        sys.exit(0)
    if not AGENT_KEY:
        raise SystemExit("TICKERNN_AGENT_KEY is not set (see .env)")
    register_capability_tools(asyncio.run(_fetch_capabilities()))
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", "8000"))
    mcp.run(transport="streamable-http", host=host, port=port)
