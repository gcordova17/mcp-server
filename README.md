# tickernn-mcp: the AI agent gateway

An MCP server (official Python SDK, `mcp==2.0.0`, Streamable HTTP on
`:8000/mcp`) that lets AI agents work with the TickerNN model pipelines
(HMM / MISA -> WMM / QWMM -> WPPO, and each user's allocator: ticker signals -> panel -> allocator) through tickerNN's Workflow Studio
framework (`bef/` in the tickerNN repo), under the same governance as people.

## What an agent can and can't do

| Can | Can't |
|---|---|
| Read capabilities, workflows, versions, policy, gates, history, lineage (`get_lineage`) and version diffs (`diff_versions`) | Approve, promote, reject or roll back |
| Search the knowledge base and read the generated wiki (`search_knowledge`, `read_knowledge_source`, `list_wiki_pages`, `get_wiki_page`) | Reindex it (it can spend Claude tokens) |
| Fork a version into a draft with changes (`fork_draft`, `configure_<capability>`) | Act as any user but the one its token names |
| Run a draft in the sandbox, follow it, read logs, cancel it, compare it with production | Write anything production serves |
| Request promotion (`request_promotion`: SANDBOX -> VALIDATED when the required gates pass) | Promote per-ticker / WMM / QWMM models |
| Ask production WPPO for a decision (`wppo_decide`) | |

The right-hand column is enforced by tickerNN, not by which tools exist
here: this server calls tickerNN with its own `AGENT_API_KEY`, and
tickerNN's `agent_gateway.py` refuses every production-changing route for
that key and records everything else it does as an `agent` action
(`TBL_BEF_Promotion.actor_kind`, audit `detail.actor_kind`).

## Authentication

Every request needs `Authorization: Bearer <token>`. A token names an agent
and the tickerNN user it acts for; tools never take a `user_name`. Only each
token's SHA-256 is stored (`secrets/agents.json`, mode 600, gitignored), and
the file is re-read when it changes, so no restart is needed:

```bash
docker exec mcp-server python server.py token create --agent claude-desktop --user gcordova
docker exec mcp-server python server.py token list
docker exec mcp-server python server.py token revoke --agent claude-desktop
```

`create` prints the token once. `.env` (gitignored) holds
`TICKERNN_AGENT_KEY` (the same value as tickerNN's `AGENT_API_KEY`) and
`TICKERNN_API_URL`.

## Connect a client

Any client that supports Streamable HTTP with a header, e.g. Claude Code:

```bash
claude mcp add --transport http tickernn http://<host>:8000/mcp \
  --header "Authorization: Bearer <token>"
```

## Tools

- **Workflows:** `list_capabilities`, `get_capability`, `get_policy`,
  `list_workflows`, `get_workflow`, `get_version`, `validate_workflow`,
  `create_draft`, `update_draft`, `fork_draft`,
  `configure_misa` / `configure_wmm` / `configure_qwmm` / `configure_wppo`
  (generated at startup from tickerNN's capability registry; each carries
  its node's parameter schema in its description).
- **Sandbox:** `run_sandbox`, `list_executions`, `get_execution`,
  `get_execution_log`, `cancel_execution`, `compare_execution`.
- **Promotion:** `get_gates`, `request_promotion`, `get_promotion_history`.
- **Production (read-only):** `wppo_decide`, `wppo_runs`.
- **Per-ticker PPO (pre-workflow):** `get_health`, `get_available_tickers`,
  `get_training_logs`, `train_model`, `get_*_chart`,
  `launch_optimizer_job`, `list_optimizer_jobs`, `get_optimizer_job`.

## Run it

```bash
docker compose up -d --build
docker compose logs -f mcp-server
```

The `configure_*` tools are registered from tickerNN's
`/api/v1/workflow/capabilities` at startup (retrying for about a minute);
restart this container after adding a capability there.
