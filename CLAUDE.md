# dsh-agent-telemetry — implementation brief (P0: offline analyzer)

You are implementing P0 of the design "DSH Agent intelligence metrics plugin". P0 is an
offline analyzer that turns DeepSeek Harness (DSH) session logs and Claude Code
transcripts into one normalized trajectory format, computes agent metrics on
both with the same code, stores them in SQLite, and renders an HTML comparison
report. No DSH plugin, no live probe, no evaluation runner in P0.

## Collaboration protocol (mandatory)

- Work milestone by milestone (M1..M6 below). After each milestone: run the
  tests, `git commit` (small, descriptive message), append a short entry to
  `PROGRESS.md` (what was done, test result, open questions), then read
  `REVIEW.md`. Fix every item under "Open" before starting the next
  milestone, and move each fixed item to "Resolved" with a one-line note.
- A reviewer checks the repo every ~5 minutes and writes findings to
  `REVIEW.md`. Treat them as binding requirements unless they conflict with
  this file; if they do, say so in PROGRESS.md instead of guessing.
- Do not ask questions and wait; decide, record the assumption in
  PROGRESS.md, and continue.
- When all milestones are done and REVIEW.md has no open items, write
  `DONE` as the last line of PROGRESS.md and stop.

## Hard rules

- This repository is public on GitHub. Nothing committed may contain real
  data: no real file paths, project or repository names, session ids, user
  names, hostnames, prompts, or tool output from `~/.dsh` or `~/.claude`.
  PROGRESS.md may only report aggregate numbers (counts, rates, percentiles).
- Read-only access to real data: never modify, move, or delete anything under
  `~/.dsh` or `~/.claude`. Open files read-only.
- Never commit real log content, real paths of other projects, or secrets.
  Test fixtures must be small synthetic files written by you that mimic the
  formats below.
- Redact secrets at the adapter boundary (API keys, bearer tokens, cookies,
  `token=` query values, `sk-...`, `ghp_...`, long base64/hex secrets) before
  anything reaches the store or the report.
- No network except installing the declared dependencies. No `git push`.
- Python 3.12 via `/opt/homebrew/bin/python3.12`; create `.venv` with it.
  Deployment must stay simple and free: a local CLI, no service, no external
  API calls, installable with one command
  (`uv tool install git+https://github.com/k1einmor4tti/dsh-agent-telemetry`).
  The only runtime dependency is `zstandard`; storage uses the standard
  library `sqlite3`; `pytest` is a dev-only extra. Nothing else without a
  recorded reason in PROGRESS.md.
- Package layout: `src/dsh_metrics/` (installable, `pyproject.toml`), tests in
  `tests/`, CLI entry point `dsh-metrics`.

## Milestones

- **M1 Skeleton + IR.** pyproject, package, `.venv`, pytest wiring.
  Immutable dataclasses for the trajectory IR:
  `Trajectory(source, session_id, cwd, model_routes, started_at, events)`,
  `ModelCall(step, started_at, first_token_at, ended_at, input_tokens,
  cache_read_tokens, cache_write_tokens, output_tokens, reasoning_tokens,
  retries, provider, model)`,
  `ToolCall(step, call_id, name, args_fingerprint, raw_args, started_at,
  ended_at, is_error, exit_code, output_bytes, truncated, available_tools)`,
  `Edit(step, path, before_hash, after_hash)`,
  `Compaction(started_at, ended_at, ok, tokens_before, tokens_after, reason)`,
  `TurnEnd(turn, reason)`, `Outcome(task_id, result, source)`.
  Times are epoch milliseconds; unknown values are `None`, never 0.
- **M2 DSH adapter.** Input: `~/.dsh/sessions/<cwd-slug>/session-<id>/session.jsonl.zstd`
  or `session.v4.jsonl.zstd` (zstd-compressed JSON lines). Facts:
  - Line 1 is `{"type":"session","id","createdAt","cwd",...}`. Other events
    have `type`, `seq`, `time` (epoch ms), `data`.
  - Chunk records `reasoning-chunks` / `text-chunks` / `tool-call-chunks` use
    `seq0`/`time0`; the first chunk time of a step is the first-token time.
  - `turn/start {turn}`, `step/start {turn,step}`, `step/end`,
    `turn/end {turn, reason:{kind: completed|error|aborted|max-tokens|...,
    error?:{code,message}}}`.
  - `request/header {header:{config:{provider,model}, system, tools:[...]}}`
    gives the tool list available for the following steps;
    `request/context {provider, model, contextWindow}`.
  - `assistant/message {turn, step, message:{content:[...]}, usage:{inputTokens,
    outputTokens, cacheReadTokens?, cacheWriteTokens?, reasoningTokens?}}`.
  - `tool/call {turn, step, callId, name, arguments(JSON string)}`;
    `tool/result {turn, step, message:{content:[{type:"tool-result",
    toolCallId, isError, content:[{type:"text",text}...]}]}}`. Bash output
    ends with `[exit code: N]` when N != 0. Spilled output contains a notice
    mentioning the omitted byte count — detect it as `truncated`.
  - `llm/retry`, `compaction/start`, `compaction/end` (may carry an error),
    `compaction/summary`, `compaction/prune`.
  - Edits come from `edit` / `write` tool calls (`path` or `file_path` arg).
  Inspect a few real files read-only to confirm, but keep tests synthetic.
- **M3 Claude Code adapter.** Input: `~/.claude/projects/<slug>/*.jsonl`
  (plain JSON lines). Facts: entries with `type` `assistant` / `user` carry
  `timestamp` (ISO 8601), `uuid`, `parentUuid`, `isSidechain`,
  `message.content` blocks (`thinking` with redacted text, `text`,
  `tool_use {id,name,input}`, `tool_result {tool_use_id, is_error, content}`),
  assistant `message.usage {input_tokens, cache_creation_input_tokens,
  cache_read_input_tokens, output_tokens, output_tokens_details.thinking_tokens}`,
  `thinkingDurationMs`, `isCompactSummary`, `system` entries with
  `compactMetadata`. Many other entry types exist (api-request, attachment,
  file-history-*, cost-state, …): ignore unknown types, never crash on them.
  Sidechain (subagent) entries become their own trajectories.
- **M4 Shared classifiers.** Command fingerprint (normalize numbers,
  whitespace, quoted strings, line ranges like `sed -n 10,20p`); command kind
  (test / build / lint / read / search / vcs / other) for go, python, node,
  make; error signature extraction (compile-class: `syntax error`,
  `undefined:`, `error TS\d+`, `SyntaxError`, `cannot find`; test-class:
  `FAIL`, `--- FAIL`, `AssertionError`, `panic:`).
- **M5 Metrics (passive, metric_version=1).** Per trajectory and aggregated:
  A: model calls & tool calls per turn; near-duplicate command rate (same
  fingerprint); repeated failures = same (command fingerprint, error
  signature) seen ≥ 2 times, plus escape latency in steps; unknown-tool
  calls (name not in `available_tools`) and argument-validation errors.
  B: compile-class error rate per 100 edits; test fail→green conversion
  within 10 steps and steps-to-green; verification rate (tests/build after
  last edit); read→edit precision proxy.
  C: token totals, cache hit ratio, model time vs tool time vs gaps.
  D: truncation rate; compaction failure rate and ratio; post-compaction
  re-read rate; retry / timeout / context-overflow rates; abort rate.
  Every metric documents its formula in a docstring.
- **M6 Store + CLI + report.** SQLite (standard library `sqlite3`, WAL mode)
  at `~/.local/share/dsh-metrics/metrics.sqlite` (override with `--db`).
  Tables: trajectories, model_calls, tool_calls, metrics (with
  metric_version), ingest_log. `dsh-metrics ingest --source dsh|claude-code
  [--since DATE]` is incremental and idempotent;
  `dsh-metrics report --out report.html` renders a self-contained HTML
  comparing DSH vs Claude Code, with a visible banner: passive data, tasks
  differ, not a fair comparison. Run both on the real data and paste the
  headline numbers into PROGRESS.md.

## Quality bar

- pytest green at every commit; each adapter and metric has tests on synthetic
  fixtures, including malformed lines and unknown event types.
- Adapters stream files (no full-file loads for large logs) and report a
  parse-failure count instead of raising.
- Keep functions small; no dead code; no placeholder modules.
