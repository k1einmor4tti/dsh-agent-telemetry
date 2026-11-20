# Progress

## M1 — Skeleton + IR

- **Done**: Created `.venv` with `/opt/homebrew/bin/python3.12`, `pyproject.toml`, `src/dsh_metrics/` package (`__init__.py`, `ir.py`, `cli.py`), `dsh-metrics` console entry point, and `tests/test_ir.py`. Implemented frozen/hashable IR dataclasses (`Trajectory`, `ModelCall`, `ToolCall`, `Edit`, `Compaction`, `TurnEnd`, `Outcome`) with epoch-ms validation (`None` for unknowns, rejecting `0`/negative/bool). Addressed review items `[R1]` (`turn` on `ModelCall`/`ToolCall`/`Edit`, `at` on `Edit`/`TurnEnd`/`Outcome`, `event_timestamp`) and `[R2]` (`parse_failures` and `unknown_event_types` on `Trajectory`).
- **Tests**: `.venv/bin/pytest -v` — 4 passed.
- **Assumptions / Notes**:
  - Installed `setuptools` in `.venv` purely as the PEP 517 build backend for editable package installation.
  - Appended trailing optional fields with defaults after the brief's positional fields so positional construction matches `CLAUDE.md`.
  - `[R3]` is an adapter-level reminder for M2/M3 and stays under `Open` until both adapters and redaction fixture tests are implemented.

## M2 — DSH Adapter + Review Fixes (R4, R5, R6)

- **Done**: Implemented streaming read-only DSH `.jsonl.zstd` adapter (`src/dsh_metrics/adapters/dsh.py`) and secret redaction boundary (`src/dsh_metrics/redact.py`) with synthetic unit tests (`tests/test_dsh_adapter.py`). Resolved `[R4]` (`ToolCall.error_signature` and `output_tail` bounded to <= 2048 UTF-8 bytes and redacted), `[R5]` (audited repo files against public-repo privacy rules), and `[R6]` (dropped `duckdb` so `zstandard` is the sole runtime dependency, switching M6 storage target to `sqlite3`).
- **Tests**: `.venv/bin/pytest -v` — 8 passed. Read-only smoke check on local DSH logs parsed 175 sessions with 0 parse failures (44,593 model calls, 47,704 tool calls, 4,845 edits, 1,483 compactions, 1,496 turn ends).
- **Assumptions / Notes**:
  - When a session directory contains both `session.jsonl.zstd` and `session.v4.jsonl.zstd`, `discover_dsh_files` prefers `session.jsonl.zstd` to avoid double-counting migrated sessions.
  - Non-zero bash `[exit code: N]` sets `exit_code = N` and `is_error = True` so failing shell commands match normalized `ToolCall` semantics across harnesses; failed `edit`/`write` calls (`isError = true`) do not emit `Edit` events.
  - `llm/retry` is counted once per retry attempt (`llm/retry-started` is not double-counted).
  - `[R3]` stays under `Open` until M3 Claude Code adapter and its redaction fixture test are also completed.

## M3 — Claude Code Adapter + Review Fixes (R3, R8, R9, R10, R11-part-1)

- **Done**: Implemented streaming read-only Claude Code `.jsonl` adapter (`src/dsh_metrics/adapters/claude_code.py`) and synthetic tests (`tests/test_claude_code_adapter.py`). Resolved `[R3]` (redaction boundary enforced across both adapters before IR construction), `[R8]` (appending post-upgrade `session.v4.jsonl.zstd` events later than `session.jsonl.zstd`'s last event), `[R9]` (grouping streamed assistant entries by `message.id` / `requestId`, taking final entry usage and `[earliest, latest]` timestamps), `[R10]` (discovering top-level `~/.claude/projects/*/*.jsonl` only), and Part 1 of `[R11]` (`ModelCall.response_id`, `ToolCall.call_id`, first-seen corpus deduplication, and two-file shared-response test).
- **Tests**: `.venv/bin/pytest -v` — 11 passed. Single read-only aggregate sanity check on local Claude Code transcripts parsed 129 top-level files into 128 deduplicated trajectories with 0 parse failures: per-file `ModelCall` count matches distinct per-file `message.id`/`requestId` count exactly (`7,413 == 7,413`), and corpus-deduplicated `ModelCall` count matches global distinct `message.id`/`requestId` count exactly (`6,583 == 6,583`; 5,526,580 output tokens, 7,728 tool calls, 445 edits, 14 compactions, 514 turn ends).
- **Assumptions / Notes**:
  - Subagent transcripts in nested session subdirectories under `~/.claude/projects/<slug>/<session>/` are out of scope for P0 per `[R10]`; inline `isSidechain: true` entries within top-level `.jsonl` files are emitted as separate sidechain trajectories.
  - `[R7]` (M5 compaction metrics excluding `reason == "prune"`) and Part 2 of `[R11]` (M6 SQLite store `UNIQUE(source, response_id)` and `UNIQUE(source, call_id)`) remain under `Open` until M5 and M6 are implemented.

## M4 — Shared Classifiers

- **Done**: Implemented `src/dsh_metrics/classifiers.py` (`fingerprint_command`, `fingerprint_tool_args`, `classify_command_kind`, `classify_tool_call_kind`, `extract_error_signature`, `classify_error_kind`, `extract_read_paths`, `normalize_repo_path`) and synthetic tests in `tests/test_classifiers.py`. Wired automatically into both DSH and Claude Code adapters.
- **Tests**: `.venv/bin/pytest -v` — 58 passed.
- **Assumptions / Notes**:
  - Compound shell commands (`&&`, `;`, `||`) take the highest-priority kind (`test > build > lint > vcs > search > read > other`), while pipelines (`|`) classify by their primary stage (`pytest | tail -20` -> `test`).
  - `tsc` (including `tsc --noEmit`) is classified as `build` so TypeScript compilation checks count toward post-edit verification alongside `error TS\d+` compile-class signatures.
  - `extract_error_signature` returns `None` when `is_error=False` so successful file reads/searches containing error keywords do not produce false positives.

## M5 — Passive Metrics (`metric_version=1`) + Review Fixes (R7, R12, R13)

- **Done**: Implemented `src/dsh_metrics/metrics.py` (`TrajectoryMetrics`, `AggregateMetrics`, `compute_trajectory_metrics`, `aggregate_trajectory_metrics`) covering Categories A, B, C, and D with formula docstrings on every metric, plus synthetic tests in `tests/test_metrics.py`. Resolved `[R7]` (excluding `reason == "prune"` from `compaction_failure_rate` and `compaction_ratio`, reporting `prune_compaction_count` separately), `[R12]` (classifying Make `lint`/`style`/`fmt`/`format`/`vet`/`staticcheck` targets as `lint` ahead of generic `check`), and `[R13]` (adding focused single-behaviour synthetic tests asserting 100% of fields on `TrajectoryMetrics` and `AggregateMetrics`, including DSH `error:CODE:message` turn-end reasons and parallel tool interval merging).
- **Tests**: `.venv/bin/pytest -v` — 75 passed.
- **Assumptions / Notes**:
  - `cache_hit_ratio` uses `cache_read_tokens / (input_tokens + cache_read_tokens + cache_write_tokens)` across both harnesses (verified on local DSH logs that `inputTokens` represents uncached input tokens, matching Claude Code's accounting).
  - Active turn timing merges overlapping model/tool intervals within each turn and excludes inter-turn user idle breaks from `gap_time_ms`.
  - `verification_rate` and `read_to_edit_precision` evaluate over editing trajectories (`edit_count > 0`) and resolve to `None` on non-editing trajectories so read-only Q&A sessions do not skew edit metrics.
  - Part 2 of `[R11]` (SQLite store `UNIQUE(source, response_id)` and `UNIQUE(source, call_id)`) remains under `Open` for M6.

## M6 — SQLite Store + CLI + HTML Report + Review Fixes (R11, R14, R15)

- **Done**: Committed the two pending adapter edge-case fixes with synthetic tests (`2de9b1b`), then implemented `src/dsh_metrics/store.py` (WAL-mode SQLite schema `trajectories`, `model_calls`, `tool_sets`, `tool_calls`, `metrics`, `ingest_log`), `src/dsh_metrics/cli.py` (`dsh-metrics ingest` and `dsh-metrics report`), `src/dsh_metrics/report.py` (self-contained HTML comparison report with light/dark theme tokens and the required observational caveat banner), and `tests/test_store_and_report.py`. Resolved `[R11]` Part 2 (`UNIQUE(source, response_id)` and `UNIQUE(source, call_id)` keeping first-seen trajectories), `[R14]`, and `[R15]` (`tool_sets(id, tools_json)` deduplication via `tool_calls.tool_set_id`, `trajectory_id` FKs, shell-only `raw_args` capped at 1,024 UTF-8 bytes, `{"path", "bytes"}` metadata for write/edit/read, and error/non-zero-exit-only `output_tail`).
- **Tests**: `.venv/bin/pytest -v` — 84 passed.
- **SQLite Store Size (Before / After R15)**:
  - **Before**: `227.07 MB` (`216.55 MiB`, `227,069,952` bytes).
  - **After**: `27.66 MB` (`26.38 MiB`, `27,656,192` bytes; `-87.8%`, well under the 30 MB target; `12` distinct `tool_sets` rows, `max(length(raw_args)) = 1,024` bytes, `0` non-error `output_tail` rows).
- **Real-Data Aggregate Headlines (`metric_version=1`, passive data — tasks differ, not a fair comparison)**:
  - **DSH** (`175` trajectories, `1,501` turns, `0` parse failures):
    - *Cat A*: `44,593` model calls (`29.71`/turn; p50 `24.25`, p90 `77.08`), `46,826` tool calls (`31.20`/turn; p50 `34.10`, p90 `97.23`), near-duplicate shell command rate `10.6%` (`3,526 / 33,127`), `47` repeated-failure groups (`112` calls; escape latency mean `243.94` steps, median `9.0`, p90 `479.2`), unknown-tool rate `0.24%` (`111`), arg-validation error rate `0.12%` (`54`).
    - *Cat B*: `4,559` edits across `52` editing sessions, compile-class errors `1.49` / 100 edits (`68`), test fail->green (`<= 10` steps) `89.3%` (`75 / 84`; steps-to-green mean `2.17`, median `2.0`, p90 `4.0`), post-edit verification rate `65.4%` (`34 / 52`), read->edit precision proxy `14.7%` (`1,050 / 7,143`).
    - *Cat C*: `9.15B` total tokens (`3.82B` uncached input, `5.30B` cache read, `25.63M` output), cache hit ratio `58.1%`, model latency p50 `7.08 s` / p90 `20.53 s`, TTFT p50 `4.31 s` / p90 `13.54 s`, active turn time split `9.4%` model / `89.9%` tool / `0.7%` gap.
    - *Cat D*: truncation rate `0.05%` (`25 / 46,773`), summary compaction failure rate `38.2%` (`468 / 1,226`; `257` separate prune runs), summary compaction token ratio `0.501`, post-compaction re-read rate `51.7%` (`6,097 / 11,795`), LLM retry rate `0.97%` (`431` calls, `827` retries), timeout turn rate `1.07%` (`16`), context-overflow turn rate `0.87%` (`13`), abort rate `8.66%` (`130`).
  - **Claude Code** (`129` deduplicated trajectories from `131` top-level files, `554` turns, `0` parse failures; `830` duplicate responses and `968` duplicate tool calls ignored via SQLite `UNIQUE` constraints):
    - *Cat A*: `7,074` model calls (`12.77`/turn; p50 `6.50`, p90 `39.28`), `8,208` tool calls (`14.82`/turn; p50 `6.75`, p90 `52.63`), near-duplicate shell command rate `5.2%` (`212 / 4,039`), `5` repeated-failure groups (`10` calls; escape latency mean `36.20` steps, median `3.0`, p90 `102.4`), unknown-tool rate `N/A` (`0` schema-checked calls), arg-validation error rate `0.00%` (`0`).
    - *Cat B*: `500` edits across `28` editing sessions, compile-class errors `1.40` / 100 edits (`7`), test fail->green (`<= 10` steps) `71.4%` (`10 / 14`; steps-to-green mean `3.40`, median `3.0`, p90 `7.0`), post-edit verification rate `39.3%` (`11 / 28`), read->edit precision proxy `5.4%` (`73 / 1,350`).
    - *Cat C*: `1.67B` total tokens (`6.92M` uncached input, `1.64B` cache read, `18.19M` cache write, `5.90M` output including `2.76M` reasoning), cache hit ratio `98.5%`, model latency p50 `739 ms` / p90 `3.84 s`, TTFT `N/A`, active turn time split `5.6%` model / `30.1%` tool / `64.3%` gap.
    - *Cat D*: truncation rate `0.95%` (`78 / 8,192`), summary compaction failure rate `0.0%` (`0 / 14`), summary compaction token ratio `0.074`, post-compaction re-read rate `20.0%` (`42 / 210`), LLM retry rate `0.28%` (`20` calls, `120` retries), timeout turn rate `0.54%` (`3`), context-overflow turn rate `0.72%` (`4`), abort rate `6.50%` (`36`).

DONE

