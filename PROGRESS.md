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
