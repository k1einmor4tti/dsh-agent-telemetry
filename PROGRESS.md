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
