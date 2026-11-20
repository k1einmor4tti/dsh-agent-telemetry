# Review

Reviewer findings. Fix every item under Open before the next milestone, then move it to Resolved with a one-line note.

## Open

- [R3] Reminder for M2/M3: `ToolCall.raw_args` must hold the already-redacted argument string. Redaction happens in the adapter before constructing IR objects, and a test must prove that a fixture containing `sk-…` / `Bearer …` never reaches `raw_args` or `args_fingerprint`.

## Resolved

- [R1] Added `turn: int | None` to `ModelCall`, `ToolCall`, and `Edit`, plus `at: int | None` epoch-ms timestamps on `Edit`, `TurnEnd`, and `Outcome` (and `event_timestamp` helper) in `src/dsh_metrics/ir.py`.
- [R2] Added `parse_failures: int = 0` and `unknown_event_types: tuple[str, ...] = ()` to `Trajectory` in `src/dsh_metrics/ir.py`.
- [R4] Replaced `ToolCall.output_text` with `error_signature: str | None` and `output_tail: str | None` (bounded to <= 2048 UTF-8 bytes and redacted at the adapter boundary), with unit tests in `tests/test_ir.py` and `tests/test_dsh_adapter.py`.
- [R5] Re-read updated Hard Rules in `CLAUDE.md` and verified zero real paths, project names, session IDs, usernames, prompts, or tool output appear in any committed file; preserved history without reset/rebase.
- [R6] Removed `duckdb` from `pyproject.toml` runtime dependencies (leaving `zstandard` only, `pytest` in `dev`) and uninstalled `duckdb` from `.venv`; M6 store will use standard-library `sqlite3` at `~/.local/share/dsh-metrics/metrics.sqlite`.
