# Review

Reviewer findings. Fix every item under Open before the next milestone, then move it to Resolved with a one-line note.

## Open

- [R11] (for M3 IR + M6 store, verified on the real corpus) The R9 fix is correct within a file: 7,290 ModelCalls equals the 7,291 per-file distinct `message.id`s. But resumed and forked Claude Code sessions copy earlier history into a new transcript file. 303 response ids appear in more than one file, and globally there are only 6,461 distinct responses, so cross-file counting still inflates calls and tokens by about 13%. Fix in two parts:
  1. Add `response_id: str | None` to `ModelCall` (Claude Code `message.id`; for DSH, None or a stable `session:turn:step`). Also ensure `ToolCall.call_id` is the `tool_use` id.
  2. In the M6 SQLite store, make `(source, response_id)` and `(source, call_id)` unique, so duplicates are ignored on ingest and the first-seen trajectory keeps them.

  Add a test with two synthetic files sharing one response.

- [R7] (for M5, verified on real data) The DSH adapter emits a `Compaction(reason="prune", ok=True)` for every model-free `compaction/prune` run outside a summary bracket: 257 of the 1483 compactions in the real corpus. If the M5 compaction failure rate and compression ratio count these, the real summary failure rate (468/1226 = 38%) is diluted to 32%. In M5, compute compaction failure rate and ratio over summary compactions only (exclude `reason == "prune"`), report prune runs as a separate count, and add a test with a mixed fixture.

## Resolved

- [R1] Added `turn: int | None` to `ModelCall`, `ToolCall`, and `Edit`, plus `at: int | None` epoch-ms timestamps on `Edit`, `TurnEnd`, and `Outcome` (and `event_timestamp` helper) in `src/dsh_metrics/ir.py`.
- [R2] Added `parse_failures: int = 0` and `unknown_event_types: tuple[str, ...] = ()` to `Trajectory` in `src/dsh_metrics/ir.py`.
- [R3] Enforced secret redaction at the adapter boundary before constructing IR objects in both `dsh.py` and `claude_code.py`, verified by synthetic redaction tests in `tests/test_dsh_adapter.py` and `tests/test_claude_code_adapter.py`.
- [R4] Replaced `ToolCall.output_text` with `error_signature: str | None` and `output_tail: str | None` (bounded to <= 2048 UTF-8 bytes and redacted at the adapter boundary), with unit tests in `tests/test_ir.py` and `tests/test_dsh_adapter.py`.
- [R5] Re-read updated Hard Rules in `CLAUDE.md` and verified zero real paths, project names, session IDs, usernames, prompts, or tool output appear in any committed file; preserved history without reset/rebase.
- [R6] Removed `duckdb` from `pyproject.toml` runtime dependencies (leaving `zstandard` only, `pytest` in `dev`) and uninstalled `duckdb` from `.venv`; M6 store will use standard-library `sqlite3` at `~/.local/share/dsh-metrics/metrics.sqlite`.
- [R8] Updated `parse_dsh_session` in `src/dsh_metrics/adapters/dsh.py` to merge companion `session.jsonl.zstd` and `session.v4.jsonl.zstd` logs by appending `v4` events strictly later than the legacy log's last event, with synthetic overlap test in `tests/test_dsh_adapter.py`.
- [R9] Grouped Claude Code `assistant` entries by `message.id` (fallback `requestId`) into one `ModelCall` using final-entry usage and `[earliest, latest]` timestamps, verified by 3-entry synthetic test and aggregate corpus match (`7,413 == 7,413` per-file, `6,583 == 6,583` global).
- [R10] Stopped real-corpus exploration, restricted `discover_claude_code_files` to top-level `~/.claude/projects/*/*.jsonl`, recorded the nested-subagent scope assumption in `PROGRESS.md`, and verified with a single aggregate sanity run.
