# Review

Reviewer findings. Fix every item under Open before the next milestone, then move it to Resolved with a one-line note.

## Open

(none)

## Resolved

- [R11] Added `ModelCall.response_id` and `ToolCall.call_id` (M3) and enforced `UNIQUE(source, response_id)` and `UNIQUE(source, call_id)` in `src/dsh_metrics/store.py` so first-seen trajectories retain shared responses/calls and cross-file duplicates are ignored, verified by synthetic two-file tests in `tests/test_claude_code_adapter.py` and `tests/test_store_and_report.py`.
- [R14] Committed the pending DSH/Claude Code adapter tweaks with synthetic tests (`2de9b1b`), then implemented `store.py`, `cli.py`, `report.py`, and `tests/test_store_and_report.py`, followed by a single real ingest + report run recorded in `PROGRESS.md`.
- [R15] Normalized distinct tool lists into `tool_sets(id, tools_json)` (`tool_calls.tool_set_id`), normalized `trajectory_id` FKs, capped shell `raw_args` at 1 KB (path + byte length only for write/edit/read, `NULL` otherwise), and kept `output_tail` only on error/non-zero exit calls, shrinking `metrics.sqlite` from 227.07 MB to 27.66 MB (26.38 MiB) with cap assertions in `tests/test_store_and_report.py`.
- [R1] Added `turn: int | None` to `ModelCall`, `ToolCall`, and `Edit`, plus `at: int | None` epoch-ms timestamps on `Edit`, `TurnEnd`, and `Outcome` (and `event_timestamp` helper) in `src/dsh_metrics/ir.py`.
- [R2] Added `parse_failures: int = 0` and `unknown_event_types: tuple[str, ...] = ()` to `Trajectory` in `src/dsh_metrics/ir.py`.
- [R3] Enforced secret redaction at the adapter boundary before constructing IR objects in both `dsh.py` and `claude_code.py`, verified by synthetic redaction tests in `tests/test_dsh_adapter.py` and `tests/test_claude_code_adapter.py`.
- [R4] Replaced `ToolCall.output_text` with `error_signature: str | None` and `output_tail: str | None` (bounded to <= 2048 UTF-8 bytes and redacted at the adapter boundary), with unit tests in `tests/test_ir.py` and `tests/test_dsh_adapter.py`.
- [R5] Re-read updated Hard Rules in `CLAUDE.md` and verified zero real paths, project names, session IDs, usernames, prompts, or tool output appear in any committed file; preserved history without reset/rebase.
- [R6] Removed `duckdb` from `pyproject.toml` runtime dependencies (leaving `zstandard` only, `pytest` in `dev`) and uninstalled `duckdb` from `.venv`; M6 store will use standard-library `sqlite3` at `~/.local/share/dsh-metrics/metrics.sqlite`.
- [R7] Computed `compaction_failure_rate` and `compaction_ratio` over summary compactions only (`reason != "prune"`), reported `prune_compaction_count` separately in `src/dsh_metrics/metrics.py`, and verified with a mixed fixture in `tests/test_metrics.py`.
- [R8] Updated `parse_dsh_session` in `src/dsh_metrics/adapters/dsh.py` to merge companion `session.jsonl.zstd` and `session.v4.jsonl.zstd` logs by appending `v4` events strictly later than the legacy log's last event, with synthetic overlap test in `tests/test_dsh_adapter.py`.
- [R9] Grouped Claude Code `assistant` entries by `message.id` (fallback `requestId`) into one `ModelCall` using final-entry usage and `[earliest, latest]` timestamps, verified by 3-entry synthetic test and aggregate corpus match (`7,413 == 7,413` per-file, `6,583 == 6,583` global).
- [R10] Stopped real-corpus exploration, restricted `discover_claude_code_files` to top-level `~/.claude/projects/*/*.jsonl`, recorded the nested-subagent scope assumption in `PROGRESS.md`, and verified with a single aggregate sanity run.
- [R12] Updated `_classify_make_targets` in `src/dsh_metrics/classifiers.py` so targets containing `lint`, `style`, `fmt`, `format`, `vet`, or `staticcheck` classify as `lint` ahead of generic `check`, with tests in `tests/test_classifiers.py`.
- [R13] Added focused single-behaviour synthetic tests in `tests/test_metrics.py` (covering `escape_latency_steps_median`, `steps_to_green_median`/`steps_to_green_values`, `post_compaction_reread_count`/`post_compaction_read_count`, `aborted_turn_count`/`timeout_turn_count`/`context_overflow_turn_count` with `error:CODE:message`, truncation counts, retry counts, parallel tool interval merging, and 100% of `TrajectoryMetrics`/`AggregateMetrics` fields).
