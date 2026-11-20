# Progress

## M1 — Skeleton + IR

- **Done**: Created `.venv` with `/opt/homebrew/bin/python3.12`, `pyproject.toml`, `src/dsh_metrics/` package (`__init__.py`, `ir.py`, `cli.py`), `dsh-metrics` console entry point, and `tests/test_ir.py`. Implemented frozen/hashable IR dataclasses (`Trajectory`, `ModelCall`, `ToolCall`, `Edit`, `Compaction`, `TurnEnd`, `Outcome`) with epoch-ms validation (`None` for unknowns, rejecting `0`/negative/bool). Addressed review items `[R1]` (`turn` on `ModelCall`/`ToolCall`/`Edit`, `at` on `Edit`/`TurnEnd`/`Outcome`, `event_timestamp`) and `[R2]` (`parse_failures` and `unknown_event_types` on `Trajectory`).
- **Tests**: `.venv/bin/pytest -v` — 4 passed.
- **Assumptions / Notes**:
  - Installed `setuptools` in `.venv` purely as the PEP 517 build backend for editable package installation; runtime dependencies remain strictly `duckdb` and `zstandard` (+ `pytest` for tests).
  - Appended `turn`, `at`, `output_text`, `parse_failures`, and `unknown_event_types` after the brief's positional fields with defaults so positional construction matches `CLAUDE.md` while satisfying `[R1]`/`[R2]`.
  - `[R3]` is an adapter-level reminder for M2/M3 and stays under `Open` until both adapters and redaction fixture tests are implemented.
