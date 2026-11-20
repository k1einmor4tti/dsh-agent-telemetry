"""SQLite storage and incremental ingestion for normalized trajectories and metrics (M6)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any

from dsh_metrics.adapters.claude_code import (
    discover_claude_code_files,
    parse_claude_code_file,
)
from dsh_metrics.adapters.dsh import discover_dsh_files, parse_dsh_session
from dsh_metrics.classifiers import (
    NATIVE_EDIT_TOOLS,
    NATIVE_READ_TOOLS,
    SHELL_TOOL_NAMES,
    CommandKind,
    classify_tool_call_kind,
)
from dsh_metrics.ir import (
    Compaction,
    Edit,
    ModelCall,
    Outcome,
    ToolCall,
    Trajectory,
    TrajectoryEvent,
    TurnEnd,
    event_timestamp,
)
from dsh_metrics.metrics import (
    METRIC_VERSION,
    AggregateMetrics,
    TrajectoryMetrics,
    _percentile_or_none,
    aggregate_trajectory_metrics,
    compute_trajectory_metrics,
)
from dsh_metrics.redact import redact_secrets

DEFAULT_DB_PATH = Path("~/.local/share/dsh-metrics/metrics.sqlite")
DEFAULT_DSH_ROOT = Path("~/.dsh/sessions")
DEFAULT_CLAUDE_CODE_ROOT = Path("~/.claude/projects")
VALID_SOURCES = ("dsh", "claude-code")

MAX_SHELL_RAW_ARGS_BYTES = 1024
MAX_OUTPUT_TAIL_BYTES = 2048
MAX_FINGERPRINT_BYTES = 48

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trajectories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    session_id TEXT NOT NULL,
    origin_file TEXT,
    cwd TEXT,
    model_routes_json TEXT NOT NULL,
    started_at INTEGER,
    parse_failures INTEGER NOT NULL DEFAULT 0,
    unknown_event_types_json TEXT NOT NULL,
    ingested_at INTEGER NOT NULL,
    UNIQUE (source, session_id)
);

CREATE TABLE IF NOT EXISTS model_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trajectory_id INTEGER NOT NULL REFERENCES trajectories(id),
    source TEXT NOT NULL,
    seq_idx INTEGER NOT NULL,
    turn INTEGER,
    step INTEGER,
    response_id TEXT,
    started_at INTEGER,
    first_token_at INTEGER,
    ended_at INTEGER,
    input_tokens INTEGER,
    cache_read_tokens INTEGER,
    cache_write_tokens INTEGER,
    output_tokens INTEGER,
    reasoning_tokens INTEGER,
    retries INTEGER,
    provider TEXT,
    model TEXT,
    UNIQUE (source, response_id)
);

CREATE TABLE IF NOT EXISTS tool_sets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tools_json TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trajectory_id INTEGER NOT NULL REFERENCES trajectories(id),
    source TEXT NOT NULL,
    seq_idx INTEGER NOT NULL,
    turn INTEGER,
    step INTEGER,
    call_id TEXT,
    name TEXT NOT NULL,
    args_fingerprint TEXT,
    raw_args TEXT,
    started_at INTEGER,
    ended_at INTEGER,
    is_error INTEGER,
    exit_code INTEGER,
    output_bytes INTEGER,
    truncated INTEGER,
    tool_set_id INTEGER REFERENCES tool_sets(id),
    error_signature TEXT,
    output_tail TEXT,
    UNIQUE (source, call_id)
);

CREATE TABLE IF NOT EXISTS metrics (
    source TEXT NOT NULL,
    session_id TEXT NOT NULL,
    metric_version INTEGER NOT NULL,
    turn_count INTEGER NOT NULL,
    model_call_count INTEGER NOT NULL,
    tool_call_count INTEGER NOT NULL,
    edit_count INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    cache_hit_ratio REAL,
    model_time_ms INTEGER NOT NULL,
    tool_time_ms INTEGER NOT NULL,
    gap_time_ms INTEGER NOT NULL,
    metrics_json TEXT NOT NULL,
    computed_at INTEGER NOT NULL,
    PRIMARY KEY (source, session_id, metric_version)
);

CREATE TABLE IF NOT EXISTS ingest_log (
    source TEXT NOT NULL,
    file_path TEXT NOT NULL,
    mtime_ns INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    session_ids_json TEXT NOT NULL,
    parse_failures INTEGER NOT NULL DEFAULT 0,
    metric_version INTEGER NOT NULL,
    ingested_at INTEGER NOT NULL,
    PRIMARY KEY (source, file_path)
);
"""


def parse_since_ms(since: str | int | None) -> int | None:
    """Parse an ISO-8601 date (`YYYY-MM-DD`) or timestamp into positive UTC epoch ms."""
    if since is None:
        return None
    if isinstance(since, int) and not isinstance(since, bool):
        if since <= 0:
            raise ValueError(f"since epoch ms must be positive, got {since!r}")
        return since
    if not isinstance(since, str) or not since.strip():
        raise ValueError(f"Invalid --since value: {since!r}")

    text = since.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid --since date {since!r}; expected YYYY-MM-DD or ISO-8601 timestamp"
        ) from exc

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ms = int(dt.timestamp() * 1000)
    if ms <= 0:
        raise ValueError(f"--since date must be after 1970-01-01, got {since!r}")
    return ms


def _bool_to_sql(value: bool | None) -> int | None:
    if value is None:
        return None
    return 1 if value else 0


def _now_ms() -> int:
    return int(time.time() * 1000)


def _cap_utf8_bytes(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _cap_json_field_bytes(key: str, value: str, max_bytes: int) -> str:
    candidate = json.dumps({key: value}, ensure_ascii=False)
    if len(candidate.encode("utf-8")) <= max_bytes:
        return candidate
    lo, hi = 0, min(len(value), max_bytes)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        trial = json.dumps({key: value[:mid]}, ensure_ascii=False)
        if len(trial.encode("utf-8")) <= max_bytes:
            lo = mid
        else:
            hi = mid - 1
    return json.dumps({key: value[:lo]}, ensure_ascii=False)


def _cap_path_and_bytes_json(path_str: str, byte_len: int, max_bytes: int) -> str:
    candidate = json.dumps({"path": path_str, "bytes": byte_len}, ensure_ascii=False)
    if len(candidate.encode("utf-8")) <= max_bytes:
        return candidate
    lo, hi = 0, min(len(path_str), max_bytes)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        trial = json.dumps(
            {"path": path_str[:mid], "bytes": byte_len}, ensure_ascii=False
        )
        if len(trial.encode("utf-8")) <= max_bytes:
            lo = mid
        else:
            hi = mid - 1
    return json.dumps({"path": path_str[:lo], "bytes": byte_len}, ensure_ascii=False)


def _compact_raw_args_for_store(name: str, raw_args: str | None) -> str | None:
    """Compact `ToolCall.raw_args` per R15 before SQLite persistence.

    - Shell-like tools: store only the redacted command, capped at 1024 UTF-8 bytes
      (omitting heredoc bodies after `<<` + newline so whole file bodies are not stored).
    - Write / edit / read tools: store only `{"path": <redacted>, "bytes": <utf8_len>}`.
    - All other tools: store `None` (the redacted `args_fingerprint` covers duplicate checks).
    """
    if not raw_args:
        return None
    redacted_raw = redact_secrets(raw_args)
    raw_bytes = len(redacted_raw.encode("utf-8"))
    tlower = name.strip().lower()

    parsed: Mapping[str, Any] | None = None
    try:
        loaded = json.loads(redacted_raw)
        if isinstance(loaded, Mapping):
            parsed = loaded
    except json.JSONDecodeError:
        parsed = None

    if tlower in NATIVE_EDIT_TOOLS or tlower in NATIVE_READ_TOOLS:
        path_val = ""
        if parsed is not None:
            raw_path = (
                parsed.get("file_path")
                or parsed.get("path")
                or parsed.get("notebook_path")
            )
            if isinstance(raw_path, str):
                path_val = redact_secrets(raw_path.strip())
        return _cap_path_and_bytes_json(path_val, raw_bytes, MAX_SHELL_RAW_ARGS_BYTES)

    cmd_val: str | None = None
    if parsed is not None:
        raw_cmd = parsed.get("command") or parsed.get("cmd")
        if isinstance(raw_cmd, str):
            cmd_val = redact_secrets(raw_cmd)
    elif tlower in SHELL_TOOL_NAMES:
        cmd_val = redacted_raw

    if tlower in SHELL_TOOL_NAMES or cmd_val is not None:
        cmd_text = cmd_val if cmd_val is not None else redacted_raw
        if "<<" in cmd_text and "\n" in cmd_text:
            first_line = cmd_text.split("\n", 1)[0]
            cmd_text = f"{first_line} # +{raw_bytes}B"
        if tlower in SHELL_TOOL_NAMES:
            return _cap_utf8_bytes(cmd_text, MAX_SHELL_RAW_ARGS_BYTES)
        return _cap_json_field_bytes("command", cmd_text, MAX_SHELL_RAW_ARGS_BYTES)

    return None


def _compact_output_tail_for_store(
    output_tail: str | None,
    is_error: bool | None,
    exit_code: int | None,
) -> str | None:
    """Keep `output_tail` only when `is_error` is true or `exit_code` is non-zero (R15)."""
    if not output_tail:
        return None
    has_error = bool(is_error) or (exit_code is not None and exit_code != 0)
    if not has_error:
        return None
    return _cap_utf8_bytes(redact_secrets(output_tail), MAX_OUTPUT_TAIL_BYTES)


def _compact_fingerprint_for_store(args_fingerprint: str | None) -> str | None:
    """Bound stored `args_fingerprint` size while preserving equality semantics."""
    if not args_fingerprint:
        return None
    clean = redact_secrets(args_fingerprint)
    raw = clean.encode("utf-8")
    if len(raw) <= MAX_FINGERPRINT_BYTES:
        return clean
    return hashlib.sha256(raw).hexdigest()[:16]


def _compact_response_id_for_store(
    source: str, session_id: str, response_id: str | None
) -> str | None:
    """Normalize `ModelCall.response_id` for SQLite `UNIQUE(source, response_id)` (R11/R15)."""
    if not response_id:
        return None
    if source == "dsh" and response_id.startswith(f"{session_id}:"):
        return None
    return redact_secrets(response_id)


def _file_signature(source: str, path: Path) -> tuple[int, int]:
    """Return `(mtime_ns, size_bytes)`, incorporating DSH's companion `.v4` log when present."""
    st = path.stat()
    mtime_ns = st.st_mtime_ns
    size_bytes = st.st_size
    if source == "dsh" and path.name == "session.jsonl.zstd":
        v4 = path.parent / "session.v4.jsonl.zstd"
        if v4.is_file():
            try:
                v4_st = v4.stat()
                mtime_ns = max(mtime_ns, v4_st.st_mtime_ns)
                size_bytes += v4_st.st_size
            except OSError:
                pass
    return mtime_ns, size_bytes


def _trajectory_latest_ms(trajectory: Trajectory) -> int | None:
    latest = trajectory.started_at
    for ev in trajectory.events:
        ts = event_timestamp(ev)
        if isinstance(ev, (ModelCall, ToolCall, Compaction)) and ev.ended_at is not None:
            ts = max(ts or 0, ev.ended_at)
        if ts is not None and (latest is None or ts > latest):
            latest = ts
    return latest


def _trajectory_matches_since(
    trajectory: Trajectory, since_ms: int | None, fallback_mtime_ms: int
) -> bool:
    if since_ms is None:
        return True
    latest = _trajectory_latest_ms(trajectory)
    anchor = latest if latest is not None else fallback_mtime_ms
    return anchor >= since_ms


@dataclass(frozen=True, slots=True)
class TrajectoryIngestStats:
    """Per-trajectory insertion and deduplication counts."""

    session_id: str
    kept: bool
    model_calls_inserted: int = 0
    model_calls_ignored_duplicate: int = 0
    tool_calls_inserted: int = 0
    tool_calls_ignored_duplicate: int = 0


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Summary returned by `MetricsStore.ingest_source`."""

    source: str
    files_discovered: int = 0
    files_ingested: int = 0
    files_skipped_unchanged: int = 0
    files_skipped_since: int = 0
    trajectories_stored: int = 0
    model_calls_inserted: int = 0
    model_calls_ignored_duplicate: int = 0
    tool_calls_inserted: int = 0
    tool_calls_ignored_duplicate: int = 0
    parse_failures: int = 0


@dataclass(frozen=True, slots=True)
class ModelRouteSummary:
    """Call and token breakdown for one `(provider, model)` route."""

    route: str
    call_count: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class SourceSupplementalStats:
    """Supplemental SQLite query statistics rendered alongside `AggregateMetrics`."""

    source: str
    files_ingested: int
    parse_failures: int
    earliest_started_at: int | None
    latest_started_at: int | None
    model_latency_ms_p50: float | None
    model_latency_ms_p90: float | None
    ttft_ms_p50: float | None
    ttft_ms_p90: float | None
    top_models: tuple[ModelRouteSummary, ...]
    tool_kind_counts: dict[CommandKind, int]


class MetricsStore:
    """SQLite-backed store for normalized trajectories and passive intelligence metrics."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        raw = str(db_path)
        if raw == ":memory:":
            self.db_path: Path | None = None
            target = ":memory:"
        else:
            resolved = Path(db_path).expanduser()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = resolved
            target = str(resolved)

        self._tool_set_cache: dict[str, int] = {}
        self.conn = sqlite3.connect(target)
        self.conn.row_factory = sqlite3.Row
        self._configure()

    def _configure(self) -> None:
        existing_tables = {
            r[0]
            for r in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if not existing_tables:
            self.conn.execute("PRAGMA page_size = 32768;")
        elif "tool_calls" in existing_tables:
            tc_cols = {
                r[1]
                for r in self.conn.execute("PRAGMA table_info(tool_calls)").fetchall()
            }
            if "tool_set_id" not in tc_cols or "trajectory_id" not in tc_cols:
                self.conn.execute("PRAGMA foreign_keys = OFF;")
                for tbl in (
                    "tool_calls",
                    "model_calls",
                    "tool_sets",
                    "metrics",
                    "ingest_log",
                    "trajectories",
                ):
                    self.conn.execute(f"DROP TABLE IF EXISTS {tbl};")
                self.conn.commit()
                self.conn.execute("PRAGMA journal_mode = DELETE;")
                self.conn.execute("PRAGMA page_size = 32768;")
                self.conn.execute("VACUUM;")

        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.executescript(_SCHEMA_SQL)
        self.conn.commit()

        for row in self.conn.execute("SELECT id, tools_json FROM tool_sets"):
            self._tool_set_cache[str(row["tools_json"])] = int(row["id"])

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> MetricsStore:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _get_or_create_tool_set_id(
        self, available_tools: tuple[str, ...] | None
    ) -> int | None:
        if available_tools is None:
            return None
        tools_json = json.dumps(
            [redact_secrets(t) for t in available_tools], ensure_ascii=False
        )
        cached = self._tool_set_cache.get(tools_json)
        if cached is not None:
            return cached
        self.conn.execute(
            "INSERT OR IGNORE INTO tool_sets (tools_json) VALUES (?)",
            (tools_json,),
        )
        row = self.conn.execute(
            "SELECT id FROM tool_sets WHERE tools_json = ?",
            (tools_json,),
        ).fetchone()
        ts_id = int(row["id"])
        self._tool_set_cache[tools_json] = ts_id
        return ts_id

    def _delete_session_rows(self, source: str, session_id: str) -> None:
        row = self.conn.execute(
            "SELECT id FROM trajectories WHERE source = ? AND session_id = ?",
            (source, session_id),
        ).fetchone()
        if row is not None:
            tid = int(row["id"])
            self.conn.execute("DELETE FROM model_calls WHERE trajectory_id = ?", (tid,))
            self.conn.execute("DELETE FROM tool_calls WHERE trajectory_id = ?", (tid,))
            self.conn.execute("DELETE FROM trajectories WHERE id = ?", (tid,))
        self.conn.execute(
            "DELETE FROM metrics WHERE source = ? AND session_id = ?",
            (source, session_id),
        )

    def _resolve_session_id(
        self, source: str, session_id: str, origin_file: str | None
    ) -> tuple[str, bool]:
        """Resolve `(source, session_id)` ownership against `origin_file`.

        Returns `(resolved_session_id, should_replace_existing)`.
        """
        row = self.conn.execute(
            "SELECT origin_file FROM trajectories WHERE source = ? AND session_id = ?",
            (source, session_id),
        ).fetchone()
        if row is None:
            return session_id, False

        existing_origin = row["origin_file"]
        if origin_file is None or existing_origin is None or existing_origin == origin_file:
            return session_id, True

        # A distinct file claims the same session_id (e.g. copied backup/fork): disambiguate
        # so the first-seen file keeps its trajectory row and unique response/call IDs.
        stem = redact_secrets(Path(origin_file).stem)
        candidate = f"{session_id}:{stem}" if stem and stem != session_id else ""
        if candidate:
            chk = self.conn.execute(
                "SELECT origin_file FROM trajectories WHERE source = ? AND session_id = ?",
                (source, candidate),
            ).fetchone()
            if chk is None:
                return candidate, False
            if chk["origin_file"] == origin_file:
                return candidate, True

        digest = hashlib.sha256(origin_file.encode("utf-8")).hexdigest()[:8]
        hashed_sid = f"{session_id}:{digest}"
        chk2 = self.conn.execute(
            "SELECT origin_file FROM trajectories WHERE source = ? AND session_id = ?",
            (source, hashed_sid),
        ).fetchone()
        return hashed_sid, (chk2 is not None and chk2["origin_file"] == origin_file)

    def ingest_trajectory(
        self,
        trajectory: Trajectory,
        *,
        origin_file: str | None = None,
        metric_version: int = METRIC_VERSION,
    ) -> TrajectoryIngestStats:
        """Insert one trajectory, ignoring duplicate `(source, response_id)` and `(source, call_id)`."""
        safe_origin = redact_secrets(origin_file) if origin_file is not None else None
        with self.conn:
            return self._ingest_trajectory_in_tx(
                trajectory,
                origin_file=safe_origin,
                metric_version=metric_version,
            )

    def _ingest_trajectory_in_tx(
        self,
        trajectory: Trajectory,
        *,
        origin_file: str | None,
        metric_version: int,
    ) -> TrajectoryIngestStats:
        source = trajectory.source
        resolved_sid, replace_existing = self._resolve_session_id(
            source, trajectory.session_id, origin_file
        )
        if replace_existing:
            self._delete_session_rows(source, resolved_sid)

        safe_cwd = redact_secrets(trajectory.cwd) if trajectory.cwd else None
        now_ms = _now_ms()
        traj_cur = self.conn.execute(
            """
            INSERT INTO trajectories (
                source, session_id, origin_file, cwd, model_routes_json,
                started_at, parse_failures, unknown_event_types_json, ingested_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source,
                resolved_sid,
                origin_file,
                safe_cwd,
                json.dumps(list(trajectory.model_routes), ensure_ascii=False),
                trajectory.started_at,
                trajectory.parse_failures,
                json.dumps(list(trajectory.unknown_event_types), ensure_ascii=False),
                now_ms,
            ),
        )
        traj_id = int(traj_cur.lastrowid)

        mc_inserted = 0
        mc_ignored = 0
        tc_inserted = 0
        tc_ignored = 0
        mc_seq = 0
        tc_seq = 0

        kept_events: list[TrajectoryEvent] = []
        turns_with_skipped_dups: set[int | None] = set()
        turns_with_kept_activity: set[int | None] = set()
        last_skipped_tool_key: tuple[int | None, int | None] | None = None
        kept_substantive_count = 0

        for ev in trajectory.events:
            if isinstance(ev, ModelCall):
                last_skipped_tool_key = None
                resp_id = _compact_response_id_for_store(
                    source, trajectory.session_id, ev.response_id
                )
                cur = self.conn.execute(
                    """
                    INSERT OR IGNORE INTO model_calls (
                        trajectory_id, source, seq_idx, turn, step, response_id,
                        started_at, first_token_at, ended_at,
                        input_tokens, cache_read_tokens, cache_write_tokens,
                        output_tokens, reasoning_tokens, retries, provider, model
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        traj_id,
                        source,
                        mc_seq,
                        ev.turn,
                        ev.step,
                        resp_id,
                        ev.started_at,
                        ev.first_token_at,
                        ev.ended_at,
                        ev.input_tokens,
                        ev.cache_read_tokens,
                        ev.cache_write_tokens,
                        ev.output_tokens,
                        ev.reasoning_tokens,
                        ev.retries,
                        redact_secrets(ev.provider) if ev.provider else None,
                        redact_secrets(ev.model) if ev.model else None,
                    ),
                )
                if cur.rowcount == 1:
                    mc_seq += 1
                    mc_inserted += 1
                    kept_substantive_count += 1
                    kept_events.append(ev)
                    turns_with_kept_activity.add(ev.turn)
                else:
                    mc_ignored += 1
                    turns_with_skipped_dups.add(ev.turn)

            elif isinstance(ev, ToolCall):
                call_id = redact_secrets(ev.call_id) if ev.call_id else None
                tool_set_id = self._get_or_create_tool_set_id(ev.available_tools)
                stored_fp = _compact_fingerprint_for_store(ev.args_fingerprint)
                stored_raw = _compact_raw_args_for_store(ev.name, ev.raw_args)
                stored_tail = _compact_output_tail_for_store(
                    ev.output_tail, ev.is_error, ev.exit_code
                )
                cur = self.conn.execute(
                    """
                    INSERT OR IGNORE INTO tool_calls (
                        trajectory_id, source, seq_idx, turn, step, call_id, name,
                        args_fingerprint, raw_args, started_at, ended_at,
                        is_error, exit_code, output_bytes, truncated,
                        tool_set_id, error_signature, output_tail
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        traj_id,
                        source,
                        tc_seq,
                        ev.turn,
                        ev.step,
                        call_id,
                        redact_secrets(ev.name),
                        stored_fp,
                        stored_raw,
                        ev.started_at,
                        ev.ended_at,
                        _bool_to_sql(ev.is_error),
                        ev.exit_code,
                        ev.output_bytes,
                        _bool_to_sql(ev.truncated),
                        tool_set_id,
                        redact_secrets(ev.error_signature)
                        if ev.error_signature
                        else None,
                        stored_tail,
                    ),
                )
                if cur.rowcount == 1:
                    tc_seq += 1
                    tc_inserted += 1
                    kept_substantive_count += 1
                    last_skipped_tool_key = None
                    kept_events.append(ev)
                    turns_with_kept_activity.add(ev.turn)
                else:
                    tc_ignored += 1
                    last_skipped_tool_key = (ev.turn, ev.step)
                    turns_with_skipped_dups.add(ev.turn)

            elif isinstance(ev, Edit):
                if last_skipped_tool_key is not None and last_skipped_tool_key == (
                    ev.turn,
                    ev.step,
                ):
                    last_skipped_tool_key = None
                    continue
                last_skipped_tool_key = None
                kept_substantive_count += 1
                kept_events.append(ev)
                turns_with_kept_activity.add(ev.turn)

            elif isinstance(ev, Compaction):
                last_skipped_tool_key = None
                kept_substantive_count += 1
                kept_events.append(ev)

            elif isinstance(ev, TurnEnd):
                last_skipped_tool_key = None
                if (
                    ev.turn in turns_with_skipped_dups
                    and ev.turn not in turns_with_kept_activity
                ):
                    continue
                kept_events.append(ev)

            elif isinstance(ev, Outcome):
                last_skipped_tool_key = None
                kept_events.append(ev)

        skipped_total = mc_ignored + tc_ignored
        if skipped_total > 0 and kept_substantive_count == 0:
            self.conn.execute("DELETE FROM trajectories WHERE id = ?", (traj_id,))
            return TrajectoryIngestStats(
                session_id=resolved_sid,
                kept=False,
                model_calls_inserted=0,
                model_calls_ignored_duplicate=mc_ignored,
                tool_calls_inserted=0,
                tool_calls_ignored_duplicate=tc_ignored,
            )

        dedup_traj = replace(
            trajectory,
            session_id=resolved_sid,
            cwd=safe_cwd,
            events=tuple(kept_events),
        )
        tm = compute_trajectory_metrics(dedup_traj)
        if metric_version != METRIC_VERSION:
            tm = replace(tm, metric_version=metric_version)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO metrics (
                source, session_id, metric_version,
                turn_count, model_call_count, tool_call_count, edit_count,
                total_tokens, cache_hit_ratio,
                model_time_ms, tool_time_ms, gap_time_ms,
                metrics_json, computed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source,
                resolved_sid,
                tm.metric_version,
                tm.turn_count,
                tm.model_call_count,
                tm.tool_call_count,
                tm.edit_count,
                tm.total_tokens,
                tm.cache_hit_ratio,
                tm.model_time_ms,
                tm.tool_time_ms,
                tm.gap_time_ms,
                json.dumps(tm.to_dict(), sort_keys=True, ensure_ascii=False),
                now_ms,
            ),
        )

        return TrajectoryIngestStats(
            session_id=resolved_sid,
            kept=True,
            model_calls_inserted=mc_inserted,
            model_calls_ignored_duplicate=mc_ignored,
            tool_calls_inserted=tc_inserted,
            tool_calls_ignored_duplicate=tc_ignored,
        )

    def ingest_source(
        self,
        source: str,
        *,
        root: str | Path | None = None,
        since: str | int | None = None,
        force: bool = False,
    ) -> IngestResult:
        """Incrementally and idempotently ingest all transcripts for `source`."""
        if source not in VALID_SOURCES:
            raise ValueError(
                f"source must be one of {VALID_SOURCES!r}, got {source!r}"
            )
        since_ms = parse_since_ms(since)

        if source == "dsh":
            base_root = Path(root) if root is not None else DEFAULT_DSH_ROOT
            discovered = discover_dsh_files(base_root)
        else:
            base_root = Path(root) if root is not None else DEFAULT_CLAUDE_CODE_ROOT
            discovered = discover_claude_code_files(base_root)

        # Order chronologically by mtime so earlier original transcripts claim shared IDs first.
        ordered_entries: list[tuple[int, int, Path]] = []
        for fpath in discovered:
            try:
                mtime_ns, size_bytes = _file_signature(source, fpath)
            except OSError:
                continue
            ordered_entries.append((mtime_ns, size_bytes, fpath))
        ordered_entries.sort(key=lambda item: (item[0], str(item[2])))

        existing_log = {
            row["file_path"]: row
            for row in self.conn.execute(
                "SELECT file_path, mtime_ns, size_bytes, session_ids_json, metric_version "
                "FROM ingest_log WHERE source = ?",
                (source,),
            ).fetchall()
        }

        files_ingested = 0
        files_skipped_unchanged = 0
        files_skipped_since = 0
        trajectories_stored = 0
        mc_ins = 0
        mc_dup = 0
        tc_ins = 0
        tc_dup = 0
        total_parse_failures = 0

        for mtime_ns, size_bytes, fpath in ordered_entries:
            file_key = redact_secrets(str(fpath.resolve()))
            prev = existing_log.get(file_key)
            if (
                not force
                and prev is not None
                and prev["mtime_ns"] == mtime_ns
                and prev["size_bytes"] == size_bytes
                and prev["metric_version"] == METRIC_VERSION
            ):
                files_skipped_unchanged += 1
                continue

            mtime_ms = max(1, mtime_ns // 1_000_000)
            if since_ms is not None and mtime_ms < since_ms:
                files_skipped_since += 1
                continue

            if source == "dsh":
                parsed_trajs = [parse_dsh_session(fpath)]
            else:
                parsed_trajs = parse_claude_code_file(fpath)

            matching_trajs = [
                t
                for t in parsed_trajs
                if _trajectory_matches_since(t, since_ms, mtime_ms)
            ]
            if since_ms is not None and not matching_trajs:
                files_skipped_since += 1
                continue

            file_parse_failures = sum(t.parse_failures for t in parsed_trajs)
            total_parse_failures += file_parse_failures

            with self.conn:
                if prev is not None:
                    try:
                        old_sids = json.loads(prev["session_ids_json"])
                    except json.JSONDecodeError:
                        old_sids = []
                    for old_sid in old_sids:
                        if isinstance(old_sid, str):
                            self._delete_session_rows(source, old_sid)

                stored_sids: list[str] = []
                for traj in matching_trajs:
                    stats = self._ingest_trajectory_in_tx(
                        traj,
                        origin_file=file_key,
                        metric_version=METRIC_VERSION,
                    )
                    mc_ins += stats.model_calls_inserted
                    mc_dup += stats.model_calls_ignored_duplicate
                    tc_ins += stats.tool_calls_inserted
                    tc_dup += stats.tool_calls_ignored_duplicate
                    if stats.kept:
                        stored_sids.append(stats.session_id)
                        trajectories_stored += 1

                # Record in ingest_log only when not partially filtered by --since
                if since_ms is None or len(matching_trajs) == len(parsed_trajs):
                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO ingest_log (
                            source, file_path, mtime_ns, size_bytes,
                            session_ids_json, parse_failures, metric_version, ingested_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            source,
                            file_key,
                            mtime_ns,
                            size_bytes,
                            json.dumps(stored_sids, ensure_ascii=False),
                            file_parse_failures,
                            METRIC_VERSION,
                            _now_ms(),
                        ),
                    )
            files_ingested += 1

        if self.db_path is not None and files_ingested > 0:
            try:
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            except sqlite3.Error:
                pass

        return IngestResult(
            source=source,
            files_discovered=len(ordered_entries),
            files_ingested=files_ingested,
            files_skipped_unchanged=files_skipped_unchanged,
            files_skipped_since=files_skipped_since,
            trajectories_stored=trajectories_stored,
            model_calls_inserted=mc_ins,
            model_calls_ignored_duplicate=mc_dup,
            tool_calls_inserted=tc_ins,
            tool_calls_ignored_duplicate=tc_dup,
            parse_failures=total_parse_failures,
        )

    def load_trajectory_metrics(
        self,
        source: str | None = None,
        *,
        metric_version: int = METRIC_VERSION,
    ) -> list[TrajectoryMetrics]:
        """Load stored `TrajectoryMetrics` rows ordered by `(source, session_id)`."""
        if source is None:
            rows = self.conn.execute(
                "SELECT metrics_json FROM metrics WHERE metric_version = ? "
                "ORDER BY source, session_id",
                (metric_version,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT metrics_json FROM metrics WHERE source = ? AND metric_version = ? "
                "ORDER BY session_id",
                (source, metric_version),
            ).fetchall()
        return [
            TrajectoryMetrics.from_dict(json.loads(r["metrics_json"])) for r in rows
        ]

    def get_aggregate_metrics(
        self,
        source: str,
        *,
        metric_version: int = METRIC_VERSION,
    ) -> AggregateMetrics:
        """Compute `AggregateMetrics` for `source` from persisted `metrics` rows."""
        items = self.load_trajectory_metrics(source, metric_version=metric_version)
        return aggregate_trajectory_metrics(items, source=source)

    def get_supplemental_stats(self, source: str) -> SourceSupplementalStats:
        """Query latency percentiles, model route totals, tool kinds, and ingest counts."""
        log_row = self.conn.execute(
            "SELECT COUNT(*) AS cnt, COALESCE(SUM(parse_failures), 0) AS pf "
            "FROM ingest_log WHERE source = ?",
            (source,),
        ).fetchone()
        files_cnt = int(log_row["cnt"]) if log_row else 0

        traj_row = self.conn.execute(
            "SELECT COUNT(*) AS cnt, MIN(started_at) AS min_s, MAX(started_at) AS max_s, "
            "COALESCE(SUM(parse_failures), 0) AS pf "
            "FROM trajectories WHERE source = ?",
            (source,),
        ).fetchone()
        if files_cnt == 0 and traj_row:
            files_cnt = int(traj_row["cnt"])
        pf_total = max(
            int(log_row["pf"]) if log_row else 0,
            int(traj_row["pf"]) if traj_row else 0,
        )
        min_s = (
            int(traj_row["min_s"])
            if traj_row and traj_row["min_s"] is not None
            else None
        )
        max_s = (
            int(traj_row["max_s"])
            if traj_row and traj_row["max_s"] is not None
            else None
        )

        latencies: list[int] = []
        ttfts: list[int] = []
        for r in self.conn.execute(
            "SELECT started_at, first_token_at, ended_at FROM model_calls WHERE source = ?",
            (source,),
        ):
            s = r["started_at"]
            f = r["first_token_at"]
            e = r["ended_at"]
            if s is not None and e is not None and e >= s:
                latencies.append(int(e - s))
            if s is not None and f is not None and f >= s:
                ttfts.append(int(f - s))

        route_rows = self.conn.execute(
            """
            SELECT provider, model, COUNT(*) AS calls,
                   COALESCE(SUM(
                       COALESCE(input_tokens, 0) +
                       COALESCE(cache_read_tokens, 0) +
                       COALESCE(cache_write_tokens, 0) +
                       COALESCE(output_tokens, 0)
                   ), 0) AS tok
            FROM model_calls
            WHERE source = ?
            GROUP BY provider, model
            ORDER BY calls DESC, tok DESC
            """,
            (source,),
        ).fetchall()
        top_models: list[ModelRouteSummary] = []
        for r in route_rows:
            prov = r["provider"]
            mod = r["model"]
            if prov and mod:
                label = f"{prov}/{mod}"
            elif mod:
                label = str(mod)
            elif prov:
                label = str(prov)
            else:
                label = "unknown"
            top_models.append(
                ModelRouteSummary(
                    route=label,
                    call_count=int(r["calls"]),
                    total_tokens=int(r["tok"]),
                )
            )

        kind_counter: Counter[CommandKind] = Counter()
        for r in self.conn.execute(
            "SELECT name, raw_args FROM tool_calls WHERE source = ?",
            (source,),
        ):
            kind = classify_tool_call_kind(
                ToolCall(
                    step=None,
                    call_id=None,
                    name=r["name"],
                    raw_args=r["raw_args"],
                )
            )
            kind_counter[kind] += 1

        ordered_kinds: tuple[CommandKind, ...] = (
            "read",
            "search",
            "test",
            "build",
            "lint",
            "vcs",
            "other",
        )
        kind_dict: dict[CommandKind, int] = {
            k: int(kind_counter.get(k, 0)) for k in ordered_kinds
        }

        return SourceSupplementalStats(
            source=source,
            files_ingested=files_cnt,
            parse_failures=pf_total,
            earliest_started_at=min_s,
            latest_started_at=max_s,
            model_latency_ms_p50=_percentile_or_none(latencies, 50.0),
            model_latency_ms_p90=_percentile_or_none(latencies, 90.0),
            ttft_ms_p50=_percentile_or_none(ttfts, 50.0),
            ttft_ms_p90=_percentile_or_none(ttfts, 90.0),
            top_models=tuple(top_models),
            tool_kind_counts=kind_dict,
        )
