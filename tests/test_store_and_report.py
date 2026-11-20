"""Synthetic unit and integration tests for M6 SQLite store, CLI, HTML report, R11, and R15."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3

import pytest
import zstandard as zstd

from dsh_metrics.cli import main
from dsh_metrics.ir import Edit, ModelCall, ToolCall, Trajectory, TurnEnd
from dsh_metrics.report import render_comparison_html
from dsh_metrics.store import (
    MAX_OUTPUT_TAIL_BYTES,
    MAX_SHELL_RAW_ARGS_BYTES,
    MetricsStore,
    parse_since_ms,
)


def _write_zstd_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = ("\n".join(json.dumps(r) for r in records) + "\n").encode("utf-8")
    cctx = zstd.ZstdCompressor()
    path.write_bytes(cctx.compress(payload))


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r) for r in records) + "\n"
    path.write_text(text, encoding="utf-8")


def test_sqlite_wal_schema_and_unique_constraints(tmp_path: Path) -> None:
    db_file = tmp_path / "nested" / "metrics.sqlite"
    with MetricsStore(db_file) as store:
        mode = store.conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert str(mode).lower() == "wal"

        tables = {
            r[0]
            for r in store.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {
            "trajectories",
            "model_calls",
            "tool_sets",
            "tool_calls",
            "metrics",
            "ingest_log",
        }.issubset(tables)

        metrics_cols = {
            r[1] for r in store.conn.execute("PRAGMA table_info(metrics)").fetchall()
        }
        assert "metric_version" in metrics_cols

        tc_cols = {
            r[1] for r in store.conn.execute("PRAGMA table_info(tool_calls)").fetchall()
        }
        assert "tool_set_id" in tc_cols
        assert "available_tools_json" not in tc_cols

        cur1 = store.conn.execute(
            "INSERT INTO trajectories (source, session_id, model_routes_json, unknown_event_types_json, ingested_at) "
            "VALUES ('claude-code', 's1', '[]', '[]', 1)"
        )
        tid1 = int(cur1.lastrowid)
        cur2 = store.conn.execute(
            "INSERT INTO trajectories (source, session_id, model_routes_json, unknown_event_types_json, ingested_at) "
            "VALUES ('claude-code', 's2', '[]', '[]', 1)"
        )
        tid2 = int(cur2.lastrowid)

        # Direct SQLite UNIQUE(source, response_id) and UNIQUE(source, call_id) checks
        store.conn.execute(
            "INSERT INTO model_calls (trajectory_id, source, seq_idx, response_id) "
            "VALUES (?, 'claude-code', 0, 'msg_dup')",
            (tid1,),
        )
        with pytest.raises(sqlite3.IntegrityError):
            store.conn.execute(
                "INSERT INTO model_calls (trajectory_id, source, seq_idx, response_id) "
                "VALUES (?, 'claude-code', 0, 'msg_dup')",
                (tid2,),
            )

        # Multiple NULL response_ids are permitted
        store.conn.execute(
            "INSERT INTO model_calls (trajectory_id, source, seq_idx, response_id) "
            "VALUES (?, 'dsh', 0, NULL)",
            (tid1,),
        )
        store.conn.execute(
            "INSERT INTO model_calls (trajectory_id, source, seq_idx, response_id) "
            "VALUES (?, 'dsh', 1, NULL)",
            (tid2,),
        )

        store.conn.execute(
            "INSERT INTO tool_calls (trajectory_id, source, seq_idx, call_id, name) "
            "VALUES (?, 'claude-code', 0, 'toolu_dup', 'Read')",
            (tid1,),
        )
        with pytest.raises(sqlite3.IntegrityError):
            store.conn.execute(
                "INSERT INTO tool_calls (trajectory_id, source, seq_idx, call_id, name) "
                "VALUES (?, 'claude-code', 0, 'toolu_dup', 'Read')",
                (tid2,),
            )


def test_r11_store_two_synthetic_files_sharing_one_response(tmp_path: Path) -> None:
    """R11 Part 2: SQLite UNIQUE(source, response_id) & UNIQUE(source, call_id) keep first-seen."""
    shared_entries = [
        {
            "type": "user",
            "sessionId": "sess-orig",
            "cwd": "/synthetic/project-alpha",
            "timestamp": "2026-10-10T10:00:00.000Z",
            "message": {"role": "user", "content": "Update module"},
        },
        {
            "type": "assistant",
            "sessionId": "sess-orig",
            "timestamp": "2026-10-10T10:00:02.000Z",
            "message": {
                "id": "msg_shared_r11",
                "model": "claude-opus-5-5",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_shared_r11",
                        "name": "Edit",
                        "input": {
                            "file_path": "src/alpha.py",
                            "old_string": "x = 1",
                            "new_string": "x = 2",
                        },
                    }
                ],
                "usage": {
                    "input_tokens": 100,
                    "cache_read_input_tokens": 400,
                    "output_tokens": 35,
                },
            },
        },
        {
            "type": "user",
            "sessionId": "sess-orig",
            "timestamp": "2026-10-10T10:00:03.000Z",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_shared_r11",
                        "is_error": False,
                        "content": "Updated src/alpha.py",
                    }
                ],
            },
        },
    ]

    resumed_delta = [
        {
            "type": "user",
            "sessionId": "sess-fork",
            "timestamp": "2026-10-10T10:05:00.000Z",
            "message": {"role": "user", "content": "Run tests now"},
        },
        {
            "type": "assistant",
            "sessionId": "sess-fork",
            "timestamp": "2026-10-10T10:05:02.000Z",
            "message": {
                "id": "msg_fork_only",
                "model": "claude-opus-5-5",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_fork_only",
                        "name": "Bash",
                        "input": {"command": "pytest -q"},
                    }
                ],
                "usage": {
                    "input_tokens": 120,
                    "cache_read_input_tokens": 500,
                    "output_tokens": 25,
                },
            },
        },
        {
            "type": "user",
            "sessionId": "sess-fork",
            "timestamp": "2026-10-10T10:05:04.000Z",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_fork_only",
                        "is_error": False,
                        "content": "3 passed in 0.05s",
                    }
                ],
            },
        },
    ]

    cc_root = tmp_path / "cc_projects"
    orig_file = cc_root / "proj-slug" / "sess-orig.jsonl"
    fork_file = cc_root / "proj-slug" / "sess-fork.jsonl"

    # Day 1: ingest only orig_file
    _write_jsonl(orig_file, shared_entries)
    os.utime(orig_file, (1_760_090_000, 1_760_090_000))

    db_path = tmp_path / "r11.sqlite"
    with MetricsStore(db_path) as store:
        r1 = store.ingest_source("claude-code", root=cc_root)
        assert r1.files_ingested == 1
        assert r1.model_calls_inserted == 1
        assert r1.model_calls_ignored_duplicate == 0
        assert r1.tool_calls_inserted == 1

        # Day 2: fork_file appears copying shared_entries
        _write_jsonl(fork_file, shared_entries + resumed_delta)
        os.utime(fork_file, (1_760_090_300, 1_760_090_300))

        # Also add an abandoned pure-copy fork with zero new messages
        pure_copy = cc_root / "proj-slug" / "sess-abandoned-copy.jsonl"
        _write_jsonl(pure_copy, shared_entries)
        os.utime(pure_copy, (1_760_090_400, 1_760_090_400))

        r2 = store.ingest_source("claude-code", root=cc_root)
        assert r2.files_skipped_unchanged == 1
        assert r2.files_ingested == 2
        assert r2.trajectories_stored == 1
        assert r2.model_calls_inserted == 1
        assert r2.model_calls_ignored_duplicate == 2
        assert r2.tool_calls_inserted == 1
        assert r2.tool_calls_ignored_duplicate == 2

        mc_rows = store.conn.execute(
            "SELECT t.session_id, mc.response_id, mc.output_tokens "
            "FROM model_calls mc JOIN trajectories t ON t.id = mc.trajectory_id "
            "ORDER BY mc.id"
        ).fetchall()
        assert [
            (r["session_id"], r["response_id"], r["output_tokens"]) for r in mc_rows
        ] == [
            ("sess-orig", "msg_shared_r11", 35),
            ("sess-fork", "msg_fork_only", 25),
        ]

        tc_rows = store.conn.execute(
            "SELECT t.session_id, tc.call_id, tc.name "
            "FROM tool_calls tc JOIN trajectories t ON t.id = tc.trajectory_id "
            "ORDER BY tc.id"
        ).fetchall()
        assert [(r["session_id"], r["call_id"], r["name"]) for r in tc_rows] == [
            ("sess-orig", "toolu_shared_r11", "Edit"),
            ("sess-fork", "toolu_fork_only", "Bash"),
        ]

        agg = store.get_aggregate_metrics("claude-code")
        assert agg.trajectory_count == 2
        assert agg.model_call_count == 2
        assert agg.tool_call_count == 2
        assert agg.edit_count == 1
        assert agg.output_tokens_total == 60
        assert agg.input_tokens_total == 220
        assert agg.cache_read_tokens_total == 900

        # Re-running ingest with --force is 100% idempotent
        r3 = store.ingest_source("claude-code", root=cc_root, force=True)
        assert r3.files_ingested == 3
        agg_after = store.get_aggregate_metrics("claude-code")
        assert agg_after == agg


def test_r15_store_size_caps_tool_sets_raw_args_and_error_only_output_tail(
    tmp_path: Path,
) -> None:
    """R15: tool_sets dedup, raw_args <= 1KB shell-only & path+bytes for write/edit/read, error-only output_tail."""
    shared_tools = ("bash", "read", "edit", "write", "grep", "run_code")
    file_body_marker = "SYNTHETIC_LARGE_FILE_BODY_LINE_" * 600  # ~18.6 KB
    long_shell_cmd = "pytest -k " + ("very_long_selector_" * 120)  # > 2.2 KB
    heredoc_cmd = f"python3 - <<'EOF'\n{file_body_marker}\nEOF"
    ok_output_marker = "PASSING_STDOUT_SHOULD_NOT_BE_STORED" * 10
    err_output_marker = "AssertionError: expected green\n" + ("stack frame\n" * 200)
    nonzero_exit_marker = "compile failure [exit code: 2]"

    write_raw = json.dumps(
        {"file_path": "src/big_module.py", "content": file_body_marker}
    )
    edit_raw = json.dumps(
        {
            "file_path": "src/big_module.py",
            "old_string": file_body_marker[:4000],
            "new_string": file_body_marker[:4000] + "\n# patched\n",
        }
    )
    read_raw = json.dumps(
        {"file_path": "src/big_module.py", "offset": 1, "limit": 200}
    )
    grep_raw = json.dumps({"pattern": "def main", "path": "src/"})
    run_code_raw = json.dumps({"code": file_body_marker})

    traj = Trajectory(
        source="dsh",
        session_id="r15-synth",
        started_at=1_760_500_000_000,
        events=(
            ModelCall(
                step=1,
                turn=1,
                started_at=1_760_500_000_100,
                ended_at=1_760_500_000_400,
                input_tokens=200,
                output_tokens=50,
                response_id="r15-synth:1:1",
            ),
            # 1. Long single-line shell command (succeeding: output_tail must be dropped)
            ToolCall(
                step=1,
                turn=1,
                call_id="tc_bash_long",
                name="bash",
                args_fingerprint="pytest -k <STR>",
                raw_args=json.dumps(
                    {"command": long_shell_cmd, "description": "run long pytest"}
                ),
                started_at=1_760_500_000_410,
                ended_at=1_760_500_000_600,
                is_error=False,
                exit_code=0,
                output_bytes=400,
                available_tools=shared_tools,
                output_tail=ok_output_marker,
            ),
            # 2. Heredoc shell command (non-zero exit_code even with is_error=False: keep output_tail)
            ToolCall(
                step=2,
                turn=1,
                call_id="tc_bash_heredoc",
                name="bash",
                args_fingerprint="cat <<<STR> > /tmp/out.txt",
                raw_args=json.dumps({"command": heredoc_cmd}),
                started_at=1_760_500_000_610,
                ended_at=1_760_500_000_700,
                is_error=False,
                exit_code=2,
                output_bytes=120,
                available_tools=shared_tools,
                output_tail=nonzero_exit_marker,
            ),
            # 3. Write call with huge body (succeeding)
            ToolCall(
                step=3,
                turn=1,
                call_id="tc_write",
                name="write",
                args_fingerprint="write:src/big_module.py",
                raw_args=write_raw,
                started_at=1_760_500_000_710,
                ended_at=1_760_500_000_750,
                is_error=False,
                output_bytes=20,
                available_tools=shared_tools,
                output_tail=ok_output_marker,
            ),
            Edit(
                step=3,
                turn=1,
                path="src/big_module.py",
                before_hash=None,
                after_hash="abcd1234",
                at=1_760_500_000_750,
            ),
            # 4. Edit call with huge body (is_error=True: keep output_tail <= 2048B)
            ToolCall(
                step=4,
                turn=1,
                call_id="tc_edit_err",
                name="edit",
                args_fingerprint="edit:src/big_module.py",
                raw_args=edit_raw,
                started_at=1_760_500_000_760,
                ended_at=1_760_500_000_800,
                is_error=True,
                output_bytes=3000,
                available_tools=shared_tools,
                error_signature="test:AssertionError",
                output_tail=err_output_marker[:2000],
            ),
            # 5. Read call
            ToolCall(
                step=5,
                turn=1,
                call_id="tc_read",
                name="read",
                args_fingerprint="read:src/big_module.py",
                raw_args=read_raw,
                started_at=1_760_500_000_810,
                ended_at=1_760_500_000_850,
                is_error=False,
                output_bytes=1500,
                available_tools=shared_tools,
                output_tail=ok_output_marker,
            ),
            # 6. Non-shell, non-write/edit/read tools -> raw_args must be NULL
            ToolCall(
                step=6,
                turn=1,
                call_id="tc_grep",
                name="grep",
                args_fingerprint="grep:<STR>:src/",
                raw_args=grep_raw,
                started_at=1_760_500_000_860,
                ended_at=1_760_500_000_900,
                is_error=False,
                available_tools=shared_tools,
                output_tail=ok_output_marker,
            ),
            ToolCall(
                step=7,
                turn=1,
                call_id="tc_run_code",
                name="run_code",
                args_fingerprint="run_code:" + ("z" * 500),
                raw_args=run_code_raw,
                started_at=1_760_500_000_910,
                ended_at=1_760_500_000_950,
                is_error=False,
                available_tools=shared_tools,
                output_tail=ok_output_marker,
            ),
            TurnEnd(turn=1, reason="completed", at=1_760_500_001_000),
        ),
    )

    db_path = tmp_path / "r15.sqlite"
    with MetricsStore(db_path) as store:
        store.ingest_trajectory(traj)

        # 1. Distinct tool list stored once in tool_sets
        ts_rows = store.conn.execute("SELECT id, tools_json FROM tool_sets").fetchall()
        assert len(ts_rows) == 1
        assert json.loads(ts_rows[0]["tools_json"]) == list(shared_tools)

        rows = {
            r["call_id"]: r
            for r in store.conn.execute(
                "SELECT call_id, raw_args, tool_set_id, output_tail, args_fingerprint FROM tool_calls"
            ).fetchall()
        }
        assert len(rows) == 7
        for r in rows.values():
            assert r["tool_set_id"] == ts_rows[0]["id"]
            if r["raw_args"] is not None:
                assert len(r["raw_args"].encode("utf-8")) <= MAX_SHELL_RAW_ARGS_BYTES
            if r["output_tail"] is not None:
                assert len(r["output_tail"].encode("utf-8")) <= MAX_OUTPUT_TAIL_BYTES

        # 2. Shell command capped at 1024 UTF-8 bytes; heredoc body omitted
        assert len(rows["tc_bash_long"]["raw_args"].encode("utf-8")) == 1024
        assert rows["tc_bash_long"]["raw_args"].startswith("pytest -k ")
        assert "SYNTHETIC_LARGE_FILE_BODY_LINE_" not in rows["tc_bash_heredoc"]["raw_args"]

        # 3. Write / edit / read store only path + byte length
        for cid, orig_raw in (
            ("tc_write", write_raw),
            ("tc_edit_err", edit_raw),
            ("tc_read", read_raw),
        ):
            meta = json.loads(rows[cid]["raw_args"])
            assert set(meta.keys()) == {"path", "bytes"}
            assert meta["path"] == "src/big_module.py"
            assert meta["bytes"] == len(orig_raw.encode("utf-8"))

        # 4. Other tools drop raw_args
        assert rows["tc_grep"]["raw_args"] is None
        assert rows["tc_run_code"]["raw_args"] is None
        assert len(rows["tc_run_code"]["args_fingerprint"]) == 16

        # 5. output_tail stored ONLY for is_error=True or exit_code != 0
        assert rows["tc_bash_long"]["output_tail"] is None
        assert rows["tc_write"]["output_tail"] is None
        assert rows["tc_read"]["output_tail"] is None
        assert rows["tc_grep"]["output_tail"] is None
        assert rows["tc_run_code"]["output_tail"] is None
        assert rows["tc_bash_heredoc"]["output_tail"] == nonzero_exit_marker
        assert "AssertionError" in rows["tc_edit_err"]["output_tail"]

        # Whole file body and passing stdout never appear anywhere in SQLite
        dump_text = "\n".join(store.conn.iterdump())
        assert "SYNTHETIC_LARGE_FILE_BODY_LINE_" not in dump_text
        assert "PASSING_STDOUT_SHOULD_NOT_BE_STORED" not in dump_text

        # Supplemental command-kind classification still works accurately on compacted rows
        sup = store.get_supplemental_stats("dsh")
        assert sup.tool_kind_counts["test"] == 1  # tc_bash_long (pytest)
        assert sup.tool_kind_counts["read"] == 1  # tc_read
        assert sup.tool_kind_counts["search"] == 1  # tc_grep


def test_incremental_ingest_detects_dsh_v4_companion_growth(tmp_path: Path) -> None:
    dsh_root = tmp_path / "dsh_sessions"
    sess_dir = dsh_root / "proj-slug" / "session-dsh-inc-1"
    legacy_path = sess_dir / "session.jsonl.zstd"
    v4_path = sess_dir / "session.v4.jsonl.zstd"

    base_events = [
        {
            "type": "session",
            "id": "dsh-inc-1",
            "createdAt": 1_760_100_000_000,
            "cwd": "/synthetic/dsh-proj",
        },
        {"type": "turn/start", "seq": 1, "time": 1_760_100_000_100, "data": {"turn": 1}},
        {
            "type": "request/header",
            "seq": 2,
            "time": 1_760_100_000_150,
            "data": {
                "header": {
                    "config": {"provider": "deepseek", "model": "deepseek-chat"},
                    "tools": [{"name": "bash"}, {"name": "read"}],
                }
            },
        },
        {
            "type": "step/start",
            "seq": 3,
            "time": 1_760_100_000_200,
            "data": {"turn": 1, "step": 1},
        },
        {
            "type": "text-chunks",
            "seq0": 4,
            "time0": 1_760_100_000_400,
            "data": {"chunks": ["Checking..."]},
        },
        {
            "type": "assistant/message",
            "seq": 5,
            "time": 1_760_100_001_000,
            "data": {
                "turn": 1,
                "step": 1,
                "message": {"content": [{"type": "text", "text": "Checking..."}]},
                "usage": {
                    "inputTokens": 200,
                    "cacheReadTokens": 800,
                    "outputTokens": 30,
                    "reasoningTokens": 10,
                },
            },
        },
        {
            "type": "turn/end",
            "seq": 6,
            "time": 1_760_100_001_100,
            "data": {"turn": 1, "reason": {"kind": "completed"}},
        },
    ]
    _write_zstd_jsonl(legacy_path, base_events)

    db_path = tmp_path / "dsh.sqlite"
    with MetricsStore(db_path) as store:
        res1 = store.ingest_source("dsh", root=dsh_root)
        assert res1.files_ingested == 1
        assert store.get_aggregate_metrics("dsh").model_call_count == 1

        # Second run with no changes skips decompression completely
        res2 = store.ingest_source("dsh", root=dsh_root)
        assert res2.files_ingested == 0
        assert res2.files_skipped_unchanged == 1

        # Companion v4 file appears with a new post-migration turn
        v4_events = base_events + [
            {
                "type": "turn/start",
                "seq": 7,
                "time": 1_760_100_010_000,
                "data": {"turn": 2},
            },
            {
                "type": "step/start",
                "seq": 8,
                "time": 1_760_100_010_100,
                "data": {"turn": 2, "step": 1},
            },
            {
                "type": "assistant/message",
                "seq": 9,
                "time": 1_760_100_011_000,
                "data": {
                    "turn": 2,
                    "step": 1,
                    "message": {"content": [{"type": "text", "text": "Done"}]},
                    "usage": {"inputTokens": 250, "outputTokens": 45},
                },
            },
            {
                "type": "turn/end",
                "seq": 10,
                "time": 1_760_100_011_100,
                "data": {"turn": 2, "reason": {"kind": "completed"}},
            },
        ]
        _write_zstd_jsonl(v4_path, v4_events)

        res3 = store.ingest_source("dsh", root=dsh_root)
        assert res3.files_ingested == 1
        agg = store.get_aggregate_metrics("dsh")
        assert agg.trajectory_count == 1
        assert agg.turn_count == 2
        assert agg.model_call_count == 2
        assert agg.output_tokens_total == 75


def test_since_filter_and_subsequent_backfill(tmp_path: Path) -> None:
    assert parse_since_ms("2026-10-15") == 1_792_022_400_000
    with pytest.raises(ValueError, match="Invalid --since"):
        parse_since_ms("not-a-date")

    cc_root = tmp_path / "cc_since"
    old_file = cc_root / "slug" / "s-old.jsonl"
    new_file = cc_root / "slug" / "s-new.jsonl"

    _write_jsonl(
        old_file,
        [
            {
                "type": "assistant",
                "sessionId": "s-old",
                "timestamp": "2026-10-01T12:00:00.000Z",
                "message": {
                    "id": "msg_old",
                    "model": "claude-sonnet-5-5",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Old"}],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            }
        ],
    )
    _write_jsonl(
        new_file,
        [
            {
                "type": "assistant",
                "sessionId": "s-new",
                "timestamp": "2026-10-20T12:00:00.000Z",
                "message": {
                    "id": "msg_new",
                    "model": "claude-sonnet-5-5",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "New"}],
                    "usage": {"input_tokens": 20, "output_tokens": 8},
                },
            }
        ],
    )

    db_path = tmp_path / "since.sqlite"
    with MetricsStore(db_path) as store:
        r_filtered = store.ingest_source(
            "claude-code", root=cc_root, since="2026-10-15"
        )
        assert r_filtered.files_ingested == 1
        assert r_filtered.files_skipped_since == 1
        assert store.get_aggregate_metrics("claude-code").trajectory_count == 1

        # Full ingest without --since skips s-new (already logged) and backfills s-old
        r_full = store.ingest_source("claude-code", root=cc_root)
        assert r_full.files_skipped_unchanged == 1
        assert r_full.files_ingested == 1
        assert store.get_aggregate_metrics("claude-code").trajectory_count == 2


def test_cli_end_to_end_and_self_contained_html_report(tmp_path: Path) -> None:
    db_path = tmp_path / "cli.sqlite"
    empty_out = tmp_path / "empty_report.html"

    # Empty DB report renders cleanly with N/A placeholders and zero crashes
    assert main(["report", "--db", str(db_path), "--out", str(empty_out)]) == 0
    empty_html = empty_out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in empty_html
    assert "N/A" in empty_html

    # Populate synthetic DSH and Claude Code roots, including a secret to verify redaction
    secret_tok = "sk-ant-api03-SyntheticSecretValueForReportCheck999"
    dsh_root = tmp_path / "dsh"
    _write_zstd_jsonl(
        dsh_root / "slug" / "session-synth-d1" / "session.jsonl.zstd",
        [
            {
                "type": "session",
                "id": "synth-d1",
                "createdAt": 1_760_200_000_000,
                "cwd": "/private/secret-repo-name",
            },
            {
                "type": "turn/start",
                "seq": 1,
                "time": 1_760_200_000_100,
                "data": {"turn": 1},
            },
            {
                "type": "step/start",
                "seq": 2,
                "time": 1_760_200_000_200,
                "data": {"turn": 1, "step": 1},
            },
            {
                "type": "assistant/message",
                "seq": 3,
                "time": 1_760_200_000_800,
                "data": {
                    "turn": 1,
                    "step": 1,
                    "message": {"content": []},
                    "usage": {"inputTokens": 500, "outputTokens": 60},
                },
            },
            {
                "type": "tool/call",
                "seq": 4,
                "time": 1_760_200_000_850,
                "data": {
                    "turn": 1,
                    "step": 1,
                    "callId": "call_synth_1",
                    "name": "bash",
                    "arguments": json.dumps(
                        {"command": f"echo {secret_tok} && pytest"}
                    ),
                },
            },
            {
                "type": "tool/result",
                "seq": 5,
                "time": 1_760_200_001_200,
                "data": {
                    "turn": 1,
                    "step": 1,
                    "message": {
                        "content": [
                            {
                                "type": "tool-result",
                                "toolCallId": "call_synth_1",
                                "isError": False,
                                "content": [{"type": "text", "text": "1 passed"}],
                            }
                        ]
                    },
                },
            },
            {
                "type": "turn/end",
                "seq": 6,
                "time": 1_760_200_001_300,
                "data": {"turn": 1, "reason": {"kind": "completed"}},
            },
        ],
    )

    cc_root = tmp_path / "cc"
    _write_jsonl(
        cc_root / "slug" / "synth-c1.jsonl",
        [
            {
                "type": "user",
                "sessionId": "synth-c1",
                "cwd": "/private/secret-repo-name",
                "timestamp": "2026-10-12T08:00:00.000Z",
                "message": {"role": "user", "content": "Run check"},
            },
            {
                "type": "assistant",
                "sessionId": "synth-c1",
                "timestamp": "2026-10-12T08:00:02.000Z",
                "message": {
                    "id": "msg_cli_1",
                    "model": "claude-opus-5-5",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "All good"}],
                    "usage": {
                        "input_tokens": 300,
                        "cache_read_input_tokens": 1200,
                        "output_tokens": 40,
                    },
                },
            },
        ],
    )

    assert (
        main(
            [
                "--db",
                str(db_path),
                "ingest",
                "--source",
                "dsh",
                "--root",
                str(dsh_root),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "ingest",
                "--db",
                str(db_path),
                "--source",
                "claude-code",
                "--root",
                str(cc_root),
            ]
        )
        == 0
    )
    # Invalid --since returns exit code 2
    assert (
        main(
            [
                "ingest",
                "--db",
                str(db_path),
                "--source",
                "dsh",
                "--root",
                str(dsh_root),
                "--since",
                "bad-date",
            ]
        )
        == 2
    )

    report_path = tmp_path / "out" / "comparison.html"
    assert main(["--db", str(db_path), "report", "--out", str(report_path)]) == 0
    html_text = report_path.read_text(encoding="utf-8")
    lower_html = html_text.lower()

    # Required self-contained banner & sections
    assert "passive data" in lower_html
    assert "tasks differ" in lower_html
    assert "not a fair comparison" in lower_html
    assert "http://" not in lower_html
    assert "https://" not in lower_html
    assert "Category A" in html_text
    assert "Category B" in html_text
    assert "Category C" in html_text
    assert "Category D" in html_text
    assert "claude-opus-5-5" in html_text

    # No secrets, cwd paths, or session IDs leak into store or report
    assert secret_tok not in html_text
    assert "/private/secret-repo-name" not in html_text
    assert "synth-d1" not in html_text
    assert "synth-c1" not in html_text

    with MetricsStore(db_path) as store:
        raw_dump = "\n".join(store.conn.iterdump())
        assert secret_tok not in raw_dump


def test_direct_trajectory_ingest_idempotency() -> None:
    t = Trajectory(
        source="dsh",
        session_id="mem-1",
        started_at=1_760_000_000_000,
        events=(
            ModelCall(
                step=1,
                turn=1,
                started_at=1_760_000_000_100,
                ended_at=1_760_000_000_500,
                input_tokens=100,
                output_tokens=20,
                response_id="mem-1:1:1",
            ),
            ToolCall(
                step=1,
                turn=1,
                call_id="tc_mem_1",
                name="read",
                raw_args='{"path": "a.py"}',
                started_at=1_760_000_000_550,
                ended_at=1_760_000_000_600,
                is_error=False,
            ),
            TurnEnd(turn=1, reason="completed", at=1_760_000_000_700),
        ),
    )
    with MetricsStore(":memory:") as store:
        s1 = store.ingest_trajectory(t)
        s2 = store.ingest_trajectory(t)
        assert s1 == s2
        assert store.get_aggregate_metrics("dsh").model_call_count == 1
        assert "<!DOCTYPE html>" in render_comparison_html(store)
