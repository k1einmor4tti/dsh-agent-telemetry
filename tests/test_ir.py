"""Tests for M1 package skeleton and trajectory IR."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import subprocess
import sys

import pytest

from dsh_metrics import (
    Compaction,
    Edit,
    ModelCall,
    Outcome,
    ToolCall,
    Trajectory,
    TurnEnd,
    __version__,
    event_timestamp,
)
from dsh_metrics.cli import main


def test_dataclass_fields_include_spec_and_review_extensions() -> None:
    assert [f.name for f in fields(Trajectory)] == [
        "source",
        "session_id",
        "cwd",
        "model_routes",
        "started_at",
        "events",
        "parse_failures",
        "unknown_event_types",
    ]
    assert [f.name for f in fields(ModelCall)] == [
        "step",
        "started_at",
        "first_token_at",
        "ended_at",
        "input_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "output_tokens",
        "reasoning_tokens",
        "retries",
        "provider",
        "model",
        "turn",
        "response_id",
    ]
    assert [f.name for f in fields(ToolCall)] == [
        "step",
        "call_id",
        "name",
        "args_fingerprint",
        "raw_args",
        "started_at",
        "ended_at",
        "is_error",
        "exit_code",
        "output_bytes",
        "truncated",
        "available_tools",
        "turn",
        "error_signature",
        "output_tail",
    ]
    assert [f.name for f in fields(Edit)] == [
        "step",
        "path",
        "before_hash",
        "after_hash",
        "turn",
        "at",
    ]
    assert [f.name for f in fields(Compaction)] == [
        "started_at",
        "ended_at",
        "ok",
        "tokens_before",
        "tokens_after",
        "reason",
    ]
    assert [f.name for f in fields(TurnEnd)] == ["turn", "reason", "at"]
    assert [f.name for f in fields(Outcome)] == ["task_id", "result", "source", "at"]


def test_ir_immutability_and_hashability() -> None:
    mc = ModelCall(
        1,
        1_700_000_000_000,
        1_700_000_000_250,
        1_700_000_001_000,
        1200,
        800,
        0,
        150,
        40,
        0,
        "deepseek",
        "deepseek-chat",
        turn=1,
    )
    tc = ToolCall(
        1,
        "call_1",
        "bash",
        "pytest",
        {"command": "pytest"},
        1_700_000_001_100,
        1_700_000_002_000,
        False,
        0,
        512,
        False,
        ["bash", "read", "edit"],
        turn=1,
    )
    ed = Edit(2, "src/app.py", "abc", "def", turn=1, at=1_700_000_002_500)
    comp = Compaction(
        1_700_000_003_000, 1_700_000_004_000, True, 50000, 12000, "auto"
    )
    te = TurnEnd(1, "completed", at=1_700_000_004_500)
    out = Outcome("task-1", "pass", "eval", at=1_700_000_005_000)
    traj = Trajectory(
        "dsh",
        "sess-001",
        "/workspace/demo",
        ["deepseek/deepseek-chat"],
        1_700_000_000_000,
        [mc, tc, ed, comp, te, out],
        parse_failures=2,
        unknown_event_types=["custom/event"],
    )

    assert isinstance(traj.model_routes, tuple)
    assert isinstance(traj.events, tuple)
    assert isinstance(traj.unknown_event_types, tuple)
    assert traj.parse_failures == 2
    assert traj.unknown_event_types == ("custom/event",)
    assert isinstance(tc.available_tools, tuple)
    assert tc.raw_args == '{"command": "pytest"}'
    assert event_timestamp(mc) == 1_700_000_000_000
    assert event_timestamp(ed) == 1_700_000_002_500
    assert event_timestamp(te) == 1_700_000_004_500
    assert event_timestamp(out) == 1_700_000_005_000
    assert hash(traj) is not None

    with pytest.raises(FrozenInstanceError):
        traj.cwd = "/other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        mc.input_tokens = 999  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        tc.is_error = True  # type: ignore[misc]


def test_unknown_values_default_to_none_and_zero_timestamps_rejected() -> None:
    mc = ModelCall(step=None)
    assert mc.turn is None
    assert mc.started_at is None
    assert mc.first_token_at is None
    assert mc.ended_at is None
    assert mc.input_tokens is None
    assert mc.cache_read_tokens is None
    assert mc.output_tokens is None
    assert mc.retries is None

    tc = ToolCall(step=1, call_id=None, name="read")
    assert tc.turn is None
    assert tc.available_tools is None
    assert tc.exit_code is None
    assert tc.output_bytes is None
    assert tc.truncated is None

    ed = Edit(step=1, path="a.py")
    assert ed.turn is None
    assert ed.at is None

    traj = Trajectory(source="dsh", session_id="s1")
    assert traj.parse_failures == 0
    assert traj.unknown_event_types == ()

    with pytest.raises(ValueError, match="positive epoch milliseconds"):
        ModelCall(step=1, started_at=0)
    with pytest.raises(ValueError, match="positive epoch milliseconds"):
        ModelCall(step=1, first_token_at=-1)
    with pytest.raises(ValueError, match="positive epoch milliseconds"):
        ToolCall(step=1, call_id="c1", name="bash", ended_at=0)
    with pytest.raises(ValueError, match="positive epoch milliseconds"):
        Edit(step=1, path="a.py", at=0)
    with pytest.raises(ValueError, match="positive epoch milliseconds"):
        TurnEnd(turn=1, reason="completed", at=0)
    with pytest.raises(ValueError, match="positive epoch milliseconds"):
        Outcome(task_id="t1", result="ok", at=-5)
    with pytest.raises(ValueError, match="positive epoch milliseconds"):
        Compaction(started_at=0)
    with pytest.raises(ValueError, match="positive epoch milliseconds"):
        Trajectory(source="dsh", session_id="s1", started_at=0)
    with pytest.raises(ValueError, match="positive epoch milliseconds"):
        Trajectory(source="dsh", session_id="s1", started_at=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="parse_failures"):
        Trajectory(source="dsh", session_id="s1", parse_failures=-1)
    with pytest.raises(ValueError, match="2048 UTF-8 bytes"):
        ToolCall(step=1, call_id="c1", name="bash", output_tail="a" * 2049)


def test_cli_entrypoint() -> None:
    assert main([]) == 0
    proc = subprocess.run(
        [sys.executable, "-m", "dsh_metrics.cli", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert __version__ in proc.stdout
