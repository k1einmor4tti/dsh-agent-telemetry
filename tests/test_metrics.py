"""Synthetic tests for M5 trajectory & corpus metrics and R7 compaction separation."""

from __future__ import annotations

import pytest

from dsh_metrics.ir import Compaction, Edit, ModelCall, ToolCall, Trajectory, TurnEnd
from dsh_metrics.metrics import (
    METRIC_VERSION,
    TrajectoryMetrics,
    aggregate_trajectory_metrics,
    compute_trajectory_metrics,
)


def test_category_a_and_b_control_loop_and_coding_metrics() -> None:
    tools = ("bash", "read", "edit")
    traj = Trajectory(
        source="dsh",
        session_id="synth-ab",
        cwd="/workspace/proj",
        events=(
            # Turn 1, Step 1: read 2 files + unknown tool + arg validation error
            ModelCall(step=1, turn=1, input_tokens=100, output_tokens=20),
            ToolCall(
                step=1,
                turn=1,
                call_id="c1",
                name="read",
                raw_args={"file_path": "/workspace/proj/src/calc.py"},
                is_error=False,
                truncated=False,
                available_tools=tools,
            ),
            ToolCall(
                step=1,
                turn=1,
                call_id="c2",
                name="bash",
                raw_args={"command": "sed -n 1,20p src/unused.py"},
                is_error=False,
                exit_code=0,
                truncated=False,
                available_tools=tools,
            ),
            ToolCall(
                step=1,
                turn=1,
                call_id="c3",
                name="bash",
                raw_args={"command": "sed -n '21,40p' src/unused.py"},
                is_error=False,
                exit_code=0,
                truncated=False,
                available_tools=tools,
            ),
            ToolCall(
                step=1,
                turn=1,
                call_id="c_unk",
                name="magic_tool",
                raw_args={"x": 1},
                is_error=True,
                error_signature="error:unknown tool",
                output_tail="Unknown tool: magic_tool",
                available_tools=tools,
            ),
            ToolCall(
                step=1,
                turn=1,
                call_id="c_arg",
                name="edit",
                raw_args={"path": "src/calc.py"},
                is_error=True,
                error_signature="error:InputValidationError",
                output_tail="InputValidationError: missing required parameter 'new_string'",
                available_tools=tools,
            ),
            # Turn 1, Step 2: failing pytest #1 (episode 1 starts at step 2)
            ModelCall(step=2, turn=1, input_tokens=120, output_tokens=30),
            ToolCall(
                step=2,
                turn=1,
                call_id="c4",
                name="bash",
                raw_args={"command": "pytest -q"},
                is_error=True,
                exit_code=1,
                error_signature="test:AssertionError",
                output_tail="AssertionError: 1 != 2",
                available_tools=tools,
            ),
            # Turn 1, Step 3: repeated failing pytest #2 (same fp + same error_signature!)
            ModelCall(step=3, turn=1, input_tokens=130, output_tokens=25),
            ToolCall(
                step=3,
                turn=1,
                call_id="c5",
                name="bash",
                raw_args={"command": "pytest -q"},
                is_error=True,
                exit_code=1,
                error_signature="test:AssertionError",
                output_tail="AssertionError: 1 != 2",
                available_tools=tools,
            ),
            TurnEnd(turn=1, reason="completed"),
            # Turn 2, Step 1 (monotonic step 4): edit calc.py -> escapes repeated failure (4 - 2 = 2 steps)
            ModelCall(step=1, turn=2, input_tokens=140, output_tokens=40),
            ToolCall(
                step=1,
                turn=2,
                call_id="c6",
                name="edit",
                raw_args={"file_path": "src/calc.py", "old_string": "a", "new_string": "b"},
                is_error=False,
                available_tools=tools,
            ),
            Edit(step=1, turn=2, path="src/calc.py"),
            # Turn 2, Step 2 (monotonic step 5): compile error after edit #1, then edit #2
            ModelCall(step=2, turn=2, input_tokens=150, output_tokens=35),
            ToolCall(
                step=2,
                turn=2,
                call_id="c7",
                name="bash",
                raw_args={"command": "python3 -m py_compile src/calc.py"},
                is_error=True,
                exit_code=1,
                error_signature="compile:SyntaxError",
                output_tail="SyntaxError: invalid syntax",
                available_tools=tools,
            ),
            ToolCall(
                step=2,
                turn=2,
                call_id="c8",
                name="edit",
                raw_args={"file_path": "src/calc.py", "old_string": "b", "new_string": "c"},
                is_error=False,
                available_tools=tools,
            ),
            Edit(step=2, turn=2, path="src/calc.py"),
            # Turn 2, Step 3 (monotonic step 6): green pytest! (6 - 2 = 4 steps <= 10, after last edit!)
            ModelCall(step=3, turn=2, input_tokens=160, output_tokens=20),
            ToolCall(
                step=3,
                turn=2,
                call_id="c9",
                name="bash",
                raw_args={"command": "pytest -q"},
                is_error=False,
                exit_code=0,
                available_tools=tools,
            ),
            TurnEnd(turn=2, reason="completed"),
        ),
    )

    m = compute_trajectory_metrics(traj)
    assert m.metric_version == METRIC_VERSION
    assert m.turn_count == 2
    assert m.step_count == 6
    assert m.model_calls_per_turn == pytest.approx(3.0)
    assert m.tool_calls_per_turn == pytest.approx(5.5)

    # Shell commands: sed#1, sed#2 (dup), pytest#1, pytest#2 (dup), py_compile, pytest#3 (dup) -> 3/6 = 0.5
    assert m.command_count == 6
    assert m.near_duplicate_command_count == 3
    assert m.near_duplicate_command_rate == pytest.approx(0.5)

    # Repeated failure: (pytest -q, test:AssertionError) at steps 2 & 3, escaped at step 4 -> latency 2
    assert m.repeated_failure_groups == 1
    assert m.repeated_failure_calls == 2
    assert m.escape_latencies == (2,)
    assert m.escape_latency_steps_mean == pytest.approx(2.0)

    # Unknown tool & arg validation errors (subprocess exit_code=1 is NOT arg validation!)
    assert m.unknown_tool_call_count == 1
    assert m.unknown_tool_call_rate == pytest.approx(1 / 11)
    assert m.arg_validation_error_count == 1
    assert m.arg_validation_error_rate == pytest.approx(1 / 11)

    # Category B: 1 compile error across 2 edits = 50 per 100 edits
    assert m.compile_error_count == 1
    assert m.edit_count == 2
    assert m.compile_errors_per_100_edits == pytest.approx(50.0)
    assert m.test_fail_episodes == 1
    assert m.test_converted_episodes == 1
    assert m.test_fail_to_green_rate == pytest.approx(1.0)
    assert m.steps_to_green_mean == pytest.approx(4.0)
    assert m.verified_after_last_edit is True
    assert m.verification_rate == pytest.approx(1.0)

    # Read files = {src/calc.py, src/unused.py}, Edited = {src/calc.py} -> 1/2 = 0.5
    assert m.read_to_edit_precision == pytest.approx(0.5)


def test_r7_compaction_metrics_exclude_prune_and_category_c_d() -> None:
    """[R7]: Compaction failure rate & ratio cover summary compactions only; prune reported separately."""
    traj = Trajectory(
        source="dsh",
        session_id="synth-cd-r7",
        cwd="/workspace/demo",
        events=(
            # Turn 1: active timing [1000..2000] model (1000ms), [2100..2600] parallel tools (500ms), TurnEnd 2700 -> gap 200ms
            ModelCall(
                step=1,
                turn=1,
                started_at=1_000,
                ended_at=2_000,
                input_tokens=2_000,
                cache_read_tokens=7_000,
                cache_write_tokens=1_000,
                output_tokens=500,
                reasoning_tokens=200,
                retries=2,
            ),
            ToolCall(
                step=1,
                turn=1,
                call_id="r1",
                name="read",
                raw_args={"file_path": "src/a.py"},
                started_at=2_100,
                ended_at=2_500,
                is_error=False,
                truncated=True,
            ),
            ToolCall(
                step=1,
                turn=1,
                call_id="r2",
                name="read",
                raw_args={"file_path": "src/b.py"},
                started_at=2_200,
                ended_at=2_600,
                is_error=False,
                truncated=False,
            ),
            # 3 model-free prune runs that MUST NOT dilute summary compaction failure rate or ratio
            Compaction(
                started_at=2_610,
                ended_at=2_615,
                ok=True,
                tokens_before=80_000,
                tokens_after=70_000,
                reason="prune",
            ),
            Compaction(
                started_at=2_620,
                ended_at=2_625,
                ok=True,
                tokens_before=70_000,
                tokens_after=65_000,
                reason="prune",
            ),
            Compaction(
                started_at=2_630,
                ended_at=2_635,
                ok=True,
                tokens_before=65_000,
                tokens_after=60_000,
                reason="prune",
            ),
            # Summary compaction #1: succeeds (100_000 -> 25_000, ratio 0.25)
            Compaction(
                started_at=2_640,
                ended_at=2_680,
                ok=True,
                tokens_before=100_000,
                tokens_after=25_000,
                reason="summary",
            ),
            # Summary compaction #2: fails (context overflow)
            Compaction(
                started_at=2_685,
                ended_at=2_695,
                ok=False,
                tokens_before=120_000,
                tokens_after=None,
                reason="maximum context length exceeded",
            ),
            TurnEnd(turn=1, reason="completed", at=2_700),
            # 2-hour user idle break before Turn 2 (must NOT inflate gap_time_ms!)
            ModelCall(
                step=1,
                turn=2,
                started_at=7_202_700,
                ended_at=7_203_700,
                input_tokens=1_000,
                cache_read_tokens=9_000,
                cache_write_tokens=0,
                output_tokens=300,
                retries=0,
            ),
            # Post-compaction reads: src/a.py (re-read!) and src/c.py (fresh read) -> 1/2 = 0.5
            ToolCall(
                step=1,
                turn=2,
                call_id="r3",
                name="read",
                raw_args={"file_path": "src/a.py"},
                started_at=7_203_800,
                ended_at=7_204_000,
                is_error=False,
                truncated=False,
            ),
            ToolCall(
                step=1,
                turn=2,
                call_id="r4",
                name="read",
                raw_args={"file_path": "src/c.py"},
                started_at=7_204_000,
                ended_at=7_204_200,
                is_error=False,
                truncated=False,
            ),
            TurnEnd(turn=2, reason="error:TIMEOUT:stream idle timeout", at=7_204_300),
            # Turn 3: aborted
            ModelCall(step=1, turn=3, started_at=8_000_000, ended_at=8_000_500, retries=0),
            TurnEnd(turn=3, reason="aborted", at=8_000_600),
        ),
    )

    m = compute_trajectory_metrics(traj)

    # Category C
    assert m.total_prompt_tokens == 20_000
    assert m.cache_read_tokens_total == 16_000
    assert m.output_tokens_total == 800
    assert m.reasoning_tokens_total == 200
    assert m.total_tokens == 20_800
    assert m.cache_hit_ratio == pytest.approx(16_000 / 20_000)

    # Turn 1: model 1000, tool union [2100, 2600] = 500, span [1000, 2700] = 1700 -> gap 200
    # Turn 2: model 1000, tool union [7203800, 7204200] = 400, span [7202700, 7204300] = 1600 -> gap 200
    # Turn 3: model 500, tool 0, span [8000000, 8000600] = 600 -> gap 100
    assert m.model_time_ms == 2_500
    assert m.tool_time_ms == 900
    assert m.gap_time_ms == 500
    assert m.active_turn_wall_ms == 3_900

    # Category D + [R7]
    assert m.prune_compaction_count == 3
    assert m.summary_compaction_count == 2
    assert m.summary_compaction_failures == 1
    assert m.compaction_failure_rate == pytest.approx(0.5)
    assert m.compaction_ratio == pytest.approx(0.25)
    assert m.post_compaction_reread_rate == pytest.approx(0.5)
    assert m.truncation_rate == pytest.approx(0.25)
    assert m.retry_rate == pytest.approx(1 / 3)
    assert m.timeout_rate == pytest.approx(1 / 3)
    assert m.context_overflow_rate == pytest.approx(1 / 3)
    assert m.abort_rate == pytest.approx(1 / 3)

    # Round-trip dict serialization & aggregation
    restored = TrajectoryMetrics.from_dict(m.to_dict())
    assert restored == m
    agg = aggregate_trajectory_metrics([m])
    assert agg.compaction_failure_rate == pytest.approx(0.5)
    assert agg.compaction_ratio == pytest.approx(0.25)
    assert agg.prune_compaction_count == 3


def test_empty_trajectory_zero_denominators_return_none_never_zero() -> None:
    empty = Trajectory(source="claude-code", session_id="empty")
    m = compute_trajectory_metrics(empty)
    assert m.model_calls_per_turn is None
    assert m.tool_calls_per_turn is None
    assert m.near_duplicate_command_rate is None
    assert m.escape_latency_steps_mean is None
    assert m.unknown_tool_call_rate is None
    assert m.arg_validation_error_rate is None
    assert m.compile_errors_per_100_edits is None
    assert m.test_fail_to_green_rate is None
    assert m.steps_to_green_mean is None
    assert m.verification_rate is None
    assert m.read_to_edit_precision is None
    assert m.cache_hit_ratio is None
    assert m.model_time_share is None
    assert m.truncation_rate is None
    assert m.compaction_failure_rate is None
    assert m.compaction_ratio is None
    assert m.post_compaction_reread_rate is None
    assert m.retry_rate is None
    assert m.timeout_rate is None
    assert m.context_overflow_rate is None
    assert m.abort_rate is None


def test_escape_latency_steps_median_focused() -> None:
    """Repeated failure followed by a different action N steps later."""
    traj = Trajectory(
        source="dsh",
        session_id="focus-escape",
        events=(
            # Group 1: fails at step 1 and step 2, different action at step 5 -> escape = 5 - 1 = 4
            ToolCall(
                step=1,
                turn=1,
                call_id="e1",
                name="bash",
                raw_args={"command": "pytest tests/test_a.py"},
                is_error=True,
                exit_code=1,
                error_signature="test:AssertionError",
            ),
            ToolCall(
                step=2,
                turn=1,
                call_id="e2",
                name="bash",
                raw_args={"command": "pytest tests/test_a.py"},
                is_error=True,
                exit_code=1,
                error_signature="test:AssertionError",
            ),
            ToolCall(
                step=5,
                turn=1,
                call_id="e3",
                name="read",
                raw_args={"file_path": "tests/test_a.py"},
                is_error=False,
            ),
            # Group 2: fails at step 6 and step 7, different action at step 8 -> escape = 8 - 6 = 2
            ToolCall(
                step=6,
                turn=1,
                call_id="e4",
                name="bash",
                raw_args={"command": "go build ./..."},
                is_error=True,
                exit_code=2,
                error_signature="compile:undefined:foo",
            ),
            ToolCall(
                step=7,
                turn=1,
                call_id="e5",
                name="bash",
                raw_args={"command": "go build ./..."},
                is_error=True,
                exit_code=2,
                error_signature="compile:undefined:foo",
            ),
            ToolCall(
                step=8,
                turn=1,
                call_id="e6",
                name="edit",
                raw_args={"file_path": "main.go"},
                is_error=False,
            ),
            # Group 3: fails at step 9 and step 10, different action at step 19 -> escape = 19 - 9 = 10
            ToolCall(
                step=9,
                turn=1,
                call_id="e7",
                name="bash",
                raw_args={"command": "npm test"},
                is_error=True,
                exit_code=1,
                error_signature="test:FAIL",
            ),
            ToolCall(
                step=10,
                turn=1,
                call_id="e8",
                name="bash",
                raw_args={"command": "npm test"},
                is_error=True,
                exit_code=1,
                error_signature="test:FAIL",
            ),
            ToolCall(
                step=19,
                turn=1,
                call_id="e9",
                name="read",
                raw_args={"file_path": "package.json"},
                is_error=False,
            ),
        ),
    )
    m = compute_trajectory_metrics(traj)
    assert m.repeated_failure_groups == 3
    assert m.repeated_failure_calls == 6
    assert m.escape_latencies == (4, 2, 10)
    assert m.escape_latency_steps_median == pytest.approx(4.0)
    assert m.escape_latency_steps_mean == pytest.approx(16 / 3)


def test_steps_to_green_window_and_median_focused() -> None:
    """Test fails then passes within 10 steps (counts); >10 steps later must not count."""
    traj = Trajectory(
        source="dsh",
        session_id="focus-stg",
        events=(
            # Episode 1: fails at step 1, passes at step 4 -> 3 steps (<= 10, counts!)
            ToolCall(
                step=1,
                turn=1,
                call_id="t1",
                name="bash",
                raw_args={"command": "pytest"},
                is_error=True,
                exit_code=1,
                error_signature="test:AssertionError",
            ),
            ToolCall(
                step=4,
                turn=1,
                call_id="t2",
                name="bash",
                raw_args={"command": "pytest"},
                is_error=False,
                exit_code=0,
            ),
            # Episode 2: fails at step 5, passes at step 12 -> 7 steps (<= 10, counts!)
            ToolCall(
                step=5,
                turn=1,
                call_id="t3",
                name="bash",
                raw_args={"command": "pytest"},
                is_error=True,
                exit_code=1,
                error_signature="test:AssertionError",
            ),
            ToolCall(
                step=12,
                turn=1,
                call_id="t4",
                name="bash",
                raw_args={"command": "pytest"},
                is_error=False,
                exit_code=0,
            ),
            # Episode 3: fails at step 15, passes at step 26 -> 11 steps (> 10, must NOT count!)
            ToolCall(
                step=15,
                turn=1,
                call_id="t5",
                name="bash",
                raw_args={"command": "pytest"},
                is_error=True,
                exit_code=1,
                error_signature="test:AssertionError",
            ),
            ToolCall(
                step=26,
                turn=1,
                call_id="t6",
                name="bash",
                raw_args={"command": "pytest"},
                is_error=False,
                exit_code=0,
            ),
        ),
    )
    m = compute_trajectory_metrics(traj)
    assert m.test_fail_episodes == 3
    assert m.test_converted_episodes == 2
    assert m.test_fail_to_green_rate == pytest.approx(2 / 3)
    assert m.steps_to_green_values == (3, 7)
    assert m.steps_to_green_median == pytest.approx(5.0)
    assert m.steps_to_green_mean == pytest.approx(5.0)


def test_post_compaction_reread_counts_focused() -> None:
    """File read before summary compaction and again after, plus one read only after."""
    traj = Trajectory(
        source="dsh",
        session_id="focus-reread",
        cwd="/repo",
        events=(
            ToolCall(
                step=1,
                turn=1,
                call_id="r1",
                name="read",
                raw_args={"file_path": "/repo/src/mod.py"},
                is_error=False,
            ),
            Compaction(
                started_at=1_000,
                ended_at=1_500,
                ok=True,
                tokens_before=100_000,
                tokens_after=20_000,
                reason="auto",
            ),
            ToolCall(
                step=2,
                turn=1,
                call_id="r2",
                name="read",
                raw_args={"file_path": "src/mod.py"},
                is_error=False,
            ),
            ToolCall(
                step=3,
                turn=1,
                call_id="r3",
                name="read",
                raw_args={"file_path": "/repo/src/fresh.py"},
                is_error=False,
            ),
        ),
    )
    m = compute_trajectory_metrics(traj)
    assert m.post_compaction_read_count == 2
    assert m.post_compaction_reread_count == 1
    assert m.post_compaction_reread_rate == pytest.approx(0.5)


def test_turn_end_reasons_including_dsh_error_code_format_focused() -> None:
    """Aborted, timeout, and context-overflow turn counts including DSH error:CODE:message."""
    traj = Trajectory(
        source="dsh",
        session_id="focus-turn-reasons",
        events=(
            TurnEnd(turn=1, reason="aborted"),
            TurnEnd(turn=2, reason="error:REQUEST_TIMEOUT:upstream stream stalled"),
            TurnEnd(
                turn=3,
                reason="error:CONTEXT_LENGTH_EXCEEDED:input tokens exceed model context limit",
            ),
            TurnEnd(turn=4, reason="completed"),
        ),
    )
    m = compute_trajectory_metrics(traj)
    assert m.turn_count == 4
    assert m.aborted_turn_count == 1
    assert m.abort_rate == pytest.approx(0.25)
    assert m.timeout_turn_count == 1
    assert m.timeout_rate == pytest.approx(0.25)
    assert m.context_overflow_turn_count == 1
    assert m.context_overflow_rate == pytest.approx(0.25)


def test_truncation_counts_focused() -> None:
    """Truncated vs non-truncated vs unchecked (None) tool calls."""
    traj = Trajectory(
        source="dsh",
        session_id="focus-trunc",
        events=(
            ToolCall(step=1, call_id="c1", name="bash", truncated=True),
            ToolCall(step=2, call_id="c2", name="bash", truncated=True),
            ToolCall(step=3, call_id="c3", name="bash", truncated=False),
            ToolCall(step=4, call_id="c4", name="read", truncated=None),
        ),
    )
    m = compute_trajectory_metrics(traj)
    assert m.tool_call_count == 4
    assert m.truncation_checked_tool_call_count == 3
    assert m.truncated_tool_call_count == 2
    assert m.truncation_rate == pytest.approx(2 / 3)


def test_retry_counts_focused() -> None:
    """Total retry attempts vs number of retried model calls."""
    traj = Trajectory(
        source="dsh",
        session_id="focus-retries",
        events=(
            ModelCall(step=1, turn=1, retries=3),
            ModelCall(step=2, turn=1, retries=1),
            ModelCall(step=3, turn=1, retries=0),
            ModelCall(step=4, turn=1, retries=None),
        ),
    )
    m = compute_trajectory_metrics(traj)
    assert m.model_call_count == 4
    assert m.retry_count == 4
    assert m.retried_model_call_count == 2
    assert m.retry_rate == pytest.approx(0.5)


def test_parallel_tool_interval_merging_and_time_shares_focused() -> None:
    """Overlapping parallel tool calls merge without double-counting in tool_time_share/gap_time_share."""
    traj = Trajectory(
        source="claude-code",
        session_id="focus-timing",
        events=(
            # Model call: [1000, 1200] -> 200ms
            ModelCall(step=1, turn=1, started_at=1_000, ended_at=1_200),
            # Parallel overlapping tool calls: [1300, 1900] and [1500, 2300]
            # Merged tool union: [1300, 2300] -> 1000ms (not 600 + 800 = 1400ms!)
            ToolCall(step=1, turn=1, call_id="p1", name="bash", started_at=1_300, ended_at=1_900),
            ToolCall(step=1, turn=1, call_id="p2", name="bash", started_at=1_500, ended_at=2_300),
            # Turn ends at 2600 -> span = 2600 - 1000 = 1600ms; union(model U tool) = 1200ms -> gap = 400ms
            TurnEnd(turn=1, reason="completed", at=2_600),
        ),
    )
    m = compute_trajectory_metrics(traj)
    assert m.model_time_ms == 200
    assert m.tool_time_ms == 1_000
    assert m.gap_time_ms == 400
    assert m.active_turn_wall_ms == 1_600
    assert m.model_time_share == pytest.approx(200 / 1_600)
    assert m.tool_time_share == pytest.approx(1_000 / 1_600)
    assert m.gap_time_share == pytest.approx(400 / 1_600)


def test_aggregate_metrics_percentiles_and_all_fields_covered() -> None:
    """Verify corpus aggregation percentiles, sums, and full field coverage."""
    t1 = Trajectory(
        source="claude-code",
        session_id="agg-1",
        cwd="/proj",
        events=(
            ModelCall(
                step=1,
                turn=1,
                started_at=1_000,
                ended_at=1_500,
                input_tokens=500,
                cache_read_tokens=1_000,
                cache_write_tokens=500,
                output_tokens=100,
                reasoning_tokens=40,
                retries=1,
            ),
            ToolCall(
                step=1,
                turn=1,
                call_id="a1",
                name="Read",
                raw_args={"file_path": "/proj/a.py"},
                started_at=1_500,
                ended_at=1_700,
                is_error=False,
                truncated=False,
                available_tools=("Read", "Edit", "Bash"),
            ),
            ToolCall(
                step=1,
                turn=1,
                call_id="a2",
                name="Edit",
                raw_args={"file_path": "/proj/a.py"},
                started_at=1_700,
                ended_at=1_800,
                is_error=False,
                truncated=False,
                available_tools=("Read", "Edit", "Bash"),
            ),
            Edit(step=1, turn=1, path="/proj/a.py", at=1_800),
            ToolCall(
                step=2,
                turn=1,
                call_id="a3",
                name="Bash",
                raw_args={"command": "pytest"},
                started_at=1_900,
                ended_at=2_200,
                is_error=False,
                exit_code=0,
                truncated=False,
                available_tools=("Read", "Edit", "Bash"),
            ),
            TurnEnd(turn=1, reason="completed", at=2_300),
        ),
    )
    t2 = Trajectory(
        source="claude-code",
        session_id="agg-2",
        events=(
            ModelCall(step=1, turn=1, input_tokens=200, output_tokens=50),
            ModelCall(step=2, turn=1, input_tokens=200, output_tokens=50),
            ModelCall(step=3, turn=1, input_tokens=200, output_tokens=50),
            TurnEnd(turn=1, reason="completed"),
        ),
    )
    m1 = compute_trajectory_metrics(t1)
    m2 = compute_trajectory_metrics(t2)
    assert m1.source == "claude-code"
    assert m1.session_id == "agg-1"
    assert m1.metric_version == METRIC_VERSION
    assert m1.step_count == 2
    assert m1.schema_checked_tool_call_count == 3
    assert m1.read_files_in_edit_session_count == 1
    assert m1.edited_read_files_count == 1
    assert m1.input_tokens_total == 500
    assert m1.cache_write_tokens_total == 500
    assert m1.compaction_tokens_before_sum == 0
    assert m1.compaction_tokens_after_sum == 0

    agg = aggregate_trajectory_metrics([m1, m2])
    assert agg.source == "claude-code"
    assert agg.trajectory_count == 2
    assert agg.metric_version == METRIC_VERSION
    assert agg.turn_count == 2
    assert agg.model_call_count == 4
    assert agg.tool_call_count == 3
    assert agg.command_count == 1
    assert agg.edit_count == 1
    assert agg.edit_session_count == 1
    assert agg.verified_edit_session_count == 1
    assert agg.verification_rate == pytest.approx(1.0)
    assert agg.read_files_in_edit_session_count == 1
    assert agg.edited_read_files_count == 1
    assert agg.input_tokens_total == 1_100
    assert agg.model_calls_per_turn == pytest.approx(2.0)
    assert agg.model_calls_per_turn_p50 == pytest.approx(2.0)
    assert agg.model_calls_per_turn_p90 == pytest.approx(2.8)
    assert agg.tool_calls_per_turn == pytest.approx(1.5)
    assert agg.tool_calls_per_turn_p50 == pytest.approx(1.5)
    assert agg.tool_calls_per_turn_p90 == pytest.approx(2.7)
    assert agg.cache_write_tokens_total == 500
    assert agg.escape_latency_steps_p90 is None
    assert agg.steps_to_green_p90 is None

