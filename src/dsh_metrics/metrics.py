"""Passive trajectory and corpus metrics (M5, metric_version=1)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import re
import statistics
from typing import Any

from dsh_metrics.classifiers import (
    SHELL_TOOL_NAMES,
    classify_error_kind,
    classify_tool_call_kind,
    extract_read_paths,
    extract_shell_command,
    fingerprint_command,
    fingerprint_tool_args,
    normalize_repo_path,
)
from dsh_metrics.ir import (
    Compaction,
    Edit,
    ModelCall,
    ToolCall,
    Trajectory,
    TurnEnd,
    event_timestamp,
)

METRIC_VERSION = 1
TEST_CONVERSION_WINDOW_STEPS = 10

_ARG_VALIDATION_RE = re.compile(
    r"(?:"
    r"\bInputValidationError\b"
    r"|\binvalid[_ -]+(?:tool[_ -]+)?(?:arguments?|parameters?|input)\b"
    r"|\bmissing\s+required\s+(?:parameter|property|argument|field)\b"
    r"|\bmust\s+have\s+required\s+property\b"
    r"|\brequired\s+property\b"
    r"|\bunexpected\s+(?:keyword\s+)?argument\b"
    r"|\bFailed\s+to\s+parse\s+(?:tool\s+)?arguments\b"
    r"|\bExpected\s+(?:string|number|boolean|object|array),\s+received\b"
    r"|\bschema\s+validation\b"
    r")",
    re.IGNORECASE,
)

_TURN_TIMEOUT_RE = re.compile(
    r"(?:timeout|timed[_\s-]*out|etimedout|deadline[_\s-]*exceeded)",
    re.IGNORECASE,
)

_TOOL_TIMEOUT_RE = re.compile(
    r"(?:\btimed\s+out\b|\bTime(?:out|dOut)Error\b|\betimedout\b|\bdeadline[_\s-]+exceeded\b|\bcommand\s+timed\s+out\b)",
    re.IGNORECASE,
)

_CONTEXT_OVERFLOW_RE = re.compile(
    r"(?:"
    r"context[_\s-]*(?:length|overflow|limit[_\s-]*exceeded|exceeded|too[_\s-]*(?:long|large))"
    r"|prompt[_\s-]*(?:is[_\s-]*)?too[_\s-]*long"
    r"|maximum[_\s-]*context[_\s-]*length"
    r"|context[_\s-]*window[^\r\n]{0,40}(?:exceed|overflow|full|limit)"
    r"|too[_\s-]*many[_\s-]*tokens"
    r"|token[_\s-]*limit[_\s-]*exceeded"
    r"|input[_\s-]*length[^\r\n]{0,30}exceed"
    r"|request[_\s-]*too[_\s-]*large"
    r")",
    re.IGNORECASE,
)

_ABORT_REASON_RE = re.compile(
    r"(?:^|:|[_\s-])(?:aborted|abort|interrupted|cancelled|canceled)(?:$|:|[_\s-])",
    re.IGNORECASE,
)


def _safe_div(num: int | float, den: int | float) -> float | None:
    """Return `num / den`, or `None` when `den <= 0`."""
    if den <= 0:
        return None
    return float(num) / float(den)


def _mean_or_none(values: Sequence[int | float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def _median_or_none(values: Sequence[int | float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _percentile_or_none(values: Sequence[int | float], q: float) -> float | None:
    """Return linear-interpolated percentile `q` in [0, 100], or `None` if empty."""
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping `[start_ms, end_ms]` intervals with `end_ms >= start_ms`."""
    valid = sorted((s, e) for s, e in intervals if e >= s)
    if not valid:
        return []
    merged: list[tuple[int, int]] = [valid[0]]
    for s, e in valid[1:]:
        prev_s, prev_e = merged[-1]
        if s <= prev_e:
            merged[-1] = (prev_s, max(prev_e, e))
        else:
            merged.append((s, e))
    return merged


def _intervals_duration(intervals: Sequence[tuple[int, int]]) -> int:
    return sum(e - s for s, e in _merge_intervals(intervals))


def _is_arg_validation_error(tc: ToolCall) -> bool:
    """Detect whether a failed ToolCall resulted from invalid/malformed tool arguments.

    Formula:
        `tc.is_error is True` and `tc.exit_code is None` (not an application exit status from
        an executed shell process), and either:
        1. `tc.raw_args` is non-empty malformed JSON on a non-shell structured tool, or
        2. `tc.output_tail` or `tc.error_signature` matches schema/argument validation markers.
    """
    if tc.is_error is not True:
        return False
    # Shell commands that ran and returned an OS exit code failed inside the subprocess,
    # not at tool argument validation.
    if tc.exit_code is not None:
        return False

    tlower = tc.name.strip().lower()
    if tlower not in SHELL_TOOL_NAMES and tc.raw_args:
        stripped = tc.raw_args.strip()
        if stripped.startswith("{"):
            try:
                json.loads(stripped)
            except json.JSONDecodeError:
                return True

    haystack = "\n".join(
        part for part in (tc.error_signature, tc.output_tail) if part
    )
    return bool(haystack and _ARG_VALIDATION_RE.search(haystack))


@dataclass(frozen=True, slots=True)
class TrajectoryMetrics:
    """Passive intelligence and reliability metrics for a single Trajectory (`metric_version=1`).

    Formulas (unknown / zero-denominator rates are `None`, never `0`):
    - Category A (Control loop & tool efficiency):
      - `model_calls_per_turn = model_call_count / turn_count` (`turn_count > 0`)
      - `tool_calls_per_turn = tool_call_count / turn_count` (`turn_count > 0`)
      - `near_duplicate_command_rate = near_duplicate_command_count / command_count`
        where a shell command call is a near-duplicate if its normalized `fingerprint_command`
        already appeared earlier in the trajectory.
      - `repeated_failure_groups`: count of distinct `(fingerprint, error_signature)` pairs
        with `is_error=True` occurring `>= 2` times in the trajectory;
        `repeated_failure_calls`: total calls belonging to those repeated-failure groups;
        `escape_latency_steps_mean`: mean of `(s_escape - s_first)` in monotonic trajectory
        steps across escaped repeated-failure groups, where `s_first` is the group's first
        failing step and `s_escape` is the first subsequent trajectory step strictly after the
        group's last failure.
      - `unknown_tool_call_rate = unknown_tool_call_count / schema_checked_tool_call_count`
        over tool calls with non-empty `available_tools` where `name not in available_tools`.
      - `arg_validation_error_rate = arg_validation_error_count / tool_call_count`.
    - Category B (Edit accuracy & verification):
      - `compile_errors_per_100_edits = (compile_error_count / edit_count) * 100` (`edit_count > 0`).
      - `test_fail_to_green_rate = test_converted_episodes / test_fail_episodes`, where each
        failing test episode begins at an unresolved failing `test`-kind tool call at step
        `s_fail` and converts if a passing `test`-kind tool call occurs at `s_green` with
        `0 <= s_green - s_fail <= 10`; `steps_to_green_mean` / `steps_to_green_median` are
        computed over `s_green - s_fail` of converted episodes.
      - `verification_rate`: `1.0` (`verified_after_last_edit=True`) if `edit_count > 0` and at
        least one `test` or `build` tool call occurs after the trajectory's last `Edit`,
        `0.0` (`False`) if `edit_count > 0` and none occurs, or `None` if `edit_count == 0`.
      - `read_to_edit_precision = len(read_files & edited_files) / len(read_files)` when
        `len(edited_files) > 0` and `len(read_files) > 0`, else `None`.
    - Category C (Tokens, caching & latency):
      - `total_prompt_tokens = input_tokens_total + cache_read_tokens_total + cache_write_tokens_total`
      - `total_tokens = total_prompt_tokens + output_tokens_total`
      - `cache_hit_ratio = cache_read_tokens_total / total_prompt_tokens` (`total_prompt_tokens > 0`).
      - `model_time_ms`, `tool_time_ms`, `gap_time_ms`: summed per active turn using merged
        interval unions for model calls and tool calls so concurrent calls do not double-count
        and inter-turn user idle breaks are excluded (`gap_time_ms = sum_turn max(0, turn_span - union(model U tool))`).
    - Category D (Context management & reliability):
      - `truncation_rate = truncated_tool_call_count / truncation_checked_tool_call_count`.
      - `compaction_failure_rate = summary_compaction_failures / summary_compaction_count`
        over summary compactions ONLY (`reason != "prune"`, per `[R7]`).
      - `compaction_ratio = compaction_tokens_after_sum / compaction_tokens_before_sum`
        over successful summary compactions (`reason != "prune"`, `ok is not False`,
        `tokens_before > 0` and `tokens_after is not None`).
      - `prune_compaction_count`: separate count of model-free `reason == "prune"` runs (`[R7]`).
      - `post_compaction_reread_rate = post_compaction_reread_count / post_compaction_read_count`,
        the share of post-summary-compaction file reads targeting paths already read before that compaction.
      - `retry_rate = retried_model_call_count / model_call_count` (`(retries or 0) > 0`).
      - `timeout_rate = timeout_turn_count / turn_count`.
      - `context_overflow_rate = context_overflow_turn_count / turn_count`.
      - `abort_rate = aborted_turn_count / turn_count`.
    """

    source: str
    session_id: str
    metric_version: int = METRIC_VERSION

    # Counts & denominators
    turn_count: int = 0
    step_count: int = 0
    model_call_count: int = 0
    tool_call_count: int = 0
    command_count: int = 0
    edit_count: int = 0

    # Category A
    model_calls_per_turn: float | None = None
    tool_calls_per_turn: float | None = None
    near_duplicate_command_count: int = 0
    near_duplicate_command_rate: float | None = None
    repeated_failure_groups: int = 0
    repeated_failure_calls: int = 0
    escape_latencies: tuple[int, ...] = ()
    escape_latency_steps_mean: float | None = None
    escape_latency_steps_median: float | None = None
    schema_checked_tool_call_count: int = 0
    unknown_tool_call_count: int = 0
    unknown_tool_call_rate: float | None = None
    arg_validation_error_count: int = 0
    arg_validation_error_rate: float | None = None

    # Category B
    compile_error_count: int = 0
    compile_errors_per_100_edits: float | None = None
    test_fail_episodes: int = 0
    test_converted_episodes: int = 0
    test_fail_to_green_rate: float | None = None
    steps_to_green_values: tuple[int, ...] = ()
    steps_to_green_mean: float | None = None
    steps_to_green_median: float | None = None
    verified_after_last_edit: bool | None = None
    verification_rate: float | None = None
    read_files_in_edit_session_count: int = 0
    edited_read_files_count: int = 0
    read_to_edit_precision: float | None = None

    # Category C
    input_tokens_total: int = 0
    cache_read_tokens_total: int = 0
    cache_write_tokens_total: int = 0
    output_tokens_total: int = 0
    reasoning_tokens_total: int = 0
    total_prompt_tokens: int = 0
    total_tokens: int = 0
    cache_hit_ratio: float | None = None
    model_time_ms: int = 0
    tool_time_ms: int = 0
    gap_time_ms: int = 0
    active_turn_wall_ms: int = 0
    model_time_share: float | None = None
    tool_time_share: float | None = None
    gap_time_share: float | None = None

    # Category D
    truncation_checked_tool_call_count: int = 0
    truncated_tool_call_count: int = 0
    truncation_rate: float | None = None
    summary_compaction_count: int = 0
    summary_compaction_failures: int = 0
    compaction_failure_rate: float | None = None
    compaction_tokens_before_sum: int = 0
    compaction_tokens_after_sum: int = 0
    compaction_ratio: float | None = None
    prune_compaction_count: int = 0
    post_compaction_read_count: int = 0
    post_compaction_reread_count: int = 0
    post_compaction_reread_rate: float | None = None
    retry_count: int = 0
    retried_model_call_count: int = 0
    retry_rate: float | None = None
    timeout_turn_count: int = 0
    timeout_rate: float | None = None
    context_overflow_turn_count: int = 0
    context_overflow_rate: float | None = None
    aborted_turn_count: int = 0
    abort_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize into a JSON-compatible dictionary."""
        d = asdict(self)
        d["escape_latencies"] = list(self.escape_latencies)
        d["steps_to_green_values"] = list(self.steps_to_green_values)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrajectoryMetrics:
        """Deserialize from a dictionary produced by `to_dict()`."""
        kwargs = dict(data)
        if "escape_latencies" in kwargs and kwargs["escape_latencies"] is not None:
            kwargs["escape_latencies"] = tuple(int(x) for x in kwargs["escape_latencies"])
        if (
            "steps_to_green_values" in kwargs
            and kwargs["steps_to_green_values"] is not None
        ):
            kwargs["steps_to_green_values"] = tuple(
                int(x) for x in kwargs["steps_to_green_values"]
            )
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    """Corpus-level aggregated metrics over a collection of trajectories (`metric_version=1`).

    Formulas:
    - Micro-averaged corpus rates use summed numerators / summed denominators across all
      trajectories in `source` so short trajectories do not disproportionately skew token or
      call rates, while trajectory-level percentiles (`p50`, `p90`) summarize per-session
      distributions.
    - All zero-denominator rates resolve to `None`.
    """

    source: str
    trajectory_count: int
    metric_version: int = METRIC_VERSION

    turn_count: int = 0
    model_call_count: int = 0
    tool_call_count: int = 0
    command_count: int = 0
    edit_count: int = 0
    edit_session_count: int = 0

    # Category A
    model_calls_per_turn: float | None = None
    model_calls_per_turn_p50: float | None = None
    model_calls_per_turn_p90: float | None = None
    tool_calls_per_turn: float | None = None
    tool_calls_per_turn_p50: float | None = None
    tool_calls_per_turn_p90: float | None = None
    near_duplicate_command_count: int = 0
    near_duplicate_command_rate: float | None = None
    repeated_failure_groups: int = 0
    repeated_failure_calls: int = 0
    escape_latency_steps_mean: float | None = None
    escape_latency_steps_median: float | None = None
    escape_latency_steps_p90: float | None = None
    unknown_tool_call_count: int = 0
    unknown_tool_call_rate: float | None = None
    arg_validation_error_count: int = 0
    arg_validation_error_rate: float | None = None

    # Category B
    compile_error_count: int = 0
    compile_errors_per_100_edits: float | None = None
    test_fail_episodes: int = 0
    test_converted_episodes: int = 0
    test_fail_to_green_rate: float | None = None
    steps_to_green_mean: float | None = None
    steps_to_green_median: float | None = None
    steps_to_green_p90: float | None = None
    verified_edit_session_count: int = 0
    verification_rate: float | None = None
    read_files_in_edit_session_count: int = 0
    edited_read_files_count: int = 0
    read_to_edit_precision: float | None = None

    # Category C
    input_tokens_total: int = 0
    cache_read_tokens_total: int = 0
    cache_write_tokens_total: int = 0
    output_tokens_total: int = 0
    reasoning_tokens_total: int = 0
    total_prompt_tokens: int = 0
    total_tokens: int = 0
    cache_hit_ratio: float | None = None
    model_time_ms: int = 0
    tool_time_ms: int = 0
    gap_time_ms: int = 0
    active_turn_wall_ms: int = 0
    model_time_share: float | None = None
    tool_time_share: float | None = None
    gap_time_share: float | None = None

    # Category D
    truncated_tool_call_count: int = 0
    truncation_checked_tool_call_count: int = 0
    truncation_rate: float | None = None
    summary_compaction_count: int = 0
    summary_compaction_failures: int = 0
    compaction_failure_rate: float | None = None
    compaction_tokens_before_sum: int = 0
    compaction_tokens_after_sum: int = 0
    compaction_ratio: float | None = None
    prune_compaction_count: int = 0
    post_compaction_read_count: int = 0
    post_compaction_reread_count: int = 0
    post_compaction_reread_rate: float | None = None
    retry_count: int = 0
    retried_model_call_count: int = 0
    retry_rate: float | None = None
    timeout_turn_count: int = 0
    timeout_rate: float | None = None
    context_overflow_turn_count: int = 0
    context_overflow_rate: float | None = None
    aborted_turn_count: int = 0
    abort_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _assign_monotonic_steps(trajectory: Trajectory) -> tuple[list[int], int]:
    """Assign 1-indexed monotonic trajectory step numbers aligned with `trajectory.events`."""
    seq_step = 0
    prev_turn: int | None = None
    prev_raw_step: int | None = None
    event_steps: list[int] = []

    for ev in trajectory.events:
        if isinstance(ev, (ModelCall, ToolCall, Edit)):
            if seq_step == 0:
                seq_step = ev.step if (ev.step is not None and ev.step > 0) else 1
                prev_turn = ev.turn
                prev_raw_step = ev.step
            elif ev.step is None:
                if isinstance(ev, ModelCall) or ev.turn != prev_turn:
                    seq_step += 1
                    prev_turn = ev.turn
                    prev_raw_step = None
            else:
                if ev.turn == prev_turn and prev_raw_step is not None:
                    if ev.step > prev_raw_step:
                        seq_step += ev.step - prev_raw_step
                    elif ev.step < prev_raw_step:
                        seq_step += max(ev.step, 1)
                else:
                    if prev_raw_step is not None and ev.step > prev_raw_step:
                        seq_step += ev.step - prev_raw_step
                    elif prev_raw_step is not None:
                        seq_step += max(ev.step, 1)
                    else:
                        seq_step += 1
                prev_turn = ev.turn
                prev_raw_step = ev.step
        event_steps.append(max(seq_step, 1) if seq_step > 0 else 0)

    return event_steps, seq_step


def _compute_turn_timing(trajectory: Trajectory) -> tuple[int, int, int, int]:
    """Compute `(model_time_ms, tool_time_ms, gap_time_ms, active_turn_wall_ms)` per turn.

    Formula:
        For each turn `u`, let `M_u` be merged `[started_at, ended_at]` intervals of `ModelCall`s,
        `T_u` be merged `[started_at, ended_at]` intervals of `ToolCall`s, `U_u = merge(M_u + T_u)`,
        and `span_u = max_ts(u) - min_ts(u)` across all timed events in turn `u`.
        Then:
          `model_time_ms = sum_u duration(M_u)`
          `tool_time_ms  = sum_u duration(T_u)`
          `gap_time_ms   = sum_u max(0, span_u - duration(U_u))`
          `active_turn_wall_ms = model_time_ms + tool_time_ms + gap_time_ms`
    """
    model_by_turn: dict[int, list[tuple[int, int]]] = defaultdict(list)
    tool_by_turn: dict[int, list[tuple[int, int]]] = defaultdict(list)
    ts_by_turn: dict[int, list[int]] = defaultdict(list)

    fallback_turn = 1
    for ev in trajectory.events:
        t_num: int
        if isinstance(ev, (ModelCall, ToolCall, Edit, TurnEnd)) and ev.turn is not None:
            t_num = ev.turn
            fallback_turn = t_num
        else:
            t_num = fallback_turn

        ts = event_timestamp(ev)
        if ts is not None:
            ts_by_turn[t_num].append(ts)

        if isinstance(ev, ModelCall):
            if ev.ended_at is not None:
                ts_by_turn[t_num].append(ev.ended_at)
            if (
                ev.started_at is not None
                and ev.ended_at is not None
                and ev.ended_at >= ev.started_at
            ):
                model_by_turn[t_num].append((ev.started_at, ev.ended_at))
        elif isinstance(ev, ToolCall):
            if ev.ended_at is not None:
                ts_by_turn[t_num].append(ev.ended_at)
            if (
                ev.started_at is not None
                and ev.ended_at is not None
                and ev.ended_at >= ev.started_at
            ):
                tool_by_turn[t_num].append((ev.started_at, ev.ended_at))
        elif isinstance(ev, Compaction):
            if ev.ended_at is not None:
                ts_by_turn[t_num].append(ev.ended_at)

        if isinstance(ev, TurnEnd):
            fallback_turn = t_num + 1

    all_turns = set(ts_by_turn) | set(model_by_turn) | set(tool_by_turn)
    total_model = 0
    total_tool = 0
    total_gap = 0

    for u in all_turns:
        m_iv = _merge_intervals(model_by_turn[u])
        t_iv = _merge_intervals(tool_by_turn[u])
        u_iv = _merge_intervals(m_iv + t_iv)
        m_dur = _intervals_duration(m_iv)
        t_dur = _intervals_duration(t_iv)
        u_dur = _intervals_duration(u_iv)
        pts = ts_by_turn[u]
        span = (max(pts) - min(pts)) if len(pts) >= 2 else u_dur
        gap = max(0, span - u_dur)
        total_model += m_dur
        total_tool += t_dur
        total_gap += gap

    active_wall = total_model + total_tool + total_gap
    return total_model, total_tool, total_gap, active_wall


def compute_trajectory_metrics(trajectory: Trajectory) -> TrajectoryMetrics:
    """Compute all M5 passive metrics (`metric_version=1`) on a normalized `Trajectory`."""
    event_steps, total_steps = _assign_monotonic_steps(trajectory)

    model_calls: list[ModelCall] = []
    tool_calls_with_step: list[tuple[int, int, ToolCall]] = []
    edits_with_idx: list[tuple[int, Edit]] = []
    compactions: list[Compaction] = []
    turn_ends: list[TurnEnd] = []
    seen_turns: set[int] = set()

    for idx, ev in enumerate(trajectory.events):
        s_num = event_steps[idx]
        if isinstance(ev, ModelCall):
            model_calls.append(ev)
            if ev.turn is not None:
                seen_turns.add(ev.turn)
        elif isinstance(ev, ToolCall):
            tool_calls_with_step.append((idx, s_num, ev))
            if ev.turn is not None:
                seen_turns.add(ev.turn)
        elif isinstance(ev, Edit):
            edits_with_idx.append((idx, ev))
            if ev.turn is not None:
                seen_turns.add(ev.turn)
        elif isinstance(ev, Compaction):
            compactions.append(ev)
        elif isinstance(ev, TurnEnd):
            turn_ends.append(ev)
            if ev.turn is not None:
                seen_turns.add(ev.turn)

    if seen_turns or turn_ends:
        turn_count = max(len(seen_turns), len(turn_ends))
    elif model_calls or tool_calls_with_step or edits_with_idx:
        turn_count = 1
    else:
        turn_count = 0

    mc_count = len(model_calls)
    tc_count = len(tool_calls_with_step)
    edit_count = len(edits_with_idx)

    # --- Category A: Calls/turn, near-duplicate commands, repeated failures, tool errors ---
    seen_cmd_fps: set[str] = set()
    command_count = 0
    near_dup_cmd_count = 0

    failure_steps_by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    schema_checked_tc = 0
    unknown_tc = 0
    arg_val_err_tc = 0

    # --- Category B: Compile errors, test fail->green, verification, read->edit ---
    compile_err_count = 0
    test_runs: list[tuple[int, bool, bool]] = []  # (seq_step, is_fail, is_green)
    all_read_files: set[str] = set()

    # --- Category D: Truncation, timeouts, context overflows ---
    trunc_checked_tc = 0
    truncated_tc = 0
    timeout_turns: set[int] = set()
    overflow_turns: set[int] = set()
    aborted_turns: set[int] = set()

    for idx, s_num, tc in tool_calls_with_step:
        cmd = extract_shell_command(tc)
        if cmd is not None or tc.name.strip().lower() in SHELL_TOOL_NAMES:
            fp_cmd = fingerprint_command(cmd) if cmd is not None else (tc.args_fingerprint or "")
            if fp_cmd:
                command_count += 1
                if fp_cmd in seen_cmd_fps:
                    near_dup_cmd_count += 1
                else:
                    seen_cmd_fps.add(fp_cmd)
            eff_fp = fp_cmd or tc.args_fingerprint
        else:
            eff_fp = tc.args_fingerprint or fingerprint_tool_args(tc.name, None, tc.raw_args)

        if tc.is_error is True and eff_fp and tc.error_signature:
            failure_steps_by_key[(eff_fp, tc.error_signature)].append(s_num)

        if tc.available_tools is not None and len(tc.available_tools) > 0:
            if not (tc.name == "unknown" and tc.started_at is None):
                schema_checked_tc += 1
                if tc.name not in tc.available_tools:
                    unknown_tc += 1

        if _is_arg_validation_error(tc):
            arg_val_err_tc += 1

        err_kind = classify_error_kind(tc.error_signature)
        if err_kind == "compile":
            compile_err_count += 1

        tc_kind = classify_tool_call_kind(tc)
        if tc_kind == "test":
            is_fail = tc.is_error is True or err_kind in ("test", "compile", "other")
            is_green = tc.is_error is False and err_kind is None
            test_runs.append((s_num, is_fail, is_green))

        for rpath in extract_read_paths(tc, cwd=trajectory.cwd):
            all_read_files.add(rpath)

        if tc.truncated is not None:
            trunc_checked_tc += 1
            if tc.truncated is True:
                truncated_tc += 1

        t_id = tc.turn if tc.turn is not None else 1
        err_blob = "\n".join(p for p in (tc.error_signature, tc.output_tail) if p)
        if tc.is_error is True and err_blob:
            if _TOOL_TIMEOUT_RE.search(err_blob):
                timeout_turns.add(t_id)
            if _CONTEXT_OVERFLOW_RE.search(err_blob):
                overflow_turns.add(t_id)

    # Repeated failures & escape latency in steps
    all_positive_steps = sorted({s for s in event_steps if s > 0})
    repeated_groups = 0
    repeated_calls = 0
    escape_latencies: list[int] = []
    for steps_list in failure_steps_by_key.values():
        if len(steps_list) >= 2:
            repeated_groups += 1
            repeated_calls += len(steps_list)
            first_s = steps_list[0]
            last_s = steps_list[-1]
            escaped_steps = [s for s in all_positive_steps if s > last_s]
            if escaped_steps:
                escape_latencies.append(escaped_steps[0] - first_s)

    # Test fail->green conversion within 10 steps
    test_fail_episodes = 0
    test_converted_episodes = 0
    steps_to_green_vals: list[int] = []
    active_fail_step: int | None = None

    for s_num, is_fail, is_green in test_runs:
        if active_fail_step is not None and s_num - active_fail_step > TEST_CONVERSION_WINDOW_STEPS:
            # Window expired without converting to green
            active_fail_step = None

        if active_fail_step is None:
            if is_fail:
                test_fail_episodes += 1
                active_fail_step = s_num
        else:
            if is_green:
                delta = s_num - active_fail_step
                if 0 <= delta <= TEST_CONVERSION_WINDOW_STEPS:
                    test_converted_episodes += 1
                    steps_to_green_vals.append(delta)
                active_fail_step = None

    # Verification rate (tests/build after last edit)
    verified_after_last_edit: bool | None = None
    verification_rate: float | None = None
    if edits_with_idx:
        last_edit_idx = edits_with_idx[-1][0]
        verified_after_last_edit = any(
            idx > last_edit_idx and classify_tool_call_kind(tc) in ("test", "build")
            for idx, _, tc in tool_calls_with_step
        )
        verification_rate = 1.0 if verified_after_last_edit else 0.0

    # Read->edit precision proxy
    edited_files = {
        normalize_repo_path(ed.path, cwd=trajectory.cwd)
        for _, ed in edits_with_idx
        if normalize_repo_path(ed.path, cwd=trajectory.cwd)
    }
    if edited_files:
        read_in_edit_cnt = len(all_read_files)
        edited_read_cnt = len(all_read_files & edited_files)
        read_to_edit_prec = _safe_div(edited_read_cnt, read_in_edit_cnt)
    else:
        read_in_edit_cnt = 0
        edited_read_cnt = 0
        read_to_edit_prec = None

    # --- Category C: Tokens, cache hit ratio, timing ---
    in_tok_total = sum(mc.input_tokens or 0 for mc in model_calls)
    cache_r_total = sum(mc.cache_read_tokens or 0 for mc in model_calls)
    cache_w_total = sum(mc.cache_write_tokens or 0 for mc in model_calls)
    out_tok_total = sum(mc.output_tokens or 0 for mc in model_calls)
    reas_tok_total = sum(mc.reasoning_tokens or 0 for mc in model_calls)
    prompt_tok_total = in_tok_total + cache_r_total + cache_w_total
    total_tok = prompt_tok_total + out_tok_total
    cache_hit_ratio = _safe_div(cache_r_total, prompt_tok_total)

    model_ms, tool_ms, gap_ms, active_wall_ms = _compute_turn_timing(trajectory)
    model_share = _safe_div(model_ms, active_wall_ms)
    tool_share = _safe_div(tool_ms, active_wall_ms)
    gap_share = _safe_div(gap_ms, active_wall_ms)

    # --- Category D: Compactions ([R7]), post-compaction re-reads, retries, turn-end errors ---
    prune_comp_count = 0
    summary_comp_count = 0
    summary_comp_failures = 0
    comp_before_sum = 0
    comp_after_sum = 0

    for comp in compactions:
        if comp.reason == "prune":
            prune_comp_count += 1
            continue
        summary_comp_count += 1
        if comp.ok is False:
            summary_comp_failures += 1
            if comp.reason and _CONTEXT_OVERFLOW_RE.search(comp.reason):
                overflow_turns.add(1)
            if comp.reason and _TURN_TIMEOUT_RE.search(comp.reason):
                timeout_turns.add(1)
        elif (
            comp.tokens_before is not None
            and comp.tokens_before > 0
            and comp.tokens_after is not None
        ):
            comp_before_sum += comp.tokens_before
            comp_after_sum += comp.tokens_after

    # Post-compaction re-read rate (over successful summary compactions)
    seen_reads_so_far: set[str] = set()
    snapshot_at_compaction: set[str] | None = None
    post_comp_reads = 0
    post_comp_rereads = 0

    for ev in trajectory.events:
        if isinstance(ev, Compaction) and ev.reason != "prune" and ev.ok is True:
            snapshot_at_compaction = set(seen_reads_so_far)
        elif isinstance(ev, ToolCall):
            rpaths = extract_read_paths(ev, cwd=trajectory.cwd)
            for rp in rpaths:
                if snapshot_at_compaction is not None:
                    post_comp_reads += 1
                    if rp in snapshot_at_compaction:
                        post_comp_rereads += 1
                seen_reads_so_far.add(rp)

    retry_count = sum(mc.retries or 0 for mc in model_calls)
    retried_mc_count = sum(1 for mc in model_calls if (mc.retries or 0) > 0)

    for idx_te, te in enumerate(turn_ends, start=1):
        t_id = te.turn if te.turn is not None else idx_te
        reason_str = te.reason or ""
        if reason_str:
            if _ABORT_REASON_RE.search(reason_str):
                aborted_turns.add(t_id)
            if _TURN_TIMEOUT_RE.search(reason_str):
                timeout_turns.add(t_id)
            if _CONTEXT_OVERFLOW_RE.search(reason_str):
                overflow_turns.add(t_id)

    timeout_turn_cnt = len(timeout_turns)
    overflow_turn_cnt = len(overflow_turns)
    aborted_turn_cnt = len(aborted_turns)

    return TrajectoryMetrics(
        source=trajectory.source,
        session_id=trajectory.session_id,
        metric_version=METRIC_VERSION,
        turn_count=turn_count,
        step_count=total_steps,
        model_call_count=mc_count,
        tool_call_count=tc_count,
        command_count=command_count,
        edit_count=edit_count,
        model_calls_per_turn=_safe_div(mc_count, turn_count),
        tool_calls_per_turn=_safe_div(tc_count, turn_count),
        near_duplicate_command_count=near_dup_cmd_count,
        near_duplicate_command_rate=_safe_div(near_dup_cmd_count, command_count),
        repeated_failure_groups=repeated_groups,
        repeated_failure_calls=repeated_calls,
        escape_latencies=tuple(escape_latencies),
        escape_latency_steps_mean=_mean_or_none(escape_latencies),
        escape_latency_steps_median=_median_or_none(escape_latencies),
        schema_checked_tool_call_count=schema_checked_tc,
        unknown_tool_call_count=unknown_tc,
        unknown_tool_call_rate=_safe_div(unknown_tc, schema_checked_tc),
        arg_validation_error_count=arg_val_err_tc,
        arg_validation_error_rate=_safe_div(arg_val_err_tc, tc_count),
        compile_error_count=compile_err_count,
        compile_errors_per_100_edits=(
            (compile_err_count / edit_count) * 100.0 if edit_count > 0 else None
        ),
        test_fail_episodes=test_fail_episodes,
        test_converted_episodes=test_converted_episodes,
        test_fail_to_green_rate=_safe_div(test_converted_episodes, test_fail_episodes),
        steps_to_green_values=tuple(steps_to_green_vals),
        steps_to_green_mean=_mean_or_none(steps_to_green_vals),
        steps_to_green_median=_median_or_none(steps_to_green_vals),
        verified_after_last_edit=verified_after_last_edit,
        verification_rate=verification_rate,
        read_files_in_edit_session_count=read_in_edit_cnt,
        edited_read_files_count=edited_read_cnt,
        read_to_edit_precision=read_to_edit_prec,
        input_tokens_total=in_tok_total,
        cache_read_tokens_total=cache_r_total,
        cache_write_tokens_total=cache_w_total,
        output_tokens_total=out_tok_total,
        reasoning_tokens_total=reas_tok_total,
        total_prompt_tokens=prompt_tok_total,
        total_tokens=total_tok,
        cache_hit_ratio=cache_hit_ratio,
        model_time_ms=model_ms,
        tool_time_ms=tool_ms,
        gap_time_ms=gap_ms,
        active_turn_wall_ms=active_wall_ms,
        model_time_share=model_share,
        tool_time_share=tool_share,
        gap_time_share=gap_share,
        truncation_checked_tool_call_count=trunc_checked_tc,
        truncated_tool_call_count=truncated_tc,
        truncation_rate=_safe_div(truncated_tc, trunc_checked_tc),
        summary_compaction_count=summary_comp_count,
        summary_compaction_failures=summary_comp_failures,
        compaction_failure_rate=_safe_div(summary_comp_failures, summary_comp_count),
        compaction_tokens_before_sum=comp_before_sum,
        compaction_tokens_after_sum=comp_after_sum,
        compaction_ratio=_safe_div(comp_after_sum, comp_before_sum),
        prune_compaction_count=prune_comp_count,
        post_compaction_read_count=post_comp_reads,
        post_compaction_reread_count=post_comp_rereads,
        post_compaction_reread_rate=_safe_div(post_comp_rereads, post_comp_reads),
        retry_count=retry_count,
        retried_model_call_count=retried_mc_count,
        retry_rate=_safe_div(retried_mc_count, mc_count),
        timeout_turn_count=timeout_turn_cnt,
        timeout_rate=_safe_div(timeout_turn_cnt, turn_count),
        context_overflow_turn_count=overflow_turn_cnt,
        context_overflow_rate=_safe_div(overflow_turn_cnt, turn_count),
        aborted_turn_count=aborted_turn_cnt,
        abort_rate=_safe_div(aborted_turn_cnt, turn_count),
    )


def aggregate_trajectory_metrics(
    items: Sequence[TrajectoryMetrics],
    *,
    source: str | None = None,
) -> AggregateMetrics:
    """Aggregate per-trajectory metrics into corpus-level `AggregateMetrics`."""
    resolved_source = source or (items[0].source if items else "unknown")
    if not items:
        return AggregateMetrics(source=resolved_source, trajectory_count=0)

    turn_cnt = sum(m.turn_count for m in items)
    mc_cnt = sum(m.model_call_count for m in items)
    tc_cnt = sum(m.tool_call_count for m in items)
    cmd_cnt = sum(m.command_count for m in items)
    edit_cnt = sum(m.edit_count for m in items)
    edit_sess_cnt = sum(1 for m in items if m.verified_after_last_edit is not None)
    verified_sess_cnt = sum(1 for m in items if m.verified_after_last_edit is True)

    mc_per_turn_vals = [
        m.model_calls_per_turn for m in items if m.model_calls_per_turn is not None
    ]
    tc_per_turn_vals = [
        m.tool_calls_per_turn for m in items if m.tool_calls_per_turn is not None
    ]

    near_dup_cnt = sum(m.near_duplicate_command_count for m in items)
    rep_groups = sum(m.repeated_failure_groups for m in items)
    rep_calls = sum(m.repeated_failure_calls for m in items)
    all_escape = [lat for m in items for lat in m.escape_latencies]

    schema_tc = sum(m.schema_checked_tool_call_count for m in items)
    unk_tc = sum(m.unknown_tool_call_count for m in items)
    arg_err_tc = sum(m.arg_validation_error_count for m in items)

    comp_err_cnt = sum(m.compile_error_count for m in items)
    fail_ep = sum(m.test_fail_episodes for m in items)
    conv_ep = sum(m.test_converted_episodes for m in items)
    all_stg = [s for m in items for s in m.steps_to_green_values]

    read_in_edit = sum(m.read_files_in_edit_session_count for m in items)
    edited_read = sum(m.edited_read_files_count for m in items)

    in_tok = sum(m.input_tokens_total for m in items)
    cr_tok = sum(m.cache_read_tokens_total for m in items)
    cw_tok = sum(m.cache_write_tokens_total for m in items)
    out_tok = sum(m.output_tokens_total for m in items)
    reas_tok = sum(m.reasoning_tokens_total for m in items)
    prompt_tok = in_tok + cr_tok + cw_tok
    tot_tok = prompt_tok + out_tok

    mod_ms = sum(m.model_time_ms for m in items)
    tl_ms = sum(m.tool_time_ms for m in items)
    gp_ms = sum(m.gap_time_ms for m in items)
    wall_ms = mod_ms + tl_ms + gp_ms

    trunc_chk = sum(m.truncation_checked_tool_call_count for m in items)
    trunc_cnt = sum(m.truncated_tool_call_count for m in items)

    sum_comp_cnt = sum(m.summary_compaction_count for m in items)
    sum_comp_fail = sum(m.summary_compaction_failures for m in items)
    comp_bef = sum(m.compaction_tokens_before_sum for m in items)
    comp_aft = sum(m.compaction_tokens_after_sum for m in items)
    prune_cnt = sum(m.prune_compaction_count for m in items)

    pc_reads = sum(m.post_compaction_read_count for m in items)
    pc_rereads = sum(m.post_compaction_reread_count for m in items)

    retries = sum(m.retry_count for m in items)
    retried_mc = sum(m.retried_model_call_count for m in items)
    to_turns = sum(m.timeout_turn_count for m in items)
    ovf_turns = sum(m.context_overflow_turn_count for m in items)
    abt_turns = sum(m.aborted_turn_count for m in items)

    return AggregateMetrics(
        source=resolved_source,
        trajectory_count=len(items),
        metric_version=METRIC_VERSION,
        turn_count=turn_cnt,
        model_call_count=mc_cnt,
        tool_call_count=tc_cnt,
        command_count=cmd_cnt,
        edit_count=edit_cnt,
        edit_session_count=edit_sess_cnt,
        model_calls_per_turn=_safe_div(mc_cnt, turn_cnt),
        model_calls_per_turn_p50=_percentile_or_none(mc_per_turn_vals, 50),
        model_calls_per_turn_p90=_percentile_or_none(mc_per_turn_vals, 90),
        tool_calls_per_turn=_safe_div(tc_cnt, turn_cnt),
        tool_calls_per_turn_p50=_percentile_or_none(tc_per_turn_vals, 50),
        tool_calls_per_turn_p90=_percentile_or_none(tc_per_turn_vals, 90),
        near_duplicate_command_count=near_dup_cnt,
        near_duplicate_command_rate=_safe_div(near_dup_cnt, cmd_cnt),
        repeated_failure_groups=rep_groups,
        repeated_failure_calls=rep_calls,
        escape_latency_steps_mean=_mean_or_none(all_escape),
        escape_latency_steps_median=_median_or_none(all_escape),
        escape_latency_steps_p90=_percentile_or_none(all_escape, 90),
        unknown_tool_call_count=unk_tc,
        unknown_tool_call_rate=_safe_div(unk_tc, schema_tc),
        arg_validation_error_count=arg_err_tc,
        arg_validation_error_rate=_safe_div(arg_err_tc, tc_cnt),
        compile_error_count=comp_err_cnt,
        compile_errors_per_100_edits=(
            (comp_err_cnt / edit_cnt) * 100.0 if edit_cnt > 0 else None
        ),
        test_fail_episodes=fail_ep,
        test_converted_episodes=conv_ep,
        test_fail_to_green_rate=_safe_div(conv_ep, fail_ep),
        steps_to_green_mean=_mean_or_none(all_stg),
        steps_to_green_median=_median_or_none(all_stg),
        steps_to_green_p90=_percentile_or_none(all_stg, 90),
        verified_edit_session_count=verified_sess_cnt,
        verification_rate=_safe_div(verified_sess_cnt, edit_sess_cnt),
        read_files_in_edit_session_count=read_in_edit,
        edited_read_files_count=edited_read,
        read_to_edit_precision=_safe_div(edited_read, read_in_edit),
        input_tokens_total=in_tok,
        cache_read_tokens_total=cr_tok,
        cache_write_tokens_total=cw_tok,
        output_tokens_total=out_tok,
        reasoning_tokens_total=reas_tok,
        total_prompt_tokens=prompt_tok,
        total_tokens=tot_tok,
        cache_hit_ratio=_safe_div(cr_tok, prompt_tok),
        model_time_ms=mod_ms,
        tool_time_ms=tl_ms,
        gap_time_ms=gp_ms,
        active_turn_wall_ms=wall_ms,
        model_time_share=_safe_div(mod_ms, wall_ms),
        tool_time_share=_safe_div(tl_ms, wall_ms),
        gap_time_share=_safe_div(gp_ms, wall_ms),
        truncated_tool_call_count=trunc_cnt,
        truncation_checked_tool_call_count=trunc_chk,
        truncation_rate=_safe_div(trunc_cnt, trunc_chk),
        summary_compaction_count=sum_comp_cnt,
        summary_compaction_failures=sum_comp_fail,
        compaction_failure_rate=_safe_div(sum_comp_fail, sum_comp_cnt),
        compaction_tokens_before_sum=comp_bef,
        compaction_tokens_after_sum=comp_aft,
        compaction_ratio=_safe_div(comp_aft, comp_bef),
        prune_compaction_count=prune_cnt,
        post_compaction_read_count=pc_reads,
        post_compaction_reread_count=pc_rereads,
        post_compaction_reread_rate=_safe_div(pc_rereads, pc_reads),
        retry_count=retries,
        retried_model_call_count=retried_mc,
        retry_rate=_safe_div(retried_mc, mc_cnt),
        timeout_turn_count=to_turns,
        timeout_rate=_safe_div(to_turns, turn_cnt),
        context_overflow_turn_count=ovf_turns,
        context_overflow_rate=_safe_div(ovf_turns, turn_cnt),
        aborted_turn_count=abt_turns,
        abort_rate=_safe_div(abt_turns, turn_cnt),
    )
