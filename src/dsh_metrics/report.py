"""Self-contained HTML comparison report for DSH vs Claude Code passive metrics (M6)."""

from __future__ import annotations

from datetime import datetime, timezone
import html
from pathlib import Path

from dsh_metrics.classifiers import CommandKind
from dsh_metrics.metrics import METRIC_VERSION, AggregateMetrics, _percentile_or_none
from dsh_metrics.redact import redact_secrets
from dsh_metrics.store import MetricsStore, SourceSupplementalStats


def _esc(value: str) -> str:
    return html.escape(redact_secrets(value), quote=True)


def _fmt_int(value: int | None) -> str:
    if value is None:
        return '<span class="na">N/A</span>'
    return f"{value:,}"


def _fmt_float(value: float | None, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return '<span class="na">N/A</span>'
    return f"{value:,.{digits}f}{_esc(suffix)}"


def _fmt_pct(
    rate: float | None,
    num: int | None = None,
    den: int | None = None,
    digits: int = 1,
) -> str:
    if rate is None:
        return '<span class="na">N/A</span>'
    pct_str = f"{rate * 100.0:.{digits}f}%"
    if num is not None and den is not None:
        return f'{pct_str} <span class="sub">({num:,} / {den:,})</span>'
    return pct_str


def _fmt_ms(ms: int | float | None) -> str:
    if ms is None:
        return '<span class="na">N/A</span>'
    val = float(ms)
    if val < 1000.0:
        return f"{val:,.0f} ms"
    sec = val / 1000.0
    if sec < 60.0:
        return f"{sec:,.2f} s"
    mins = sec / 60.0
    if mins < 60.0:
        return f"{mins:,.1f} min"
    hrs = mins / 60.0
    return f"{hrs:,.2f} h"


def _fmt_date_range(min_ms: int | None, max_ms: int | None) -> str:
    if min_ms is None or max_ms is None:
        return '<span class="na">N/A</span>'
    d1 = datetime.fromtimestamp(min_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
    d2 = datetime.fromtimestamp(max_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
    return d1 if d1 == d2 else f"{d1} &rarr; {d2}"


def _fmt_tokens_compact(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,}"


def _render_time_bar(agg: AggregateMetrics, label: str) -> str:
    if agg.active_turn_wall_ms <= 0:
        return (
            f'<div class="bar-group"><div class="bar-label">{_esc(label)}</div>'
            '<div class="bar-empty">No timed active turns</div></div>'
        )
    m_pct = max(0.0, min(100.0, (agg.model_time_share or 0.0) * 100.0))
    t_pct = max(0.0, min(100.0, (agg.tool_time_share or 0.0) * 100.0))
    g_pct = max(0.0, min(100.0, (agg.gap_time_share or 0.0) * 100.0))
    total = m_pct + t_pct + g_pct
    if total > 0:
        scale = 100.0 / total
        mw, tw, gw = m_pct * scale, t_pct * scale, g_pct * scale
    else:
        mw = tw = gw = 0.0

    return f"""
    <div class="bar-group">
      <div class="bar-header">
        <strong>{_esc(label)}</strong>
        <span class="sub">Active turn wall time: {_fmt_ms(agg.active_turn_wall_ms)}</span>
      </div>
      <div class="stacked-bar" role="img" aria-label="{_esc(label)} active turn time split">
        <div class="seg seg-model" tabindex="0" style="width:{mw:.2f}%" title="Model: {m_pct:.1f}%"></div>
        <div class="seg seg-tool" tabindex="0" style="width:{tw:.2f}%" title="Tool: {t_pct:.1f}%"></div>
        <div class="seg seg-gap" tabindex="0" style="width:{gw:.2f}%" title="Gap / Overhead: {g_pct:.1f}%"></div>
      </div>
      <div class="bar-legend">
        <span><i class="dot seg-model"></i>Model {_fmt_ms(agg.model_time_ms)} ({m_pct:.1f}%)</span>
        <span><i class="dot seg-tool"></i>Tool {_fmt_ms(agg.tool_time_ms)} ({t_pct:.1f}%)</span>
        <span><i class="dot seg-gap"></i>Gap {_fmt_ms(agg.gap_time_ms)} ({g_pct:.1f}%)</span>
      </div>
    </div>
    """


def _render_row(
    metric_label: str,
    formula_note: str,
    dsh_cell: str,
    cc_cell: str,
) -> str:
    return (
        "<tr>"
        f'<td><div class="m-name">{_esc(metric_label)}</div>'
        f'<div class="m-note">{_esc(formula_note)}</div></td>'
        f'<td class="num">{dsh_cell}</td>'
        f'<td class="num">{cc_cell}</td>'
        "</tr>"
    )


def _render_models_table(title: str, swatch_cls: str, stats: SourceSupplementalStats) -> str:
    if not stats.top_models:
        rows_html = '<tr><td colspan="3" class="na">No model calls recorded</td></tr>'
    else:
        rows = []
        for m in stats.top_models[:8]:
            rows.append(
                "<tr>"
                f"<td><code>{_esc(m.route)}</code></td>"
                f'<td class="num">{m.call_count:,}</td>'
                f'<td class="num">{_fmt_tokens_compact(m.total_tokens)} '
                f'<span class="sub">({m.total_tokens:,})</span></td>'
                "</tr>"
            )
        rows_html = "\n".join(rows)
    return f"""
    <div class="card">
      <h3><i class="dot {swatch_cls}"></i>{_esc(title)}</h3>
      <table class="compact-table">
        <thead>
          <tr><th>Provider / Model</th><th class="num">Model Calls</th><th class="num">Total Tokens</th></tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
    """


def _render_tool_kinds_rows(
    dsh_stats: SourceSupplementalStats,
    cc_stats: SourceSupplementalStats,
    dsh_total: int,
    cc_total: int,
) -> str:
    kinds: tuple[tuple[CommandKind, str], ...] = (
        ("read", "Read (file inspection / cat / head / sed -n)"),
        ("search", "Search (grep / rg / glob / find)"),
        ("test", "Test (pytest / go test / npm test / vitest)"),
        ("build", "Build (go build / tsc / cargo build / npm build)"),
        ("lint", "Lint / Format (ruff / eslint / golangci-lint / make lint)"),
        ("vcs", "VCS (git / gh)"),
        ("other", "Other (edit / write / subagent / general shell)"),
    )
    rows: list[str] = []
    for k, desc in kinds:
        dc = dsh_stats.tool_kind_counts.get(k, 0)
        cc = cc_stats.tool_kind_counts.get(k, 0)
        dr = (dc / dsh_total) if dsh_total > 0 else None
        cr = (cc / cc_total) if cc_total > 0 else None
        d_cell = f"{dc:,}" + (
            f' <span class="sub">({dr * 100:.1f}%)</span>' if dr is not None else ""
        )
        c_cell = f"{cc:,}" + (
            f' <span class="sub">({cr * 100:.1f}%)</span>' if cr is not None else ""
        )
        rows.append(_render_row(f"Kind: {k}", desc, d_cell, c_cell))
    return "\n".join(rows)


def render_comparison_html(
    store: MetricsStore,
    *,
    metric_version: int = METRIC_VERSION,
) -> str:
    """Render a self-contained offline HTML report comparing DSH vs Claude Code."""
    dsh = store.get_aggregate_metrics("dsh", metric_version=metric_version)
    cc = store.get_aggregate_metrics("claude-code", metric_version=metric_version)
    dsh_sup = store.get_supplemental_stats("dsh")
    cc_sup = store.get_supplemental_stats("claude-code")

    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    dsh_trajs = store.load_trajectory_metrics("dsh", metric_version=metric_version)
    cc_trajs = store.load_trajectory_metrics(
        "claude-code", metric_version=metric_version
    )
    dsh_schema_checked = sum(m.schema_checked_tool_call_count for m in dsh_trajs)
    cc_schema_checked = sum(m.schema_checked_tool_call_count for m in cc_trajs)
    dsh_r2e_med = _percentile_or_none(
        [m.read_to_edit_precision for m in dsh_trajs if m.read_to_edit_precision is not None],
        50.0,
    )
    cc_r2e_med = _percentile_or_none(
        [m.read_to_edit_precision for m in cc_trajs if m.read_to_edit_precision is not None],
        50.0,
    )
    dsh_cache_med = _percentile_or_none(
        [m.cache_hit_ratio for m in dsh_trajs if m.cache_hit_ratio is not None],
        50.0,
    )
    cc_cache_med = _percentile_or_none(
        [m.cache_hit_ratio for m in cc_trajs if m.cache_hit_ratio is not None],
        50.0,
    )
    dsh_comp_med = _percentile_or_none(
        [m.compaction_ratio for m in dsh_trajs if m.compaction_ratio is not None],
        50.0,
    )
    cc_comp_med = _percentile_or_none(
        [m.compaction_ratio for m in cc_trajs if m.compaction_ratio is not None],
        50.0,
    )

    cat_a_rows = "\n".join(
        [
            _render_row(
                "Model calls per turn",
                "model_call_count / turn_count (micro-avg; session p50 / p90)",
                f"{_fmt_float(dsh.model_calls_per_turn)} "
                f'<span class="sub">(p50 {_fmt_float(dsh.model_calls_per_turn_p50)}, '
                f"p90 {_fmt_float(dsh.model_calls_per_turn_p90)})</span>",
                f"{_fmt_float(cc.model_calls_per_turn)} "
                f'<span class="sub">(p50 {_fmt_float(cc.model_calls_per_turn_p50)}, '
                f"p90 {_fmt_float(cc.model_calls_per_turn_p90)})</span>",
            ),
            _render_row(
                "Tool calls per turn",
                "tool_call_count / turn_count (micro-avg; session p50 / p90)",
                f"{_fmt_float(dsh.tool_calls_per_turn)} "
                f'<span class="sub">(p50 {_fmt_float(dsh.tool_calls_per_turn_p50)}, '
                f"p90 {_fmt_float(dsh.tool_calls_per_turn_p90)})</span>",
                f"{_fmt_float(cc.tool_calls_per_turn)} "
                f'<span class="sub">(p50 {_fmt_float(cc.tool_calls_per_turn_p50)}, '
                f"p90 {_fmt_float(cc.tool_calls_per_turn_p90)})</span>",
            ),
            _render_row(
                "Near-duplicate shell command rate",
                "Repeated normalized command fingerprint within a trajectory / command_count",
                _fmt_pct(
                    dsh.near_duplicate_command_rate,
                    dsh.near_duplicate_command_count,
                    dsh.command_count,
                ),
                _fmt_pct(
                    cc.near_duplicate_command_rate,
                    cc.near_duplicate_command_count,
                    cc.command_count,
                ),
            ),
            _render_row(
                "Repeated failure groups (calls)",
                "Distinct (fingerprint, error_signature) pairs seen >= 2 times in a trajectory",
                f"{dsh.repeated_failure_groups:,} groups "
                f'<span class="sub">({dsh.repeated_failure_calls:,} calls)</span>',
                f"{cc.repeated_failure_groups:,} groups "
                f'<span class="sub">({cc.repeated_failure_calls:,} calls)</span>',
            ),
            _render_row(
                "Failure-loop escape latency (steps)",
                "Steps from first repeated failure to first step strictly after the group's last failure",
                f"mean {_fmt_float(dsh.escape_latency_steps_mean)} "
                f'<span class="sub">(med {_fmt_float(dsh.escape_latency_steps_median)}, '
                f"p90 {_fmt_float(dsh.escape_latency_steps_p90)})</span>",
                f"mean {_fmt_float(cc.escape_latency_steps_mean)} "
                f'<span class="sub">(med {_fmt_float(cc.escape_latency_steps_median)}, '
                f"p90 {_fmt_float(cc.escape_latency_steps_p90)})</span>",
            ),
            _render_row(
                "Unknown-tool call rate",
                "Tool calls whose name is absent from non-empty available_tools",
                _fmt_pct(
                    dsh.unknown_tool_call_rate,
                    dsh.unknown_tool_call_count,
                    dsh_schema_checked,
                    digits=2,
                ),
                _fmt_pct(
                    cc.unknown_tool_call_rate,
                    cc.unknown_tool_call_count,
                    cc_schema_checked,
                    digits=2,
                ),
            ),
            _render_row(
                "Argument-validation error rate",
                "Schema / JSON tool argument validation failures / tool_call_count",
                _fmt_pct(
                    dsh.arg_validation_error_rate,
                    dsh.arg_validation_error_count,
                    dsh.tool_call_count,
                    digits=2,
                ),
                _fmt_pct(
                    cc.arg_validation_error_rate,
                    cc.arg_validation_error_count,
                    cc.tool_call_count,
                    digits=2,
                ),
            ),
        ]
    )

    cat_b_rows = "\n".join(
        [
            _render_row(
                "Compile-class errors per 100 edits",
                "(compile_error_count / edit_count) * 100",
                f"{_fmt_float(dsh.compile_errors_per_100_edits)} "
                f'<span class="sub">({dsh.compile_error_count:,} errors / {dsh.edit_count:,} edits)</span>',
                f"{_fmt_float(cc.compile_errors_per_100_edits)} "
                f'<span class="sub">({cc.compile_error_count:,} errors / {cc.edit_count:,} edits)</span>',
            ),
            _render_row(
                "Test fail -> green conversion (<= 10 steps)",
                "Failing test episodes followed by a passing test run within 10 steps",
                _fmt_pct(
                    dsh.test_fail_to_green_rate,
                    dsh.test_converted_episodes,
                    dsh.test_fail_episodes,
                ),
                _fmt_pct(
                    cc.test_fail_to_green_rate,
                    cc.test_converted_episodes,
                    cc.test_fail_episodes,
                ),
            ),
            _render_row(
                "Steps to green (converted test episodes)",
                "s_green - s_fail across converted test failure episodes",
                f"mean {_fmt_float(dsh.steps_to_green_mean)} "
                f'<span class="sub">(med {_fmt_float(dsh.steps_to_green_median)}, '
                f"p90 {_fmt_float(dsh.steps_to_green_p90)})</span>",
                f"mean {_fmt_float(cc.steps_to_green_mean)} "
                f'<span class="sub">(med {_fmt_float(cc.steps_to_green_median)}, '
                f"p90 {_fmt_float(cc.steps_to_green_p90)})</span>",
            ),
            _render_row(
                "Post-edit verification rate",
                "Editing sessions running a test or build command at or after the last edit",
                _fmt_pct(
                    dsh.verification_rate,
                    dsh.verified_edit_session_count,
                    dsh.edit_session_count,
                ),
                _fmt_pct(
                    cc.verification_rate,
                    cc.verified_edit_session_count,
                    cc.edit_session_count,
                ),
            ),
            _render_row(
                "Read -> edit precision proxy",
                "|distinct edited & read files| / |distinct read files| in editing sessions (session med)",
                f"{_fmt_pct(dsh.read_to_edit_precision, dsh.edited_read_files_count, dsh.read_files_in_edit_session_count)} "
                f'<span class="sub">(med {_fmt_pct(dsh_r2e_med)})</span>',
                f"{_fmt_pct(cc.read_to_edit_precision, cc.edited_read_files_count, cc.read_files_in_edit_session_count)} "
                f'<span class="sub">(med {_fmt_pct(cc_r2e_med)})</span>',
            ),
        ]
    )

    cat_c_rows = "\n".join(
        [
            _render_row(
                "Total tokens (input + cache_read + cache_write + output)",
                "Sum across deduplicated ModelCalls",
                f"{_fmt_tokens_compact(dsh.total_tokens)} "
                f'<span class="sub">({dsh.total_tokens:,})</span>',
                f"{_fmt_tokens_compact(cc.total_tokens)} "
                f'<span class="sub">({cc.total_tokens:,})</span>',
            ),
            _render_row(
                "Uncached input tokens",
                "Sum of ModelCall.input_tokens",
                _fmt_int(dsh.input_tokens_total),
                _fmt_int(cc.input_tokens_total),
            ),
            _render_row(
                "Cache read tokens",
                "Sum of ModelCall.cache_read_tokens",
                _fmt_int(dsh.cache_read_tokens_total),
                _fmt_int(cc.cache_read_tokens_total),
            ),
            _render_row(
                "Cache write / creation tokens",
                "Sum of ModelCall.cache_write_tokens",
                _fmt_int(dsh.cache_write_tokens_total),
                _fmt_int(cc.cache_write_tokens_total),
            ),
            _render_row(
                "Output tokens (reasoning tokens)",
                "Sum of ModelCall.output_tokens and reasoning_tokens",
                f"{_fmt_int(dsh.output_tokens_total)} "
                f'<span class="sub">(reasoning: {_fmt_int(dsh.reasoning_tokens_total)})</span>',
                f"{_fmt_int(cc.output_tokens_total)} "
                f'<span class="sub">(reasoning: {_fmt_int(cc.reasoning_tokens_total)})</span>',
            ),
            _render_row(
                "Prompt cache hit ratio",
                "cache_read / (input + cache_read + cache_write) (session med)",
                f"{_fmt_pct(dsh.cache_hit_ratio)} "
                f'<span class="sub">(med {_fmt_pct(dsh_cache_med)})</span>',
                f"{_fmt_pct(cc.cache_hit_ratio)} "
                f'<span class="sub">(med {_fmt_pct(cc_cache_med)})</span>',
            ),
            _render_row(
                "Model call latency (p50 / p90)",
                "ended_at - started_at per ModelCall",
                f"p50 {_fmt_ms(dsh_sup.model_latency_ms_p50)} "
                f'<span class="sub">(p90 {_fmt_ms(dsh_sup.model_latency_ms_p90)})</span>',
                f"p50 {_fmt_ms(cc_sup.model_latency_ms_p50)} "
                f'<span class="sub">(p90 {_fmt_ms(cc_sup.model_latency_ms_p90)})</span>',
            ),
            _render_row(
                "Time to first token — TTFT (p50 / p90)",
                "first_token_at - started_at when streaming chunks are recorded",
                f"p50 {_fmt_ms(dsh_sup.ttft_ms_p50)} "
                f'<span class="sub">(p90 {_fmt_ms(dsh_sup.ttft_ms_p90)})</span>',
                f"p50 {_fmt_ms(cc_sup.ttft_ms_p50)} "
                f'<span class="sub">(p90 {_fmt_ms(cc_sup.ttft_ms_p90)})</span>',
            ),
            _render_row(
                "Active turn time split (Model / Tool / Gap)",
                "Interval-merged active turn time excluding inter-turn user idle breaks",
                f"{_fmt_pct(dsh.model_time_share)} / {_fmt_pct(dsh.tool_time_share)} / {_fmt_pct(dsh.gap_time_share)}",
                f"{_fmt_pct(cc.model_time_share)} / {_fmt_pct(cc.tool_time_share)} / {_fmt_pct(cc.gap_time_share)}",
            ),
        ]
    )

    cat_d_rows = "\n".join(
        [
            _render_row(
                "Tool output truncation rate",
                "truncated_tool_call_count / truncation_checked_tool_call_count",
                _fmt_pct(
                    dsh.truncation_rate,
                    dsh.truncated_tool_call_count,
                    dsh.truncation_checked_tool_call_count,
                    digits=2,
                ),
                _fmt_pct(
                    cc.truncation_rate,
                    cc.truncated_tool_call_count,
                    cc.truncation_checked_tool_call_count,
                    digits=2,
                ),
            ),
            _render_row(
                "Summary compaction failure rate",
                "summary_compaction_failures / summary_compaction_count (excludes prune)",
                _fmt_pct(
                    dsh.compaction_failure_rate,
                    dsh.summary_compaction_failures,
                    dsh.summary_compaction_count,
                ),
                _fmt_pct(
                    cc.compaction_failure_rate,
                    cc.summary_compaction_failures,
                    cc.summary_compaction_count,
                ),
            ),
            _render_row(
                "Summary compaction token ratio (tokens_after / tokens_before)",
                "compaction_tokens_after_sum / compaction_tokens_before_sum (prune count separate)",
                f"{_fmt_float(dsh.compaction_ratio, digits=3)} "
                f'<span class="sub">(med {_fmt_float(dsh_comp_med, digits=3)}; '
                f"prune: {dsh.prune_compaction_count:,})</span>",
                f"{_fmt_float(cc.compaction_ratio, digits=3)} "
                f'<span class="sub">(med {_fmt_float(cc_comp_med, digits=3)}; '
                f"prune: {cc.prune_compaction_count:,})</span>",
            ),
            _render_row(
                "Post-compaction file re-read rate",
                "Reads after first summary compaction targeting files already read pre-compaction",
                _fmt_pct(
                    dsh.post_compaction_reread_rate,
                    dsh.post_compaction_reread_count,
                    dsh.post_compaction_read_count,
                ),
                _fmt_pct(
                    cc.post_compaction_reread_rate,
                    cc.post_compaction_reread_count,
                    cc.post_compaction_read_count,
                ),
            ),
            _render_row(
                "LLM retry rate (calls with >= 1 retry)",
                "retried_model_call_count / model_call_count (total retries in parens)",
                f"{_fmt_pct(dsh.retry_rate, dsh.retried_model_call_count, dsh.model_call_count, digits=2)} "
                f'<span class="sub">(total retries: {dsh.retry_count:,})</span>',
                f"{_fmt_pct(cc.retry_rate, cc.retried_model_call_count, cc.model_call_count, digits=2)} "
                f'<span class="sub">(total retries: {cc.retry_count:,})</span>',
            ),
            _render_row(
                "Timeout turn rate",
                "timeout_turn_count / turn_count",
                _fmt_pct(
                    dsh.timeout_rate, dsh.timeout_turn_count, dsh.turn_count, digits=2
                ),
                _fmt_pct(
                    cc.timeout_rate, cc.timeout_turn_count, cc.turn_count, digits=2
                ),
            ),
            _render_row(
                "Context overflow turn rate",
                "context_overflow_turn_count / turn_count",
                _fmt_pct(
                    dsh.context_overflow_rate,
                    dsh.context_overflow_turn_count,
                    dsh.turn_count,
                    digits=2,
                ),
                _fmt_pct(
                    cc.context_overflow_rate,
                    cc.context_overflow_turn_count,
                    cc.turn_count,
                    digits=2,
                ),
            ),
            _render_row(
                "Turn abort / interrupt rate",
                "aborted_turn_count / turn_count",
                _fmt_pct(
                    dsh.abort_rate, dsh.aborted_turn_count, dsh.turn_count, digits=2
                ),
                _fmt_pct(
                    cc.abort_rate, cc.aborted_turn_count, cc.turn_count, digits=2
                ),
            ),
        ]
    )

    tool_kind_rows = _render_tool_kinds_rows(
        dsh_sup, cc_sup, dsh.tool_call_count, cc.tool_call_count
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>dsh-metrics — Passive Trajectory Telemetry Report (v{metric_version})</title>
  <style>
    :root {{
      color-scheme: light;
      --page-bg: #f9f9f7;
      --surface-1: #fcfcfb;
      --surface-2: #f2f1ec;
      --border: #e1e0d9;
      --text-primary: #0b0b0b;
      --text-secondary: #52514e;
      --text-muted: #898781;
      --series-1: #2a78d6;
      --series-2: #eb6834;
      --series-3: #1baf7a;
      --series-gap: #c3c2b7;
      --banner-bg: #fef7e0;
      --banner-border: #fab219;
    }}
    @media (prefers-color-scheme: dark) {{
      :root:where(:not([data-theme="light"])) {{
        color-scheme: dark;
        --page-bg: #0d0d0d;
        --surface-1: #1a1a19;
        --surface-2: #242422;
        --border: #2c2c2a;
        --text-primary: #f0efec;
        --text-secondary: #c3c2b7;
        --text-muted: #898781;
        --series-1: #3987e5;
        --series-2: #d95926;
        --series-3: #199e70;
        --series-gap: #383835;
        --banner-bg: #2a2110;
        --banner-border: #fab219;
      }}
    }}
    :root[data-theme="dark"] {{
      color-scheme: dark;
      --page-bg: #0d0d0d;
      --surface-1: #1a1a19;
      --surface-2: #242422;
      --border: #2c2c2a;
      --text-primary: #f0efec;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --series-1: #3987e5;
      --series-2: #d95926;
      --series-3: #199e70;
      --series-gap: #383835;
      --banner-bg: #2a2110;
      --banner-border: #fab219;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 28px 20px 56px;
      font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
      background: var(--page-bg);
      color: var(--text-primary);
      line-height: 1.5;
    }}
    .container {{ max-width: 1140px; margin: 0 auto; }}
    header {{ margin-bottom: 18px; }}
    h1 {{ margin: 0 0 4px; font-size: 1.65rem; letter-spacing: -0.02em; }}
    .meta {{ color: var(--text-secondary); font-size: 0.88rem; }}
    .caveat-banner {{
      background: var(--banner-bg);
      border: 1px solid var(--banner-border);
      border-left-width: 6px;
      color: var(--text-primary);
      padding: 14px 18px;
      border-radius: 8px;
      margin: 16px 0 24px;
      font-size: 0.94rem;
    }}
    .caveat-banner strong {{ display: block; font-size: 1.0rem; margin-bottom: 4px; }}
    .grid-2 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .card {{
      background: var(--surface-1);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 18px 20px;
    }}
    .card h2, .card h3 {{ margin: 0 0 12px; font-size: 1.06rem; color: var(--text-primary); }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-top: 10px;
    }}
    .kpi {{
      background: var(--surface-2);
      border-radius: 6px;
      padding: 9px 11px;
    }}
    .kpi .k-label {{ font-size: 0.74rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.02em; }}
    .kpi .k-val {{ font-size: 1.08rem; font-weight: 600; }}
    section {{ margin-bottom: 26px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--surface-1);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      font-size: 0.91rem;
    }}
    .compact-table {{ border: none; }}
    th, td {{
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
      text-align: left;
    }}
    th {{
      background: var(--surface-2);
      font-weight: 600;
      font-size: 0.81rem;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      color: var(--text-secondary);
    }}
    tr:last-child td {{ border-bottom: none; }}
    td.num, th.num {{
      text-align: right;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}
    .m-name {{ font-weight: 600; color: var(--text-primary); }}
    .m-note {{ font-size: 0.79rem; color: var(--text-secondary); margin-top: 2px; }}
    .sub {{ color: var(--text-secondary); font-size: 0.82rem; font-weight: 400; }}
    .na {{ color: var(--text-muted); font-style: italic; }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.83rem;
      background: var(--surface-2);
      padding: 2px 5px;
      border-radius: 4px;
    }}
    .bar-group {{ margin-bottom: 14px; }}
    .bar-group:last-child {{ margin-bottom: 0; }}
    .bar-header {{
      display: flex;
      justify-content: space-between;
      font-size: 0.88rem;
      margin-bottom: 6px;
    }}
    .stacked-bar {{
      display: flex;
      gap: 2px;
      height: 16px;
      border-radius: 4px;
      overflow: hidden;
      background: var(--surface-1);
    }}
    .seg {{ height: 100%; border-radius: 2px; }}
    .seg-model {{ background: var(--series-1); }}
    .seg-tool {{ background: var(--series-2); }}
    .seg-cc {{ background: var(--series-3); }}
    .seg-gap {{ background: var(--series-gap); }}
    .bar-legend {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      font-size: 0.81rem;
      color: var(--text-secondary);
      margin-top: 6px;
    }}
    .dot {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 2px;
      margin-right: 6px;
      vertical-align: middle;
    }}
    .bar-empty {{ font-size: 0.85rem; color: var(--text-muted); font-style: italic; }}
    footer {{
      margin-top: 32px;
      color: var(--text-secondary);
      font-size: 0.82rem;
      text-align: center;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>DSH vs Claude Code — Offline Trajectory Telemetry</h1>
      <div class="meta">Generated {_esc(generated_utc)} &middot; metric_version={metric_version} &middot; Deduplicated SQLite store</div>
    </header>

    <div class="caveat-banner" role="note">
      <strong>&#9888; Observational Caveat — Passive Data, Tasks Differ, Not a Fair Comparison</strong>
      These metrics are computed from passively collected local sessions where DSH and Claude Code
      were used on different tasks, repositories, prompts, session lengths, and underlying models.
      Treat every figure below as an observational harness baseline rather than a controlled
      head-to-head benchmark.
    </div>

    <div class="grid-2">
      <div class="card">
        <h2><i class="dot seg-model"></i>DeepSeek Harness (DSH)</h2>
        <div class="meta">Window: {_fmt_date_range(dsh_sup.earliest_started_at, dsh_sup.latest_started_at)} &middot; Files: {dsh_sup.files_ingested:,} &middot; Parse failures: {dsh_sup.parse_failures:,}</div>
        <div class="kpi-grid">
          <div class="kpi"><div class="k-label">Trajectories</div><div class="k-val">{dsh.trajectory_count:,}</div></div>
          <div class="kpi"><div class="k-label">Turns</div><div class="k-val">{dsh.turn_count:,}</div></div>
          <div class="kpi"><div class="k-label">Model Calls</div><div class="k-val">{dsh.model_call_count:,}</div></div>
          <div class="kpi"><div class="k-label">Tool Calls</div><div class="k-val">{dsh.tool_call_count:,}</div></div>
          <div class="kpi"><div class="k-label">Edits</div><div class="k-val">{dsh.edit_count:,}</div></div>
          <div class="kpi"><div class="k-label">Cache Hit</div><div class="k-val">{_fmt_pct(dsh.cache_hit_ratio)}</div></div>
        </div>
      </div>

      <div class="card">
        <h2><i class="dot seg-cc"></i>Claude Code</h2>
        <div class="meta">Window: {_fmt_date_range(cc_sup.earliest_started_at, cc_sup.latest_started_at)} &middot; Files: {cc_sup.files_ingested:,} &middot; Parse failures: {cc_sup.parse_failures:,}</div>
        <div class="kpi-grid">
          <div class="kpi"><div class="k-label">Trajectories</div><div class="k-val">{cc.trajectory_count:,}</div></div>
          <div class="kpi"><div class="k-label">Turns</div><div class="k-val">{cc.turn_count:,}</div></div>
          <div class="kpi"><div class="k-label">Model Calls</div><div class="k-val">{cc.model_call_count:,}</div></div>
          <div class="kpi"><div class="k-label">Tool Calls</div><div class="k-val">{cc.tool_call_count:,}</div></div>
          <div class="kpi"><div class="k-label">Edits</div><div class="k-val">{cc.edit_count:,}</div></div>
          <div class="kpi"><div class="k-label">Cache Hit</div><div class="k-val">{_fmt_pct(cc.cache_hit_ratio)}</div></div>
        </div>
      </div>
    </div>

    <section class="card">
      <h2>Active Turn Time Allocation</h2>
      {_render_time_bar(dsh, "DSH")}
      {_render_time_bar(cc, "Claude Code")}
    </section>

    <section>
      <table>
        <thead>
          <tr>
            <th>Category A — Control Loop &amp; Tool Efficiency</th>
            <th class="num">DSH</th>
            <th class="num">Claude Code</th>
          </tr>
        </thead>
        <tbody>
          {cat_a_rows}
        </tbody>
      </table>
    </section>

    <section>
      <table>
        <thead>
          <tr>
            <th>Category B — Edit Accuracy &amp; Verification</th>
            <th class="num">DSH</th>
            <th class="num">Claude Code</th>
          </tr>
        </thead>
        <tbody>
          {cat_b_rows}
        </tbody>
      </table>
    </section>

    <section>
      <table>
        <thead>
          <tr>
            <th>Category C — Tokens, Caching &amp; Latency</th>
            <th class="num">DSH</th>
            <th class="num">Claude Code</th>
          </tr>
        </thead>
        <tbody>
          {cat_c_rows}
        </tbody>
      </table>
    </section>

    <section>
      <table>
        <thead>
          <tr>
            <th>Category D — Context Management &amp; Reliability</th>
            <th class="num">DSH</th>
            <th class="num">Claude Code</th>
          </tr>
        </thead>
        <tbody>
          {cat_d_rows}
        </tbody>
      </table>
    </section>

    <section>
      <table>
        <thead>
          <tr>
            <th>Normalized Tool &amp; Shell Command Kinds</th>
            <th class="num">DSH</th>
            <th class="num">Claude Code</th>
          </tr>
        </thead>
        <tbody>
          {tool_kind_rows}
        </tbody>
      </table>
    </section>

    <div class="grid-2">
      {_render_models_table("DSH — Top Model Routes", "seg-model", dsh_sup)}
      {_render_models_table("Claude Code — Top Model Routes", "seg-cc", cc_sup)}
    </div>

    <footer>
      dsh-metrics P0 offline analyzer &middot; Zero-denominator rates render as N/A &middot;
      Secrets redacted at adapter boundary; cross-file duplicate responses and tool calls ignored via SQLite UNIQUE constraints.
    </footer>
  </div>
</body>
</html>
"""


def write_comparison_report(
    store: MetricsStore,
    out_path: str | Path,
    *,
    metric_version: int = METRIC_VERSION,
) -> Path:
    """Render and write the self-contained HTML comparison report to `out_path`."""
    dest = Path(out_path).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    html_text = render_comparison_html(store, metric_version=metric_version)
    dest.write_text(html_text, encoding="utf-8")
    return dest
