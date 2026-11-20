"""Offline trajectory telemetry analyzer for DSH and Claude Code."""

from dsh_metrics.classifiers import (
    CommandKind,
    ErrorKind,
    classify_command_kind,
    classify_error_kind,
    classify_tool_call_kind,
    extract_error_signature,
    extract_read_paths,
    fingerprint_command,
    fingerprint_tool_args,
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
    aggregate_trajectory_metrics,
    compute_trajectory_metrics,
)
from dsh_metrics.report import render_comparison_html, write_comparison_report
from dsh_metrics.store import DEFAULT_DB_PATH, IngestResult, MetricsStore

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DEFAULT_DB_PATH",
    "METRIC_VERSION",
    "AggregateMetrics",
    "CommandKind",
    "Compaction",
    "Edit",
    "ErrorKind",
    "IngestResult",
    "MetricsStore",
    "ModelCall",
    "Outcome",
    "ToolCall",
    "Trajectory",
    "TrajectoryEvent",
    "TrajectoryMetrics",
    "TurnEnd",
    "aggregate_trajectory_metrics",
    "classify_command_kind",
    "classify_error_kind",
    "classify_tool_call_kind",
    "compute_trajectory_metrics",
    "event_timestamp",
    "extract_error_signature",
    "extract_read_paths",
    "fingerprint_command",
    "fingerprint_tool_args",
    "render_comparison_html",
    "write_comparison_report",
]
