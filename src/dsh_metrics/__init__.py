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

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "METRIC_VERSION",
    "AggregateMetrics",
    "CommandKind",
    "Compaction",
    "Edit",
    "ErrorKind",
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
]
