"""Transcript adapters normalizing DSH and Claude Code logs into Trajectory IR."""

from dsh_metrics.adapters.dsh import discover_dsh_files, parse_dsh_session

__all__ = [
    "discover_dsh_files",
    "parse_dsh_session",
]
