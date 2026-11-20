"""Command-line interface for dsh-metrics (M6)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from dsh_metrics import __version__
from dsh_metrics.report import write_comparison_report
from dsh_metrics.store import DEFAULT_DB_PATH, VALID_SOURCES, MetricsStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsh-metrics",
        description="Offline trajectory analyzer for DSH and Claude Code transcripts.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite store (default: {DEFAULT_DB_PATH})",
    )

    subparsers = parser.add_subparsers(dest="command")

    db_parent = argparse.ArgumentParser(add_help=False)
    db_parent.add_argument(
        "--db",
        dest="sub_db",
        type=Path,
        default=None,
        help="Path to SQLite store (overrides default ~/.local/share/dsh-metrics/metrics.sqlite)",
    )

    ingest_p = subparsers.add_parser(
        "ingest",
        parents=[db_parent],
        help="Incrementally ingest DSH and/or Claude Code session logs into SQLite.",
    )
    ingest_p.add_argument(
        "--source",
        choices=(*VALID_SOURCES, "all"),
        default="all",
        help="Transcript source to ingest: dsh, claude-code, or all (default: all).",
    )
    ingest_p.add_argument(
        "--since",
        type=str,
        default=None,
        help="Only ingest sessions active on or after DATE (YYYY-MM-DD or ISO-8601).",
    )
    ingest_p.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Override root transcript directory (valid when --source is dsh or claude-code).",
    )
    ingest_p.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest files even if mtime and byte size match ingest_log.",
    )

    report_p = subparsers.add_parser(
        "report",
        parents=[db_parent],
        help="Render a self-contained HTML comparison report from the SQLite store.",
    )
    report_p.add_argument(
        "--out",
        type=Path,
        default=Path("report.html"),
        help="Output HTML file path (default: report.html).",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        return 0

    db_path: Path = getattr(args, "sub_db", None) or args.db

    if args.command == "ingest":
        if args.root is not None and args.source == "all":
            print(
                "error: --root requires an explicit --source (dsh or claude-code)",
                file=sys.stderr,
            )
            return 2

        sources = VALID_SOURCES if args.source == "all" else (args.source,)
        try:
            with MetricsStore(db_path) as store:
                for src in sources:
                    res = store.ingest_source(
                        src,
                        root=args.root,
                        since=args.since,
                        force=args.force,
                    )
                    print(
                        f"[{res.source}] discovered={res.files_discovered} "
                        f"ingested={res.files_ingested} "
                        f"unchanged={res.files_skipped_unchanged} "
                        f"skipped_since={res.files_skipped_since} "
                        f"trajectories={res.trajectories_stored} "
                        f"model_calls={res.model_calls_inserted} "
                        f"(dup_ignored={res.model_calls_ignored_duplicate}) "
                        f"tool_calls={res.tool_calls_inserted} "
                        f"(dup_ignored={res.tool_calls_ignored_duplicate}) "
                        f"parse_failures={res.parse_failures}"
                    )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.command == "report":
        with MetricsStore(db_path) as store:
            write_comparison_report(store, args.out)
            dsh_agg = store.get_aggregate_metrics("dsh")
            cc_agg = store.get_aggregate_metrics("claude-code")
        print(
            f"Wrote comparison report (dsh={dsh_agg.trajectory_count} trajectories, "
            f"claude-code={cc_agg.trajectory_count} trajectories)."
        )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
