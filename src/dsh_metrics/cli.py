"""Command-line interface for dsh-metrics."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from dsh_metrics import __version__


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
