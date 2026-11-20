"""Synthetic tests for M4 shared classifiers."""

from __future__ import annotations

import pytest

from dsh_metrics.classifiers import (
    CommandKind,
    classify_command_kind,
    classify_error_kind,
    classify_tool_call_kind,
    extract_error_signature,
    extract_read_paths,
    fingerprint_command,
    fingerprint_tool_args,
    normalize_repo_path,
)
from dsh_metrics.ir import ToolCall


def test_fingerprint_command_normalizes_numbers_whitespace_quotes_and_sed_ranges() -> None:
    # Line ranges (unquoted, single-quoted, double-quoted, with quit command)
    fp1 = fingerprint_command("sed -n 10,20p src/app.py")
    fp2 = fingerprint_command("  sed   -n   '80,160p'   src/app.py  ")
    fp3 = fingerprint_command('sed -n "161,240p;241q" src/app.py')
    assert fp1 == fp2 == fp3 == "sed -n <RANGE> src/app.py"

    # Numbers and line specifiers
    assert fingerprint_command("pytest tests/test_x.py:14 --maxfail=1") == (
        fingerprint_command("pytest tests/test_x.py:98 --maxfail=3")
    )
    assert fingerprint_command("head -20 README.md") == fingerprint_command(
        "head -100 README.md"
    )

    # Quoted strings
    assert fingerprint_command('pytest -k "test_alpha"') == fingerprint_command(
        "pytest -k 'test_beta and not slow'"
    )

    #Identifiers like python3 / utf-8 are not mangled, while secrets are redacted
    fp_sec = fingerprint_command(
        "python3 -m pytest --token=sk-proj-SecretToken123456"
    )
    assert "python3" in fp_sec
    assert "SecretToken" not in fp_sec


@pytest.mark.parametrize(
    ("cmd", "expected"),
    [
        # Go
        ("go test ./...", "test"),
        ("CGO_ENABLED=0 go build ./cmd/server", "build"),
        ("go vet ./...", "lint"),
        ("golangci-lint run", "lint"),
        # Python
        ("pytest -v", "test"),
        ("cd /workspace/app && .venv/bin/python3.12 -m unittest discover", "test"),
        ("uv run pytest -q | tail -n 20", "test"),
        ("python -m build", "build"),
        ("uv sync", "build"),
        ("ruff check src/", "lint"),
        ("python3 -m mypy src/", "lint"),
        ("python3 scripts/seed.py", "other"),
        # Node
        ("npm test", "test"),
        ("npx vitest run", "test"),
        ("pnpm run test:unit", "test"),
        ("npm run build", "build"),
        ("npx tsc --noEmit", "build"),
        ("eslint src/", "lint"),
        ("npm run lint:check", "lint"),
        ("node index.js", "other"),
        # Make
        ("make", "build"),
        ("make -j4 build", "build"),
        ("make test", "test"),
        ("make check", "test"),
        ("make lint", "lint"),
        ("make fmt", "lint"),
        # Read / Search / VCS / Other
        ("sed -n '10,40p' src/main.go", "read"),
        ("sed -i 's/foo/bar/g' src/main.go", "other"),
        ("nl -ba src/main.go | sed -n '1,30p'", "read"),
        ("cat pyproject.toml", "read"),
        ("rg -n 'ModelCall' src/", "search"),
        ("find src -name '*.py' | xargs grep 'dataclass'", "search"),
        ("git status -s && git diff", "vcs"),
        ("mkdir -p dist/out", "other"),
    ],
)
def test_classify_command_kind_ecosystems(cmd: str, expected: CommandKind) -> None:
    assert classify_command_kind(cmd) == expected


def test_classify_tool_call_and_extract_read_paths() -> None:
    tc_read = ToolCall(
        step=1,
        call_id="c1",
        name="Read",
        raw_args={"file_path": "/workspace/demo/src/a.py", "offset": 1, "limit": 50},
    )
    assert classify_tool_call_kind(tc_read) == "read"
    assert extract_read_paths(tc_read, cwd="/workspace/demo") == ("src/a.py",)
    assert fingerprint_tool_args("Read", {"file_path": "src/a.py", "offset": 10}) == (
        "read:src/a.py"
    )

    tc_sed = ToolCall(
        step=2,
        call_id="c2",
        name="bash",
        raw_args={"command": "sed -n '20,60p' /workspace/demo/src/b.py"},
    )
    assert classify_tool_call_kind(tc_sed) == "read"
    assert extract_read_paths(tc_sed, cwd="/workspace/demo") == ("src/b.py",)

    tc_grep = ToolCall(
        step=3,
        call_id="c3",
        name="Grep",
        raw_args={"pattern": "foo", "path": "src"},
    )
    assert classify_tool_call_kind(tc_grep) == "search"
    assert normalize_repo_path("./src/dsh_metrics/../a.py") == "src/a.py"


@pytest.mark.parametrize(
    ("output", "expected_kind", "expected_prefix"),
    [
        ("pkg/calc.go:12:4: syntax error: unexpected newline", "compile", "compile:syntax error"),
        ("pkg/calc.go:19:9: undefined: computeTotal", "compile", "compile:undefined:computeTotal"),
        ("src/view.ts(42,7): error TS2304: Cannot find name 'Widget'.", "compile", "compile:error TS2304"),
        ("  File 'a.py', line 4\nSyntaxError: invalid syntax", "compile", "compile:SyntaxError"),
        ("Error: Cannot find module './missing'", "compile", "compile:cannot find"),
        ("--- FAIL: TestParser (0.02s)", "test", "test:--- FAIL:TestParser"),
        ("E       AssertionError: assert 1 == 2", "test", "test:AssertionError"),
        ("panic: runtime error: index out of range [3] with length 3", "test", "test:panic:"),
        ("FAIL\texample.com/pkg\t0.114s", "test", "test:FAIL"),
        ("PermissionError: [Errno 13] Permission denied: '/etc/shadow'", "other", "error:"),
    ],
)
def test_extract_error_signature_compile_and_test_classes(
    output: str,
    expected_kind: str,
    expected_prefix: str,
) -> None:
    sig = extract_error_signature(output)
    assert sig is not None
    assert sig.startswith(expected_prefix)
    assert classify_error_kind(sig) == expected_kind


def test_error_signature_stability_and_non_error_suppression() -> None:
    # Same TypeScript error at two shifted line numbers yields identical signature
    sig1 = extract_error_signature("src/x.ts(10,2): error TS2322: bad type")
    sig2 = extract_error_signature("src/x.ts(28,2): error TS2322: bad type")
    assert sig1 == sig2 == "compile:error TS2322"

    # Successful Read/Grep mentioning FAIL or SyntaxError returns None when is_error=False
    assert (
        extract_error_signature(
            "# Docs mentioning SyntaxError and --- FAIL", is_error=False
        )
        is None
    )
