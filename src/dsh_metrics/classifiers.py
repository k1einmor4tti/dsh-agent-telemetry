"""Shared command fingerprinter, command-kind classifier, and error-signature extractor."""

from __future__ import annotations

from collections.abc import Mapping
import json
import posixpath
import re
import shlex
from typing import Any, Literal

from dsh_metrics.ir import ToolCall
from dsh_metrics.redact import redact_obj, redact_secrets

CommandKind = Literal["test", "build", "lint", "read", "search", "vcs", "other"]
ErrorKind = Literal["compile", "test", "other"]

SHELL_TOOL_NAMES: frozenset[str] = frozenset(
    {"bash", "tools__bash", "sh", "zsh", "shell"}
)
NATIVE_READ_TOOLS: frozenset[str] = frozenset(
    {"read", "tools.read", "view", "notebookread", "read_file"}
)
NATIVE_SEARCH_TOOLS: frozenset[str] = frozenset(
    {"grep", "glob", "search", "find", "ls", "tools.grep", "tools.glob", "list_dir"}
)
NATIVE_EDIT_TOOLS: frozenset[str] = frozenset(
    {
        "edit",
        "write",
        "multiedit",
        "notebookedit",
        "tools.edit",
        "tools.write",
        "str_replace_editor",
    }
)

# Line-range expressions such as `sed -n 10,20p`, `sed -n '10,20p;21q'`, `sed -n "5,$p"`
_SED_RANGE_ARG_RE = re.compile(
    r"""(?P<prefix>\bsed\s+(?:-[A-Za-z]+\s+)*)(?:['"]?\d+\s*,\s*(?:\d+|\$)[pdq]?(?:\s*;\s*\d+[pdq]?)?['"]?|['"]\d+[pq]['"])"""
)
_GENERIC_LINE_RANGE_RE = re.compile(
    r"""['"]?(?<![A-Za-z0-9_./-])\d+\s*,\s*(?:\d+|\$)[pdq]?(?:\s*;\s*\d+[pdq]?)?['"]?(?![A-Za-z0-9_./-])"""
)

# Quoted string literals (double-quoted or single-quoted, preserving escaped quotes)
_QUOTED_STR_RE = re.compile(
    r"""(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')"""
)

# Hex numbers and standalone/flag/colon numeric literals (avoids mangling utf-8, python3, sha256, v4)
_HEX_LIT_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
_FLAG_DASH_NUM_RE = re.compile(r"(?<=\s)[+-]\d+(?:\.\d+)?(?![A-Za-z0-9_])")
_STANDALONE_NUM_RE = re.compile(
    r"(?<![A-Za-z0-9_-])\d+(?:\.\d+)?(?![A-Za-z0-9_])"
)

# Wrapper commands / environment assignments preceding the real executable
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")

_KIND_PRIORITY: dict[CommandKind, int] = {
    "test": 6,
    "build": 5,
    "lint": 4,
    "vcs": 3,
    "search": 2,
    "read": 1,
    "other": 0,
}

# Compile-class error patterns from M4 spec:
# `syntax error`, `undefined:`, `error TS\d+`, `SyntaxError`, `cannot find`
_TS_ERROR_RE = re.compile(r"\berror\s+(TS\d+)\b")
_SYNTAX_ERROR_PY_JS_RE = re.compile(r"\bSyntaxError\b")
_SYNTAX_ERROR_LOWER_RE = re.compile(r"\bsyntax error\b", re.IGNORECASE)
_GO_UNDEFINED_RE = re.compile(r"\bundefined:\s*([A-Za-z0-9_.$]+)?")
_CANNOT_FIND_RE = re.compile(r"\bcannot find\b", re.IGNORECASE)

# Test-class error patterns from M4 spec:
# `FAIL`, `--- FAIL`, `AssertionError`, `panic:`
_GO_TEST_FAIL_RE = re.compile(r"---\s*FAIL(?::\s*([A-Za-z0-9_./-]+))?")
_ASSERTION_ERR_RE = re.compile(r"\bAssertionError\b")
_PANIC_RE = re.compile(r"\bpanic:\s*([^\r\n]*)")
_GENERIC_FAIL_RE = re.compile(r"\bFAIL(?:ED)?\b")

_EXIT_BANNER_RE = re.compile(
    r"(?:^Exit code\s+-?\d+\s*$|\[exit code:\s*-?\d+\]\s*$)",
    re.MULTILINE,
)
_SALIENT_ERR_LINE_RE = re.compile(
    r"(?:\b[A-Za-z0-9_]*(?:Error|Exception)\b|\bfatal:|\berror:)",
    re.IGNORECASE,
)


def fingerprint_command(command: str) -> str:
    """Normalize a shell command into a canonical redacted fingerprint.

    Normalizes:
    - secrets (redacted at boundary)
    - line ranges such as `sed -n 10,20p` or `sed -n '10,20p;21q'` -> `<RANGE>`
    - single and double quoted strings -> `<STR>`
    - numbers (integers, floats, `:line`, `=N`, `-20`, `0x...`) -> `<NUM>`
    - runs of whitespace -> single space
    """
    if not command:
        return ""
    out = redact_secrets(command)
    out = _SED_RANGE_ARG_RE.sub(r"\g<prefix><RANGE>", out)
    out = _GENERIC_LINE_RANGE_RE.sub("<RANGE>", out)
    out = _QUOTED_STR_RE.sub("<STR>", out)
    out = _HEX_LIT_RE.sub("<NUM>", out)
    out = _FLAG_DASH_NUM_RE.sub("-<NUM>", out)
    out = _STANDALONE_NUM_RE.sub("<NUM>", out)
    out = " ".join(out.split())
    return out


def _normalize_scalar_for_fp(val: Any) -> Any:
    if isinstance(val, bool) or val is None:
        return val
    if isinstance(val, (int, float)):
        return "<NUM>"
    if isinstance(val, str):
        return redact_secrets(val.strip())
    if isinstance(val, list):
        return [_normalize_scalar_for_fp(x) for x in val]
    if isinstance(val, Mapping):
        return {str(k): _normalize_scalar_for_fp(v) for k, v in sorted(val.items())}
    return str(val)


def fingerprint_tool_args(
    tool_name: str,
    parsed_args: Any = None,
    raw_args: str | None = None,
) -> str | None:
    """Compute a normalized, secret-redacted fingerprint for a ToolCall's arguments."""
    tname = (tool_name or "").strip()
    tlower = tname.lower()

    if parsed_args is None and raw_args:
        try:
            parsed_args = json.loads(raw_args)
        except json.JSONDecodeError:
            parsed_args = None

    if isinstance(parsed_args, Mapping):
        redacted_map = redact_obj(parsed_args)
        cmd = redacted_map.get("command") or redacted_map.get("cmd")
        if tlower in SHELL_TOOL_NAMES or isinstance(cmd, str):
            if isinstance(cmd, str):
                return fingerprint_command(cmd)

        path = (
            redacted_map.get("file_path")
            or redacted_map.get("path")
            or redacted_map.get("notebook_path")
        )
        if tlower in NATIVE_READ_TOOLS and isinstance(path, str):
            return f"read:{path.strip()}"
        if tlower in NATIVE_EDIT_TOOLS and isinstance(path, str):
            return f"{tlower}:{path.strip()}"
        if tlower in ("grep", "tools.grep") and isinstance(
            redacted_map.get("pattern"), str
        ):
            target = str(redacted_map.get("path") or ".").strip()
            return f"grep:<STR>:{target}"
        if tlower in ("glob", "tools.glob") and isinstance(
            redacted_map.get("pattern"), str
        ):
            target = str(redacted_map.get("path") or ".").strip()
            return f"glob:{redacted_map['pattern'].strip()}:{target}"

        norm_obj = _normalize_scalar_for_fp(redacted_map)
        return f"{tlower}:{json.dumps(norm_obj, sort_keys=True, ensure_ascii=False)}"

    if raw_args is not None:
        clean_raw = redact_secrets(raw_args.strip())
        if tlower in SHELL_TOOL_NAMES:
            return fingerprint_command(clean_raw)
        return f"{tlower}:{fingerprint_command(clean_raw)}"
    return None


def _split_compound_segments(command: str) -> list[str]:
    """Split a shell string on top-level `;`, `&&`, `||`, `\n` outside quotes."""
    segments: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    escaped = False
    i = 0
    n = len(command)

    while i < n:
        ch = command[i]
        if escaped:
            buf.append(ch)
            escaped = False
            i += 1
            continue
        if ch == "\\" and quote != "'":
            buf.append(ch)
            escaped = True
            i += 1
            continue
        if quote is not None:
            buf.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch in (";", "\n"):
            seg = "".join(buf).strip()
            if seg:
                segments.append(seg)
            buf.clear()
            i += 1
            continue
        if ch in ("&", "|") and i + 1 < n and command[i + 1] == ch:
            seg = "".join(buf).strip()
            if seg:
                segments.append(seg)
            buf.clear()
            i += 2
            continue
        buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        segments.append(tail)
    return segments


def _split_pipeline(segment: str) -> list[str]:
    """Split a single command segment on unquoted `|` pipes."""
    stages: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    escaped = False
    for ch in segment:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if ch == "\\" and quote != "'":
            buf.append(ch)
            escaped = True
            continue
        if quote is not None:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            continue
        if ch == "|":
            st = "".join(buf).strip()
            if st:
                stages.append(st)
            buf.clear()
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        stages.append(tail)
    return stages


def _tokenize_stage(stage: str) -> list[str]:
    try:
        tokens = shlex.split(stage, posix=True)
    except ValueError:
        tokens = stage.split()
    # Strip trailing redirections like `2>&1`, `>/dev/null`
    cleaned: list[str] = []
    skip_next = False
    for tok in tokens:
        if skip_next:
            skip_next = False
            continue
        if tok in (">", ">>", "<", "2>", "2>>", "&>"):
            skip_next = True
            continue
        if re.match(r"^(?:\d*>>?|\d*<|2>&1|&>).*", tok):
            continue
        cleaned.append(tok)
    return cleaned


def _unwrap_command_tokens(tokens: list[str]) -> list[str]:
    """Strip env vars and execution wrappers (`sudo`, `uv run`, `npx`, `timeout 30`, etc.)."""
    idx = 0
    n = len(tokens)
    while idx < n:
        tok = tokens[idx]
        if _ENV_ASSIGN_RE.match(tok):
            idx += 1
            continue
        base = posixpath.basename(tok)
        if base in ("env", "sudo", "nohup", "time", "command", "nice"):
            idx += 1
            while idx < n and (
                tokens[idx].startswith("-") or _ENV_ASSIGN_RE.match(tokens[idx])
            ):
                idx += 1
            continue
        if base == "timeout":
            idx += 1
            while idx < n and tokens[idx].startswith("-"):
                if tokens[idx] in ("-k", "-s", "--kill-after", "--signal"):
                    idx += 2
                else:
                    idx += 1
            if idx < n:
                idx += 1  # skip duration
            continue
        if base in ("uv", "poetry", "pipenv") and idx + 1 < n and tokens[idx + 1] == "run":
            idx += 2
            while idx < n and tokens[idx].startswith("-"):
                idx += 1
            continue
        if base == "bundle" and idx + 1 < n and tokens[idx + 1] == "exec":
            idx += 2
            continue
        if base in ("npx", "bunx"):
            idx += 1
            while idx < n and tokens[idx].startswith("-"):
                idx += 1
            continue
        if base == "pnpm" and idx + 1 < n and tokens[idx + 1] in ("exec", "dlx"):
            idx += 2
            continue
        break
    return tokens[idx:]


def _classify_npm_script(script_name: str) -> CommandKind:
    s = script_name.lower()
    if any(
        k in s
        for k in ("test", "vitest", "jest", "mocha", "ava", "playwright", "cypress", "e2e", "spec")
    ):
        return "test"
    if any(
        k in s
        for k in ("lint", "eslint", "prettier", "biome", "format", "fmt", "typecheck", "check-types", "style")
    ):
        return "lint"
    if any(k in s for k in ("build", "compile", "bundle", "tsc", "dist")):
        return "build"
    return "other"


def _classify_make_targets(args: list[str]) -> CommandKind:
    targets: list[str] = []
    idx = 0
    n = len(args)
    while idx < n:
        a = args[idx]
        if a in ("-C", "-f", "--directory", "--file"):
            idx += 2
            continue
        if a in ("-j", "--jobs"):
            if idx + 1 < n and args[idx + 1].isdigit():
                idx += 2
            else:
                idx += 1
            continue
        if a.startswith("-") or "=" in a:
            idx += 1
            continue
        targets.append(a.lower())
        idx += 1

    if not targets:
        return "build"

    best: CommandKind = "other"
    for t in targets:
        if any(
            k in t
            for k in (
                "lint",
                "style",
                "fmt",
                "format",
                "vet",
                "staticcheck",
                "typecheck",
                "mypy",
                "ruff",
                "eslint",
                "prettier",
                "flake8",
                "clippy",
            )
        ):
            cand: CommandKind = "lint"
        elif any(k in t for k in ("test", "check", "pytest", "unittest", "spec")):
            cand = "test"
        else:
            cand = "build"
        if _KIND_PRIORITY[cand] > _KIND_PRIORITY[best]:
            best = cand
    return best


def _classify_python_invocation(args: list[str]) -> CommandKind:
    idx = 0
    n = len(args)
    while idx < n:
        a = args[idx]
        if a == "-m" and idx + 1 < n:
            mod = args[idx + 1].lower()
            if mod in ("pytest", "unittest", "nose", "nose2", "tox", "nox"):
                return "test"
            if mod in (
                "ruff",
                "flake8",
                "pylint",
                "mypy",
                "pyright",
                "basedpyright",
                "black",
                "isort",
                "bandit",
                "pycodestyle",
            ):
                return "lint"
            if mod in ("build", "compileall", "py_compile", "pip", "cython"):
                return "build"
            return "other"
        if a == "-c":
            return "other"
        if a.startswith("-"):
            idx += 1
            continue
        break
    return "other"


def _classify_single_stage(stage: str) -> CommandKind:
    raw_tokens = _tokenize_stage(stage)
    tokens = _unwrap_command_tokens(raw_tokens)
    if not tokens:
        return "other"

    exe = posixpath.basename(tokens[0]).lower()
    rest = tokens[1:]

    # Version control
    if exe in ("git", "gh", "hg", "svn", "jj"):
        return "vcs"

    # Go ecosystem
    if exe == "go":
        sub = rest[0].lower() if rest else ""
        if sub == "test":
            return "test"
        if sub in ("vet", "fmt"):
            return "lint"
        if sub in ("build", "install", "generate", "get", "mod"):
            return "build"
        return "other"
    if exe in ("gofmt", "goimports", "golangci-lint", "staticcheck", "revive"):
        return "lint"

    # Python ecosystem
    if exe in ("pytest", "py.test", "tox", "nox") or exe.startswith("pytest-"):
        return "test"
    if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", exe) or exe == "pypy3":
        return _classify_python_invocation(rest)
    if exe in (
        "ruff",
        "flake8",
        "pylint",
        "mypy",
        "pyright",
        "basedpyright",
        "black",
        "isort",
        "autopep8",
        "bandit",
        "pycodestyle",
        "pydocstyle",
    ):
        return "lint"
    if exe in ("pip", "pip3", "maturin", "cython"):
        return "build"
    if exe == "uv":
        sub = rest[0].lower() if rest else ""
        if sub in ("pip", "sync", "build", "lock"):
            return "build"
        return "other"
    if exe == "poetry":
        sub = rest[0].lower() if rest else ""
        if sub in ("build", "install"):
            return "build"
        return "other"

    # Node / JS / TS ecosystem
    if exe in ("jest", "vitest", "mocha", "ava", "tap"):
        return "test"
    if exe in ("playwright", "cypress"):
        sub = rest[0].lower() if rest else ""
        if sub in ("test", "run"):
            return "test"
    if exe in ("eslint", "prettier", "oxlint", "stylelint", "tslint", "biome"):
        return "lint"
    if exe in ("tsc", "esbuild", "webpack", "rollup", "tsup", "vite"):
        if exe == "vite" and rest and rest[0].lower() not in ("build",):
            return "other"
        return "build"
    if exe == "node":
        if "--test" in rest:
            return "test"
        if "--check" in rest or "-c" in rest:
            return "build"
        return "other"
    if exe in ("npm", "pnpm", "yarn", "bun", "deno"):
        non_flag = [r for r in rest if not r.startswith("-")]
        sub = non_flag[0].lower() if non_flag else ""
        if sub in ("test", "t", "tst"):
            return "test"
        if sub in ("lint", "typecheck", "fmt", "format", "check"):
            return "lint"
        if sub in ("build", "compile", "install", "i", "ci", "add"):
            return "build"
        if sub == "run" and len(non_flag) >= 2:
            return _classify_npm_script(non_flag[1])
        if exe == "yarn" and sub:
            return _classify_npm_script(sub)
        return "other"

    # Make / C / Rust build systems
    if exe in ("make", "gmake"):
        return _classify_make_targets(rest)
    if exe == "ctest":
        return "test"
    if exe in ("cmake", "ninja", "gcc", "g++", "clang", "clang++", "rustc"):
        return "build"
    if exe == "cargo":
        sub = rest[0].lower() if rest else ""
        if sub == "test":
            return "test"
        if sub == "clippy" or sub == "fmt":
            return "lint"
        if sub in ("build", "check", "install"):
            return "build"
        return "other"

    # Search
    if exe in (
        "grep",
        "egrep",
        "fgrep",
        "rg",
        "ripgrep",
        "ag",
        "ack",
        "find",
        "fd",
        "fdfind",
        "locate",
        "ls",
        "tree",
        "which",
        "whereis",
    ):
        return "search"
    if exe == "xargs" and rest:
        return _classify_single_stage(" ".join(rest))

    # Read
    if exe in ("cat", "head", "tail", "less", "more", "nl", "bat", "wc", "stat", "file"):
        return "read"
    if exe == "sed":
        if any(a == "-i" or a.startswith("-i") for a in rest):
            return "other"
        if any(a == "-n" or (a.startswith("-") and "n" in a[1:]) for a in rest):
            return "read"
        return "other"

    return "other"


def classify_command_kind(command: str) -> CommandKind:
    """Classify a shell command into test / build / lint / read / search / vcs / other."""
    if not command or not command.strip():
        return "other"

    best: CommandKind = "other"
    for seg in _split_compound_segments(command):
        stages = _split_pipeline(seg)
        if not stages:
            continue
        # Primary action of a pipeline is its first stage (e.g., `pytest | tail -20` -> `test`);
        # if the first stage is `other` (e.g. `echo ... | ...`), check downstream search/read ONLY if non-echo.
        stage_kind = _classify_single_stage(stages[0])
        if stage_kind == "other" and len(stages) > 1:
            first_tokens = _unwrap_command_tokens(_tokenize_stage(stages[0]))
            first_exe = posixpath.basename(first_tokens[0]).lower() if first_tokens else ""
            if first_exe in ("xargs",):
                for st in stages[1:]:
                    cand = _classify_single_stage(st)
                    if _KIND_PRIORITY[cand] > _KIND_PRIORITY[stage_kind]:
                        stage_kind = cand
        if _KIND_PRIORITY[stage_kind] > _KIND_PRIORITY[best]:
            best = stage_kind
    return best


def extract_shell_command(tc: ToolCall) -> str | None:
    """Return the shell command string from a ToolCall if it is a shell invocation."""
    tlower = tc.name.strip().lower()
    if tc.raw_args:
        try:
            parsed = json.loads(tc.raw_args)
            if isinstance(parsed, Mapping):
                cmd = parsed.get("command") or parsed.get("cmd")
                if isinstance(cmd, str):
                    return cmd
        except json.JSONDecodeError:
            if tlower in SHELL_TOOL_NAMES:
                return tc.raw_args
    return None


def classify_tool_call_kind(tc: ToolCall) -> CommandKind:
    """Classify any IR ToolCall (shell or structured harness tool) into CommandKind."""
    tlower = tc.name.strip().lower()
    cmd = extract_shell_command(tc)
    if tlower in SHELL_TOOL_NAMES or cmd is not None:
        return classify_command_kind(cmd or "")
    if tlower in NATIVE_READ_TOOLS:
        return "read"
    if tlower in NATIVE_SEARCH_TOOLS:
        return "search"
    return "other"


def extract_error_signature(text: str, *, is_error: bool = True) -> str | None:
    """Extract a normalized error signature (`compile:...`, `test:...`, or `error:...`).

    Returns `None` when `is_error` is False or `text` is empty without error flag.
    """
    if not is_error:
        return None
    if not text or not text.strip():
        return "error:unknown"

    redacted = redact_secrets(text)

    # 1. Compile-class signatures (`error TS\d+`, `SyntaxError`, `syntax error`, `undefined:`, `cannot find`)
    m_ts = _TS_ERROR_RE.search(redacted)
    if m_ts:
        return f"compile:error {m_ts.group(1)}"

    if _SYNTAX_ERROR_PY_JS_RE.search(redacted):
        return "compile:SyntaxError"

    if _SYNTAX_ERROR_LOWER_RE.search(redacted):
        return "compile:syntax error"

    m_undef = _GO_UNDEFINED_RE.search(redacted)
    if m_undef:
        sym = m_undef.group(1)
        return f"compile:undefined:{sym}" if sym else "compile:undefined:"

    if _CANNOT_FIND_RE.search(redacted):
        return "compile:cannot find"

    # 2. Test-class signatures (`--- FAIL`, `AssertionError`, `panic:`, `FAIL`)
    m_gofail = _GO_TEST_FAIL_RE.search(redacted)
    if m_gofail:
        tname = m_gofail.group(1)
        return f"test:--- FAIL:{tname}" if tname else "test:--- FAIL"

    if _ASSERTION_ERR_RE.search(redacted):
        return "test:AssertionError"

    m_panic = _PANIC_RE.search(redacted)
    if m_panic:
        detail = m_panic.group(1).strip()
        if detail:
            norm_detail = fingerprint_command(detail)[:80]
            return f"test:panic:{norm_detail}"
        return "test:panic:"

    if _GENERIC_FAIL_RE.search(redacted):
        return "test:FAIL"

    # 3. Generic error fallback (normalize numbers/quotes so line shifts still group)
    cleaned = _EXIT_BANNER_RE.sub("", redacted)
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    for ln in lines:
        if _SALIENT_ERR_LINE_RE.search(ln):
            return f"error:{fingerprint_command(ln)[:120]}"
    if lines:
        return f"error:{fingerprint_command(lines[0])[:120]}"
    return "error:unknown"


def classify_error_kind(sig_or_text: str | None) -> ErrorKind | None:
    """Classify an error signature (or raw error text) into compile / test / other / None."""
    if not sig_or_text:
        return None
    if sig_or_text.startswith("compile:"):
        return "compile"
    if sig_or_text.startswith("test:"):
        return "test"
    if sig_or_text.startswith("error:"):
        return "other"
    sig = extract_error_signature(sig_or_text, is_error=True)
    if sig is None:
        return None
    if sig.startswith("compile:"):
        return "compile"
    if sig.startswith("test:"):
        return "test"
    return "other"


def normalize_repo_path(path: str, cwd: str | None = None) -> str:
    """Normalize a file path relative to trajectory `cwd` when possible."""
    raw = path.strip()
    if not raw:
        return ""
    norm = posixpath.normpath(raw)
    if cwd:
        norm_cwd = posixpath.normpath(cwd.strip())
        if norm_cwd and norm_cwd != "/" and norm.startswith(norm_cwd + "/"):
            norm = norm[len(norm_cwd) + 1 :]
    if norm.startswith("./"):
        norm = norm[2:]
    return norm


def extract_read_paths(tc: ToolCall, cwd: str | None = None) -> tuple[str, ...]:
    """Extract normalized file paths read by a native Read tool or shell read command."""
    tlower = tc.name.strip().lower()
    if tlower in NATIVE_READ_TOOLS and tc.raw_args:
        try:
            parsed = json.loads(tc.raw_args)
            if isinstance(parsed, Mapping):
                p = (
                    parsed.get("file_path")
                    or parsed.get("path")
                    or parsed.get("notebook_path")
                )
                if isinstance(p, str) and p.strip():
                    norm = normalize_repo_path(p, cwd)
                    return (norm,) if norm else ()
        except json.JSONDecodeError:
            pass

    cmd = extract_shell_command(tc)
    if cmd is None:
        return ()

    paths: list[str] = []
    seen: set[str] = set()
    for seg in _split_compound_segments(cmd):
        stages = _split_pipeline(seg)
        for st in stages:
            if _classify_single_stage(st) != "read":
                continue
            tokens = _unwrap_command_tokens(_tokenize_stage(st))
            if not tokens:
                continue
            exe = posixpath.basename(tokens[0]).lower()
            rest = tokens[1:]
            candidates: list[str] = []
            if exe == "sed":
                # Skip flags and the script/range expression; remaining args are file paths
                script_consumed = False
                skip_next = False
                for arg in rest:
                    if skip_next:
                        skip_next = False
                        continue
                    if arg in ("-e", "-f"):
                        script_consumed = True
                        skip_next = True
                        continue
                    if arg.startswith("-"):
                        continue
                    if not script_consumed:
                        script_consumed = True
                        continue
                    candidates.append(arg)
            else:
                skip_next = False
                for arg in rest:
                    if skip_next:
                        skip_next = False
                        continue
                    if arg in ("-n", "-c", "-s", "--lines", "--bytes"):
                        skip_next = True
                        continue
                    if arg.startswith("-") or arg.startswith("+"):
                        continue
                    candidates.append(arg)

            for cand in candidates:
                norm = normalize_repo_path(cand, cwd)
                if norm and norm not in seen:
                    seen.add(norm)
                    paths.append(norm)
    return tuple(paths)
