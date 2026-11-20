"""DeepSeek Harness (DSH) session log adapter (.jsonl.zstd -> Trajectory IR)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Any

import zstandard as zstd

from dsh_metrics.ir import (
    Compaction,
    Edit,
    ModelCall,
    ToolCall,
    Trajectory,
    TrajectoryEvent,
    TurnEnd,
)
from dsh_metrics.redact import make_output_tail, redact_obj, redact_secrets

KNOWN_DSH_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "session",
        "turn/start",
        "turn/end",
        "step/start",
        "step/end",
        "request/header",
        "request/context",
        "reasoning-chunks",
        "text-chunks",
        "tool-call-chunks",
        "assistant/chunk",
        "assistant/attempt",
        "assistant/message",
        "user/message",
        "system/message",
        "tool/call",
        "tool/result",
        "tool/code-dispatch-start",
        "tool/code-dispatch",
        "llm/retry",
        "llm/retry-started",
        "compaction/start",
        "compaction/end",
        "compaction/summary",
        "compaction/prune",
        "agent/inbox/spliced",
        "permission/preset",
        "sandbox/mode",
        "approval/policy",
        "approval/asked",
        "approval/decided",
        "session/title",
        "session/title-llm-request",
        "session/end-seed",
        "todo/write",
        "subagent/descriptor",
        "web/deepseek-search-llm-request",
        "command/run",
        "command/done",
        "goal/change",
        "agent-preset/selected",
    }
)

_CHUNK_EVENT_TYPES: frozenset[str] = frozenset(
    {"reasoning-chunks", "text-chunks", "tool-call-chunks", "assistant/chunk"}
)
_SHELL_TOOL_NAMES: frozenset[str] = frozenset({"bash", "tools__bash", "sh", "zsh"})
_EDIT_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "edit",
        "write",
        "Edit",
        "Write",
        "MultiEdit",
        "tools.edit",
        "tools.write",
        "str_replace_editor",
    }
)

_EXIT_CODE_RE = re.compile(r"\[exit code:\s*(-?\d+)\]\s*$")
_TRUNCATED_RE = re.compile(
    r"(?:"
    r"\bomitted\s+\d[\d,]*\s+bytes\b"
    r"|\b\d[\d,]*\s+bytes\s+omitted\b"
    r"|^\[output truncated;\s*full output:"
    r"|\((?:Showing\s+\d+\s+of\s+\d+[^)]*\.\s*)?Full\s+(?:grep|sorted|formatted)?\s*result\s+stored\s+at:"
    r"|<persisted-output>"
    r"|Output\s+too\s+large\s*\([^)]+\)\.\s*Full\s+output\s+saved\s+to:"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

_FALLBACK_COMPILE_ERR_RE = re.compile(
    r"(?:syntax error|undefined:|error TS\d+|SyntaxError|cannot find)",
    re.IGNORECASE,
)
_FALLBACK_TEST_ERR_RE = re.compile(r"(?:---\s*FAIL|AssertionError|panic:|\bFAIL\b)")


def _pos_int(val: Any) -> int | None:
    if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
        return None
    return val


def _opt_int(val: Any, default: int | None = None) -> int | None:
    if isinstance(val, bool) or not isinstance(val, int):
        return default
    return val


def _short_hash(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _derive_error_signature(text: str, is_error: bool) -> str | None:
    """Derive normalized error signature at the adapter boundary."""
    if not text and not is_error:
        return None
    try:
        from dsh_metrics.classifiers import extract_error_signature

        sig = extract_error_signature(text, is_error=is_error)
        return redact_secrets(sig) if sig else None
    except ImportError:
        pass

    if not is_error:
        return None
    redacted = redact_secrets(text)
    m = _FALLBACK_COMPILE_ERR_RE.search(redacted)
    if m:
        return f"compile:{m.group(0).lower()}"
    m = _FALLBACK_TEST_ERR_RE.search(redacted)
    if m:
        return f"test:{m.group(0).strip()}"
    for line in redacted.splitlines():
        s = line.strip()
        if s:
            return f"error:{s[:120]}"
    return "error:unknown"


def detect_truncated(text: str) -> bool:
    """Return True if tool output carries a DSH spilled/omitted-bytes notice."""
    if not text:
        return False
    return bool(_TRUNCATED_RE.search(text))


def extract_bash_exit_code(text: str, raw_is_error: bool) -> int | None:
    """Extract bash exit code from trailing '[exit code: N]' notice."""
    match = _EXIT_CODE_RE.search(text)
    if match:
        return int(match.group(1))
    if not raw_is_error:
        return 0
    return None


def _basic_args_fingerprint(
    tool_name: str, redacted_parsed: Any, redacted_raw: str | None
) -> str | None:
    """Compute a deterministic redacted argument fingerprint (refined in M4)."""
    # Try to use M4 classifiers if installed; fallback cleanly during M2/M3.
    try:
        from dsh_metrics.classifiers import fingerprint_tool_args

        return fingerprint_tool_args(tool_name, redacted_parsed, redacted_raw)
    except ImportError:
        pass

    if isinstance(redacted_parsed, Mapping):
        cmd = redacted_parsed.get("command") or redacted_parsed.get("cmd")
        if isinstance(cmd, str):
            return " ".join(cmd.strip().split())
        path = redacted_parsed.get("file_path") or redacted_parsed.get("path")
        if isinstance(path, str):
            return f"{tool_name}:{path.strip()}"
    if redacted_raw is not None:
        return f"{tool_name}:{redacted_raw.strip()}"
    return None


def discover_dsh_files(root: str | Path) -> list[Path]:
    """Discover DSH session log files under root without duplicating migrated v4 logs."""
    root_path = Path(root).expanduser()
    if root_path.is_file():
        return [root_path]
    if not root_path.is_dir():
        return []

    discovered: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames.sort()
        files_set = set(filenames)
        if "session.jsonl.zstd" in files_set:
            discovered.append(Path(dirpath) / "session.jsonl.zstd")
        elif "session.v4.jsonl.zstd" in files_set:
            discovered.append(Path(dirpath) / "session.v4.jsonl.zstd")
        elif "session.jsonl" in files_set:
            discovered.append(Path(dirpath) / "session.jsonl")
    discovered.sort()
    return discovered


def _stream_lines(path: Path) -> Iterator[tuple[str | None, bool]]:
    """Stream lines read-only from a .zstd or plain .jsonl file.

    Yields (line, is_stream_error). Never raises on corrupt compressed frames.
    """
    try:
        with open(path, "rb") as fh:
            if path.suffix == ".zstd" or path.name.endswith(".zstd"):
                dctx = zstd.ZstdDecompressor()
                try:
                    with dctx.stream_reader(fh) as reader:
                        wrapper = io.TextIOWrapper(
                            reader, encoding="utf-8", errors="replace"
                        )
                        for line in wrapper:
                            yield line, False
                except (zstd.ZstdError, OSError):
                    yield None, True
            else:
                wrapper = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
                for line in wrapper:
                    yield line, False
    except OSError:
        yield None, True


@dataclass
class _StepState:
    turn: int | None
    step: int | None
    started_at: int | None = None
    first_token_at: int | None = None
    ended_at: int | None = None
    retries: int = 0
    provider: str | None = None
    model: str | None = None
    emitted: bool = False


@dataclass
class _PendingToolCall:
    turn: int | None
    step: int | None
    call_id: str | None
    name: str
    args_fingerprint: str | None
    raw_args: str | None
    redacted_parsed: Any
    started_at: int | None
    available_tools: tuple[str, ...] | None
    ended_at: int | None = None
    is_error: bool | None = None
    exit_code: int | None = None
    output_bytes: int | None = None
    truncated: bool | None = None
    error_signature: str | None = None
    output_tail: str | None = None


@dataclass
class _PendingCompaction:
    started_at: int | None
    ended_at: int | None = None
    ok: bool | None = None
    tokens_before: int | None = None
    tokens_after: int | None = None
    reason: str | None = None
    shadowed_tokens: int = 0
    summary_output_tokens: int | None = None


def _extract_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for sub in content:
            if isinstance(sub, Mapping) and isinstance(sub.get("text"), str):
                parts.append(sub["text"])
            elif isinstance(sub, str):
                parts.append(sub)
        return "\n".join(parts)
    return ""


def _format_turn_end_reason(reason_obj: Any) -> str | None:
    if isinstance(reason_obj, Mapping):
        kind = str(reason_obj.get("kind") or "unknown")
        err = reason_obj.get("error")
        if isinstance(err, Mapping):
            code = str(err.get("code") or "").strip()
            msg = redact_secrets(str(err.get("message") or "").strip())[:200]
            if code and msg:
                return f"{kind}:{code}:{msg}"
            if code:
                return f"{kind}:{code}"
            if msg:
                return f"{kind}:{msg}"
        elif err:
            return f"{kind}:{redact_secrets(str(err))[:200]}"
        return kind
    if isinstance(reason_obj, str):
        return redact_secrets(reason_obj)
    return None


def _maybe_build_edit(tc: _PendingToolCall) -> Edit | None:
    if tc.name not in _EDIT_TOOL_NAMES or tc.is_error is True:
        return None
    if not isinstance(tc.redacted_parsed, Mapping):
        return None
    raw_path = tc.redacted_parsed.get("file_path") or tc.redacted_parsed.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None

    old_str = tc.redacted_parsed.get("old_string")
    new_str = tc.redacted_parsed.get("new_string")
    content = tc.redacted_parsed.get("content")
    before_hash = _short_hash(old_str) if isinstance(old_str, str) else None
    if isinstance(new_str, str):
        after_hash = _short_hash(new_str)
    elif isinstance(content, str):
        after_hash = _short_hash(content)
    else:
        after_hash = None

    return Edit(
        step=tc.step,
        path=raw_path.strip(),
        before_hash=before_hash,
        after_hash=after_hash,
        turn=tc.turn,
        at=tc.ended_at if tc.ended_at is not None else tc.started_at,
    )


def _fallback_session_id(path: Path) -> str:
    parent = path.parent.name
    if parent.startswith("session-"):
        return parent[len("session-") :]
    if parent and parent not in (".", "/"):
        return parent
    return path.stem


def _resolve_dsh_log_pair(file_path: Path) -> tuple[Path, Path | None]:
    """Return (primary_path, post_upgrade_v4_path) when a session dir has both legacy and v4 logs."""
    if file_path.is_dir():
        legacy = file_path / "session.jsonl.zstd"
        v4 = file_path / "session.v4.jsonl.zstd"
        if legacy.is_file() and v4.is_file():
            return legacy, v4
        if legacy.is_file():
            return legacy, None
        if v4.is_file():
            return v4, None
    if file_path.name in ("session.jsonl.zstd", "session.v4.jsonl.zstd"):
        legacy = file_path.parent / "session.jsonl.zstd"
        v4 = file_path.parent / "session.v4.jsonl.zstd"
        if legacy.is_file() and v4.is_file():
            return legacy, v4
    return file_path, None


def _iter_parsed_dsh_records(
    primary_path: Path, post_upgrade_v4: Path | None
) -> Iterator[tuple[Mapping[str, Any] | None, str | None, int | None, bool]]:
    """Stream records from primary log, appending post-upgrade v4 events strictly after legacy's last event."""
    last_legacy_time: int | None = None

    for raw_line, stream_err in _stream_lines(primary_path):
        if stream_err:
            yield None, None, None, True
            break
        assert raw_line is not None
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            yield None, None, None, True
            continue
        if not isinstance(obj, Mapping):
            yield None, None, None, True
            continue
        ev_type = obj.get("type")
        if not isinstance(ev_type, str) or not ev_type:
            yield None, None, None, True
            continue
        ev_time = _pos_int(obj.get("time")) or _pos_int(obj.get("time0"))
        if ev_time is not None and (
            last_legacy_time is None or ev_time > last_legacy_time
        ):
            last_legacy_time = ev_time
        yield obj, ev_type, ev_time, False

    if post_upgrade_v4 is None:
        return

    for raw_line, stream_err in _stream_lines(post_upgrade_v4):
        if stream_err:
            yield None, None, None, True
            break
        assert raw_line is not None
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            yield None, None, None, True
            continue
        if not isinstance(obj, Mapping):
            yield None, None, None, True
            continue
        ev_type = obj.get("type")
        if not isinstance(ev_type, str) or not ev_type:
            yield None, None, None, True
            continue
        ev_time = _pos_int(obj.get("time")) or _pos_int(obj.get("time0"))
        if ev_type == "session":
            yield obj, ev_type, ev_time, False
            continue
        if last_legacy_time is None or (
            ev_time is not None and ev_time > last_legacy_time
        ):
            yield obj, ev_type, ev_time, False


def parse_dsh_session(path: str | Path) -> Trajectory:
    """Stream-parse a DSH session (.jsonl.zstd or .jsonl) into a Trajectory."""
    file_path = Path(path).expanduser()
    primary_path, post_upgrade_v4 = _resolve_dsh_log_pair(file_path)
    session_id: str | None = None
    started_at: int | None = None
    cwd: str | None = None
    parse_failures = 0
    unknown_types: list[str] = []
    unknown_seen: set[str] = set()
    routes: list[str] = []
    routes_seen: set[str] = set()

    current_turn: int | None = None
    current_step: int | None = None
    current_provider: str | None = None
    current_model: str | None = None
    available_tools: tuple[str, ...] | None = None

    steps: dict[tuple[int | None, int | None], _StepState] = {}
    pending_tools_by_id: dict[str, _PendingToolCall] = {}
    active_compaction: _PendingCompaction | None = None
    active_prune: _PendingCompaction | None = None
    awaiting_tokens_after: list[_PendingCompaction] = []
    last_prompt_tokens: int | None = None
    earliest_time: int | None = None

    slots: list[TrajectoryEvent | _PendingToolCall | _PendingCompaction] = []

    def record_route(provider: str | None, model: str | None) -> None:
        if provider and model:
            route = f"{provider}/{model}"
        elif model:
            route = model
        elif provider:
            route = provider
        else:
            return
        if route not in routes_seen:
            routes_seen.add(route)
            routes.append(route)

    def get_step_state(
        turn: int | None, step: int | None, started: int | None = None
    ) -> _StepState:
        key = (turn, step)
        st = steps.get(key)
        if st is None:
            st = _StepState(
                turn=turn,
                step=step,
                started_at=started,
                provider=current_provider,
                model=current_model,
            )
            steps[key] = st
        elif st.started_at is None and started is not None:
            st.started_at = started
        return st

    def flush_prune() -> None:
        nonlocal active_prune
        if active_prune is not None:
            awaiting_tokens_after.append(active_prune)
            active_prune = None

    for obj, ev_type, ev_time, is_failure in _iter_parsed_dsh_records(
        primary_path, post_upgrade_v4
    ):
        if is_failure:
            parse_failures += 1
            continue
        assert obj is not None and ev_type is not None

        if ev_type not in KNOWN_DSH_EVENT_TYPES and ev_type not in unknown_seen:
            unknown_seen.add(ev_type)
            unknown_types.append(ev_type)

        if ev_time is not None and (earliest_time is None or ev_time < earliest_time):
            earliest_time = ev_time

        if ev_type == "session":
            raw_id = obj.get("id")
            if session_id is None and isinstance(raw_id, str) and raw_id:
                session_id = raw_id
            created = _pos_int(obj.get("createdAt"))
            if started_at is None and created is not None:
                started_at = created
            raw_cwd = obj.get("cwd")
            if cwd is None and isinstance(raw_cwd, str) and raw_cwd:
                cwd = redact_secrets(raw_cwd)
            continue

        data = obj.get("data")
        if not isinstance(data, Mapping):
            data = {}

        if ev_type == "request/header":
            header = data.get("header")
            if isinstance(header, Mapping):
                cfg = header.get("config")
                if isinstance(cfg, Mapping):
                    if isinstance(cfg.get("provider"), str) and cfg["provider"]:
                        current_provider = cfg["provider"]
                    if isinstance(cfg.get("model"), str) and cfg["model"]:
                        current_model = cfg["model"]
                    record_route(current_provider, current_model)
                tools = header.get("tools")
                if isinstance(tools, list):
                    available_tools = tuple(
                        str(t["name"])
                        for t in tools
                        if isinstance(t, Mapping) and isinstance(t.get("name"), str)
                    )

        elif ev_type == "request/context":
            if isinstance(data.get("provider"), str) and data["provider"]:
                current_provider = data["provider"]
            if isinstance(data.get("model"), str) and data["model"]:
                current_model = data["model"]
            record_route(current_provider, current_model)

        elif ev_type == "turn/start":
            flush_prune()
            current_turn = _opt_int(data.get("turn"), default=current_turn)
            current_step = None

        elif ev_type == "step/start":
            flush_prune()
            current_turn = _opt_int(data.get("turn"), default=current_turn)
            current_step = _opt_int(data.get("step"))
            st = _StepState(
                turn=current_turn,
                step=current_step,
                started_at=ev_time,
                provider=current_provider,
                model=current_model,
            )
            steps[(current_turn, current_step)] = st

        elif ev_type in _CHUNK_EVENT_TYPES:
            turn = _opt_int(data.get("turn"), default=current_turn)
            step = _opt_int(data.get("step"), default=current_step)
            st = get_step_state(turn, step)
            if ev_time is not None and (
                st.first_token_at is None or ev_time < st.first_token_at
            ):
                st.first_token_at = ev_time

        elif ev_type == "llm/retry":
            turn = _opt_int(data.get("turn"), default=current_turn)
            step = _opt_int(data.get("step"), default=current_step)
            st = get_step_state(turn, step)
            st.retries += 1
            retry_num = _opt_int(data.get("retry"))
            if retry_num is not None and retry_num > st.retries:
                st.retries = retry_num

        elif ev_type == "assistant/message":
            flush_prune()
            turn = _opt_int(data.get("turn"), default=current_turn)
            step = _opt_int(data.get("step"), default=current_step)
            st = get_step_state(turn, step)

            msg = data.get("message") if isinstance(data.get("message"), Mapping) else {}
            src = msg.get("source") if isinstance(msg.get("source"), Mapping) else {}
            prov = (
                src.get("provider")
                if isinstance(src.get("provider"), str) and src.get("provider")
                else (st.provider or current_provider)
            )
            mdl = (
                src.get("model")
                if isinstance(src.get("model"), str) and src.get("model")
                else (st.model or current_model)
            )
            record_route(prov, mdl)

            usage = data.get("usage") if isinstance(data.get("usage"), Mapping) else {}
            in_tok = _opt_int(usage.get("inputTokens"))
            cache_r = _opt_int(usage.get("cacheReadTokens"))
            cache_w = _opt_int(usage.get("cacheWriteTokens"))
            out_tok = _opt_int(usage.get("outputTokens"))
            reas_tok = _opt_int(usage.get("reasoningTokens"))

            slots.append(
                ModelCall(
                    step=step,
                    started_at=st.started_at,
                    first_token_at=st.first_token_at,
                    ended_at=ev_time,
                    input_tokens=in_tok,
                    cache_read_tokens=cache_r,
                    cache_write_tokens=cache_w,
                    output_tokens=out_tok,
                    reasoning_tokens=reas_tok,
                    retries=st.retries,
                    provider=prov,
                    model=mdl,
                    turn=turn,
                )
            )
            st.emitted = True

            if any(x is not None for x in (in_tok, cache_r, cache_w)):
                prompt_tokens = (in_tok or 0) + (cache_r or 0) + (cache_w or 0)
                if awaiting_tokens_after:
                    for comp in awaiting_tokens_after:
                        if comp.tokens_after is None:
                            comp.tokens_after = prompt_tokens
                    awaiting_tokens_after.clear()
                last_prompt_tokens = prompt_tokens

        elif ev_type == "step/end":
            turn = _opt_int(data.get("turn"), default=current_turn)
            step = _opt_int(data.get("step"), default=current_step)
            st = steps.get((turn, step))
            if st is not None and not st.emitted:
                st.ended_at = ev_time
                if (
                    st.started_at is not None
                    or st.first_token_at is not None
                    or st.retries > 0
                ):
                    slots.append(
                        ModelCall(
                            step=step,
                            started_at=st.started_at,
                            first_token_at=st.first_token_at,
                            ended_at=st.ended_at,
                            retries=st.retries,
                            provider=st.provider or current_provider,
                            model=st.model or current_model,
                            turn=turn,
                        )
                    )
                    st.emitted = True

        elif ev_type == "tool/call":
            flush_prune()
            turn = _opt_int(data.get("turn"), default=current_turn)
            step = _opt_int(data.get("step"), default=current_step)
            call_id = (
                data["callId"]
                if isinstance(data.get("callId"), str) and data["callId"]
                else None
            )
            name = str(data.get("name") or "")
            raw_arguments = data.get("arguments")
            redacted_parsed: Any = None
            raw_args_str: str | None = None

            if isinstance(raw_arguments, str):
                try:
                    parsed = json.loads(raw_arguments)
                    redacted_parsed = redact_obj(parsed)
                    raw_args_str = json.dumps(
                        redacted_parsed, sort_keys=True, ensure_ascii=False
                    )
                except json.JSONDecodeError:
                    raw_args_str = redact_secrets(raw_arguments)
            elif isinstance(raw_arguments, Mapping):
                redacted_parsed = redact_obj(raw_arguments)
                raw_args_str = json.dumps(
                    redacted_parsed, sort_keys=True, ensure_ascii=False
                )
            elif raw_arguments is not None:
                raw_args_str = redact_secrets(str(raw_arguments))

            fp = _basic_args_fingerprint(name, redacted_parsed, raw_args_str)
            pending_tc = _PendingToolCall(
                turn=turn,
                step=step,
                call_id=call_id,
                name=name,
                args_fingerprint=fp,
                raw_args=raw_args_str,
                redacted_parsed=redacted_parsed,
                started_at=ev_time,
                available_tools=available_tools,
            )
            slots.append(pending_tc)
            if call_id is not None:
                pending_tools_by_id[call_id] = pending_tc

        elif ev_type == "tool/result":
            turn = _opt_int(data.get("turn"), default=current_turn)
            step = _opt_int(data.get("step"), default=current_step)
            msg = data.get("message") if isinstance(data.get("message"), Mapping) else {}
            src = msg.get("source") if isinstance(msg.get("source"), Mapping) else {}
            fallback_cid = (
                src["callId"]
                if isinstance(src.get("callId"), str) and src["callId"]
                else None
            )
            blocks = msg.get("content") if isinstance(msg.get("content"), list) else []
            for blk in blocks:
                if not isinstance(blk, Mapping):
                    continue
                cid = (
                    blk["toolCallId"]
                    if isinstance(blk.get("toolCallId"), str) and blk["toolCallId"]
                    else fallback_cid
                )
                full_text = _extract_result_text(blk.get("content"))
                out_bytes = len(full_text.encode("utf-8"))
                trunc = detect_truncated(full_text)
                raw_err = bool(blk.get("isError"))
                bounded_tail = make_output_tail(full_text)

                target = pending_tools_by_id.get(cid) if cid else None
                if target is not None:
                    exit_code = (
                        extract_bash_exit_code(full_text, raw_err)
                        if target.name.lower() in _SHELL_TOOL_NAMES
                        else None
                    )
                    is_err = raw_err or (
                        exit_code is not None and exit_code != 0
                    )
                    target.ended_at = ev_time
                    target.exit_code = exit_code
                    target.is_error = is_err
                    target.output_bytes = out_bytes
                    target.truncated = trunc
                    target.error_signature = _derive_error_signature(full_text, is_err)
                    target.output_tail = bounded_tail
                else:
                    orphan_tc = _PendingToolCall(
                        turn=turn,
                        step=step,
                        call_id=cid,
                        name="unknown",
                        args_fingerprint=None,
                        raw_args=None,
                        redacted_parsed=None,
                        started_at=None,
                        available_tools=available_tools,
                        ended_at=ev_time,
                        is_error=raw_err,
                        exit_code=None,
                        output_bytes=out_bytes,
                        truncated=trunc,
                        error_signature=_derive_error_signature(full_text, raw_err),
                        output_tail=bounded_tail,
                    )
                    slots.append(orphan_tc)

        elif ev_type == "compaction/start":
            flush_prune()
            active_compaction = _PendingCompaction(
                started_at=ev_time,
                tokens_before=last_prompt_tokens,
            )
            slots.append(active_compaction)

        elif ev_type == "compaction/summary":
            if active_compaction is None:
                active_compaction = _PendingCompaction(
                    started_at=ev_time,
                    tokens_before=last_prompt_tokens,
                )
                slots.append(active_compaction)
            shadowed = _opt_int(data.get("shadowedTokenCount"))
            if shadowed is not None:
                active_compaction.shadowed_tokens += shadowed
                if active_compaction.tokens_before is None:
                    active_compaction.tokens_before = shadowed
            usage = data.get("usage") if isinstance(data.get("usage"), Mapping) else {}
            out_tok = _opt_int(usage.get("outputTokens"))
            if out_tok is not None:
                active_compaction.summary_output_tokens = out_tok

        elif ev_type == "compaction/end":
            if active_compaction is None:
                active_compaction = _PendingCompaction(
                    started_at=None,
                    tokens_before=last_prompt_tokens,
                )
                slots.append(active_compaction)
            active_compaction.ended_at = ev_time
            err = data.get("error")
            if err:
                active_compaction.ok = False
                active_compaction.reason = redact_secrets(str(err))[:200]
            else:
                active_compaction.ok = True
                active_compaction.reason = (
                    "manual" if data.get("sourceCommandId") else "summary"
                )
                awaiting_tokens_after.append(active_compaction)
            active_compaction = None

        elif ev_type == "compaction/prune":
            shadowed = _opt_int(data.get("shadowedTokenCount")) or 0
            if active_compaction is not None:
                active_compaction.shadowed_tokens += shadowed
            else:
                if active_prune is None:
                    active_prune = _PendingCompaction(
                        started_at=ev_time,
                        ended_at=ev_time,
                        ok=True,
                        tokens_before=last_prompt_tokens,
                        reason="prune",
                        shadowed_tokens=shadowed,
                    )
                    slots.append(active_prune)
                else:
                    active_prune.ended_at = ev_time or active_prune.ended_at
                    active_prune.shadowed_tokens += shadowed

        elif ev_type == "turn/end":
            flush_prune()
            turn = _opt_int(data.get("turn"), default=current_turn)
            # If the current step never emitted ModelCall (e.g. error/abort before step/end)
            st = steps.get((turn, current_step))
            if st is not None and not st.emitted:
                if (
                    st.started_at is not None
                    or st.first_token_at is not None
                    or st.retries > 0
                ):
                    slots.append(
                        ModelCall(
                            step=st.step,
                            started_at=st.started_at,
                            first_token_at=st.first_token_at,
                            ended_at=ev_time,
                            retries=st.retries,
                            provider=st.provider or current_provider,
                            model=st.model or current_model,
                            turn=turn,
                        )
                    )
                    st.emitted = True
            slots.append(
                TurnEnd(
                    turn=turn,
                    reason=_format_turn_end_reason(data.get("reason")),
                    at=ev_time,
                )
            )

    flush_prune()

    # Flush any trailing unfinished step at EOF
    for st in steps.values():
        if not st.emitted and (
            st.first_token_at is not None or st.retries > 0
        ):
            slots.append(
                ModelCall(
                    step=st.step,
                    started_at=st.started_at,
                    first_token_at=st.first_token_at,
                    ended_at=st.ended_at,
                    retries=st.retries,
                    provider=st.provider or current_provider,
                    model=st.model or current_model,
                    turn=st.turn,
                )
            )
            st.emitted = True

    resolved_sid = session_id or _fallback_session_id(primary_path)
    seen_resp_counts: dict[str, int] = {}

    # Materialize frozen IR events
    final_events: list[TrajectoryEvent] = []
    for slot in slots:
        if isinstance(slot, _PendingToolCall):
            final_events.append(
                ToolCall(
                    step=slot.step,
                    call_id=slot.call_id,
                    name=slot.name,
                    args_fingerprint=slot.args_fingerprint,
                    raw_args=slot.raw_args,
                    started_at=slot.started_at,
                    ended_at=slot.ended_at,
                    is_error=slot.is_error,
                    exit_code=slot.exit_code,
                    output_bytes=slot.output_bytes,
                    truncated=slot.truncated,
                    available_tools=slot.available_tools,
                    turn=slot.turn,
                    error_signature=slot.error_signature,
                    output_tail=slot.output_tail,
                )
            )
            edit_ev = _maybe_build_edit(slot)
            if edit_ev is not None:
                final_events.append(edit_ev)
        elif isinstance(slot, _PendingCompaction):
            tok_after = slot.tokens_after
            if (
                tok_after is None
                and slot.ok is True
                and slot.tokens_before is not None
                and slot.shadowed_tokens > 0
            ):
                estimated = (
                    slot.tokens_before
                    - slot.shadowed_tokens
                    + (slot.summary_output_tokens or 0)
                )
                if estimated >= 0:
                    tok_after = estimated
            final_events.append(
                Compaction(
                    started_at=slot.started_at,
                    ended_at=slot.ended_at,
                    ok=slot.ok,
                    tokens_before=slot.tokens_before,
                    tokens_after=tok_after,
                    reason=slot.reason,
                )
            )
        elif isinstance(slot, ModelCall):
            resp_id = slot.response_id
            if (
                resp_id is None
                and slot.turn is not None
                and slot.step is not None
            ):
                base_resp = f"{resolved_sid}:{slot.turn}:{slot.step}"
                cnt = seen_resp_counts.get(base_resp, 0) + 1
                seen_resp_counts[base_resp] = cnt
                resp_id = base_resp if cnt == 1 else f"{base_resp}#{cnt}"
            final_events.append(replace(slot, response_id=resp_id))
        else:
            final_events.append(slot)

    return Trajectory(
        source="dsh",
        session_id=resolved_sid,
        cwd=cwd,
        model_routes=tuple(routes),
        started_at=started_at if started_at is not None else earliest_time,
        events=tuple(final_events),
        parse_failures=parse_failures,
        unknown_event_types=tuple(unknown_types),
    )
