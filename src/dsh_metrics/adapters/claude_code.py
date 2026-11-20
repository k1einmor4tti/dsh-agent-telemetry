"""Claude Code transcript adapter (~/.claude/projects/<slug>/*.jsonl -> Trajectory IR)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any

from dsh_metrics.adapters.dsh import (
    _basic_args_fingerprint,
    _derive_error_signature,
    detect_truncated,
)
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

KNOWN_CLAUDE_ENTRY_TYPES: frozenset[str] = frozenset(
    {
        "assistant",
        "user",
        "system",
        "summary",
        "api-request",
        "api-request-blob",
        "api-request-shape",
        "attachment",
        "last-prompt",
        "atis-latch",
        "custom-title",
        "agent-name",
        "mode",
        "permission-mode",
        "relocated",
        "queue-operation",
        "pr-link",
        "ai-title",
        "file-history-snapshot",
        "file-history-delta",
        "worktree-state",
        "session-owner",
        "cost-state",
        "tools-used",
        "progress",
    }
)

_SHELL_TOOL_NAMES: frozenset[str] = frozenset({"bash", "sh", "zsh"})
_EDIT_TOOL_NAMES: frozenset[str] = frozenset(
    {"Edit", "Write", "MultiEdit", "NotebookEdit", "edit", "write"}
)

_CC_EXIT_CODE_RE = re.compile(
    r"(?:^Exit code\s+(-?\d+)\b|\[exit code:\s*(-?\d+)\]\s*$)",
    re.MULTILINE,
)


def parse_iso_ms(ts: Any) -> int | None:
    """Convert ISO-8601 timestamp string or epoch-ms int into positive epoch ms."""
    if isinstance(ts, bool):
        return None
    if isinstance(ts, int):
        return ts if ts > 0 else None
    if not isinstance(ts, str) or not ts.strip():
        return None
    raw = ts.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ms = int(round(dt.timestamp() * 1000))
    return ms if ms > 0 else None


def _opt_int(val: Any) -> int | None:
    if isinstance(val, bool) or not isinstance(val, int):
        return None
    return val


def _short_hash(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _extract_cc_bash_exit_code(text: str, raw_is_error: bool) -> int | None:
    match = _CC_EXIT_CODE_RE.search(text)
    if match:
        val = match.group(1) if match.group(1) is not None else match.group(2)
        return int(val)
    if not raw_is_error:
        return 0
    return None


def _extract_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), (str, list)):
                    nested = _extract_content_text(item["content"])
                    if nested:
                        parts.append(nested)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def _split_provider_model(raw_model: str | None) -> tuple[str | None, str | None]:
    if not raw_model or raw_model == "<synthetic>":
        return None, None
    if "/" in raw_model:
        prov, mdl = raw_model.split("/", 1)
        return (prov or None), (mdl or None)
    return "anthropic", raw_model


def _extract_group_id(obj: Mapping[str, Any], msg: Mapping[str, Any]) -> str | None:
    """Return message.id, falling back to requestId, or None if neither is present."""
    mid = msg.get("id")
    if isinstance(mid, str) and mid:
        return mid
    req_id = obj.get("requestId")
    if isinstance(req_id, str) and req_id:
        return req_id
    return None


def discover_claude_code_files(root: str | Path) -> list[Path]:
    """Discover top-level Claude Code transcripts (~/.claude/projects/*/*.jsonl).

    Subagent transcripts nested under session subdirectories are out of scope for P0.
    Also accepts a single .jsonl file or a single project slug directory.
    """
    root_path = Path(root).expanduser()
    if root_path.is_file():
        return [root_path]
    if not root_path.is_dir():
        return []

    direct_jsonl = sorted(p for p in root_path.glob("*.jsonl") if p.is_file())
    nested_top_level: list[Path] = []
    for child in sorted(root_path.iterdir()):
        if child.is_dir():
            for candidate in sorted(child.glob("*.jsonl")):
                if candidate.is_file():
                    nested_top_level.append(candidate)

    if nested_top_level:
        return sorted(direct_jsonl + nested_top_level)
    return direct_jsonl


@dataclass
class _PendingModelCall:
    msg_id: str
    response_id: str | None
    turn: int
    step: int
    started_at: int | None
    ended_at: int | None
    input_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    retries: int = 0
    provider: str | None = None
    model: str | None = None
    seen_tool_ids: set[str] = field(default_factory=set)


@dataclass
class _PendingToolCall:
    turn: int
    step: int
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
    ended_at: int | None
    ok: bool | None
    tokens_before: int | None
    tokens_after: int | None
    reason: str | None


def _build_edit_from_cc_tool(tc: _PendingToolCall) -> Edit | None:
    if tc.name not in _EDIT_TOOL_NAMES or tc.is_error is True:
        return None
    if not isinstance(tc.redacted_parsed, Mapping):
        return None
    raw_path = (
        tc.redacted_parsed.get("file_path")
        or tc.redacted_parsed.get("path")
        or tc.redacted_parsed.get("notebook_path")
    )
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None

    before_hash: str | None = None
    after_hash: str | None = None
    if isinstance(tc.redacted_parsed.get("old_string"), str):
        before_hash = _short_hash(tc.redacted_parsed["old_string"])
    if isinstance(tc.redacted_parsed.get("new_string"), str):
        after_hash = _short_hash(tc.redacted_parsed["new_string"])
    elif isinstance(tc.redacted_parsed.get("content"), str):
        after_hash = _short_hash(tc.redacted_parsed["content"])
    elif isinstance(tc.redacted_parsed.get("new_source"), str):
        after_hash = _short_hash(tc.redacted_parsed["new_source"])
    elif isinstance(tc.redacted_parsed.get("edits"), list):
        edits_list = tc.redacted_parsed["edits"]
        olds = [
            e.get("old_string", "")
            for e in edits_list
            if isinstance(e, Mapping) and isinstance(e.get("old_string"), str)
        ]
        news = [
            e.get("new_string", "")
            for e in edits_list
            if isinstance(e, Mapping) and isinstance(e.get("new_string"), str)
        ]
        if olds:
            before_hash = _short_hash("\n".join(olds))
        if news:
            after_hash = _short_hash("\n".join(news))

    return Edit(
        step=tc.step,
        path=raw_path.strip(),
        before_hash=before_hash,
        after_hash=after_hash,
        turn=tc.turn,
        at=tc.ended_at if tc.ended_at is not None else tc.started_at,
    )


class _ChainBuilder:
    """Stateful trajectory builder for either the main chain or one sidechain."""

    def __init__(
        self,
        key: str,
        seen_message_ids: set[str] | None = None,
        seen_tool_ids: set[str] | None = None,
    ) -> None:
        self.key = key
        self.seen_message_ids = seen_message_ids
        self.seen_tool_ids = seen_tool_ids
        self.session_id: str | None = None
        self.agent_id: str | None = None
        self.cwd: str | None = None
        self.started_at: int | None = None
        self.available_tools: tuple[str, ...] | None = None
        self.routes: list[str] = []
        self.routes_seen: set[str] = set()

        self.turn = 0
        self.step = 0
        self.turn_open = False
        self.turn_has_activity = False
        self.turn_last_ts: int | None = None
        self.trigger_ts: int | None = None
        self.pending_retries = 0
        self.pending_terminal_error: tuple[str, int | None] | None = None

        self.calls_by_msg_id: dict[str, _PendingModelCall] = {}
        self.pending_tools: dict[str, _PendingToolCall] = {}
        self.skipped_tool_ids: set[str] = set()
        self.skipped_duplicate_calls = 0
        self.awaiting_tokens_after: list[_PendingCompaction] = []
        self.last_prompt_tokens: int | None = None
        self.just_emitted_compact_meta = False

        self.slots: list[
            TrajectoryEvent | _PendingModelCall | _PendingToolCall | _PendingCompaction
        ] = []

    @property
    def has_meaningful_content(self) -> bool:
        return bool(self.slots)

    def _record_ts(self, ts: int | None) -> None:
        if ts is None:
            return
        if self.started_at is None or ts < self.started_at:
            self.started_at = ts
        self.turn_last_ts = ts

    def _record_route(self, provider: str | None, model: str | None) -> None:
        if provider and model:
            route = f"{provider}/{model}"
        elif model:
            route = model
        elif provider:
            route = provider
        else:
            return
        if route not in self.routes_seen:
            self.routes_seen.add(route)
            self.routes.append(route)

    def _ensure_turn_open(self) -> None:
        if not self.turn_open:
            self.turn += 1
            self.step = 0
            self.turn_open = True
            self.turn_has_activity = False

    def close_open_turn(self, reason: str, at: int | None) -> None:
        if not self.turn_open:
            return
        if self.pending_terminal_error is not None and reason == "completed":
            err_reason, err_at = self.pending_terminal_error
            reason = err_reason
            at = err_at or at
            self.turn_has_activity = True
        self.pending_terminal_error = None
        if self.turn_has_activity or reason != "completed":
            self.slots.append(
                TurnEnd(turn=self.turn, reason=reason, at=at or self.turn_last_ts)
            )
        else:
            # Reclaim unused turn number if a copied prefix prompt had all responses deduplicated
            self.turn -= 1
        self.turn_open = False
        self.turn_has_activity = False
        self.pending_retries = 0

    def handle_entry(self, obj: Mapping[str, Any]) -> None:
        ts = parse_iso_ms(obj.get("timestamp"))

        if isinstance(obj.get("sessionId"), str) and obj["sessionId"]:
            self.session_id = obj["sessionId"]
        if (
            self.agent_id is None
            and isinstance(obj.get("agentId"), str)
            and obj["agentId"]
        ):
            self.agent_id = obj["agentId"]
        if self.cwd is None and isinstance(obj.get("cwd"), str) and obj["cwd"]:
            self.cwd = redact_secrets(obj["cwd"])

        ev_type = obj.get("type")

        if ev_type == "tools-used":
            return

        if ev_type == "system":
            subtype = obj.get("subtype")
            if subtype == "init" and isinstance(obj.get("tools"), list):
                self.available_tools = tuple(
                    str(t) for t in obj["tools"] if isinstance(t, str)
                )
            elif "compactMetadata" in obj or subtype == "compact_boundary":
                self._record_ts(ts)
                meta = (
                    obj.get("compactMetadata")
                    if isinstance(obj.get("compactMetadata"), Mapping)
                    else {}
                )
                dur = _opt_int(meta.get("durationMs"))
                start_ts = (
                    ts - dur
                    if (ts is not None and dur and dur > 0 and ts - dur > 0)
                    else ts
                )
                pre_tok = _opt_int(meta.get("preTokens")) or self.last_prompt_tokens
                post_tok = _opt_int(meta.get("postTokens"))
                ok = obj.get("level") != "error" and not bool(obj.get("error"))
                trigger = (
                    redact_secrets(str(meta.get("trigger")))
                    if meta.get("trigger")
                    else "auto"
                )
                comp = _PendingCompaction(
                    started_at=start_ts,
                    ended_at=ts,
                    ok=ok,
                    tokens_before=pre_tok,
                    tokens_after=post_tok if ok else None,
                    reason=trigger,
                )
                self.slots.append(comp)
                if ok and post_tok is None:
                    self.awaiting_tokens_after.append(comp)
                self.just_emitted_compact_meta = True
                self.turn_has_activity = True
            elif subtype == "api_error":
                self._record_ts(ts)
                self.pending_retries += 1
                err = obj.get("error") if isinstance(obj.get("error"), Mapping) else {}
                status = err.get("status")
                msg = redact_secrets(str(err.get("message") or ""))[:200]
                err_str = f"error:{status}:{msg}" if status else f"error:{msg}"
                retry_attempt = _opt_int(obj.get("retryAttempt"))
                max_retries = _opt_int(obj.get("maxRetries"))
                if (
                    retry_attempt is not None
                    and max_retries is not None
                    and retry_attempt >= max_retries
                ) or obj.get("retryInMs") is None:
                    self.pending_terminal_error = (err_str, ts)
            elif subtype == "turn_duration":
                self._record_ts(ts)
                self.close_open_turn("completed", ts)
            return

        if ev_type == "user":
            if obj.get("isCompactSummary"):
                self._record_ts(ts)
                if not self.just_emitted_compact_meta:
                    comp = _PendingCompaction(
                        started_at=ts,
                        ended_at=ts,
                        ok=True,
                        tokens_before=self.last_prompt_tokens,
                        tokens_after=None,
                        reason="summary",
                    )
                    self.slots.append(comp)
                    self.awaiting_tokens_after.append(comp)
                    self.turn_has_activity = True
                self.just_emitted_compact_meta = False
                self.trigger_ts = ts or self.trigger_ts
                return

            self.just_emitted_compact_meta = False
            msg = obj.get("message") if isinstance(obj.get("message"), Mapping) else {}
            content = msg.get("content")

            tool_results: list[Mapping[str, Any]] = []
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, Mapping) and blk.get("type") == "tool_result":
                        tool_results.append(blk)

            full_user_text = _extract_content_text(content)
            if "[Request interrupted by user" in full_user_text:
                self._record_ts(ts)
                self._ensure_turn_open()
                self.turn_has_activity = True
                self.close_open_turn("aborted", ts)
                return

            if tool_results:
                for blk in tool_results:
                    cid = (
                        blk["tool_use_id"]
                        if isinstance(blk.get("tool_use_id"), str)
                        and blk["tool_use_id"]
                        else None
                    )
                    if cid and cid in self.skipped_tool_ids:
                        continue
                    self._record_ts(ts)
                    txt = _extract_content_text(blk.get("content"))
                    out_bytes = len(txt.encode("utf-8"))
                    trunc = detect_truncated(txt)
                    raw_err = bool(blk.get("is_error"))
                    tail = make_output_tail(txt)

                    target = self.pending_tools.get(cid) if cid else None
                    if target is not None:
                        exit_code = (
                            _extract_cc_bash_exit_code(txt, raw_err)
                            if target.name.lower() in _SHELL_TOOL_NAMES
                            else None
                        )
                        is_err = raw_err or (
                            exit_code is not None and exit_code != 0
                        )
                        target.ended_at = ts
                        target.exit_code = exit_code
                        target.is_error = is_err
                        target.output_bytes = out_bytes
                        target.truncated = trunc
                        target.error_signature = _derive_error_signature(txt, is_err)
                        target.output_tail = tail
                        self.turn_has_activity = True
                    else:
                        self._ensure_turn_open()
                        self.turn_has_activity = True
                        self.slots.append(
                            _PendingToolCall(
                                turn=self.turn,
                                step=max(self.step, 1),
                                call_id=cid,
                                name="unknown",
                                args_fingerprint=None,
                                raw_args=None,
                                redacted_parsed=None,
                                started_at=None,
                                available_tools=self.available_tools,
                                ended_at=ts,
                                is_error=raw_err,
                                exit_code=None,
                                output_bytes=out_bytes,
                                truncated=trunc,
                                error_signature=_derive_error_signature(txt, raw_err),
                                output_tail=tail,
                            )
                        )
                self.trigger_ts = ts
            else:
                if self.turn_open:
                    self.close_open_turn("completed", self.turn_last_ts)
                self._record_ts(ts)
                self.turn += 1
                self.step = 0
                self.turn_open = True
                self.turn_has_activity = False
                self.trigger_ts = ts
            return

        if ev_type == "assistant":
            self.just_emitted_compact_meta = False
            msg = obj.get("message") if isinstance(obj.get("message"), Mapping) else {}
            group_id = _extract_group_id(obj, msg)
            msg_id = group_id or f"anon_{obj.get('uuid') or id(obj)}"

            call = self.calls_by_msg_id.get(msg_id)
            if call is None:
                if (
                    group_id is not None
                    and self.seen_message_ids is not None
                    and group_id in self.seen_message_ids
                ):
                    self.skipped_duplicate_calls += 1
                    blocks = (
                        msg.get("content")
                        if isinstance(msg.get("content"), list)
                        else []
                    )
                    for blk in blocks:
                        if (
                            isinstance(blk, Mapping)
                            and blk.get("type") == "tool_use"
                            and isinstance(blk.get("id"), str)
                        ):
                            self.skipped_tool_ids.add(blk["id"])
                    return
                if group_id is not None and self.seen_message_ids is not None:
                    self.seen_message_ids.add(group_id)

            self._record_ts(ts)
            self._ensure_turn_open()
            self.turn_has_activity = True

            raw_model = msg.get("model") if isinstance(msg.get("model"), str) else None
            if obj.get("isApiErrorMessage"):
                err_txt = _extract_content_text(msg.get("content"))
                if err_txt:
                    self.pending_terminal_error = (
                        f"error:{redact_secrets(err_txt)[:200]}",
                        ts,
                    )

            prov, mdl = _split_provider_model(raw_model)
            self._record_route(prov, mdl)

            if call is None:
                self.step += 1
                call = _PendingModelCall(
                    msg_id=msg_id,
                    response_id=group_id,
                    turn=self.turn,
                    step=self.step,
                    started_at=ts,
                    ended_at=ts,
                    retries=self.pending_retries,
                    provider=prov,
                    model=mdl,
                )
                self.calls_by_msg_id[msg_id] = call
                self.slots.append(call)
                self.pending_retries = 0
                if not obj.get("isApiErrorMessage"):
                    self.pending_terminal_error = None
            else:
                if ts is not None:
                    if call.started_at is None or ts < call.started_at:
                        call.started_at = ts
                    if call.ended_at is None or ts > call.ended_at:
                        call.ended_at = ts
                if prov and not call.provider:
                    call.provider = prov
                if mdl and not call.model:
                    call.model = mdl

            # Always take usage from the latest entry in the group (R9.2)
            usage = msg.get("usage")
            if isinstance(usage, Mapping) and usage:
                in_tok = _opt_int(usage.get("input_tokens"))
                cache_r = _opt_int(usage.get("cache_read_input_tokens"))
                cache_w = _opt_int(usage.get("cache_creation_input_tokens"))
                out_tok = _opt_int(usage.get("output_tokens"))
                details = (
                    usage.get("output_tokens_details")
                    if isinstance(usage.get("output_tokens_details"), Mapping)
                    else {}
                )
                think_tok = _opt_int(details.get("thinking_tokens"))

                call.input_tokens = in_tok
                call.cache_read_tokens = cache_r
                call.cache_write_tokens = cache_w
                call.output_tokens = out_tok
                call.reasoning_tokens = think_tok

                if any(x is not None for x in (in_tok, cache_r, cache_w)):
                    prompt_tok = (in_tok or 0) + (cache_r or 0) + (cache_w or 0)
                    if prompt_tok > 0:
                        if self.awaiting_tokens_after:
                            for comp in self.awaiting_tokens_after:
                                if comp.tokens_after is None:
                                    comp.tokens_after = prompt_tok
                            self.awaiting_tokens_after.clear()
                        self.last_prompt_tokens = prompt_tok

            stop_reason = msg.get("stop_reason")
            if stop_reason == "max_tokens":
                self.pending_terminal_error = ("max-tokens", ts)

            blocks = msg.get("content") if isinstance(msg.get("content"), list) else []
            for blk in blocks:
                if not isinstance(blk, Mapping):
                    continue
                if blk.get("type") == "tool_use":
                    cid = (
                        blk["id"]
                        if isinstance(blk.get("id"), str) and blk["id"]
                        else None
                    )
                    if cid and cid in call.seen_tool_ids:
                        continue
                    if cid:
                        call.seen_tool_ids.add(cid)
                    if (
                        cid
                        and self.seen_tool_ids is not None
                        and cid in self.seen_tool_ids
                    ):
                        self.skipped_tool_ids.add(cid)
                        continue
                    if cid and self.seen_tool_ids is not None:
                        self.seen_tool_ids.add(cid)

                    tname = str(blk.get("name") or "")
                    raw_input = blk.get("input")
                    redacted_parsed: Any = None
                    raw_args_str: str | None = None
                    if isinstance(raw_input, Mapping):
                        redacted_parsed = redact_obj(raw_input)
                        raw_args_str = json.dumps(
                            redacted_parsed, sort_keys=True, ensure_ascii=False
                        )
                    elif isinstance(raw_input, str):
                        raw_args_str = redact_secrets(raw_input)
                    elif raw_input is not None:
                        redacted_parsed = redact_obj(raw_input)
                        raw_args_str = json.dumps(
                            redacted_parsed, sort_keys=True, ensure_ascii=False
                        )

                    fp = _basic_args_fingerprint(tname, redacted_parsed, raw_args_str)
                    tc = _PendingToolCall(
                        turn=call.turn,
                        step=call.step,
                        call_id=cid,
                        name=tname,
                        args_fingerprint=fp,
                        raw_args=raw_args_str,
                        redacted_parsed=redacted_parsed,
                        started_at=ts,
                        available_tools=self.available_tools,
                    )
                    self.slots.append(tc)
                    if cid:
                        self.pending_tools[cid] = tc

    def build(
        self,
        fallback_session_id: str,
        parse_failures: int = 0,
        unknown_event_types: tuple[str, ...] = (),
    ) -> Trajectory:
        if self.turn_open:
            if self.pending_retries > 0 and self.pending_terminal_error is not None:
                self.step += 1
                err_reason, err_at = self.pending_terminal_error
                self.turn_has_activity = True
                self.slots.append(
                    ModelCall(
                        step=self.step,
                        started_at=self.trigger_ts,
                        ended_at=err_at,
                        retries=self.pending_retries,
                        turn=self.turn,
                    )
                )
            self.close_open_turn("completed", self.turn_last_ts)

        base_sid = self.session_id or fallback_session_id
        if self.key == "main":
            final_sid = base_sid
        elif self.agent_id:
            final_sid = f"{base_sid}:sidechain:{self.agent_id}"
        else:
            suffix = self.key.split(":", 1)[-1]
            final_sid = f"{base_sid}:sidechain:{suffix}"

        final_events: list[TrajectoryEvent] = []
        for slot in self.slots:
            if isinstance(slot, _PendingModelCall):
                final_events.append(
                    ModelCall(
                        step=slot.step,
                        started_at=slot.started_at,
                        first_token_at=None,
                        ended_at=slot.ended_at,
                        input_tokens=slot.input_tokens,
                        cache_read_tokens=slot.cache_read_tokens,
                        cache_write_tokens=slot.cache_write_tokens,
                        output_tokens=slot.output_tokens,
                        reasoning_tokens=slot.reasoning_tokens,
                        retries=slot.retries,
                        provider=slot.provider,
                        model=slot.model,
                        turn=slot.turn,
                        response_id=slot.response_id,
                    )
                )
            elif isinstance(slot, _PendingToolCall):
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
                ed = _build_edit_from_cc_tool(slot)
                if ed is not None:
                    final_events.append(ed)
            elif isinstance(slot, _PendingCompaction):
                final_events.append(
                    Compaction(
                        started_at=slot.started_at,
                        ended_at=slot.ended_at,
                        ok=slot.ok,
                        tokens_before=slot.tokens_before,
                        tokens_after=slot.tokens_after,
                        reason=slot.reason,
                    )
                )
            else:
                final_events.append(slot)

        return Trajectory(
            source="claude-code",
            session_id=final_sid,
            cwd=self.cwd,
            model_routes=tuple(self.routes),
            started_at=self.started_at,
            events=tuple(final_events),
            parse_failures=parse_failures,
            unknown_event_types=unknown_event_types,
        )


def parse_claude_code_file(
    path: str | Path,
    *,
    seen_message_ids: set[str] | None = None,
    seen_tool_ids: set[str] | None = None,
) -> list[Trajectory]:
    """Stream-parse a Claude Code .jsonl file into main + inline-sidechain trajectories."""
    file_path = Path(path).expanduser()
    fallback_sid = file_path.stem
    parse_failures = 0
    unknown_types: list[str] = []
    unknown_seen: set[str] = set()

    chains: dict[str, _ChainBuilder] = {
        "main": _ChainBuilder(
            "main",
            seen_message_ids=seen_message_ids,
            seen_tool_ids=seen_tool_ids,
        )
    }
    uuid_to_chain_key: dict[str, str] = {}
    sidechain_seq = 0

    def resolve_chain_key(obj: Mapping[str, Any]) -> str:
        if not obj.get("isSidechain"):
            return "main"
        agent_id = obj.get("agentId")
        if isinstance(agent_id, str) and agent_id:
            return f"agent:{agent_id}"
        parent_uuid = obj.get("parentUuid")
        if isinstance(parent_uuid, str) and parent_uuid in uuid_to_chain_key:
            parent_key = uuid_to_chain_key[parent_uuid]
            if parent_key != "main":
                return parent_key
        nonlocal sidechain_seq
        sidechain_seq += 1
        uid = obj.get("uuid") if isinstance(obj.get("uuid"), str) else str(sidechain_seq)
        return f"sc:{uid}"

    try:
        with open(file_path, "rb") as fh:
            wrapper = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
            for raw_line in wrapper:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError:
                    parse_failures += 1
                    continue
                if not isinstance(obj, Mapping):
                    parse_failures += 1
                    continue

                ev_type = obj.get("type")
                if not isinstance(ev_type, str) or not ev_type:
                    parse_failures += 1
                    continue

                if (
                    ev_type not in KNOWN_CLAUDE_ENTRY_TYPES
                    and ev_type not in unknown_seen
                ):
                    unknown_seen.add(ev_type)
                    unknown_types.append(ev_type)

                key = resolve_chain_key(obj)
                uid = obj.get("uuid")
                if isinstance(uid, str) and uid:
                    uuid_to_chain_key[uid] = key

                builder = chains.get(key)
                if builder is None:
                    builder = _ChainBuilder(
                        key,
                        seen_message_ids=seen_message_ids,
                        seen_tool_ids=seen_tool_ids,
                    )
                    chains[key] = builder
                builder.handle_entry(obj)
    except OSError:
        parse_failures += 1

    active_builders = [
        b
        for k, b in chains.items()
        if b.has_meaningful_content
        or (
            k == "main"
            and len(chains) == 1
            and b.skipped_duplicate_calls == 0
        )
    ]
    if not active_builders and seen_message_ids is None:
        active_builders = [chains["main"]]

    trajectories: list[Trajectory] = []
    for idx, builder in enumerate(active_builders):
        trajectories.append(
            builder.build(
                fallback_session_id=fallback_sid,
                parse_failures=parse_failures if idx == 0 else 0,
                unknown_event_types=tuple(unknown_types) if idx == 0 else (),
            )
        )
    return trajectories


def parse_claude_code_corpus(root: str | Path) -> list[Trajectory]:
    """Parse all top-level Claude Code transcripts under root without double-counting forked/resumed history."""
    files = discover_claude_code_files(root)
    ordered_files = sorted(
        files,
        key=lambda p: (_safe_file_mtime_ns(p), str(p)),
    )
    seen_message_ids: set[str] = set()
    seen_tool_ids: set[str] = set()
    trajectories: list[Trajectory] = []
    for fpath in ordered_files:
        trajectories.extend(
            parse_claude_code_file(
                fpath,
                seen_message_ids=seen_message_ids,
                seen_tool_ids=seen_tool_ids,
            )
        )
    trajectories.sort(key=lambda t: (t.started_at or 0, t.session_id))
    return trajectories


def _safe_file_mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0
