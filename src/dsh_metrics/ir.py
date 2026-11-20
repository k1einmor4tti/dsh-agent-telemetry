"""Normalized trajectory intermediate representation (IR)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json


def _validate_epoch_ms(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"{name} must be positive epoch milliseconds or None, got {value!r}"
        )


def _validate_optional_int(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be int or None, got {value!r}")


@dataclass(frozen=True, slots=True)
class ModelCall:
    step: int | None
    started_at: int | None = None
    first_token_at: int | None = None
    ended_at: int | None = None
    input_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    retries: int | None = None
    provider: str | None = None
    model: str | None = None
    turn: int | None = None

    def __post_init__(self) -> None:
        _validate_optional_int("step", self.step)
        _validate_optional_int("turn", self.turn)
        _validate_epoch_ms("started_at", self.started_at)
        _validate_epoch_ms("first_token_at", self.first_token_at)
        _validate_epoch_ms("ended_at", self.ended_at)
        _validate_optional_int("input_tokens", self.input_tokens)
        _validate_optional_int("cache_read_tokens", self.cache_read_tokens)
        _validate_optional_int("cache_write_tokens", self.cache_write_tokens)
        _validate_optional_int("output_tokens", self.output_tokens)
        _validate_optional_int("reasoning_tokens", self.reasoning_tokens)
        _validate_optional_int("retries", self.retries)


@dataclass(frozen=True, slots=True)
class ToolCall:
    step: int | None
    call_id: str | None
    name: str
    args_fingerprint: str | None = None
    raw_args: str | None = None
    started_at: int | None = None
    ended_at: int | None = None
    is_error: bool | None = None
    exit_code: int | None = None
    output_bytes: int | None = None
    truncated: bool | None = None
    available_tools: tuple[str, ...] | None = None
    turn: int | None = None
    error_signature: str | None = None
    output_tail: str | None = None

    def __post_init__(self) -> None:
        _validate_optional_int("step", self.step)
        _validate_optional_int("turn", self.turn)
        _validate_epoch_ms("started_at", self.started_at)
        _validate_epoch_ms("ended_at", self.ended_at)
        _validate_optional_int("exit_code", self.exit_code)
        _validate_optional_int("output_bytes", self.output_bytes)
        if self.output_tail is not None and len(self.output_tail.encode("utf-8")) > 2048:
            raise ValueError("output_tail must be at most 2048 UTF-8 bytes")
        if self.raw_args is not None and not isinstance(self.raw_args, str):
            if isinstance(self.raw_args, (Mapping, list, tuple)):
                serialized = json.dumps(
                    self.raw_args, sort_keys=True, ensure_ascii=False
                )
                object.__setattr__(self, "raw_args", serialized)
            else:
                raise TypeError(
                    f"raw_args must be str, mapping, sequence, or None, got {type(self.raw_args)!r}"
                )
        if self.available_tools is not None and not isinstance(
            self.available_tools, tuple
        ):
            if isinstance(self.available_tools, str) or not isinstance(
                self.available_tools, Iterable
            ):
                raise TypeError(
                    "available_tools must be an iterable of tool names or None"
                )
            object.__setattr__(self, "available_tools", tuple(self.available_tools))


@dataclass(frozen=True, slots=True)
class Edit:
    step: int | None
    path: str
    before_hash: str | None = None
    after_hash: str | None = None
    turn: int | None = None
    at: int | None = None

    def __post_init__(self) -> None:
        _validate_optional_int("step", self.step)
        _validate_optional_int("turn", self.turn)
        _validate_epoch_ms("at", self.at)


@dataclass(frozen=True, slots=True)
class Compaction:
    started_at: int | None = None
    ended_at: int | None = None
    ok: bool | None = None
    tokens_before: int | None = None
    tokens_after: int | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        _validate_epoch_ms("started_at", self.started_at)
        _validate_epoch_ms("ended_at", self.ended_at)
        _validate_optional_int("tokens_before", self.tokens_before)
        _validate_optional_int("tokens_after", self.tokens_after)


@dataclass(frozen=True, slots=True)
class TurnEnd:
    turn: int | None
    reason: str | None = None
    at: int | None = None

    def __post_init__(self) -> None:
        _validate_optional_int("turn", self.turn)
        _validate_epoch_ms("at", self.at)


@dataclass(frozen=True, slots=True)
class Outcome:
    task_id: str | None
    result: str | None
    source: str | None = None
    at: int | None = None

    def __post_init__(self) -> None:
        _validate_epoch_ms("at", self.at)


TrajectoryEvent = ModelCall | ToolCall | Edit | Compaction | TurnEnd | Outcome


def event_timestamp(event: TrajectoryEvent) -> int | None:
    """Return the primary epoch-ms timestamp of an IR event for timeline placement."""
    if isinstance(event, (ModelCall, ToolCall, Compaction)):
        return event.started_at if event.started_at is not None else event.ended_at
    return event.at


@dataclass(frozen=True, slots=True)
class Trajectory:
    source: str
    session_id: str
    cwd: str | None = None
    model_routes: tuple[str, ...] = ()
    started_at: int | None = None
    events: tuple[TrajectoryEvent, ...] = ()
    parse_failures: int = 0
    unknown_event_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_epoch_ms("started_at", self.started_at)
        if (
            isinstance(self.parse_failures, bool)
            or not isinstance(self.parse_failures, int)
            or self.parse_failures < 0
        ):
            raise ValueError(
                f"parse_failures must be a non-negative int, got {self.parse_failures!r}"
            )
        if not isinstance(self.model_routes, tuple):
            if isinstance(self.model_routes, str) or not isinstance(
                self.model_routes, Iterable
            ):
                raise TypeError("model_routes must be an iterable of strings")
            object.__setattr__(self, "model_routes", tuple(self.model_routes))
        if not isinstance(self.events, tuple):
            if not isinstance(self.events, Iterable):
                raise TypeError("events must be an iterable of trajectory events")
            object.__setattr__(self, "events", tuple(self.events))
        if not isinstance(self.unknown_event_types, tuple):
            if isinstance(self.unknown_event_types, str) or not isinstance(
                self.unknown_event_types, Iterable
            ):
                raise TypeError("unknown_event_types must be an iterable of strings")
            object.__setattr__(
                self, "unknown_event_types", tuple(self.unknown_event_types)
            )
