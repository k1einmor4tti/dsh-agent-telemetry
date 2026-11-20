"""Synthetic tests for M3 Claude Code adapter, R9 message.id grouping, R10 discovery, and R3 redaction."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dsh_metrics.adapters.claude_code import (
    discover_claude_code_files,
    parse_claude_code_corpus,
    parse_claude_code_file,
    parse_iso_ms,
)
from dsh_metrics.ir import Compaction, Edit, ModelCall, ToolCall, TurnEnd


def _write_jsonl(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_r9_split_response_three_entries_final_usage_and_interleaved_tool_result(
    tmp_path: Path,
) -> None:
    """R9: 3 entries sharing message.id (even with interim usage & interleaved tool_result) -> 1 ModelCall."""
    t_earliest = "2026-10-01T10:00:02.000Z"
    t_middle = "2026-10-01T10:00:02.500Z"
    t_tool_res_1 = "2026-10-01T10:00:02.650Z"
    t_latest = "2026-10-01T10:00:02.900Z"

    entries = [
        {"type": "file-history-snapshot", "messageId": "m0", "snapshot": {}},
        {"type": "cost-state", "totalCostUsd": 0.12},
        {"type": "attachment", "uuid": "att-1"},
        {
            "type": "user",
            "sessionId": "cc-sess-01",
            "cwd": "/workspace/synth-cc",
            "uuid": "u1",
            "parentUuid": None,
            "isSidechain": False,
            "timestamp": "2026-10-01T10:00:00.000Z",
            "message": {"role": "user", "content": "Fix failing test"},
        },
        {
            "type": "system",
            "subtype": "api_error",
            "sessionId": "cc-sess-01",
            "uuid": "err1",
            "parentUuid": "u1",
            "isSidechain": False,
            "retryAttempt": 1,
            "maxRetries": 5,
            "retryInMs": 500,
            "error": {"status": 529, "message": "Overloaded"},
            "timestamp": "2026-10-01T10:00:00.500Z",
        },
        # Entry 1 of 3 for msg_01: smaller interim output_tokens (15)
        {
            "type": "assistant",
            "sessionId": "cc-sess-01",
            "uuid": "a1_think",
            "parentUuid": "u1",
            "isSidechain": False,
            "thinkingDurationMs": 800,
            "timestamp": t_earliest,
            "message": {
                "id": "msg_01",
                "model": "claude-opus-5-5",
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": ""}],
                "usage": {
                    "input_tokens": 300,
                    "cache_read_input_tokens": 12000,
                    "cache_creation_input_tokens": 400,
                    "output_tokens": 15,
                    "output_tokens_details": {"thinking_tokens": 15},
                },
            },
        },
        # Entry 2 of 3 for msg_01: interim output_tokens (85) + first tool_use
        {
            "type": "assistant",
            "sessionId": "cc-sess-01",
            "uuid": "a1_tool1",
            "parentUuid": "a1_think",
            "isSidechain": False,
            "timestamp": t_middle,
            "message": {
                "id": "msg_01",
                "model": "claude-opus-5-5",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_bash_1",
                        "name": "Bash",
                        "input": {"command": "pytest -q"},
                    }
                ],
                "usage": {
                    "input_tokens": 300,
                    "cache_read_input_tokens": 12000,
                    "cache_creation_input_tokens": 400,
                    "output_tokens": 85,
                    "output_tokens_details": {"thinking_tokens": 60},
                },
            },
        },
        # Interleaved user tool_result arriving before Entry 3 of msg_01 finishes!
        {
            "type": "user",
            "sessionId": "cc-sess-01",
            "uuid": "u_res_1",
            "parentUuid": "a1_tool1",
            "isSidechain": False,
            "timestamp": t_tool_res_1,
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_bash_1",
                        "is_error": True,
                        "content": "Exit code 2\nAssertionError: expected 4\n<persisted-output>",
                    }
                ],
            },
        },
        # Entry 3 of 3 for msg_01: final output_tokens (180) + second tool_use
        {
            "type": "assistant",
            "sessionId": "cc-sess-01",
            "uuid": "a1_tool2",
            "parentUuid": "a1_tool1",
            "isSidechain": False,
            "timestamp": t_latest,
            "message": {
                "id": "msg_01",
                "model": "claude-opus-5-5",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_edit_1",
                        "name": "Edit",
                        "input": {
                            "file_path": "src/calc.py",
                            "old_string": "a - b",
                            "new_string": "a + b",
                        },
                    }
                ],
                "usage": {
                    "input_tokens": 300,
                    "cache_read_input_tokens": 12000,
                    "cache_creation_input_tokens": 400,
                    "output_tokens": 180,
                    "output_tokens_details": {"thinking_tokens": 60},
                },
            },
        },
        # Sidechain subagent entries interleaved in the same file
        {
            "type": "user",
            "sessionId": "cc-sess-01",
            "agentId": "sub-explore-1",
            "uuid": "sc_u1",
            "parentUuid": "a1_tool1",
            "isSidechain": True,
            "timestamp": "2026-10-01T10:00:02.950Z",
            "message": {"role": "user", "content": "Search helper"},
        },
        {
            "type": "assistant",
            "sessionId": "cc-sess-01",
            "agentId": "sub-explore-1",
            "requestId": "req_fallback_sub_01",
            "uuid": "sc_a1",
            "parentUuid": "sc_u1",
            "isSidechain": True,
            "timestamp": "2026-10-01T10:00:03.100Z",
            "message": {
                "model": "claude-haiku-5-5",
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": ""}],
                "usage": {"input_tokens": 150, "output_tokens": 5},
            },
        },
        {
            "type": "assistant",
            "sessionId": "cc-sess-01",
            "agentId": "sub-explore-1",
            "requestId": "req_fallback_sub_01",
            "uuid": "sc_a2",
            "parentUuid": "sc_a1",
            "isSidechain": True,
            "timestamp": "2026-10-01T10:00:03.400Z",
            "message": {
                "model": "claude-haiku-5-5",
                "role": "assistant",
                "content": [{"type": "text", "text": "Found helper"}],
                "usage": {"input_tokens": 150, "output_tokens": 25},
            },
        },
        # Second tool result on main chain
        {
            "type": "user",
            "sessionId": "cc-sess-01",
            "uuid": "u_res_2",
            "parentUuid": "a1_tool2",
            "isSidechain": False,
            "timestamp": "2026-10-01T10:00:04.000Z",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_edit_1",
                        "is_error": False,
                        "content": "Updated src/calc.py",
                    }
                ],
            },
        },
        # Compaction boundary + compact summary
        {
            "type": "system",
            "subtype": "compact_boundary",
            "sessionId": "cc-sess-01",
            "uuid": "sys_comp",
            "isSidechain": False,
            "timestamp": "2026-10-01T10:00:05.000Z",
            "compactMetadata": {
                "trigger": "auto",
                "preTokens": 12700,
                "postTokens": 4200,
                "durationMs": 600,
            },
        },
        {
            "type": "user",
            "sessionId": "cc-sess-01",
            "uuid": "u_comp_sum",
            "isSidechain": False,
            "isCompactSummary": True,
            "timestamp": "2026-10-01T10:00:05.010Z",
            "message": {"role": "user", "content": "Summary of prior context"},
        },
        {
            "type": "system",
            "subtype": "turn_duration",
            "sessionId": "cc-sess-01",
            "isSidechain": False,
            "durationMs": 5100,
            "timestamp": "2026-10-01T10:00:05.100Z",
        },
    ]

    f = tmp_path / "proj-slug" / "cc-sess-01.jsonl"
    nested_subagent = (
        tmp_path / "proj-slug" / "cc-sess-01" / "subagents" / "agent-ignored.jsonl"
    )
    _write_jsonl(f, [json.dumps(e) for e in entries])
    _write_jsonl(nested_subagent, [json.dumps(entries[0])])

    # R10.2: only top-level <slug>/*.jsonl discovered, nested subagents ignored
    discovered = discover_claude_code_files(tmp_path)
    assert discovered == [f]

    trajs = parse_claude_code_file(f)
    assert len(trajs) == 2

    main_traj = trajs[0]
    sub_traj = trajs[1]

    assert main_traj.source == "claude-code"
    assert main_traj.session_id == "cc-sess-01"
    assert main_traj.cwd == "/workspace/synth-cc"
    assert main_traj.model_routes == ("anthropic/claude-opus-5-5",)
    assert main_traj.parse_failures == 0

    mcs = [e for e in main_traj.events if isinstance(e, ModelCall)]
    tcs = [e for e in main_traj.events if isinstance(e, ToolCall)]
    edits = [e for e in main_traj.events if isinstance(e, Edit)]
    comps = [e for e in main_traj.events if isinstance(e, Compaction)]
    tes = [e for e in main_traj.events if isinstance(e, TurnEnd)]

    # R9: 3 assistant entries -> 1 ModelCall with final usage & [earliest, latest] timestamps
    assert len(mcs) == 1
    assert mcs[0].response_id == "msg_01"
    assert mcs[0].input_tokens == 300
    assert mcs[0].cache_read_tokens == 12000
    assert mcs[0].cache_write_tokens == 400
    assert mcs[0].output_tokens == 180
    assert mcs[0].reasoning_tokens == 60
    assert mcs[0].retries == 1
    assert mcs[0].started_at == parse_iso_ms(t_earliest)
    assert mcs[0].ended_at == parse_iso_ms(t_latest)

    assert len(tcs) == 2
    assert tcs[0].call_id == "toolu_bash_1"
    assert tcs[0].name == "Bash"
    assert tcs[0].exit_code == 2
    assert tcs[0].is_error is True
    assert tcs[0].truncated is True
    assert tcs[1].call_id == "toolu_edit_1"
    assert tcs[1].name == "Edit"
    assert tcs[1].is_error is False

    assert len(edits) == 1
    assert edits[0].path == "src/calc.py"
    assert edits[0].before_hash != edits[0].after_hash

    assert len(comps) == 1
    assert comps[0].ok is True
    assert comps[0].tokens_before == 12700
    assert comps[0].tokens_after == 4200
    assert comps[0].reason == "auto"

    assert len(tes) == 1
    assert tes[0].reason == "completed"

    # Sidechain grouped via requestId fallback with final usage
    assert sub_traj.session_id == "cc-sess-01:sidechain:sub-explore-1"
    assert sub_traj.model_routes == ("anthropic/claude-haiku-5-5",)
    sub_mcs = [e for e in sub_traj.events if isinstance(e, ModelCall)]
    assert len(sub_mcs) == 1
    assert sub_mcs[0].response_id == "req_fallback_sub_01"
    assert sub_mcs[0].input_tokens == 150
    assert sub_mcs[0].output_tokens == 25


def test_r11_two_files_sharing_response_first_seen_keeps_duplicates_ignored(
    tmp_path: Path,
) -> None:
    shared_turn = [
        {
            "type": "user",
            "sessionId": "s-orig",
            "timestamp": "2026-10-03T01:00:00.000Z",
            "message": {"role": "user", "content": "Hello"},
        },
        {
            "type": "assistant",
            "sessionId": "s-orig",
            "timestamp": "2026-10-03T01:00:01.000Z",
            "message": {
                "id": "msg_shared_1",
                "model": "claude-opus-5-5",
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hi"},
                    {
                        "type": "tool_use",
                        "id": "toolu_shared_1",
                        "name": "Read",
                        "input": {"file_path": "README.md"},
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 40},
            },
        },
        {
            "type": "user",
            "sessionId": "s-orig",
            "timestamp": "2026-10-03T01:00:02.000Z",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_shared_1",
                        "is_error": False,
                        "content": "# Readme",
                    }
                ],
            },
        },
    ]
    continued_turn = [
        {
            "type": "user",
            "sessionId": "s-cont",
            "timestamp": "2026-10-03T01:01:00.000Z",
            "message": {"role": "user", "content": "Next"},
        },
        {
            "type": "assistant",
            "sessionId": "s-cont",
            "timestamp": "2026-10-03T01:01:02.000Z",
            "message": {
                "id": "msg_unique_2",
                "model": "claude-opus-5-5",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_unique_2",
                        "name": "Bash",
                        "input": {"command": "pytest"},
                    }
                ],
                "usage": {"input_tokens": 50, "output_tokens": 60},
            },
        },
        {
            "type": "user",
            "sessionId": "s-cont",
            "timestamp": "2026-10-03T01:01:03.000Z",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_unique_2",
                        "is_error": False,
                        "content": "1 passed",
                    }
                ],
            },
        },
    ]

    orig_path = tmp_path / "slug-a" / "s-orig.jsonl"
    cont_path = tmp_path / "slug-b" / "s-cont.jsonl"
    _write_jsonl(orig_path, [json.dumps(e) for e in shared_turn])
    _write_jsonl(
        cont_path,
        [json.dumps({**e, "sessionId": "s-cont"}) for e in shared_turn + continued_turn],
    )
    os.utime(orig_path, (1_760_000_000, 1_760_000_000))
    os.utime(cont_path, (1_760_000_060, 1_760_000_060))

    # Single-file parse populates response_id and call_id on every call
    orig_single = parse_claude_code_file(orig_path)[0]
    cont_single = parse_claude_code_file(cont_path)[0]
    assert [
        e.response_id for e in orig_single.events if isinstance(e, ModelCall)
    ] == ["msg_shared_1"]
    assert [
        e.response_id for e in cont_single.events if isinstance(e, ModelCall)
    ] == ["msg_shared_1", "msg_unique_2"]
    assert [e.call_id for e in cont_single.events if isinstance(e, ToolCall)] == [
        "toolu_shared_1",
        "toolu_unique_2",
    ]

    # Corpus parse: first-seen trajectory keeps shared response & tool call; second keeps only delta
    trajs = parse_claude_code_corpus(tmp_path)
    assert len(trajs) == 2
    by_sid = {t.session_id: t for t in trajs}
    orig_mcs = [e for e in by_sid["s-orig"].events if isinstance(e, ModelCall)]
    cont_mcs = [e for e in by_sid["s-cont"].events if isinstance(e, ModelCall)]
    orig_tcs = [e for e in by_sid["s-orig"].events if isinstance(e, ToolCall)]
    cont_tcs = [e for e in by_sid["s-cont"].events if isinstance(e, ToolCall)]

    assert [m.response_id for m in orig_mcs] == ["msg_shared_1"]
    assert [t.call_id for t in orig_tcs] == ["toolu_shared_1"]
    assert [m.response_id for m in cont_mcs] == ["msg_unique_2"]
    assert [t.call_id for t in cont_tcs] == ["toolu_unique_2"]
    assert sum(m.output_tokens or 0 for m in orig_mcs + cont_mcs) == 100


def test_claude_code_malformed_lines_unknown_types_and_redaction(
    tmp_path: Path,
) -> None:
    sk_secret = "sk-ant-api03-SuperSecretClaudeKey123456"
    bearer_secret = "ccBearerTokenSecretValue999"
    huge_out = (
        ("synthetic line\n" * 500)
        + f"Authorization: Bearer {bearer_secret} key={sk_secret}\nExit code 1"
    )
    raw_lines = [
        "{bad json line",
        "[1, 2, 3]",
        json.dumps({"no_type_field": True}),
        json.dumps(
            {
                "type": "future-cc-telemetry",
                "sessionId": "cc-red-1",
                "timestamp": "2026-10-02T00:00:00.000Z",
            }
        ),
        json.dumps(
            {
                "type": "assistant",
                "sessionId": "cc-red-1",
                "uuid": "a1",
                "timestamp": "2026-10-02T00:00:01.000Z",
                "message": {
                    "id": "msg_red_1",
                    "model": "claude-sonnet-5-5",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_1",
                            "name": "Bash",
                            "input": {
                                "command": f'curl -H "Authorization: Bearer {bearer_secret}" https://api.example.test?token={sk_secret}'
                            },
                        }
                    ],
                    "usage": {"input_tokens": 50, "output_tokens": 20},
                },
            }
        ),
        json.dumps(
            {
                "type": "user",
                "sessionId": "cc-red-1",
                "uuid": "u1",
                "timestamp": "2026-10-02T00:00:02.000Z",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_1",
                            "is_error": True,
                            "content": huge_out,
                        }
                    ],
                },
            }
        ),
        json.dumps(
            {
                "type": "user",
                "sessionId": "cc-red-1",
                "uuid": "u_abort",
                "timestamp": "2026-10-02T00:00:03.000Z",
                "message": {
                    "role": "user",
                    "content": "[Request interrupted by user]",
                },
            }
        ),
    ]

    f = tmp_path / "slug" / "cc-red-1.jsonl"
    _write_jsonl(f, raw_lines)

    trajs = parse_claude_code_file(f)
    assert len(trajs) == 1
    traj = trajs[0]
    assert traj.parse_failures == 3
    assert traj.unknown_event_types == ("future-cc-telemetry",)

    tc = [e for e in traj.events if isinstance(e, ToolCall)][0]
    assert tc.raw_args is not None
    assert tc.args_fingerprint is not None
    assert tc.output_tail is not None
    assert len(tc.output_tail.encode("utf-8")) <= 2048

    for secret in (sk_secret, bearer_secret):
        assert secret not in tc.raw_args
        assert secret not in tc.args_fingerprint
        assert secret not in tc.output_tail
        assert secret not in (tc.error_signature or "")
        assert secret not in repr(traj)

    te = [e for e in traj.events if isinstance(e, TurnEnd)][0]
    assert te.reason == "aborted"


def test_claude_code_resumed_session_id_and_inter_turn_idle_boundary(
    tmp_path: Path,
) -> None:
    """R14: resumed transcripts adopt latest sessionId, ignore tools-used, and close turns at turn_last_ts."""
    entries = [
        {
            "type": "tools-used",
            "sessionId": "cc-old-prefix",
            "tools": ["Read"],
            "timestamp": "2026-10-03T09:00:00.000Z",
        },
        {
            "type": "user",
            "sessionId": "cc-old-prefix",
            "cwd": "/synth/repo",
            "timestamp": "2026-10-03T09:00:01.000Z",
            "message": {"role": "user", "content": "Turn 1"},
        },
        {
            "type": "assistant",
            "sessionId": "cc-old-prefix",
            "timestamp": "2026-10-03T09:00:02.000Z",
            "message": {
                "id": "msg_t1",
                "model": "claude-sonnet-5-5",
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": ""}],
                "usage": {"input_tokens": 40, "output_tokens": 5},
            },
        },
        {
            "type": "assistant",
            "sessionId": "cc-old-prefix",
            "timestamp": "2026-10-03T09:00:04.000Z",
            "message": {
                "id": "msg_t1",
                "model": "claude-sonnet-5-5",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_t1",
                        "name": "Bash",
                        "input": {"command": "pytest -q"},
                    }
                ],
                "usage": {"input_tokens": 40, "output_tokens": 18},
            },
        },
        {
            "type": "user",
            "sessionId": "cc-old-prefix",
            "timestamp": "2026-10-03T09:00:05.000Z",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_t1",
                        "is_error": False,
                        "content": "1 passed",
                    }
                ],
            },
        },
        # Resumed 30 minutes later with new sessionId: Turn 1 must close at 09:00:05Z, not 09:30:00Z
        {
            "type": "user",
            "sessionId": "cc-resumed-final",
            "timestamp": "2026-10-03T09:30:00.000Z",
            "message": {"role": "user", "content": "Turn 2"},
        },
        {
            "type": "assistant",
            "sessionId": "cc-resumed-final",
            "timestamp": "2026-10-03T09:30:02.000Z",
            "message": {
                "id": "msg_t2",
                "model": "claude-sonnet-5-5",
                "role": "assistant",
                "content": [{"type": "text", "text": "Done"}],
                "usage": {"input_tokens": 50, "output_tokens": 10},
            },
        },
    ]
    f = tmp_path / "slug" / "cc-resumed-final.jsonl"
    _write_jsonl(f, [json.dumps(e) for e in entries])

    trajs = parse_claude_code_file(f)
    assert len(trajs) == 1
    traj = trajs[0]
    assert traj.session_id == "cc-resumed-final"

    mcs = [e for e in traj.events if isinstance(e, ModelCall)]
    tcs = [e for e in traj.events if isinstance(e, ToolCall)]
    tes = [e for e in traj.events if isinstance(e, TurnEnd)]

    assert mcs[0].first_token_at is None
    assert tcs[0].available_tools is None
    assert tes[0].turn == 1
    assert tes[0].at == parse_iso_ms("2026-10-03T09:00:05.000Z")

