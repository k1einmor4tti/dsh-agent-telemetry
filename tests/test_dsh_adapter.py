"""Synthetic tests for M2 DSH adapter and secret redaction."""

from __future__ import annotations

import json
from pathlib import Path

import zstandard as zstd

from dsh_metrics.adapters.dsh import discover_dsh_files, parse_dsh_session
from dsh_metrics.ir import Compaction, Edit, ModelCall, ToolCall, TurnEnd
from dsh_metrics.redact import redact_secrets


def _write_zstd_lines(path: Path, lines: list[str], trailing_garbage: bytes = b"") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cctx = zstd.ZstdCompressor()
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    compressed = cctx.compress(payload) + trailing_garbage
    path.write_bytes(compressed)


def test_parse_dsh_session_end_to_end(tmp_path: Path) -> None:
    events = [
        {
            "type": "session",
            "version": 0,
            "id": "sess-synth-01",
            "createdAt": 1_700_000_000_000,
            "cwd": "/workspace/synthetic-project",
        },
        {
            "type": "request/header",
            "seq": 1,
            "time": 1_700_000_000_010,
            "data": {
                "header": {
                    "config": {"provider": "deepseek", "model": "deepseek-chat"},
                    "system": "You are helpful.",
                    "tools": [
                        {"name": "bash", "description": "shell", "parameters": {}},
                        {"name": "read", "description": "read", "parameters": {}},
                        {"name": "edit", "description": "edit", "parameters": {}},
                        {"name": "write", "description": "write", "parameters": {}},
                    ],
                }
            },
        },
        {
            "type": "request/context",
            "seq": 2,
            "time": 1_700_000_000_012,
            "data": {
                "provider": "deepseek",
                "model": "deepseek-chat",
                "contextWindow": 128000,
            },
        },
        {
            "type": "turn/start",
            "seq": 3,
            "time": 1_700_000_000_020,
            "data": {"turn": 1},
        },
        {
            "type": "step/start",
            "seq": 4,
            "time": 1_700_000_000_050,
            "data": {"turn": 1, "step": 1},
        },
        {
            "type": "llm/retry",
            "seq": 5,
            "time": 1_700_000_000_150,
            "data": {
                "retryId": "r1",
                "turn": 1,
                "step": 1,
                "retry": 1,
                "failure": {"code": "TIMEOUT", "message": "Request timed out."},
            },
        },
        {
            "type": "llm/retry-started",
            "seq": 6,
            "time": 1_700_000_000_200,
            "data": {"retryId": "r1", "turn": 1, "step": 1, "retry": 1},
        },
        {
            "type": "reasoning-chunks",
            "seq0": 7,
            "time0": 1_700_000_000_350,
            "data": {"turn": 1, "step": 1, "index": 0, "dt": [0], "texts": ["hmm"]},
        },
        {
            "type": "tool-call-chunks",
            "seq0": 8,
            "time0": 1_700_000_000_450,
            "data": {
                "turn": 1,
                "step": 1,
                "index": 1,
                "dt": [0],
                "id": "call_bash_1",
                "name": "bash",
                "args": ['{"command":"pytest"}'],
            },
        },
        {
            "type": "assistant/message",
            "seq": 9,
            "time": 1_700_000_000_600,
            "data": {
                "turn": 1,
                "step": 1,
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Running tests"}],
                    "source": {"provider": "deepseek", "model": "deepseek-chat"},
                },
                "usage": {
                    "inputTokens": 4000,
                    "cacheReadTokens": 26000,
                    "cacheWriteTokens": 500,
                    "outputTokens": 120,
                    "reasoningTokens": 45,
                },
            },
        },
        {
            "type": "tool/call",
            "seq": 10,
            "time": 1_700_000_000_610,
            "data": {
                "turn": 1,
                "step": 1,
                "callId": "call_bash_1",
                "name": "bash",
                "arguments": json.dumps({"command": "pytest -q"}),
            },
        },
        {
            "type": "tool/result",
            "seq": 11,
            "time": 1_700_000_001_200,
            "data": {
                "turn": 1,
                "step": 1,
                "message": {
                    "content": [
                        {
                            "type": "tool-result",
                            "toolCallId": "call_bash_1",
                            "isError": False,
                            "content": [
                                {
                                    "type": "text",
                                    "text": "FAILED test_app.py::test_x\n(Omitted 4096 bytes. Full formatted result stored at: /tmp/spill.txt)\n[exit code: 1]",
                                }
                            ],
                        }
                    ]
                },
            },
        },
        {
            "type": "step/end",
            "seq": 12,
            "time": 1_700_000_001_210,
            "data": {"turn": 1, "step": 1},
        },
        {
            "type": "compaction/start",
            "seq": 13,
            "time": 1_700_000_001_300,
            "data": {"compactionId": "c1", "turn": 1},
        },
        {
            "type": "compaction/summary",
            "seq": 14,
            "time": 1_700_000_001_800,
            "data": {
                "compactionId": "c1",
                "shadowedTokenCount": 20000,
                "usage": {"inputTokens": 1000, "outputTokens": 1500},
            },
        },
        {
            "type": "compaction/end",
            "seq": 15,
            "time": 1_700_000_001_850,
            "data": {"compactionId": "c1", "turn": 1},
        },
        {
            "type": "step/start",
            "seq": 16,
            "time": 1_700_000_002_000,
            "data": {"turn": 1, "step": 2},
        },
        {
            "type": "text-chunks",
            "seq0": 17,
            "time0": 1_700_000_002_100,
            "data": {"turn": 1, "step": 2, "index": 0, "dt": [0], "texts": ["Fix"]},
        },
        {
            "type": "assistant/message",
            "seq": 18,
            "time": 1_700_000_002_400,
            "data": {
                "turn": 1,
                "step": 2,
                "message": {"role": "assistant", "content": []},
                "usage": {
                    "inputTokens": 2000,
                    "cacheReadTokens": 10000,
                    "outputTokens": 90,
                },
            },
        },
        {
            "type": "tool/call",
            "seq": 19,
            "time": 1_700_000_002_410,
            "data": {
                "turn": 1,
                "step": 2,
                "callId": "call_edit_1",
                "name": "edit",
                "arguments": json.dumps(
                    {
                        "file_path": "src/app.py",
                        "old_string": "return 1",
                        "new_string": "return 2",
                    }
                ),
            },
        },
        {
            "type": "tool/result",
            "seq": 20,
            "time": 1_700_000_002_450,
            "data": {
                "turn": 1,
                "step": 2,
                "message": {
                    "content": [
                        {
                            "type": "tool-result",
                            "toolCallId": "call_edit_1",
                            "isError": False,
                            "content": [{"type": "text", "text": "Updated src/app.py"}],
                        }
                    ]
                },
            },
        },
        {
            "type": "tool/call",
            "seq": 21,
            "time": 1_700_000_002_460,
            "data": {
                "turn": 1,
                "step": 2,
                "callId": "call_edit_fail",
                "name": "edit",
                "arguments": json.dumps(
                    {
                        "file_path": "src/app.py",
                        "old_string": "missing",
                        "new_string": "x",
                    }
                ),
            },
        },
        {
            "type": "tool/result",
            "seq": 22,
            "time": 1_700_000_002_480,
            "data": {
                "turn": 1,
                "step": 2,
                "message": {
                    "content": [
                        {
                            "type": "tool-result",
                            "toolCallId": "call_edit_fail",
                            "isError": True,
                            "content": [
                                {"type": "text", "text": "Error: old_string not found"}
                            ],
                        }
                    ]
                },
            },
        },
        {
            "type": "step/end",
            "seq": 23,
            "time": 1_700_000_002_500,
            "data": {"turn": 1, "step": 2},
        },
        {
            "type": "turn/end",
            "seq": 24,
            "time": 1_700_000_002_510,
            "data": {"turn": 1, "reason": {"kind": "completed"}},
        },
    ]

    zstd_path = tmp_path / "slug" / "session-sess-synth-01" / "session.jsonl.zstd"
    _write_zstd_lines(zstd_path, [json.dumps(ev) for ev in events])

    traj = parse_dsh_session(zstd_path)
    assert traj.source == "dsh"
    assert traj.session_id == "sess-synth-01"
    assert traj.cwd == "/workspace/synthetic-project"
    assert traj.started_at == 1_700_000_000_000
    assert traj.model_routes == ("deepseek/deepseek-chat",)
    assert traj.parse_failures == 0
    assert traj.unknown_event_types == ()

    model_calls = [e for e in traj.events if isinstance(e, ModelCall)]
    tool_calls = [e for e in traj.events if isinstance(e, ToolCall)]
    edits = [e for e in traj.events if isinstance(e, Edit)]
    compactions = [e for e in traj.events if isinstance(e, Compaction)]
    turn_ends = [e for e in traj.events if isinstance(e, TurnEnd)]

    assert len(model_calls) == 2
    mc1 = model_calls[0]
    assert mc1.turn == 1
    assert mc1.step == 1
    assert mc1.started_at == 1_700_000_000_050
    assert mc1.first_token_at == 1_700_000_000_350
    assert mc1.ended_at == 1_700_000_000_600
    assert mc1.retries == 1
    assert mc1.input_tokens == 4000
    assert mc1.cache_read_tokens == 26000
    assert mc1.cache_write_tokens == 500
    assert mc1.output_tokens == 120
    assert mc1.reasoning_tokens == 45

    assert len(tool_calls) == 3
    tc_bash = tool_calls[0]
    assert tc_bash.name == "bash"
    assert tc_bash.exit_code == 1
    assert tc_bash.is_error is True
    assert tc_bash.truncated is True
    assert tc_bash.available_tools == ("bash", "read", "edit", "write")
    assert tc_bash.output_bytes is not None and tc_bash.output_bytes > 50

    # Only the non-error edit produces an Edit event
    assert len(edits) == 1
    assert edits[0].path == "src/app.py"
    assert edits[0].before_hash is not None
    assert edits[0].after_hash is not None
    assert edits[0].before_hash != edits[0].after_hash
    assert edits[0].turn == 1
    assert edits[0].step == 2
    assert edits[0].at == 1_700_000_002_450

    assert len(compactions) == 1
    assert compactions[0].ok is True
    assert compactions[0].tokens_before == 30500  # 4000 + 26000 + 500
    assert compactions[0].tokens_after == 12000  # 2000 + 10000

    assert len(turn_ends) == 1
    assert turn_ends[0].reason == "completed"
    assert turn_ends[0].at == 1_700_000_002_510


def test_dsh_malformed_lines_unknown_events_and_failed_compaction(
    tmp_path: Path,
) -> None:
    raw_lines = [
        json.dumps(
            {"type": "session", "id": "sess-err", "createdAt": 1_700_000_100_000}
        ),
        "{not valid json",
        "12345",
        json.dumps({"seq": 1, "time": 1_700_000_100_010}),
        json.dumps(
            {
                "type": "experimental/new-event",
                "seq": 2,
                "time": 1_700_000_100_020,
                "data": {},
            }
        ),
        json.dumps(
            {
                "type": "compaction/start",
                "seq": 3,
                "time": 1_700_000_100_030,
                "data": {"compactionId": "c_fail", "turn": 1},
            }
        ),
        json.dumps(
            {
                "type": "compaction/end",
                "seq": 4,
                "time": 1_700_000_100_040,
                "data": {"compactionId": "c_fail", "turn": 1, "error": "llm boom"},
            }
        ),
        json.dumps(
            {
                "type": "turn/end",
                "seq": 5,
                "time": 1_700_000_100_050,
                "data": {
                    "turn": 1,
                    "reason": {
                        "kind": "error",
                        "error": {"code": "TIMEOUT", "message": "stream idle timeout"},
                    },
                },
            }
        ),
    ]
    zstd_path = tmp_path / "sess-err" / "session.v4.jsonl.zstd"
    _write_zstd_lines(zstd_path, raw_lines, trailing_garbage=b"corrupt_frame_bytes")

    traj = parse_dsh_session(zstd_path)
    assert traj.session_id == "sess-err"
    # 3 bad JSON/schema lines + 1 corrupt zstd frame tail = 4 parse failures
    assert traj.parse_failures == 4
    assert traj.unknown_event_types == ("experimental/new-event",)

    comp = [e for e in traj.events if isinstance(e, Compaction)][0]
    assert comp.ok is False
    assert comp.reason == "llm boom"

    te = [e for e in traj.events if isinstance(e, TurnEnd)][0]
    assert te.reason == "error:TIMEOUT:stream idle timeout"


def test_dsh_secret_redaction_boundary(tmp_path: Path) -> None:
    sk_secret = "sk-proj-SuperSecretKey1234567890"
    bearer_secret = "myBearerTokenValue987654321"
    ghp_secret = "ghp_AbCdEfGhIjKlMnOpQrStUvWxYz123456"
    cookie_secret = "sessionid=xyzSecretCookieValue999"
    query_secret = "urlQueryTokenSecret123"
    hex_secret = "deadbeefcafebabe0123456789abcdef01234567"
    b64_secret = "YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY3ODkw=="

    cmd = (
        f'curl -H "Authorization: Bearer {bearer_secret}" '
        f'-H "Cookie: {cookie_secret}" '
        f'"https://example.test/api?token={query_secret}" '
        f"--data 'key={sk_secret}&gh={ghp_secret}&hex={hex_secret}&b64={b64_secret}'"
    )
    # Large 50 KB output with an early compile error and secrets in both head & tail
    huge_output = (
        f"SyntaxError: invalid token {sk_secret}\n"
        + ("line of synthetic build log output\n" * 1600)
        + f"tail leaked {sk_secret} and Bearer {bearer_secret}\n[exit code: 1]"
    )
    events = [
        {
            "type": "session",
            "id": "sess-redact",
            "createdAt": 1_700_000_200_000,
            "cwd": f"/tmp/work?token={query_secret}",
        },
        {
            "type": "tool/call",
            "seq": 1,
            "time": 1_700_000_200_100,
            "data": {
                "turn": 1,
                "step": 1,
                "callId": "c1",
                "name": "bash",
                "arguments": json.dumps({"command": cmd, "api_key": sk_secret}),
            },
        },
        {
            "type": "tool/result",
            "seq": 2,
            "time": 1_700_000_200_200,
            "data": {
                "turn": 1,
                "step": 1,
                "message": {
                    "content": [
                        {
                            "type": "tool-result",
                            "toolCallId": "c1",
                            "isError": False,
                            "content": [
                                {
                                    "type": "text",
                                    "text": huge_output,
                                }
                            ],
                        }
                    ]
                },
            },
        },
    ]
    zstd_path = tmp_path / "session-redact" / "session.jsonl.zstd"
    _write_zstd_lines(zstd_path, [json.dumps(e) for e in events])

    traj = parse_dsh_session(zstd_path)
    tc = [e for e in traj.events if isinstance(e, ToolCall)][0]
    assert tc.raw_args is not None
    assert tc.args_fingerprint is not None
    assert tc.output_tail is not None
    assert tc.error_signature is not None
    assert tc.output_bytes == len(huge_output.encode("utf-8"))
    assert tc.output_bytes > 50_000
    assert len(tc.output_tail.encode("utf-8")) <= 2048

    for secret in (
        sk_secret,
        bearer_secret,
        ghp_secret,
        "xyzSecretCookieValue999",
        query_secret,
        hex_secret,
        b64_secret,
    ):
        assert secret not in tc.raw_args
        assert secret not in tc.args_fingerprint
        assert secret not in tc.output_tail
        assert secret not in tc.error_signature
        assert secret not in (traj.cwd or "")
        assert secret not in repr(traj)

    # Normal file paths are not mangled by base64 redaction
    normal_path = "/workspace/demo-repo/src/dsh_metrics/adapters/dsh.py"
    assert redact_secrets(normal_path) == normal_path


def test_discover_dsh_files_prefers_latest_over_v4(tmp_path: Path) -> None:
    both_dir = tmp_path / "slug1" / "session-a"
    _write_zstd_lines(
        both_dir / "session.jsonl.zstd",
        [json.dumps({"type": "session", "id": "a", "createdAt": 1000})],
    )
    _write_zstd_lines(
        both_dir / "session.v4.jsonl.zstd",
        [json.dumps({"type": "session", "id": "a", "createdAt": 1000})],
    )

    v4_only_dir = tmp_path / "slug1" / "session-b"
    _write_zstd_lines(
        v4_only_dir / "session.v4.jsonl.zstd",
        [json.dumps({"type": "session", "id": "b", "createdAt": 2000})],
    )

    found = discover_dsh_files(tmp_path)
    assert found == [
        both_dir / "session.jsonl.zstd",
        v4_only_dir / "session.v4.jsonl.zstd",
    ]
