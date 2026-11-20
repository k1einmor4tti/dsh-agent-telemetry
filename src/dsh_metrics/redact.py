"""Secret redaction utilities applied at the adapter boundary."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

REDACTED = "[REDACTED]"
MAX_OUTPUT_TAIL_BYTES = 2048

# Prefixed API tokens (OpenAI, Anthropic, DeepSeek, GitHub, AWS, Slack, JWT)
_PREFIXED_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    r"sk-(?:ant-|proj-|live-|test-)?[A-Za-z0-9_-]{6,}"
    r"|(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{8,}"
    r"|github_pat_[A-Za-z0-9_]{10,}"
    r"|(?:AKIA|ASIA)[0-9A-Z]{16}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
    r")(?![A-Za-z0-9_-])"
)

# Authorization schemes: Bearer <tok>, Basic <tok>
_AUTH_SCHEME_RE = re.compile(
    r"(?i)\b(Bearer|Basic)(\s+)[A-Za-z0-9._~+/=-]{4,}"
)

# Authorization header key/value without standard scheme
_AUTH_HEADER_RE = re.compile(
    r"""(?i)\b(authorization["']?\s*[:=]\s*["']?)(?!\[REDACTED\])([^\s"',;}{]+(?:\s+[^\s"',;}{]+)?)"""
)

# Cookie headers and CLI flags (--cookie, Cookie:, Set-Cookie:)
_COOKIE_HEADER_RE = re.compile(
    r"""(?i)(\b(?:set-)?cookie["']?\s*[:=]\s*["']?)([^"'\r\n;}{]+(?:;\s*[^"'\r\n;}{]+)*)"""
)
_CURL_COOKIE_FLAG_RE = re.compile(
    r"""(?i)((?:--cookie|-b)\s+["']?)([^"'\s]+)"""
)

# URL query parameters holding credentials
_QUERY_SECRET_RE = re.compile(
    r"(?i)([?&](?:token|access_token|refresh_token|id_token|api_key|apikey|secret|password|passwd|client_secret|auth_token|sig|signature)=)([^&\s\"'#]+)"
)

# Key=value / key: value credential assignments
_KV_SECRET_RE = re.compile(
    r"""(?i)\b(api[_-]?key|apikey|access[_-]?token|refresh[_-]?token|id[_-]?token|auth[_-]?token|secret[_-]?key|client[_-]?secret|private[_-]?key|password|passwd|x-api-key|token)(["']?\s*[:=]\s*["']?)(?!\[REDACTED\])([A-Za-z0-9._~+/=-]{4,})"""
)

# Sensitive dictionary keys when redacting structured JSON objects
_SENSITIVE_KEY_RE = re.compile(
    r"(?i)^(?:authorization|cookie|set-cookie|x-api-key|api[_-]?key|apikey|"
    r"access[_-]?token|refresh[_-]?token|id[_-]?token|auth[_-]?token|"
    r"client[_-]?secret|secret[_-]?key|private[_-]?key|password|passwd|secret|token)$"
)

# Long contiguous hex strings (>= 32 hex chars, not hyphenated UUIDs)
_LONG_HEX_RE = re.compile(r"(?<![A-Za-z0-9_-])[0-9a-fA-F]{32,}(?![A-Za-z0-9_-])")

# Candidate long base64-like tokens (>= 40 chars)
_CANDIDATE_B64_RE = re.compile(
    r"(?<![A-Za-z0-9_./~+-])[A-Za-z0-9+/=_-]{40,}(?![A-Za-z0-9_./-])"
)

_PATH_EXT_RE = re.compile(
    r"\.(?:py|go|ts|tsx|js|jsx|mjs|cjs|json|jsonl|zstd|md|txt|yaml|yml|toml|rs|java|c|cc|cpp|h|hpp|sh|zsh|sql|html|css|lock| proto)$",
    re.IGNORECASE,
)


def _looks_like_base64_secret(token: str) -> bool:
    """Heuristic distinguishing high-entropy base64 secrets from paths/identifiers."""
    if token.startswith(("/", "./", "../", "~/")) or "//" in token:
        return False
    if _PATH_EXT_RE.search(token):
        return False
    # Multiple slash-separated lowercase/identifier segments look like relative paths
    if "/" in token:
        parts = token.split("/")
        if len(parts) >= 3 and all(
            p and re.fullmatch(r"[A-Za-z0-9_.-]+", p) for p in parts
        ):
            # Unless it has explicit '=' padding at the end, treat multi-segment '/' as a path
            if not token.endswith("="):
                return False
    # Snake-case or kebab-case identifiers without digits/padding are code identifiers
    if ("_" in token or "-" in token) and not any(ch.isdigit() for ch in token):
        return False

    has_upper = any(ch.isupper() for ch in token)
    has_lower = any(ch.islower() for ch in token)
    has_digit = any(ch.isdigit() for ch in token)
    has_b64_sym = any(ch in "+/=" for ch in token)
    classes = sum((has_upper, has_lower, has_digit, has_b64_sym))
    if token.endswith("=") and classes >= 2:
        return True
    return has_upper and has_lower and has_digit


def _replace_b64_candidate(match: re.Match[str]) -> str:
    tok = match.group(0)
    return REDACTED if _looks_like_base64_secret(tok) else tok


def redact_secrets(text: str) -> str:
    """Redact API keys, bearer tokens, cookies, token params, and long hex/base64 secrets."""
    if not text:
        return text
    out = _PREFIXED_TOKEN_RE.sub(REDACTED, text)
    out = _AUTH_SCHEME_RE.sub(rf"\1\2{REDACTED}", out)
    out = _AUTH_HEADER_RE.sub(rf"\1{REDACTED}", out)
    out = _COOKIE_HEADER_RE.sub(rf"\1{REDACTED}", out)
    out = _CURL_COOKIE_FLAG_RE.sub(rf"\1{REDACTED}", out)
    out = _QUERY_SECRET_RE.sub(rf"\1{REDACTED}", out)
    out = _KV_SECRET_RE.sub(rf"\1\2{REDACTED}", out)
    out = _LONG_HEX_RE.sub(REDACTED, out)
    out = _CANDIDATE_B64_RE.sub(_replace_b64_candidate, out)
    return out


def redact_obj(value: Any) -> Any:
    """Recursively redact secrets in JSON-like Python structures."""
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for k, v in value.items():
            key_str = redact_secrets(str(k))
            if _SENSITIVE_KEY_RE.match(key_str) and v not in (None, "", [], {}):
                result[key_str] = REDACTED
            else:
                result[key_str] = redact_obj(v)
        return result
    if isinstance(value, list):
        return [redact_obj(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_obj(item) for item in value)
    return value


def make_output_tail(
    text: str | None, max_bytes: int = MAX_OUTPUT_TAIL_BYTES
) -> str | None:
    """Return at most the last `max_bytes` UTF-8 bytes of `text` after redaction."""
    if text is None:
        return None
    if not text:
        return ""
    window = text[-(max_bytes * 2) :] if len(text) > max_bytes * 2 else text
    redacted = redact_secrets(window)
    encoded = redacted.encode("utf-8")
    if len(encoded) <= max_bytes:
        return redacted
    trimmed = encoded[-max_bytes:].decode("utf-8", errors="ignore")
    if "]" in trimmed[:10] and not trimmed.startswith("["):
        idx = trimmed.find("]")
        trimmed = trimmed[idx + 1 :]
    return trimmed

