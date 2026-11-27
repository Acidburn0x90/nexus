"""Multi-harness execution backends for Nexus (OpenCode, OpenAI-compatible HTTP, Ollama, external CLI)."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import itertools
import json
import os
from pathlib import Path
import queue
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Optional
import urllib.error
import urllib.request

from nexus.drops import SEND_FILE_TOOL_SCHEMA, write_drop_file
from nexus.events import DEFAULT_EVENTS_PATH, emit_event, new_span_id
from nexus.router import load_routes, resolve_route_api_key
from nexus.square import DEFAULT_BOARD_PATH, DEFAULT_SQUARE_PATH, format_square_context
from nexus.visualizer import (
    format_duration,
    format_quota_badge,
    format_tokens,
    status_line,
    visible_len,
)


_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# Live spinner state keyed by `id(stop_event)`: {"chars": int, "preview": str}.
_SPIN_STATES: dict[int, dict[str, object]] = {}

_THINKING_PREVIEW_CHARS = 42
_THINKING_WRAP_CHARS = 110


def _status_enabled(verbosity: int, minimum: int = 1) -> bool:
    return verbosity >= minimum


def _clip_preview(text: str, limit: int = _THINKING_PREVIEW_CHARS) -> str:
    """Collapse whitespace and clip one preview line cleanly to `limit` chars."""
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 1)].rstrip() + "…"


def _preview_from_chunk(chunk: str) -> str:
    """Return the last non-empty, whitespace-collapsed line of a thinking chunk."""
    for line in reversed(chunk.splitlines()):
        if line.strip():
            return _clip_preview(line)
    return ""


def _wrap_thinking(text: str, width: int = _THINKING_WRAP_CHARS) -> list[str]:
    """Wrap one buffered span at word boundaries, hard-splitting overlong words."""
    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
        while len(word) > width:
            lines.append(word[:width])
            word = word[width:]
        current = word
    if current:
        lines.append(current)
    return lines


def _spin_stderr(
    route_key: str,
    stop: threading.Event,
    *,
    color: Optional[bool] = None,
    verbosity: int = 1,
    state: Optional[dict] = None,
) -> None:
    """Rotate a thinking spinner on stderr while the backend runs (TTY only).

    When live thinking state is available, the frame also shows how many
    reasoning characters arrived and a preview of the latest line.
    """
    if not sys.stderr.isatty() or not _status_enabled(verbosity):
        return
    enabled = sys.stderr.isatty() if color is None else color
    info = state if state is not None else _SPIN_STATES.get(id(stop))
    width = 0
    for ch in itertools.cycle(_SPINNER_FRAMES):
        if stop.is_set():
            break
        chars = int(info.get("chars", 0) or 0) if info else 0
        if chars > 0:
            preview = str(info.get("preview", "") or "") if info else ""
            label = _route_label(route_key)
            if preview:
                text = f"{label} thinking ({format_tokens(chars)}c) · {preview} {ch}"
            else:
                text = f"{label} thinking ({format_tokens(chars)}c)…{ch}"
        else:
            text = f"{_route_label(route_key)} thinking…{ch}"
        frame = status_line("spin", text, color=enabled)
        width = max(width, visible_len(frame))
        sys.stderr.write("\r" + frame)
        sys.stderr.flush()
        stop.wait(0.2)
    # Clear exactly what was drawn so a longer route never leaves debris.
    if width:
        sys.stderr.write("\r" + " " * width + "\r")
    sys.stderr.flush()


def _stop_spin(stop: threading.Event, thread: threading.Thread) -> None:
    """Stop the spinner and wait for it to clear its line before printing anything else."""
    stop.set()
    thread.join(timeout=1.0)
    _SPIN_STATES.pop(id(stop), None)


def _route_label(route_key: str) -> str:
    """Human label for a span: `in-charge:code` -> `@code (in charge)`."""
    if route_key.startswith("in-charge:"):
        return f"@{route_key[len('in-charge:'):]} (in charge)"
    return f"@{route_key}"


class _ThinkingStderrPrinter:
    """Stream reasoning lines to stderr at verbosity >= 1 without touching stdout."""

    def __init__(
        self,
        route_key: str,
        verbosity: int,
        emit_stderr: bool,
        color_enabled: bool,
        spin_state: dict,
        stop_spinner: Callable[[], None],
    ) -> None:
        self.route_key = route_key
        self.verbosity = verbosity
        self.emit_stderr = emit_stderr
        self.color_enabled = color_enabled
        self.spin_state = spin_state
        self.stop_spinner = stop_spinner
        self.chars = 0
        self._buf = ""
        self._started = False
        self._lock = threading.Lock()

    def _live(self) -> bool:
        return self.emit_stderr and self.verbosity >= 1

    def _write_line(self, line: str) -> None:
        body = line.rstrip()
        if self.color_enabled:
            sys.stderr.write(f"\033[2;3m  ┆ {body}\033[0m\n")
        else:
            sys.stderr.write(f"  ┆ {body}\n")

    def _start_header(self) -> None:
        if self._started:
            return
        self.stop_spinner()
        header = status_line("info", f"{_route_label(self.route_key)} thinking:", color=self.color_enabled)
        sys.stderr.write(header + "\n")
        self._started = True

    def _drain(self) -> None:
        while "\n" in self._buf:
            line, _, self._buf = self._buf.partition("\n")
            self._write_line(line)
        if len(self._buf) > _THINKING_WRAP_CHARS:
            wrapped = _wrap_thinking(self._buf)
            for line in wrapped[:-1]:
                self._write_line(line)
            self._buf = wrapped[-1] if wrapped else ""

    def push(self, chunk: str) -> None:
        if not chunk:
            return
        with self._lock:
            self.chars += len(chunk)
            self.spin_state["chars"] = self.chars
            preview = _preview_from_chunk(chunk)
            if preview:
                self.spin_state["preview"] = preview
            if not self._live():
                return
            self._buf += str(chunk).replace("\r\n", "\n").replace("\r", "")
            if not (self._started or self._buf.strip()):
                return
            self._start_header()
            self._drain()

    def flush(self) -> None:
        with self._lock:
            if not self._started and self._live() and self._buf.strip():
                self._start_header()
            if not self._started:
                return
            if self._buf.strip():
                for line in _wrap_thinking(self._buf):
                    self._write_line(line)
            self._buf = ""

    def reset_attempt(self) -> None:
        """Re-arm the header for a fallback attempt without losing the char count."""
        with self._lock:
            self._buf = ""
            self._started = False


def _receipt(
    kind: str,
    route_key: str,
    backend: str,
    model: str,
    quota_tier: str,
    elapsed_ms: int,
    *,
    tokens_in: int = 0,
    tokens_out: int = 0,
    tokens_cached: int = 0,
    tools: int = 0,
    detail: str = "",
    color: Optional[bool] = None,
    verbosity: int = 1,
    thinking_chars: int = 0,
) -> None:
    """Print a one-line outcome receipt for a delegated span."""
    if not _status_enabled(verbosity):
        return
    enabled = sys.stderr.isatty() if color is None else color
    parts = [_route_label(route_key), f"{backend}:{model}", format_quota_badge(quota_tier, color=enabled)]
    parts.append(format_duration(elapsed_ms))
    if tokens_in or tokens_out:
        tok_part = f"{format_tokens(tokens_in)}→{format_tokens(tokens_out)} tok"
        parts.append(tok_part)
    if tokens_cached and verbosity >= 2:
        parts.append(f"({format_tokens(tokens_cached)} cached)")
    if tools:
        parts.append(f"{tools} tool{'s' if tools != 1 else ''}")
    if thinking_chars > 0:
        parts.append(f"thought {format_tokens(thinking_chars)}c")
    if detail:
        parts.append(detail)
    print(status_line(kind, " · ".join(parts), color=enabled), file=sys.stderr)


@dataclass
class OpenCodeStream:
    """Parsed `opencode run --format json` transcript."""

    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0
    last_step: str = "completed"
    session_id: str = ""
    thinking: str = ""
    has_text: bool = False


@dataclass
class ExecutionResult:
    trace_id: str
    span_id: str
    route: str
    backend: str
    model: str
    quota_tier: str
    status: str
    output: str
    tokens_in: int = 0
    tokens_out: int = 0
    elapsed_ms: int = 0
    error: str = ""
    session_id: str = ""
    tokens_cached: int = 0
    thinking: str = ""


StepCallback = Optional[Callable[[dict], None]]
# kind is one of: 'text', 'thinking', 'tool', 'session', 'status'.
StreamCallback = Optional[Callable[[str, str], None]]


def _emit(
    trace_id: str,
    span_id: str,
    *,
    parent_id: Optional[str],
    route: str,
    intent: str,
    backend: str,
    model: str,
    quota_tier: str,
    status: str,
    step: str,
    prompt: str = "",
    tokens_in: int = 0,
    tokens_out: int = 0,
    elapsed_ms: int = 0,
    error: str = "",
    profile: str = "",
    dry_run: bool = False,
    events_path: Path = DEFAULT_EVENTS_PATH,
    on_event: StepCallback = None,
) -> dict:
    ev = emit_event(
        trace_id,
        span_id,
        parent_id=parent_id,
        route=route,
        intent=intent,
        backend=backend,
        model=model,
        quota_tier=quota_tier,
        status=status,
        step=step,
        prompt=prompt,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        elapsed_ms=elapsed_ms,
        error=error,
        profile=profile,
        dry_run=dry_run,
        path=events_path,
    )
    if on_event is not None:
        on_event(ev)
    return ev


def _require_binary(backend: str) -> Optional[str]:
    """Return a human error when a subprocess backend's binary is missing from PATH."""
    binary = {"opencode": "opencode", "agy": "agy"}.get(backend)
    if binary and shutil.which(binary) is None:
        return f"`{binary}` not found on PATH — install it or route to another backend"
    return None


def _json_error_message(obj: object) -> str:
    """Return the first human error string carried by one JSON event object."""
    if not isinstance(obj, dict):
        return ""
    err = obj.get("error")
    if isinstance(err, dict):
        data = err.get("data") if isinstance(err.get("data"), dict) else {}
        msg = data.get("message") or err.get("message") or err.get("type") or ""
        if msg:
            return str(msg)
    if obj.get("message") and obj.get("type") in (None, "", "error"):
        return str(obj["message"])
    return ""


def _humanize_error(raw: str, limit: int = 160) -> str:
    """Reduce a backend error blob to one readable line (opencode emits JSON errors)."""
    text = (raw or "").strip()
    if not text:
        return ""
    lines = text.splitlines()
    first = lines[0]
    for line in lines:
        s = line.strip()
        if not s.startswith("{"):
            continue
        try:
            msg = _json_error_message(json.loads(s))
        except json.JSONDecodeError:
            continue
        if msg:
            return msg[:limit]
    return first[:limit]


_HISTORY_MAX_ITEMS = 12
_HISTORY_MAX_CHARS = 16_000
_ATTACH_MAX_BYTES = 64 * 1024
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"})
_MARKDOWN_BLOCKS = ("#", "- ", "* ", "```")


def _infer_fallback_tier(fallback_model: str, default_tier: str) -> str:
    """Infer the quota tier a fallback model spends; never guesses PREMIUM."""
    if fallback_model.endswith("-free"):
        return "FREE"
    if fallback_model.startswith("opencode-go/"):
        return "GO"
    norm = str(default_tier or "").strip().upper().replace("_", "-")
    if norm in ("PREMIUM", "PAID", "ARGON", "ARGON-QUOTA"):
        return "FREE"
    return default_tier


_HTTP_STATUS_RE = re.compile(r"\b(?:HTTP(?:\s+Error)?\s+)?(40[0-9]|429|50[0-9]|5[1-9][0-9])\b", re.IGNORECASE)
_HISTORY_KEEP_RECENT_TURNS = 3
_HISTORY_TOOL_CLEAR_CHARS = 240
_TOOL_OUTPUT_MARKERS = ("[Tool", "[Attached file:", "Command output:", "Tool output:", "stdout:", "stderr:")


def classify_http_error(error_or_status: object) -> str:
    """Classify an HTTP status code or error message into auth|quota|transient|other."""
    if isinstance(error_or_status, int):
        code: Optional[int] = error_or_status
        text = str(error_or_status)
    else:
        text = str(error_or_status or "").strip()
        m = _HTTP_STATUS_RE.search(text)
        code = int(m.group(1)) if m else None
    low = text.lower()
    if code in (401, 403) or any(tok in low for tok in ("unauthorized", "forbidden", "invalid api key", "authentication")):
        return "auth"
    if code in (402, 429) or any(tok in low for tok in ("rate limit", "too many requests", "quota", "insufficient credits")):
        return "quota"
    if code == 408 or (code is not None and code >= 500) or any(
        tok in low for tok in ("timed out", "timeout", "connection refused", "connection reset", "urlopen error", "bad gateway", "service unavailable")
    ):
        return "transient"
    return "other"


class RouteHealthTracker:
    """Thread-safe lazy 3-scope cooldown tracker with EWMA latency (alpha=0.3)."""

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = float(alpha)
        self._lock = threading.Lock()
        self._cooldowns: dict[str, tuple[float, str]] = {}
        self._failures: dict[str, int] = {}
        self._ewma_ms: dict[str, float] = {}

    @staticmethod
    def _scopes(endpoint: str, route_key: str, model: str) -> tuple[str, str, str]:
        return (
            f"endpoint:{endpoint}",
            f"route_key:{route_key}",
            f"model:{endpoint}::{model}",
        )

    def _scope_for_kind(self, kind: str, endpoint: str, route_key: str, model: str) -> tuple[str, float, float]:
        ep_s, rk_s, md_s = self._scopes(endpoint, route_key, model)
        if kind == "transient":
            return ep_s, 5.0, 60.0
        if kind == "quota":
            return rk_s, 15.0, 120.0
        if kind == "auth":
            return rk_s, 300.0, 300.0
        return md_s, 10.0, 120.0

    def record_failure(self, *, endpoint: str, route_key: str, model: str, kind: str) -> float:
        """Apply capped jittered cooldown on the affected scope and return cooldown seconds."""
        scope, base_s, cap_s = self._scope_for_kind(kind, endpoint, route_key, model)
        with self._lock:
            n = self._failures.get(scope, 0) + 1
            self._failures[scope] = n
            delay = min(cap_s, base_s * (2 ** min(n - 1, 5)))
            jittered = delay * random.uniform(0.85, 1.15)
            self._cooldowns[scope] = (time.monotonic() + jittered, kind)
            return jittered

    def record_success(self, *, endpoint: str, route_key: str, model: str, elapsed_ms: float) -> None:
        """Clear scope cooldowns/failures and update EWMA latency."""
        ep_s, rk_s, md_s = self._scopes(endpoint, route_key, model)
        sample = max(0.0, float(elapsed_ms))
        with self._lock:
            for scope in (ep_s, rk_s, md_s):
                self._failures.pop(scope, None)
                self._cooldowns.pop(scope, None)
            for scope in (rk_s, md_s):
                prev = self._ewma_ms.get(scope)
                self._ewma_ms[scope] = sample if prev is None else (self.alpha * sample + (1.0 - self.alpha) * prev)

    def is_cooled_down(self, *, endpoint: str, route_key: str, model: str) -> Optional[str]:
        """Return the active cooldown reason if any scope is cooling down, pruning expired entries."""
        now = time.monotonic()
        with self._lock:
            for scope in self._scopes(endpoint, route_key, model):
                entry = self._cooldowns.get(scope)
                if entry is None:
                    continue
                until, reason = entry
                if now < until:
                    return f"{scope} ({reason})"
                self._cooldowns.pop(scope, None)
        return None

    def latency_ewma(self, *, endpoint: str, route_key: str, model: str) -> Optional[float]:
        """Return model-level or route-level EWMA latency in ms, if recorded."""
        _, rk_s, md_s = self._scopes(endpoint, route_key, model)
        with self._lock:
            return self._ewma_ms.get(md_s, self._ewma_ms.get(rk_s))

    def reset(self) -> None:
        """Clear all recorded cooldowns, failures, and latency estimates."""
        with self._lock:
            self._cooldowns.clear()
            self._failures.clear()
            self._ewma_ms.clear()


_HEALTH_TRACKER = RouteHealthTracker(alpha=0.3)


def reset_health_state() -> None:
    """Reset the shared RouteHealthTracker state (useful in tests and diagnostics)."""
    _HEALTH_TRACKER.reset()


def _markdown_piece(acc: str, chunk: str, saw_tool: bool) -> str:
    """Prefix a streamed chunk with a paragraph break when Markdown would glue."""
    if not acc or acc.endswith("\n\n"):
        return chunk
    glued = saw_tool or (not acc[-1].isspace() and chunk.startswith(_MARKDOWN_BLOCKS))
    if not glued:
        return chunk
    return ("\n" if acc.endswith("\n") else "\n\n") + chunk


def _normalize_think(value: object) -> object:
    """Map a `think` option to an Ollama payload value; None omits the key."""
    if value is None or isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("off", "none", "false"):
        return False
    if text == "true":
        return True
    if text in ("minimal", "low", "medium", "high", "xhigh", "max"):
        return text
    return None


def _clear_old_tool_output(content: str) -> str:
    """Deterministically replace bulky old tool outputs while preserving concise turns."""
    if len(content) <= _HISTORY_TOOL_CLEAR_CHARS:
        return content
    stripped = content.lstrip()
    if any(stripped.startswith(m) or f"\n{m}" in content for m in _TOOL_OUTPUT_MARKERS):
        head = stripped.splitlines()[0][:120].strip()
        return f"{head}\n[cleared old tool output ({len(content)} chars)]"
    return content


def _bounded_history(history_messages: Optional[list[dict]]) -> list[dict]:
    """Keep the newest 12 user/assistant turns within 16k chars, clearing old tool outputs (>240 chars)."""
    valid: list[dict] = []
    for msg in list(history_messages or []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            continue
        valid.append({"role": role, "content": content})

    # Keep the newest 3 conversation turns (up to 6 user/assistant messages) verbatim.
    tail_cutoff = max(0, len(valid) - (_HISTORY_KEEP_RECENT_TURNS * 2))
    normalized: list[dict] = []
    for idx, msg in enumerate(valid):
        content = msg["content"] if idx >= tail_cutoff else _clear_old_tool_output(msg["content"])
        normalized.append({"role": msg["role"], "content": content})

    kept: list[dict] = []
    total = 0
    for msg in reversed(normalized):
        if len(kept) >= _HISTORY_MAX_ITEMS:
            break
        content = msg["content"]
        remaining = _HISTORY_MAX_CHARS - total
        if remaining <= 0:
            break
        content = content[-remaining:]
        kept.append({"role": msg["role"], "content": content})
        total += len(content)
    kept.reverse()
    return kept


def _attachment_parts(attach_file: Path, backend: str) -> tuple[str, str]:
    """Return (text_prefix, image_b64) for one attachment; raise on a missing file."""
    path = Path(attach_file)
    if not path.is_file():
        raise FileNotFoundError(f"attachment not found: {attach_file}")
    if backend == "ollama" and path.suffix.lower() in _IMAGE_SUFFIXES:
        return "", base64.b64encode(path.read_bytes()).decode("ascii")
    text = path.read_bytes()[:_ATTACH_MAX_BYTES].decode("utf-8", errors="replace")
    return f"[Attached file: {path.name}]\n{text}\n\n", ""


def _stream_objects(line: str, backend: str) -> list[dict]:
    """Extract JSON objects from one Ollama NDJSON or OpenAI-compatible SSE line."""
    s = line.strip()
    if not s:
        return []
    if s.startswith("data:"):
        s = s[len("data:"):].strip()
        if not s or s == "[DONE]":
            return []
    if not s.startswith("{"):
        return []
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return []
    return [obj] if isinstance(obj, dict) else []


def _ollama_stream_chunk(obj: dict) -> tuple[str, str]:
    """Return (thinking, text) from one Ollama NDJSON object."""
    msg = obj.get("message") if isinstance(obj.get("message"), dict) else {}
    thinking = msg.get("thinking") or ""
    text = msg.get("content") or obj.get("response") or ""
    return str(thinking), str(text)


def _openai_compat_stream_chunk(obj: dict) -> tuple[str, str]:
    """Return (thinking, text) from one OpenAI-compatible chat-completions object."""
    choices = obj.get("choices") if isinstance(obj.get("choices"), list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        delta = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    thinking = delta.get("reasoning_content") or delta.get("reasoning") or ""
    return str(thinking), str(delta.get("content") or "")


def _parse_tool_args(raw: Any) -> tuple[dict[str, Any], str]:
    """Normalize dict or JSON-string tool arguments to `(args_dict, json_str)`."""
    if isinstance(raw, dict):
        return raw, json.dumps(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed, text
            except json.JSONDecodeError:
                pass
        return {}, raw
    return {}, "{}"


def _ollama_tool_calls(obj: dict) -> list[dict[str, Any]]:
    """Extract normalized tool calls emitted on one Ollama NDJSON chunk."""
    msg = obj.get("message") if isinstance(obj.get("message"), dict) else {}
    raw_calls = msg.get("tool_calls")
    if not isinstance(raw_calls, list):
        return []
    out: list[dict[str, Any]] = []
    for call in raw_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        args, raw_str = _parse_tool_args(fn.get("arguments"))
        out.append({"id": str(call.get("id") or ""), "name": name, "arguments": args, "raw_args": raw_str})
    return out


def _openai_compat_tool_delta(obj: dict, slots: dict[int, dict[str, Any]]) -> None:
    """Merge OpenAI-style streamed or non-streamed tool calls into `slots` keyed by index."""
    choices = obj.get("choices") if isinstance(obj.get("choices"), list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    container = choice.get("delta")
    if not isinstance(container, dict):
        container = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    raw_calls = container.get("tool_calls")
    if not isinstance(raw_calls, list):
        return
    for pos, chunk in enumerate(raw_calls):
        if not isinstance(chunk, dict):
            continue
        idx = int(chunk.get("index") if chunk.get("index") is not None else pos)
        slot = slots.setdefault(idx, {"id": "", "name": "", "raw_args": ""})
        if chunk.get("id"):
            slot["id"] = str(chunk["id"])
        fn = chunk.get("function") if isinstance(chunk.get("function"), dict) else {}
        if fn.get("name"):
            slot["name"] = str(fn["name"]).strip()
        args = fn.get("arguments")
        if isinstance(args, dict):
            slot["raw_args"] = json.dumps(args)
        elif isinstance(args, str):
            slot["raw_args"] += args


def _finalize_openai_compat_calls(slots: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for _idx, slot in sorted(slots.items()):
        name = str(slot.get("name") or "").strip()
        if not name:
            continue
        args, raw_str = _parse_tool_args(slot.get("raw_args", ""))
        calls.append({"id": str(slot.get("id") or ""), "name": name, "arguments": args, "raw_args": raw_str})
    return calls


def _run_drop_tool_call(call: dict[str, Any], *, drop_dir: Optional[Path] = None) -> tuple[dict[str, Any], str]:
    """Execute one local drop-folder tool call safely; never raises on bad arguments."""
    name = str(call.get("name") or "")
    if name != "send_file":
        return {"ok": False, "error": f"unsupported tool: {name}"}, f"{name} · unsupported tool"
    args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    exec_flag = args.get("executable")
    if exec_flag is not None and not isinstance(exec_flag, bool):
        exec_flag = None
    try:
        res = write_drop_file(
            args.get("filename", ""),
            args.get("content", ""),
            executable=exec_flag,
            drop_dir=drop_dir,
        )
    except (ValueError, OSError, TypeError) as exc:
        msg = str(exc)
        return {"ok": False, "error": msg}, f"send_file · error: {msg[:80]}"
    exec_note = ", +x" if res["executable"] else ""
    summary = f"send_file · {res['relative']} ({res['bytes']} B{exec_note})"
    return res, summary


def _read_http_response(
    endpoint: str,
    payload: bytes,
    timeout_s: float,
    closed: threading.Event,
    resp_box: list,
    errors: list[str],
    sink: "queue.Queue[Optional[str]]",
    headers: Optional[dict[str, str]] = None,
) -> None:
    """POST a payload and push decoded response lines into `sink`; None terminates."""
    resp = None
    try:
        req_headers = {"Content-Type": "application/json"}
        if headers:
            req_headers.update(headers)
        req = urllib.request.Request(endpoint, data=payload, headers=req_headers, method="POST")
        resp = urllib.request.urlopen(req, timeout=max(1.0, min(timeout_s, 30.0)))
        # Expose the raw socket so the caller can unblock a read without touching
        # the buffered response object (closing it would wait on the reader lock).
        raw = getattr(getattr(resp, "fp", None), "raw", None)
        resp_box.append(getattr(raw, "_sock", None))
        for line in resp:
            if closed.is_set():
                break
            sink.put(line.decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:2000]
        except Exception:
            pass
        finally:
            try:
                exc.close()
            except Exception:
                pass
        detail = _humanize_error(body, 220) if body else str(exc.reason or "")
        errors.append(f"HTTP {exc.code}: {detail}" if detail else f"HTTP {exc.code}")
    except Exception as exc:  # urlopen/read errors are reported to the caller
        errors.append(str(exc))
    finally:
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass
        sink.put(None)


def _pump(stream, sink: list[str], on_line: Optional[Callable[[str], None]], stop: threading.Event) -> None:
    """Drain a child pipe into `sink`, feeding `on_line` as data arrives."""
    try:
        for line in stream:
            if stop.is_set():
                break
            sink.append(line)
            if on_line is not None:
                on_line(line)
    except (OSError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except (OSError, ValueError):
            pass


def _run_subprocess_bounded(
    argv: list[str],
    timeout_s: float,
    on_stdout_line: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> tuple[int, str, str, bool]:
    """Run a subprocess in its own process group with a wall-clock timeout that actually fires.

    Both pipes are drained by reader threads, so a chatty or silent child can never
    deadlock the wait: the deadline and the optional cancel event are enforced by
    polling `proc.wait()` in short slices, not by the arrival of the next stdout line.
    ``timed_out`` is True only for a wall-clock timeout; a cancellation returns False.
    """
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    stop = threading.Event()
    readers = [
        threading.Thread(target=_pump, args=(proc.stdout, stdout_lines, on_stdout_line, stop), daemon=True),
        threading.Thread(target=_pump, args=(proc.stderr, stderr_lines, None, stop), daemon=True),
    ]
    for reader in readers:
        reader.start()

    timed_out = False
    cancelled = False
    try:
        deadline = time.monotonic() + max(0.1, timeout_s)
        while proc.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            slice_s = min(0.15, remaining) if cancel_event is not None else remaining
            try:
                proc.wait(timeout=slice_s)
            except subprocess.TimeoutExpired:
                continue
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=2.0)
                except Exception:
                    pass
        join_timeout = 0.5 if (timed_out or cancelled) else 5.0
        for reader in readers:
            reader.join(timeout=join_timeout)
        stop.set()

    if cancelled:
        timed_out = False
    rc = proc.returncode if proc.returncode is not None else -1
    return rc, "".join(stdout_lines), "".join(stderr_lines), timed_out


def _parse_opencode_json_stream(raw_stdout: str) -> "OpenCodeStream":
    """Extract text, token accounting, last step and session from `opencode run --format json`.

    opencode reports per-step usage as `{"input", "output", "reasoning",
    "cache": {"read", "write"}}`. `input` excludes cached prompt tokens, so a
    warm-cache run looks tiny; summing `input + cache.read` gives the real prompt
    size and stays stable across cache states.
    """
    text_chunks: list[str] = []
    thinking_chunks: list[str] = []
    text_out = ""
    saw_tool_since_text = False
    step_in = step_out = step_cached = 0
    seen_step_finish = False
    max_in = max_out = max_cached = 0
    last_step = "completed"
    discovered_session = ""

    for line in raw_stdout.splitlines():
        s = line.strip()
        if not s.startswith("{"):
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue

        if not discovered_session and obj.get("sessionID"):
            discovered_session = str(obj["sessionID"])
        ev_type = obj.get("type", "")
        part = obj.get("part") if isinstance(obj.get("part"), dict) else obj
        if not discovered_session and part.get("sessionID"):
            discovered_session = str(part["sessionID"])
        if ev_type == "text" or part.get("type") == "text":
            txt = part.get("text") or obj.get("text") or ""
            if txt:
                piece = _markdown_piece(text_out, str(txt), saw_tool_since_text)
                text_out += piece
                text_chunks.append(piece)
                saw_tool_since_text = False
        elif ev_type in ("reasoning", "thinking") or part.get("type") in ("reasoning", "thinking"):
            txt = part.get("text") or obj.get("text") or ""
            if txt:
                thinking_chunks.append(str(txt))
        elif ev_type in ("tool", "tool_use") or part.get("type") == "tool":
            tool_name = part.get("tool") or part.get("name") or "tool"
            last_step = f"tool:{tool_name}"
            saw_tool_since_text = True

        usage = part.get("tokens") if isinstance(part.get("tokens"), dict) else None
        if usage is None:
            continue
        cache = usage.get("cache") if isinstance(usage.get("cache"), dict) else {}
        cached = int(cache.get("read", 0) or 0)
        in_tok = int(usage.get("input", 0) or 0)
        out_tok = int(usage.get("output", 0) or 0) + int(usage.get("reasoning", 0) or 0)
        max_in = max(max_in, in_tok)
        max_out = max(max_out, out_tok)
        max_cached = max(max_cached, cached)
        if part.get("type") == "step-finish" or ev_type == "step-finish":
            seen_step_finish = True
            step_in += in_tok
            step_out += out_tok
            step_cached += cached

    if seen_step_finish:
        tokens_in, tokens_out, tokens_cached = step_in + step_cached, step_out, step_cached
    else:
        tokens_in, tokens_out, tokens_cached = max_in + max_cached, max_out, max_cached

    text = "".join(text_chunks).strip() if text_chunks else raw_stdout.strip()
    return OpenCodeStream(
        text,
        tokens_in,
        tokens_out,
        tokens_cached,
        last_step,
        discovered_session,
        "".join(thinking_chunks).strip(),
        bool(text_chunks),
    )


def _agy_stream_text(obj: dict) -> str:
    """Pull the human-readable text out of an `agy --output-format stream-json` event."""
    for key in ("text", "content", "message", "response"):
        val = obj.get(key)
        if isinstance(val, str):
            return val
        if isinstance(val, dict) and isinstance(val.get("text"), str):
            return val["text"]
    return ""


def execute_route(
    trace_id: str,
    route_key: str,
    route_cfg: dict,
    prompt: str,
    *,
    parent_id: Optional[str] = None,
    span_id: Optional[str] = None,
    attach_file: Optional[Path] = None,
    session_id: Optional[str] = None,
    events_path: Path = DEFAULT_EVENTS_PATH,
    square_path: Path = DEFAULT_SQUARE_PATH,
    board_path: Path = DEFAULT_BOARD_PATH,
    inject_square: bool = True,
    on_event: StepCallback = None,
    dry_run: bool = False,
    color: Optional[bool] = None,
    verbosity: int = 1,
    on_stream: StreamCallback = None,
    cancel_event: Optional[threading.Event] = None,
    emit_stderr: bool = True,
    history_messages: Optional[list[dict]] = None,
    think: Optional[str | bool] = None,
    allow_fallback: bool = True,
    drop_dir: Optional[Path] = None,
    system_persona: str = "",
    handoff_context: str = "",
    profile: str = "",
    routes_path: Optional[Path] = None,
    _fallback_depth: int = 0,
) -> ExecutionResult:
    """Execute a single delegated span against its configured harness backend."""
    span_id = span_id or new_span_id()
    backend = route_cfg.get("backend", "opencode")
    model = route_cfg.get("model", "opencode/deepseek-v4.1-flash-free")
    intent = route_cfg.get("intent", route_key)
    quota_tier = route_cfg.get("quota_tier", "FREE")
    timeout_s = float(route_cfg.get("timeout_s", 120))
    t0 = time.monotonic()

    sq_ctx = ""
    if inject_square and not dry_run:
        sq_ctx = format_square_context(limit=5, square_path=square_path, board_path=board_path)
    effective_prompt = f"{sq_ctx}\n{prompt}" if sq_ctx else prompt
    if system_persona or handoff_context:
        from nexus.entity import compose_entity_prompt

        subprocess_prompt = compose_entity_prompt(
            effective_prompt,
            system_persona=system_persona,
            handoff_context=handoff_context,
            include_handoff=bool(handoff_context) and not session_id,
        )
    else:
        subprocess_prompt = effective_prompt

    _emit(
        trace_id,
        span_id,
        parent_id=parent_id,
        route=route_key,
        intent=intent,
        backend=backend,
        model=model,
        quota_tier=quota_tier,
        status="RUNNING",
        step=f"dispatch->{backend}:{model}",
        prompt=prompt[:240],
        profile=profile,
        dry_run=dry_run,
        events_path=events_path,
        on_event=on_event,
    )

    if dry_run:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        out = f"[dry-run] {route_key} -> {backend}:{model} ({quota_tier}): {prompt}"
        if on_stream is not None:
            on_stream("text", out)
        _emit(
            trace_id,
            span_id,
            parent_id=parent_id,
            route=route_key,
            intent=intent,
            backend=backend,
            model=model,
            quota_tier=quota_tier,
            status="DONE",
            step="dry-run complete",
            prompt=prompt[:240],
            tokens_in=max(1, len(prompt) // 4),
            tokens_out=max(1, len(out) // 4),
            elapsed_ms=elapsed_ms,
            profile=profile,
            dry_run=True,
            events_path=events_path,
            on_event=on_event,
        )
        if emit_stderr and _status_enabled(verbosity, 2):
            _receipt(
                "step",
                route_key,
                backend,
                model,
                quota_tier,
                elapsed_ms,
                detail="dry-run",
                color=color,
                verbosity=verbosity,
            )
        return ExecutionResult(
            trace_id=trace_id,
            span_id=span_id,
            route=route_key,
            backend=backend,
            model=model,
            quota_tier=quota_tier,
            status="DONE",
            output=out,
            tokens_in=max(1, len(prompt) // 4),
            tokens_out=max(1, len(out) // 4),
            elapsed_ms=elapsed_ms,
        )

    missing = _require_binary(backend)
    if missing:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        _emit(
            trace_id,
            span_id,
            parent_id=parent_id,
            route=route_key,
            intent=intent,
            backend=backend,
            model=model,
            quota_tier=quota_tier,
            status="FAIL",
            step="preflight: missing backend binary",
            elapsed_ms=elapsed_ms,
            error=missing,
            profile=profile,
            events_path=events_path,
            on_event=on_event,
        )
        if emit_stderr:
            _receipt(
                "fail",
                route_key,
                backend,
                model,
                quota_tier,
                elapsed_ms,
                detail=missing,
                color=color,
                verbosity=verbosity,
            )
        return ExecutionResult(
            trace_id, span_id, route_key, backend, model, quota_tier, "FAIL", "", 0, 0, elapsed_ms, missing
        )

    enabled = sys.stderr.isatty() if color is None else color
    tool_count = 0

    def _finish(result: ExecutionResult) -> ExecutionResult:
        """Stop the spinner, then print one receipt covering every outcome."""
        stop_spinner_once()
        think_printer.flush()
        if not emit_stderr:
            return result
        kind = {
            "DONE": "ok",
            "FAIL": "fail",
            "TIMEOUT": "timeout",
            "INTERRUPTED": "stop",
        }.get(result.status, "info")
        detail = ""
        if result.status in ("FAIL", "TIMEOUT", "INTERRUPTED"):
            detail = (result.error or "").strip().splitlines()[0][:110] if result.error else ""
        _receipt(
            kind,
            result.route,
            result.backend,
            result.model,
            result.quota_tier,
            result.elapsed_ms,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            tokens_cached=result.tokens_cached,
            tools=tool_count,
            detail=detail,
            color=color,
            verbosity=verbosity,
            thinking_chars=max(think_printer.chars, len(result.thinking)),
        )
        return result

    def _interrupted(
        out_text: str = "",
        tok_in: int = 0,
        tok_out: int = 0,
        sid: str = "",
        *,
        run_model: Optional[str] = None,
        run_tier: Optional[str] = None,
        thinking: str = "",
    ) -> ExecutionResult:
        """Emit the terminal span for a user cancellation and return the result."""
        use_model = run_model or model
        use_tier = run_tier or quota_tier
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        _emit(
            trace_id,
            span_id,
            parent_id=parent_id,
            route=route_key,
            intent=intent,
            backend=backend,
            model=use_model,
            quota_tier=use_tier,
            status="FAIL",
            step="interrupted (Esc/Ctrl-C)",
            tokens_in=tok_in,
            tokens_out=tok_out,
            elapsed_ms=elapsed_ms,
            error="Cancelled by user",
            profile=profile,
            events_path=events_path,
            on_event=on_event,
        )
        return ExecutionResult(
            trace_id=trace_id,
            span_id=span_id,
            route=route_key,
            backend=backend,
            model=use_model,
            quota_tier=use_tier,
            status="INTERRUPTED",
            output=out_text,
            tokens_in=tok_in,
            tokens_out=tok_out,
            elapsed_ms=elapsed_ms,
            error="Cancelled by user",
            session_id=sid,
            thinking=thinking,
        )

    stop_spin = threading.Event()
    spin_thread: Optional[threading.Thread] = None
    spin_state: dict[str, object] = {"chars": 0, "preview": ""}
    spin_lock = threading.Lock()

    def stop_spinner_once() -> None:
        """Stop the spinner at most once and drop its live state."""
        nonlocal spin_thread
        with spin_lock:
            thread, spin_thread = spin_thread, None
        _SPIN_STATES.pop(id(stop_spin), None)
        if thread is not None:
            _stop_spin(stop_spin, thread)

    think_printer = _ThinkingStderrPrinter(
        route_key, verbosity, emit_stderr, enabled, spin_state, stop_spinner_once
    )

    if emit_stderr:
        _SPIN_STATES[id(stop_spin)] = spin_state
        spin_thread = threading.Thread(
            target=_spin_stderr,
            args=(route_key, stop_spin),
            kwargs={"color": color, "verbosity": verbosity},
            daemon=True,
        )
        spin_thread.start()

    try:
        def _resolve_fallback_route(from_key: str, from_cfg: dict) -> Optional[tuple[str, dict]]:
            fb_key = str(from_cfg.get("fallback_route") or "").strip()
            if not fb_key or fb_key == from_key:
                return None
            all_routes = load_routes(routes_path).get("routes", {})
            fb_cfg = all_routes.get(fb_key)
            if not isinstance(fb_cfg, dict):
                return None
            if fb_cfg.get("backend") not in ("opencode", "openai-compat", "colibri", "ollama"):
                return None
            return fb_key, dict(fb_cfg)

        def _handoff_to_fallback_route(fb_key: str, fb_cfg: dict) -> ExecutionResult:
            fb_backend = str(fb_cfg.get("backend", "opencode"))
            fb_model = str(fb_cfg.get("model", ""))
            fb_tier = str(fb_cfg.get("quota_tier", "FREE"))
            _emit(
                trace_id,
                span_id,
                parent_id=parent_id,
                route=fb_key,
                intent=intent,
                backend=fb_backend,
                model=fb_model,
                quota_tier=fb_tier,
                status="RUNNING",
                step=f"fallback->{fb_key}:{fb_model}",
                elapsed_ms=int((time.monotonic() - t0) * 1000),
                profile=profile,
                dry_run=dry_run,
                events_path=events_path,
                on_event=on_event,
            )
            if on_stream is not None:
                on_stream("status", f"fallback -> @{fb_key} ({fb_tier})")
            if emit_stderr and _status_enabled(verbosity):
                print(
                    status_line("warn", f"{_route_label(route_key)} · fallback -> @{fb_key} ({fb_tier})", color=enabled),
                    file=sys.stderr,
                )
            think_printer.flush()
            think_printer.reset_attempt()
            stop_spinner_once()
            return execute_route(
                trace_id,
                fb_key,
                fb_cfg,
                prompt,
                parent_id=parent_id,
                span_id=span_id,
                attach_file=attach_file,
                session_id=None,
                events_path=events_path,
                square_path=square_path,
                board_path=board_path,
                inject_square=inject_square,
                on_event=on_event,
                dry_run=dry_run,
                color=color,
                verbosity=verbosity,
                on_stream=on_stream,
                cancel_event=cancel_event,
                emit_stderr=emit_stderr,
                history_messages=history_messages,
                think=think,
                allow_fallback=False,
                drop_dir=drop_dir,
                system_persona=system_persona,
                handoff_context=handoff_context,
                profile=profile,
                routes_path=routes_path,
                _fallback_depth=_fallback_depth + 1,
            )

        if backend == "opencode":

            def _opencode_attempt(run_model: str, run_tier: str) -> tuple[ExecutionResult, bool]:
                """Run one OpenCode attempt on the shared span, returning (result, streamed_text)."""
                argv = ["opencode", "run", "--format", "json"]
                if route_cfg.get("auto_approve") is not False:
                    argv.append("--auto")
                argv.extend(["--model", run_model])
                if session_id:
                    argv.extend(["--session", session_id])
                else:
                    argv.append("--archived")
                if attach_file is not None:
                    argv.extend(["--file", str(attach_file)])
                argv.extend(["--", subprocess_prompt])

                acc_text = ""
                saw_tool = False
                stream_session = ""

                def _on_line(line: str) -> None:
                    nonlocal tool_count, stream_session, acc_text, saw_tool
                    s = line.strip()
                    if not s.startswith("{"):
                        return
                    try:
                        obj = json.loads(s)
                    except json.JSONDecodeError:
                        return
                    part = obj.get("part") if isinstance(obj.get("part"), dict) else obj
                    sid = obj.get("sessionID") or part.get("sessionID")
                    if sid and str(sid) != stream_session:
                        stream_session = str(sid)
                        if on_stream is not None:
                            on_stream("session", stream_session)
                    ev_type = obj.get("type", "")
                    ptype = part.get("type")
                    if ev_type == "text" or ptype == "text":
                        txt = part.get("text") or obj.get("text") or ""
                        if txt:
                            piece = _markdown_piece(acc_text, str(txt), saw_tool)
                            acc_text += piece
                            saw_tool = False
                            think_printer.flush()
                            if on_stream is not None:
                                on_stream("text", piece)
                        return
                    if ev_type in ("reasoning", "thinking") or ptype in ("reasoning", "thinking"):
                        txt = part.get("text") or obj.get("text") or ""
                        if txt:
                            think_printer.push(str(txt))
                            if on_stream is not None:
                                on_stream("thinking", str(txt))
                        return
                    if ptype == "tool" or ev_type in ("tool", "tool_use"):
                        tname = part.get("tool") or part.get("name") or "tool"
                        state = part.get("state") if isinstance(part.get("state"), dict) else {}
                        title = state.get("title") or part.get("title") or ""
                        summary = f"{tname} · {str(title)[:120]}" if title else str(tname)
                        saw_tool = True
                        tool_count += 1
                        think_printer.flush()
                        _emit(
                            trace_id,
                            span_id,
                            parent_id=parent_id,
                            route=route_key,
                            intent=intent,
                            backend=backend,
                            model=run_model,
                            quota_tier=run_tier,
                            status="RUNNING",
                            step=f"tool:{tname}",
                            elapsed_ms=int((time.monotonic() - t0) * 1000),
                            profile=profile,
                            events_path=events_path,
                            on_event=on_event,
                        )
                        if on_stream is not None:
                            on_stream("tool", summary)
                        if emit_stderr and _status_enabled(verbosity):
                            print(
                                status_line("step", f"{_route_label(route_key)} · tool {tool_count}: {summary}", color=enabled),
                                file=sys.stderr,
                            )

                if cancel_event is None:
                    rc, out_raw, err_raw, timed_out = _run_subprocess_bounded(argv, timeout_s, _on_line)
                else:
                    rc, out_raw, err_raw, timed_out = _run_subprocess_bounded(
                        argv, timeout_s, _on_line, cancel_event=cancel_event
                    )
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                stream = _parse_opencode_json_stream(out_raw)
                out_text, tok_in, tok_out, tok_cached, last_step = (
                    stream.text,
                    stream.tokens_in,
                    stream.tokens_out,
                    stream.tokens_cached,
                    stream.last_step,
                )
                disc_sid = stream.session_id
                final_sid = session_id or disc_sid
                if final_sid and not stream_session and on_stream is not None:
                    on_stream("session", final_sid)
                if tok_in == 0:
                    tok_in = max(1, len(prompt) // 4)
                if tok_out == 0 and out_text:
                    tok_out = max(1, len(out_text) // 4)

                if cancel_event is not None and cancel_event.is_set():
                    return _interrupted(
                        out_text,
                        tok_in,
                        tok_out,
                        final_sid,
                        run_model=run_model,
                        run_tier=run_tier,
                        thinking=stream.thinking,
                    ), bool(acc_text.strip() or stream.has_text)

                saw_text = bool(acc_text.strip() or stream.has_text)
                if timed_out:
                    err_msg = f"Timed out after {timeout_s:.0f}s"
                    _emit(
                        trace_id,
                        span_id,
                        parent_id=parent_id,
                        route=route_key,
                        intent=intent,
                        backend=backend,
                        model=run_model,
                        quota_tier=run_tier,
                        status="TIMEOUT",
                        step=err_msg,
                        tokens_in=tok_in,
                        tokens_out=tok_out,
                        elapsed_ms=elapsed_ms,
                        error=err_msg,
                        profile=profile,
                        events_path=events_path,
                        on_event=on_event,
                    )
                    return ExecutionResult(
                        trace_id, span_id, route_key, backend, run_model, run_tier, "TIMEOUT", out_text,
                        tok_in, tok_out, elapsed_ms, err_msg, final_sid, thinking=stream.thinking,
                    ), saw_text

                if rc != 0:
                    err_msg = _humanize_error(err_raw.strip() or out_raw.strip() or f"exit {rc}", 240)
                    _emit(
                        trace_id,
                        span_id,
                        parent_id=parent_id,
                        route=route_key,
                        intent=intent,
                        backend=backend,
                        model=run_model,
                        quota_tier=run_tier,
                        status="FAIL",
                        step=f"exit {rc}",
                        tokens_in=tok_in,
                        tokens_out=tok_out,
                        elapsed_ms=elapsed_ms,
                        error=err_msg,
                        profile=profile,
                        events_path=events_path,
                        on_event=on_event,
                    )
                    return ExecutionResult(
                        trace_id, span_id, route_key, backend, run_model, run_tier, "FAIL", out_text,
                        tok_in, tok_out, elapsed_ms, err_msg, final_sid, thinking=stream.thinking,
                    ), saw_text

                step_label = f"{last_step} (ses:{final_sid[:12]})" if final_sid else last_step
                _emit(
                    trace_id,
                    span_id,
                    parent_id=parent_id,
                    route=route_key,
                    intent=intent,
                    backend=backend,
                    model=run_model,
                    quota_tier=run_tier,
                    status="DONE",
                    step=step_label,
                    tokens_in=tok_in,
                    tokens_out=tok_out,
                    elapsed_ms=elapsed_ms,
                    profile=profile,
                    events_path=events_path,
                    on_event=on_event,
                )
                return ExecutionResult(
                    trace_id, span_id, route_key, backend, run_model, run_tier, "DONE", out_text,
                    tok_in, tok_out, elapsed_ms, "", final_sid, tok_cached, thinking=stream.thinking,
                ), saw_text

            result, saw_opencode_text = _opencode_attempt(model, quota_tier)
            fb_model = route_cfg.get("fallback")
            if (
                result.status == "FAIL"
                and allow_fallback
                and not saw_opencode_text
                and classify_http_error(result.error) != "auth"
                and isinstance(fb_model, str)
                and fb_model.strip()
                and fb_model.strip() != model
            ):
                fb_model = fb_model.strip()
                fb_tier = _infer_fallback_tier(fb_model, quota_tier)
                _emit(
                    trace_id,
                    span_id,
                    parent_id=parent_id,
                    route=route_key,
                    intent=intent,
                    backend=backend,
                    model=fb_model,
                    quota_tier=fb_tier,
                    status="RUNNING",
                    step=f"fallback->{fb_model}",
                    elapsed_ms=int((time.monotonic() - t0) * 1000),
                    profile=profile,
                    events_path=events_path,
                    on_event=on_event,
                )
                if on_stream is not None:
                    on_stream("status", f"fallback -> {fb_model} ({fb_tier})")
                if emit_stderr and _status_enabled(verbosity):
                    print(
                        status_line("warn", f"{_route_label(route_key)} · fallback -> {fb_model} ({fb_tier})", color=enabled),
                        file=sys.stderr,
                    )
                think_printer.flush()
                think_printer.reset_attempt()
                result, saw_opencode_text = _opencode_attempt(fb_model, fb_tier)
            if (
                result.status in ("FAIL", "TIMEOUT")
                and allow_fallback
                and _fallback_depth == 0
                and not saw_opencode_text
                and classify_http_error(result.error) != "auth"
            ):
                fb_route_target = _resolve_fallback_route(route_key, route_cfg)
                if fb_route_target is not None:
                    return _handoff_to_fallback_route(*fb_route_target)
            return _finish(result)

        if backend in ("openai-compat", "colibri", "ollama"):
            if cancel_event is not None and cancel_event.is_set():
                return _finish(_interrupted())

            def _http_endpoint(cfg: dict, kind: str) -> str:
                return str(
                    cfg.get(
                        "endpoint",
                        "http://127.0.0.1:8000/v1/chat/completions" if kind == "openai-compat" else "http://127.0.0.1:11434/api/chat",
                    )
                )

            def _http_attempt(attempt_route: str, attempt_cfg: dict) -> tuple[ExecutionResult, bool]:
                a_backend = attempt_cfg.get("backend", backend)
                a_kind = "ollama" if a_backend == "ollama" else "openai-compat"
                a_model = str(attempt_cfg.get("model", model))
                a_tier = str(attempt_cfg.get("quota_tier", quota_tier))
                a_timeout_s = float(attempt_cfg.get("timeout_s", timeout_s))
                a_endpoint = _http_endpoint(attempt_cfg, a_kind)
                resolved_think = _normalize_think(think if think is not None else attempt_cfg.get("think"))
                api_key = resolve_route_api_key(attempt_cfg)
                req_headers = {"Authorization": f"Bearer {api_key}"} if api_key else None

                messages: list[dict] = []
                sys_parts = [part for part in ((system_persona or "").strip(), sq_ctx.strip()) if part]
                if sys_parts:
                    messages.append({"role": "system", "content": "\n\n".join(sys_parts)})
                if history_messages:
                    messages.extend(_bounded_history(history_messages))
                elif (handoff_context or "").strip():
                    messages.append({"role": "system", "content": handoff_context.strip()})
                user_msg: dict = {"role": "user", "content": prompt}
                if attach_file is not None:
                    try:
                        prefix, image_b64 = _attachment_parts(attach_file, a_kind)
                    except OSError as exc:
                        elapsed_ms = int((time.monotonic() - t0) * 1000)
                        err_msg = str(exc)
                        _emit(
                            trace_id,
                            span_id,
                            parent_id=parent_id,
                            route=attempt_route,
                            intent=intent,
                            backend=a_backend,
                            model=a_model,
                            quota_tier=a_tier,
                            status="FAIL",
                            step=err_msg,
                            elapsed_ms=elapsed_ms,
                            error=err_msg,
                            profile=profile,
                            events_path=events_path,
                            on_event=on_event,
                        )
                        return ExecutionResult(
                            trace_id, span_id, attempt_route, a_backend, a_model, a_tier, "FAIL", "", 0, 0, elapsed_ms, err_msg
                        ), False
                    if prefix:
                        user_msg["content"] = prefix + prompt
                    if image_b64:
                        user_msg["images"] = [image_b64]
                messages.append(user_msg)

                out_chunks: list[str] = []
                thinking_chunks: list[str] = []
                tok_in = tok_out = 0
                cancelled = timed_out = saw_first_byte = False
                errors: list[str] = []
                saw_tool_after_call = False
                a_t0 = time.monotonic()
                deadline = a_t0 + max(0.1, a_timeout_s)

                for _round in range(4):
                    if cancel_event is not None and cancel_event.is_set():
                        cancelled = True
                        break
                    if deadline - time.monotonic() <= 0:
                        timed_out = True
                        break

                    body: dict[str, Any] = {"model": a_model, "messages": messages, "stream": True}
                    if a_kind == "ollama" and resolved_think is not None:
                        body["think"] = resolved_think
                    if attempt_cfg.get("tools") is not False:
                        body["tools"] = [SEND_FILE_TOOL_SCHEMA]
                    payload = json.dumps(body).encode("utf-8")

                    round_chunks: list[str] = []
                    ollama_calls: list[dict[str, Any]] = []
                    openai_compat_slots: dict[int, dict[str, Any]] = {}
                    round_in = round_out = 0
                    resp_box: list = []
                    closed = threading.Event()
                    sink: "queue.Queue[Optional[str]]" = queue.Queue()
                    reader_kwargs = {"headers": req_headers} if req_headers else {}
                    reader = threading.Thread(
                        target=_read_http_response,
                        args=(a_endpoint, payload, a_timeout_s, closed, resp_box, errors, sink),
                        kwargs=reader_kwargs,
                        daemon=True,
                    )
                    reader.start()
                    try:
                        while True:
                            if cancel_event is not None and cancel_event.is_set():
                                cancelled = True
                                break
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                timed_out = True
                                break
                            try:
                                line = sink.get(timeout=min(0.1, remaining))
                            except queue.Empty:
                                continue
                            if line is None:
                                break
                            for obj in _stream_objects(line, a_kind):
                                if a_kind == "ollama":
                                    thinking, text = _ollama_stream_chunk(obj)
                                    ollama_calls.extend(_ollama_tool_calls(obj))
                                    if obj.get("done"):
                                        round_in = int(obj.get("prompt_eval_count") or round_in)
                                        round_out = int(obj.get("eval_count") or round_out)
                                else:
                                    thinking, text = _openai_compat_stream_chunk(obj)
                                    _openai_compat_tool_delta(obj, openai_compat_slots)
                                    usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else {}
                                    round_in = int(usage.get("prompt_tokens") or round_in)
                                    round_out = int(usage.get("completion_tokens") or round_out)
                                if thinking:
                                    saw_first_byte = True
                                    thinking_chunks.append(thinking)
                                    think_printer.push(thinking)
                                    if on_stream is not None:
                                        on_stream("thinking", thinking)
                                if text:
                                    saw_first_byte = True
                                    think_printer.flush()
                                    piece = _markdown_piece("".join(out_chunks), text, True) if saw_tool_after_call else text
                                    saw_tool_after_call = False
                                    out_chunks.append(piece)
                                    round_chunks.append(piece)
                                    if on_stream is not None:
                                        on_stream("text", piece)
                    finally:
                        closed.set()
                        for sock in resp_box:
                            if sock is None:
                                continue
                            try:
                                sock.shutdown(socket.SHUT_RDWR)
                            except OSError:
                                pass
                        reader.join(timeout=1.0)

                    tok_in += round_in
                    tok_out += round_out
                    round_calls = ollama_calls if a_kind == "ollama" else _finalize_openai_compat_calls(openai_compat_slots)
                    if round_calls:
                        saw_first_byte = True
                    if cancelled or timed_out or (errors and not round_calls) or not round_calls:
                        break

                    messages.append(
                        {
                            "role": "assistant",
                            "content": "".join(round_chunks),
                            "tool_calls": [
                                {
                                    "type": "function",
                                    **({"id": tc["id"]} if tc.get("id") else {}),
                                    "function": {
                                        "name": tc["name"],
                                        "arguments": tc["arguments"] if a_kind == "ollama" else tc["raw_args"],
                                    },
                                }
                                for tc in round_calls
                            ],
                        }
                    )
                    for tc in round_calls:
                        t_name = tc["name"] or "send_file"
                        tool_res, summary = _run_drop_tool_call(tc, drop_dir=drop_dir)
                        nonlocal tool_count
                        tool_count += 1
                        think_printer.flush()
                        _emit(
                            trace_id,
                            span_id,
                            parent_id=parent_id,
                            route=attempt_route,
                            intent=intent,
                            backend=a_backend,
                            model=a_model,
                            quota_tier=a_tier,
                            status="RUNNING",
                            step=f"tool:{t_name}",
                            elapsed_ms=int((time.monotonic() - t0) * 1000),
                            profile=profile,
                            events_path=events_path,
                            on_event=on_event,
                        )
                        if on_stream is not None:
                            on_stream("tool", summary)
                        if emit_stderr and _status_enabled(verbosity):
                            print(
                                status_line("step", f"{_route_label(attempt_route)} · tool {tool_count}: {summary}", color=enabled),
                                file=sys.stderr,
                            )
                        tool_msg: dict[str, Any] = {
                            "role": "tool",
                            "tool_name": t_name,
                            "content": json.dumps(tool_res),
                        }
                        if tc.get("id"):
                            tool_msg["tool_call_id"] = tc["id"]
                        messages.append(tool_msg)
                    saw_tool_after_call = True

                elapsed_ms = int((time.monotonic() - t0) * 1000)
                attempt_ms = (time.monotonic() - a_t0) * 1000.0
                out_text = "".join(out_chunks).strip()
                if not tok_in:
                    tok_in = max(1, len(prompt) // 4)
                if not tok_out and out_text:
                    tok_out = max(1, len(out_text) // 4)

                if cancelled:
                    return (
                        _interrupted(out_text, tok_in, tok_out, thinking="".join(thinking_chunks).strip()),
                        saw_first_byte,
                    )
                if timed_out:
                    err_msg = f"Timed out after {a_timeout_s:.0f}s"
                    _HEALTH_TRACKER.record_failure(
                        endpoint=a_endpoint, route_key=attempt_route, model=a_model, kind="transient"
                    )
                    _emit(
                        trace_id,
                        span_id,
                        parent_id=parent_id,
                        route=attempt_route,
                        intent=intent,
                        backend=a_backend,
                        model=a_model,
                        quota_tier=a_tier,
                        status="TIMEOUT",
                        step=err_msg,
                        tokens_in=tok_in,
                        tokens_out=tok_out,
                        elapsed_ms=elapsed_ms,
                        error=err_msg,
                        profile=profile,
                        events_path=events_path,
                        on_event=on_event,
                    )
                    return ExecutionResult(
                        trace_id, span_id, attempt_route, a_backend, a_model, a_tier, "TIMEOUT", out_text,
                        tok_in, tok_out, elapsed_ms, err_msg,
                        thinking="".join(thinking_chunks).strip(),
                    ), saw_first_byte
                if errors and not out_text:
                    err_msg = _humanize_error(errors[0], 240)
                    _HEALTH_TRACKER.record_failure(
                        endpoint=a_endpoint,
                        route_key=attempt_route,
                        model=a_model,
                        kind=classify_http_error(err_msg),
                    )
                    _emit(
                        trace_id,
                        span_id,
                        parent_id=parent_id,
                        route=attempt_route,
                        intent=intent,
                        backend=a_backend,
                        model=a_model,
                        quota_tier=a_tier,
                        status="FAIL",
                        step="http stream error",
                        elapsed_ms=elapsed_ms,
                        error=err_msg,
                        profile=profile,
                        events_path=events_path,
                        on_event=on_event,
                    )
                    return ExecutionResult(
                        trace_id, span_id, attempt_route, a_backend, a_model, a_tier, "FAIL", "", 0, 0, elapsed_ms, err_msg
                    ), saw_first_byte

                _HEALTH_TRACKER.record_success(
                    endpoint=a_endpoint, route_key=attempt_route, model=a_model, elapsed_ms=attempt_ms
                )
                _emit(
                    trace_id,
                    span_id,
                    parent_id=parent_id,
                    route=attempt_route,
                    intent=intent,
                    backend=a_backend,
                    model=a_model,
                    quota_tier=a_tier,
                    status="DONE",
                    step="http stream complete",
                    tokens_in=tok_in,
                    tokens_out=tok_out,
                    elapsed_ms=elapsed_ms,
                    profile=profile,
                    events_path=events_path,
                    on_event=on_event,
                )
                return ExecutionResult(
                    trace_id, span_id, attempt_route, a_backend, a_model, a_tier, "DONE", out_text, tok_in, tok_out, elapsed_ms,
                    thinking="".join(thinking_chunks).strip(),
                ), saw_first_byte

            primary_kind = "ollama" if backend == "ollama" else "openai-compat"
            primary_ep = _http_endpoint(route_cfg, primary_kind)
            fb_target = _resolve_fallback_route(route_key, route_cfg) if (allow_fallback and _fallback_depth == 0) else None
            if fb_target and _HEALTH_TRACKER.is_cooled_down(endpoint=primary_ep, route_key=route_key, model=model):
                fb_key, fb_cfg = fb_target
                if fb_cfg.get("backend") in ("openai-compat", "colibri", "ollama"):
                    fb_kind = "ollama" if fb_cfg.get("backend") == "ollama" else "openai-compat"
                    fb_ep = _http_endpoint(fb_cfg, fb_kind)
                    if not _HEALTH_TRACKER.is_cooled_down(endpoint=fb_ep, route_key=fb_key, model=str(fb_cfg.get("model", ""))):
                        if on_stream is not None:
                            on_stream("status", f"cooldown fallback -> @{fb_key}")
                        result, _ = _http_attempt(fb_key, fb_cfg)
                        return _finish(result)
                else:
                    return _handoff_to_fallback_route(fb_key, fb_cfg)

            result, saw_first_byte = _http_attempt(route_key, route_cfg)
            if (
                result.status in ("FAIL", "TIMEOUT")
                and allow_fallback
                and _fallback_depth == 0
                and not saw_first_byte
                and classify_http_error(result.error) != "auth"
                and fb_target is not None
            ):
                fb_key, fb_cfg = fb_target
                if fb_cfg.get("backend") not in ("openai-compat", "colibri", "ollama"):
                    return _handoff_to_fallback_route(fb_key, fb_cfg)
                fb_model = str(fb_cfg.get("model", ""))
                fb_tier = str(fb_cfg.get("quota_tier", "FREE"))
                _emit(
                    trace_id,
                    span_id,
                    parent_id=parent_id,
                    route=fb_key,
                    intent=intent,
                    backend=fb_cfg.get("backend", backend),
                    model=fb_model,
                    quota_tier=fb_tier,
                    status="RUNNING",
                    step=f"fallback->{fb_key}:{fb_model}",
                    elapsed_ms=int((time.monotonic() - t0) * 1000),
                    profile=profile,
                    dry_run=dry_run,
                    events_path=events_path,
                    on_event=on_event,
                )
                if on_stream is not None:
                    on_stream("status", f"fallback -> @{fb_key} ({fb_tier})")
                if emit_stderr and _status_enabled(verbosity):
                    print(
                        status_line("warn", f"{_route_label(route_key)} · fallback -> @{fb_key} ({fb_tier})", color=enabled),
                        file=sys.stderr,
                    )
                think_printer.flush()
                think_printer.reset_attempt()
                result, _ = _http_attempt(fb_key, fb_cfg)
            return _finish(result)

        if backend == "agy":
            if system_persona or handoff_context:
                from nexus.entity import compose_entity_prompt

                agy_prompt = compose_entity_prompt(
                    prompt,
                    system_persona=system_persona,
                    handoff_context=handoff_context,
                    include_handoff=bool(handoff_context),
                )
            else:
                agy_prompt = prompt
            argv = ["agy", "-p", agy_prompt, "--output-format", "stream-json", "--print-timeout", f"{int(timeout_s)}s"]
            agy_thinking: list[str] = []

            def _on_agy_line(line: str) -> None:
                nonlocal tool_count
                s = line.strip()
                if not s.startswith("{"):
                    return
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    return
                ev = obj.get("event") or obj.get("type") or ""
                if ev == "result":
                    res = obj.get("result") if isinstance(obj.get("result"), dict) else {}
                    txt = res.get("response") or _agy_stream_text(obj) or _agy_stream_text(res)
                    if txt:
                        think_printer.flush()
                        if on_stream is not None:
                            on_stream("text", str(txt))
                elif ev in ("thinking", "reasoning"):
                    txt = _agy_stream_text(obj)
                    if txt:
                        agy_thinking.append(str(txt))
                        think_printer.push(str(txt))
                        if on_stream is not None:
                            on_stream("thinking", str(txt))
                elif ev in ("tool", "tool_use"):
                    tname = obj.get("tool") or obj.get("name") or "tool"
                    state = obj.get("state") if isinstance(obj.get("state"), dict) else {}
                    title = obj.get("title") or state.get("title") or ""
                    summary = f"{tname} · {str(title)[:120]}" if title else str(tname)
                    tool_count += 1
                    think_printer.flush()
                    if on_stream is not None:
                        on_stream("tool", summary)
                else:
                    txt = _agy_stream_text(obj)
                    if txt:
                        think_printer.flush()
                        if on_stream is not None:
                            on_stream("text", str(txt))

            if cancel_event is None:
                rc, out_raw, err_raw, timed_out = _run_subprocess_bounded(argv, timeout_s, _on_agy_line)
            else:
                rc, out_raw, err_raw, timed_out = _run_subprocess_bounded(
                    argv, timeout_s, _on_agy_line, cancel_event=cancel_event
                )
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            out_text = out_raw.strip()
            tok_in = max(1, len(prompt) // 4)
            tok_out = max(1, len(out_text) // 4)
            for line in out_raw.splitlines():
                if line.strip().startswith("{"):
                    try:
                        ev = json.loads(line)
                        if ev.get("event") == "result":
                            res = ev.get("result", {})
                            out_text = res.get("response", out_text)
                            u = res.get("usage", {})
                            tok_in = int(u.get("input_tokens", tok_in) or tok_in)
                            tok_out = int(u.get("output_tokens", tok_out) or tok_out)
                    except json.JSONDecodeError:
                        pass

            if cancel_event is not None and cancel_event.is_set():
                return _finish(
                    _interrupted(out_text, tok_in, tok_out, thinking="".join(agy_thinking).strip())
                )

            status = "TIMEOUT" if timed_out else ("DONE" if rc == 0 else "FAIL")
            err_msg = f"Timed out after {timeout_s:.0f}s" if timed_out else (_humanize_error(err_raw, 240) if rc != 0 else "")
            _emit(
                trace_id,
                span_id,
                parent_id=parent_id,
                route=route_key,
                intent=intent,
                backend=backend,
                model=model,
                quota_tier=quota_tier,
                status=status,
                step="agy complete" if status == "DONE" else err_msg,
                tokens_in=tok_in,
                tokens_out=tok_out,
                elapsed_ms=elapsed_ms,
                error=err_msg,
                events_path=events_path,
                on_event=on_event,
            )
            return _finish(ExecutionResult(
                trace_id, span_id, route_key, backend, model, quota_tier, status, out_text, tok_in, tok_out, elapsed_ms, err_msg,
                thinking="".join(agy_thinking).strip(),
            ))

        raise ValueError(f"Unsupported backend: {backend}")

    except KeyboardInterrupt:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        err_msg = "Interrupted by user (Ctrl-C)"
        _emit(
            trace_id,
            span_id,
            parent_id=parent_id,
            route=route_key,
            intent=intent,
            backend=backend,
            model=model,
            quota_tier=quota_tier,
            status="FAIL",
            step="interrupted (Ctrl-C)",
            elapsed_ms=elapsed_ms,
            error=err_msg,
            profile=profile,
            events_path=events_path,
            on_event=on_event,
        )
        return _finish(ExecutionResult(
            trace_id, span_id, route_key, backend, model, quota_tier, "INTERRUPTED", "", 0, 0, elapsed_ms, err_msg
        ))

    except Exception as exc:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        err_msg = _humanize_error(str(exc), 240)
        _emit(
            trace_id,
            span_id,
            parent_id=parent_id,
            route=route_key,
            intent=intent,
            backend=backend,
            model=model,
            quota_tier=quota_tier,
            status="FAIL",
            step="exception",
            elapsed_ms=elapsed_ms,
            error=err_msg,
            profile=profile,
            events_path=events_path,
            on_event=on_event,
        )
        return _finish(ExecutionResult(
            trace_id, span_id, route_key, backend, model, quota_tier, "FAIL", "", 0, 0, elapsed_ms, err_msg
        ))
    finally:
        stop_spinner_once()
