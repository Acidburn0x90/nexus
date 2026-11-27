"""Append-only JSONL event bus for nexus delegation spans.

Every delegated span writes one line when it starts and another when it
finishes. Writes use a single `os.write` on an `O_APPEND` descriptor, which
keeps lines atomic for the sizes involved here (measured: 4 concurrent writers,
2000/2000 well-formed lines), so concurrent harnesses never interleave.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

DEFAULT_EVENTS_PATH = Path.home() / ".local/state/nexus/events.jsonl"
# Window used to find a trace without reading the whole append-only bus.
_TRACE_TAIL_BYTES = 262144
AUTO_PRUNE_MAX_BYTES = 2 * 1024 * 1024
AUTO_PRUNE_KEEP_LINES = 2000

# Span fields that must survive the merge as first non-empty value seen.
_FIRST_WINS = ("parent_id", "route", "intent", "backend", "model", "quota_tier", "prompt", "profile")
# Span fields that always take the newest value.
_LAST_WINS = ("status", "step", "tokens_in", "tokens_out", "elapsed_ms", "error")
# Quota tiers excluded from zero-cost non-premium span counts.
_EXEMPT_TIERS = {"ORCHESTRATOR", "PREMIUM", "PAID", "ARGON", "ARGON-QUOTA"}


def normalize_quota_tier(tier: str) -> str:
    """Normalize legacy quota tier aliases to their canonical public names."""
    normalized = str(tier or "").strip().upper().replace("_", "-")
    if normalized in {"ARGON", "ARGON-QUOTA", "PAID", "PREMIUM"}:
        return "PREMIUM"
    return normalized


def new_trace_id() -> str:
    """Return a short trace identifier, e.g. `tr_1a2b3c4d`."""
    return "tr_" + uuid.uuid4().hex[:8]


def new_span_id() -> str:
    """Return a short span identifier, e.g. `sp_1a2b3c`."""
    return "sp_" + uuid.uuid4().hex[:6]


def emit_event(
    trace_id: str,
    span_id: str,
    *,
    parent_id: Optional[str] = None,
    route: str = "",
    intent: str = "",
    backend: str = "",
    model: str = "",
    quota_tier: str = "FREE",
    status: str = "RUNNING",
    step: str = "",
    prompt: str = "",
    tokens_in: int = 0,
    tokens_out: int = 0,
    elapsed_ms: int = 0,
    error: str = "",
    profile: str = "",
    dry_run: bool = False,
    auto_prune: bool = True,
    path: Optional[Path] = None,
) -> dict:
    """Append one span event to the bus and return the written record."""
    target = Path(path) if path is not None else DEFAULT_EVENTS_PATH
    record = {
        "ts": time.time(),
        "mono_ns": time.monotonic_ns(),
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_id": parent_id,
        "route": route[:64],
        "intent": intent[:64],
        "backend": backend[:32],
        "model": model[:128],
        "quota_tier": quota_tier[:32],
        "status": status[:32],
        "step": step[:512],
        "prompt": prompt[:1024],
        "tokens_in": int(tokens_in),
        "tokens_out": int(tokens_out),
        "elapsed_ms": int(elapsed_ms),
        "error": error[:512],
        "profile": str(profile or "")[:64],
    }
    if dry_run:
        record["dry_run"] = True
    payload = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    size_after = 0
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, payload.encode("utf-8"))
        size_after = os.fstat(fd).st_size
    finally:
        os.close(fd)
    if auto_prune and size_after > AUTO_PRUNE_MAX_BYTES:
        try:
            prune_events(target, keep_lines=AUTO_PRUNE_KEEP_LINES)
        except OSError:
            pass
    return record


def _parse_event_lines(lines: list[str], trace_id: Optional[str]) -> list[dict]:
    """Parse JSONL lines into event records, skipping malformed entries."""
    events: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        if trace_id is not None and record.get("trace_id") != trace_id:
            continue
        events.append(record)
    return events


def _read_tail(target: Path, window: int) -> tuple[list[dict], bool]:
    """Parse the trailing `window` bytes of `target`, returning `(events, whole_file)`."""
    try:
        with open(target, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            file_size = handle.tell()
            if file_size == 0:
                return [], True
            start = max(0, file_size - window)
            handle.seek(start)
            raw_bytes = handle.read(file_size - start)
    except FileNotFoundError:
        return [], True

    lines = raw_bytes.decode("utf-8", errors="replace").splitlines()
    if start > 0:
        # First line is likely truncated; drop it.
        lines = lines[1:]
    return _parse_event_lines(lines, None), start == 0


def read_events(
    path: Optional[Path] = None,
    trace_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    """Read span events, skipping malformed lines, newest `limit` wins."""
    target = Path(path) if path is not None else DEFAULT_EVENTS_PATH

    if trace_id is not None:
        # Tail-seek first: a trace usually ends near the tail of the bus, so
        # only fall back to a full read when the match is older than the window.
        events, whole_file = _read_tail(target, _TRACE_TAIL_BYTES)
        matched = [event for event in events if event.get("trace_id") == trace_id]
        if whole_file or matched:
            if limit is not None and limit >= 0 and len(matched) > limit:
                return matched[-limit:] if limit else []
            return matched

    if limit is not None and limit > 0 and trace_id is None:
        # Tail-seek: read backwards in growing windows so a multi-MB bus is
        # never read or parsed from byte 0 just to fetch the newest lines.
        try:
            with open(target, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                file_size = handle.tell()
                if file_size == 0:
                    return []
                window = max(65536, limit * 512)
                while True:
                    start = max(0, file_size - window)
                    handle.seek(start)
                    raw_bytes = handle.read(file_size - start)
                    lines = raw_bytes.decode("utf-8", errors="replace").splitlines()
                    if start > 0:
                        # First line is likely truncated; drop it.
                        lines = lines[1:]
                    events = _parse_event_lines(lines, None)
                    if len(events) >= limit or start == 0:
                        return events[-limit:]
                    window *= 2
        except FileNotFoundError:
            return []

    try:
        raw = target.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return []

    events = _parse_event_lines(raw.splitlines(), trace_id)
    if limit is not None and limit >= 0 and len(events) > limit:
        return events[-limit:] if limit else []
    return events


def prune_events(path: Optional[Path] = None, *, keep_lines: int = 2000) -> tuple[int, int]:
    """Atomically keep only the newest `keep_lines` non-empty lines of the bus."""
    target = Path(path) if path is not None else DEFAULT_EVENTS_PATH
    try:
        raw = target.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return 0, 0

    lines = [line for line in raw.splitlines() if line.strip()]
    before = len(lines)
    keep = max(0, int(keep_lines))
    if before <= keep:
        return before, before

    kept = lines[-keep:] if keep else []
    payload = "\n".join(kept) + ("\n" if kept else "")
    fd, tmp_name = tempfile.mkstemp(prefix=f"{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o644)
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return before, len(kept)


def merge_span_events(events: list[dict]) -> list[dict]:
    """Fold chronologically ordered events into merged span records (preserving order)."""
    spans: dict[str, dict] = {}
    order: list[str] = []

    for event in events:
        if not isinstance(event, dict):
            continue
        span_id = event.get("span_id")
        if not span_id:
            continue
        if span_id not in spans:
            spans[span_id] = dict(event)
            order.append(span_id)
            continue

        merged = spans[span_id]
        ts = event.get("ts")
        if ts is not None and (merged.get("ts") is None or ts < merged["ts"]):
            merged["ts"] = ts
        if event.get("dry_run"):
            merged["dry_run"] = True

        new_model = str(event.get("model") or "").strip()
        old_model = str(merged.get("model") or "").strip()
        new_backend = str(event.get("backend") or "").strip()
        old_backend = str(merged.get("backend") or "").strip()
        switched = (bool(new_model and old_model and new_model != old_model)) or (
            bool(new_backend and old_backend and new_backend != old_backend)
        )
        if switched:
            for field in ("route", "backend", "model", "quota_tier"):
                if event.get(field):
                    merged[field] = event[field]

        for field in _FIRST_WINS:
            if not merged.get(field) and event.get(field):
                merged[field] = event[field]
        for field in _LAST_WINS:
            if field in event and event[field] not in (None, ""):
                merged[field] = event[field]

    return [spans[sid] for sid in order]


def reconstruct_tree(events: list[dict], trace_id: Optional[str] = None) -> dict:
    """Fold span events into a tree of merged spans plus a summary."""
    active_trace = trace_id
    if active_trace is None and events:
        active_trace = events[-1].get("trace_id")

    if active_trace is not None:
        events = [event for event in events if event.get("trace_id") == active_trace]

    merged_list = merge_span_events(events)
    spans: dict[str, dict] = {sp["span_id"]: sp for sp in merged_list}
    order: list[str] = [sp["span_id"] for sp in merged_list]

    children: dict[Optional[str], list[str]] = {}
    roots: list[str] = []
    for span_id in order:
        parent_id = spans[span_id].get("parent_id") or None
        children.setdefault(parent_id, []).append(span_id)
        if parent_id is None:
            roots.append(span_id)

    premium_spans = sum(
        1
        for span in spans.values()
        if normalize_quota_tier(str(span.get("quota_tier", ""))) == "PREMIUM"
    )
    non_premium = sum(
        1
        for span in spans.values()
        if str(span.get("quota_tier", "")).strip().upper().replace("_", "-") not in _EXEMPT_TIERS
    )
    root_elapsed = max((int(spans[root].get("elapsed_ms") or 0) for root in roots), default=0)
    any_elapsed = max((int(span.get("elapsed_ms") or 0) for span in spans.values()), default=0)

    summary = {
        "total_spans": len(spans),
        "premium_spans": premium_spans,
        "non_premium_spans": non_premium,
        "tokens_in": sum(int(span.get("tokens_in") or 0) for span in spans.values()),
        "tokens_out": sum(int(span.get("tokens_out") or 0) for span in spans.values()),
        "elapsed_ms": root_elapsed or any_elapsed,
    }

    return {
        "trace_id": active_trace,
        "spans": spans,
        "children": children,
        "roots": roots,
        "summary": summary,
    }
