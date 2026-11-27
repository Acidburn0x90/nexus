"""Verbatim write-ahead complaint and false-pass ledger (LifeOS Errata pattern)."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from nexus.events import DEFAULT_EVENTS_PATH, merge_span_events, read_events
from nexus.visualizer import render_panel, status_line, truncate_ansi

DEFAULT_ERRATA_PATH = Path.home() / ".local/state/nexus/errata.jsonl"
ERRATA_KINDS = ("complaint", "false-pass", "bug", "correction", "feature-request")
_VALID_SEVERITIES = ("low", "medium", "high")
_EXEMPT_TIERS = {"ORCHESTRATOR"}


def resolve_errata_path(path: Optional[Path] = None) -> Path:
    """Resolve errata ledger path, honoring `NEXUS_ERRATA_PATH` over the default."""
    if path is not None:
        return Path(path).expanduser()
    env_path = os.environ.get("NEXUS_ERRATA_PATH", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return DEFAULT_ERRATA_PATH


def _pick_span_context(
    events_path: Path,
    *,
    trace_id: str = "",
) -> dict[str, str]:
    """Find the most relevant non-dry-run, non-ORCHESTRATOR span for errata joins."""
    wanted_trace = str(trace_id or "").strip()
    raw = read_events(
        path=events_path,
        trace_id=wanted_trace or None,
        limit=80,
    )
    merged = merge_span_events(raw)
    candidates = [
        sp
        for sp in merged
        if not sp.get("dry_run")
        and str(sp.get("quota_tier") or "").strip().upper() not in _EXEMPT_TIERS
    ]
    if not candidates:
        candidates = [sp for sp in merged if not sp.get("dry_run")]
    if not candidates:
        return {}
    chosen = candidates[-1]
    raw_route = str(chosen.get("route") or "").strip()
    clean_route = (
        raw_route[len("in-charge:") :]
        if raw_route.startswith("in-charge:")
        else raw_route
    )
    return {
        "trace_id": str(chosen.get("trace_id") or ""),
        "span_id": str(chosen.get("span_id") or ""),
        "route": clean_route,
        "model": str(chosen.get("model") or ""),
        "span_status": str(chosen.get("status") or ""),
    }


def record_erratum(
    message: str,
    *,
    trace_id: str = "",
    route: str = "",
    model: str = "",
    kind: str = "complaint",
    severity: str = "medium",
    false_pass: Optional[bool] = None,
    source: str = "user",
    events_path: Path = DEFAULT_EVENTS_PATH,
    path: Optional[Path] = None,
) -> dict:
    """Append a verbatim erratum record atomically with zero LLM calls on hot path."""
    verbatim = str(message or "")
    if not verbatim.strip():
        raise ValueError("erratum message cannot be empty")

    target = resolve_errata_path(path)
    ctx = _pick_span_context(Path(events_path), trace_id=trace_id)

    resolved_trace = str(trace_id or ctx.get("trace_id") or "").strip()
    resolved_route = str(route or ctx.get("route") or "").strip()
    if resolved_route.startswith("@"):
        resolved_route = resolved_route[1:]
    resolved_model = str(model or ctx.get("model") or "").strip()
    resolved_span_status = str(ctx.get("span_status") or "").strip().upper()

    norm_kind = str(kind or "complaint").strip().lower()
    if norm_kind not in ERRATA_KINDS:
        norm_kind = "complaint"

    norm_sev = str(severity or "medium").strip().lower()
    if norm_sev not in _VALID_SEVERITIES:
        norm_sev = "medium"

    if false_pass is None:
        is_false_pass = resolved_span_status == "DONE"
    else:
        is_false_pass = bool(false_pass)
    if is_false_pass and norm_kind == "complaint":
        norm_kind = "false-pass"

    record: dict[str, Any] = {
        "id": "err_" + uuid.uuid4().hex[:8],
        "ts": time.time(),
        "source": str(source or "user").strip()[:32] or "user",
        "kind": norm_kind,
        "severity": norm_sev,
        "message": verbatim,
        "trace_id": resolved_trace,
        "span_id": str(ctx.get("span_id") or ""),
        "route": resolved_route,
        "model": resolved_model,
        "span_status": resolved_span_status,
        "false_pass": is_false_pass,
    }
    payload = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, payload.encode("utf-8"))
    finally:
        os.close(fd)
    return record


def read_errata(
    path: Optional[Path] = None,
    *,
    limit: Optional[int] = None,
    route: Optional[str] = None,
) -> list[dict]:
    """Read errata entries newest-last, optionally filtered by route and capped by limit."""
    target = resolve_errata_path(path)
    try:
        raw = target.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return []

    wanted_route = str(route or "").strip().lstrip("@")
    out: list[dict] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if wanted_route and str(obj.get("route") or "").lstrip("@") != wanted_route:
            continue
        out.append(obj)

    if limit is not None and limit >= 0 and len(out) > limit:
        return out[-limit:] if limit else []
    return out


def summarize_errata(entries: list[dict]) -> dict:
    """Aggregate counts by kind, severity, false_pass, and route."""
    by_kind: dict[str, int] = {}
    by_route: dict[str, dict[str, int]] = {}
    by_severity: dict[str, int] = {}
    false_passes = 0

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        k = str(entry.get("kind") or "complaint")
        sev = str(entry.get("severity") or "medium")
        rt = str(entry.get("route") or "unknown").strip() or "unknown"
        is_fp = bool(entry.get("false_pass"))
        by_kind[k] = by_kind.get(k, 0) + 1
        by_severity[sev] = by_severity.get(sev, 0) + 1
        if is_fp:
            false_passes += 1
        bucket = by_route.setdefault(rt, {"total": 0, "false_passes": 0})
        bucket["total"] += 1
        if is_fp:
            bucket["false_passes"] += 1

    return {
        "total": sum(by_kind.values()),
        "false_passes": false_passes,
        "by_kind": by_kind,
        "by_severity": by_severity,
        "by_route": by_route,
    }


def render_errata_report(
    entries: list[dict],
    *,
    color: bool = True,
    width: Optional[int] = None,
) -> str:
    """Render a width-safe wireframe panel summarizing recorded errata."""
    w = max(64, min(116, int(width or 92)))
    summary = summarize_errata(entries)
    if not entries:
        return render_panel(
            ["(no errata recorded yet — run 'nexus errata add \"...\"' or '/erratum ...')"],
            title="NEXUS ERRATA LEDGER",
            footer="0 total · 0 false-pass",
            width=w,
        )

    inner = w - 4
    rows: list[str] = []
    for entry in entries[-12:]:
        eid = str(entry.get("id") or "err_?")
        kind = str(entry.get("kind") or "complaint")
        sev = str(entry.get("severity") or "medium").upper()
        rt = str(entry.get("route") or "unknown")
        tr = str(entry.get("trace_id") or "-")
        fp_tag = " FALSE-PASS" if entry.get("false_pass") else ""
        msg = " ".join(str(entry.get("message") or "").split())
        header = status_line(
            "fail" if entry.get("false_pass") else "warn",
            f"{eid} [{sev}] {kind}{fp_tag} · @{rt} · {tr}",
            color=color,
            prefix="",
        )
        rows.append(truncate_ansi(header, inner, "…"))
        rows.append(truncate_ansi(f"  {msg}", inner, "…"))

    footer = (
        f"{summary['total']} total · {summary['false_passes']} false-pass"
        f" · routes={len(summary['by_route'])}"
    )
    return render_panel(rows, title="NEXUS ERRATA LEDGER", footer=footer, width=w)
