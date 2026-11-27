"""Assay verification receipts, staged checks, shadow calibration, and ISA parsing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

from nexus.visualizer import render_panel, truncate_ansi

ASSAY_SCHEMA = "nexus-assay/v1"
DEFAULT_SHADOW_PATH = Path.home() / ".local/state/nexus/shadow_decisions.jsonl"

_PROBE_RE = re.compile(r"^\s*-\s+\*\*Probe:\*\*\s*(.+)$")
_TASK_RE = re.compile(r"^\s*-\s+\[([ xX])\]\s+(.+)$")
_IDEAL_HEADER_RE = re.compile(r"^\s*##\s+Ideal State\b", re.IGNORECASE)


def resolve_shadow_path(path: Optional[Path] = None) -> Path:
    """Resolve shadow-decision log path, honoring `NEXUS_SHADOW_PATH`."""
    if path is not None:
        return Path(path).expanduser()
    env_path = os.environ.get("NEXUS_SHADOW_PATH", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return DEFAULT_SHADOW_PATH


def _subject_bytes(subject: bytes | str) -> bytes:
    if isinstance(subject, (bytes, bytearray)):
        return bytes(subject)
    return str(subject).encode("utf-8", errors="replace")


def build_assay_receipt(
    subject: bytes | str,
    *,
    ideal_state: str,
    stages: list[dict],
    evaluator_version: str = "1.0",
    target: str = "",
) -> dict:
    """Construct a byte-bound receipt; unmeasured stage sets never claim PASS."""
    raw = _subject_bytes(subject)
    digest = hashlib.sha256(raw).hexdigest()
    normalized_stages: list[dict] = []
    passed = 0
    total = 0
    had_error = False

    for st in stages:
        if not isinstance(st, dict):
            continue
        name = str(st.get("name") or f"stage-{len(normalized_stages) + 1}")
        ok = bool(st.get("passed"))
        skipped = bool(st.get("skipped"))
        detail = str(st.get("detail") or "")
        err = str(st.get("error") or "")
        if err:
            had_error = True
        total += 1
        if ok and not skipped and not err:
            passed += 1
        normalized_stages.append(
            {
                "name": name,
                "passed": ok and not skipped and not err,
                "skipped": skipped,
                "detail": detail,
                "error": err,
            }
        )

    if total == 0:
        status = "UNMEASURED"
    elif passed == total and not had_error:
        status = "PASS"
    else:
        status = "FAIL"

    return {
        "schema": ASSAY_SCHEMA,
        "ts": time.time(),
        "target": str(target or ""),
        "ideal_state": str(ideal_state or "").strip(),
        "evaluator_version": str(evaluator_version or "1.0"),
        "subject_sha256": digest,
        "subject_bytes": len(raw),
        "status": status,
        "passed": passed,
        "total": total,
        "stages": normalized_stages,
    }


def run_staged_assay(
    subject: bytes | str,
    *,
    ideal_state: str,
    stages: list[tuple[str, Callable[[bytes], tuple[bool, str]]]],
    target: str = "",
    evaluator_version: str = "1.0",
) -> dict:
    """Run staged checks in order, skipping later stages once an earlier stage fails."""
    raw = _subject_bytes(subject)
    results: list[dict] = []
    blocked = False
    for name, fn in stages:
        if blocked:
            results.append(
                {"name": name, "passed": False, "skipped": True, "detail": "skipped due to earlier stage failure", "error": ""}
            )
            continue
        try:
            ok, detail = fn(raw)
        except Exception as exc:
            results.append(
                {"name": name, "passed": False, "skipped": False, "detail": "", "error": f"{type(exc).__name__}: {exc}"}
            )
            blocked = True
            continue
        passed = bool(ok)
        results.append(
            {"name": name, "passed": passed, "skipped": False, "detail": str(detail or ""), "error": ""}
        )
        if not passed:
            blocked = True

    return build_assay_receipt(
        raw,
        ideal_state=ideal_state,
        stages=results,
        evaluator_version=evaluator_version,
        target=target,
    )


def record_shadow_decision(
    *,
    prompt: str,
    keyword_route: str,
    llm_route: str,
    agreed: bool,
    elapsed_ms: int,
    decision_model: str,
    path: Optional[Path] = None,
) -> dict:
    """Append one shadow routing comparison record atomically."""
    target = resolve_shadow_path(path)
    record = {
        "ts": time.time(),
        "prompt_sha256": hashlib.sha256(str(prompt or "").encode("utf-8", errors="replace")).hexdigest()[:16],
        "prompt_preview": " ".join(str(prompt or "").split())[:120],
        "keyword_route": str(keyword_route or ""),
        "llm_route": str(llm_route or ""),
        "agreed": bool(agreed),
        "elapsed_ms": max(0, int(elapsed_ms)),
        "decision_model": str(decision_model or ""),
    }
    payload = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, payload.encode("utf-8"))
    finally:
        os.close(fd)
    return record


def _pct(sorted_vals: list[int], frac: float) -> int:
    if not sorted_vals:
        return 0
    if len(sorted_vals) == 1:
        return int(sorted_vals[0])
    idx = max(0, min(len(sorted_vals) - 1, math.ceil(frac * len(sorted_vals)) - 1))
    return int(sorted_vals[idx])


def summarize_shadow_log(path: Optional[Path] = None, *, limit: int = 500) -> dict:
    """Summarize shadow routing calibration and report authority promotion readiness."""
    target = resolve_shadow_path(path)
    try:
        raw = target.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return {
            "total": 0,
            "agreed": 0,
            "agreement_rate": 0.0,
            "p50_ms": 0,
            "p95_ms": 0,
            "ready_to_promote": False,
        }

    records: list[dict] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    if limit > 0 and len(records) > limit:
        records = records[-limit:]

    total = len(records)
    if total == 0:
        return {
            "total": 0,
            "agreed": 0,
            "agreement_rate": 0.0,
            "p50_ms": 0,
            "p95_ms": 0,
            "ready_to_promote": False,
        }
    agreed = sum(1 for r in records if r.get("agreed"))
    rate = agreed / total
    latencies = sorted(max(0, int(r.get("elapsed_ms") or 0)) for r in records)
    p50_ms = _pct(latencies, 0.50)
    p95_ms = _pct(latencies, 0.95)
    return {
        "total": total,
        "agreed": agreed,
        "agreement_rate": round(rate, 4),
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "ready_to_promote": bool(total >= 20 and rate >= 0.85 and p95_ms <= 5000),
    }


def parse_isa_markdown(path: Path) -> dict:
    """Parse an ISA markdown document for Ideal State and task/probe coverage."""
    target = Path(path)
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return {
            "path": str(target),
            "ideal_state": "",
            "checked": 0,
            "open": 0,
            "measured": 0,
            "unmeasured": 0,
            "tasks": [],
        }

    ideal_state = ""
    in_ideal = False
    tasks: list[dict[str, Any]] = []

    for line in lines:
        if _IDEAL_HEADER_RE.match(line):
            in_ideal = True
            continue
        if in_ideal:
            stripped = line.strip()
            if stripped.startswith("##"):
                in_ideal = False
            elif stripped and not ideal_state:
                ideal_state = stripped.lstrip("> ").strip()

        m_task = _TASK_RE.match(line)
        if m_task:
            tasks.append(
                {
                    "checked": m_task.group(1).lower() == "x",
                    "title": m_task.group(2).strip(),
                    "probe": "",
                }
            )
            continue

        m_probe = _PROBE_RE.match(line)
        if m_probe and tasks and not tasks[-1]["probe"]:
            tasks[-1]["probe"] = m_probe.group(1).strip()

    checked = sum(1 for t in tasks if t["checked"])
    open_count = len(tasks) - checked
    measured = sum(1 for t in tasks if bool(t["probe"]))
    unmeasured = len(tasks) - measured
    return {
        "path": str(target),
        "ideal_state": ideal_state,
        "checked": checked,
        "open": open_count,
        "measured": measured,
        "unmeasured": unmeasured,
        "tasks": tasks,
    }


def render_isa_summary(isa: dict, *, width: Optional[int] = None) -> str:
    """Render an ISA checklist & probe summary panel."""
    w = max(64, min(116, int(width or 92)))
    inner = w - 4
    rows = [
        truncate_ansi(f"ideal: {isa.get('ideal_state') or '(missing Ideal State section)'}", inner, "…"),
        truncate_ansi(
            f"tasks: {isa.get('checked', 0)} done · {isa.get('open', 0)} open"
            f" · probes: {isa.get('measured', 0)} measured · {isa.get('unmeasured', 0)} unmeasured",
            inner,
            "…",
        ),
    ]
    for task in (isa.get("tasks") or []):
        if not task.get("checked"):
            mark = "[ ]"
            probe = f"probe={task['probe']}" if task.get("probe") else "UNMEASURED"
            rows.append(truncate_ansi(f"  {mark} {task['title']} ({probe})", inner, "…"))
    return render_panel(
        rows,
        title="NEXUS ISA & PROBE COVERAGE",
        footer=Path(str(isa.get("path") or "tasks/todo.md")).name,
        width=w,
    )
