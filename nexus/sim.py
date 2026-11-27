"""Self-Improvement Memory (SIM) ledger for Nexus.

The ledger preserves the full provenance of an optimisation campaign: every
accepted *and* rejected attempt, the tasks an edit was predicted to fix, the
tasks it actually fixed or missed, regressions, alias merges and schema breaks.
Compaction distills dead ends and per-model notes instead of deleting them, so a
future proposer never recycles a patch that was already disproven.

The JSON document is written atomically (temp file + ``os.replace``) and every
reader tolerates a missing or corrupt file unless ``strict=True`` is requested.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Optional

from nexus.visualizer import (
    BOLD,
    DIM,
    GREEN,
    RED,
    RESET,
    YELLOW,
    render_panel,
)

SCHEMA_VERSION = "1.0"
DEFAULT_SIM_PATH = Path.home() / ".local" / "state" / "nexus" / "sim.json"

ACTION_VALUES = ("new", "retain", "revise", "replace", "remove")
TIER_VALUES = ("invariant", "adaptable", "model_specific")

_PROBLEM_ID_RE = re.compile(r"^P(\d+)$")


def resolve_sim_path(path: Optional[Path] = None) -> Path:
    """Return the ledger path: explicit argument, ``NEXUS_SIM_PATH``, then default."""
    if path is not None:
        return Path(path).expanduser()
    env = os.environ.get("NEXUS_SIM_PATH", "")
    if env:
        return Path(env).expanduser()
    return DEFAULT_SIM_PATH


def new_ledger(
    *,
    domain: str = "nexus",
    policy_model: str = "",
    benchmark: str = "",
    interface_name: str = "nexus-harness",
    interface_version: str = "1.0",
) -> dict:
    """Return a fresh canonical SIM ledger."""
    return {
        "schema_version": SCHEMA_VERSION,
        "interface": {"name": interface_name, "version": interface_version},
        "carried_from": None,
        "schema_breaks": [],
        "context": {
            "domain": domain,
            "policy_model": policy_model,
            "benchmark": benchmark,
            "created_at": time.time(),
        },
        "aliases": {},
        "problems": [],
    }


def load_sim(path: Optional[Path] = None, *, strict: bool = False) -> dict:
    """Load a ledger, defaulting missing optional keys.

    A missing file yields a new ledger unless ``strict`` is set, in which case it
    raises ``FileNotFoundError``. Corrupt JSON or a non-dict root raises
    ``ValueError`` when ``strict`` is set and otherwise yields a new ledger.
    """
    target = resolve_sim_path(path)
    try:
        raw = target.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        if strict:
            raise
        return new_ledger()
    except OSError:
        if strict:
            raise
        return new_ledger()

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        if strict:
            raise ValueError(f"corrupt SIM ledger: {target}")
        return new_ledger()

    if not isinstance(data, dict):
        if strict:
            raise ValueError(f"SIM ledger root must be an object: {target}")
        return new_ledger()

    defaults = new_ledger()
    for key, value in defaults.items():
        if key not in data:
            data[key] = value

    if not isinstance(data.get("interface"), dict):
        data["interface"] = {"name": "nexus-harness", "version": SCHEMA_VERSION}
    if not isinstance(data.get("context"), dict):
        data["context"] = defaults["context"]
    if not isinstance(data.get("aliases"), dict):
        data["aliases"] = {}
    if not isinstance(data.get("problems"), list):
        data["problems"] = []
    if not isinstance(data.get("schema_breaks"), list):
        data["schema_breaks"] = []
    if not isinstance(data.get("schema_version"), str):
        data["schema_version"] = SCHEMA_VERSION
    return data


def save_sim(ledger: dict, path: Optional[Path] = None) -> Path:
    """Atomically write a pretty JSON ledger and return the resolved path."""
    target = resolve_sim_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(ledger, indent=2, ensure_ascii=False) + "\n"

    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
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
    return target


def resolve_problem_id(ledger: dict, problem_id: str) -> str:
    """Follow alias links transitively and return the canonical problem id."""
    aliases = ledger.get("aliases") or {}
    if not isinstance(aliases, dict):
        aliases = {}
    current = str(problem_id)
    seen = {current}
    while current in aliases:
        nxt = str(aliases[current])
        if nxt in seen:
            raise ValueError(f"alias cycle detected involving '{problem_id}'")
        seen.add(nxt)
        current = nxt
    return current


def find_problem(ledger: dict, problem_id: str) -> Optional[dict]:
    """Return the problem dict for an id after resolving aliases."""
    canonical = resolve_problem_id(ledger, problem_id)
    for problem in ledger.get("problems") or []:
        if isinstance(problem, dict) and problem.get("id") == canonical:
            return problem
    return None


def next_problem_id(ledger: dict) -> str:
    """Return the next free ``P<N>`` id, never reusing merged or aliased ids."""
    highest = 0
    problems = ledger.get("problems") or []
    for problem in problems:
        if not isinstance(problem, dict):
            continue
        match = _PROBLEM_ID_RE.match(str(problem.get("id") or ""))
        if match:
            highest = max(highest, int(match.group(1)))
    aliases = ledger.get("aliases") or {}
    for source, target in aliases.items():
        for candidate in (source, target):
            match = _PROBLEM_ID_RE.match(str(candidate or ""))
            if match:
                highest = max(highest, int(match.group(1)))
    return f"P{highest + 1}"


def open_problem(
    ledger: dict,
    name: str,
    *,
    opened_round: int = 1,
    tier: str = "adaptable",
    problem_id: Optional[str] = None,
) -> dict:
    """Append a new problem to the ledger and return it."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("problem name must be non-empty")
    if tier not in TIER_VALUES:
        raise ValueError(f"invalid tier '{tier}'; expected one of {TIER_VALUES}")

    pid = str(problem_id) if problem_id else next_problem_id(ledger)
    aliases = ledger.get("aliases") or {}
    if pid in aliases or find_problem(ledger, pid) is not None:
        raise ValueError(f"problem '{pid}' already exists")

    problem = {
        "id": pid,
        "name": name.strip(),
        "opened_round": int(opened_round),
        "tier": tier,
        "invariant_summary": "",
        "model_notes": {},
        "dead_ends": [],
        "attempts": [],
    }
    ledger.setdefault("problems", []).append(problem)
    return problem


def merge_problems(ledger: dict, source_id: str, target_id: str) -> dict:
    """Merge ``source_id`` into ``target_id`` and alias the source to the target."""
    source_resolved = resolve_problem_id(ledger, source_id)
    target_resolved = resolve_problem_id(ledger, target_id)
    if source_resolved == target_resolved:
        raise ValueError("cannot merge a problem into itself")

    source = find_problem(ledger, source_resolved)
    target = find_problem(ledger, target_resolved)
    if source is None:
        raise ValueError(f"unknown source problem '{source_id}'")
    if target is None:
        raise ValueError(f"unknown target problem '{target_id}'")

    combined = list(target.get("attempts") or []) + list(source.get("attempts") or [])
    combined.sort(key=lambda a: (int(a.get("round", 0) or 0), str(a.get("sample", ""))))
    target["attempts"] = combined

    source_notes = source.get("model_notes") or {}
    target_notes = target.setdefault("model_notes", {})
    if isinstance(source_notes, dict):
        for model, notes in source_notes.items():
            existing = target_notes.get(model)
            if isinstance(existing, list) and isinstance(notes, list):
                target_notes[model] = existing + [n for n in notes if n not in existing]
            else:
                target_notes.setdefault(model, notes)

    target_dead = target.setdefault("dead_ends", [])
    for entry in source.get("dead_ends") or []:
        if entry not in target_dead:
            target_dead.append(entry)

    ledger["problems"] = [
        problem
        for problem in ledger.get("problems") or []
        if not (isinstance(problem, dict) and problem.get("id") == source_resolved)
    ]
    ledger.setdefault("aliases", {})[source_resolved] = target_resolved
    return target


def compute_task_outcomes(
    before: dict[str, bool],
    after: dict[str, bool],
    predicted: list[str],
) -> dict:
    """Compute fixed, missed and regressed tasks plus the aggregate score.

    ``fixed`` lists predicted tasks that flipped from fail to pass first, then
    any other newly passing tasks in sorted order, so an edit that fixes an
    unpredicted task is still credited.
    """
    fixed: list[str] = []
    for tid in predicted:
        if after.get(tid, False) and not before.get(tid, False) and tid not in fixed:
            fixed.append(tid)
    extra = sorted(
        tid
        for tid, ok in after.items()
        if ok and not before.get(tid, False) and tid not in fixed
    )
    fixed.extend(extra)

    missed = [tid for tid in predicted if not after.get(tid, False)]
    regressed = [
        tid
        for tid, was_ok in before.items()
        if was_ok and not after.get(tid, False)
    ]
    passed = sum(1 for value in after.values() if value)
    total = len(after)
    score = (passed / total) if total else 0.0
    return {
        "fixed": fixed,
        "missed": missed,
        "regressed": regressed,
        "passed": passed,
        "total": total,
        "score": score,
    }


def record_attempt(
    ledger: dict,
    *,
    problem_id: str,
    round_num: int,
    sample: str,
    action: str,
    accepted: bool,
    domain: str,
    what: str,
    files: list[str],
    predicted: list[str],
    fixed: list[str],
    missed: list[str],
    passed: int,
    total: int,
    regressed: Optional[list[str]] = None,
    backfilled: bool = False,
    policy_model: str | list[str] = "",
    optimizer_model: str = "",
    benchmark: str = "",
    score: Optional[float] = None,
    diff_ref: str = "",
    problem_name: Optional[str] = None,
    tier: str = "adaptable",
) -> dict:
    """Append one attempt (accepted or rejected) and return it.

    Attempts are append-only: earlier attempts are never overwritten or removed.
    A missing problem is auto-opened only when ``problem_name`` is supplied.
    """
    if action not in ACTION_VALUES:
        raise ValueError(f"invalid action '{action}'; expected one of {ACTION_VALUES}")
    if not isinstance(what, str) or not what.strip():
        raise ValueError("attempt 'what' must be non-empty")
    if not isinstance(predicted, list):
        raise ValueError("attempt 'predicted' must be a list")

    problem = find_problem(ledger, problem_id)
    if problem is None:
        if problem_name is None:
            raise KeyError(f"unknown problem '{problem_id}'")
        problem = open_problem(
            ledger,
            problem_name,
            opened_round=round_num,
            tier=tier,
            problem_id=problem_id,
        )

    total_int = int(total)
    passed_int = int(passed)
    if score is not None:
        score_value = float(score)
    else:
        score_value = (passed_int / total_int) if total_int > 0 else 0.0

    attempt = {
        "round": int(round_num),
        "sample": str(sample),
        "action": action,
        "accepted": bool(accepted),
        "backfilled": bool(backfilled),
        "domain": str(domain),
        "what": what.strip(),
        "files": [str(f) for f in files],
        "predicted": [str(p) for p in predicted],
        "fixed": [str(x) for x in fixed],
        "missed": [str(x) for x in missed],
        "regressed": [str(x) for x in (regressed or [])],
        "passed": passed_int,
        "total": total_int,
        "score": score_value,
        "policy_model": (
            list(policy_model)
            if isinstance(policy_model, (list, tuple))
            else str(policy_model)
        ),
        "optimizer_model": str(optimizer_model),
        "benchmark": str(benchmark),
        "diff_ref": str(diff_ref),
        "ts": time.time(),
    }
    problem.setdefault("attempts", []).append(attempt)
    return attempt


def carry_forward(source_ledger: dict, *, from_run: str, into_run: str) -> dict:
    """Deep-copy a ledger into a new run and record the carry-forward chain."""
    carried = copy.deepcopy(source_ledger)
    carried["carried_from"] = {
        "from": str(from_run),
        "into": str(into_run),
        "parent": source_ledger.get("carried_from"),
        "carried_at": time.time(),
    }
    return carried


def record_schema_break(ledger: dict, *, new_version: str, reason: str) -> dict:
    """Record an interface schema break and advance the interface version."""
    interface = ledger.setdefault("interface", {})
    if not isinstance(interface, dict):
        interface = {}
        ledger["interface"] = interface
    record = {
        "ts": time.time(),
        "from_version": interface.get("version"),
        "to_version": str(new_version),
        "reason": str(reason),
    }
    ledger.setdefault("schema_breaks", []).append(record)
    interface["version"] = str(new_version)
    return record


def _policy_model_key(value: Any) -> str:
    """Normalise a policy model (string or list) into one label."""
    if isinstance(value, (list, tuple)):
        label = ", ".join(str(item) for item in value)
    else:
        label = str(value or "")
    return label.strip() or "unknown"


def _dead_end_entry(attempt: dict) -> Optional[str]:
    """Return a tagged dead-end line for an attempt, or ``None`` if it worked."""
    what = str(attempt.get("what") or "").strip()
    round_num = int(attempt.get("round", 0) or 0)
    if not attempt.get("accepted", False):
        tag = "rejected"
    elif attempt.get("regressed"):
        tag = "regressed"
    elif not attempt.get("fixed"):
        tag = "zero-fix"
    else:
        return None
    return f"[{tag}] r{round_num} {what}".rstrip()


def _distill_problem(problem: dict, keep_recent_attempts: int) -> None:
    """Rewrite one problem in place with distilled fields and trimmed attempts."""
    name = str(problem.get("name") or "").strip()
    attempts = [a for a in (problem.get("attempts") or []) if isinstance(a, dict)]

    # Invariant summary: keep older distilled segments, then add new accepted
    # fixes, so compaction never drops previously learned invariants.
    segments: list[str] = []
    existing_summary = str(problem.get("invariant_summary") or "").strip()
    if existing_summary:
        prefix = f"{name}: " if name else ""
        if prefix and existing_summary.startswith(prefix):
            body = existing_summary[len(prefix):]
        else:
            body = existing_summary
        if body and body != name:
            segments.extend(part.strip() for part in body.split(";") if part.strip())
    for attempt in attempts:
        if not attempt.get("accepted"):
            continue
        if attempt.get("action") == "remove":
            continue
        if not attempt.get("fixed"):
            continue
        note = str(attempt.get("what") or "").strip()
        if note and note not in segments:
            segments.append(note)
    segments = segments[:8]
    problem["invariant_summary"] = (
        f"{name}: " + "; ".join(segments) if segments else name
    )

    # Per-model notes: merge distilled history with newly accepted attempts.
    notes: dict[str, list[str]] = {}
    existing_notes = problem.get("model_notes") or {}
    if isinstance(existing_notes, dict):
        for model, entries in existing_notes.items():
            if isinstance(entries, list):
                notes[str(model)] = [str(entry) for entry in entries]
    for attempt in attempts:
        if not attempt.get("accepted") or attempt.get("action") == "remove":
            continue
        what = str(attempt.get("what") or "").strip()
        if not what:
            continue
        model = _policy_model_key(attempt.get("policy_model"))
        note = f"r{int(attempt.get('round', 0) or 0)} {attempt.get('action')}: {what}"
        bucket = notes.setdefault(model, [])
        if note not in bucket:
            bucket.append(note)
    for model, entries in notes.items():
        notes[model] = entries[:12]
    problem["model_notes"] = notes

    # Dead ends: keep prior distilled entries and add every disproven attempt.
    dead_ends: list[str] = []
    for entry in problem.get("dead_ends") or []:
        if isinstance(entry, str) and entry not in dead_ends:
            dead_ends.append(entry)
    for attempt in attempts:
        entry = _dead_end_entry(attempt)
        if entry and entry not in dead_ends:
            dead_ends.append(entry)
    problem["dead_ends"] = dead_ends[:32]

    keep = max(0, int(keep_recent_attempts))
    retained = attempts[-keep:] if keep else []
    trimmed = len(attempts) - len(retained)
    if trimmed > 0:
        problem["archived_attempts_count"] = (
            int(problem.get("archived_attempts_count", 0) or 0) + trimmed
        )
    problem["attempts"] = retained


def compact_sim(
    ledger: dict,
    *,
    archive_dir: Optional[Path] = None,
    keep_recent_attempts: int = 3,
) -> tuple[dict, Optional[Path]]:
    """Return a losslessly compacted ledger and an optional archive path.

    The archive holds the uncompacted ledger. The compacted copy distills
    invariants, per-model notes and dead ends before trimming raw attempts to
    the newest ``keep_recent_attempts`` entries per problem.
    """
    compacted = copy.deepcopy(ledger)

    archive_path: Optional[Path] = None
    if archive_dir is not None:
        directory = Path(archive_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        archive_path = directory / f"sim_archive_{stamp}.json"
        save_sim(ledger, archive_path)

    for problem in compacted.get("problems") or []:
        if isinstance(problem, dict):
            _distill_problem(problem, keep_recent_attempts)

    compacted["compacted_at"] = time.time()
    return compacted, archive_path


def validate_sim(ledger: dict) -> tuple[bool, list[str]]:
    """Return ``(ok, errors)`` describing schema integrity problems."""
    errors: list[str] = []
    if not isinstance(ledger, dict):
        return False, ["ledger root must be an object"]

    problems = ledger.get("problems")
    if problems is None:
        problems = []
    elif not isinstance(problems, list):
        errors.append("'problems' must be a list")
        problems = []

    aliases = ledger.get("aliases")
    if aliases is None:
        aliases = {}
    elif not isinstance(aliases, dict):
        errors.append("'aliases' must be an object")
        aliases = {}

    problem_ids: set[str] = set()
    for index, problem in enumerate(problems):
        if not isinstance(problem, dict):
            errors.append(f"problem[{index}] must be an object")
            continue
        pid = problem.get("id")
        if not isinstance(pid, str) or not pid:
            errors.append(f"problem[{index}] missing 'id'")
        elif pid in problem_ids:
            errors.append(f"duplicate problem id '{pid}'")
        else:
            problem_ids.add(pid)

        name = problem.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"problem '{pid or index}' missing 'name'")
        opened_round = problem.get("opened_round")
        if not isinstance(opened_round, int) or isinstance(opened_round, bool):
            errors.append(f"problem '{pid or index}' has invalid 'opened_round'")
        if problem.get("tier") not in TIER_VALUES:
            errors.append(f"problem '{pid or index}' has invalid tier")
        attempts = problem.get("attempts")
        if not isinstance(attempts, list):
            errors.append(f"problem '{pid or index}' has non-list 'attempts'")
            continue

        for a_index, attempt in enumerate(attempts):
            label = f"problem '{pid or index}' attempt[{a_index}]"
            if not isinstance(attempt, dict):
                errors.append(f"{label} must be an object")
                continue
            if attempt.get("action") not in ACTION_VALUES:
                errors.append(f"{label} has invalid action")
            what = attempt.get("what")
            if not isinstance(what, str) or not what.strip():
                errors.append(f"{label} missing 'what'")
            for field in ("predicted", "fixed", "missed"):
                if not isinstance(attempt.get(field), list):
                    errors.append(f"{label} '{field}' must be a list")
            passed = attempt.get("passed")
            total = attempt.get("total")
            if isinstance(passed, int) and isinstance(total, int):
                if total > 0 and passed > total:
                    errors.append(f"{label} passed ({passed}) exceeds total ({total})")
            if not attempt.get("policy_model") and not attempt.get("domain"):
                errors.append(f"{label} missing 'policy_model' and 'domain'")

    for source, target in aliases.items():
        try:
            resolved = resolve_problem_id(ledger, str(source))
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if resolved not in problem_ids:
            errors.append(f"alias '{source}' points to missing problem '{resolved}'")

    return (not errors, errors)


def format_sim_for_prompt(ledger: dict) -> str:
    """Return the complete ledger JSON for proposer and selector prompts."""
    return json.dumps(ledger, indent=2, ensure_ascii=False)


def _carried_chain(entry: Any) -> list[str]:
    """Return human-readable labels for a nested carry-forward chain."""
    labels: list[str] = []
    node = entry
    seen: set[int] = set()
    while isinstance(node, dict) and id(node) not in seen:
        seen.add(id(node))
        source = node.get("from")
        target = node.get("into")
        if source or target:
            labels.append(f"{source or '?'}→{target or '?'}")
        node = node.get("parent")
    return labels


def _score_text(value: Any) -> str:
    """Format a score for the summary panel, tolerating missing values."""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def render_sim_summary(ledger: dict, *, color: bool = True, width: int = 96) -> str:
    """Render a width-safe wireframe summary of the SIM ledger."""
    width = max(60, int(width))
    problems = [p for p in (ledger.get("problems") or []) if isinstance(p, dict)]
    interface = ledger.get("interface") if isinstance(ledger.get("interface"), dict) else {}
    context = ledger.get("context") if isinstance(ledger.get("context"), dict) else {}
    aliases = ledger.get("aliases") if isinstance(ledger.get("aliases"), dict) else {}

    def _paint(code: str, text: str) -> str:
        return f"{code}{text}{RESET}" if color else text

    lines: list[str] = []
    schema = ledger.get("schema_version", SCHEMA_VERSION)
    interface_label = f"{interface.get('name', 'nexus-harness')}:{interface.get('version', '?')}"
    lines.append(f"{_paint(DIM, 'schema')}  {schema}  ·  interface {interface_label}")

    domain = context.get("domain", "") or "-"
    benchmark = context.get("benchmark", "") or "-"
    policy = _policy_model_key(context.get("policy_model"))
    lines.append(f"{_paint(DIM, 'domain')}  {domain}  ·  benchmark {benchmark}  ·  policy {policy}")

    chain = _carried_chain(ledger.get("carried_from"))
    lines.append(
        f"{_paint(DIM, 'carried')} {' · '.join(chain) if chain else 'none'}"
    )
    alias_text = ", ".join(f"{src}→{dst}" for src, dst in aliases.items()) or "none"
    lines.append(f"{_paint(DIM, 'aliases')} {alias_text}")
    lines.append("")

    if not problems:
        lines.append("(no problems recorded)")
    total_attempts = 0
    for problem in problems:
        pid = problem.get("id", "?")
        tier = problem.get("tier", "?")
        opened = problem.get("opened_round", "?")
        name = str(problem.get("name") or "").strip()
        attempts = [a for a in (problem.get("attempts") or []) if isinstance(a, dict)]
        total_attempts += len(attempts)
        accepted = sum(1 for a in attempts if a.get("accepted"))
        rejected = sum(1 for a in attempts if not a.get("accepted"))
        zero_fix = sum(1 for a in attempts if a.get("accepted") and not a.get("fixed"))
        latest = attempts[-1].get("score") if attempts else None
        dead = len(problem.get("dead_ends") or [])

        header = (
            f"{_paint(BOLD, str(pid))} [{tier}] r{opened} {name}"
        )
        lines.append(header)
        stats = (
            f"   {_paint(GREEN, f'ok {accepted}')} · "
            f"{_paint(RED, f'rej {rejected}')} · "
            f"{_paint(YELLOW, f'zero {zero_fix}')} · "
            f"score {_score_text(latest)} · dead {dead}"
        )
        lines.append(stats)

    footer = f"schema {schema} · {len(problems)} problems · {total_attempts} attempts"
    border = DIM if color else ""
    return render_panel(
        lines,
        title="SIM LEDGER",
        footer=footer,
        width=width,
        border_color=border,
    )
