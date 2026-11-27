"""Unified entity abstraction, pluggable profiles, handoff context, and performance lab."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Optional

from nexus.events import _FIRST_WINS, _LAST_WINS, normalize_quota_tier
from nexus.router import load_routes
from nexus.visualizer import (
    BOLD,
    CYAN,
    DIM,
    GREEN,
    RESET,
    YELLOW,
    pad_ansi,
    render_panel,
    truncate_ansi,
    visible_len,
)

BUILTIN_PROFILES_DIR = Path(__file__).resolve().parent.parent / "config" / "profiles"
USER_PROFILES_DIR = Path.home() / ".config" / "nexus" / "profiles"
DEFAULT_PROFILE_NAME = "default"
VALID_MODES = ("entity", "single", "ensemble")

DEFAULT_SYSTEM_PERSONA = (
    "You are Nexus, a unified AI engineering assistant. Respond directly in one "
    "cohesive, authoritative voice. Never expose internal model routing, dispatch "
    "mechanics, or ask the user to manually switch models. Continue the conversation "
    "seamlessly using any shared context provided. Continuously look for concrete, "
    "measurable ways to improve accuracy, latency, and open-model efficiency without "
    "over-engineering."
)


@dataclass
class EntityProfile:
    """Declarative Nexus configuration profile controlling entity behavior and routes."""

    name: str
    description: str = ""
    mode: str = "entity"
    system_persona: str = DEFAULT_SYSTEM_PERSONA
    handoff_max_turns: int = 8
    handoff_max_chars: int = 6000
    default_route: Optional[str] = None
    decision_model: Optional[str] = None
    council_roles: Optional[dict[str, str]] = None
    replace_routes: bool = False
    routes: dict[str, dict] = field(default_factory=dict)
    precedence: Optional[list[str]] = None
    source_path: Optional[Path] = None


def _profile_from_dict(data: Any, *, fallback_name: str, source_path: Optional[Path] = None) -> EntityProfile:
    if not isinstance(data, dict):
        raise ValueError(f"profile at {source_path or fallback_name} must be a JSON object")
    mode = str(data.get("mode") or "entity").strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"invalid profile mode '{mode}'; expected one of {VALID_MODES}")

    name = str(data.get("name") or fallback_name).strip()
    if not name:
        raise ValueError("profile name must be non-empty")

    raw_persona = data.get("system_persona")
    system_persona = DEFAULT_SYSTEM_PERSONA if raw_persona is None else str(raw_persona)

    routes = data.get("routes")
    if routes is None:
        routes_dict: dict[str, dict] = {}
    elif isinstance(routes, dict):
        routes_dict = {str(k): dict(v) for k, v in routes.items() if isinstance(v, dict)}
    else:
        raise ValueError("profile 'routes' must be an object")

    council_raw = data.get("council_roles")
    council_roles = (
        {str(k): str(v) for k, v in council_raw.items()}
        if isinstance(council_raw, dict)
        else None
    )

    prec_raw = data.get("precedence")
    precedence = [str(item) for item in prec_raw] if isinstance(prec_raw, list) else None

    return EntityProfile(
        name=name,
        description=str(data.get("description") or ""),
        mode=mode,
        system_persona=system_persona,
        handoff_max_turns=max(0, int(data.get("handoff_max_turns", 8))),
        handoff_max_chars=max(0, int(data.get("handoff_max_chars", 6000))),
        default_route=(str(data["default_route"]) if data.get("default_route") else None),
        decision_model=(str(data["decision_model"]) if data.get("decision_model") else None),
        council_roles=council_roles,
        replace_routes=bool(data.get("replace_routes", False)),
        routes=routes_dict,
        precedence=precedence,
        source_path=source_path,
    )


def _read_profile_file(path: Path) -> EntityProfile:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FileNotFoundError(f"cannot read profile '{path}': {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in profile '{path}': {exc}") from exc
    return _profile_from_dict(data, fallback_name=path.stem, source_path=path)


def list_profiles(
    user_dir: Optional[Path] = None,
    builtin_dir: Optional[Path] = None,
) -> list[EntityProfile]:
    """Discover built-in and user profiles (`user_dir` overrides built-ins by name)."""
    b_dir = Path(builtin_dir).expanduser() if builtin_dir is not None else BUILTIN_PROFILES_DIR
    u_dir = Path(user_dir).expanduser() if user_dir is not None else USER_PROFILES_DIR
    by_name: dict[str, EntityProfile] = {}

    for directory in (b_dir, u_dir):
        if not directory.is_dir():
            continue
        for entry in sorted(directory.glob("*.json")):
            if not entry.is_file():
                continue
            try:
                prof = _read_profile_file(entry)
            except (OSError, ValueError):
                continue
            by_name[prof.name] = prof

    return [by_name[key] for key in sorted(by_name)]


def load_profile(
    name_or_path: Optional[str | Path] = None,
    *,
    user_dir: Optional[Path] = None,
    builtin_dir: Optional[Path] = None,
) -> EntityProfile:
    """Load an `EntityProfile` by explicit JSON file path, user profile name, or built-in name."""
    if name_or_path is None or (isinstance(name_or_path, str) and not name_or_path.strip()):
        target_name = DEFAULT_PROFILE_NAME
    else:
        candidate = Path(str(name_or_path)).expanduser()
        if candidate.is_file():
            return _read_profile_file(candidate)
        target_name = str(name_or_path).strip()
        if target_name.endswith(".json") or "/" in target_name or "\\" in target_name:
            raise FileNotFoundError(f"profile file not found: {name_or_path}")

    u_dir = Path(user_dir).expanduser() if user_dir is not None else USER_PROFILES_DIR
    b_dir = Path(builtin_dir).expanduser() if builtin_dir is not None else BUILTIN_PROFILES_DIR

    for directory in (u_dir, b_dir):
        file_path = directory / f"{target_name}.json"
        if file_path.is_file():
            return _read_profile_file(file_path)

    if target_name == DEFAULT_PROFILE_NAME:
        return EntityProfile(
            name=DEFAULT_PROFILE_NAME,
            description="Built-in default unified entity profile",
            mode="entity",
        )
    raise FileNotFoundError(f"unknown profile '{target_name}'")


def save_user_profile(
    profile: EntityProfile,
    *,
    user_dir: Optional[Path] = None,
) -> Path:
    """Atomically save an `EntityProfile` JSON file into the user profiles directory."""
    if profile.mode not in VALID_MODES:
        raise ValueError(f"invalid profile mode '{profile.mode}'; expected one of {VALID_MODES}")
    clean_name = profile.name.strip()
    if not clean_name or "/" in clean_name or "\\" in clean_name or clean_name in (".", ".."):
        raise ValueError(f"invalid profile name: {profile.name!r}")

    target_dir = Path(user_dir).expanduser() if user_dir is not None else USER_PROFILES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / f"{clean_name}.json"

    payload: dict[str, Any] = {
        "name": clean_name,
        "description": profile.description,
        "mode": profile.mode,
        "system_persona": profile.system_persona,
        "handoff_max_turns": int(profile.handoff_max_turns),
        "handoff_max_chars": int(profile.handoff_max_chars),
    }
    if profile.default_route:
        payload["default_route"] = profile.default_route
    if profile.decision_model:
        payload["decision_model"] = profile.decision_model
    if profile.council_roles is not None:
        payload["council_roles"] = dict(profile.council_roles)
    if profile.replace_routes:
        payload["replace_routes"] = True
    if profile.routes:
        payload["routes"] = copy.deepcopy(profile.routes)
    if profile.precedence is not None:
        payload["precedence"] = list(profile.precedence)

    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=str(target_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o644)
        os.replace(tmp_name, dest)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    profile.source_path = dest
    return dest


def resolve_profile_routes(
    profile: EntityProfile,
    base_routes_data: Optional[dict] = None,
) -> dict:
    """Resolve an `EntityProfile` into a complete route-table dict without mutating inputs."""
    resolved = copy.deepcopy(base_routes_data if base_routes_data is not None else load_routes())
    base_routes = resolved.get("routes")
    if not isinstance(base_routes, dict):
        base_routes = {}

    if profile.replace_routes:
        routes: dict[str, dict] = copy.deepcopy(profile.routes)
    else:
        routes = copy.deepcopy(base_routes)
        for key, patch in (profile.routes or {}).items():
            if not isinstance(patch, dict):
                continue
            if key in routes and isinstance(routes[key], dict):
                merged = dict(routes[key])
                merged.update(copy.deepcopy(patch))
                routes[key] = merged
            else:
                routes[key] = copy.deepcopy(patch)

    resolved["routes"] = routes
    if profile.default_route:
        resolved["default_route"] = profile.default_route
    elif profile.replace_routes and resolved.get("default_route") not in routes and routes:
        resolved["default_route"] = next(iter(routes))

    if profile.decision_model:
        resolved["decision_model"] = profile.decision_model

    if profile.council_roles is not None:
        if profile.replace_routes or not isinstance(resolved.get("council_roles"), dict):
            resolved["council_roles"] = dict(profile.council_roles)
        else:
            merged_council = dict(resolved["council_roles"])
            merged_council.update(profile.council_roles)
            resolved["council_roles"] = merged_council

    if profile.precedence is not None:
        resolved["precedence"] = list(profile.precedence)
    elif profile.replace_routes:
        resolved["precedence"] = list(routes.keys())

    return resolved


def _clip_turn_text(text: str, limit: int) -> str:
    clean = text.strip()
    if len(clean) <= limit:
        return clean
    if limit <= 1:
        return clean[: max(0, limit)]
    return clean[: limit - 1].rstrip() + "…"


def build_handoff_context(
    turns: list[dict],
    *,
    max_turns: int = 8,
    max_chars: int = 6000,
) -> str:
    """Format bounded cross-model shared conversation context from chronological turns."""
    if max_turns <= 0 or max_chars <= 0 or not turns:
        return ""

    filtered: list[tuple[str, str]] = []
    for item in turns:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()
        if not content or role not in ("user", "assistant"):
            continue
        if role == "assistant" and content.startswith("[error]"):
            continue
        label = "User" if role == "user" else "Nexus"
        filtered.append((label, content))

    if not filtered:
        return ""

    recent = filtered[-max_turns:]
    per_turn_cap = max(240, max_chars // max(1, min(len(recent), 4)))
    header = "[Nexus Shared Conversation Context]"
    budget = max(0, max_chars - len(header) - 1)
    selected: list[str] = []
    used = 0

    for label, content in reversed(recent):
        clipped = _clip_turn_text(content, per_turn_cap)
        line = f"{label}: {clipped}"
        needed = len(line) + (1 if selected else 0)
        if used + needed > budget:
            remaining = budget - used - (1 if selected else 0)
            if not selected and remaining > len(label) + 8:
                line = f"{label}: {_clip_turn_text(content, remaining - len(label) - 2)}"
                selected.append(line)
            break
        selected.append(line)
        used += needed

    if not selected:
        return ""
    selected.reverse()
    return header + "\n" + "\n".join(selected)


def compose_entity_prompt(
    prompt: str,
    *,
    system_persona: str = "",
    handoff_context: str = "",
    include_handoff: bool = True,
) -> str:
    """Prepend persona and/or handoff context only when present; else return `prompt` verbatim."""
    blocks: list[str] = []
    persona = (system_persona or "").strip()
    if persona:
        blocks.append(f"[Nexus System Persona]\n{persona}")
    handoff = (handoff_context or "").strip()
    if include_handoff and handoff:
        blocks.append(handoff)
    if not blocks:
        return prompt
    blocks.append(f"[Current Request]\n{prompt}")
    return "\n\n".join(blocks)


def build_synthesis_prompt(
    user_prompt: str,
    seat_outputs: list[tuple[str, str]],
    *,
    system_persona: str = "",
) -> str:
    """Build a single-entity synthesis prompt from parallel internal specialist drafts."""
    persona = (system_persona or "").strip() or DEFAULT_SYSTEM_PERSONA
    drafts: list[str] = []
    for idx, (role_name, text) in enumerate(seat_outputs, start=1):
        body = (text or "").strip()
        if not body:
            continue
        title = role_name.strip().upper().replace("_", "-") or f"DRAFT-{idx}"
        drafts.append(f"### Perspective {idx} ({title})\n{body}")

    draft_block = "\n\n".join(drafts) if drafts else "(no specialist drafts succeeded)"
    return (
        f"[Nexus System Persona]\n{persona}\n\n"
        "Synthesize the internal drafts below into ONE concise, authoritative response directly "
        "answering the user. Resolve disagreements, omit redundant boilerplate, and never mention "
        "internal drafts, seats, routes, or model names unless explicitly asked.\n\n"
        f"[User Request]\n{user_prompt}\n\n"
        f"[Internal Drafts]\n{draft_block}"
    )


def _percentile(sorted_vals: list[int], pct: float) -> int:
    if not sorted_vals:
        return 0
    if len(sorted_vals) == 1:
        return int(sorted_vals[0])
    rank = max(0, min(len(sorted_vals) - 1, math.ceil(pct * len(sorted_vals)) - 1))
    return int(sorted_vals[rank])


def _merge_spans(events: list[dict]) -> list[dict]:
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
        for field_name in _FIRST_WINS:
            if not merged.get(field_name) and event.get(field_name):
                merged[field_name] = event[field_name]
        for field_name in _LAST_WINS:
            if field_name in event and event[field_name] not in (None, ""):
                merged[field_name] = event[field_name]
    return [spans[sid] for sid in order]


def compute_performance_stats(events: list[dict], *, group_by: str = "profile") -> dict:
    """Aggregate execution telemetry across completed/failed spans by profile, route, or model."""
    if group_by not in ("profile", "route", "model"):
        raise ValueError(f"invalid group_by '{group_by}'; expected profile, route, or model")

    merged = _merge_spans(events)
    terminal_statuses = {"DONE", "FAIL", "TIMEOUT", "INTERRUPTED"}
    leaf_spans = [
        sp
        for sp in merged
        if normalize_quota_tier(str(sp.get("quota_tier") or "")) != "ORCHESTRATOR"
        and str(sp.get("status") or "").upper() in terminal_statuses
    ]

    buckets: dict[str, list[dict]] = {}
    for sp in leaf_spans:
        if group_by == "profile":
            key = str(sp.get("profile") or "").strip() or "(default)"
        elif group_by == "route":
            raw_route = str(sp.get("route") or "").strip() or "unknown"
            key = raw_route[len("in-charge:") :] if raw_route.startswith("in-charge:") else raw_route
        else:
            key = str(sp.get("model") or "").strip() or "unknown"
        buckets.setdefault(key, []).append(sp)

    groups: dict[str, dict[str, Any]] = {}
    ordered_keys = sorted(buckets.keys(), key=lambda k: (-len(buckets[k]), k))
    for key in ordered_keys:
        items = buckets[key]
        runs = len(items)
        done = sum(1 for sp in items if str(sp.get("status") or "").upper() == "DONE")
        fail = sum(1 for sp in items if str(sp.get("status") or "").upper() == "FAIL")
        timeout = sum(1 for sp in items if str(sp.get("status") or "").upper() == "TIMEOUT")
        interrupted = sum(1 for sp in items if str(sp.get("status") or "").upper() == "INTERRUPTED")
        latencies = sorted(max(0, int(sp.get("elapsed_ms") or 0)) for sp in items)
        tok_in = sum(max(0, int(sp.get("tokens_in") or 0)) for sp in items)
        tok_out = sum(max(0, int(sp.get("tokens_out") or 0)) for sp in items)
        premium = sum(
            1 for sp in items if normalize_quota_tier(str(sp.get("quota_tier") or "")) == "PREMIUM"
        )
        groups[key] = {
            "runs": runs,
            "done": done,
            "fail": fail,
            "timeout": timeout,
            "interrupted": interrupted,
            "success_rate": (done / runs) if runs else 0.0,
            "avg_ms": int(sum(latencies) / runs) if runs else 0,
            "p50_ms": _percentile(latencies, 0.50),
            "p95_ms": _percentile(latencies, 0.95),
            "tokens_in": tok_in,
            "tokens_out": tok_out,
            "premium_spans": premium,
            "zero_cost_rate": ((runs - premium) / runs) if runs else 1.0,
        }

    return {
        "group_by": group_by,
        "total_runs": len(leaf_spans),
        "groups": groups,
    }


def _fmt_ms(ms: int) -> str:
    if ms < 1000:
        return f"{ms}ms"
    return f"{ms / 1000:.1f}s"


def _fmt_tok(count: int) -> str:
    if count < 1000:
        return str(count)
    if count < 1_000_000:
        return f"{count / 1000:.1f}k"
    return f"{count / 1_000_000:.1f}M"


def render_stats_table(
    stats: dict,
    *,
    group_by: str = "profile",
    color: bool = True,
    width: Optional[int] = None,
) -> str:
    """Render a width-safe wireframe performance table from `compute_performance_stats`."""
    target_w = max(72, min(116, int(width or 96)))
    groups = stats.get("groups") or {}
    g_label = str(stats.get("group_by") or group_by).upper()

    if not groups:
        return render_panel(
            ["(no completed execution spans found in telemetry bus)"],
            title=f"NEXUS PERFORMANCE STATS · BY {g_label}",
            footer="0 runs",
            width=target_w,
            border_color=DIM if color else "",
        )

    name_w = max(14, min(28, target_w - 60))
    header = (
        f"{pad_ansi(g_label, name_w)} │ "
        f"{pad_ansi('RUNS', 4, 'right')} │ "
        f"{pad_ansi('OK%', 5, 'right')} │ "
        f"{pad_ansi('P50', 6, 'right')} │ "
        f"{pad_ansi('P95', 6, 'right')} │ "
        f"{pad_ansi('TOKENS', 13)} │ "
        f"{pad_ansi('$0%', 5, 'right')}"
    )
    rows = [f"{BOLD}{header}{RESET}" if color else header, "─" * (target_w - 4)]

    for key, info in groups.items():
        ok_pct = f"{info['success_rate'] * 100:.0f}%"
        z_pct = f"{info['zero_cost_rate'] * 100:.0f}%"
        tok = f"{_fmt_tok(info['tokens_in'])}→{_fmt_tok(info['tokens_out'])}"
        name = truncate_ansi(str(key), name_w, "…")
        row = (
            f"{pad_ansi(name, name_w)} │ "
            f"{pad_ansi(str(info['runs']), 4, 'right')} │ "
            f"{pad_ansi(ok_pct, 5, 'right')} │ "
            f"{pad_ansi(_fmt_ms(info['p50_ms']), 6, 'right')} │ "
            f"{pad_ansi(_fmt_ms(info['p95_ms']), 6, 'right')} │ "
            f"{pad_ansi(truncate_ansi(tok, 13, '…'), 13)} │ "
            f"{pad_ansi(z_pct, 5, 'right')}"
        )
        rows.append(row)

    total_runs = int(stats.get("total_runs") or 0)
    footer = f"{len(groups)} group(s) · {total_runs} span(s)"
    return render_panel(
        rows,
        title=f"NEXUS PERFORMANCE STATS · BY {g_label}",
        footer=footer,
        width=target_w,
        border_color=CYAN if color else "",
    )


def benchmark_profiles(
    workspace: Path,
    profile_names: list[str],
    *,
    benchmark_path: Optional[Path] = None,
    split: str = "evolve",
    routes_base: Optional[dict] = None,
    user_dir: Optional[Path] = None,
    builtin_dir: Optional[Path] = None,
) -> list[dict]:
    """Evaluate multiple configuration profiles head-to-head on a benchmark split."""
    from nexus.evolve import BenchmarkSpec, evaluate_split

    spec = BenchmarkSpec.load(benchmark_path)
    if routes_base is not None:
        base = copy.deepcopy(routes_base)
    else:
        routes_file = Path(workspace) / "config" / "routes.json"
        base = load_routes(routes_file) if routes_file.is_file() else load_routes()

    names = [n.strip() for n in profile_names if str(n).strip()]
    if not names:
        names = [p.name for p in list_profiles(user_dir=user_dir, builtin_dir=builtin_dir)]

    results: list[dict] = []
    for name in names:
        profile = load_profile(name, user_dir=user_dir, builtin_dir=builtin_dir)
        resolved = resolve_profile_routes(profile, base)
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory(prefix=f"nexus-bench-{profile.name}-") as tmp:
            ws = Path(tmp)
            cfg_dir = ws / "config"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            (cfg_dir / "routes.json").write_text(
                json.dumps(resolved, indent=2), encoding="utf-8"
            )
            report = evaluate_split(ws, spec, split=split)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        tiers = sorted(
            {
                normalize_quota_tier(str(cfg.get("quota_tier") or "FREE"))
                for cfg in (resolved.get("routes") or {}).values()
                if isinstance(cfg, dict)
            }
        )
        results.append(
            {
                "profile": profile.name,
                "mode": profile.mode,
                "split": report.split,
                "passed": report.passed,
                "total": report.total,
                "score": report.score,
                "elapsed_ms": elapsed_ms,
                "tiers": tiers,
                "route_count": len(resolved.get("routes") or {}),
            }
        )

    results.sort(key=lambda r: (-float(r["score"]), int(r["elapsed_ms"]), str(r["profile"])))
    return results


def render_bench_table(
    rows: list[dict],
    *,
    split: str = "evolve",
    color: bool = True,
    width: Optional[int] = None,
) -> str:
    """Render a side-by-side configuration benchmark leaderboard."""
    target_w = max(72, min(116, int(width or 96)))
    if not rows:
        return render_panel(
            ["(no profiles benchmarked)"],
            title=f"NEXUS CONFIG BENCHMARK · {split.upper()}",
            width=target_w,
            border_color=DIM if color else "",
        )

    name_w = max(12, min(22, target_w - 64))
    header = (
        f"{pad_ansi('PROFILE', name_w)} │ "
        f"{pad_ansi('MODE', 8)} │ "
        f"{pad_ansi('PASS', 7, 'right')} │ "
        f"{pad_ansi('SCORE', 6, 'right')} │ "
        f"{pad_ansi('TIME', 6, 'right')} │ "
        f"{pad_ansi('TIERS', 18)}"
    )
    lines = [f"{BOLD}{header}{RESET}" if color else header, "─" * (target_w - 4)]

    for row in rows:
        pass_str = f"{row['passed']}/{row['total']}"
        score_str = f"{float(row['score']) * 100:.1f}%"
        tiers_str = ",".join(row.get("tiers") or [])
        line = (
            f"{pad_ansi(truncate_ansi(str(row['profile']), name_w, '…'), name_w)} │ "
            f"{pad_ansi(str(row['mode']), 8)} │ "
            f"{pad_ansi(pass_str, 7, 'right')} │ "
            f"{pad_ansi(score_str, 6, 'right')} │ "
            f"{pad_ansi(_fmt_ms(int(row['elapsed_ms'])), 6, 'right')} │ "
            f"{pad_ansi(truncate_ansi(tiers_str, 18, '…'), 18)}"
        )
        lines.append(line)

    best = rows[0]
    footer = f"split={split} · best={best['profile']} ({float(best['score']) * 100:.1f}%)"
    return render_panel(
        lines,
        title=f"NEXUS CONFIG BENCHMARK · {split.upper()}",
        footer=footer,
        width=target_w,
        border_color=GREEN if color else "",
    )
