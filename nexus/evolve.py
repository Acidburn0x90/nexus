"""Continual Recursive Self-Improvement (CRSI) outer-loop optimizer.

The evolve loop proposes edits to the policy's editable files (the routing
table by default), verifies every proposal inside a scratch sandbox against
the ``evolve`` split of a benchmark, and lets a Selector promote at most one
candidate per round.  Every attempt -- accepted or rejected -- is preserved
in the SIM ledger so later rounds never repeat a dead end, while held-out
tasks stay strictly outside the loop: prompt builders reject non-evolve
baselines and a leak scanner checks every model-facing string for held-out
ids and prompts before dispatch.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from statistics import median
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# Imported at module level so tests (and the CLI) can monkeypatch the
# attribute on this module and capture the dispatch verbatim.
from nexus.backends import execute_route
from nexus import sim
from nexus.events import DEFAULT_EVENTS_PATH, emit_event, new_span_id, new_trace_id
from nexus.router import load_routes, resolve_route

DEFAULT_POLICY_PATH = Path(__file__).resolve().parent.parent / "config" / "sim_policy.json"
DEFAULT_BENCHMARK_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "evals" / "routing_benchmark.json"
)

# Security boundaries that are permanently outside editable scope, no matter
# what a policy file or caller requests.
IMMUTABLE_LOCKED_FILES: tuple[str, ...] = (
    "nexus/sim.py",
    "nexus/evolve.py",
    "nexus/drops.py",
    "config/sim_policy.json",
    "config/policies",
    "config/evals",
    "tests",
    ".git",
    ".env",
)

# Supporting config mirrored into every sandbox (read-only reference material
# for verifiers; workspace files always overlay it).
REPO_CONFIG_DIR = DEFAULT_POLICY_PATH.parent

# Fallback used when the policy file is missing or unreadable.
_DEFAULT_POLICY_DATA = {
    "editable_files": ["config/routes.json"],
    "locked_files": list(IMMUTABLE_LOCKED_FILES),
    "max_file_bytes": 524288,
}

OBJECTIVES: tuple[str, ...] = (
    "accuracy",
    "speed",
    "efficiency",
    "open_model",
    "balanced",
)

IMPROVEMENT_LADDER: tuple[tuple[str, str], ...] = (
    (
        "prompt_tune",
        "Refine system_persona, prompt framing, few-shot/output structure, think level, and handoff compaction to squeeze better accuracy and reasoning from open/free/local models.",
    ),
    (
        "profile_tune",
        "Tune route tables (config/routes.json), keyword rules, precedence, fallbacks, ensembles, and config/profiles/*.json to maximize $0 open-model solve rate and eliminate paid/premium escalations.",
    ),
    (
        "code_refactor",
        "Refactor Nexus Python modules (router.py, entity.py, backends.py, visualizer.py, events.py, sessions.py) for algorithmic speed, caching, lower token overhead, and streaming throughput.",
    ),
    (
        "polyglot_rewrite",
        "Rewrite verified hot paths in compiled languages (Rust, C, Zig, Go) under native/ with transparent Python fallback, verified by sandbox compilation, tests, and measured speedup.",
    ),
)

SCOPE_PRESETS: dict[str, dict] = {
    "routes": {
        "editable_files": ("config/routes.json",),
        "editable_dirs": (),
        "description": "Routing table only (classic safe SIM preset)",
    },
    "prompts": {
        "editable_files": ("config/routes.json",),
        "editable_dirs": ("config/profiles",),
        "description": "Prompts, personas, profiles, and routing rules",
    },
    "full": {
        "editable_files": (
            "config/routes.json",
            "nexus/router.py",
            "nexus/entity.py",
            "nexus/backends.py",
            "nexus/visualizer.py",
            "nexus/events.py",
            "nexus/sessions.py",
        ),
        "editable_dirs": ("config/profiles",),
        "description": "Prompts, profiles, routes, and core Python harness modules",
    },
    "polyglot": {
        "editable_files": (
            "config/routes.json",
            "nexus/router.py",
            "nexus/entity.py",
            "nexus/backends.py",
            "nexus/visualizer.py",
            "nexus/events.py",
            "nexus/sessions.py",
        ),
        "editable_dirs": ("config/profiles", "native"),
        "description": "Full Python harness plus native/ compiled language rewrites (Rust/C/Zig/Go)",
    },
}


def normalize_objective(raw: Optional[str]) -> str:
    """Normalize an optimization objective (`open-model` -> `open_model`)."""
    key = (raw or "balanced").strip().lower().replace("-", "_")
    if key not in OBJECTIVES:
        raise ValueError(f"unknown objective '{raw}'; expected one of {OBJECTIVES}")
    return key


CANONICAL_DIRECTIVES: tuple[str, ...] = (
    "D1 Inject the complete SIM ledger into both the Proposer and Selector prompts without silent truncation.",
    "D2 Enforce strict Evolve / Held-Out isolation: held-out outcomes and traces never enter the Proposer, the Selector, or the SIM ledger.",
    "D3 Require explicit root-cause attribution (an existing problem_id, or a new problem_name plus hypothesis) before editing any file.",
    "D4 Log falsifiable pre-evaluation `predicted` task ids before scoring a proposal.",
    "D5 Read provenance (id, opened_round, domain, policy_model, carried_from) before judging whether inherited knowledge is relevant.",
    "D6 When inherited code interacts with a failure, choose explicitly among retain, revise, replace, and remove (or open a new problem).",
    "D7 Preserve history even when inherited checks are revised, replaced, or removed.",
    "D8 Merge duplicate root-cause diagnoses through SIM aliases without dropping historical attempts.",
    "D9 Retain failures, misses (fixed == []), rejected proposals, and regressions explicitly in the ledger.",
    "D10 Lock security boundaries, sandboxing, verifiers, and approval gates permanently outside editable scope.",
)

PROACTIVE_RESEARCH_DIRECTIVES: tuple[str, ...] = (
    "R1 Treat every roadblock, failure, or dead end as an unanswered question—state the concrete question and falsifiable hypothesis before editing.",
    "R2 Check the latest AI/CS research papers and proven open-source methods before inventing custom mechanisms.",
    "R3 Surface unresolved assumptions, missing metrics, or architectural trade-offs explicitly as open_questions for the operator or next round.",
    "R4 Proactively pursue higher-leverage improvements (prompt framing, routing resilience, compression, or native speedups) even when baseline accuracy is 100%.",
)

TRANSFER_TIERS: dict[str, str] = {
    "invariant": (
        "Tier 1 Durable Invariants: retain core execution loops, interfaces, and structural checks."
    ),
    "adaptable": (
        "Tier 2 Portable Rationales Requiring New Code: revise or replace the mechanism while "
        "preserving the root-cause rationale in the SIM ledger."
    ),
    "model_specific": (
        "Tier 3 Policy-Locked & Domain-Locked Traps: prune dead predicates and foreign-model parser "
        "repairs, and inspect fixed == [] zero-fix records before reusing anything."
    ),
}

_PROPOSAL_SCHEMA = (
    '{"problem_id": "P<n>|new", "problem_name": "...", '
    '"tier": "invariant|adaptable|model_specific", '
    '"action": "new|retain|revise|replace|remove", '
    '"strategy": "prompt_tune|profile_tune|code_refactor|polyglot_rewrite", "what": "...", '
    '"predicted": ["task_id", ...], "edits": [{"path": "...", '
    '"mode": "write|replace", "content": "...", "search": "...", "replace": "..."}]}'
)

_SELECTOR_SCHEMA = '{"selected_sample": "<sample_id>", "rationale": "..."}'

_JSON_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*(.*?)```", re.DOTALL)


# ---------------------------------------------------------------------------
# Scope policy


def _normalize_rel(raw: object) -> str:
    """Normalise a workspace-relative path for scope comparisons."""
    value = str(raw or "").strip()
    if not value:
        return ""
    while value.startswith("./"):
        value = value[2:]
    while len(value) > 1 and value.endswith("/"):
        value = value[:-1]
    return value


def _locked_match(path: str, locked: object) -> bool:
    """True when `path` equals a locked entry or lives inside a locked directory."""
    return any(path == entry or path.startswith(str(entry) + "/") for entry in locked)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _dir_match(path: str, dirs: tuple[str, ...]) -> bool:
    """True when `path` equals an allowed directory or resides inside one."""
    return any(path == entry or path.startswith(str(entry) + "/") for entry in dirs)


@dataclass
class ScopePolicy:
    """Edit-scope contract: editable files/dirs, immutable lock set, build cmd, size cap."""

    editable_files: tuple[str, ...] = ()
    locked_files: tuple[str, ...] = IMMUTABLE_LOCKED_FILES
    max_file_bytes: int = 524288
    editable_dirs: tuple[str, ...] = ()
    build_cmd: tuple[str, ...] = ()
    build_timeout_s: float = 120.0

    def __post_init__(self) -> None:
        locked = [_normalize_rel(entry) for entry in self.locked_files]
        for entry in IMMUTABLE_LOCKED_FILES:
            if entry not in locked:
                locked.append(entry)
        self.locked_files = tuple(locked)
        # The lock always wins: an editable entry inside a locked tree is dropped.
        self.editable_files = tuple(
            rel
            for rel in (_normalize_rel(entry) for entry in self.editable_files)
            if rel and not _locked_match(rel, self.locked_files)
        )
        self.editable_dirs = tuple(
            rel
            for rel in (_normalize_rel(entry) for entry in self.editable_dirs)
            if rel and not _locked_match(rel, self.locked_files)
        )
        self.build_cmd = tuple(str(part) for part in (self.build_cmd or ()) if str(part))
        self.build_timeout_s = max(1.0, float(self.build_timeout_s or 120.0))
        self.max_file_bytes = int(self.max_file_bytes)

    @classmethod
    def load(
        cls,
        path: Optional[Path] = None,
        *,
        extra_editable: Optional[list[str]] = None,
        extra_locked: Optional[list[str]] = None,
        extra_dirs: Optional[list[str]] = None,
    ) -> "ScopePolicy":
        """Load a policy file, keeping the immutable security defaults locked.

        A missing or corrupt file falls back to the canonical defaults.
        ``extra_editable`` / ``extra_dirs`` entries that collide with a locked
        path are dropped: ``locked_files`` always wins.
        """
        source = Path(path).expanduser() if path is not None else DEFAULT_POLICY_PATH
        data: dict = dict(_DEFAULT_POLICY_DATA)
        if source.is_file():
            try:
                loaded = json.loads(source.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError, ValueError):
                loaded = None
            if isinstance(loaded, dict):
                data = loaded

        editable: list[str] = []
        for entry in list(data.get("editable_files") or []) + list(extra_editable or []):
            rel = _normalize_rel(entry)
            if rel and rel not in editable:
                editable.append(rel)

        dirs: list[str] = []
        for entry in list(data.get("editable_dirs") or []) + list(extra_dirs or []):
            rel = _normalize_rel(entry)
            if rel and rel not in dirs:
                dirs.append(rel)

        locked: list[str] = []
        for entry in list(data.get("locked_files") or []) + list(extra_locked or []):
            rel = _normalize_rel(entry)
            if rel and rel not in locked:
                locked.append(rel)

        try:
            max_file_bytes = int(data.get("max_file_bytes") or 524288)
        except (TypeError, ValueError):
            max_file_bytes = 524288

        build_cmd = tuple(str(part) for part in (data.get("build_cmd") or []) if str(part))
        try:
            build_timeout_s = max(1.0, float(data.get("build_timeout_s") or 120.0))
        except (TypeError, ValueError):
            build_timeout_s = 120.0

        return cls(
            editable_files=tuple(editable),
            locked_files=tuple(locked),
            max_file_bytes=max_file_bytes,
            editable_dirs=tuple(dirs),
            build_cmd=build_cmd,
            build_timeout_s=build_timeout_s,
        )

    @classmethod
    def for_scope(
        cls,
        scope: str,
        *,
        build_cmd: Optional[list[str]] = None,
        build_timeout_s: float = 120.0,
        extra_locked: Optional[list[str]] = None,
    ) -> "ScopePolicy":
        """Construct a named `ScopePolicy` preset (`routes`, `prompts`, `full`, `polyglot`)."""
        key = str(scope or "").strip().lower()
        if key not in SCOPE_PRESETS:
            raise ValueError(
                f"unknown scope '{scope}'; expected one of {tuple(SCOPE_PRESETS)}"
            )
        preset = SCOPE_PRESETS[key]
        locked = list(IMMUTABLE_LOCKED_FILES) + list(extra_locked or [])
        return cls(
            editable_files=tuple(preset.get("editable_files") or ()),
            editable_dirs=tuple(preset.get("editable_dirs") or ()),
            locked_files=tuple(locked),
            build_cmd=tuple(str(part) for part in (build_cmd or []) if str(part)),
            build_timeout_s=build_timeout_s,
        )

    def validate_paths(self, paths: list[str], workspace: Path) -> tuple[bool, list[str]]:
        """Return ``(ok, errors)`` for a batch of proposed edit paths.

        Rejects empty paths, absolute paths, ``..`` traversal, symlinked edit
        targets, parent symlinks that escape the workspace, anything locked
        by policy, and anything not explicitly editable via `editable_files`
        or `editable_dirs`.  A parent symlink resolving inside the workspace
        is an allowed alias.
        """
        workspace = Path(workspace)
        ws_root = workspace.resolve()

        errors: list[str] = []
        for raw in paths:
            label = str(raw) if str(raw) else "<empty>"
            rel = _normalize_rel(raw)
            if not rel:
                errors.append(f"path '{label}' is empty")
                continue
            candidate = Path(rel)
            if candidate.is_absolute() or str(raw).startswith("~"):
                errors.append(f"path '{label}' must be relative to the workspace")
                continue
            if ".." in candidate.parts:
                errors.append(f"path '{label}' must not traverse outside the workspace")
                continue
            if _locked_match(rel, self.locked_files):
                errors.append(f"path '{label}' is locked by policy")
                continue

            current = workspace
            for index, part in enumerate(candidate.parts):
                current = current / part
                if not current.is_symlink():
                    continue
                if index == len(candidate.parts) - 1:
                    # The edit target itself may never be a symlink.
                    errors.append(f"path '{label}' is a symlink")
                    break
                resolved = current.resolve()
                if not _is_within(resolved, ws_root):
                    errors.append(f"path '{label}': parent symlink escapes workspace")
                    break
                # In-workspace alias: keep walking the resolved location.
                current = resolved
            else:
                if rel not in self.editable_files and not _dir_match(rel, self.editable_dirs):
                    errors.append(f"path '{label}' is not listed as editable")

        return (not errors, errors)


# ---------------------------------------------------------------------------
# Benchmark spec


def _validate_tasks(raw: object, split: str, source: Path) -> list[dict]:
    if not isinstance(raw, list):
        raise ValueError(f"benchmark at {source}: '{split}_tasks' must be a list")
    tasks: list[dict] = []
    seen: set[str] = set()
    for index, task in enumerate(raw):
        if not isinstance(task, dict):
            raise ValueError(f"benchmark at {source}: {split} task[{index}] must be an object")
        kind = str(task.get("kind") or "routing").strip().lower()
        tid = str(task.get("id") or "").strip()
        prompt = str(task.get("prompt") or "")
        expected = str(task.get("expected_route") or "").strip()
        if not tid:
            raise ValueError(f"benchmark at {source}: {split} task[{index}] is missing 'id'")
        if tid in seen:
            raise ValueError(f"benchmark at {source}: duplicate task id '{tid}' in {split}")
        if not prompt:
            raise ValueError(f"benchmark at {source}: task '{tid}' is missing 'prompt'")
        if kind == "routing" and not expected:
            raise ValueError(f"benchmark at {source}: task '{tid}' is missing 'expected_route'")
        seen.add(tid)
        entry = dict(task)
        entry["id"] = tid
        entry["prompt"] = prompt
        if expected:
            entry["expected_route"] = expected
        if entry.get("forbidden_routes"):
            entry["forbidden_routes"] = [str(route) for route in task["forbidden_routes"]]
        tasks.append(entry)
    return tasks


@dataclass
class BenchmarkSpec:
    """Deterministic benchmark with strictly disjoint evolve and held-out splits."""

    name: str
    domain: str
    editable_files: list[str]
    locked_files: list[str]
    evolve_tasks: list[dict]
    held_out_tasks: list[dict]
    verifier: str = "builtin:routing"
    verifier_cmd: list[str] = field(default_factory=list)
    build_cmd: list[str] = field(default_factory=list)

    def tasks_for(self, split: str) -> list[dict]:
        """Return the task list for ``split`` ('evolve' or 'held_out')."""
        if split == "evolve":
            return self.evolve_tasks
        if split == "held_out":
            return self.held_out_tasks
        raise ValueError(f"unknown split '{split}'; expected 'evolve' or 'held_out'")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "BenchmarkSpec":
        """Load and validate a benchmark; overlapping splits raise ``ValueError``."""
        source = Path(path).expanduser() if path is not None else DEFAULT_BENCHMARK_PATH
        data = json.loads(source.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(data, dict):
            raise ValueError(f"benchmark at {source} must be a JSON object")
        evolve = _validate_tasks(data.get("evolve_tasks"), "evolve", source)
        held = _validate_tasks(data.get("held_out_tasks"), "held_out", source)
        overlap = sorted({t["id"] for t in evolve} & {t["id"] for t in held})
        if overlap:
            raise ValueError(
                f"benchmark at {source}: evolve and held_out task ids overlap: "
                + ", ".join(overlap)
            )
        return cls(
            name=str(data.get("name") or source.stem),
            domain=str(data.get("domain") or "routing"),
            editable_files=[str(entry) for entry in (data.get("editable_files") or [])],
            locked_files=[str(entry) for entry in (data.get("locked_files") or [])],
            evolve_tasks=evolve,
            held_out_tasks=held,
            verifier=str(data.get("verifier") or "builtin:routing"),
            verifier_cmd=[str(part) for part in (data.get("verifier_cmd") or [])],
            build_cmd=[str(part) for part in (data.get("build_cmd") or [])],
        )


# ---------------------------------------------------------------------------
# Evaluation


@dataclass
class EvalReport:
    """Deterministic verification outcome for one benchmark split."""

    split: str
    results: dict[str, bool] = field(default_factory=dict)
    traces: dict[str, str] = field(default_factory=dict)
    passed: int = 0
    total: int = 0
    score: float = 0.0
    policy_models: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    tokens_total: int = 0
    metrics: dict[str, float] = field(default_factory=dict)

    def failing_ids(self) -> list[str]:
        """Return the task ids that did not pass, in report order."""
        return [tid for tid, ok in self.results.items() if not ok]


def _report(
    split: str,
    results: dict,
    traces: dict,
    policy_models: list,
    *,
    latency_ms: float = 0.0,
    tokens_total: int = 0,
    metrics: Optional[dict[str, float]] = None,
) -> EvalReport:
    passed = sum(1 for value in results.values() if value)
    total = len(results)
    return EvalReport(
        split=split,
        results=results,
        traces=traces,
        passed=passed,
        total=total,
        score=(passed / total) if total else 0.0,
        policy_models=list(policy_models),
        latency_ms=float(latency_ms or 0.0),
        tokens_total=int(tokens_total or 0),
        metrics=dict(metrics or {}),
    )


def _route_cfg_problem(route_key: str, cfg: object) -> tuple[bool, str]:
    if not isinstance(cfg, dict):
        return False, f"policy route '{route_key}' missing or invalid"
    if not str(cfg.get("backend") or "").strip():
        return False, f"policy route '{route_key}' missing or invalid"
    if not str(cfg.get("model") or "").strip():
        return False, f"policy route '{route_key}' missing or invalid"
    return True, ""


def _check_routing_task(task: dict, routes_data: dict) -> tuple[bool, str]:
    expected = str(task["expected_route"])
    forbidden = [str(route) for route in (task.get("forbidden_routes") or [])]
    try:
        got, _cfg, _cleaned = resolve_route(str(task["prompt"]), routes_data=routes_data)
    except (KeyError, ValueError) as exc:
        return False, f"route resolution failed: {exc}"
    if got != expected:
        return False, f"expected {expected}, got {got}"
    if got in forbidden:
        return False, f"route '{got}' is forbidden for this task"
    return True, ""


def _parse_numeric_metrics(raw: object) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    clean: dict[str, float] = {}
    for k, v in raw.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            clean[str(k)] = float(v)
    return clean


def evaluate_split(
    workspace: Path,
    spec: BenchmarkSpec,
    *,
    split: str = "evolve",
    policy_routes: Optional[list[str]] = None,
    timeout_s: float = 30.0,
    repeats: int = 1,
) -> EvalReport:
    """Verify one split of `spec` inside `workspace`.

    ``builtin:routing`` resolves every task prompt against the workspace route
    table and measures median ``latency_ms`` across ``repeats`` passes.
    ``command`` verifiers receive ``NEXUS_EVAL_SPLIT`` and
    ``NEXUS_EVAL_WORKSPACE`` and may emit optional ``latency_ms``,
    ``tokens_total``, and numeric ``metrics`` alongside ``results``.
    """
    if split not in ("evolve", "held_out"):
        raise ValueError(f"unknown split '{split}'; expected 'evolve' or 'held_out'")
    workspace = Path(workspace)
    tasks = spec.tasks_for(split)
    pooled = [str(route) for route in (policy_routes or [])]
    policy_models = list(pooled) if pooled else ["default"]

    if spec.verifier == "builtin:routing":
        routes_data = load_routes(workspace / "config" / "routes.json")
        runs = max(1, int(repeats))
        timings_ms: list[float] = []
        results: dict[str, bool] = {}
        traces: dict[str, str] = {}
        for _ in range(runs):
            pass_results: dict[str, bool] = {}
            pass_traces: dict[str, str] = {}
            t0 = time.monotonic()
            if len(pooled) > 1:
                routes = routes_data.get("routes") or {}
                for route_key in pooled:
                    cfg_ok, cfg_error = _route_cfg_problem(route_key, routes.get(route_key))
                    for task in tasks:
                        key = f"{route_key}::{task['id']}"
                        if not cfg_ok:
                            pass_results[key] = False
                            pass_traces[key] = cfg_error
                            continue
                        ok, trace = _check_routing_task(task, routes_data)
                        pass_results[key] = ok
                        if not ok:
                            pass_traces[key] = trace
            else:
                for task in tasks:
                    ok, trace = _check_routing_task(task, routes_data)
                    pass_results[task["id"]] = ok
                    if not ok:
                        pass_traces[task["id"]] = trace
            timings_ms.append((time.monotonic() - t0) * 1000.0)
            results, traces = pass_results, pass_traces
        return _report(
            split,
            results,
            traces,
            policy_models,
            latency_ms=median(timings_ms) if timings_ms else 0.0,
        )

    if spec.verifier == "command":
        if not spec.verifier_cmd:
            raise ValueError(f"benchmark '{spec.name}' verifier 'command' has no verifier_cmd")
        results = {str(task["id"]): False for task in tasks}
        traces = {
            str(task["id"]): "verifier produced no verdict for this task" for task in tasks
        }
        env = dict(os.environ)
        env["NEXUS_EVAL_SPLIT"] = split
        env["NEXUS_EVAL_WORKSPACE"] = str(workspace)
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [str(part) for part in spec.verifier_cmd],
                cwd=str(workspace),
                shell=False,
                timeout=timeout_s,
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            for tid in traces:
                traces[tid] = f"verifier timed out after {timeout_s:g}s"
            return _report(
                split,
                results,
                traces,
                policy_models,
                latency_ms=(time.monotonic() - t0) * 1000.0,
            )
        except OSError as exc:
            for tid in traces:
                traces[tid] = f"verifier failed to start: {exc}"
            return _report(
                split,
                results,
                traces,
                policy_models,
                latency_ms=(time.monotonic() - t0) * 1000.0,
            )

        wall_ms = (time.monotonic() - t0) * 1000.0
        payload: Optional[dict] = None
        stdout = proc.stdout or ""
        if stdout.strip():
            try:
                parsed = json.loads(stdout)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and isinstance(parsed.get("results"), dict):
                payload = parsed

        reported_latency = wall_ms
        reported_tokens = 0
        reported_metrics: dict[str, float] = {}
        if payload is not None:
            reported = payload["results"]
            reported_traces = payload.get("traces")
            reported_traces = reported_traces if isinstance(reported_traces, dict) else {}
            if isinstance(payload.get("latency_ms"), (int, float)) and not isinstance(
                payload.get("latency_ms"), bool
            ):
                reported_latency = float(payload["latency_ms"])
            if isinstance(payload.get("tokens_total"), (int, float)) and not isinstance(
                payload.get("tokens_total"), bool
            ):
                reported_tokens = int(payload["tokens_total"])
            reported_metrics = _parse_numeric_metrics(payload.get("metrics"))
            for tid in results:
                value = reported.get(tid, False)
                results[tid] = value is True or value == 1
                if tid in reported_traces:
                    traces[tid] = str(reported_traces[tid])
                elif not results[tid]:
                    traces[tid] = "verifier reported failure"
        elif proc.returncode == 0:
            for tid in results:
                results[tid] = True
                traces.pop(tid, None)
        else:
            stderr_lines = (proc.stderr or "").strip().splitlines()
            trace = (
                stderr_lines[0]
                if stderr_lines
                else f"verifier exited with code {proc.returncode}"
            )
            for tid in traces:
                traces[tid] = trace
        return _report(
            split,
            results,
            traces,
            policy_models,
            latency_ms=reported_latency,
            tokens_total=reported_tokens,
            metrics=reported_metrics,
        )

    if spec.verifier == "builtin:core_quality":
        from nexus.assay import build_assay_receipt
        from nexus.backends import _bounded_history, classify_http_error

        routes_path = workspace / "config" / "routes.json"
        routes_data = load_routes(routes_path)
        routes_raw = routes_path.read_bytes() if routes_path.is_file() else json.dumps(routes_data).encode("utf-8")

        results: dict[str, bool] = {}
        traces: dict[str, str] = {}
        stages: list[dict] = []
        latencies_ms: list[float] = []
        comp_ratios: list[float] = []

        for task in tasks:
            tid = str(task["id"])
            kind = str(task.get("kind") or "routing")
            t0 = time.monotonic()
            ok = False
            detail = ""
            if kind == "routing":
                try:
                    got, _cfg, _cleaned = resolve_route(str(task["prompt"]), routes_data=routes_data)
                    exp = str(task["expected_route"])
                    ok = got == exp
                    detail = f"got={got} expected={exp}"
                except Exception as exc:
                    detail = f"routing error: {exc}"
            elif kind == "compression":
                turns = max(4, int(task.get("turns", 8)))
                tool_chars = max(260, int(task.get("tool_chars", 900)))
                min_ratio = float(task.get("min_savings_ratio", 0.35))
                hist = []
                for idx in range(turns - 3):
                    hist.append({"role": "user", "content": f"turn {idx}"})
                    hist.append({"role": "assistant", "content": "[Tool output: scan]\n" + ("a" * tool_chars)})
                for idx in range(3):
                    hist.append({"role": "user", "content": f"recent ask {idx}"})
                    hist.append({"role": "assistant", "content": f"recent answer {idx}"})
                orig_chars = sum(len(m["content"]) for m in hist)
                kept_chars = sum(len(m["content"]) for m in _bounded_history(hist))
                ratio = (orig_chars - kept_chars) / max(1, orig_chars)
                comp_ratios.append(ratio)
                ok = ratio >= min_ratio
                detail = f"savings={ratio:.2%} min={min_ratio:.2%}"
            elif kind == "health_classify":
                pairs = task.get("inputs") or []
                mismatches = []
                for inp, exp in pairs:
                    got = classify_http_error(inp)
                    if got != exp:
                        mismatches.append(f"{inp!r}->{got}!={exp}")
                ok = bool(pairs) and not mismatches
                detail = "ok" if ok else ", ".join(mismatches)
            elif kind == "latency_budget":
                resolve_route(str(task.get("prompt") or ""), routes_data=routes_data)
                dt_ms = (time.monotonic() - t0) * 1000.0
                latencies_ms.append(dt_ms)
                max_ms = float(task.get("max_ms", 50.0))
                ok = dt_ms <= max_ms
                detail = f"elapsed={dt_ms:.2f}ms max={max_ms:.1f}ms"
            else:
                detail = f"unsupported core_quality task kind '{kind}'"

            results[tid] = ok
            if not ok:
                traces[tid] = detail
            stages.append({"name": f"{tid}:{kind}", "passed": ok, "detail": detail})

        receipt = build_assay_receipt(
            routes_raw,
            ideal_state=f"{spec.name} ({split})",
            stages=stages,
            target=spec.name,
        )
        metrics: dict[str, float] = {
            "assay_passed": float(receipt["passed"]),
            "assay_total": float(receipt["total"]),
            "compression_ratio": round(sum(comp_ratios) / len(comp_ratios), 4) if comp_ratios else 0.0,
            "routing_latency_ms": round(max(latencies_ms), 3) if latencies_ms else 0.0,
        }
        return _report(
            split,
            results,
            traces,
            policy_models,
            latency_ms=sum(latencies_ms) if latencies_ms else 1.0,
            metrics=metrics,
        )

    raise ValueError(f"unsupported verifier '{spec.verifier}' for benchmark '{spec.name}'")


# ---------------------------------------------------------------------------
# Held-out leak scanner


def scan_held_out_leaks(text: str, spec: BenchmarkSpec) -> list[str]:
    """Return held-out ids whose id or full prompt appears in `text`.

    Ids are matched as standalone tokens (word-ish boundaries) so short ids
    like ``h1`` never false-positive inside unrelated content; full prompts
    are matched verbatim.
    """
    text = str(text or "")
    leaks: list[str] = []
    for task in spec.held_out_tasks or []:
        tid = str(task.get("id") or "")
        prompt = str(task.get("prompt") or "")
        hit = False
        if tid:
            pattern = rf"(?<![A-Za-z0-9_-]){re.escape(tid)}(?![A-Za-z0-9_-])"
            hit = re.search(pattern, text) is not None
        if not hit and prompt:
            hit = prompt in text
        if hit:
            leaks.append(tid or prompt)
    return leaks


def _assert_no_held_out_leak(text: str, spec: BenchmarkSpec) -> None:
    """Raise on a model-facing string that carries held-out data.

    Leaked ids are deliberately not repeated in the message: the exception
    text can end up in ledger records and must stay free of held-out data.
    """
    if scan_held_out_leaks(text, spec):
        raise ValueError("held-out firewall breach: model-facing prompt contains held-out tasks")


def _assert_no_proposal_leak(candidate: dict, spec: BenchmarkSpec) -> None:
    """Raise when a parsed proposal itself references held-out data.

    The scan covers the fields that would flow into prompts and the SIM
    ledger: problem_name, what, predicted and the raw edits list.
    """
    payload = json.dumps(
        {
            "problem_name": candidate.get("problem_name"),
            "what": candidate.get("what"),
            "predicted": candidate.get("predicted"),
            "edits": candidate.get("edits"),
        },
        ensure_ascii=False,
    )
    if scan_held_out_leaks(payload, spec):
        raise ValueError("held-out firewall breach: proposal references held-out tasks")


# ---------------------------------------------------------------------------
# Prompts


def _models_label(policy_models: list) -> str:
    return ", ".join(str(model) for model in (policy_models or ["default"])) or "default"


def build_proposer_prompt(
    ledger: dict,
    baseline: EvalReport,
    editable_contents: dict[str, str],
    *,
    round_num: int,
    domain: str,
    policy_models: list[str],
    objective: str = "balanced",
    policy: Optional[ScopePolicy] = None,
    spec: Optional[BenchmarkSpec] = None,
) -> str:
    """Build the Proposer prompt; only the evolve split may ever reach it."""
    if baseline.split != "evolve":
        raise ValueError(
            f"proposer input must be the evolve split, got '{baseline.split}' "
            "(held-out firewall)"
        )
    obj = normalize_objective(objective)

    lines: list[str] = []
    lines.append("You are the Proposer in the Nexus self-improvement loop.")
    lines.append(
        f"Domain: {domain}. Benchmark: evolve split only. Round: {round_num}. "
        f"Objective: {obj}. Policy models: {_models_label(policy_models)}."
    )
    lines.append("")
    lines.append("## Canonical directives (mandatory)")
    for directive in CANONICAL_DIRECTIVES:
        lines.append(f"- {directive}")
    lines.append("")
    lines.append("## Proactive research & inquiry directives")
    for directive in PROACTIVE_RESEARCH_DIRECTIVES:
        lines.append(f"- {directive}")
    lines.append("")
    lines.append("## Improvement Incentive & Ladder")
    lines.append(
        "Always look for the simplest high-leverage win across the ladder below "
        "(prefer prompt/profile optimization to squeeze open models first; escalate "
        "to Python refactoring or native Rust/C/Zig/Go rewrites under native/ when "
        "latency or throughput bottlenecks require it):"
    )
    for lever, desc in IMPROVEMENT_LADDER:
        lines.append(f"- {lever}: {desc}")
    if policy is not None:
        if policy.editable_dirs:
            lines.append(f"Allowed editable directories (new files supported): {', '.join(policy.editable_dirs)}")
        effective_build = policy.build_cmd or (tuple(spec.build_cmd) if spec is not None else ())
        if effective_build:
            lines.append(f"Sandbox build gate before verification: {' '.join(effective_build)}")
    lines.append("")
    lines.append("## Transfer tiers")
    for tier, rule in TRANSFER_TIERS.items():
        lines.append(f"- {tier}: {rule}")
    lines.append("")
    lines.append("## Current evolve baseline")
    perf_bits = []
    if baseline.latency_ms > 0:
        perf_bits.append(f"latency_ms {baseline.latency_ms:.2f}")
    if baseline.tokens_total > 0:
        perf_bits.append(f"tokens_total {baseline.tokens_total}")
    if baseline.metrics:
        perf_bits.append("metrics " + json.dumps(baseline.metrics, ensure_ascii=False))
    perf_suffix = (" | " + ", ".join(perf_bits)) if perf_bits else ""
    lines.append(f"passed {baseline.passed}/{baseline.total} (score {baseline.score:.4f}){perf_suffix}")
    failing = [(tid, baseline.traces.get(tid, "")) for tid, ok in sorted(baseline.results.items()) if not ok]
    if failing:
        lines.append("Failing evolve tasks (held-out data is never shown):")
        for tid, trace in failing:
            lines.append(f"- {tid}: {trace or 'failed'}")
    else:
        lines.append(
            "Baseline task accuracy is already 100%; focus on squeezing open-model "
            "reliability, prompt conciseness, latency, token efficiency, or polyglot "
            "native/ rewrites where editable (avoid unnecessary no-op churn)."
        )
    lines.append("")
    lines.append("## Complete SIM LEDGER (injected without truncation)")
    lines.append(sim.format_sim_for_prompt(ledger))
    lines.append("")
    lines.append("## Editable file contents")
    if editable_contents:
        for rel, content in sorted(editable_contents.items()):
            lines.append(f"### {rel}")
            lines.append(content)
            lines.append("")
    else:
        lines.append("(no editable file is currently readable)")
    lines.append("## Required response")
    lines.append("Return ONLY one JSON object with these keys:")
    lines.append(_PROPOSAL_SCHEMA)
    lines.append("Rules:")
    lines.append(
        "- Attribute the root cause: reuse an existing problem_id, or open a new "
        "problem with problem_name plus the hypothesis recorded in 'what'."
    )
    lines.append(
        "- 'predicted' lists the evolve task ids your edits will fix; it is logged "
        "before scoring and is falsifiable."
    )
    lines.append(
        "- 'edits' may only touch the editable paths/dirs allowed by policy; mode 'write' "
        "creates or replaces the file, mode 'replace' swaps every exact 'search' "
        "occurrence for 'replace'."
    )
    lines.append("- Never reference held-out tasks: their outcomes stay outside this loop.")
    return "\n".join(lines)


def build_selector_prompt(
    ledger: dict,
    baseline: EvalReport,
    candidates: list[dict],
    *,
    round_num: int,
    policy_models: list[str],
    objective: str = "balanced",
) -> str:
    """Build the Selector prompt; only the evolve split may ever reach it."""
    if baseline.split != "evolve":
        raise ValueError(
            f"selector input must be the evolve split, got '{baseline.split}' "
            "(held-out firewall)"
        )
    obj = normalize_objective(objective)

    def _fmt(value: object) -> str:
        return "-" if value in (None, "") else str(value)

    lines: list[str] = []
    lines.append("You are the Selector in the Nexus self-improvement loop.")
    lines.append(
        f"Round: {round_num}. Benchmark: evolve split only. Objective: {obj}. "
        f"Policy models: {_models_label(policy_models)}."
    )
    lines.append(
        f"Baseline: passed {baseline.passed}/{baseline.total} (score {baseline.score:.4f}, "
        f"latency_ms {baseline.latency_ms:.2f}, tokens_total {baseline.tokens_total})."
    )
    lines.append("")
    lines.append("## Canonical directives (mandatory)")
    for directive in CANONICAL_DIRECTIVES:
        lines.append(f"- {directive}")
    lines.append("")
    lines.append("## Proactive research & inquiry directives")
    for directive in PROACTIVE_RESEARCH_DIRECTIVES:
        lines.append(f"- {directive}")
    lines.append("")
    lines.append("## Complete SIM LEDGER (injected without truncation)")
    lines.append(sim.format_sim_for_prompt(ledger))
    lines.append("")
    lines.append("## Candidate proposals evaluated on the evolve split")
    for candidate in candidates:
        score = candidate.get("score")
        summary = {
            "sample": candidate.get("sample"),
            "strategy": candidate.get("strategy", ""),
            "action": candidate.get("action"),
            "problem_id": candidate.get("problem_id"),
            "what": candidate.get("what"),
            "predicted": list(candidate.get("predicted") or []),
            "fixed": list(candidate.get("fixed") or []),
            "missed": list(candidate.get("missed") or []),
            "regressed": list(candidate.get("regressed") or []),
            "passed": candidate.get("passed", 0),
            "total": candidate.get("total", 0),
            "score": float(score) if isinstance(score, (int, float)) else None,
            "latency_ms": float(candidate.get("latency_ms") or 0.0),
            "tokens_total": int(candidate.get("tokens_total") or 0),
            "metrics": dict(candidate.get("metrics") or {}),
            "error": _fmt(candidate.get("error")),
        }
        lines.append(f"- {json.dumps(summary, ensure_ascii=False)}")
    lines.append("")
    lines.append("## Required response")
    lines.append(f"Return ONLY one JSON object: {_SELECTOR_SCHEMA}")
    lines.append("Rules:")
    lines.append(
        "- Never select a candidate with regressions over the baseline, a failed build, "
        "or a rejected scope check."
    )
    lines.append(
        "- Prefer the highest score; at equal accuracy, prefer measurable gains on "
        f"objective '{obj}' (faster latency_ms, lower tokens_total, higher open_model_rate)."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Proposal handling


def parse_json_object(text: str) -> dict:
    """Extract a JSON object from raw text or a fenced block; else raise."""
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty response: expected a JSON object")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    for block in _JSON_FENCE_RE.findall(raw):
        candidate = block.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    start, end = raw.find("{"), raw.rfind("}")
    if 0 <= start < end:
        try:
            parsed = json.loads(raw[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError("could not parse a JSON object from the response")


def _enforce_size(content: str, policy: ScopePolicy, rel: str) -> None:
    if len(content.encode("utf-8")) > policy.max_file_bytes:
        raise ValueError(
            f"edit for '{rel}' exceeds max_file_bytes ({policy.max_file_bytes})"
        )


def _copy_tree_files(src_root: Path, dst_root: Path) -> None:
    """Copy regular files only: symlinks and special entries are skipped."""
    for src in sorted(src_root.rglob("*")):
        if src.is_symlink() or not src.is_file():
            continue
        dst = dst_root / src.relative_to(src_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _seed_sandbox(
    workspace: Path, sandbox_root: Path, policy: ScopePolicy, spec: BenchmarkSpec
) -> None:
    """Mirror supporting config plus editable files and directories into the sandbox.

    The repository config tree provides read-only reference material for
    verifiers; the workspace overlay and editable copies then win for every
    file actually under test.  ``.git`` and locked directories are never copied.
    """
    if REPO_CONFIG_DIR.is_dir():
        _copy_tree_files(REPO_CONFIG_DIR, sandbox_root / "config")
    workspace_config = workspace / "config"
    if workspace_config.is_dir():
        _copy_tree_files(workspace_config, sandbox_root / "config")
    wanted: list[str] = []
    for rel in list(policy.editable_files) + list(spec.editable_files):
        if rel not in wanted:
            wanted.append(rel)
    for rel in wanted:
        src = workspace / rel
        if src.is_symlink() or not src.is_file():
            continue
        dst = sandbox_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for rel_dir in policy.editable_dirs:
        if rel_dir == "config" or rel_dir.startswith("config/"):
            continue
        src_dir = workspace / rel_dir
        if src_dir.is_symlink() or not src_dir.is_dir():
            continue
        for src in sorted(src_dir.rglob("*")):
            if src.is_symlink() or not src.is_file():
                continue
            rel = src.relative_to(workspace).as_posix()
            if _locked_match(rel, policy.locked_files):
                continue
            if "__pycache__" in src.parts:
                continue
            dst = sandbox_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def apply_edits_in_sandbox(
    workspace: Path,
    spec: BenchmarkSpec,
    edits: list[dict],
    policy: ScopePolicy,
) -> tuple[Path, list[str], "tempfile.TemporaryDirectory"]:
    """Validate every edit, then apply them inside a scratch sandbox.

    Returns ``(sandbox_root, touched_rel_paths, tempdir)``; the caller owns
    ``tempdir`` and must clean it up.  Scope violations raise
    ``PermissionError``; malformed edits raise ``ValueError``.  Nothing is
    written before every path passes ``policy.validate_paths``.
    """
    workspace = Path(workspace)
    if not isinstance(edits, list):
        raise ValueError("edits must be a list of objects")

    norm_edits: list[dict] = []
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict):
            raise ValueError(f"edit[{index}] must be an object")
        rel = _normalize_rel(edit.get("path"))
        if not rel:
            raise ValueError(f"edit[{index}] is missing a 'path'")
        norm_edits.append({**edit, "path": rel})

    ok, errors = policy.validate_paths([edit["path"] for edit in norm_edits], workspace)
    if not ok:
        first = errors[0]
        if any(tag in first for tag in ("is empty", "must be relative", "must not traverse")):
            raise ValueError(first)
        raise PermissionError(first)

    tempdir = tempfile.TemporaryDirectory(prefix="nexus-evolve-sandbox-")
    sandbox_root = Path(tempdir.name)
    try:
        _seed_sandbox(workspace, sandbox_root, policy, spec)
        touched: list[str] = []
        seen: set[str] = set()
        for edit in norm_edits:
            rel = edit["path"]
            mode = str(edit.get("mode") or "").strip()
            target = sandbox_root / rel
            if mode == "write":
                content = edit.get("content")
                if not isinstance(content, str):
                    raise ValueError(
                        f"edit for '{rel}' with mode 'write' needs string 'content'"
                    )
                _enforce_size(content, policy, rel)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            elif mode == "replace":
                search = edit.get("search")
                replacement = edit.get("replace")
                if not isinstance(search, str) or not isinstance(replacement, str):
                    raise ValueError(
                        f"edit for '{rel}' with mode 'replace' needs string "
                        "'search' and 'replace'"
                    )
                if not target.is_file():
                    raise ValueError(f"replace target '{rel}' not found in the sandbox")
                content = target.read_text(encoding="utf-8", errors="replace")
                if search not in content:
                    raise ValueError(f"search text not found in '{rel}'")
                content = content.replace(search, replacement)
                _enforce_size(content, policy, rel)
                target.write_text(content, encoding="utf-8")
            else:
                raise ValueError(
                    f"edit for '{rel}': unknown edit mode '{mode or '<missing>'}'"
                )
            if rel not in seen:
                seen.add(rel)
                touched.append(rel)
    except BaseException:
        tempdir.cleanup()
        raise
    return sandbox_root, touched, tempdir


# ---------------------------------------------------------------------------
# Round execution


def _read_editable_contents(workspace: Path, policy: ScopePolicy) -> dict[str, str]:
    contents: dict[str, str] = {}
    for rel in policy.editable_files:
        source = Path(workspace) / rel
        if source.is_symlink() or not source.is_file():
            continue
        try:
            contents[rel] = source.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    for rel_dir in policy.editable_dirs:
        src_dir = Path(workspace) / rel_dir
        if src_dir.is_symlink() or not src_dir.is_dir():
            continue
        for entry in sorted(src_dir.rglob("*")):
            if entry.is_symlink() or not entry.is_file():
                continue
            rel = entry.relative_to(Path(workspace)).as_posix()
            if rel in contents or _locked_match(rel, policy.locked_files):
                continue
            if "__pycache__" in entry.parts:
                continue
            try:
                if entry.stat().st_size > 65536:
                    continue
                contents[rel] = entry.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
    return contents


def _execute_route_span(
    trace_id: str,
    parent_span_id: Optional[str],
    workspace: Path,
    route_key: str,
    prompt: str,
    *,
    events_path: Path,
    allow_fallback: bool,
    intent_label: str,
):
    """Dispatch one optimizer span through the configured backend, or fail softly."""
    try:
        routes_data = load_routes(Path(workspace) / "config" / "routes.json")
    except (OSError, ValueError) as exc:
        return None, f"route table unavailable: {exc}"
    cfg = (routes_data.get("routes") or {}).get(route_key)
    if not isinstance(cfg, dict) or not cfg:
        return None, f"route '{route_key}' missing from the workspace route table"
    span_cfg = dict(cfg)
    span_cfg["intent"] = intent_label
    span_cfg["quota_tier"] = "ORCHESTRATOR"
    result = execute_route(
        trace_id,
        route_key,
        span_cfg,
        prompt,
        parent_id=parent_span_id,
        span_id=new_span_id(),
        events_path=events_path,
        inject_square=False,
        emit_stderr=False,
        allow_fallback=allow_fallback,
    )
    return result, ""


def _dry_run_proposal(
    workspace: Path, baseline: EvalReport, policy: ScopePolicy
) -> dict:
    """Deterministic synthetic proposal: verbatim rewrite, zero backend calls."""
    failing = sorted(tid for tid, ok in baseline.results.items() if not ok)
    edits: list[dict] = []
    for rel in policy.editable_files:
        source = Path(workspace) / rel
        if not source.is_symlink() and source.is_file():
            edits.append(
                {
                    "path": rel,
                    "mode": "write",
                    "content": source.read_text(encoding="utf-8", errors="replace"),
                }
            )
            break
    if edits:
        what = f"dry-run plumbing probe: verbatim rewrite of {edits[0]['path']}"
    else:
        what = "dry-run plumbing probe: no editable file present in the workspace"
    return {
        "problem_id": "new",
        "problem_name": "dry-run plumbing check",
        "tier": "adaptable",
        "action": "new",
        "what": what,
        "predicted": failing,
        "edits": edits,
    }


def _merge_proposal(candidate: dict, proposal: dict) -> None:
    """Normalise a parsed proposal into the candidate record, or raise."""
    if not isinstance(proposal, dict):
        raise ValueError("proposal must be a JSON object")
    problem_id = str(proposal.get("problem_id") or "new").strip() or "new"
    problem_name = str(proposal.get("problem_name") or "").strip()
    tier = str(proposal.get("tier") or "adaptable").strip()
    if tier not in sim.TIER_VALUES:
        tier = "adaptable"
    action = str(proposal.get("action") or "new").strip()
    if action not in sim.ACTION_VALUES:
        action = "new"
    strategy = str(proposal.get("strategy") or "").strip()
    what = str(proposal.get("what") or "").strip()
    if not what:
        raise ValueError("proposal is missing a non-empty 'what'")
    predicted_raw = proposal.get("predicted")
    if predicted_raw is None:
        predicted_raw = []
    if not isinstance(predicted_raw, list):
        raise ValueError("proposal 'predicted' must be a list of task ids")
    edits_raw = proposal.get("edits")
    if edits_raw is None:
        edits_raw = []
    if not isinstance(edits_raw, list):
        raise ValueError("proposal 'edits' must be a list")
    candidate.update(
        problem_id=problem_id,
        problem_name=problem_name,
        tier=tier,
        action=action,
        strategy=strategy,
        what=what,
        predicted=[str(tid) for tid in predicted_raw],
        edits=edits_raw,
    )


def _record_candidate(
    ledger: dict,
    candidate: dict,
    *,
    spec: BenchmarkSpec,
    round_num: int,
    policy_models: list,
    accepted: bool,
) -> dict:
    """Log one candidate (winner or dead end) into the SIM ledger."""
    problem_id = str(candidate.get("problem_id") or "new").strip()
    if problem_id.lower() == "new":
        problem_id = sim.next_problem_id(ledger)
    candidate["problem_id"] = problem_id
    problem_name = str(candidate.get("problem_name") or "").strip()
    if not problem_name:
        problem_name = f"{spec.domain}: {str(candidate.get('what') or 'attempt')[:120]}"
    score = candidate.get("score")
    return sim.record_attempt(
        ledger,
        problem_id=problem_id,
        round_num=int(round_num),
        sample=str(candidate.get("sample") or "s0"),
        action=str(candidate.get("action") or "new"),
        accepted=bool(accepted),
        domain=str(spec.domain),
        what=str(candidate.get("what") or "unnamed attempt"),
        files=[str(rel) for rel in (candidate.get("touched") or [])],
        predicted=[str(tid) for tid in (candidate.get("predicted") or [])],
        fixed=[str(tid) for tid in (candidate.get("fixed") or [])],
        missed=[str(tid) for tid in (candidate.get("missed") or [])],
        passed=int(candidate.get("passed") or 0),
        total=int(candidate.get("total") or 0),
        regressed=[str(tid) for tid in (candidate.get("regressed") or [])],
        policy_model=list(policy_models),
        optimizer_model=str(candidate.get("optimizer_model") or ""),
        benchmark=str(spec.name),
        score=float(score) if isinstance(score, (int, float)) else None,
        diff_ref="",
        problem_name=problem_name,
        tier=str(candidate.get("tier") or "adaptable"),
    )


def _best_candidate(candidates: list, *, objective: str = "balanced") -> dict:
    """Deterministic ranking across accuracy, open-model share, latency, and tokens."""
    obj = normalize_objective(objective)

    def _key(cand: dict) -> tuple:
        metrics = cand.get("metrics") or {}
        open_rate = float(metrics.get("open_model_rate") or metrics.get("primary_metric") or 0.0)
        lat = float(cand.get("latency_ms") or 0.0)
        tok = int(cand.get("tokens_total") or 0)
        # Integer-ms buckets keep sub-millisecond timing noise from breaking ties.
        lat_rank = -int(round(lat)) if lat >= 1.0 else 0
        tok_rank = -tok if tok > 0 else 0
        base_score = float(cand.get("score") or 0.0)
        base_passed = int(cand.get("passed") or 0)
        fixed_len = len(cand.get("fixed") or [])
        if obj == "speed":
            return (base_score, base_passed, lat_rank, fixed_len, open_rate, tok_rank)
        if obj == "efficiency":
            return (base_score, base_passed, tok_rank, fixed_len, open_rate, lat_rank)
        if obj == "open_model":
            return (base_score, base_passed, open_rate, fixed_len, lat_rank, tok_rank)
        return (base_score, base_passed, fixed_len, open_rate, lat_rank, tok_rank)

    return max(candidates, key=_key)


def _run_sandbox_build(
    sandbox_root: Path,
    build_cmd: tuple[str, ...] | list[str],
    timeout_s: float = 120.0,
) -> tuple[bool, str]:
    """Run trusted `build_cmd` inside `sandbox_root` before verifier evaluation."""
    cmd = [str(part) for part in build_cmd if str(part)]
    if not cmd:
        return True, ""
    env = dict(os.environ)
    env["NEXUS_EVAL_WORKSPACE"] = str(sandbox_root)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(sandbox_root),
            shell=False,
            timeout=max(1.0, float(timeout_s or 120.0)),
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"build timed out after {timeout_s:g}s"
    except OSError as exc:
        return False, f"build failed to start: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        msg = detail[-1][:200] if detail else f"exit code {proc.returncode}"
        return False, f"build failed (exit {proc.returncode}): {msg}"
    return True, ""


def beats_baseline(
    candidate: dict,
    baseline: EvalReport,
    *,
    objective: str = "balanced",
    min_speedup_pct: float = 0.05,
    min_token_reduction_pct: float = 0.05,
) -> tuple[bool, str]:
    """Return `(eligible, reason)` for multi-axis zero-regression promotion."""
    obj = normalize_objective(objective)
    if candidate.get("error") or not candidate.get("scope_ok"):
        return False, "error or scope violation"
    if not candidate.get("touched"):
        return False, "no editable files modified"
    if candidate.get("regressed"):
        return False, "task regression detected"

    raw_score = candidate.get("score")
    if raw_score is None:
        return False, "missing candidate score"
    cand_score = float(raw_score)
    if cand_score < float(baseline.score):
        return False, "score below baseline"
    if cand_score > float(baseline.score):
        return True, f"accuracy +{cand_score - float(baseline.score):.4f}"

    if int(candidate.get("passed") or 0) >= int(baseline.passed) and candidate.get("fixed"):
        return True, f"fixed {len(candidate.get('fixed') or [])} task(s)"

    cand_lat = float(candidate.get("latency_ms") or 0.0)
    if (
        obj in ("speed", "balanced")
        and baseline.latency_ms > 0
        and cand_lat > 0
        and cand_lat <= float(baseline.latency_ms) * (1.0 - min_speedup_pct)
    ):
        gain = 1.0 - (cand_lat / float(baseline.latency_ms))
        return True, f"speedup {gain:.1%}"

    cand_tok = int(candidate.get("tokens_total") or 0)
    if (
        obj in ("efficiency", "balanced")
        and baseline.tokens_total > 0
        and cand_tok > 0
        and cand_tok <= float(baseline.tokens_total) * (1.0 - min_token_reduction_pct)
    ):
        gain = 1.0 - (cand_tok / float(baseline.tokens_total))
        return True, f"tokens -{gain:.1%}"

    if obj in ("open_model", "balanced"):
        cand_metrics = candidate.get("metrics") or {}
        base_metrics = baseline.metrics or {}
        for key in ("open_model_rate", "primary_metric"):
            if key in cand_metrics and key in base_metrics:
                delta = float(cand_metrics[key]) - float(base_metrics[key])
                if delta >= 0.01:
                    return True, f"{key} +{delta:.4f}"

    return False, "no accuracy or performance gain over baseline"


def _apply_sandbox_files(
    workspace: Path, sandbox_root: Path, touched: list, policy: ScopePolicy
) -> list[str]:
    """Atomically copy promoted sandbox files back into the workspace."""
    workspace = Path(workspace)
    rels = [str(rel) for rel in touched]
    if not rels:
        return []
    ok, errors = policy.validate_paths(rels, workspace)
    if not ok:
        raise PermissionError("; ".join(errors))
    applied: list[str] = []
    for rel in rels:
        source = Path(sandbox_root) / rel
        if not source.is_file():
            continue
        data = source.read_bytes()
        if len(data) > policy.max_file_bytes:
            raise ValueError(
                f"promoted file '{rel}' exceeds the {policy.max_file_bytes}-byte cap"
            )
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp_name, target)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        applied.append(rel)
    return applied


def _run_candidate(
    *,
    workspace: Path,
    spec: BenchmarkSpec,
    ledger: dict,
    policy: ScopePolicy,
    baseline: EvalReport,
    editable_contents: dict,
    sample: int,
    round_num: int,
    trace_id: str,
    round_span: str,
    policy_routes: Optional[list[str]],
    proposer_route: str,
    dry_run: bool,
    optimizer_fn: Optional[Callable[[str, str], str]],
    events_path: Path,
    allow_fallback: bool,
    sandboxes: dict,
    objective: str = "balanced",
    repeats: int = 1,
) -> dict:
    """Obtain, verify and score one candidate proposal."""
    sample_id = f"r{round_num}s{sample}"
    candidate: dict = {
        "sample": sample_id,
        "problem_id": "",
        "problem_name": "",
        "tier": "adaptable",
        "action": "new",
        "strategy": "",
        "what": "",
        "predicted": [],
        "edits": [],
        "fixed": [],
        "missed": [],
        "regressed": [],
        "passed": baseline.passed,
        "total": baseline.total,
        "score": None,
        "latency_ms": 0.0,
        "tokens_total": 0,
        "metrics": {},
        "results": {},
        "traces": {},
        "touched": [],
        "scope_ok": False,
        "error": "",
        "failure_class": "",
        "accepted": False,
        "recorded": False,
        "optimizer_model": "",
        "policy_models": list(baseline.policy_models),
    }

    proposer_prompt = build_proposer_prompt(
        ledger,
        baseline,
        editable_contents,
        round_num=round_num,
        domain=spec.domain,
        policy_models=list(baseline.policy_models),
        objective=objective,
        policy=policy,
        spec=spec,
    )
    # Held-out firewall: no model-facing string may carry held-out data.
    _assert_no_held_out_leak(proposer_prompt, spec)

    proposal: Optional[dict] = None
    raw_output = ""
    tokens_in = tokens_out = elapsed_ms = 0
    if dry_run:
        proposal = _dry_run_proposal(workspace, baseline, policy)
        raw_output = json.dumps(proposal, indent=2)
        candidate["optimizer_model"] = "dry-run"
        tokens_in = max(1, len(proposer_prompt) // 4)
        tokens_out = max(1, len(raw_output) // 4)
    elif optimizer_fn is not None:
        raw_output = str(optimizer_fn("proposer", proposer_prompt) or "")
        candidate["optimizer_model"] = "optimizer_fn"
        tokens_in = max(1, len(proposer_prompt) // 4)
        tokens_out = max(1, len(raw_output) // 4)
    else:
        result, dispatch_error = _execute_route_span(
            trace_id,
            round_span,
            workspace,
            proposer_route,
            proposer_prompt,
            events_path=events_path,
            allow_fallback=allow_fallback,
            intent_label="evolve:propose",
        )
        if result is None:
            candidate["failure_class"] = "backend_error"
            candidate["error"] = f"proposer dispatch failed: {dispatch_error}"
        else:
            raw_output = result.output
            candidate["optimizer_model"] = result.model
            tokens_in, tokens_out = result.tokens_in, result.tokens_out
            elapsed_ms = result.elapsed_ms
            if result.status != "DONE" and not raw_output.strip():
                candidate["failure_class"] = "backend_error"
                candidate["error"] = (
                    f"proposer backend {result.status}: {result.error or 'no output'}"
                )

    if proposal is None and not candidate["error"]:
        try:
            proposal = parse_json_object(raw_output)
        except ValueError as exc:
            repaired_raw = ""
            if raw_output.strip():
                repair_prompt = (
                    "Convert your previous response into ONLY a valid single JSON object matching: "
                    '{"problem_id":"new","problem_name":"...","tier":"adaptable","action":"new","what":"...","predicted":[],"edits":[{"path":"...","mode":"write","content":"..."}]}. '
                    f"Parse error was: {exc}\n\nOriginal output:\n{raw_output[:4000]}"
                )
                if optimizer_fn is not None:
                    repaired_raw = str(optimizer_fn("proposer_repair", repair_prompt) or "")
                    tokens_in += max(1, len(repair_prompt) // 4)
                    tokens_out += max(1, len(repaired_raw) // 4)
                else:
                    r_res, _r_err = _execute_route_span(
                        trace_id,
                        round_span,
                        workspace,
                        proposer_route,
                        repair_prompt,
                        events_path=events_path,
                        allow_fallback=allow_fallback,
                        intent_label="evolve:propose-repair",
                    )
                    if r_res is not None:
                        repaired_raw = r_res.output
                        tokens_in += r_res.tokens_in
                        tokens_out += r_res.tokens_out
                        elapsed_ms += r_res.elapsed_ms
            if repaired_raw.strip():
                try:
                    proposal = parse_json_object(repaired_raw)
                except ValueError as exc2:
                    candidate["failure_class"] = "parse_error"
                    candidate["error"] = f"proposal rejected: {type(exc2).__name__}: {exc2}"
            else:
                candidate["failure_class"] = "parse_error"
                candidate["error"] = f"proposal rejected: {type(exc).__name__}: {exc}"
    if proposal is not None and not candidate["error"]:
        try:
            _merge_proposal(candidate, proposal)
            _assert_no_proposal_leak(candidate, spec)
        except ValueError as exc:
            candidate["failure_class"] = "parse_error"
            candidate["error"] = f"proposal rejected: {type(exc).__name__}: {exc}"
            # A leaky proposal must leave no trace in the SIM ledger: drop
            # every field the proposal contributed before the fallback text.
            candidate["what"] = ""
            candidate["problem_name"] = ""
            candidate["predicted"] = []
            candidate["edits"] = []

    if not candidate["error"]:
        try:
            sandbox_root, touched, tempdir = apply_edits_in_sandbox(
                workspace, spec, candidate["edits"], policy
            )
        except PermissionError as exc:
            candidate["failure_class"] = "scope_violation"
            candidate["error"] = f"scope violation: {type(exc).__name__}: {exc}"
        except ValueError as exc:
            candidate["failure_class"] = "scope_violation"
            candidate["error"] = f"edits rejected: {type(exc).__name__}: {exc}"
        else:
            sandboxes[sample_id] = (sandbox_root, tempdir)
            candidate["touched"] = touched
            candidate["scope_ok"] = True
            effective_build = policy.build_cmd or tuple(spec.build_cmd or ())
            if effective_build:
                build_ok, build_err = _run_sandbox_build(
                    sandbox_root, effective_build, policy.build_timeout_s
                )
                if not build_ok:
                    candidate["error"] = build_err
            if not candidate["error"]:
                try:
                    cand_eval = evaluate_split(
                        sandbox_root,
                        spec,
                        split="evolve",
                        policy_routes=policy_routes,
                        repeats=repeats,
                    )
                except (OSError, ValueError, KeyError) as exc:
                    candidate["error"] = (
                        f"sandbox evaluation failed: {type(exc).__name__}: {exc}"
                    )
                else:
                    outcomes = sim.compute_task_outcomes(
                        baseline.results, cand_eval.results, candidate["predicted"]
                    )
                    candidate["fixed"] = outcomes["fixed"]
                    candidate["missed"] = outcomes["missed"]
                    candidate["regressed"] = outcomes["regressed"]
                    candidate["passed"] = outcomes["passed"]
                    candidate["total"] = outcomes["total"]
                    candidate["score"] = outcomes["score"]
                    candidate["latency_ms"] = cand_eval.latency_ms
                    candidate["tokens_total"] = cand_eval.tokens_total
                    candidate["metrics"] = dict(cand_eval.metrics)
                    candidate["results"] = dict(cand_eval.results)
                    candidate["traces"] = dict(cand_eval.traces)

    if candidate["error"]:
        # Directive D9: rejected proposals still enter the ledger with
        # fixed == [] and missed == predicted so the dead end is never retried.
        candidate["what"] = candidate["what"] or f"rejected candidate ({candidate['error']})"
        candidate["fixed"] = []
        candidate["missed"] = list(candidate["predicted"])
        candidate["regressed"] = []
        candidate["score"] = None

    emit_event(
        trace_id,
        new_span_id(),
        parent_id=round_span,
        route=proposer_route,
        intent="evolve:propose",
        backend="nexus",
        model=candidate["optimizer_model"] or spec.name,
        quota_tier="ORCHESTRATOR",
        status="FAIL" if candidate["error"] else "DONE",
        step=(
            f"sample {sample_id}: {candidate['error']}"
            if candidate["error"]
            else (
                f"sample {sample_id}: score {candidate['score']:.4f}, "
                f"fixed {len(candidate['fixed'])}, regressed {len(candidate['regressed'])}"
            )
        ),
        prompt=proposer_prompt,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        elapsed_ms=elapsed_ms,
        error=candidate["error"][:500],
        dry_run=dry_run,
        path=events_path,
    )
    return candidate


def run_evolve_round(
    workspace: Path,
    spec: BenchmarkSpec,
    ledger: dict,
    *,
    round_num: int = 1,
    proposer_route: str = "refactor",
    selector_route: str = "reason",
    policy_routes: Optional[list[str]] = None,
    candidates_k: int = 2,
    policy: Optional[ScopePolicy] = None,
    apply_changes: bool = False,
    dry_run: bool = False,
    optimizer_fn: Optional[Callable[[str, str], str]] = None,
    events_path: Path = DEFAULT_EVENTS_PATH,
    parent_trace_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    allow_fallback: bool = True,
    objective: str = "balanced",
    repeats: int = 1,
) -> dict:
    """Run one propose/verify/select round and return a summary dict.

    The baseline is always the ``evolve`` split (held-out tasks never enter
    the round).  Every candidate -- winning or rejected -- is recorded in the
    SIM ledger.  Promotion requires beating the baseline on accuracy or on
    the active performance ``objective`` (latency, token efficiency, or
    open-model solve rate) with zero regressions.
    """
    workspace = Path(workspace)
    obj = normalize_objective(objective)
    if policy is None:
        policy = ScopePolicy.load(
            extra_editable=list(spec.editable_files),
            extra_locked=list(spec.locked_files),
        )
    trace_id = parent_trace_id or new_trace_id()
    round_span = new_span_id()
    started = time.monotonic()

    emit_event(
        trace_id,
        round_span,
        parent_id=parent_span_id,
        route="evolve",
        intent="evolve:round",
        backend="nexus",
        model=spec.name,
        quota_tier="ORCHESTRATOR",
        status="RUNNING",
        step=f"round {round_num} start ({obj}) on benchmark '{spec.name}'",
        dry_run=dry_run,
        path=events_path,
    )

    sandboxes: dict = {}
    try:
        baseline = evaluate_split(
            workspace, spec, split="evolve", policy_routes=policy_routes, repeats=repeats
        )
        policy_models = list(baseline.policy_models)
        editable_contents = _read_editable_contents(workspace, policy)

        candidates: list[dict] = []
        for sample in range(1, max(0, int(candidates_k)) + 1):
            candidates.append(
                _run_candidate(
                    workspace=workspace,
                    spec=spec,
                    ledger=ledger,
                    policy=policy,
                    baseline=baseline,
                    editable_contents=editable_contents,
                    sample=sample,
                    round_num=round_num,
                    trace_id=trace_id,
                    round_span=round_span,
                    policy_routes=policy_routes,
                    proposer_route=proposer_route,
                    dry_run=dry_run,
                    optimizer_fn=optimizer_fn,
                    events_path=events_path,
                    allow_fallback=allow_fallback,
                    sandboxes=sandboxes,
                    objective=obj,
                    repeats=repeats,
                )
            )

        # --- selection ----------------------------------------------------
        valid = [
            cand
            for cand in candidates
            if cand["scope_ok"] and not cand["error"] and not cand["regressed"]
        ]
        winner: Optional[dict] = None
        selection_mode = "none"
        if len(valid) == 1:
            winner = valid[0]
            selection_mode = "single-valid-candidate"
        elif valid:
            selected_sample = ""
            rationale = ""
            selection_mode = "deterministic-fallback"
            if not dry_run:
                selector_prompt = build_selector_prompt(
                    ledger,
                    baseline,
                    valid,
                    round_num=round_num,
                    policy_models=policy_models,
                    objective=obj,
                )
                # Held-out firewall applies to the Selector prompt as well.
                _assert_no_held_out_leak(selector_prompt, spec)
                raw_selection = ""
                if optimizer_fn is not None:
                    raw_selection = str(optimizer_fn("selector", selector_prompt) or "")
                else:
                    result, _dispatch_error = _execute_route_span(
                        trace_id,
                        round_span,
                        workspace,
                        selector_route,
                        selector_prompt,
                        events_path=events_path,
                        allow_fallback=allow_fallback,
                        intent_label="evolve:select",
                    )
                    if result is not None:
                        raw_selection = result.output
                if raw_selection.strip():
                    try:
                        picked = parse_json_object(raw_selection)
                        selected_sample = str(picked.get("selected_sample") or "").strip()
                        rationale = str(picked.get("rationale") or "").strip()
                    except ValueError:
                        selected_sample = ""
            by_sample = {cand["sample"]: cand for cand in valid}
            if selected_sample and selected_sample in by_sample:
                winner = by_sample[selected_sample]
                selection_mode = "selector"
            else:
                winner = _best_candidate(valid, objective=obj)
                if selected_sample:
                    # Deterministic guard: never promote a regressing or
                    # scope-rejected candidate, whatever the Selector said.
                    selection_mode = "selector-guard"

        # --- promotion gate -----------------------------------------------
        promoted = False
        promotion_reason = "no eligible winner"
        if winner is not None and not dry_run:
            promoted, promotion_reason = beats_baseline(
                winner, baseline, objective=obj
            )
            winner["promotion_reason"] = promotion_reason

        # --- ledger: every candidate is logged, winner or not ---------------
        for candidate in candidates:
            accepted = (
                (not dry_run)
                and winner is not None
                and candidate["sample"] == winner["sample"]
            )
            _record_candidate(
                ledger,
                candidate,
                spec=spec,
                round_num=round_num,
                policy_models=policy_models,
                accepted=accepted,
            )
            candidate["accepted"] = bool(accepted)
            candidate["recorded"] = True

        # --- apply ----------------------------------------------------------
        applied = False
        if promoted and apply_changes and not dry_run and winner is not None:
            entry = sandboxes.get(winner["sample"])
            if entry is not None:
                sandbox_root, _tempdir = entry
                _apply_sandbox_files(workspace, sandbox_root, winner["touched"], policy)
                applied = True

        if valid:
            emit_event(
                trace_id,
                new_span_id(),
                parent_id=round_span,
                route=selector_route,
                intent="evolve:select",
                backend="nexus",
                model=spec.name,
                quota_tier="ORCHESTRATOR",
                status="DONE",
                step=(
                    f"selection mode {selection_mode}: winner "
                    f"{winner['sample'] if winner is not None else 'none'}"
                ),
                dry_run=dry_run,
                path=events_path,
            )

        summary = {
            "round": int(round_num),
            "trace_id": trace_id,
            "span_id": round_span,
            "dry_run": bool(dry_run),
            "objective": obj,
            "promotion_reason": promotion_reason,
            "baseline": {
                "split": baseline.split,
                "passed": baseline.passed,
                "total": baseline.total,
                "score": baseline.score,
                "latency_ms": baseline.latency_ms,
                "tokens_total": baseline.tokens_total,
                "metrics": dict(baseline.metrics),
                "failing": baseline.failing_ids(),
                "policy_models": policy_models,
                "results": dict(baseline.results),
                "traces": dict(baseline.traces),
            },
            "candidates": candidates,
            "winner": winner["sample"] if winner is not None else None,
            "promoted": bool(promoted),
            "applied": bool(applied),
            "selection_mode": selection_mode,
        }

        emit_event(
            trace_id,
            round_span,
            route="evolve",
            intent="evolve:round",
            backend="nexus",
            model=spec.name,
            quota_tier="ORCHESTRATOR",
            status="DONE",
            step=(
                f"round {round_num}: baseline {baseline.passed}/{baseline.total}, "
                f"winner {summary['winner']}, promoted {promoted}, applied {applied}"
            ),
            elapsed_ms=int((time.monotonic() - started) * 1000),
            dry_run=dry_run,
            path=events_path,
        )
        return summary
    except BaseException as exc:
        emit_event(
            trace_id,
            round_span,
            route="evolve",
            intent="evolve:round",
            backend="nexus",
            model=spec.name,
            quota_tier="ORCHESTRATOR",
            status="FAIL",
            step=f"round {round_num} failed",
            error=str(exc)[:500],
            elapsed_ms=int((time.monotonic() - started) * 1000),
            dry_run=dry_run,
            path=events_path,
        )
        raise
    finally:
        for _root, tempdir in sandboxes.values():
            tempdir.cleanup()


# ---------------------------------------------------------------------------
# Loop and cold-start challenger


def _max_recorded_round(ledger: dict) -> int:
    highest = 0
    for problem in ledger.get("problems") or []:
        if not isinstance(problem, dict):
            continue
        for attempt in problem.get("attempts") or []:
            if isinstance(attempt, dict):
                try:
                    highest = max(highest, int(attempt.get("round") or 0))
                except (TypeError, ValueError):
                    continue
    return highest


def run_evolve_loop(
    workspace: Path,
    spec: BenchmarkSpec,
    ledger: Optional[dict] = None,
    *,
    rounds: int = 1,
    start_round: Optional[int] = None,
    sim_path: Optional[Path] = None,
    candidates_k: int = 2,
    proposer_route: str = "refactor",
    selector_route: str = "reason",
    policy_routes: Optional[list[str]] = None,
    policy: Optional[ScopePolicy] = None,
    apply_changes: bool = False,
    dry_run: bool = False,
    optimizer_fn: Optional[Callable[[str, str], str]] = None,
    events_path: Path = DEFAULT_EVENTS_PATH,
    parent_trace_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    allow_fallback: bool = True,
    objective: str = "balanced",
    repeats: int = 1,
) -> dict:
    """Run ``rounds`` consecutive evolve rounds, saving the ledger each round.

    The first round starts at ``start_round`` when given, otherwise at one
    past the highest round already recorded in the ledger.
    """
    workspace = Path(workspace)
    obj = normalize_objective(objective)
    if ledger is None:
        if sim_path is not None and Path(sim_path).is_file():
            ledger = sim.load_sim(Path(sim_path))
        else:
            ledger = sim.new_ledger(domain=spec.domain, benchmark=spec.name)
    trace_id = parent_trace_id or new_trace_id()
    if start_round is None:
        start_round = _max_recorded_round(ledger) + 1
    start_round = max(1, int(start_round))
    count = max(0, int(rounds))

    summaries: list[dict] = []
    for offset in range(count):
        summary = run_evolve_round(
            workspace,
            spec,
            ledger,
            round_num=start_round + offset,
            proposer_route=proposer_route,
            selector_route=selector_route,
            policy_routes=policy_routes,
            candidates_k=candidates_k,
            policy=policy,
            apply_changes=apply_changes,
            dry_run=dry_run,
            optimizer_fn=optimizer_fn,
            events_path=events_path,
            parent_trace_id=trace_id,
            parent_span_id=parent_span_id,
            allow_fallback=allow_fallback,
            objective=obj,
            repeats=repeats,
        )
        summaries.append(summary)
        if sim_path is not None:
            sim.save_sim(ledger, Path(sim_path))

    return {
        "trace_id": trace_id,
        "workspace": str(workspace),
        "start_round": start_round,
        "rounds_requested": count,
        "rounds_run": len(summaries),
        "objective": obj,
        "rounds": summaries,
        "round_summaries": summaries,
        "promoted_rounds": [entry["round"] for entry in summaries if entry.get("promoted")],
        "applied_rounds": [entry["round"] for entry in summaries if entry.get("applied")],
        "ledger": ledger,
        "final_ledger": ledger,
    }


def _clone_workspace(source: Path, destination: Path) -> Path:
    source = Path(source)
    destination = Path(destination)
    if not source.is_dir():
        raise NotADirectoryError(f"workspace to clone is not a directory: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        symlinks=True,
        ignore=shutil.ignore_patterns(".git"),
    )
    return destination


def run_cold_start_challenger(
    workspace: Path,
    seed_dir: Path,
    spec: BenchmarkSpec,
    inherited_sim: dict,
    *,
    rounds: int = 2,
    policy_routes: Optional[list[str]] = None,
    proposer_route: str = "refactor",
    selector_route: str = "reason",
    optimizer_fn: Optional[Callable[[str, str], str]] = None,
    dry_run: bool = False,
    events_path: Path = DEFAULT_EVENTS_PATH,
    candidates_k: int = 2,
    policy: Optional[ScopePolicy] = None,
) -> dict:
    """Race the evolved workspace (continued) against a cold start (fresh).

    Both clones run the same evolve loop; only afterwards are they scored on
    the held-out split, which never entered either loop.
    """
    workspace = Path(workspace)
    seed_dir = Path(seed_dir)
    policy_label = ", ".join(str(route) for route in (policy_routes or [])) or "default"

    continued_sim = sim.carry_forward(inherited_sim, from_run="evolved", into_run="continued")
    fresh_sim = sim.new_ledger(
        domain=spec.domain, policy_model=policy_label, benchmark=spec.name
    )

    with tempfile.TemporaryDirectory(prefix="nexus-coldstart-continued-") as dir_a:
        with tempfile.TemporaryDirectory(prefix="nexus-coldstart-fresh-") as dir_b:
            continued_ws = _clone_workspace(workspace, Path(dir_a))
            fresh_ws = _clone_workspace(seed_dir, Path(dir_b))

            run_evolve_loop(
                continued_ws,
                spec,
                continued_sim,
                rounds=rounds,
                apply_changes=True,
                candidates_k=candidates_k,
                proposer_route=proposer_route,
                selector_route=selector_route,
                policy_routes=policy_routes,
                optimizer_fn=optimizer_fn,
                dry_run=dry_run,
                events_path=events_path,
                policy=policy,
            )
            run_evolve_loop(
                fresh_ws,
                spec,
                fresh_sim,
                rounds=rounds,
                apply_changes=True,
                candidates_k=candidates_k,
                proposer_route=proposer_route,
                selector_route=selector_route,
                policy_routes=policy_routes,
                optimizer_fn=optimizer_fn,
                dry_run=dry_run,
                events_path=events_path,
                policy=policy,
            )

            continued_eval = evaluate_split(
                continued_ws, spec, split="held_out", policy_routes=policy_routes
            )
            fresh_eval = evaluate_split(
                fresh_ws, spec, split="held_out", policy_routes=policy_routes
            )

    continued_score = continued_eval.score
    fresh_score = fresh_eval.score
    delta = continued_score - fresh_score
    if continued_score > fresh_score:
        winner = "continued"
        recommendation = (
            "recommendation: continue from the evolved workspace -- it generalizes "
            f"better on held-out tasks ({continued_score:.4f} vs {fresh_score:.4f})."
        )
    elif fresh_score > continued_score:
        winner = "fresh"
        recommendation = (
            "recommendation: restart from the seed -- the inherited SIM did not "
            f"transfer and the cold start generalizes better "
            f"({fresh_score:.4f} vs {continued_score:.4f})."
        )
    else:
        winner = "tie"
        recommendation = (
            f"recommendation: held-out scores are equal ({continued_score:.4f}); "
            "prefer the cheaper start."
        )

    return {
        "winner": winner,
        "continued_held_out": continued_score,
        "fresh_held_out": fresh_score,
        "delta": delta,
        "continued_sim": continued_sim,
        "fresh_sim": fresh_sim,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Passive ingrained improvement monitor


def detect_improvement_opportunities(
    events: list[dict],
    *,
    ledger: Optional[dict] = None,
    routes_data: Optional[dict] = None,
    errata: Optional[list[dict]] = None,
) -> list[dict]:
    """Scan telemetry and SIM dead ends for concrete optimization opportunities."""
    from nexus.entity import compute_performance_stats

    stats = compute_performance_stats(events, group_by="route")
    groups = stats.get("groups") or {}
    routes = (routes_data or {}).get("routes") if isinstance(routes_data, dict) else {}
    routes = routes if isinstance(routes, dict) else {}

    opportunities: list[dict] = []

    for route_name, metrics in groups.items():
        runs = int(metrics.get("runs") or 0)
        success_rate = float(metrics.get("success_rate") or 0.0)
        p95_ms = int(metrics.get("p95_ms") or 0)
        premium_spans = int(metrics.get("premium_spans") or 0)
        cfg = routes.get(route_name) if isinstance(routes, dict) else None
        tier = str((cfg or {}).get("quota_tier") or "").upper()
        is_open = tier in ("", "FREE", "LOCAL")

        if runs >= 2 and success_rate < 0.85 and is_open:
            opportunities.append(
                {
                    "id": f"route-reliability:{route_name}",
                    "priority": "high",
                    "lever": "prompt_tune|profile_tune",
                    "target": f"@{route_name}",
                    "evidence": f"{int(metrics.get('done') or 0)}/{runs} succeeded ({success_rate:.0%})",
                    "open_question": (
                        f"Why is @{route_name} failing ({int(metrics.get('done') or 0)}/{runs}), and can "
                        "prompt framing, free-tier failover, or recent OSS routing patterns raise it >=95%?"
                    ),
                    "recommendation": (
                        f"Squeeze @{route_name} with sharper persona framing, tighter fallbacks, "
                        "or ensemble cross-checks to lift open-model reliability."
                    ),
                    "command": "nexus evolve --scope prompts --objective open-model",
                }
            )
        if premium_spans > 0:
            opportunities.append(
                {
                    "id": f"premium-leakage:{route_name}",
                    "priority": "high",
                    "lever": "prompt_tune|profile_tune",
                    "target": f"@{route_name}",
                    "evidence": f"{premium_spans} premium span(s) in {runs} run(s)",
                    "recommendation": (
                        "Tune prompts, routing keywords, and free-only/local-only fallbacks so "
                        "open models handle these tasks without paid escalations."
                    ),
                    "command": "nexus evolve --scope prompts --objective open-model",
                }
            )
        if runs >= 2 and p95_ms >= 15000:
            opportunities.append(
                {
                    "id": f"latency-hotspot:{route_name}",
                    "priority": "medium",
                    "lever": "code_refactor|polyglot_rewrite",
                    "target": f"@{route_name}",
                    "evidence": f"p95 latency {p95_ms / 1000:.1f}s over {runs} run(s)",
                    "open_question": (
                        f"Is @{route_name}'s {p95_ms / 1000:.1f}s p95 caused by prompt bloat, queuing, or "
                        "Python hot paths solvable via compression or native/ Rust/C code?"
                    ),
                    "recommendation": (
                        "Profile slow paths (streaming, parsing, router dispatch) and consider "
                        "Python optimization or a compiled native/ implementation."
                    ),
                    "command": "nexus evolve --scope polyglot --objective speed",
                }
            )

    if isinstance(ledger, dict):
        for problem in ledger.get("problems") or []:
            if not isinstance(problem, dict):
                continue
            attempts = [a for a in (problem.get("attempts") or []) if isinstance(a, dict)]
            zero_fix = sum(1 for a in attempts if not a.get("fixed"))
            if zero_fix >= 2:
                pid = str(problem.get("id") or "P?")
                pname = str(problem.get("name") or pid)
                opportunities.append(
                    {
                        "id": f"sim-dead-end:{pid}",
                        "priority": "medium",
                        "lever": "code_refactor|polyglot_rewrite",
                        "target": pid,
                        "evidence": f"{zero_fix} zero-fix attempt(s) on '{pname}'",
                        "open_question": (
                            f"What false assumption caused {zero_fix} zero-fix retries on '{pname}', "
                            "and what recent AI/CS paper or OSS architecture solves it cleanly?"
                        ),
                        "recommendation": (
                            f"Treat {pid} as an unanswered question: check recent AI/CS papers and "
                            "existing OSS methods before refactoring harness or native/ code."
                        ),
                        "command": "nexus evolve --scope full --objective balanced",
                    }
                )

    if errata:
        by_route_err: dict[str, dict[str, int]] = {}
        for item in errata:
            if not isinstance(item, dict):
                continue
            rt = str(item.get("route") or "unknown").strip().lstrip("@") or "unknown"
            bucket = by_route_err.setdefault(rt, {"total": 0, "false_passes": 0})
            bucket["total"] += 1
            if item.get("false_pass"):
                bucket["false_passes"] += 1
        for rt, counts in by_route_err.items():
            if counts["false_passes"] >= 1 or counts["total"] >= 2:
                opportunities.append(
                    {
                        "id": f"errata-false-pass:{rt}",
                        "priority": "high",
                        "lever": "prompt_tune|profile_tune|code_refactor",
                        "target": f"@{rt}",
                        "evidence": (
                            f"{counts['total']} erratum(a) ({counts['false_passes']} false-pass) "
                            "captured on completed runs"
                        ),
                        "open_question": (
                            f"Why did @{rt} accumulate {counts['total']} user erratum(a) "
                            f"({counts['false_passes']} false-pass despite DONE), and what ISA/Assay "
                            "check catches this failure before exit?"
                        ),
                        "recommendation": (
                            "Inspect `nexus errata list --route "
                            f"{rt}`, encode a verifiable Assay stage check, and tighten persona/routing."
                        ),
                        "command": "nexus errata list",
                    }
                )

    priority_rank = {"high": 0, "medium": 1, "low": 2}
    opportunities.sort(key=lambda item: (priority_rank.get(str(item.get("priority")), 9), str(item.get("id"))))
    if not opportunities:
        opportunities.append(
            {
                "id": "proactive:open-model-squeeze",
                "priority": "low",
                "lever": "prompt_tune|code_refactor|polyglot_rewrite",
                "target": "all",
                "evidence": f"{int(stats.get('total_runs') or 0)} terminal span(s) healthy",
                "open_question": (
                    "Which recent AI/CS paper or OSS technique (compression, EWMA routing, or native/ hot path) "
                    "yields the biggest zero-regression open-model gain next?"
                ),
                "recommendation": (
                    "Check recent AI/CS papers and OSS methods, benchmark profiles head-to-head, "
                    "and probe prompt, compression, or native/ speedups with 0 regressions."
                ),
                "command": "nexus bench --profiles default,free-only,local-only --split held-out",
            }
        )
    return opportunities


def render_improvement_opportunities(
    opportunities: list[dict],
    *,
    color: bool = True,
    width: int = 96,
) -> str:
    """Render improvement opportunities in a width-safe wireframe panel."""
    from nexus.visualizer import BOLD, CYAN, DIM, GREEN, RED, RESET, YELLOW, render_panel, truncate_ansi

    target_w = max(72, min(116, int(width or 96)))
    lines: list[str] = []
    badge_color = {"high": RED, "medium": YELLOW, "low": CYAN}
    for opp in opportunities:
        prio = str(opp.get("priority") or "low").lower()
        prio_tag = f"[{prio.upper():<6}]"
        if color:
            prio_tag = f"{BOLD}{badge_color.get(prio, CYAN)}{prio_tag}{RESET}"
        target = str(opp.get("target") or "")
        evidence = str(opp.get("evidence") or "")
        lever = str(opp.get("lever") or "")
        header = f"{prio_tag} {target} · {evidence} · lever={lever}"
        lines.append(truncate_ansi(header, target_w - 4))
        lines.append(truncate_ansi(f"  why: {opp.get('recommendation', '')}", target_w - 4))
        if opp.get("open_question"):
            lines.append(truncate_ansi(f"  ask: {opp['open_question']}", target_w - 4))
        lines.append(
            truncate_ansi(
                f"  try: {GREEN if color else ''}{opp.get('command', '')}{RESET if color else ''}",
                target_w - 4,
            )
        )
    return render_panel(
        lines or ["(no improvement opportunities detected)"],
        title="NEXUS SELF-IMPROVEMENT SUGGESTIONS",
        footer=f"{len(opportunities)} suggestion(s) · prompt->profile->code->polyglot",
        width=target_w,
        border_color=DIM if color else "",
    )
