"""Tests for the CRSI outer-loop optimizer (nexus/evolve.py).

Zero network calls: every optimizer interaction uses ``optimizer_fn`` mocks,
``dry_run=True``, or a monkeypatched ``execute_route``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nexus.evolve as evolve_mod
from nexus import sim
from nexus.backends import ExecutionResult
from nexus.events import read_events, reconstruct_tree
from nexus.evolve import (
    CANONICAL_DIRECTIVES,
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_POLICY_PATH,
    TRANSFER_TIERS,
    BenchmarkSpec,
    EvalReport,
    ScopePolicy,
    apply_edits_in_sandbox,
    build_proposer_prompt,
    build_selector_prompt,
    evaluate_split,
    parse_json_object,
    run_cold_start_challenger,
    run_evolve_loop,
    run_evolve_round,
)
from nexus.router import resolve_route

IMMUTABLE_LOCKED = (
    "nexus/sim.py",
    "nexus/evolve.py",
    "nexus/drops.py",
    "config/sim_policy.json",
    "config/evals",
    "tests",
    ".git",
    ".env",
)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture()
def routes_data() -> dict:
    return json.loads((PROJECT_ROOT / "config" / "routes.json").read_text(encoding="utf-8"))


def _write_routes(workspace: Path, data: dict) -> None:
    config = workspace / "config"
    config.mkdir(parents=True, exist_ok=True)
    (config / "routes.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


@pytest.fixture()
def workspace(tmp_path: Path, routes_data: dict) -> Path:
    _write_routes(tmp_path, routes_data)
    return tmp_path


@pytest.fixture()
def broken_workspace(tmp_path: Path, routes_data: dict) -> Path:
    """Workspace whose triage route lost its keywords: t1 falls through to code."""
    broken = copy.deepcopy(routes_data)
    broken["routes"]["triage"]["keywords"] = []
    _write_routes(tmp_path, broken)
    return tmp_path


@pytest.fixture()
def mini_spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        name="mini-routing",
        domain="routing",
        editable_files=["config/routes.json"],
        locked_files=[],
        evolve_tasks=[
            {"id": "t1", "prompt": "triage the alerts", "expected_route": "triage"},
            {"id": "t2", "prompt": "fix the bug", "expected_route": "code"},
            {"id": "t3", "prompt": "quick snippet please", "expected_route": "fast-code"},
            {"id": "t4", "prompt": "@local stay offline", "expected_route": "local"},
        ],
        held_out_tasks=[
            {"id": "h1", "prompt": "audit security", "expected_route": "reason"},
            {"id": "h2", "prompt": "read the long document", "expected_route": "context"},
        ],
    )


@pytest.fixture()
def real_spec() -> BenchmarkSpec:
    return BenchmarkSpec.load(DEFAULT_BENCHMARK_PATH)


def _fix_proposal(routes_data: dict, **overrides) -> str:
    proposal = {
        "problem_id": "new",
        "problem_name": "triage keywords dropped",
        "tier": "adaptable",
        "action": "revise",
        "what": "restore triage keywords so triage prompts stop falling through to code",
        "predicted": ["t1"],
        "edits": [
            {"path": "config/routes.json", "mode": "write", "content": json.dumps(routes_data)}
        ],
    }
    proposal.update(overrides)
    return json.dumps(proposal)


def _all_attempts(ledger: dict) -> list[dict]:
    return [
        attempt
        for problem in ledger.get("problems") or []
        for attempt in problem.get("attempts") or []
    ]


# --------------------------------------------------------------------------- constants & policy file


def test_default_paths_and_constants():
    assert DEFAULT_POLICY_PATH == PROJECT_ROOT / "config" / "sim_policy.json"
    assert DEFAULT_BENCHMARK_PATH == PROJECT_ROOT / "config" / "evals" / "routing_benchmark.json"
    assert DEFAULT_POLICY_PATH.is_file()
    assert DEFAULT_BENCHMARK_PATH.is_file()
    assert len(CANONICAL_DIRECTIVES) == 10
    assert all(isinstance(d, str) and d.strip() for d in CANONICAL_DIRECTIVES)
    assert set(TRANSFER_TIERS) == {"invariant", "adaptable", "model_specific"}
    assert TRANSFER_TIERS == {k: v for k, v in TRANSFER_TIERS.items() if isinstance(v, str)}


def test_policy_file_matches_required_defaults():
    data = json.loads(DEFAULT_POLICY_PATH.read_text(encoding="utf-8"))
    assert data == {
        "editable_files": ["config/routes.json"],
        "locked_files": list(IMMUTABLE_LOCKED),
        "max_file_bytes": 524288,
    }


def test_scope_policy_load_defaults():
    policy = ScopePolicy.load()
    assert "config/routes.json" in policy.editable_files
    for locked in IMMUTABLE_LOCKED:
        assert locked in policy.locked_files
    assert policy.max_file_bytes == 524288


def test_scope_policy_locked_wins_over_extra_editable():
    policy = ScopePolicy.load(
        extra_editable=["nexus/evolve.py", "tests", "config/evals/x.json", ".git/config", "nexus/sim.py"]
    )
    for forbidden in ("nexus/evolve.py", "tests", "config/evals/x.json", ".git/config", "nexus/sim.py"):
        assert forbidden not in policy.editable_files
    for locked in IMMUTABLE_LOCKED:
        assert locked in policy.locked_files


def test_scope_policy_extras_and_custom_file(tmp_path: Path):
    custom = tmp_path / "policy.json"
    custom.write_text(
        json.dumps(
            {
                "editable_files": ["config/routes.json", "src/app.py"],
                "locked_files": ["src/secret.py"],
                "max_file_bytes": 1024,
            }
        ),
        encoding="utf-8",
    )
    policy = ScopePolicy.load(custom, extra_editable=["src/extra.py"], extra_locked=["vendor"])
    assert {"config/routes.json", "src/app.py", "src/extra.py"} <= set(policy.editable_files)
    assert {"src/secret.py", "vendor"} <= set(policy.locked_files)
    for locked in IMMUTABLE_LOCKED:
        assert locked in policy.locked_files
    assert policy.max_file_bytes == 1024


def test_scope_policy_missing_file_falls_back(tmp_path: Path):
    policy = ScopePolicy.load(tmp_path / "missing.json")
    assert policy.editable_files == ("config/routes.json",)
    assert set(IMMUTABLE_LOCKED) <= set(policy.locked_files)


# --------------------------------------------------------------------------- path validation


def test_validate_paths_accepts_editable(workspace: Path):
    policy = ScopePolicy.load()
    ok, errors = policy.validate_paths(["config/routes.json"], workspace)
    assert ok and errors == []


@pytest.mark.parametrize(
    "bad_path",
    [
        "",
        "   ",
        "/etc/passwd",
        "~/routes.json",
        "../escape.json",
        "config/../../escape.json",
        "nexus/sim.py",
        "nexus/evolve.py",
        "nexus/drops.py",
        "config/sim_policy.json",
        "config/evals/routing_benchmark.json",
        "tests/test_evolve.py",
        ".git/config",
        ".env",
        "nexus/router.py",
    ],
)
def test_validate_paths_rejects(workspace: Path, bad_path: str):
    policy = ScopePolicy.load()
    ok, errors = policy.validate_paths([bad_path], workspace)
    assert not ok
    assert errors


def test_validate_paths_rejects_symlink_target(workspace: Path, routes_data: dict):
    routes = workspace / "config" / "routes.json"
    real = workspace / "config" / "real.json"
    routes.unlink()
    real.write_text(json.dumps(routes_data), encoding="utf-8")
    routes.symlink_to(real)
    ok, errors = ScopePolicy.load().validate_paths(["config/routes.json"], workspace)
    assert not ok
    assert any("symlink" in e for e in errors)


def test_validate_paths_rejects_parent_symlink_escape(tmp_path: Path):
    workspace = tmp_path / "ws"
    (workspace / "config").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "routes.json").write_text("{}", encoding="utf-8")
    (workspace / "linkdir").symlink_to(outside, target_is_directory=True)
    ok, errors = ScopePolicy.load().validate_paths(["linkdir/routes.json"], workspace)
    assert not ok
    assert any("parent symlink escapes workspace" in e for e in errors)


def test_validate_paths_allows_parent_symlink_inside_workspace(tmp_path: Path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "routes.json").write_text("{}", encoding="utf-8")
    (tmp_path / "inner").symlink_to(tmp_path / "config", target_is_directory=True)
    policy = ScopePolicy.load(extra_editable=["inner/routes.json"])
    ok, errors = policy.validate_paths(["inner/routes.json"], tmp_path)
    assert ok, errors


# --------------------------------------------------------------------------- benchmark spec


def test_real_benchmark_shape(real_spec: BenchmarkSpec):
    assert real_spec.name == "nexus-routing-v1"
    assert real_spec.domain == "routing"
    assert real_spec.verifier == "builtin:routing"
    assert real_spec.editable_files == ["config/routes.json"]
    assert real_spec.locked_files == []
    assert len(real_spec.evolve_tasks) >= 14
    assert len(real_spec.held_out_tasks) >= 10
    evolve_ids = {t["id"] for t in real_spec.evolve_tasks}
    held_ids = {t["id"] for t in real_spec.held_out_tasks}
    assert evolve_ids.isdisjoint(held_ids)
    covered = {t["expected_route"] for t in real_spec.evolve_tasks + real_spec.held_out_tasks}
    assert {
        "triage", "vision", "context", "free-reason", "reason",
        "refactor", "fast-code", "code", "local", "ollama", "premium",
    } <= covered


def test_benchmark_overlap_raises(tmp_path: Path):
    task = {"id": "shared", "prompt": "fix the bug", "expected_route": "code"}
    path = tmp_path / "bench.json"
    path.write_text(
        json.dumps(
            {
                "name": "bad", "domain": "routing",
                "editable_files": [], "locked_files": [],
                "evolve_tasks": [task], "held_out_tasks": [dict(task)],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlap"):
        BenchmarkSpec.load(path)


def test_benchmark_duplicate_id_raises(tmp_path: Path):
    task = {"id": "dup", "prompt": "fix the bug", "expected_route": "code"}
    path = tmp_path / "bench.json"
    path.write_text(
        json.dumps(
            {
                "name": "bad", "domain": "routing",
                "editable_files": [], "locked_files": [],
                "evolve_tasks": [task, dict(task)],
                "held_out_tasks": [{"id": "other", "prompt": "audit security", "expected_route": "reason"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        BenchmarkSpec.load(path)


def test_real_benchmark_passes_on_repo_routes(real_spec: BenchmarkSpec):
    evolve = evaluate_split(PROJECT_ROOT, real_spec, split="evolve")
    held = evaluate_split(PROJECT_ROOT, real_spec, split="held_out")
    assert evolve.score == 1.0, evolve.traces
    assert held.score == 1.0, held.traces
    assert evolve.passed == evolve.total == len(real_spec.evolve_tasks)
    assert held.passed == held.total == len(real_spec.held_out_tasks)


def test_non_prefixed_prompts_never_hit_explicit_only_routes(real_spec: BenchmarkSpec):
    routes = json.loads((PROJECT_ROOT / "config" / "routes.json").read_text(encoding="utf-8"))
    for task in real_spec.evolve_tasks + real_spec.held_out_tasks:
        if task["prompt"].lstrip().startswith("@"):
            continue
        got, _cfg, _cleaned = resolve_route(task["prompt"], routes_data=routes)
        assert got not in {"local", "ollama", "premium"}, task["id"]


# --------------------------------------------------------------------------- evaluate_split


def test_evaluate_split_rejects_unknown_split(workspace: Path, mini_spec: BenchmarkSpec):
    with pytest.raises(ValueError):
        evaluate_split(workspace, mini_spec, split="validation")


def test_evaluate_split_failure_trace(broken_workspace: Path, mini_spec: BenchmarkSpec):
    report = evaluate_split(broken_workspace, mini_spec, split="evolve")
    assert report.results["t1"] is False
    assert report.results["t2"] is True
    assert report.traces["t1"] == "expected triage, got code"
    assert report.passed == 3 and report.total == 4
    assert report.score == pytest.approx(0.75)
    assert report.policy_models == ["default"]


def test_evaluate_split_pooling(workspace: Path, real_spec: BenchmarkSpec):
    pooled = evaluate_split(workspace, real_spec, split="evolve", policy_routes=["code", "reason"])
    expected_keys = {
        f"{route}::{task['id']}"
        for route in ("code", "reason")
        for task in real_spec.evolve_tasks
    }
    assert set(pooled.results) == expected_keys
    assert all(pooled.results.values())
    assert pooled.total == len(real_spec.evolve_tasks) * 2
    assert pooled.policy_models == ["code", "reason"]

    single = evaluate_split(workspace, real_spec, split="evolve", policy_routes=["code"])
    assert set(single.results) == {t["id"] for t in real_spec.evolve_tasks}
    plain = evaluate_split(workspace, real_spec, split="evolve")
    assert set(plain.results) == {t["id"] for t in real_spec.evolve_tasks}


def test_evaluate_split_pooling_invalid_arm(workspace: Path, mini_spec: BenchmarkSpec):
    report = evaluate_split(workspace, mini_spec, split="evolve", policy_routes=["code", "ghost"])
    assert report.results["code::t1"] is True
    assert report.results["ghost::t1"] is False
    assert report.traces["ghost::t1"] == "policy route 'ghost' missing or invalid"


def _command_spec(cmd: list[str]) -> BenchmarkSpec:
    return BenchmarkSpec(
        name="cmd-bench", domain="routing",
        editable_files=["config/routes.json"], locked_files=[],
        evolve_tasks=[
            {"id": "t1", "prompt": "p1", "expected_route": "code"},
            {"id": "t2", "prompt": "p2", "expected_route": "code"},
        ],
        held_out_tasks=[{"id": "h1", "prompt": "p3", "expected_route": "code"}],
        verifier="command", verifier_cmd=cmd,
    )


def test_evaluate_split_command_json(workspace: Path):
    script = (
        "import json, os\n"
        "assert os.environ['NEXUS_EVAL_SPLIT'] == 'evolve'\n"
        "assert os.environ['NEXUS_EVAL_WORKSPACE']\n"
        "print(json.dumps({'results': {'t1': True, 't2': False}, 'traces': {'t2': 'boom'}}))\n"
    )
    report = evaluate_split(workspace, _command_spec([sys.executable, "-c", script]), split="evolve")
    assert report.results == {"t1": True, "t2": False}
    assert report.traces["t2"] == "boom"
    assert report.passed == 1 and report.total == 2


def test_evaluate_split_command_exit_code_fallback(workspace: Path):
    ok = evaluate_split(workspace, _command_spec([sys.executable, "-c", "raise SystemExit(0)"]))
    assert ok.results == {"t1": True, "t2": True}
    bad = evaluate_split(
        workspace,
        _command_spec([sys.executable, "-c", "import sys; print('bad', file=sys.stderr); sys.exit(3)"]),
    )
    assert bad.results == {"t1": False, "t2": False}
    assert "bad" in bad.traces["t1"]


def test_evaluate_split_command_timeout(workspace: Path):
    spec = _command_spec([sys.executable, "-c", "import time; time.sleep(5)"])
    report = evaluate_split(workspace, spec, timeout_s=0.2)
    assert report.results == {"t1": False, "t2": False}
    assert "timed out" in report.traces["t1"]


# --------------------------------------------------------------------------- JSON parsing & prompts


def test_parse_json_object_variants():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure!\n```json\n{"a": {"b": 2}}\n```\nDone.') == {"a": {"b": 2}}
    assert parse_json_object('prefix {"a": 1} suffix') == {"a": 1}


@pytest.mark.parametrize("bad", ["", "   ", "no json here", "[1, 2]", '"text"', "```json\n[1]\n```"])
def test_parse_json_object_rejects(bad: str):
    with pytest.raises(ValueError):
        parse_json_object(bad)


def _evolve_baseline() -> EvalReport:
    return EvalReport(
        split="evolve",
        results={"t1": False, "t2": True},
        traces={"t1": "expected triage, got code"},
        passed=1, total=2, score=0.5, policy_models=["default"],
    )


def _ledger_with_history() -> dict:
    ledger = sim.new_ledger(domain="routing", benchmark="mini-routing")
    sim.open_problem(ledger, "triage fallthrough", opened_round=1, tier="adaptable")
    sim.record_attempt(
        ledger, problem_id="P1", round_num=1, sample="r1s1", action="revise",
        accepted=False, domain="routing", what="dead end: broad keyword stuffing",
        files=["config/routes.json"], predicted=["t1"], fixed=[], missed=["t1"],
        passed=1, total=2, policy_model="default", benchmark="mini-routing",
    )
    return ledger


def test_proposer_prompt_firewall():
    held = EvalReport(
        split="held_out", results={"h1": True}, traces={}, passed=1, total=1,
        score=1.0, policy_models=["default"],
    )
    with pytest.raises(ValueError):
        build_proposer_prompt(
            _ledger_with_history(), held, {}, round_num=1, domain="routing", policy_models=["default"]
        )


def test_proposer_prompt_content():
    ledger = _ledger_with_history()
    prompt = build_proposer_prompt(
        ledger, _evolve_baseline(), {"config/routes.json": '{"routes": {}}'},
        round_num=3, domain="routing", policy_models=["default"],
    )
    assert sim.format_sim_for_prompt(ledger) in prompt  # complete, untruncated
    for directive in CANONICAL_DIRECTIVES:
        assert directive in prompt
    for tier, desc in TRANSFER_TIERS.items():
        assert tier in prompt and desc in prompt
    assert "Round: 3" in prompt
    assert "t1" in prompt and "expected triage, got code" in prompt
    assert '{"routes": {}}' in prompt
    assert '"problem_id"' in prompt and '"predicted"' in prompt and '"edits"' in prompt
    assert "dead end: broad keyword stuffing" in prompt


def test_selector_prompt_firewall_and_content():
    held = EvalReport(
        split="held_out", results={"h1": True}, traces={}, passed=1, total=1,
        score=1.0, policy_models=["default"],
    )
    candidates = [
        {
            "sample": "r1s1", "action": "revise", "problem_id": "P1", "what": "fix",
            "predicted": ["t1"], "fixed": ["t1"], "missed": [], "regressed": [],
            "passed": 2, "total": 2, "score": 1.0, "error": "", "scope_ok": True,
        }
    ]
    with pytest.raises(ValueError):
        build_selector_prompt(_ledger_with_history(), held, candidates, round_num=1, policy_models=["default"])
    ledger = _ledger_with_history()
    prompt = build_selector_prompt(
        ledger, _evolve_baseline(), candidates, round_num=2, policy_models=["default"]
    )
    assert sim.format_sim_for_prompt(ledger) in prompt
    assert '"selected_sample"' in prompt
    assert "r1s1" in prompt
    for field in ("predicted", "fixed", "missed", "regressed", "passed", "total", "score"):
        assert f'"{field}"' in prompt
    assert "Never select a candidate with regressions" in prompt


# --------------------------------------------------------------------------- sandbox edits


def test_apply_edits_write_and_replace(workspace: Path, mini_spec: BenchmarkSpec, routes_data: dict):
    policy = ScopePolicy.load()
    shutil.copy2(DEFAULT_POLICY_PATH, workspace / "config" / "sim_policy.json")
    original = (workspace / "config" / "routes.json").read_text(encoding="utf-8")
    edits = [
        {"path": "config/routes.json", "mode": "replace",
         "search": '"default_route": "code"', "replace": '"default_route": "triage"'},
    ]
    sandbox, touched, tmp = apply_edits_in_sandbox(workspace, mini_spec, edits, policy)
    try:
        assert touched == ["config/routes.json"]
        content = (sandbox / "config" / "routes.json").read_text(encoding="utf-8")
        assert '"default_route": "triage"' in content
        assert (workspace / "config" / "routes.json").read_text(encoding="utf-8") == original
        assert (sandbox / "config" / "sim_policy.json").is_file()  # supporting config copied
    finally:
        tmp.cleanup()
    assert not sandbox.exists()

    sandbox2, touched2, tmp2 = apply_edits_in_sandbox(
        workspace, mini_spec,
        [{"path": "config/routes.json", "mode": "write", "content": json.dumps(routes_data)}],
        policy,
    )
    try:
        assert touched2 == ["config/routes.json"]
        assert json.loads((sandbox2 / "config" / "routes.json").read_text(encoding="utf-8")) == routes_data
    finally:
        tmp2.cleanup()


def test_apply_edits_locked_rejected_before_any_write(workspace: Path, mini_spec: BenchmarkSpec):
    policy = ScopePolicy.load()
    original = (workspace / "config" / "routes.json").read_text(encoding="utf-8")
    edits = [
        {"path": "config/routes.json", "mode": "write", "content": "{}"},
        {"path": "nexus/evolve.py", "mode": "write", "content": "# pwned"},
    ]
    with pytest.raises(PermissionError):
        apply_edits_in_sandbox(workspace, mini_spec, edits, policy)
    assert (workspace / "config" / "routes.json").read_text(encoding="utf-8") == original
    assert not (workspace / "nexus").exists()


def test_apply_edits_replace_search_missing(workspace: Path, mini_spec: BenchmarkSpec):
    policy = ScopePolicy.load()
    edits = [{"path": "config/routes.json", "mode": "replace", "search": "NO_SUCH_TEXT", "replace": "x"}]
    with pytest.raises(ValueError, match="search text not found"):
        apply_edits_in_sandbox(workspace, mini_spec, edits, policy)


def test_apply_edits_size_cap_and_bad_mode(workspace: Path, mini_spec: BenchmarkSpec):
    policy = ScopePolicy.load()
    tiny = ScopePolicy(
        editable_files=policy.editable_files, locked_files=policy.locked_files, max_file_bytes=10
    )
    with pytest.raises(ValueError, match="max_file_bytes"):
        apply_edits_in_sandbox(
            workspace, mini_spec,
            [{"path": "config/routes.json", "mode": "write", "content": "x" * 32}],
            tiny,
        )
    with pytest.raises(ValueError, match="unknown edit mode"):
        apply_edits_in_sandbox(
            workspace, mini_spec,
            [{"path": "config/routes.json", "mode": "append", "content": "x"}],
            policy,
        )


# --------------------------------------------------------------------------- evolve round


def test_round_dry_run_plumbing(workspace: Path, mini_spec: BenchmarkSpec, tmp_path: Path):
    ledger = sim.new_ledger(domain="routing", benchmark="mini-routing")
    events = tmp_path / "events.jsonl"
    original = (workspace / "config" / "routes.json").read_bytes()
    summary = run_evolve_round(
        workspace, mini_spec, ledger, round_num=1, dry_run=True,
        candidates_k=2, events_path=events,
    )
    assert set(summary) >= {"round", "baseline", "candidates", "winner", "promoted", "applied", "trace_id"}
    assert summary["promoted"] is False
    assert summary["applied"] is False
    assert len(summary["candidates"]) == 2
    assert summary["winner"] in {"r1s1", "r1s2"}
    attempts = _all_attempts(ledger)
    assert len(attempts) == 2
    assert all(a["accepted"] is False for a in attempts)
    assert sim.validate_sim(ledger)[0]
    assert (workspace / "config" / "routes.json").read_bytes() == original

    bus = read_events(events, trace_id=summary["trace_id"])
    intents = {e["intent"] for e in bus}
    assert {"evolve:round", "evolve:propose", "evolve:select"} <= intents
    assert all(e["quota_tier"] == "ORCHESTRATOR" for e in bus)
    tree = reconstruct_tree(bus, summary["trace_id"])
    assert tree["roots"] == [summary["span_id"]]


def test_round_dry_run_records_misses(broken_workspace: Path, mini_spec: BenchmarkSpec, tmp_path: Path):
    ledger = sim.new_ledger(domain="routing")
    summary = run_evolve_round(
        broken_workspace, mini_spec, ledger, dry_run=True,
        candidates_k=1, events_path=tmp_path / "events.jsonl",
    )
    attempt = _all_attempts(ledger)[0]
    assert attempt["predicted"] == ["t1"]
    assert attempt["missed"] == ["t1"]
    assert attempt["fixed"] == []
    assert summary["baseline"]["score"] == pytest.approx(0.75)


def test_round_optimizer_promotes_and_applies(
    broken_workspace: Path, mini_spec: BenchmarkSpec, routes_data: dict, tmp_path: Path
):
    ledger = sim.new_ledger(domain="routing", benchmark="mini-routing")
    calls: list[str] = []

    def optimizer(role: str, prompt: str) -> str:
        calls.append(role)
        assert "SIM LEDGER" in prompt
        return _fix_proposal(routes_data)

    summary = run_evolve_round(
        broken_workspace, mini_spec, ledger, round_num=1,
        optimizer_fn=optimizer, candidates_k=1, apply_changes=True,
        events_path=tmp_path / "events.jsonl",
    )
    assert calls == ["proposer"]  # single valid candidate: no selector call
    assert summary["promoted"] is True
    assert summary["applied"] is True
    assert summary["winner"] == "r1s1"
    cand = summary["candidates"][0]
    assert cand["fixed"] == ["t1"]
    assert cand["missed"] == []
    assert cand["regressed"] == []
    assert cand["score"] == 1.0
    assert summary["baseline"]["score"] == pytest.approx(0.75)

    applied = json.loads((broken_workspace / "config" / "routes.json").read_text(encoding="utf-8"))
    assert applied == routes_data
    assert evaluate_split(broken_workspace, mini_spec, split="evolve").score == 1.0

    attempt = _all_attempts(ledger)[0]
    assert attempt["accepted"] is True
    assert attempt["action"] == "revise"
    assert attempt["fixed"] == ["t1"]
    problem = ledger["problems"][0]
    assert problem["name"] == "triage keywords dropped"
    assert problem["tier"] == "adaptable"


def test_round_scope_violation_logged_as_rejected(
    broken_workspace: Path, mini_spec: BenchmarkSpec, tmp_path: Path
):
    ledger = sim.new_ledger(domain="routing")
    proposal = json.loads(_fix_proposal({}))
    proposal["edits"] = [{"path": "nexus/evolve.py", "mode": "write", "content": "# pwned"}]
    original = (broken_workspace / "config" / "routes.json").read_bytes()
    summary = run_evolve_round(
        broken_workspace, mini_spec, ledger, optimizer_fn=lambda role, prompt: json.dumps(proposal),
        candidates_k=1, apply_changes=True, events_path=tmp_path / "events.jsonl",
    )
    assert summary["promoted"] is False
    assert summary["applied"] is False
    cand = summary["candidates"][0]
    assert "PermissionError" in cand["error"]
    assert cand["scope_ok"] is False
    attempt = _all_attempts(ledger)[0]
    assert attempt["accepted"] is False
    assert attempt["fixed"] == []
    assert attempt["missed"] == ["t1"]
    assert (broken_workspace / "config" / "routes.json").read_bytes() == original


def test_round_malformed_proposal_logged(broken_workspace: Path, mini_spec: BenchmarkSpec, tmp_path: Path):
    ledger = sim.new_ledger(domain="routing")
    summary = run_evolve_round(
        broken_workspace, mini_spec, ledger,
        optimizer_fn=lambda role, prompt: "I refuse to answer in JSON",
        candidates_k=1, events_path=tmp_path / "events.jsonl",
    )
    assert summary["promoted"] is False
    assert summary["winner"] is None
    attempt = _all_attempts(ledger)[0]
    assert attempt["accepted"] is False
    assert "ValueError" in attempt["what"]
    assert sim.validate_sim(ledger)[0]


def test_round_selector_choice_and_guard(
    broken_workspace: Path, mini_spec: BenchmarkSpec, routes_data: dict, tmp_path: Path
):
    ledger = sim.new_ledger(domain="routing")
    roles: list[str] = []

    def optimizer(role: str, prompt: str) -> str:
        roles.append(role)
        if role == "proposer":
            return _fix_proposal(routes_data)
        return json.dumps({"selected_sample": "r1s2", "rationale": "deterministic tie-break"})

    summary = run_evolve_round(
        broken_workspace, mini_spec, ledger, optimizer_fn=optimizer,
        candidates_k=2, events_path=tmp_path / "events.jsonl",
    )
    assert roles == ["proposer", "proposer", "selector"]
    assert summary["winner"] == "r1s2"
    assert summary["promoted"] is True
    accepted = [a for a in _all_attempts(ledger) if a["accepted"]]
    assert len(accepted) == 1
    assert accepted[0]["sample"] == "r1s2"


def test_round_selector_invalid_pick_falls_back(
    broken_workspace: Path, mini_spec: BenchmarkSpec, routes_data: dict, tmp_path: Path
):
    ledger = sim.new_ledger(domain="routing")

    def optimizer(role: str, prompt: str) -> str:
        if role == "proposer":
            return _fix_proposal(routes_data)
        return json.dumps({"selected_sample": "ghost", "rationale": "bad pick"})

    summary = run_evolve_round(
        broken_workspace, mini_spec, ledger, optimizer_fn=optimizer,
        candidates_k=2, events_path=tmp_path / "events.jsonl",
    )
    assert summary["winner"] == "r1s1"  # deterministic best-score tie-break
    assert summary["promoted"] is True


def test_round_regressing_candidate_never_promoted(
    broken_workspace: Path, mini_spec: BenchmarkSpec, routes_data: dict, tmp_path: Path
):
    ledger = sim.new_ledger(domain="routing")
    regressing = copy.deepcopy(routes_data)
    regressing["routes"]["fast-code"]["keywords"] = []  # fixes t1, breaks t3
    calls = {"n": 0}

    def optimizer(role: str, prompt: str) -> str:
        if role == "proposer":
            calls["n"] += 1
            if calls["n"] == 3:
                return _fix_proposal(regressing)
            return _fix_proposal(routes_data)
        # Selector tries to promote the regressing candidate; the guard must refuse.
        return json.dumps({"selected_sample": "r1s3", "rationale": "tempting but regressing"})

    summary = run_evolve_round(
        broken_workspace, mini_spec, ledger, optimizer_fn=optimizer,
        candidates_k=3, events_path=tmp_path / "events.jsonl",
    )
    assert summary["winner"] in {"r1s1", "r1s2"}
    assert summary["promoted"] is True
    regressed_cand = next(c for c in summary["candidates"] if c["sample"] == "r1s3")
    assert regressed_cand["regressed"] == ["t3"]
    assert regressed_cand["accepted"] is False
    attempts = {a["sample"]: a for a in _all_attempts(ledger)}
    assert attempts["r1s3"]["regressed"] == ["t3"]
    assert attempts["r1s3"]["accepted"] is False


def test_round_single_regressing_candidate_no_winner(
    broken_workspace: Path, mini_spec: BenchmarkSpec, routes_data: dict, tmp_path: Path
):
    ledger = sim.new_ledger(domain="routing")
    regressing = copy.deepcopy(routes_data)
    regressing["routes"]["fast-code"]["keywords"] = []
    summary = run_evolve_round(
        broken_workspace, mini_spec, ledger,
        optimizer_fn=lambda role, prompt: _fix_proposal(regressing),
        candidates_k=1, apply_changes=True, events_path=tmp_path / "events.jsonl",
    )
    assert summary["winner"] is None
    assert summary["promoted"] is False
    assert summary["applied"] is False
    attempt = _all_attempts(ledger)[0]
    assert attempt["accepted"] is False
    assert attempt["regressed"] == ["t3"]


def test_round_backend_wiring(
    broken_workspace: Path, mini_spec: BenchmarkSpec, routes_data: dict, tmp_path: Path, monkeypatch
):
    ledger = sim.new_ledger(domain="routing")
    events = tmp_path / "events.jsonl"
    captured: dict = {}

    def fake_execute(trace_id, route_key, route_cfg, prompt, **kwargs):
        captured.update(kwargs)
        captured["route"] = route_key
        captured["cfg"] = route_cfg
        return ExecutionResult(
            trace_id=trace_id, span_id=kwargs.get("span_id") or "sp_fake",
            route=route_key, backend="opencode", model="mock/model", quota_tier="GO",
            status="DONE", output=_fix_proposal(routes_data), tokens_in=10, tokens_out=20,
        )

    monkeypatch.setattr(evolve_mod, "execute_route", fake_execute)
    summary = run_evolve_round(
        broken_workspace, mini_spec, ledger, candidates_k=1,
        events_path=events, allow_fallback=False,
    )
    assert captured["route"] == "refactor"
    assert captured["cfg"]["intent"] == "evolve:propose"
    assert captured["cfg"]["quota_tier"] == "ORCHESTRATOR"
    assert captured["allow_fallback"] is False
    assert captured["inject_square"] is False
    assert captured["emit_stderr"] is False
    assert captured["events_path"] == events
    assert summary["promoted"] is True
    assert summary["candidates"][0]["optimizer_model"] == "mock/model"


# --------------------------------------------------------------------------- firewall scanner & CLI contract


def test_eval_report_failing_ids():
    report = EvalReport(
        split="evolve",
        results={"a": True, "b": False, "c": False},
        traces={}, passed=1, total=3, score=1 / 3, policy_models=["default"],
    )
    assert report.failing_ids() == ["b", "c"]


def test_round_firewall_leak_scanner(workspace: Path, mini_spec: BenchmarkSpec, tmp_path: Path):
    ledger = sim.new_ledger(domain="routing")
    sim.open_problem(ledger, "contaminated", opened_round=1)
    sim.record_attempt(
        ledger, problem_id="P1", round_num=1, sample="r1s1", action="new",
        accepted=False, domain="routing", what="leaked reference to h1 outcome",
        files=[], predicted=[], fixed=[], missed=[], passed=0, total=1,
        policy_model="default",
    )
    events = tmp_path / "events.jsonl"
    with pytest.raises(ValueError, match="held-out firewall breach"):
        run_evolve_round(
            workspace, mini_spec, ledger, dry_run=True, candidates_k=1, events_path=events,
        )
    assert len(_all_attempts(ledger)) == 1  # nothing new logged
    bus = read_events(events)
    assert any(e["intent"] == "evolve:round" and e["status"] == "FAIL" for e in bus)


def test_round_proposal_leak_rejected_without_sim_contamination(
    workspace: Path, mini_spec: BenchmarkSpec, tmp_path: Path
):
    ledger = sim.new_ledger(domain="routing")
    leaky = json.dumps(
        {
            "problem_id": "new",
            "problem_name": "leak",
            "tier": "adaptable",
            "action": "new",
            "what": "optimize for audit security held-out flow",
            "predicted": [],
            "edits": [],
        }
    )
    summary = run_evolve_round(
        workspace, mini_spec, ledger, optimizer_fn=lambda role, prompt: leaky,
        candidates_k=1, events_path=tmp_path / "events.jsonl",
    )
    assert summary["promoted"] is False
    cand = summary["candidates"][0]
    assert "firewall" in cand["error"]
    attempt = _all_attempts(ledger)[0]
    assert attempt["accepted"] is False
    assert "audit security" not in attempt["what"]
    assert "h1" not in json.dumps(ledger)
    assert "h2" not in json.dumps(ledger)


def test_loop_cli_contract_keys(workspace: Path, mini_spec: BenchmarkSpec, tmp_path: Path):
    loop = run_evolve_loop(
        workspace, mini_spec, rounds=2, dry_run=True, candidates_k=1,
        sim_path=tmp_path / "sim.json", events_path=tmp_path / "events.jsonl",
    )
    assert loop["final_ledger"] is loop["ledger"]
    assert loop["round_summaries"] == loop["rounds"]
    assert loop["promoted_rounds"] == []
    assert loop["applied_rounds"] == []
    assert loop["trace_id"] == loop["rounds"][-1]["trace_id"]
    assert loop["rounds_run"] == 2


# --------------------------------------------------------------------------- loop & cold start


def test_loop_saves_sim_and_continues_rounds(workspace: Path, mini_spec: BenchmarkSpec, tmp_path: Path):
    sim_path = tmp_path / "sim.json"
    events = tmp_path / "events.jsonl"
    loop = run_evolve_loop(
        workspace, mini_spec, rounds=2, sim_path=sim_path,
        dry_run=True, candidates_k=1, events_path=events,
    )
    assert loop["start_round"] == 1
    assert loop["rounds_run"] == 2
    assert sim_path.is_file()
    ledger = sim.load_sim(sim_path)
    assert {a["round"] for a in _all_attempts(ledger)} == {1, 2}

    loop2 = run_evolve_loop(
        workspace, mini_spec, rounds=1, sim_path=sim_path,
        dry_run=True, candidates_k=1, events_path=events,
    )
    assert loop2["start_round"] == 3
    ledger2 = sim.load_sim(sim_path)
    assert {a["round"] for a in _all_attempts(ledger2)} == {1, 2, 3}


def _seed_workspace(path: Path, routes_data: dict, *, break_reason: bool = False) -> Path:
    data = copy.deepcopy(routes_data)
    if break_reason:
        data["routes"]["reason"]["keywords"] = []
    _write_routes(path, data)
    return path


def test_cold_start_challenger_continued_wins(tmp_path: Path, mini_spec: BenchmarkSpec, routes_data: dict):
    evolved = _seed_workspace(tmp_path / "evolved", routes_data)
    seed = _seed_workspace(tmp_path / "seed", routes_data, break_reason=True)
    inherited = sim.new_ledger(domain="routing", benchmark="mini-routing")
    sim.open_problem(inherited, "legacy issue", opened_round=1)
    sim.record_attempt(
        inherited, problem_id="P1", round_num=1, sample="r1s1", action="new",
        accepted=False, domain="routing", what="historical dead end",
        files=["config/routes.json"], predicted=["t1"], fixed=[], missed=["t1"],
        passed=0, total=1, policy_model="default",
    )
    result = run_cold_start_challenger(
        evolved, seed, mini_spec, inherited, rounds=1, dry_run=True,
        events_path=tmp_path / "events.jsonl",
    )
    assert result["winner"] == "continued"
    assert result["continued_held_out"] == 1.0
    assert result["fresh_held_out"] == 0.5
    assert result["delta"] == pytest.approx(0.5)
    assert result["continued_sim"]["carried_from"]["into"] == "continued"
    assert result["fresh_sim"]["carried_from"] is None
    assert "continue" in result["recommendation"]

    # Held-out firewall: no held-out task id ever entered either evolve ledger.
    held_ids = {t["id"] for t in mini_spec.held_out_tasks}
    for ledger in (result["continued_sim"], result["fresh_sim"]):
        for attempt in _all_attempts(ledger):
            touched = set(attempt["predicted"]) | set(attempt["fixed"]) | set(attempt["missed"]) | set(attempt["regressed"])
            assert not touched & held_ids
    # Inherited history survived the carry-forward.
    assert any(a["what"] == "historical dead end" for a in _all_attempts(result["continued_sim"]))


def test_cold_start_challenger_tie(tmp_path: Path, mini_spec: BenchmarkSpec, routes_data: dict):
    evolved = _seed_workspace(tmp_path / "evolved", routes_data)
    seed = _seed_workspace(tmp_path / "seed", routes_data)
    result = run_cold_start_challenger(
        evolved, seed, mini_spec, sim.new_ledger(domain="routing"),
        rounds=1, dry_run=True, events_path=tmp_path / "events.jsonl",
    )
    assert result["winner"] == "tie"
    assert result["delta"] == 0.0
    assert result["continued_held_out"] == result["fresh_held_out"] == 1.0


# --------------------------------------------------------------------------- hygiene


def test_no_personal_strings_in_owned_files():
    needles = ["/" + "home/", "acid" + "burn", "Sync/" + "antigravity", "100." + "102."]
    for rel in ("nexus/evolve.py", "config/sim_policy.json", "config/evals/routing_benchmark.json"):
        text = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
        for needle in needles:
            assert needle not in text, f"{rel} leaks personal string {needle!r}"