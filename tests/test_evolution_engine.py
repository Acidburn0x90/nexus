"""Tests for E2 Pareto archive, E3 GEPA trajectory feedback, E4 sandbox egress/integrity, and E6 explorer split."""

from __future__ import annotations

from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
import json
from pathlib import Path
import threading
import time

import pytest

from nexus import sim
from nexus.errata import record_erratum
from nexus.events import emit_event
from nexus.evolve import (
    IMMUTABLE_LOCKED_FILES,
    BenchmarkSpec,
    EvalReport,
    ScopePolicy,
    apply_edits_in_sandbox,
    build_proposer_prompt,
    build_sandbox_env,
    build_selector_prompt,
    collect_trajectory_feedback,
    compute_pareto_front,
    load_pareto_archive,
    resolve_explorer_routes,
    run_evolve_round,
    update_pareto_archive,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_cli(name: str = "nexus_cli_evo_test"):
    loader = SourceFileLoader(name, str(PROJECT_ROOT / "bin" / "nexus"))
    spec = spec_from_loader(name, loader)
    assert spec is not None
    mod = module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _make_spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        name="test-bench",
        domain="routing",
        editable_files=["config/routes.json"],
        locked_files=[],
        evolve_tasks=[
            {"id": "e1", "prompt": "review code bug", "expected_route": "code"},
            {"id": "e2", "prompt": "summarize logs", "expected_route": "triage"},
        ],
        held_out_tasks=[
            {"id": "h1", "prompt": "unique_held_out_secret_prompt_alpha", "expected_route": "reason"},
        ],
    )


def _seed_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    (ws / "config").mkdir(parents=True, exist_ok=True)
    routes = {
        "default_route": "code",
        "routes": {
            "omni": {"backend": "openai-compat", "model": "openrouter/openrouter/free", "quota_tier": "FREE", "keywords": ["ok"]},
            "fast-code": {"backend": "opencode", "model": "opencode/mimo-v2.6-flash-free", "quota_tier": "FREE", "keywords": ["snippet"]},
            "code": {"backend": "opencode", "model": "opencode/deepseek-v4.1-flash-free", "quota_tier": "FREE", "keywords": ["code", "bug"]},
            "triage": {"backend": "opencode", "model": "opencode/nemotron-3.5-lightning-free", "quota_tier": "FREE", "keywords": ["summarize", "logs"]},
            "refactor": {"backend": "opencode", "model": "opencode-go/kimi-k3.1-pro", "quota_tier": "GO", "keywords": ["refactor"]},
            "reason": {"backend": "opencode", "model": "opencode-go/glm-5.3", "quota_tier": "GO", "keywords": ["prove"]},
        },
        "precedence": ["triage", "code", "fast-code", "refactor", "reason", "omni"],
    }
    (ws / "config" / "routes.json").write_text(json.dumps(routes, indent=2), encoding="utf-8")
    return ws


# ---------------------------------------------------------------------------
# E4: Sandbox egress & trajectory integrity guard


def test_sandbox_env_strips_secrets_and_blocks_egress(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-key-placeholder")
    monkeypatch.setenv("NEXUS_TOKEN_SECRET", "tok-xyz")
    monkeypatch.setenv("SAFE_PATH_VAR", "/usr/bin")

    env = build_sandbox_env(tmp_path, extra={"NEXUS_EVAL_SPLIT": "evolve"})
    assert "OPENROUTER_API_KEY" not in env
    assert "NEXUS_TOKEN_SECRET" not in env
    assert env["SAFE_PATH_VAR"] == "/usr/bin"
    assert env["NEXUS_SANDBOX"] == "1"
    assert env["NEXUS_EVAL_WORKSPACE"] == str(tmp_path)
    assert env["NEXUS_EVAL_SPLIT"] == "evolve"
    assert env["HTTP_PROXY"] == "http://127.0.0.1:9"
    assert env["HTTPS_PROXY"] == "http://127.0.0.1:9"
    assert env["PIP_NO_INDEX"] == "1"


def test_immutable_lock_includes_lifeos_modules_and_integrity_guard(tmp_path: Path) -> None:
    for required in ("nexus/assay.py", "nexus/errata.py", "nexus/contract.py"):
        assert required in IMMUTABLE_LOCKED_FILES

    ws = _seed_workspace(tmp_path)
    spec = _make_spec()
    policy = ScopePolicy.load(PROJECT_ROOT / "config" / "sim_policy.json")

    with pytest.raises(PermissionError, match="held-out"):
        apply_edits_in_sandbox(
            ws,
            spec,
            [{"path": "config/routes.json", "mode": "write", "content": "unique_held_out_secret_prompt_alpha"}],
            policy,
        )

    with pytest.raises(PermissionError, match="bypass"):
        apply_edits_in_sandbox(
            ws,
            spec,
            [{"path": "config/routes.json", "mode": "write", "content": '{"note": "IMMUTABLE_LOCKED_FILES"}'}],
            policy,
        )


# ---------------------------------------------------------------------------
# E2: Pareto multi-objective front & archive


def test_pareto_front_and_atomic_archive_respects_save_flag(tmp_path: Path) -> None:
    candidates = [
        {"sample": "s1", "scope_ok": True, "error": "", "regressed": [], "score": 0.9, "latency_ms": 200.0, "tokens_total": 1000, "metrics": {"open_model_rate": 0.8}, "what": "fast open", "touched": ["config/routes.json"]},
        {"sample": "s2", "scope_ok": True, "error": "", "regressed": [], "score": 0.9, "latency_ms": 50.0, "tokens_total": 900, "metrics": {"open_model_rate": 0.8}, "what": "dominates s1", "touched": ["config/routes.json"]},
        {"sample": "s3", "scope_ok": True, "error": "", "regressed": [], "score": 0.8, "latency_ms": 20.0, "tokens_total": 300, "metrics": {"open_model_rate": 1.0}, "what": "fastest 100% open", "touched": ["config/routes.json"]},
        {"sample": "s4", "scope_ok": False, "error": "bad", "regressed": [], "score": 1.0, "latency_ms": 1.0, "tokens_total": 10, "metrics": {"open_model_rate": 1.0}},
    ]
    front = compute_pareto_front(candidates)
    front_ids = [c["sample"] for c in front]
    assert "s2" in front_ids
    assert "s3" in front_ids
    assert "s1" not in front_ids
    assert "s4" not in front_ids

    archive_path = tmp_path / "pareto_archive.json"
    update_pareto_archive(candidates, round_num=1, benchmark="test-bench", objective="balanced", path=archive_path, save=False)
    assert not archive_path.exists()

    elites = update_pareto_archive(candidates, round_num=1, benchmark="test-bench", objective="balanced", path=archive_path, save=True)
    assert archive_path.is_file()
    loaded = load_pareto_archive(archive_path)
    assert len(loaded["elites"]) == len(elites) >= 2


# ---------------------------------------------------------------------------
# E3: Trajectory-feedback prompting (GEPA)


def test_collect_trajectory_feedback_and_prompt_injection(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    errata_path = tmp_path / "errata.jsonl"

    emit_event("tr1", "sp1", route="code", backend="opencode", model="m-free", quota_tier="FREE", status="DONE", prompt="PRIVATE PROMPT", elapsed_ms=1200, tokens_in=500, tokens_out=50, path=events_path)
    emit_event("tr2", "sp2", route="refactor", backend="opencode", model="m-go", quota_tier="GO", status="FAIL", error="timeout expired", prompt="SECRET", elapsed_ms=1000, path=events_path)
    emit_event("tr3", "sp3", route="code", backend="opencode", model="m-free", quota_tier="FREE", status="DONE", dry_run=True, elapsed_ms=90000, path=events_path)
    record_erratum("misrouted regex edge", trace_id="tr1", events_path=events_path, path=errata_path)

    feedback = collect_trajectory_feedback(events_path=events_path, errata_path=errata_path)
    assert any(item["source"] == "erratum:false_pass" and item["route"] == "code" for item in feedback)
    assert any(item["source"] == "span:FAIL" and item["route"] == "refactor" for item in feedback)
    serialized = json.dumps(feedback)
    assert "PRIVATE PROMPT" not in serialized
    assert "tr3" not in serialized

    spec = _make_spec()
    baseline = EvalReport(split="evolve", results={"e1": True, "e2": False}, traces={"e2": "mismatch"}, passed=1, total=2, score=0.5, policy_models=["code"])
    ledger = sim.new_ledger(domain="routing", benchmark="test-bench")
    prompt = build_proposer_prompt(
        ledger,
        baseline,
        {"config/routes.json": "{}"},
        round_num=1,
        domain="routing",
        policy_models=["code"],
        trajectory_feedback=feedback,
        pareto_elites=[{"sample": "r1s1", "score": 0.9, "open_model_rate": 1.0, "latency_ms": 12.0}],
    )
    assert "## Live trajectory feedback (GEPA signals)" in prompt
    assert "## Pareto archive elites" in prompt
    sel_prompt = build_selector_prompt(
        ledger,
        baseline,
        [{"sample": "r1s1", "passed": 2, "total": 2, "score": 1.0, "latency_ms": 10.0, "tokens_total": 100}],
        round_num=1,
        policy_models=["code"],
        pareto_front=[{"sample": "r1s1", "score": 1.0}],
    )
    assert "Pareto non-dominated samples" in sel_prompt


# ---------------------------------------------------------------------------
# E6: Free-explorer + reasoning-synthesizer split & CLI flags


def test_resolve_explorer_routes_and_parallel_split_round(monkeypatch, tmp_path: Path) -> None:
    ws = _seed_workspace(tmp_path)
    routes_data = json.loads((ws / "config" / "routes.json").read_text(encoding="utf-8"))
    explorers = resolve_explorer_routes(routes_data)
    assert explorers == ["omni", "fast-code", "code", "triage"] or set(explorers) >= {"omni", "fast-code", "code"}
    assert explorers[0] == "omni"

    spec = _make_spec()
    ledger = sim.new_ledger(domain="routing", benchmark="test-bench")
    policy = ScopePolicy.load(PROJECT_ROOT / "config" / "sim_policy.json")
    seen_routes: list[str] = []
    lock = threading.Lock()

    class _FakeRes:
        def __init__(self, output: str, route: str) -> None:
            self.output = output
            self.model = f"model:{route}"
            self.tokens_in = 120
            self.tokens_out = 40
            self.elapsed_ms = 50
            self.status = "DONE"
            self.error = ""

    def fake_execute(trace_id, route_key, cfg, prompt, **kwargs):
        time.sleep(0.02)
        with lock:
            seen_routes.append(route_key)
        if cfg.get("intent") == "evolve:select":
            return _FakeRes(json.dumps({"selected_sample": "r1s1", "rationale": "fastest"}), route_key)
        routes_payload = json.loads((ws / "config" / "routes.json").read_text(encoding="utf-8"))
        prop = {
            "problem_id": "new",
            "problem_name": f"prop via {route_key}",
            "tier": "adaptable",
            "action": "new",
            "what": f"proposal from {route_key}",
            "predicted": [],
            "edits": [{"path": "config/routes.json", "mode": "write", "content": json.dumps(routes_payload, indent=2)}],
        }
        return _FakeRes(json.dumps(prop), route_key)

    monkeypatch.setattr("nexus.evolve.execute_route", fake_execute)
    events_path = tmp_path / "events.jsonl"
    pareto_path = tmp_path / "pareto.json"
    summary = run_evolve_round(
        ws,
        spec,
        ledger,
        round_num=1,
        proposer_route="refactor",
        selector_route="reason",
        explorer_routes=["omni", "fast-code"],
        candidates_k=2,
        policy=policy,
        parallel_candidates=True,
        events_path=events_path,
        pareto_path=pareto_path,
    )
    assert summary["explorer_routes"] == ["omni", "fast-code"]
    assert "pareto_front" in summary and len(summary["pareto_front"]) >= 1
    assert pareto_path.is_file()
    assert set(seen_routes) >= {"omni", "fast-code", "reason"}
    assert [c["proposer_route"] for c in summary["candidates"]] == ["omni", "fast-code"]


def test_cli_evolve_explorer_and_parallel_dry_run_envelope(monkeypatch, tmp_path: Path, capsys) -> None:
    cli = _load_cli()
    sim_path = tmp_path / "sim.json"
    ev_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("NEXUS_SIM_PATH", str(sim_path))
    rc = cli.main(
        [
            "--events-file",
            str(ev_path),
            "--envelope",
            "evolve",
            "--explorer-route",
            "omni",
            "--parallel",
            "--dry-run",
            "--rounds",
            "1",
        ]
    )
    assert rc == 0
    env = json.loads(capsys.readouterr().out)
    assert env["ok"] is True
    assert "pareto_front_sizes" in env["data"]


def test_execute_route_span_falls_back_to_user_routes_override(monkeypatch, tmp_path: Path) -> None:
    from nexus.evolve import _execute_route_span
    import nexus.router as router_mod

    ws = tmp_path / "ws"
    (ws / "config").mkdir(parents=True, exist_ok=True)
    (ws / "config" / "routes.json").write_text(
        json.dumps({"default_route": "code", "routes": {"code": {"backend": "opencode", "model": "m-free"}}}),
        encoding="utf-8",
    )
    user_routes = tmp_path / "user_routes.json"
    user_routes.write_text(
        json.dumps({"default_route": "code", "routes": {"omni": {"backend": "openai-compat", "model": "openrouter/openrouter/free", "quota_tier": "FREE"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(router_mod, "USER_ROUTES_PATH", user_routes)

    dispatched: list[tuple[str, dict]] = []

    class _Res:
        output = "ok"
        model = "openrouter/openrouter/free"
        tokens_in = 10
        tokens_out = 2
        elapsed_ms = 5
        status = "DONE"
        error = ""

    def fake_exec(trace_id, route_key, cfg, prompt, **kwargs):
        dispatched.append((route_key, cfg))
        return _Res()

    monkeypatch.setattr("nexus.evolve.execute_route", fake_exec)
    res, err = _execute_route_span(
        "tr_user_ovr",
        "sp_parent",
        ws,
        "omni",
        "prompt",
        events_path=tmp_path / "events.jsonl",
        allow_fallback=True,
        intent_label="evolve:propose",
    )
    assert err == "" and res is not None
    assert dispatched and dispatched[0][0] == "omni"
    assert dispatched[0][1]["model"] == "openrouter/openrouter/free"
