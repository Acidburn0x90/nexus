"""Tests for Phase 18 ingrained self-improvement, polyglot scope, and multi-axis promotion."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nexus import sim
from nexus.events import emit_event
from nexus.evolve import (
    DEFAULT_BENCHMARK_PATH,
    IMPROVEMENT_LADDER,
    OBJECTIVES,
    BenchmarkSpec,
    EvalReport,
    ScopePolicy,
    beats_baseline,
    detect_improvement_opportunities,
    render_improvement_opportunities,
    run_evolve_round,
)
from nexus.router import load_routes
from nexus.visualizer import strip_ansi, visible_len


def test_scope_policy_presets_and_locked_boundaries(tmp_path: Path) -> None:
    routes_pol = ScopePolicy.for_scope("routes")
    assert routes_pol.validate_paths(["config/routes.json"], tmp_path)[0] is True
    assert routes_pol.validate_paths(["config/profiles/custom.json"], tmp_path)[0] is False

    prompts_pol = ScopePolicy.for_scope("prompts")
    assert prompts_pol.validate_paths(["config/profiles/custom.json"], tmp_path)[0] is True
    assert prompts_pol.validate_paths(["nexus/router.py"], tmp_path)[0] is False

    full_pol = ScopePolicy.for_scope("full")
    assert full_pol.validate_paths(["nexus/router.py", "nexus/entity.py"], tmp_path)[0] is True
    assert full_pol.validate_paths(["native/fast_router.c"], tmp_path)[0] is False

    poly_pol = ScopePolicy.for_scope("polyglot")
    assert poly_pol.validate_paths(
        ["config/profiles/tuned.json", "nexus/router.py", "native/src/lib.rs", "native/fast.c"],
        tmp_path,
    )[0] is True

    # Bundled JSON policies match the same security contract.
    self_json = ScopePolicy.load(PROJECT_ROOT / "config" / "policies" / "self_rewrite.json")
    poly_json = ScopePolicy.load(PROJECT_ROOT / "config" / "policies" / "polyglot.json")
    assert self_json.validate_paths(["nexus/backends.py", "config/profiles/x.json"], tmp_path)[0] is True
    assert poly_json.validate_paths(["native/fast.zig"], tmp_path)[0] is True

    # Locked files, policy definitions, traversal, and symlinks remain forbidden across every preset.
    for pol in (routes_pol, prompts_pol, full_pol, poly_pol, self_json, poly_json):
        for locked in (
            "nexus/sim.py",
            "nexus/evolve.py",
            "nexus/drops.py",
            "config/sim_policy.json",
            "config/policies/polyglot.json",
            "config/evals/routing_benchmark.json",
            "tests/test_evolve.py",
            ".git/config",
            ".env",
            "../outside.py",
        ):
            ok, _errs = pol.validate_paths([locked], tmp_path)
            assert ok is False, f"{locked} should remain locked under {pol}"

    link_dir = tmp_path / "native"
    link_dir.mkdir(parents=True)
    evil = link_dir / "link.rs"
    evil.symlink_to(tmp_path / "outside.rs")
    assert poly_pol.validate_paths(["native/link.rs"], tmp_path)[0] is False


def test_sandbox_polyglot_build_failure_and_speedup_promotion(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    (ws / "config").mkdir(parents=True)
    routes = load_routes(PROJECT_ROOT / "config" / "routes.json")
    (ws / "config" / "routes.json").write_text(json.dumps(routes), encoding="utf-8")

    builder = tmp_path / "build_check.py"
    builder.write_text(
        "import pathlib, sys\n"
        "src = pathlib.Path('native/fast_kernel.c')\n"
        "if not src.exists():\n"
        "    sys.exit(0)\n"
        "text = src.read_text(encoding='utf-8')\n"
        "if 'COMPILE_ERROR' in text:\n"
        "    sys.stderr.write('syntax error in fast_kernel.c\\n')\n"
        "    sys.exit(2)\n"
        "pathlib.Path('native/fast_kernel.ok').write_text('compiled', encoding='utf-8')\n",
        encoding="utf-8",
    )
    verifier = tmp_path / "verify_perf.py"
    verifier.write_text(
        "import json, pathlib\n"
        "fast = pathlib.Path('native/fast_kernel.ok').is_file()\n"
        "print(json.dumps({\n"
        "    'results': {'t1': True, 't2': True},\n"
        "    'latency_ms': 14.0 if fast else 85.0,\n"
        "    'tokens_total': 420,\n"
        "    'metrics': {'open_model_rate': 1.0},\n"
        "}))\n",
        encoding="utf-8",
    )

    spec = BenchmarkSpec(
        name="polyglot-speed",
        domain="harness",
        editable_files=["config/routes.json"],
        locked_files=[],
        evolve_tasks=[
            {"id": "t1", "prompt": "p1", "expected_route": "code"},
            {"id": "t2", "prompt": "p2", "expected_route": "code"},
        ],
        held_out_tasks=[
            {"id": "h1", "prompt": "hp1", "expected_route": "code"},
        ],
        verifier="command",
        verifier_cmd=[sys.executable, str(verifier)],
        build_cmd=[sys.executable, str(builder)],
    )
    policy = ScopePolicy.for_scope("polyglot")

    ledger = sim.new_ledger(domain="harness", benchmark="polyglot-speed")
    events_path = tmp_path / "events.jsonl"

    # 1. Failing native build must reject promotion and log negative evidence in SIM.
    def failing_optimizer(role: str, prompt: str) -> str:
        assert "polyglot_rewrite" in prompt
        return json.dumps(
            {
                "problem_id": "new",
                "problem_name": "native hot path kernel",
                "tier": "adaptable",
                "action": "new",
                "strategy": "polyglot_rewrite",
                "what": "rewrite dispatch loop in C",
                "predicted": ["t1"],
                "edits": [
                    {"path": "native/fast_kernel.c", "mode": "write", "content": "COMPILE_ERROR;"}
                ],
            }
        )

    fail_summary = run_evolve_round(
        ws,
        spec,
        ledger,
        policy=policy,
        candidates_k=1,
        apply_changes=True,
        optimizer_fn=failing_optimizer,
        events_path=events_path,
        objective="speed",
    )
    assert fail_summary["promoted"] is False
    assert fail_summary["applied"] is False
    assert "build failed" in fail_summary["candidates"][0]["error"]
    assert not (ws / "native" / "fast_kernel.c").exists()
    assert ledger["problems"][0]["attempts"][0]["accepted"] is False

    # 2. Valid native build that yields a measurable speedup promotes & applies cleanly.
    def fast_optimizer(role: str, prompt: str) -> str:
        return json.dumps(
            {
                "problem_id": "P1",
                "problem_name": "native hot path kernel",
                "tier": "adaptable",
                "action": "revise",
                "strategy": "polyglot_rewrite",
                "what": "compile clean C kernel for native/fast_kernel.c",
                "predicted": [],
                "edits": [
                    {"path": "native/fast_kernel.c", "mode": "write", "content": "int fast_ok(void){return 1;}"}
                ],
            }
        )

    win_summary = run_evolve_round(
        ws,
        spec,
        ledger,
        policy=policy,
        candidates_k=1,
        apply_changes=True,
        optimizer_fn=fast_optimizer,
        events_path=events_path,
        objective="speed",
    )
    assert win_summary["promoted"] is True
    assert win_summary["applied"] is True
    assert "speedup" in win_summary["promotion_reason"]
    assert (ws / "native" / "fast_kernel.c").is_file()


def test_beats_baseline_all_objectives() -> None:
    baseline = EvalReport(
        split="evolve",
        results={"t1": True, "t2": True},
        passed=2,
        total=2,
        score=1.0,
        latency_ms=100.0,
        tokens_total=1000,
        metrics={"open_model_rate": 0.60},
    )

    def _cand(**overrides) -> dict:
        base = {
            "error": "",
            "scope_ok": True,
            "touched": ["config/routes.json"],
            "regressed": [],
            "fixed": [],
            "passed": 2,
            "total": 2,
            "score": 1.0,
            "latency_ms": 100.0,
            "tokens_total": 1000,
            "metrics": {"open_model_rate": 0.60},
        }
        base.update(overrides)
        return base

    assert set(OBJECTIVES) == {"accuracy", "speed", "efficiency", "open_model", "balanced"}
    assert len(IMPROVEMENT_LADDER) == 4

    # Small noise (< 5%) does not promote.
    assert beats_baseline(_cand(latency_ms=97.0), baseline, objective="speed")[0] is False

    # Clear speedup promotes under speed and balanced, not accuracy.
    assert beats_baseline(_cand(latency_ms=80.0), baseline, objective="speed")[0] is True
    assert beats_baseline(_cand(latency_ms=80.0), baseline, objective="balanced")[0] is True
    assert beats_baseline(_cand(latency_ms=80.0), baseline, objective="accuracy")[0] is False

    # Token reduction promotes under efficiency and balanced.
    assert beats_baseline(_cand(tokens_total=800), baseline, objective="efficiency")[0] is True
    assert beats_baseline(_cand(tokens_total=800), baseline, objective="balanced")[0] is True

    # Open-model squeeze gain promotes under open_model and balanced.
    assert beats_baseline(_cand(metrics={"open_model_rate": 0.90}), baseline, objective="open-model")[0] is True
    assert beats_baseline(_cand(metrics={"open_model_rate": 0.90}), baseline, objective="balanced")[0] is True

    # Any regression blocks promotion regardless of speed/efficiency.
    assert (
        beats_baseline(
            _cand(score=0.5, regressed=["t2"], latency_ms=5.0, tokens_total=10),
            baseline,
            objective="speed",
        )[0]
        is False
    )


def test_detect_improvement_opportunities_telemetry_and_sim(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    routes_data = load_routes(PROJECT_ROOT / "config" / "routes.json")

    # Healthy baseline yields proactive suggestion.
    proactive = detect_improvement_opportunities([], routes_data=routes_data)
    assert len(proactive) == 1
    assert proactive[0]["priority"] == "low"

    # Inject failing open route, premium leakage, and latency hotspot.
    for idx in range(3):
        emit_event(
            f"tr_c{idx}",
            f"sp_c{idx}",
            route="code",
            model="opencode/deepseek-v4.1-flash-free",
            quota_tier="FREE",
            status="DONE" if idx == 0 else "FAIL",
            elapsed_ms=400,
            path=events_path,
        )
    emit_event(
        "tr_p1",
        "sp_p1",
        route="premium",
        model="premium-cli",
        quota_tier="PREMIUM",
        status="DONE",
        elapsed_ms=800,
        path=events_path,
    )
    for idx in range(2):
        emit_event(
            f"tr_r{idx}",
            f"sp_r{idx}",
            route="reason",
            model="opencode-go/glm-5.3",
            quota_tier="GO",
            status="DONE",
            elapsed_ms=19000,
            path=events_path,
        )

    ledger = sim.new_ledger(domain="routing")
    sim.open_problem(ledger, "stuck route tie", opened_round=1, problem_id="P1")
    for r in (1, 2):
        sim.record_attempt(
            ledger,
            problem_id="P1",
            round_num=r,
            sample=f"r{r}s1",
            action="revise",
            accepted=False,
            domain="routing",
            what="tweak keywords",
            files=["config/routes.json"],
            predicted=["t1"],
            fixed=[],
            missed=["t1"],
            passed=1,
            total=2,
        )

    from nexus.events import read_events

    events = read_events(events_path, limit=100)
    opps = detect_improvement_opportunities(events, ledger=ledger, routes_data=routes_data)
    ids = {o["id"] for o in opps}
    assert {
        "route-reliability:code",
        "premium-leakage:premium",
        "latency-hotspot:reason",
        "sim-dead-end:P1",
    } <= ids

    rendered = render_improvement_opportunities(opps, color=False, width=88)
    assert "NEXUS SELF-IMPROVEMENT SUGGESTIONS" in rendered
    assert all(visible_len(line) == 88 for line in strip_ansi(rendered).splitlines())


def test_cli_evolve_suggest_and_scope_flags(tmp_path: Path) -> None:
    bin_path = PROJECT_ROOT / "bin" / "nexus"
    sim_file = tmp_path / "sim.json"
    events_file = tmp_path / "events.jsonl"
    env = {"HOME": str(tmp_path), "PATH": os.environ.get("PATH", ""), "NEXUS_SIM_PATH": str(sim_file)}

    emit_event(
        "tr_s1",
        "sp_s1",
        route="premium",
        model="premium-cli",
        quota_tier="PREMIUM",
        status="DONE",
        elapsed_ms=200,
        path=events_file,
    )

    sug = subprocess.run(
        [sys.executable, str(bin_path), "--events-file", str(events_file), "--no-color", "evolve", "suggest"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert "NEXUS SELF-IMPROVEMENT SUGGESTIONS" in sug.stdout
    assert "@premium" in sug.stdout
    assert "[HIGH" in sug.stdout

    sug_json = subprocess.run(
        [sys.executable, str(bin_path), "--events-file", str(events_file), "--no-color", "evolve", "suggest", "--json"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    parsed = json.loads(sug_json.stdout)
    assert any(item["id"] == "premium-leakage:premium" for item in parsed)

    ev = subprocess.run(
        [
            sys.executable,
            str(bin_path),
            "--events-file",
            str(events_file),
            "--no-color",
            "evolve",
            "--scope",
            "prompts",
            "--objective",
            "open-model",
            "--dry-run",
            "--rounds",
            "1",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert "EVOLVE | round=" in ev.stdout
    assert "obj=open_model" in ev.stderr
