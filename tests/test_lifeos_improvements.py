"""Tests for LifeOS-inspired Nexus improvements and live-telemetry fixes."""

from __future__ import annotations

import asyncio
import importlib.machinery
import importlib.util
import http.server
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from nexus import backends
from nexus.assay import (
    ASSAY_SCHEMA,
    build_assay_receipt,
    parse_isa_markdown,
    record_shadow_decision,
    render_isa_summary,
    run_staged_assay,
    summarize_shadow_log,
)
from nexus.contract import (
    CONTRACT_SCHEMA,
    EXIT_INTEGRITY,
    EXIT_NOT_FOUND,
    EXIT_OK,
    EXIT_REFUSED,
    check_version_drift,
    make_envelope,
    validate_trust_root,
)
from nexus.entity import compute_performance_stats
from nexus.errata import (
    read_errata,
    record_erratum,
    render_errata_report,
    summarize_errata,
)
from nexus.events import emit_event, merge_span_events, read_events, reconstruct_tree
from nexus.evolve import (
    BenchmarkSpec,
    ScopePolicy,
    detect_improvement_opportunities,
    evaluate_split,
    run_evolve_round,
)
from nexus.router import front_desk_decide, load_routes, resolve_route
from nexus.sessions import TranscriptItem
from nexus import sim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROUTES_PATH = PROJECT_ROOT / "config" / "routes.json"


def _load_cli(name: str = "nexus_cli_lifeos_test"):
    bin_path = PROJECT_ROOT / "bin" / "nexus"
    loader = importlib.machinery.SourceFileLoader(name, str(bin_path))
    spec = importlib.util.spec_from_loader(name, loader)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


def test_events_fallback_overwrite_dry_run_exclusion_and_auto_prune(monkeypatch, tmp_path: Path) -> None:
    bus = tmp_path / "events.jsonl"
    # 1) Normal follow-up without model change keeps _FIRST_WINS (PREMIUM intact).
    emit_event("tr_p1", "sp_p1", route="premium", backend="agy", model="premium-cli", quota_tier="PREMIUM", status="RUNNING", path=bus)
    emit_event("tr_p1", "sp_p1", route="premium", backend="agy", model="premium-cli", quota_tier="", status="DONE", elapsed_ms=12, path=bus)

    # 2) Fallback follow-up with switched model overwrites model/quota_tier/route/backend cleanly.
    emit_event("tr_fb", "sp_fb", route="code", backend="opencode", model="opencode/deepseek-v4.1-flash-free", quota_tier="FREE", status="RUNNING", path=bus)
    emit_event("tr_fb", "sp_fb", route="omni", backend="openai-compat", model="openrouter/openrouter/free", quota_tier="FREE", status="RUNNING", step="fallback->omni", path=bus)
    emit_event("tr_fb", "sp_fb", route="code", backend="opencode", model="opencode-go/deepseek-v4.1-flash", quota_tier="GO", status="DONE", elapsed_ms=99, path=bus)

    # 3) Dry-run span emitted directly + legacy dry-run complete step.
    emit_event("tr_dr", "sp_dr1", route="code", backend="opencode", model="opencode/deepseek-v4.1-flash-free", quota_tier="FREE", status="DONE", step="ok", dry_run=True, path=bus)
    emit_event("tr_dr", "sp_dr2", route="triage", backend="opencode", model="opencode/nemotron-3.5-lightning-free", quota_tier="FREE", status="DONE", step="dry-run complete", path=bus)

    evs = read_events(path=bus, limit=50)
    tree_p = reconstruct_tree(evs, "tr_p1")
    assert tree_p["spans"]["sp_p1"]["quota_tier"] == "PREMIUM"
    assert tree_p["summary"]["premium_spans"] == 1

    tree_fb = reconstruct_tree(evs, "tr_fb")
    assert tree_fb["spans"]["sp_fb"]["model"] == "opencode-go/deepseek-v4.1-flash"
    assert tree_fb["spans"]["sp_fb"]["quota_tier"] == "GO"

    stats_live = compute_performance_stats(evs, group_by="route")
    assert stats_live["total_runs"] == 2
    assert stats_live["excluded_dry_run"] == 2
    assert "triage" not in stats_live["groups"]

    stats_all = compute_performance_stats(evs, group_by="route", include_dry_run=True)
    assert stats_all["total_runs"] == 4
    assert stats_all["excluded_dry_run"] == 0

    # Auto-prune trigger when file exceeds threshold.
    import nexus.events as ev_mod

    monkeypatch.setattr(ev_mod, "AUTO_PRUNE_MAX_BYTES", 200)
    monkeypatch.setattr(ev_mod, "AUTO_PRUNE_KEEP_LINES", 3)
    prune_bus = tmp_path / "prune.jsonl"
    for idx in range(8):
        emit_event(f"tr_a{idx}", f"sp_a{idx}", route="code", status="DONE", path=prune_bus)
    kept_lines = [ln for ln in prune_bus.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(kept_lines) <= 4


def test_cross_backend_fallback_opencode_to_http_updates_tree(monkeypatch, tmp_path: Path) -> None:
    backends.reset_health_state()
    http_calls: list[dict] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length", "0"))
            http_calls.append(json.loads(self.rfile.read(n).decode("utf-8")))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"omni ok"}}]}\ndata: [DONE]\n')

        def log_message(self, *args) -> None:
            return

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        endpoint = f"http://127.0.0.1:{srv.server_address[1]}/v1/chat/completions"
        routes_file = tmp_path / "routes.json"
        code_cfg = {
            "backend": "opencode",
            "model": "opencode/deepseek-v4.1-flash-free",
            "quota_tier": "FREE",
            "timeout_s": 5,
            "fallback_route": "omni",
        }
        omni_cfg = {
            "backend": "openai-compat",
            "endpoint": endpoint,
            "model": "openrouter/openrouter/free",
            "quota_tier": "FREE",
            "timeout_s": 5,
        }
        routes_file.write_text(json.dumps({"routes": {"code": code_cfg, "omni": omni_cfg}}), encoding="utf-8")

        monkeypatch.setattr(backends, "_require_binary", lambda _b: None)
        monkeypatch.setattr(
            backends,
            "_run_subprocess_bounded",
            lambda *a, **k: (1, '{"type":"error","error":{"message":"429 rate limit"}}', "", False),
        )
        ev_file = tmp_path / "events.jsonl"
        res = backends.execute_route(
            "tr_xback",
            "code",
            code_cfg,
            "hello",
            events_path=ev_file,
            inject_square=False,
            emit_stderr=False,
            routes_path=routes_file,
        )
        assert res.status == "DONE"
        assert res.route == "omni"
        assert res.backend == "openai-compat"
        assert res.output == "omni ok"
        assert len(http_calls) == 1
        tree = reconstruct_tree(read_events(path=ev_file, trace_id="tr_xback"), "tr_xback")
        span = list(tree["spans"].values())[0]
        assert span["backend"] == "openai-compat"
        assert span["model"] == "openrouter/openrouter/free"
    finally:
        srv.shutdown()
        srv.server_close()
        backends.reset_health_state()


def test_errata_write_ahead_join_false_pass_and_suggestions(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NEXUS_ERRATA_PATH", str(tmp_path / "errata.jsonl"))
    ev_file = tmp_path / "events.jsonl"
    emit_event("tr_ok1", "sp_r1", route="front-desk", backend="nexus", model="router", quota_tier="ORCHESTRATOR", status="DONE", path=ev_file)
    emit_event("tr_ok1", "sp_c1", parent_id="sp_r1", route="in-charge:code", backend="opencode", model="deepseek-v4.1-flash-free", quota_tier="FREE", status="DONE", path=ev_file)

    with pytest.raises(ValueError):
        record_erratum("   ", events_path=ev_file)

    verbatim = "The answer clipped off step 3  with exact whitespace!"
    rec = record_erratum(verbatim, events_path=ev_file)
    assert rec["message"] == verbatim
    assert rec["trace_id"] == "tr_ok1"
    assert rec["route"] == "code"
    assert rec["model"] == "deepseek-v4.1-flash-free"
    assert rec["false_pass"] is True
    assert rec["kind"] == "false-pass"

    entries = read_errata(route="code")
    assert len(entries) == 1
    assert summarize_errata(entries)["false_passes"] == 1
    panel = render_errata_report(entries, color=False, width=80)
    assert "FALSE-PASS" in panel

    opps = detect_improvement_opportunities([], routes_data=load_routes(ROUTES_PATH), errata=entries)
    fp_opps = [o for o in opps if o["id"] == "errata-false-pass:code"]
    assert fp_opps and fp_opps[0]["priority"] == "high"
    assert "open_question" in fp_opps[0]


def test_assay_receipts_staged_shadow_and_isa(tmp_path: Path) -> None:
    empty_receipt = build_assay_receipt(b"abc", ideal_state="clean", stages=[])
    assert empty_receipt["schema"] == ASSAY_SCHEMA
    assert empty_receipt["status"] == "UNMEASURED"
    assert len(empty_receipt["subject_sha256"]) == 64

    called: list[str] = []

    def s1(_raw: bytes) -> tuple[bool, str]:
        called.append("s1")
        return False, "stage 1 failed"

    def s2(_raw: bytes) -> tuple[bool, str]:
        called.append("s2")
        return True, "should not run"

    staged = run_staged_assay("payload", ideal_state="pass both", stages=[("lint", s1), ("judge", s2)])
    assert staged["status"] == "FAIL"
    assert called == ["s1"]
    assert staged["stages"][1]["skipped"] is True

    shadow_path = tmp_path / "shadow.jsonl"
    for i in range(20):
        record_shadow_decision(
            prompt=f"prompt {i}",
            keyword_route="code",
            llm_route="code" if i < 18 else "triage",
            agreed=(i < 18),
            elapsed_ms=250 + i,
            decision_model="omni",
            path=shadow_path,
        )
    summary = summarize_shadow_log(shadow_path)
    assert summary["total"] == 20
    assert summary["agreed"] == 18
    assert summary["ready_to_promote"] is True

    isa_file = tmp_path / "ISA.md"
    isa_file.write_text(
        "# Plan\n\n## Ideal State\n\nZero-regression router.\n\n- [x] Done item\n  - **Probe:** `pytest -q`\n- [ ] Open item\n",
        encoding="utf-8",
    )
    isa = parse_isa_markdown(isa_file)
    assert isa["ideal_state"] == "Zero-regression router."
    assert (isa["checked"], isa["open"], isa["measured"], isa["unmeasured"]) == (1, 1, 1, 1)
    assert "UNMEASURED" in render_isa_summary(isa, width=80)


def test_contract_envelopes_trust_root_and_version_drift(tmp_path: Path) -> None:
    ok_env = make_envelope("doctor", ok=True, data={"ready": True})
    assert set(ok_env) == {"schema", "ok", "command", "data", "error"}
    assert ok_env["schema"] == CONTRACT_SCHEMA and ok_env["error"] is None

    err_env = make_envelope("sim.init", ok=False, error_code="governance_refused", error_message="exists")
    assert set(err_env) == {"schema", "ok", "command", "data", "error"}
    assert err_env["data"] is None and err_env["error"]["code"] == "governance_refused"

    root = tmp_path / "trust"
    root.mkdir()
    (root / "ok.json").write_text("{}", encoding="utf-8")
    assert validate_trust_root(root) == (True, [])
    (root / "bad_link").symlink_to(tmp_path)
    valid, errs = validate_trust_root(root)
    assert not valid and "integrity_error" in errs[0]

    # VersionDrift detects >= threshold core .py edits with zero doc edits.
    repo = tmp_path / "repo"
    (repo / "nexus").mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    (repo / "nexus" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "nexus" / "b.py").write_text("y = 2\n", encoding="utf-8")
    drift = check_version_drift(repo, threshold=2)
    assert drift["git_available"] is True
    assert drift["drifted"] is True
    (repo / "AGENTS.md").write_text("# Docs\n", encoding="utf-8")
    assert check_version_drift(repo, threshold=2)["drifted"] is False


def test_router_http_front_desk_shadow_and_fast_qa(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length", "0"))
            calls.append({"body": json.loads(self.rfile.read(n).decode("utf-8")), "auth": self.headers.get("Authorization")})
            payload = json.dumps(
                {
                    "choices": [{"message": {"content": '{"orchestrator":"opencode","route":"reason","take_charge":true,"reason":"http fast classify"}'}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 10},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args) -> None:
            return

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        base_routes = load_routes(ROUTES_PATH)
        key_file = tmp_path / "dec.key"
        key_file.write_text("sk-dec-1\n", encoding="utf-8")
        base_routes["routes"]["omni"] = {
            "backend": "openai-compat",
            "endpoint": f"http://127.0.0.1:{srv.server_address[1]}/v1/chat/completions",
            "model": "openrouter/openrouter/free",
            "api_key_file": str(key_file),
            "quota_tier": "FREE",
            "explicit_only": True,
        }
        base_routes["decision_route"] = "omni"
        base_routes["fast_qa_route"] = "omni"

        monkeypatch.setattr(backends, "_run_subprocess_bounded", lambda *a, **k: pytest.fail("should use HTTP front desk, not subprocess"))

        dec = front_desk_decide("compare pros and cons", routes_data=base_routes, use_llm=True)
        assert dec.route_key == "reason"
        assert dec.reason == "http fast classify"
        assert calls[0]["auth"] == "Bearer sk-dec-1"

        shadow_file = tmp_path / "shadow.jsonl"
        kw_dec = front_desk_decide(
            "audit security vulnerability",
            routes_data=base_routes,
            use_llm=True,
            shadow_mode=True,
            shadow_log_path=shadow_file,
        )
        assert kw_dec.route_key == "reason"
        sh = summarize_shadow_log(shadow_file)
        assert sh["total"] == 1 and sh["agreed"] == 1

        # fast_qa_route catches simple Q&A while leaving code/file tasks on @code.
        qa_key, _, _ = resolve_route("What is the capital of France?", routes_data=base_routes)
        code_key, _, _ = resolve_route("edit nexus/router.py to handle empty prompts", routes_data=base_routes)
        assert qa_key == "omni"
        assert code_key == "code"
    finally:
        srv.shutdown()
        srv.server_close()


def test_evolve_core_quality_json_repair_and_dry_run_isolation(monkeypatch, tmp_path: Path) -> None:
    core_bench = BenchmarkSpec.load(PROJECT_ROOT / "config" / "evals" / "core_quality_benchmark.json")
    ev_report = evaluate_split(PROJECT_ROOT, core_bench, split="evolve")
    ho_report = evaluate_split(PROJECT_ROOT, core_bench, split="held_out")
    assert ev_report.passed == ev_report.total and ev_report.total == 5
    assert ho_report.passed == ho_report.total and ho_report.total == 5
    assert ev_report.metrics["compression_ratio"] > 0.4

    # JSON repair retry on malformed initial proposer output.
    ws = tmp_path / "ws"
    (ws / "config").mkdir(parents=True)
    routes_data = load_routes(ROUTES_PATH)
    (ws / "config" / "routes.json").write_text(json.dumps(routes_data), encoding="utf-8")
    mini = BenchmarkSpec.load(PROJECT_ROOT / "config" / "evals" / "routing_benchmark.json")
    seen_roles: list[str] = []

    def repair_opt(role: str, prompt: str) -> str:
        seen_roles.append(role)
        if role == "proposer":
            return "Thinking... oops here is raw prose not json"
        if role == "proposer_repair":
            return json.dumps(
                {
                    "problem_id": "new",
                    "problem_name": "repaired",
                    "tier": "adaptable",
                    "action": "new",
                    "what": "keep valid routes",
                    "predicted": [],
                    "edits": [{"path": "config/routes.json", "mode": "write", "content": json.dumps(routes_data)}],
                }
            )
        return json.dumps({"selected_sample": "r1s1", "rationale": "clean"})

    ledger = sim.new_ledger(domain="routing")
    summary = run_evolve_round(
        ws,
        mini,
        ledger,
        candidates_k=1,
        optimizer_fn=repair_opt,
        events_path=tmp_path / "ev.jsonl",
    )
    assert "proposer_repair" in seen_roles
    assert summary["candidates"][0]["error"] == ""
    assert summary["candidates"][0]["failure_class"] == ""

    # CLI evolve --dry-run must NOT create or mutate default SIM path, and emits dry_run=True events.
    sim_target = tmp_path / "sim.json"
    ev_bus = tmp_path / "cli_ev.jsonl"
    monkeypatch.setenv("NEXUS_SIM_PATH", str(sim_target))
    cli = _load_cli("nexus_cli_lifeos_evolve")
    rc = cli.main(["--events-file", str(ev_bus), "evolve", "--dry-run", "--rounds", "1"])
    assert rc == 0
    assert not sim_target.exists()
    evs = read_events(path=ev_bus, limit=20)
    assert evs and all(ev.get("dry_run") is True for ev in evs)


def test_cli_contract_subcommands_and_tui_erratum(monkeypatch, tmp_path: Path, capsys) -> None:
    err_file = tmp_path / "errata.jsonl"
    ev_file = tmp_path / "events.jsonl"
    sim_file = tmp_path / "sim.json"
    monkeypatch.setenv("NEXUS_ERRATA_PATH", str(err_file))
    monkeypatch.setenv("NEXUS_SIM_PATH", str(sim_file))

    emit_event("tr_t1", "sp_t1", route="code", backend="opencode", model="m1", quota_tier="FREE", status="DONE", path=ev_file)
    cli = _load_cli("nexus_cli_lifeos_subcmds")

    assert cli.main(["--events-file", str(ev_file), "errata", "add", "--envelope", "missed edge case"]) == EXIT_OK
    env_add = json.loads(capsys.readouterr().out)
    assert env_add["schema"] == CONTRACT_SCHEMA and env_add["ok"] is True
    assert env_add["data"]["false_pass"] is True

    assert cli.main(["errata", "summary", "--envelope"]) == EXIT_OK
    env_sum = json.loads(capsys.readouterr().out)
    assert env_sum["data"]["total"] == 1

    assert cli.main(["assay", "core", "--envelope"]) == EXIT_OK
    env_core = json.loads(capsys.readouterr().out)
    assert env_core["ok"] is True and env_core["data"]["passed"] == env_core["data"]["total"]

    assert cli.main(["assay", "isa", "--envelope"]) == EXIT_OK
    env_isa = json.loads(capsys.readouterr().out)
    assert env_isa["data"]["measured"] >= 1

    sim_file.write_text(json.dumps(sim.new_ledger(domain="nexus")), encoding="utf-8")
    assert cli.main(["sim", "init", "--envelope"]) == EXIT_REFUSED
    env_ref = json.loads(capsys.readouterr().out)
    assert env_ref["ok"] is False and env_ref["error"]["code"] == "governance_refused"

    # Missing file returns EXIT_NOT_FOUND under --envelope.
    assert cli.main(["assay", "isa", "--path", str(tmp_path / "nope.md"), "--envelope"]) == EXIT_NOT_FOUND
    capsys.readouterr()

    # TUI /erratum joins active session's newest trace_id.
    from nexus.tui.app import NexusApp

    async def tui_scenario() -> None:
        app = NexusApp(
            events_path=ev_file,
            sessions_path=tmp_path / "sessions.json",
            config_path=tmp_path / "tui.json",
            square_path=tmp_path / "square.jsonl",
            board_path=tmp_path / "BOARD.md",
            history_path=tmp_path / "history",
            routes_path=ROUTES_PATH,
            dry_run=True,
            ui_mode="cli",
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.active_session.items.append(
                TranscriptItem(kind="assistant", content="bad reply", route="code", model="m1", trace_id="tr_t1")
            )
            app._handle_slash("/erratum incomplete explanation")
            await pilot.pause()

    asyncio.run(tui_scenario())
    logged = read_errata(err_file)
    assert len(logged) == 2
    assert logged[-1]["source"] == "tui"
    assert logged[-1]["trace_id"] == "tr_t1"
