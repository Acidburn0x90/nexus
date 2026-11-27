"""Tests for nexus.entity: unified entity continuity, profiles, stats, and benchmarking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nexus import backends
from nexus.entity import (
    DEFAULT_PROFILE_NAME,
    DEFAULT_SYSTEM_PERSONA,
    EntityProfile,
    benchmark_profiles,
    build_handoff_context,
    build_synthesis_prompt,
    compose_entity_prompt,
    compute_performance_stats,
    list_profiles,
    load_profile,
    render_bench_table,
    render_stats_table,
    resolve_profile_routes,
    save_user_profile,
)
from nexus.events import emit_event, read_events
from nexus.router import resolve_route
from nexus.sessions import TuiSession, load_sessions
from nexus.visualizer import visible_len

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_builtin_profiles_discover_and_load() -> None:
    profiles = list_profiles()
    names = {p.name for p in profiles}
    assert {"default", "free-only", "local-only", "ensemble"} <= names

    free_prof = load_profile("free-only")
    assert free_prof.mode == "entity"
    resolved_free = resolve_profile_routes(free_prof)
    for key in ("refactor", "reason", "context"):
        assert resolved_free["routes"][key]["quota_tier"] == "FREE"

    local_prof = load_profile("local-only")
    assert local_prof.replace_routes is True
    resolved_local = resolve_profile_routes(local_prof)
    assert set(resolved_local["routes"]) == {"local", "ollama"}
    got, _, _ = resolve_route("audit security design", routes_data=resolved_local)
    assert got == "ollama"
    got_fast, _, _ = resolve_route("quick snippet", routes_data=resolved_local)
    assert got_fast == "local"

    ens = load_profile("ensemble")
    assert ens.mode == "ensemble"


def test_user_profile_override_save_and_explicit_path(tmp_path: Path) -> None:
    user_dir = tmp_path / "profiles"
    custom = EntityProfile(
        name="free-only",
        description="Custom user override",
        mode="entity",
        system_persona="Custom persona",
        routes={"code": {"model": "opencode/mimo-v2.6-flash-free", "quota_tier": "FREE"}},
    )
    saved_path = save_user_profile(custom, user_dir=user_dir)
    assert saved_path.is_file()

    loaded = load_profile("free-only", user_dir=user_dir)
    assert loaded.description == "Custom user override"
    assert loaded.system_persona == "Custom persona"

    by_path = load_profile(saved_path)
    assert by_path.name == "free-only"

    listed = {p.name: p for p in list_profiles(user_dir=user_dir)}
    assert listed["free-only"].description == "Custom user override"

    with pytest.raises(FileNotFoundError):
        load_profile("no-such-profile", user_dir=user_dir)


def test_handoff_context_bounds_and_compose_entity_prompt() -> None:
    turns = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "[error] transient timeout"},
        {"role": "assistant", "content": "answer 1"},
        {"role": "user", "content": "turn 2 " + ("x" * 180)},
        {"role": "assistant", "content": "answer 2"},
    ]
    ctx = build_handoff_context(turns, max_turns=3, max_chars=260)
    assert ctx.startswith("[Nexus Shared Conversation Context]")
    assert "[error]" not in ctx
    assert "Nexus: answer 2" in ctx
    assert len(ctx) <= 260

    assert compose_entity_prompt("hello") == "hello"
    with_persona = compose_entity_prompt(
        "hello", system_persona=DEFAULT_SYSTEM_PERSONA, handoff_context=ctx, include_handoff=False
    )
    assert "[Nexus System Persona]" in with_persona
    assert "[Nexus Shared Conversation Context]" not in with_persona
    assert "[Current Request]\nhello" in with_persona

    full = compose_entity_prompt(
        "hello", system_persona=DEFAULT_SYSTEM_PERSONA, handoff_context=ctx, include_handoff=True
    )
    assert "[Nexus Shared Conversation Context]" in full


def test_build_synthesis_prompt() -> None:
    prompt = build_synthesis_prompt(
        "Should we use SQLite or JSONL?",
        [("pioneer", "Use JSONL for O_APPEND."), ("red_teamer", "Watch file growth.")],
    )
    assert "Perspective 1 (PIONEER)" in prompt
    assert "Perspective 2 (RED-TEAMER)" in prompt
    assert "Should we use SQLite or JSONL?" in prompt
    assert "never mention internal drafts" in prompt


def test_compute_performance_stats_and_render(tmp_path: Path) -> None:
    ev_path = tmp_path / "events.jsonl"
    emit_event(
        "tr_1", "sp_root1", parent_id=None, route="front-desk",
        backend="nexus", model="nexus-router", quota_tier="ORCHESTRATOR",
        status="DONE", elapsed_ms=5, path=ev_path,
    )
    emit_event(
        "tr_1", "sp_c1", parent_id="sp_root1", route="in-charge:code",
        backend="opencode", model="opencode/deepseek-v4.1-flash-free", quota_tier="FREE",
        status="DONE", tokens_in=100, tokens_out=50, elapsed_ms=200, profile="default", path=ev_path,
    )
    emit_event(
        "tr_2", "sp_c2", parent_id=None, route="reason",
        backend="opencode", model="opencode-go/glm-5.3", quota_tier="GO",
        status="FAIL", tokens_in=300, tokens_out=20, elapsed_ms=800, profile="default", path=ev_path,
    )
    emit_event(
        "tr_3", "sp_c3", parent_id=None, route="code",
        backend="opencode", model="opencode/deepseek-v4.1-flash-free", quota_tier="FREE",
        status="DONE", tokens_in=150, tokens_out=75, elapsed_ms=300, profile="free-only", path=ev_path,
    )

    events = read_events(path=ev_path, limit=50)
    by_prof = compute_performance_stats(events, group_by="profile")
    assert by_prof["total_runs"] == 3
    assert by_prof["groups"]["default"]["runs"] == 2
    assert by_prof["groups"]["default"]["success_rate"] == pytest.approx(0.5)
    assert by_prof["groups"]["free-only"]["success_rate"] == pytest.approx(1.0)

    by_route = compute_performance_stats(events, group_by="route")
    assert by_route["groups"]["code"]["runs"] == 2  # strips `in-charge:` prefix

    table = render_stats_table(by_route, group_by="route", color=False, width=88)
    for line in table.splitlines():
        assert visible_len(line) == 88


def test_benchmark_profiles_and_render() -> None:
    rows = benchmark_profiles(
        REPO_ROOT, ["default", "free-only", "local-only"], split="held_out"
    )
    assert len(rows) == 3
    by_name = {r["profile"]: r for r in rows}
    assert by_name["default"]["score"] == pytest.approx(1.0)
    assert by_name["free-only"]["score"] == pytest.approx(1.0)
    assert "GO" not in by_name["free-only"]["tiers"]
    assert by_name["local-only"]["tiers"] == ["LOCAL"]

    table = render_bench_table(rows, split="held_out", color=False, width=90)
    assert "free-only" in table
    for line in table.splitlines():
        assert visible_len(line) == 90


def test_execute_route_forwards_profile_and_session_migrates(tmp_path: Path) -> None:
    ev_path = tmp_path / "events.jsonl"
    res = backends.execute_route(
        "tr_ent1",
        "code",
        {"backend": "opencode", "model": "opencode/deepseek-v4.1-flash-free", "quota_tier": "FREE"},
        "hello entity",
        events_path=ev_path,
        dry_run=True,
        emit_stderr=False,
        profile="free-only",
        system_persona=DEFAULT_SYSTEM_PERSONA,
        handoff_context="[Nexus Shared Conversation Context]\nUser: hi",
    )
    assert res.status == "DONE"
    assert res.output.startswith("[dry-run] code ->")
    evs = read_events(path=ev_path, trace_id="tr_ent1")
    assert evs and evs[-1]["profile"] == "free-only"

    sessions_file = tmp_path / "tui_sessions.json"
    sessions_file.write_text(
        json.dumps([{"id": "tui_legacy", "title": "Old", "created_at": 1.0, "updated_at": 2.0, "argon_spans": 1}]),
        encoding="utf-8",
    )
    loaded = load_sessions(sessions_file)
    assert len(loaded) == 1
    assert loaded[0].profile == DEFAULT_PROFILE_NAME
    assert loaded[0].entity_mode is True
    assert loaded[0].premium_spans == 1
    assert isinstance(loaded[0], TuiSession)
