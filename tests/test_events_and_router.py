"""Tests for the nexus event bus and route resolver (Phase 1 contract)."""

from __future__ import annotations

import http.server
import json
from pathlib import Path
import socket
import sys
import threading

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nexus.events import (
    DEFAULT_EVENTS_PATH,
    emit_event,
    new_span_id,
    new_trace_id,
    prune_events,
    read_events,
    reconstruct_tree,
)
from nexus.router import (
    DEFAULT_ROUTES_PATH,
    budget_allows,
    front_desk_decide,
    load_budgets,
    load_routes,
    list_opencode_models,
    probe_local_endpoints,
    resolve_route,
    validate_routes,
)


# --------------------------------------------------------------------------- events


def test_emit_and_read_roundtrip(tmp_path: Path) -> None:
    bus = tmp_path / "events.jsonl"
    trace = new_trace_id()
    span = new_span_id()

    record = emit_event(trace, span, route="code", backend="opencode", model="m/x", prompt="hi", path=bus)
    assert record["trace_id"] == trace
    assert record["span_id"] == span
    assert trace.startswith("tr_")
    assert span.startswith("sp_")

    events = read_events(path=bus)
    assert len(events) == 1
    assert events[0]["route"] == "code"
    assert events[0]["quota_tier"] == "FREE"
    assert events[0]["status"] == "RUNNING"


def test_read_events_filters_trace_and_limit_and_skips_garbage(tmp_path: Path) -> None:
    bus = tmp_path / "events.jsonl"
    emit_event("tr_aaaa1111", "sp_1", route="code", path=bus)
    emit_event("tr_bbbb2222", "sp_2", route="reason", path=bus)
    emit_event("tr_aaaa1111", "sp_3", route="triage", path=bus)

    with bus.open("a", encoding="utf-8") as handle:
        handle.write("this is not json\n")
        handle.write("\n")

    only_a = read_events(path=bus, trace_id="tr_aaaa1111")
    assert [event["span_id"] for event in only_a] == ["sp_1", "sp_3"]

    newest = read_events(path=bus, limit=2)
    assert len(newest) == 2
    assert newest[-1]["span_id"] == "sp_3"

    assert read_events(path=tmp_path / "missing.jsonl") == []


def test_reconstruct_tree_merges_updates_and_summarises(tmp_path: Path) -> None:
    bus = tmp_path / "events.jsonl"
    trace = "tr_merged01"

    # start + finish for the same span: earliest ts, newest status/usage kept
    emit_event(trace, "sp_root", route="entrypoint", backend="nexus", model="nexus-router",
               quota_tier="ORCHESTRATOR", status="RUNNING", step="dispatch", path=bus)
    emit_event(trace, "sp_child", parent_id="sp_root", route="reason", backend="opencode",
               model="opencode-go/deepseek-v4-pro", quota_tier="GO", status="RUNNING",
               prompt="audit this", path=bus)
    emit_event(trace, "sp_child", parent_id="sp_root", route="reason", backend="opencode",
               model="opencode-go/deepseek-v4-pro", quota_tier="GO", status="DONE",
               step="completed", tokens_in=100, tokens_out=40, elapsed_ms=900, path=bus)
    emit_event(trace, "sp_root", route="entrypoint", backend="nexus", model="nexus-router",
               quota_tier="ORCHESTRATOR", status="DONE", step="completed @reason",
               elapsed_ms=1000, path=bus)

    tree = reconstruct_tree(read_events(path=bus, trace_id=trace), trace_id=trace)

    assert tree["trace_id"] == trace
    assert tree["roots"] == ["sp_root"]
    assert tree["children"]["sp_root"] == ["sp_child"]

    child = tree["spans"]["sp_child"]
    assert child["status"] == "DONE"
    assert child["step"] == "completed"
    assert child["tokens_in"] == 100
    assert child["tokens_out"] == 40
    assert child["elapsed_ms"] == 900
    # first non-empty values are preserved through the merge
    assert child["prompt"] == "audit this"
    assert child["parent_id"] == "sp_root"

    root = tree["spans"]["sp_root"]
    assert root["status"] == "DONE"
    assert root["elapsed_ms"] == 1000

    summary = tree["summary"]
    assert summary["total_spans"] == 2
    assert summary["premium_spans"] == 0
    assert summary["non_premium_spans"] == 1  # GO child counts, ORCHESTRATOR root does not
    assert summary["tokens_in"] == 100
    assert summary["tokens_out"] == 40
    assert summary["elapsed_ms"] == 1000


def test_reconstruct_tree_flags_premium_spans(tmp_path: Path) -> None:
    bus = tmp_path / "events.jsonl"
    trace = "tr_premium01"
    emit_event(trace, "sp_root", route="entrypoint", backend="nexus", quota_tier="ORCHESTRATOR", path=bus)
    emit_event(trace, "sp_a", parent_id="sp_root", route="premium", backend="agy",
               model="premium-cli", quota_tier="PREMIUM", path=bus)

    summary = reconstruct_tree(read_events(path=bus, trace_id=trace), trace_id=trace)["summary"]
    assert summary["premium_spans"] == 1
    assert summary["non_premium_spans"] == 0


@pytest.mark.parametrize("legacy_tier", ["ARGON-QUOTA", "ARGON"])
def test_reconstruct_tree_counts_legacy_premium_tiers(tmp_path: Path, legacy_tier: str) -> None:
    """Legacy paid-tier names still contribute to the canonical premium count."""
    bus = tmp_path / "events.jsonl"
    trace = "tr_legacy_premium"
    emit_event(trace, "sp_root", route="entrypoint", backend="nexus", quota_tier="ORCHESTRATOR", path=bus)
    emit_event(trace, "sp_legacy", parent_id="sp_root", route="premium", backend="agy",
               model="premium-cli", quota_tier=legacy_tier, path=bus)

    summary = reconstruct_tree(read_events(path=bus, trace_id=trace), trace_id=trace)["summary"]
    assert summary["premium_spans"] == 1
    assert summary["non_premium_spans"] == 0


def test_reconstruct_tree_defaults_to_last_trace(tmp_path: Path) -> None:
    bus = tmp_path / "events.jsonl"
    emit_event("tr_first000", "sp_a", route="code", path=bus)
    emit_event("tr_second00", "sp_b", route="reason", path=bus)

    tree = reconstruct_tree(read_events(path=bus))
    assert tree["trace_id"] == "tr_second00"
    assert list(tree["spans"]) == ["sp_b"]

    empty = reconstruct_tree([])
    assert empty["trace_id"] is None
    assert empty["summary"]["total_spans"] == 0


def test_default_events_path_location() -> None:
    assert DEFAULT_EVENTS_PATH == Path.home() / ".local/state/nexus/events.jsonl"


def _append_bulk(bus: Path, count: int, trace: str) -> None:
    """Append `count` padded events so the bus comfortably exceeds the 256 KiB tail."""
    with bus.open("a", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(
                json.dumps(
                    {
                        "ts": index,
                        "trace_id": trace,
                        "span_id": f"sp_bulk{index}",
                        "route": "code",
                        "prompt": "x" * 200,
                    }
                )
                + "\n"
            )


def test_read_events_trace_tail_seek_avoids_full_file_read(tmp_path: Path, monkeypatch) -> None:
    bus = tmp_path / "events.jsonl"
    _append_bulk(bus, 1400, "tr_other001")
    for index in range(3):
        emit_event("tr_target01", f"sp_t{index}", route="code", path=bus)
    assert bus.stat().st_size > 300_000

    def _explode(*_args, **_kwargs):
        raise AssertionError("full read used for a trace that sits in the tail window")

    monkeypatch.setattr(Path, "read_text", _explode)

    events = read_events(path=bus, trace_id="tr_target01")
    assert [event["span_id"] for event in events] == ["sp_t0", "sp_t1", "sp_t2"]
    assert read_events(path=bus, trace_id="tr_target01", limit=2) == events[-2:]


def test_read_events_trace_falls_back_when_match_predates_tail(tmp_path: Path, monkeypatch) -> None:
    bus = tmp_path / "events.jsonl"
    emit_event("tr_old00001", "sp_old0", path=bus)
    emit_event("tr_old00001", "sp_old1", path=bus)
    _append_bulk(bus, 1500, "tr_new00001")
    assert bus.stat().st_size > 300_000

    original = Path.read_text
    reads: list[Path] = []

    def _tracked(self, *args, **kwargs):
        reads.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _tracked)

    events = read_events(path=bus, trace_id="tr_old00001")
    assert [event["span_id"] for event in events] == ["sp_old0", "sp_old1"]
    assert bus in reads


def test_prune_events_keeps_newest_lines_atomically(tmp_path: Path) -> None:
    bus = tmp_path / "events.jsonl"
    for index in range(10):
        emit_event("tr_prune01", f"sp_{index}", path=bus)
    with bus.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    before, after = prune_events(bus, keep_lines=3)
    assert (before, after) == (10, 3)
    assert [event["span_id"] for event in read_events(path=bus)] == ["sp_7", "sp_8", "sp_9"]
    assert not list(tmp_path.glob("*.tmp"))

    # A file already within budget is never rewritten.
    assert prune_events(bus, keep_lines=10) == (3, 3)
    assert [event["span_id"] for event in read_events(path=bus)] == ["sp_7", "sp_8", "sp_9"]

    assert prune_events(tmp_path / "missing.jsonl", keep_lines=5) == (0, 0)


# --------------------------------------------------------------------------- router


def test_load_routes_ships_expected_keys() -> None:
    data = load_routes(DEFAULT_ROUTES_PATH)
    assert data["default_route"] == "code"
    assert data["council_roles"] == {
        "pioneer": "fast-code",
        "red_teamer": "free-reason",
        "synthesizer": "refactor",
    }
    routes = data["routes"]
    assert {
        "triage", "code", "fast-code", "refactor", "reason", "context", "vision",
        "local", "ollama", "premium",
    } <= set(routes)
    assert routes["local"]["backend"] == "openai-compat"
    assert routes["ollama"]["backend"] == "ollama"
    assert routes["ollama"]["model"] == "llama3.2:latest"
    assert routes["premium"]["quota_tier"] == "PREMIUM"


def test_resolve_route_at_prefix_strips_prefix() -> None:
    key, cfg, cleaned = resolve_route("@reason Verify security invariants")
    assert key == "reason"
    assert cfg["model"] == "opencode-go/glm-5.3"
    assert cleaned.strip() == "Verify security invariants"


def test_resolve_route_explicit_flag_wins() -> None:
    key, cfg, cleaned = resolve_route("@reason whatever", explicit_route="local")
    assert key == "local"
    assert cfg["backend"] == "openai-compat"
    assert cleaned == "@reason whatever"


def test_resolve_route_keywords_and_default() -> None:
    key, _, _ = resolve_route("please refactor and modularize this module")
    assert key == "refactor"

    key, _, _ = resolve_route("take a screenshot of the ui")
    assert key == "vision"

    key, cfg, _ = resolve_route("zzz no keywords here")
    assert key == "code"
    assert cfg["model"] == "opencode/deepseek-v4.1-flash-free"

    # Route names alone must never auto-select an explicit-only backend.
    key_local_word, _, _ = resolve_route("fix a local variable in an ollama helper")
    assert key_local_word == "code"


def test_resolve_route_unknown_key_raises() -> None:
    with pytest.raises(KeyError):
        resolve_route("hello", explicit_route="nope")
    with pytest.raises(KeyError):
        resolve_route("@nope hello")


def test_validate_routes_passes_with_full_inventory() -> None:
    data = load_routes(DEFAULT_ROUTES_PATH)
    available = {cfg["model"] for cfg in data["routes"].values() if cfg["backend"] == "opencode"}
    available |= {cfg["fallback"] for cfg in data["routes"].values() if cfg.get("fallback")}
    ok, errors = validate_routes(data, available_models=available)
    assert ok is True
    assert errors == []


def test_validate_routes_fails_loudly_on_missing_model() -> None:
    data = load_routes(DEFAULT_ROUTES_PATH)
    ok, errors = validate_routes(data, available_models={"opencode/some-other-model"})
    assert ok is False
    assert any("reason" in error and "missing model" in error for error in errors)


def test_validate_routes_fails_when_inventory_is_empty() -> None:
    data = load_routes(DEFAULT_ROUTES_PATH)
    ok, errors = validate_routes(data, available_models=set())
    assert ok is False
    assert errors and "could not enumerate" in errors[0]


def test_validate_routes_checks_decision_model_and_council_roles() -> None:
    data = load_routes(DEFAULT_ROUTES_PATH)
    available = {cfg["model"] for cfg in data["routes"].values() if cfg["backend"] == "opencode"}
    available |= {cfg["fallback"] for cfg in data["routes"].values() if cfg.get("fallback")}

    broken = dict(
        data,
        decision_model="opencode/ghost-model",
        council_roles={"pioneer": "code", "ghost": "missing-route"},
    )
    ok, errors = validate_routes(broken, available_models=available)

    assert ok is False
    assert "decision_model pins missing model 'opencode/ghost-model'" in errors
    assert "council role 'ghost' references unknown route 'missing-route'" in errors
    assert not any("council role 'pioneer'" in error for error in errors)


@pytest.fixture()
def local_http_server():
    """In-process HTTP server mocking OpenAI-compatible and Ollama model lists."""

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
            if self.path == "/api/tags":
                body = {"models": [{"name": "llama3.2:latest"}]}
            elif self.path == "/v1/models":
                body = {"data": [{"id": "local-model"}]}
            else:
                self.send_error(404)
                return
            payload = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _dead_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_probe_local_endpoints_reports_reachability_and_models(local_http_server: str) -> None:
    data = {
        "routes": {
            "local": {
                "backend": "openai-compat",
                "model": "local-model",
                "endpoint": f"{local_http_server}/v1/chat/completions",
            },
            "ollama": {
                "backend": "ollama",
                "model": "llama3.2",
                "endpoint": f"{local_http_server}/api/chat",
            },
            "dead": {
                "backend": "ollama",
                "model": "llama3.2",
                "endpoint": f"http://127.0.0.1:{_dead_port()}/api/chat",
            },
            "cloud": {"backend": "opencode", "model": "opencode/x"},
        }
    }

    probes = probe_local_endpoints(data, timeout_s=1.0)

    assert set(probes) == {"local", "ollama", "dead"}
    assert probes["local"]["reachable"] is True
    assert probes["local"]["model_present"] is True
    assert probes["local"]["endpoint"] == f"{local_http_server}/v1/chat/completions"
    assert probes["ollama"]["reachable"] is True
    assert probes["ollama"]["model_present"] is True  # llama3.2 matches llama3.2:latest
    assert probes["dead"]["reachable"] is False
    assert probes["dead"]["model_present"] is False
    assert probes["dead"]["detail"]


def test_probe_local_endpoints_supports_legacy_colibri_backend(local_http_server: str) -> None:
    """The legacy backend alias still uses the OpenAI-compatible model endpoint."""
    data = {
        "routes": {
            "legacy-local": {
                "backend": "colibri",
                "model": "local-model",
                "endpoint": f"{local_http_server}/v1/chat/completions",
            },
        },
    }

    probes = probe_local_endpoints(data, timeout_s=1.0)

    assert probes["legacy-local"]["reachable"] is True
    assert probes["legacy-local"]["model_present"] is True
    assert probes["legacy-local"]["detail"] == "1 model(s) advertised"


def test_validate_routes_check_local_flags_unreachable_and_missing(local_http_server: str) -> None:
    data = {
        "default_route": "code",
        "decision_model": "opencode/x",
        "routes": {
            "code": {"backend": "opencode", "model": "opencode/x"},
            "local": {
                "backend": "openai-compat",
                "model": "local-model",
                "endpoint": f"{local_http_server}/v1/chat/completions",
            },
            "ollama": {
                "backend": "ollama",
                "model": "absent-model",
                "endpoint": f"{local_http_server}/api/chat",
            },
            "dead": {
                "backend": "ollama",
                "model": "llama3.2",
                "endpoint": f"http://127.0.0.1:{_dead_port()}/api/chat",
            },
        },
    }

    ok, errors = validate_routes(data, available_models={"opencode/x"}, check_local=True)

    assert ok is False
    assert any("'dead'" in error and "unreachable" in error for error in errors)
    assert any("'ollama'" in error and "missing model 'absent-model'" in error for error in errors)
    assert not any("'local'" in error for error in errors)


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("triage these build logs", "triage"),
        ("implement a csv parser", "code"),
        ("write a quick one-liner snippet", "fast-code"),
        ("refactor this module across multi-file boundaries", "refactor"),
        ("audit the security of this spec", "reason"),
        ("evaluate the tradeoffs and compare options", "free-reason"),
        ("summarize this long document transcript", "context"),
        ("describe the screenshot of the ui", "vision"),
    ],
)
def test_route_accuracy_from_keywords(prompt: str, expected: str) -> None:
    """Route-accuracy check straight from intent text (cheapest eval per Anthropic's agent guidance)."""
    key, _, _ = resolve_route(prompt)
    assert key == expected


def test_local_ollama_and_premium_are_explicit_only() -> None:
    """Sensitive routes require an explicit prefix, never a keyword match."""
    for prompt, unexpected in (
        ("run this offline on an OpenAI-compatible server", "local"),
        ("use ollama for this", "ollama"),
        ("use a premium model for this", "premium"),
    ):
        key, _, _ = resolve_route(prompt)
        assert key != unexpected

    for prefix, expected_backend in (
        ("@local summarize the README", "openai-compat"),
        ("@ollama reverse engineer this binary", "ollama"),
        ("@premium reverse engineer this binary", "agy"),
    ):
        key, cfg, _ = resolve_route(prefix)
        assert key == prefix.split()[0][1:]
        assert cfg["backend"] == expected_backend


def test_list_opencode_models_prefers_fresh_cache(tmp_path: Path) -> None:
    cache = tmp_path / "models.json"
    cache.write_text(
        json.dumps({"ts": 9_999_999_999, "models": ["opencode/cached-a", "opencode/cached-b"]}),
        encoding="utf-8",
    )
    models = list_opencode_models(cache_path=cache, ttl_s=3600)
    assert models == {"opencode/cached-a", "opencode/cached-b"}


# --------------------------------------------------------------------------- budgets + precedence


def test_budgets_default_and_override() -> None:
    data = load_routes(DEFAULT_ROUTES_PATH)
    budgets = load_budgets(data)
    assert budgets["max_spans_per_trace"] == 32
    assert budgets["max_children"] == 8

    custom = dict(data, budgets={"max_spans_per_trace": 4, "max_children": 2, "max_depth": 1})
    tightened = load_budgets(custom)
    assert tightened["max_spans_per_trace"] == 4


def test_budget_allows_rejects_overruns() -> None:
    budgets = {"max_spans_per_trace": 32, "max_children": 8, "max_depth": 2}
    assert budget_allows(budgets, spans=2, children=1, depth=1)[0] is True

    ok, why = budget_allows(budgets, spans=32)
    assert ok is False and "span budget" in why
    ok, why = budget_allows(budgets, children=9)
    assert ok is False and "max_children" in why
    ok, why = budget_allows(budgets, depth=2)
    assert ok is False and "max_depth" in why


def test_precedence_breaks_keyword_ties() -> None:
    """Two routes matching equally should not silently fall to dict order."""
    routes_data = {
        "default_route": "code",
        "precedence": ["vision", "code"],
        "routes": {
            "code": {"backend": "opencode", "model": "opencode/x", "keywords": ["diagram"]},
            "vision": {"backend": "opencode", "model": "opencode/y", "keywords": ["diagram"]},
        },
    }
    key, _, _ = resolve_route("draw a diagram", routes_data=routes_data)
    assert key == "vision"


# --------------------------------------------------------------------------- router verbosity / thinking indicators


def test_front_desk_decide_explicit_route_no_llm() -> None:
    """Explicit route bypasses the LLM decision call entirely (deterministic, no subprocess)."""
    decision = front_desk_decide("anything at all", explicit_route="reason", use_llm=False)
    assert decision.orchestrator == "opencode"
    assert decision.route_key == "reason"
    assert decision.take_charge is True
    assert decision.reason == "explicit @reason override"
    assert decision.cleaned_prompt == "anything at all"
    assert decision.decision_model == "opencode/nemotron-3.5-lightning-free"


def test_front_desk_decide_explicit_route_premium_backend() -> None:
    """An explicit premium route still selects its external CLI orchestrator."""
    decision = front_desk_decide("reverse engineer this", explicit_route="premium", use_llm=False)
    assert decision.orchestrator == "agy"
    assert decision.route_key == "premium"
    assert decision.take_charge is True


@pytest.mark.parametrize("explicit_route", ["local", "ollama", "premium"])
def test_front_desk_decide_excludes_explicit_only_routes(monkeypatch, explicit_route: str) -> None:
    """The LLM catalog and its response guard both exclude explicit-only routes."""
    seen: dict[str, str] = {}
    selected = json.dumps({
        "orchestrator": "opencode",
        "route": explicit_route,
        "take_charge": True,
        "reason": "attempt explicit-only selection",
    })
    reply = json.dumps({"type": "text", "part": {"type": "text", "text": selected}})

    def _fake(argv, **_kwargs):
        seen["prompt"] = argv[-1]
        return 0, reply + "\n", "", False

    monkeypatch.setattr("nexus.backends._run_subprocess_bounded", _fake)
    decision = front_desk_decide("zz no matching route keywords", use_llm=True)

    assert f"{explicit_route} (" not in seen["prompt"]
    assert decision.route_key == "code"


def test_front_desk_decide_unknown_explicit_route_raises() -> None:
    """Unknown explicit route raises KeyError — never silently falls back."""
    with pytest.raises(KeyError):
        front_desk_decide("hello", explicit_route="nope", use_llm=False)


def test_list_opencode_models_cache_miss_mocks_subprocess(monkeypatch, tmp_path: Path) -> None:
    """Cache-miss path calls `opencode models`; mock subprocess to avoid live network/subprocess."""
    cache = tmp_path / "models.json"

    class _FakeProc:
        stdout = "opencode/mock-a\nopencode/mock-b\n"

    def _fake_run(*_args, **_kwargs):
        return _FakeProc()

    monkeypatch.setattr("subprocess.run", _fake_run)
    models = list_opencode_models(cache_path=cache, refresh_cache=True)
    assert models == {"opencode/mock-a", "opencode/mock-b"}

    # Cache file written with timestamp + sorted model list
    import json
    data = json.loads(cache.read_text(encoding="utf-8"))
    assert data["models"] == ["opencode/mock-a", "opencode/mock-b"]
    assert "ts" in data


def test_front_desk_decide_skips_llm_when_already_cancelled(monkeypatch) -> None:
    """A pre-set cancel event must never spawn the decision subprocess."""
    calls: list[list[str]] = []

    def _fake(argv, **_kwargs):
        calls.append(argv)
        return 0, "", "", False

    monkeypatch.setattr("nexus.backends._run_subprocess_bounded", _fake)

    cancelled = threading.Event()
    cancelled.set()
    decision = front_desk_decide("implement a csv parser", cancel_event=cancelled)

    assert calls == []
    assert decision.route_key == "code"
    assert decision.take_charge is True
    assert "cancel" in decision.reason


def test_front_desk_decide_passes_auto_and_stops_when_cancelled(monkeypatch) -> None:
    """The decision call is bounded, cancellable and runs `opencode run --auto`."""
    seen: dict[str, list[str]] = {}

    def _fake(argv, *, timeout_s, cancel_event):
        seen["argv"] = argv
        cancel_event.set()
        return 0, "", "", False

    monkeypatch.setattr("nexus.backends._run_subprocess_bounded", _fake)

    cancel = threading.Event()
    decision = front_desk_decide("implement a csv parser", cancel_event=cancel)

    assert "--auto" in seen["argv"]
    assert "--archived" in seen["argv"]
    assert decision.take_charge is True
    assert "cancel" in decision.reason


def test_front_desk_decide_parses_json_and_cached_tokens(monkeypatch) -> None:
    reply = json.dumps(
        {
            "type": "text",
            "part": {
                "type": "text",
                "text": '{"orchestrator":"council","route":"free-reason","take_charge":true,"reason":"debate"}',
            },
        }
    )
    usage = json.dumps(
        {"type": "step_finish", "part": {"tokens": {"input": 10, "output": 5, "cache": {"read": 90}}}}
    )
    monkeypatch.setattr(
        "nexus.backends._run_subprocess_bounded",
        lambda *args, **kwargs: (0, f"{reply}\n{usage}\n", "", False),
    )

    decision = front_desk_decide("evaluate the tradeoffs between options")

    assert decision.orchestrator == "council"
    assert decision.route_key == "free-reason"
    assert decision.tokens_in == 100  # 10 prompt + 90 cache-read tokens
    assert decision.tokens_out == 5
    assert decision.reason == "debate"
