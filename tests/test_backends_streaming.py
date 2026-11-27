"""Tests for live streaming callbacks, stderr suppression, cancellation, and TUI state."""

from __future__ import annotations

import base64
import contextlib
import http.server
import json
import os
from pathlib import Path
import sys
import threading
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nexus.backends as backends
import nexus.sessions as sessions
from nexus.events import read_events


def _cfg(backend: str = "opencode") -> dict:
    return {
        "backend": backend,
        "model": "opencode/mock-free",
        "quota_tier": "FREE",
        "timeout_s": 5,
    }


def _patch_binary(monkeypatch) -> None:
    monkeypatch.setattr(backends.shutil, "which", lambda name: f"/usr/bin/{name}")


def _stream_run(lines: list[str]):
    """Return a fake `_run_subprocess_bounded` that replays JSON lines to the callback."""

    def _run(argv, timeout_s, on_stdout_line=None, **_kwargs):
        for line in lines:
            if on_stdout_line is not None:
                on_stdout_line(line)
        return 0, "\n".join(lines) + "\n", "", False

    return _run


@contextlib.contextmanager
def _http_stream(lines: list[str], *, hold: float = 0.0, delay: float = 0.0):
    """Serve scripted response lines over a local HTTP server; yield (url, captured)."""
    captured: list[dict] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            length = int(self.headers.get("Content-Length", "0"))
            captured.append(json.loads(self.rfile.read(length).decode("utf-8")))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            try:
                for line in lines:
                    self.wfile.write((line + "\n").encode("utf-8"))
                    self.wfile.flush()
                    if delay:
                        time.sleep(delay)
                if hold:
                    time.sleep(hold)
            except OSError:
                pass
            self.close_connection = True

        def log_message(self, *args) -> None:
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/api/chat", captured
    finally:
        server.shutdown()
        server.server_close()


def _http_cfg(backend: str, endpoint: str, **extra) -> dict:
    cfg = {
        "backend": backend,
        "model": "llama3.2",
        "quota_tier": "LOCAL",
        "timeout_s": 5,
        "endpoint": endpoint,
    }
    cfg.update(extra)
    return cfg


def test_on_stream_receives_session_thinking_tool_and_text(monkeypatch, tmp_path: Path) -> None:
    _patch_binary(monkeypatch)
    lines = [
        json.dumps({"sessionID": "ses_stream01", "type": "text", "part": {"type": "text", "text": "hello "}}),
        json.dumps({"type": "reasoning", "part": {"type": "reasoning", "text": "thinking hard"}}),
        json.dumps({"type": "tool", "part": {"type": "tool", "tool": "bash", "state": {"title": "Run tests"}}}),
        json.dumps({"type": "text", "part": {"type": "text", "text": "world"}}),
    ]
    monkeypatch.setattr(backends, "_run_subprocess_bounded", _stream_run(lines))

    events: list[tuple[str, str]] = []
    res = backends.execute_route(
        "tr_stream01",
        "code",
        _cfg(),
        "hi",
        events_path=tmp_path / "events.jsonl",
        on_stream=lambda kind, payload: events.append((kind, payload)),
        emit_stderr=False,
    )

    assert res.status == "DONE"
    kinds = [kind for kind, _ in events]
    assert "session" in kinds
    assert "thinking" in kinds
    assert "tool" in kinds
    assert kinds.count("text") >= 2

    assert ("session", "ses_stream01") in events
    assert ("thinking", "thinking hard") in events
    assert any(kind == "tool" and "bash" in payload and "Run tests" in payload for kind, payload in events)
    text_payloads = [payload for kind, payload in events if kind == "text"]
    assert any("hello " in payload for payload in text_payloads)
    assert any("world" in payload for payload in text_payloads)

    # The event bus still records the span even with stderr suppressed.
    emitted = read_events(path=tmp_path / "events.jsonl", trace_id="tr_stream01")
    assert any(event.get("status") == "DONE" for event in emitted)


def test_emit_stderr_false_writes_nothing_to_stderr(monkeypatch, capsys, tmp_path: Path) -> None:
    _patch_binary(monkeypatch)
    lines = [
        json.dumps({"type": "tool", "part": {"type": "tool", "tool": "read", "state": {"title": "Open file"}}}),
        json.dumps({"type": "text", "part": {"type": "text", "text": "done"}}),
    ]
    monkeypatch.setattr(backends, "_run_subprocess_bounded", _stream_run(lines))

    capsys.readouterr()
    res = backends.execute_route(
        "tr_quiet_stream",
        "code",
        _cfg(),
        "hi",
        events_path=tmp_path / "events.jsonl",
        emit_stderr=False,
    )
    captured = capsys.readouterr()

    assert res.status == "DONE"
    assert captured.err == "", f"emit_stderr=False leaked status output: {captured.err!r}"
    assert captured.out == ""
    emitted = read_events(path=tmp_path / "events.jsonl", trace_id="tr_quiet_stream")
    assert any(event.get("status") == "DONE" for event in emitted)
    assert any(str(event.get("step", "")).startswith("tool:") for event in emitted)


def test_dry_run_emits_text_on_stream(monkeypatch, tmp_path: Path) -> None:
    events: list[tuple[str, str]] = []
    res = backends.execute_route(
        "tr_dry_stream",
        "code",
        _cfg(),
        "hello dry",
        events_path=tmp_path / "events.jsonl",
        dry_run=True,
        on_stream=lambda kind, payload: events.append((kind, payload)),
        emit_stderr=False,
    )

    assert res.status == "DONE"
    assert res.output.startswith("[dry-run]")
    assert events == [("text", res.output)]


def test_cancel_event_stops_slow_subprocess(monkeypatch, tmp_path: Path) -> None:
    """A real child process is terminated promptly when the cancel event fires."""
    fake_bin = tmp_path / "opencode"
    fake_bin.write_text(
        f"#!{sys.executable}\n"
        "import time\n"
        "print('{\"type\":\"text\",\"part\":{\"type\":\"text\",\"text\":\"started\"}}', flush=True)\n"
        "time.sleep(10)\n",
        encoding="utf-8",
    )
    fake_bin.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")

    cancel = threading.Event()
    timer = threading.Timer(0.4, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        res = backends.execute_route(
            "tr_cancel01",
            "code",
            _cfg(),
            "hi",
            events_path=tmp_path / "events.jsonl",
            cancel_event=cancel,
            emit_stderr=False,
        )
    finally:
        timer.cancel()
    elapsed = time.monotonic() - started

    assert res.status == "INTERRUPTED"
    assert res.error == "Cancelled by user"
    assert elapsed < 2.5, f"cancellation took {elapsed:.1f}s"

    emitted = read_events(path=tmp_path / "events.jsonl", trace_id="tr_cancel01")
    terminal = emitted[-1]
    assert terminal["status"] == "FAIL"
    assert terminal["step"] == "interrupted (Esc/Ctrl-C)"
    assert terminal["error"] == "Cancelled by user"


def test_sessions_save_load_roundtrip_and_trimming(tmp_path: Path) -> None:
    path = tmp_path / "tui_sessions.json"
    session = sessions.new_session("Audit run")
    session.total_tokens_in = 42
    session.items.append(
        sessions.TranscriptItem(
            kind="user",
            content="hello",
            route="code",
            trace_id="tr_round01",
            tools=["read"],
        )
    )
    sessions.save_sessions([session], path=path)

    loaded = sessions.load_sessions(path=path)
    assert len(loaded) == 1
    assert loaded[0].id == session.id
    assert loaded[0].title == "Audit run"
    assert loaded[0].total_tokens_in == 42
    assert loaded[0].items[0].content == "hello"
    assert loaded[0].items[0].tools == ["read"]

    base = time.time()
    many = []
    for index in range(5):
        item = sessions.new_session(f"session {index}")
        item.updated_at = base + index
        many.append(item)
    sessions.save_sessions(many, path=path, max_sessions=2)
    trimmed = sessions.load_sessions(path=path)
    assert [item.title for item in trimmed] == ["session 3", "session 4"]


def test_sessions_corrupt_and_missing_files(tmp_path: Path) -> None:
    path = tmp_path / "tui_sessions.json"
    assert sessions.load_sessions(path=path) == []

    path.write_text("{not valid json", encoding="utf-8")
    assert sessions.load_sessions(path=path) == []

    path.write_text("\"not a list\"", encoding="utf-8")
    assert sessions.load_sessions(path=path) == []


def test_tui_config_roundtrip_and_defaults(tmp_path: Path) -> None:
    path = tmp_path / "tui.json"

    defaults = sessions.load_tui_config(path=path)
    assert defaults == {
        "theme": "nexus-charcoal",
        "sidebar_open": True,
        "use_llm_decision": True,
        "verbosity": 1,
        "profile": "default",
        "entity_mode": True,
        "ui_mode": "webui",
    }

    sessions.save_tui_config({"theme": "solar", "verbosity": 2}, path=path)
    merged = sessions.load_tui_config(path=path)
    assert merged["theme"] == "solar"
    assert merged["verbosity"] == 2
    assert merged["sidebar_open"] is True

    path.write_text("{broken", encoding="utf-8")
    assert sessions.load_tui_config(path=path)["theme"] == "nexus-charcoal"

    assert list(tmp_path.glob("*.tmp")) == [], "atomic writes must not leave temp files behind"


# --------------------------------------------------------------------------- opencode fallback


def test_infer_fallback_tier() -> None:
    assert backends._infer_fallback_tier("opencode/x-free", "GO") == "FREE"
    assert backends._infer_fallback_tier("opencode-go/x", "FREE") == "GO"
    assert backends._infer_fallback_tier("mystery/model", "LOCAL") == "LOCAL"
    assert backends._infer_fallback_tier("mystery/model", "PREMIUM") == "FREE"


def test_infer_fallback_tier_supports_legacy_premium_alias() -> None:
    assert backends._infer_fallback_tier("mystery/model", "ARGON-QUOTA") == "FREE"


def test_opencode_fallback_success_reports_fallback_model_and_tier(monkeypatch, tmp_path: Path) -> None:
    _patch_binary(monkeypatch)
    calls: list[list[str]] = []

    def _run(argv, timeout_s, on_stdout_line=None, **_kwargs):
        calls.append(list(argv))
        if len(calls) == 1:
            return 1, "", '{"type":"error","error":{"data":{"message":"no route"}}}\n', False
        line = json.dumps({"type": "text", "part": {"type": "text", "text": "recovered"}})
        on_stdout_line(line)
        return 0, line + "\n", "", False

    monkeypatch.setattr(backends, "_run_subprocess_bounded", _run)
    cfg = {
        "backend": "opencode",
        "model": "opencode/mock-free",
        "fallback": "opencode-go/mock-go",
        "quota_tier": "FREE",
        "timeout_s": 5,
    }
    events: list[tuple[str, str]] = []
    res = backends.execute_route(
        "tr_fb01",
        "code",
        cfg,
        "hi",
        events_path=tmp_path / "events.jsonl",
        on_stream=lambda kind, payload: events.append((kind, payload)),
        emit_stderr=False,
    )

    assert len(calls) == 2
    assert "--auto" in calls[0]
    assert res.status == "DONE"
    assert res.output == "recovered"
    assert res.model == "opencode-go/mock-go"
    assert res.quota_tier == "GO"
    assert ("status", "fallback -> opencode-go/mock-go (GO)") in events

    emitted = read_events(path=tmp_path / "events.jsonl", trace_id="tr_fb01")
    assert any(str(event.get("step", "")).startswith("fallback->") for event in emitted)


def test_opencode_fallback_runs_at_most_once(monkeypatch, tmp_path: Path) -> None:
    _patch_binary(monkeypatch)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        backends,
        "_run_subprocess_bounded",
        lambda argv, timeout_s, on_stdout_line=None, **k: (calls.append(list(argv)) or (1, "", "boom", False)),
    )
    cfg = {
        "backend": "opencode",
        "model": "opencode/mock-free",
        "fallback": "opencode-go/mock-go",
        "quota_tier": "FREE",
        "timeout_s": 5,
    }
    res = backends.execute_route("tr_fb02", "code", cfg, "hi", events_path=tmp_path / "e.jsonl", emit_stderr=False)

    assert len(calls) == 2
    assert res.status == "FAIL"
    assert res.error == "boom"


def test_opencode_no_fallback_on_timeout_or_interrupt(monkeypatch, tmp_path: Path) -> None:
    _patch_binary(monkeypatch)
    cfg = {
        "backend": "opencode",
        "model": "opencode/mock-free",
        "fallback": "opencode-go/mock-go",
        "quota_tier": "FREE",
        "timeout_s": 5,
    }

    calls: list[list[str]] = []
    monkeypatch.setattr(
        backends,
        "_run_subprocess_bounded",
        lambda argv, timeout_s, on_stdout_line=None, **k: (calls.append(list(argv)) or (0, "", "", True)),
    )
    res = backends.execute_route("tr_fb03", "code", cfg, "hi", events_path=tmp_path / "e1.jsonl", emit_stderr=False)
    assert res.status == "TIMEOUT"
    assert len(calls) == 1

    calls.clear()
    monkeypatch.setattr(
        backends,
        "_run_subprocess_bounded",
        lambda argv, timeout_s, on_stdout_line=None, **k: (calls.append(list(argv)) or (0, "", "", False)),
    )
    cancel = threading.Event()
    cancel.set()
    res = backends.execute_route(
        "tr_fb04", "code", cfg, "hi", events_path=tmp_path / "e2.jsonl", emit_stderr=False, cancel_event=cancel
    )
    assert res.status == "INTERRUPTED"
    assert len(calls) == 1


def test_opencode_allow_fallback_false(monkeypatch, tmp_path: Path) -> None:
    _patch_binary(monkeypatch)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        backends,
        "_run_subprocess_bounded",
        lambda argv, timeout_s, on_stdout_line=None, **k: (calls.append(list(argv)) or (1, "", "boom", False)),
    )
    cfg = {
        "backend": "opencode",
        "model": "opencode/mock-free",
        "fallback": "opencode-go/mock-go",
        "quota_tier": "FREE",
        "timeout_s": 5,
    }
    res = backends.execute_route(
        "tr_fb05", "code", cfg, "hi", events_path=tmp_path / "e.jsonl", emit_stderr=False, allow_fallback=False
    )
    assert res.status == "FAIL"
    assert len(calls) == 1


# --------------------------------------------------------------------------- humanize errors


def test_humanize_error_scans_multiline_json_stream() -> None:
    blob = "\n".join(
        [
            '{"type":"step_start","part":{"type":"step-start"}}',
            '{"type":"text","part":{"type":"text","text":"hello"}}',
            '{"type":"error","error":{"type":"provider","data":{"message":"Model unavailable"}}}',
        ]
    )
    assert backends._humanize_error(blob) == "Model unavailable"

    flat = '{"type":"step_start"}\n{"type":"error","message":"rate limited"}'
    assert backends._humanize_error(flat) == "rate limited"

    # no error JSON anywhere -> first line, exactly as before
    plain = '{"type":"step_start"}\nplain failure'
    assert backends._humanize_error(plain) == '{"type":"step_start"}'
    assert backends._humanize_error("") == ""
    assert backends._humanize_error("line one\nline two") == "line one"
    assert backends._humanize_error("x" * 400, limit=10) == "x" * 10


# --------------------------------------------------------------------------- opencode argv + markdown


def test_opencode_passes_auto_unless_disabled(monkeypatch, tmp_path: Path) -> None:
    _patch_binary(monkeypatch)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        backends,
        "_run_subprocess_bounded",
        lambda argv, timeout_s, on_stdout_line=None, **k: (calls.append(list(argv)) or (0, "", "", False)),
    )
    cfg = {"backend": "opencode", "model": "opencode/mock-free", "quota_tier": "FREE", "timeout_s": 5}
    backends.execute_route("tr_auto01", "code", cfg, "hi", events_path=tmp_path / "e1.jsonl", emit_stderr=False)
    assert "--auto" in calls[0]

    cfg["auto_approve"] = False
    backends.execute_route("tr_auto02", "code", cfg, "hi", events_path=tmp_path / "e2.jsonl", emit_stderr=False)
    assert "--auto" not in calls[1]


def test_markdown_headings_are_separated_across_tool_boundaries(monkeypatch, tmp_path: Path) -> None:
    _patch_binary(monkeypatch)
    lines = [
        json.dumps({"type": "text", "part": {"type": "text", "text": "Here is the patch:"}}),
        json.dumps({"type": "tool", "part": {"type": "tool", "tool": "edit", "state": {"title": "Apply patch"}}}),
        json.dumps({"type": "text", "part": {"type": "text", "text": "## Summary"}}),
        json.dumps({"type": "text", "part": {"type": "text", "text": "All done."}}),
    ]
    monkeypatch.setattr(backends, "_run_subprocess_bounded", _stream_run(lines))

    text_payloads: list[str] = []
    res = backends.execute_route(
        "tr_md01",
        "code",
        _cfg(),
        "hi",
        events_path=tmp_path / "e.jsonl",
        on_stream=lambda kind, payload: text_payloads.append(payload) if kind == "text" else None,
        emit_stderr=False,
    )

    assert res.output.startswith("Here is the patch:")
    assert "Here is the patch:\n\n## Summary" in res.output
    assert any(payload.startswith("\n\n## Summary") for payload in text_payloads)

    # A plain continuation still glues, and an existing paragraph break gains nothing.
    parsed = backends._parse_opencode_json_stream(
        "\n".join(
            [
                json.dumps({"type": "text", "part": {"type": "text", "text": "para\n\n"}}),
                json.dumps({"type": "tool", "part": {"type": "tool", "tool": "bash"}}),
                json.dumps({"type": "text", "part": {"type": "text", "text": "## Next"}}),
                json.dumps({"type": "text", "part": {"type": "text", "text": " tail"}}),
            ]
        )
    )
    assert parsed.text == "para\n\n## Next tail"


# --------------------------------------------------------------------------- ollama streaming


def _ndjson(obj: dict) -> str:
    return json.dumps(obj)


def test_ollama_streams_thinking_and_text_with_usage(monkeypatch, tmp_path: Path) -> None:
    with _http_stream(
        [
            _ndjson({"message": {"thinking": "step one "}, "done": False}),
            _ndjson({"message": {"thinking": "and two"}, "done": False}),
            _ndjson({"message": {"content": "answer "}, "done": False}),
            _ndjson(
                {"message": {"content": "done"}, "done": True, "prompt_eval_count": 11, "eval_count": 7}
            ),
        ]
    ) as (url, captured):
        events: list[tuple[str, str]] = []
        res = backends.execute_route(
            "tr_oll01",
            "ollama",
            _http_cfg("ollama", url),
            "hello",
            events_path=tmp_path / "e.jsonl",
            inject_square=False,
            think="medium",
            on_stream=lambda kind, payload: events.append((kind, payload)),
            emit_stderr=False,
        )

    assert res.status == "DONE"
    assert res.output == "answer done"
    assert (res.tokens_in, res.tokens_out) == (11, 7)
    assert [p for k, p in events if k == "thinking"] == ["step one ", "and two"]
    assert [p for k, p in events if k == "text"] == ["answer ", "done"]

    body = captured[0]
    assert body["stream"] is True
    assert body["think"] == "medium"
    assert body["messages"] == [{"role": "user", "content": "hello"}]


def test_ollama_think_normalization_and_config_default(monkeypatch, tmp_path: Path) -> None:
    assert backends._normalize_think(None) is None
    assert backends._normalize_think(False) is False
    assert backends._normalize_think(True) is True
    assert backends._normalize_think("off") is False
    assert backends._normalize_think("none") is False
    assert backends._normalize_think("false") is False
    assert backends._normalize_think("true") is True
    assert backends._normalize_think("HIGH") == "high"
    assert backends._normalize_think("xhigh") == "xhigh"
    assert backends._normalize_think("banana") is None

    lines = [_ndjson({"message": {"content": "ok"}, "done": True})]
    with _http_stream(lines) as (url, captured):
        base = _http_cfg("ollama", url)
        backends.execute_route(
            "tr_think01", "ollama", base, "p1", events_path=tmp_path / "e1.jsonl",
            inject_square=False, emit_stderr=False,
        )
        assert "think" not in captured[0]

        backends.execute_route(
            "tr_think02", "ollama", base, "p2", events_path=tmp_path / "e2.jsonl",
            inject_square=False, think="off", emit_stderr=False,
        )
        assert captured[1]["think"] is False

        backends.execute_route(
            "tr_think03", "ollama", {**base, "think": "medium"}, "p3",
            events_path=tmp_path / "e3.jsonl", inject_square=False, emit_stderr=False,
        )
        assert captured[2]["think"] == "medium"


def test_ollama_history_forwarding_and_truncation(monkeypatch, tmp_path: Path) -> None:
    lines = [_ndjson({"message": {"content": "ok"}, "done": True})]
    history = [{"role": "system", "content": "ignored"}]
    for i in range(20):
        history.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"msg-{i}"})
    history.append({"role": "tool", "content": "nope"})
    history.append({"role": "user", "content": "   "})

    with _http_stream(lines) as (url, captured):
        res = backends.execute_route(
            "tr_hist01", "ollama", _http_cfg("ollama", url), "current",
            events_path=tmp_path / "e1.jsonl", inject_square=False,
            history_messages=history, emit_stderr=False,
        )
    assert res.status == "DONE"
    sent = captured[0]["messages"]
    assert sent[-1] == {"role": "user", "content": "current"}
    assert [m["content"] for m in sent[:-1]] == [f"msg-{i}" for i in range(8, 20)]
    assert all(m["role"] in ("user", "assistant") for m in sent)

    big = [{"role": "user", "content": "x" * 1000} for _ in range(20)]
    with _http_stream(lines) as (url, captured):
        backends.execute_route(
            "tr_hist02", "ollama", _http_cfg("ollama", url), "current",
            events_path=tmp_path / "e2.jsonl", inject_square=False,
            history_messages=big, emit_stderr=False,
        )
    sent = captured[0]["messages"]
    assert sum(len(m["content"]) for m in sent[:-1]) <= backends._HISTORY_MAX_CHARS
    assert len(sent) - 1 <= backends._HISTORY_MAX_ITEMS


def test_ollama_image_attachment_is_base64(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "shot.png"
    payload_bytes = b"\x89PNG\r\n\x1a\nfake-image"
    image.write_bytes(payload_bytes)
    lines = [_ndjson({"message": {"content": "seen"}, "done": True})]

    with _http_stream(lines) as (url, captured):
        res = backends.execute_route(
            "tr_img01", "ollama", _http_cfg("ollama", url), "what is this?",
            events_path=tmp_path / "e.jsonl", inject_square=False,
            attach_file=image, emit_stderr=False,
        )

    assert res.status == "DONE"
    last = captured[0]["messages"][-1]
    assert last["content"] == "what is this?"
    assert last["images"] == [base64.b64encode(payload_bytes).decode("ascii")]


def test_ollama_text_attachment_is_prefixed(tmp_path: Path) -> None:
    note = tmp_path / "notes.txt"
    note.write_text("line one\nline two", encoding="utf-8")
    lines = [_ndjson({"message": {"content": "ok"}, "done": True})]

    with _http_stream(lines) as (url, captured):
        backends.execute_route(
            "tr_att01", "ollama", _http_cfg("ollama", url), "summarize",
            events_path=tmp_path / "e.jsonl", inject_square=False,
            attach_file=note, emit_stderr=False,
        )

    content = captured[0]["messages"][-1]["content"]
    assert content.startswith("[Attached file: notes.txt]\nline one\nline two")
    assert content.endswith("summarize")


def test_missing_attachment_fails_clearly(tmp_path: Path) -> None:
    res = backends.execute_route(
        "tr_att02", "ollama", _http_cfg("ollama", "http://127.0.0.1:9/api/chat"), "hi",
        events_path=tmp_path / "e.jsonl", inject_square=False,
        attach_file=tmp_path / "missing.png", emit_stderr=False,
    )
    assert res.status == "FAIL"
    assert res.error == f"attachment not found: {tmp_path / 'missing.png'}"


def test_ollama_cancel_aborts_fast_and_preserves_partial(tmp_path: Path) -> None:
    lines = [
        _ndjson({"message": {"content": "partial answer"}, "done": False}),
        _ndjson({"message": {"thinking": "more"}, "done": False}),
    ]
    cancel = threading.Event()
    timer = threading.Timer(0.3, cancel.set)
    started = time.monotonic()
    timer.start()
    try:
        with _http_stream(lines, hold=5.0) as (url, _captured):
            res = backends.execute_route(
                "tr_oc01", "ollama", _http_cfg("ollama", url, timeout_s=10), "hi",
                events_path=tmp_path / "e.jsonl", inject_square=False,
                cancel_event=cancel, emit_stderr=False,
            )
    finally:
        timer.cancel()
    elapsed = time.monotonic() - started

    assert res.status == "INTERRUPTED"
    assert res.error == "Cancelled by user"
    assert "partial answer" in res.output
    assert elapsed < 1.5, f"cancellation took {elapsed:.1f}s"


def test_ollama_timeout_preserves_partial_output(tmp_path: Path) -> None:
    lines = [_ndjson({"message": {"content": "partial answer"}, "done": False})]
    started = time.monotonic()
    with _http_stream(lines, hold=5.0) as (url, _captured):
        res = backends.execute_route(
            "tr_ot01", "ollama", _http_cfg("ollama", url, timeout_s=1), "hi",
            events_path=tmp_path / "e.jsonl", inject_square=False, emit_stderr=False,
        )
    elapsed = time.monotonic() - started

    assert res.status == "TIMEOUT"
    assert res.error == "Timed out after 1s"
    assert "partial answer" in res.output
    assert elapsed < 2.5, f"timeout took {elapsed:.1f}s"


# --------------------------------------------------------------------------- OpenAI-compatible streaming


def _exercise_openai_compatible_stream(tmp_path: Path, backend: str) -> None:
    lines = [
        'data: {"choices":[{"delta":{"reasoning_content":"think "}}]}',
        'data: {"choices":[{"delta":{"reasoning":"more"}}]}',
        'data: {"choices":[{"delta":{"content":"hello "}}]}',
        'data: {"choices":[{"delta":{"content":"world"}}],"usage":{"prompt_tokens":9,"completion_tokens":4}}',
        "data: [DONE]",
    ]
    with _http_stream(lines) as (url, captured):
        events: list[tuple[str, str]] = []
        res = backends.execute_route(
            "tr_openai01", "local", _http_cfg(backend, url), "hello",
            events_path=tmp_path / "e.jsonl", inject_square=False,
            on_stream=lambda kind, payload: events.append((kind, payload)),
            emit_stderr=False,
        )

    assert res.status == "DONE"
    assert res.output == "hello world"
    assert (res.tokens_in, res.tokens_out) == (9, 4)
    assert [p for k, p in events if k == "thinking"] == ["think ", "more"]
    assert [p for k, p in events if k == "text"] == ["hello ", "world"]
    assert captured[0]["stream"] is True


def test_openai_compat_streams_reasoning_and_content_with_usage(tmp_path: Path) -> None:
    _exercise_openai_compatible_stream(tmp_path, "openai-compat")


def test_legacy_colibri_backend_streams_openai_compatible_sse(tmp_path: Path) -> None:
    """The legacy backend alias retains the same SSE streaming contract."""
    _exercise_openai_compatible_stream(tmp_path, "colibri")


def test_openai_compat_accepts_non_sse_single_json_response(tmp_path: Path) -> None:
    lines = [
        json.dumps(
            {
                "choices": [{"message": {"content": "single shot"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }
        )
    ]
    with _http_stream(lines) as (url, _captured):
        res = backends.execute_route(
            "tr_openai02", "local", _http_cfg("openai-compat", url), "hi",
            events_path=tmp_path / "e.jsonl", inject_square=False, emit_stderr=False,
        )
    assert res.status == "DONE"
    assert res.output == "single shot"
    assert (res.tokens_in, res.tokens_out) == (3, 2)


# --------------------------------------------------------------------------- thinking visibility


def _fake_tty_stderr(writes: list[str]):
    class _FakeStderr:
        def isatty(self) -> bool:
            return True

        def write(self, s: str) -> int:
            writes.append(s)
            return len(s)

        def flush(self) -> None:
            return None

    return _FakeStderr()


def test_ollama_verbosity_streams_thinking_to_stderr_not_stdout(
    tmp_path: Path, capsys
) -> None:
    """Default verbosity (>= 1) mirrors reasoning to stderr while stdout stays pure answer text."""
    lines = [
        _ndjson({"message": {"thinking": "step one "}, "done": False}),
        _ndjson({"message": {"thinking": "and two"}, "done": False}),
        _ndjson({"message": {"content": "answer "}, "done": False}),
        _ndjson(
            {"message": {"content": "done"}, "done": True, "prompt_eval_count": 11, "eval_count": 7}
        ),
    ]
    with _http_stream(lines) as (url, _captured):
        res = backends.execute_route(
            "tr_think_live",
            "ollama",
            _http_cfg("ollama", url),
            "hello",
            events_path=tmp_path / "e.jsonl",
            inject_square=False,
            think="medium",
            color=False,
            verbosity=1,
        )
    captured = capsys.readouterr()

    assert res.status == "DONE"
    assert res.output == "answer done"
    assert res.thinking == "step one and two"
    assert captured.out == "", "thinking must never touch stdout"
    assert "  ┆ step one and two" in captured.err
    assert f"thought {backends.format_tokens(len(res.thinking))}c" in captured.err
    assert "\x1b[" not in captured.err, "color=False must emit no ANSI"


def test_opencode_verbosity_streams_thinking_and_respects_quiet(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """Reasoning is mirrored before tool/answer output by default (v1) and muted at v0."""
    _patch_binary(monkeypatch)
    lines = [
        json.dumps({"type": "reasoning", "part": {"type": "reasoning", "text": "look at the tests\n"}}),
        json.dumps({"type": "tool", "part": {"type": "tool", "tool": "bash", "state": {"title": "Run tests"}}}),
        json.dumps({"type": "text", "part": {"type": "text", "text": "all good"}}),
    ]
    monkeypatch.setattr(backends, "_run_subprocess_bounded", _stream_run(lines))
    capsys.readouterr()

    res = backends.execute_route(
        "tr_open_live",
        "code",
        _cfg(),
        "hi",
        events_path=tmp_path / "e.jsonl",
        color=False,
        verbosity=1,
    )
    loud = capsys.readouterr()

    assert res.status == "DONE"
    assert res.output == "all good"
    assert loud.out == ""
    assert "  ┆ look at the tests" in loud.err
    assert "thought " in loud.err
    assert loud.err.index("  ┆ look at the tests") < loud.err.index("tool 1:")

    res_quiet = backends.execute_route(
        "tr_open_quiet",
        "code",
        _cfg(),
        "hi",
        events_path=tmp_path / "e2.jsonl",
        color=False,
        verbosity=0,
    )
    quiet = capsys.readouterr()

    assert res_quiet.status == "DONE"
    assert quiet.err == "", f"verbosity=0 leaked stderr: {quiet.err!r}"
    assert quiet.out == ""


def test_spin_stderr_shows_thinking_preview_and_clears_width(monkeypatch) -> None:
    """A live state turns the spinner into a char-count preview without ANSI debris."""
    writes: list[str] = []
    monkeypatch.setattr(sys, "stderr", _fake_tty_stderr(writes))

    stop = threading.Event()
    spin = threading.Thread(
        target=backends._spin_stderr,
        args=("code", stop),
        kwargs={"color": False, "state": {"chars": 1500, "preview": "checking parser edge cases"}},
        daemon=True,
    )
    spin.start()

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not any("(1.5kc)" in w for w in writes):
        time.sleep(0.01)
    assert any("(1.5kc)" in w for w in writes), f"preview frame never drew: {writes!r}"
    assert any("checking parser edge cases" in w for w in writes)
    assert not any("\x1b[" in w for w in writes), "color=False must emit no ANSI"

    backends._stop_spin(stop, spin)
    assert not spin.is_alive()

    clear = writes[-1]
    assert set(clear) <= {"\r", " "}, f"final write is not a blanking clear: {clear!r}"
    widest = max(len(w.lstrip("\r")) for w in writes if "thinking" in w)
    assert len(clear) - 2 >= widest, "clear shorter than the widest spinner frame"


def test_agy_collects_thinking_without_on_stream(monkeypatch, tmp_path: Path, capsys) -> None:
    """agy still streams and returns reasoning when no on_stream callback is wired."""
    _patch_binary(monkeypatch)
    lines = [
        json.dumps({"event": "thinking", "text": "plan carefully"}),
        json.dumps(
            {
                "event": "result",
                "result": {"response": "final answer", "usage": {"input_tokens": 3, "output_tokens": 2}},
            }
        ),
    ]
    monkeypatch.setattr(backends, "_run_subprocess_bounded", _stream_run(lines))
    capsys.readouterr()

    res = backends.execute_route(
        "tr_agy_live",
        "code",
        _cfg("agy"),
        "hi",
        events_path=tmp_path / "e.jsonl",
        color=False,
        verbosity=2,
    )
    captured = capsys.readouterr()

    assert res.status == "DONE"
    assert res.thinking == "plan carefully"
    assert res.output == "final answer"
    assert (res.tokens_in, res.tokens_out) == (3, 2)
    assert captured.out == ""
    assert "  ┆ plan carefully" in captured.err
    assert "thought " in captured.err


# --------------------------------------------------------------------------- drop-folder delivery & tool-calling


@contextlib.contextmanager
def _http_rounds(rounds: list[list[str]]):
    """Serve one scripted line batch per sequential POST request; yield (url, captured)."""
    captured: list[dict] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            length = int(self.headers.get("Content-Length", "0"))
            captured.append(json.loads(self.rfile.read(length).decode("utf-8")))
            idx = min(len(captured) - 1, len(rounds) - 1)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for line in rounds[idx]:
                self.wfile.write((line + "\n").encode("utf-8"))
                self.wfile.flush()
            self.close_connection = True

        def log_message(self, *args) -> None:
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/api/chat", captured
    finally:
        server.shutdown()
        server.server_close()


def test_write_drop_file_enforces_sandbox_permissions_and_size(tmp_path: Path) -> None:
    import pytest
    from nexus import drops

    drop_root = tmp_path / "drops"
    script_res = drops.write_drop_file(
        "tools/run.py", "#!/usr/bin/env python3\nprint('hi')\n", drop_dir=drop_root
    )
    script_path = drop_root / "tools" / "run.py"
    assert script_res["executable"] is True
    assert (script_path.stat().st_mode & 0o777) == 0o755

    note_res = drops.write_drop_file("note.txt", "hello", executable=False, drop_dir=drop_root)
    assert note_res["executable"] is False
    assert ((drop_root / "note.txt").stat().st_mode & 0o777) == 0o644

    for bad in ("", "../escape.sh", "/etc/passwd", "~/secret", ".stfolder/meta", "a\\b.txt"):
        with pytest.raises(ValueError):
            drops.write_drop_file(bad, "x", drop_dir=drop_root)

    outside = tmp_path / "outside.txt"
    outside.write_text("target", encoding="utf-8")
    (drop_root / "link.txt").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        drops.write_drop_file("link.txt", "overwrite", drop_dir=drop_root)

    sym_dir = drop_root / "symdir"
    sym_dir.symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        drops.write_drop_file("symdir/bad.txt", "nope", drop_dir=drop_root)

    with pytest.raises(ValueError, match="limit"):
        drops.write_drop_file("huge.txt", "x" * (drops.MAX_DROP_BYTES + 1), drop_dir=drop_root)



def test_default_drop_dir_uses_local_data_directory() -> None:
    from nexus import drops

    assert drops.DEFAULT_DROP_DIR.as_posix().endswith(".local/share/nexus/drops")


def test_save_text_artifact_rejects_protected_paths(tmp_path: Path, monkeypatch) -> None:
    from nexus import drops
    import pytest

    env_root = tmp_path / "env-protected"
    arg_root = tmp_path / "argument-protected"
    monkeypatch.setenv("NEXUS_PROTECTED_PATHS", os.pathsep.join([str(env_root)]))

    env_target = env_root / "nested" / "leak.md"
    with pytest.raises(ValueError, match="protected path"):
        drops.save_text_artifact("notes", str(env_target), raw_path=True)

    arg_target = arg_root / "leak.md"
    with pytest.raises(ValueError, match="protected path"):
        drops.save_text_artifact(
            "notes",
            str(arg_target),
            raw_path=True,
            protected_paths=[arg_root],
        )

    assert not env_target.exists()
    assert not arg_target.exists()


def test_extract_code_block_prefers_largest_block_and_falls_back() -> None:
    from nexus.drops import extract_code_block

    reply = (
        "Here is the script:\n\n"
        "```python\nimport sys\nfor line in sys.stdin:\n    print(line.upper())\n```\n\n"
        "Run it via:\n```bash\npython3 upper.py\n```\n"
    )
    body, lang = extract_code_block(reply)
    assert lang == "python"
    assert "import sys" in body and "python3 upper.py" not in body

    plain_body, plain_lang = extract_code_block("  just plain answer  ")
    assert (plain_body, plain_lang) == ("just plain answer", "")


def test_ollama_send_file_tool_loop_writes_drop_and_streams_confirmation(tmp_path: Path) -> None:
    drop_root = tmp_path / "drops"
    rounds = [
        [
            _ndjson({"message": {"thinking": "writing script"}, "done": False}),
            _ndjson(
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "send_file",
                                    "arguments": {
                                        "filename": "hello.sh",
                                        "content": "#!/bin/sh\necho ok\n",
                                        "executable": True,
                                    },
                                },
                            }
                        ],
                    },
                    "done": True,
                    "prompt_eval_count": 10,
                    "eval_count": 5,
                }
            ),
        ],
        [
            _ndjson({"message": {"content": "Sent hello.sh to laptop."}, "done": True, "prompt_eval_count": 15, "eval_count": 4}),
        ],
    ]
    events: list[tuple[str, str]] = []
    with _http_rounds(rounds) as (url, captured):
        res = backends.execute_route(
            "tr_drop01",
            "ollama",
            _http_cfg("ollama", url),
            "send hello.sh",
            events_path=tmp_path / "e.jsonl",
            inject_square=False,
            drop_dir=drop_root,
            on_stream=lambda kind, payload: events.append((kind, payload)),
            emit_stderr=False,
        )

    assert res.status == "DONE"
    assert res.output == "Sent hello.sh to laptop."
    assert (res.tokens_in, res.tokens_out) == (25, 9)
    written = drop_root / "hello.sh"
    assert written.read_text(encoding="utf-8") == "#!/bin/sh\necho ok\n"
    assert (written.stat().st_mode & 0o777) == 0o755
    assert any(k == "tool" and "hello.sh" in p and "+x" in p for k, p in events)

    assert len(captured) == 2
    assert captured[0]["tools"][0]["function"]["name"] == "send_file"
    tool_msg = captured[1]["messages"][-1]
    assert tool_msg["role"] == "tool"
    assert json.loads(tool_msg["content"])["ok"] is True


def test_ollama_send_file_rejects_path_traversal_gracefully(tmp_path: Path) -> None:
    drop_root = tmp_path / "drops"
    rounds = [
        [
            _ndjson(
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_bad",
                                "function": {
                                    "name": "send_file",
                                    "arguments": {"filename": "../evil.sh", "content": "rm"},
                                },
                            }
                        ]
                    },
                    "done": True,
                }
            )
        ],
        [_ndjson({"message": {"content": "Blocked bad path."}, "done": True})],
    ]
    with _http_rounds(rounds) as (url, captured):
        res = backends.execute_route(
            "tr_drop_bad",
            "ollama",
            _http_cfg("ollama", url),
            "bad",
            events_path=tmp_path / "e.jsonl",
            inject_square=False,
            drop_dir=drop_root,
            emit_stderr=False,
        )

    assert res.status == "DONE"
    assert res.output == "Blocked bad path."
    assert not (tmp_path / "evil.sh").exists()
    tool_payload = json.loads(captured[1]["messages"][-1]["content"])
    assert tool_payload["ok"] is False


def _exercise_openai_compatible_send_file(tmp_path: Path, backend: str) -> None:
    drop_root = tmp_path / "drops"
    arg_str = json.dumps({"filename": "artifact.txt", "content": "saved content\n", "executable": False})
    rounds = [
        [
            f'data: {{"choices":[{{"delta":{{"tool_calls":[{{"index":0,"id":"c1","function":{{"name":"send_file","arguments":{json.dumps(arg_str[:18])}}}}}]}}}}]}}',
            f'data: {{"choices":[{{"delta":{{"tool_calls":[{{"index":0,"function":{{"arguments":{json.dumps(arg_str[18:])}}}}}]}}}}],"usage":{{"prompt_tokens":6,"completion_tokens":3}}}}',
            "data: [DONE]",
        ],
        [
            'data: {"choices":[{"delta":{"content":"Saved artifact.txt."}}],"usage":{"prompt_tokens":9,"completion_tokens":4}}',
            "data: [DONE]",
        ],
    ]
    with _http_rounds(rounds) as (url, captured):
        res = backends.execute_route(
            "tr_openai_drop",
            "local",
            _http_cfg(backend, url),
            "save the artifact",
            events_path=tmp_path / "e.jsonl",
            inject_square=False,
            drop_dir=drop_root,
            emit_stderr=False,
        )

    assert res.status == "DONE"
    assert res.output == "Saved artifact.txt."
    assert (res.tokens_in, res.tokens_out) == (15, 7)
    assert (drop_root / "artifact.txt").read_text(encoding="utf-8") == "saved content\n"
    assert len(captured) == 2
    assert captured[1]["messages"][-1]["role"] == "tool"


def test_openai_compat_sse_send_file_tool_loop(tmp_path: Path) -> None:
    _exercise_openai_compatible_send_file(tmp_path, "openai-compat")


def test_legacy_colibri_backend_sse_send_file_tool_loop(tmp_path: Path) -> None:
    """The legacy backend alias retains streamed send_file tool calls."""
    _exercise_openai_compatible_send_file(tmp_path, "colibri")


def test_classify_http_error_and_health_tracker_ewma(monkeypatch) -> None:
    assert backends.classify_http_error(401) == "auth"
    assert backends.classify_http_error("HTTP 403: forbidden") == "auth"
    assert backends.classify_http_error(429) == "quota"
    assert backends.classify_http_error("HTTP 402 insufficient credits") == "quota"
    assert backends.classify_http_error(503) == "transient"
    assert backends.classify_http_error("timed out waiting") == "transient"
    assert backends.classify_http_error("HTTP 400 bad request") == "other"

    tracker = backends.RouteHealthTracker(alpha=0.3)
    tracker.record_failure(endpoint="http://127.0.0.1:1/v1", route_key="r1", model="m1", kind="transient")
    # Transient affects endpoint scope (even a different route sharing the endpoint sees it).
    assert tracker.is_cooled_down(endpoint="http://127.0.0.1:1/v1", route_key="r2", model="m2")
    assert not tracker.is_cooled_down(endpoint="http://127.0.0.1:2/v1", route_key="r2", model="m2")

    tracker.record_failure(endpoint="http://127.0.0.1:2/v1", route_key="r2", model="m2", kind="quota")
    assert tracker.is_cooled_down(endpoint="http://127.0.0.1:2/v1", route_key="r2", model="m2")
    assert not tracker.is_cooled_down(endpoint="http://127.0.0.1:2/v1", route_key="r3", model="m2")

    tracker.record_failure(endpoint="http://127.0.0.1:3/v1", route_key="r3", model="m3", kind="other")
    assert tracker.is_cooled_down(endpoint="http://127.0.0.1:3/v1", route_key="r3", model="m3")
    assert not tracker.is_cooled_down(endpoint="http://127.0.0.1:3/v1", route_key="r3", model="m4")

    tracker.record_success(endpoint="http://127.0.0.1:3/v1", route_key="r3", model="m3", elapsed_ms=100.0)
    assert not tracker.is_cooled_down(endpoint="http://127.0.0.1:3/v1", route_key="r3", model="m3")
    assert abs((tracker.latency_ewma(endpoint="http://127.0.0.1:3/v1", route_key="r3", model="m3") or 0.0) - 100.0) < 1e-6
    tracker.record_success(endpoint="http://127.0.0.1:3/v1", route_key="r3", model="m3", elapsed_ms=200.0)
    assert abs((tracker.latency_ewma(endpoint="http://127.0.0.1:3/v1", route_key="r3", model="m3") or 0.0) - 130.0) < 1e-6

    backends.reset_health_state()


def _scripted_http_server(handler_fn):
    @contextlib.contextmanager
    def _ctx():
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8", errors="replace")
                handler_fn(self, raw)

            def log_message(self, *args) -> None:
                return

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            yield f"http://127.0.0.1:{srv.server_address[1]}/v1/chat/completions"
        finally:
            srv.shutdown()
            srv.server_close()

    return _ctx()


def test_http_bearer_auth_fallback_and_guards(tmp_path: Path) -> None:
    backends.reset_health_state()
    try:
        calls: list[dict] = []
        mode = {"case": "503_then_ok"}

        def handle(req, raw: str) -> None:
            path = req.path
            auth = str(req.headers.get("Authorization") or "")
            calls.append({"path": path, "auth": auth, "body": json.loads(raw)})
            if mode["case"] == "401":
                body = json.dumps({"error": {"message": "bad token"}}).encode("utf-8")
                req.send_response(401)
                req.send_header("Content-Type", "application/json")
                req.send_header("Content-Length", str(len(body)))
                req.end_headers()
                req.wfile.write(body)
                return
            if mode["case"] == "503_then_ok" and path.endswith("/primary"):
                body = json.dumps({"error": {"message": "upstream overloaded"}}).encode("utf-8")
                req.send_response(503)
                req.send_header("Content-Type", "application/json")
                req.send_header("Content-Length", str(len(body)))
                req.end_headers()
                req.wfile.write(body)
                return
            if mode["case"] == "partial_then_drop" and path.endswith("/primary"):
                req.send_response(200)
                req.send_header("Content-Type", "text/event-stream")
                req.end_headers()
                req.wfile.write(b'data: {"choices":[{"delta":{"content":"partial "}}]}\n')
                req.wfile.flush()
                req.wfile.write(b"not-valid-finish")
                return
            req.send_response(200)
            req.send_header("Content-Type", "text/event-stream")
            req.end_headers()
            req.wfile.write(b'data: {"choices":[{"delta":{"content":"fallback ok"}}]}\ndata: [DONE]\n')

        with _scripted_http_server(handle) as base_url:
            base = base_url.rsplit("/v1/", 1)[0]
            key_file = tmp_path / "gw.key"
            key_file.write_text("sk-gw-123\n", encoding="utf-8")
            routes_file = tmp_path / "routes.json"
            primary_cfg = {
                "backend": "openai-compat",
                "endpoint": f"{base}/primary",
                "model": "m-primary",
                "api_key_file": str(key_file),
                "quota_tier": "FREE",
                "timeout_s": 5,
                "fallback_route": "local_fb",
            }
            fb_cfg = {
                "backend": "openai-compat",
                "endpoint": f"{base}/fb",
                "model": "m-fb",
                "quota_tier": "LOCAL",
                "timeout_s": 5,
            }
            routes_file.write_text(json.dumps({"routes": {"omni": primary_cfg, "local_fb": fb_cfg}}), encoding="utf-8")

            # 1) Pre-first-byte 503 captures HTTP body & falls back to fallback_route cleanly.
            res1 = backends.execute_route(
                "tr_fb1", "omni", primary_cfg, "hello",
                events_path=tmp_path / "e1.jsonl", inject_square=False, emit_stderr=False, routes_path=routes_file,
            )
            assert res1.status == "DONE"
            assert res1.route == "local_fb"
            assert res1.output == "fallback ok"
            assert calls[0]["auth"] == "Bearer sk-gw-123"

            # 2) 401 auth error captures HTTP body and never retries or falls back blindly.
            backends.reset_health_state()
            calls.clear()
            mode["case"] = "401"
            res2 = backends.execute_route(
                "tr_auth", "omni", primary_cfg, "hello",
                events_path=tmp_path / "e2.jsonl", inject_square=False, emit_stderr=False, routes_path=routes_file,
            )
            assert res2.status == "FAIL"
            assert "401" in res2.error and "bad token" in res2.error
            assert len(calls) == 1

            # 3) Post-first-byte guard: partial output is returned instead of switching streams mid-reply.
            backends.reset_health_state()
            calls.clear()
            mode["case"] = "partial_then_drop"
            res3 = backends.execute_route(
                "tr_pfb", "omni", primary_cfg, "hello",
                events_path=tmp_path / "e3.jsonl", inject_square=False, emit_stderr=False, routes_path=routes_file,
            )
            assert res3.output == "partial"
            assert len(calls) == 1
    finally:
        backends.reset_health_state()


def test_bounded_history_clears_old_tool_output_keeping_recent_verbatim() -> None:
    bulky_tool = "[Tool output: ls -la]\n" + ("file-entry-line\n" * 30)
    history = [
        {"role": "user", "content": "old ask"},
        {"role": "assistant", "content": bulky_tool},
    ]
    for i in range(6):
        history.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"recent-{i} " + ("x" * 260)})
    kept = backends._bounded_history(history)
    assert "[cleared old tool output" in kept[1]["content"]
    assert kept[-1]["content"].startswith("recent-5 ")
    assert len(kept[-1]["content"]) > 240
