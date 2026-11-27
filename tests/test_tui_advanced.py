"""Advanced end-to-end tests for the Nexus Textual TUI and its CLI entrypoint.

Every test isolates persistent state under `tmp_path`: events, sessions, config,
town square, board and prompt history. The app is driven through
`app.run_test(...)` + `Pilot`, so key bindings, modal screens and the background
worker thread are exercised the same way a user would.
"""

from __future__ import annotations

import asyncio
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
import time

import pytest
from textual.command import CommandPalette
from textual.css.query import NoMatches
from textual.screen import Screen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nexus.backends import ExecutionResult
from nexus.events import read_events
import nexus.tui.app as tui_app
from nexus.tui.app import NexusApp
from nexus.tui.modals import HelpModal, ThemePickerModal
from nexus.tui.widgets import (
    ComposerBox,
    CouncilGridCard,
    ToolActivityGroup,
    UserMessageCard,
)


def _paths(tmp_path: Path) -> dict[str, Path]:
    """All persistent state the TUI can touch, isolated per test."""
    return {
        "events_path": tmp_path / "events.jsonl",
        "sessions_path": tmp_path / "tui_sessions.json",
        "config_path": tmp_path / "tui.json",
        "square_path": tmp_path / "square.jsonl",
        "board_path": tmp_path / "BOARD.md",
        "history_path": tmp_path / "history",
    }


def _make_app(tmp_path: Path, **overrides) -> NexusApp:
    paths = _paths(tmp_path)
    paths.update(overrides)
    return NexusApp(**paths)


async def _wait_for(predicate, timeout: float = 5.0) -> bool:
    """Yield to the Textual loop until `predicate` is true or time runs out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return bool(predicate())


async def _submit(pilot, app: NexusApp, text: str) -> None:
    """Put `text` in the composer and press Enter like a user would."""
    composer = app.query_one(ComposerBox)
    composer.input.load_text(text)
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


def _session(app: NexusApp, index: int):
    return app.sessions[index]


def _route_result(
    trace_id: str,
    route: str,
    route_cfg: dict,
    output: str,
    *,
    session_id: str = "",
) -> ExecutionResult:
    return ExecutionResult(
        trace_id=trace_id,
        span_id="span_test",
        route=route,
        backend=str(route_cfg.get("backend", "opencode")),
        model=str(route_cfg.get("model", "")),
        quota_tier=str(route_cfg.get("quota_tier", "FREE")),
        status="DONE",
        output=output,
        elapsed_ms=1,
        session_id=session_id,
    )


def test_tui_theme_picker_live_preview_and_persistence(tmp_path: Path) -> None:
    async def flow() -> None:
        paths = _paths(tmp_path)
        app = _make_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.theme == "nexus-charcoal"

            # Ctrl+T opens the picker; moving the highlight previews live.
            await pilot.press("ctrl+t")
            await pilot.pause()
            assert isinstance(app.screen, ThemePickerModal)
            await pilot.press("down")
            await pilot.pause()
            assert app.theme == "oled-black"

            # Enter applies the highlighted theme and persists it.
            await pilot.press("enter")
            await pilot.pause()
            assert not isinstance(app.screen, ThemePickerModal)
            assert app.theme == "oled-black"
            saved = json.loads(paths["config_path"].read_text(encoding="utf-8"))
            assert saved["theme"] == "oled-black"

            # A second open previews another theme; Esc restores the saved one.
            await pilot.press("ctrl+t")
            await pilot.pause()
            assert isinstance(app.screen, ThemePickerModal)
            await pilot.press("down")
            await pilot.pause()
            assert app.theme == "opencode-dark"
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, ThemePickerModal)
            assert app.theme == "oled-black"

    asyncio.run(flow())


def test_tui_multi_tab_creation_and_isolation(tmp_path: Path) -> None:
    async def flow() -> None:
        app = _make_app(tmp_path, dry_run=True)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            first_id = app.active_id
            assert len(app.sessions) == 1

            # Ctrl+N creates a second, active tab.
            await pilot.press("ctrl+n")
            await pilot.pause()
            assert len(app.sessions) == 2
            second_id = app.active_id
            assert second_id != first_id

            await _submit(pilot, app, "@code tab two")
            ready = await _wait_for(
                lambda: second_id not in app._busy
                and any(item.kind == "assistant" for item in _session(app, 1).items)
            )
            assert ready, "tab 2 never completed its dry-run turn"

            second = _session(app, 1)
            assert [item.kind for item in second.items][0] == "user"
            assert second.items[0].content == "@code tab two"
            assert second.active_route is None
            assert second.last_route == "code"
            assert second.id == second_id

            # Alt+Left returns to tab 1: empty transcript, no leaked route.
            await pilot.press("alt+left")
            await pilot.pause()
            await _wait_for(
                lambda: app.active_id == first_id
                and len(app.query(UserMessageCard)) == 0
            )
            first = _session(app, 0)
            assert app.active_id == first_id
            assert first.items == []
            assert first.active_route is None
            assert first.last_route is None
            assert len(app.query(UserMessageCard)) == 0

            # Alt+Right restores tab 2 with its own transcript and route state,
            # and /pin code locks only tab 2.
            await pilot.press("alt+right")
            await pilot.pause()
            await _wait_for(
                lambda: app.active_id == second_id
                and len(app.query(UserMessageCard)) == 1
            )
            await _submit(pilot, app, "/pin code")
            assert app.active_id == second_id
            assert len(app.query(UserMessageCard)) == 1
            assert _session(app, 1).active_route == "code"
            assert _session(app, 0).active_route is None

    asyncio.run(flow())


def test_tui_council_grid_renders_three_seats(tmp_path: Path) -> None:
    async def flow() -> None:
        app = _make_app(tmp_path, dry_run=True)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _submit(pilot, app, "/council Compare JSONL vs SQLite")

            def _ready() -> bool:
                try:
                    card = app.query_one(CouncilGridCard)
                except NoMatches:
                    return False
                receipts = [
                    str(widget.content)
                    for widget in card.query(".council-seat-receipt")
                ]
                return len(receipts) == 3 and all("DONE" in text for text in receipts)

            assert await _wait_for(_ready), "council seats never finalized"
            assert await _wait_for(
                lambda: any(item.kind == "council" for item in app.active_session.items)
            )

            card = app.query_one(CouncilGridCard)
            titles = [str(widget.content) for widget in card.query(".council-seat-title")]
            assert titles == ["PIONEER", "RED-TEAMER", "SYNTHESIZER"]

            receipts = [str(widget.content) for widget in card.query(".council-seat-receipt")]
            assert len(receipts) == 3
            assert all("DONE" in text for text in receipts)
            assert all("✗" not in text for text in receipts)

            council_item = next(
                item for item in app.active_session.items if item.kind == "council"
            )
            assert council_item.status == "DONE"
            assert {seat["seat"] for seat in council_item.council_seats} == {
                "pioneer",
                "red_teamer",
                "synthesizer",
            }
            events = read_events(
                path=app.events_path,
                trace_id=council_item.trace_id,
                limit=100,
            )
            root_events = [event for event in events if event.get("route") == "council"]
            assert {event.get("status") for event in root_events} == {"RUNNING", "DONE"}
            root_ids = {event.get("span_id") for event in root_events}
            assert len(root_ids) == 1
            root_id = root_ids.pop()
            children = [event for event in events if event.get("parent_id") == root_id]
            assert len({event.get("span_id") for event in children}) == 3

    asyncio.run(flow())


def test_tui_tool_toggle_and_esc_cancellation(tmp_path: Path, monkeypatch) -> None:
    async def tool_flow() -> None:
        app = _make_app(tmp_path / "tools", dry_run=True)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _submit(pilot, app, "@code tool activity test")

            await _wait_for(
                lambda: bool(app.query(ToolActivityGroup)) and not app._busy
            )
            groups = list(app.query(ToolActivityGroup))
            assert groups, "a completed dry-run turn should mount a tool group"
            group = groups[0]
            assert group.collapsed is True

            await pilot.press("ctrl+o")
            await pilot.pause()
            assert group.collapsed is False

            await pilot.press("ctrl+o")
            await pilot.pause()
            assert group.collapsed is True

    asyncio.run(tool_flow())

    observed: dict[str, object] = {"cancel_event": None}

    def slow_execute(trace_id, route_label, route_cfg, prompt, **kwargs):
        """Block until the app's cancel event is set, honouring it promptly."""
        cancel_event = kwargs.get("cancel_event")
        observed["cancel_event"] = cancel_event
        started = time.monotonic()
        while time.monotonic() - started < 5.0:
            if cancel_event is not None and cancel_event.is_set():
                return ExecutionResult(
                    trace_id,
                    "span_cancelled",
                    str(route_label),
                    str(route_cfg.get("backend", "")),
                    str(route_cfg.get("model", "")),
                    str(route_cfg.get("quota_tier", "FREE")),
                    "INTERRUPTED",
                    "",
                    0,
                    0,
                    int((time.monotonic() - started) * 1000),
                    "Interrupted by user (Ctrl-C)",
                )
            time.sleep(0.02)
        return ExecutionResult(
            trace_id,
            "span_timeout",
            str(route_label),
            str(route_cfg.get("backend", "")),
            str(route_cfg.get("model", "")),
            str(route_cfg.get("quota_tier", "FREE")),
            "TIMEOUT",
            "",
            elapsed_ms=5000,
            error="mock timeout",
        )

    async def cancel_flow() -> None:
        app = _make_app(tmp_path / "cancel", dry_run=False)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _submit(pilot, app, "@code slow task")

            busy = await _wait_for(
                lambda: app.active_id in app._busy
                and app.active_id in app._cancel_events,
                timeout=3.0,
            )
            assert busy, "the mocked route never started"

            started = time.monotonic()
            await pilot.press("escape")
            stopped = await _wait_for(
                lambda: app.active_id not in app._busy,
                timeout=2.0,
            )
            elapsed = time.monotonic() - started

            assert stopped, "Escape did not cancel the in-flight route"
            assert elapsed < 2.0
            cancel_event = observed["cancel_event"]
            assert cancel_event is not None and cancel_event.is_set()

            session = app.active_session
            assert session is not None
            assert session.items[-1].status == "INTERRUPTED"

            # The app is still alive and accepts key bindings after cancelling.
            before = app._sidebar_open
            await pilot.press("ctrl+b")
            await pilot.pause()
            assert app.is_running
            assert app._sidebar_open is not before

    monkeypatch.setattr(tui_app, "execute_route", slow_execute)
    asyncio.run(cancel_flow())


def test_tui_command_palette_and_help_modal(tmp_path: Path) -> None:
    async def flow() -> None:
        app = _make_app(tmp_path, dry_run=True)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            # Ctrl+P opens the Textual command palette; Escape dismisses it.
            await pilot.press("ctrl+p")
            await pilot.pause()
            assert isinstance(app.screen, CommandPalette)
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, Screen)
            assert not isinstance(app.screen, CommandPalette)

            # F1 opens the keymap / slash-command reference.
            await pilot.press("f1")
            await pilot.pause()
            assert isinstance(app.screen, HelpModal)
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, Screen)

            # "/help" is an equivalent path into the same modal.
            await _submit(pilot, app, "/help ")
            await pilot.pause()
            assert isinstance(app.screen, HelpModal)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, HelpModal)

    asyncio.run(flow())


def test_tui_responsive_narrow_terminal(tmp_path: Path) -> None:
    async def flow() -> None:
        app = _make_app(tmp_path, dry_run=True)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()

            def _sidebar_hidden() -> bool:
                try:
                    return (
                        not app._sidebar_open
                        and app.query_one("#sidebar").has_class("hidden")
                    )
                except NoMatches:
                    return False

            assert await _wait_for(_sidebar_hidden, timeout=2.0)
            assert app.size.width == 80

            await _submit(pilot, app, "/council narrow terminal layout")

            def _stacked() -> bool:
                try:
                    card = app.query_one(CouncilGridCard)
                    return card.query_one(".council-grid").has_class("stacked")
                except NoMatches:
                    return False

            assert await _wait_for(_stacked), "council grid did not stack at 80 cols"
            assert app.is_running
            assert app.query_one("#main-pane").size.width > 0

    asyncio.run(flow())


def test_tui_at_route_switch_does_not_leak_backend_session(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[dict[str, object]] = []

    def fake_execute(trace_id, route_label, route_cfg, prompt, **kwargs):
        session_id = "code-session" if route_label.endswith("code") else "reason-session"
        calls.append({"route": route_label, "session_id": kwargs.get("session_id")})
        on_stream = kwargs.get("on_stream")
        if on_stream is not None:
            on_stream("session", session_id)
            on_stream("text", f"{route_label} answer")
        return _route_result(
            trace_id,
            route_label,
            route_cfg,
            f"{route_label} answer",
            session_id=session_id,
        )

    monkeypatch.setattr(tui_app, "execute_route", fake_execute)

    async def flow() -> None:
        app = _make_app(tmp_path, dry_run=True)
        async with app.run_test(size=(120, 40)) as pilot:
            await _submit(pilot, app, "@code first turn")
            assert await _wait_for(lambda: len(calls) == 1 and not app._busy)
            assert app.active_session.backend_session_id == "code-session"

            await _submit(pilot, app, "@reason second turn")
            assert await _wait_for(lambda: len(calls) == 2 and not app._busy)
            assert calls[1]["session_id"] is None
            assert app.active_session.backend_session_id == "reason-session"

            await _submit(pilot, app, "@reason third turn")
            assert await _wait_for(lambda: len(calls) == 3 and not app._busy)
            assert calls[2]["session_id"] == "reason-session"

    asyncio.run(flow())


def test_tui_http_backend_receives_prior_transcript_history(
    tmp_path: Path, monkeypatch
) -> None:
    histories: list[list[dict]] = []

    def fake_execute(trace_id, route_label, route_cfg, prompt, **kwargs):
        histories.append(list(kwargs.get("history_messages") or []))
        output = f"answer {len(histories)}"
        on_stream = kwargs.get("on_stream")
        if on_stream is not None:
            on_stream("text", output)
        return _route_result(trace_id, route_label, route_cfg, output)

    monkeypatch.setattr(tui_app, "execute_route", fake_execute)

    async def flow() -> None:
        app = _make_app(tmp_path, dry_run=True)
        async with app.run_test(size=(120, 40)) as pilot:
            await _submit(pilot, app, "@local first question")
            assert await _wait_for(lambda: len(histories) == 1 and not app._busy)
            await _submit(pilot, app, "@local second question")
            assert await _wait_for(lambda: len(histories) == 2 and not app._busy)

        assert histories[0] == []
        assert histories[1] == [
            {"role": "user", "content": "@local first question"},
            {"role": "assistant", "content": "answer 1"},
        ]

    asyncio.run(flow())


def test_tui_think_and_attach_state_reach_exactly_one_prompt(
    tmp_path: Path, monkeypatch
) -> None:
    attachment = tmp_path / "notes.txt"
    attachment.write_text("useful context", encoding="utf-8")
    calls: list[dict] = []

    def fake_execute(trace_id, route_label, route_cfg, prompt, **kwargs):
        calls.append(kwargs)
        on_stream = kwargs.get("on_stream")
        if on_stream is not None:
            on_stream("text", "attachment answer")
        return _route_result(trace_id, route_label, route_cfg, "attachment answer")

    monkeypatch.setattr(tui_app, "execute_route", fake_execute)

    async def flow() -> None:
        app = _make_app(tmp_path, dry_run=True)
        async with app.run_test(size=(120, 40)) as pilot:
            await _submit(pilot, app, "/think high")
            assert app._think_override == "high"
            mode = str(app.query_one("#composer-mode").content)
            assert "think:high" in mode

            await _submit(pilot, app, "/think extreme")
            assert app._think_override == "high"
            await _submit(pilot, app, f"/attach {tmp_path / 'missing.txt'}")
            assert app._attached_file is None

            await _submit(pilot, app, f"/attach {attachment}")
            assert app._attached_file == attachment
            mode = str(app.query_one("#composer-mode").content)
            assert "+notes.txt" in mode

            await _submit(pilot, app, "@local inspect the attachment")
            assert await _wait_for(lambda: len(calls) == 1 and not app._busy)
            assert calls[0]["think"] == "high"
            assert calls[0]["attach_file"] == attachment
            assert app._attached_file is None
            mode = str(app.query_one("#composer-mode").content)
            assert "think:high" in mode
            assert "+notes.txt" not in mode

            await _submit(pilot, app, "@local no attachment now")
            assert await _wait_for(lambda: len(calls) == 2 and not app._busy)
            assert calls[1]["attach_file"] is None

            await _submit(pilot, app, "/think default")
            assert app._think_override is None
            assert "think:" not in str(app.query_one("#composer-mode").content)

    asyncio.run(flow())


def test_tui_copy_uses_active_session_after_switch_and_restart(tmp_path: Path) -> None:
    async def flow() -> None:
        app = _make_app(tmp_path, dry_run=True)
        expected: dict[str, str] = {}
        async with app.run_test(size=(120, 40)) as pilot:
            await _submit(pilot, app, "@code first tab answer")
            assert await _wait_for(lambda: not app._busy)
            first_id = app.active_id
            expected["first"] = next(
                item.content
                for item in reversed(app.active_session.items)
                if item.kind == "assistant"
            )

            await pilot.press("ctrl+n")
            await pilot.pause()
            second_id = app.active_id
            assert second_id != first_id
            await _submit(pilot, app, "@code second tab answer")
            assert await _wait_for(lambda: not app._busy)
            expected["second"] = next(
                item.content
                for item in reversed(app.active_session.items)
                if item.kind == "assistant"
            )

            copied: list[str] = []
            app.copy_text = copied.append  # type: ignore[method-assign]
            app.switch_session(first_id)
            await pilot.pause()
            app.action_copy_last()
            assert copied[-1] == expected["first"]
            assert "first tab answer" in copied[-1]

            app.switch_session(second_id)
            await pilot.pause()
            app.action_copy_last()
            assert copied[-1] == expected["second"]
            assert "second tab answer" in copied[-1]

        restarted = _make_app(tmp_path, dry_run=True)
        assert restarted.active_id in (first_id, second_id)
        expected_active = expected["first" if restarted.active_id == first_id else "second"]
        restarted_copied: list[str] = []
        restarted.copy_text = restarted_copied.append  # type: ignore[method-assign]
        restarted.action_copy_last()
        assert restarted_copied == [expected_active]

    asyncio.run(flow())


def test_tui_verbosity_controls_thinking_visibility_and_retention(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[dict[str, object]] = []

    def fake_execute(trace_id, route_label, route_cfg, prompt, **kwargs):
        calls.append(kwargs)
        on_stream = kwargs.get("on_stream")
        output = f"answer {len(calls)}"
        if on_stream is not None:
            for _ in range(10):
                on_stream("thinking", "x" * 250)
            on_stream("text", output)
        return _route_result(trace_id, route_label, route_cfg, output)

    monkeypatch.setattr(tui_app, "execute_route", fake_execute)

    async def flow() -> None:
        app = _make_app(tmp_path, dry_run=True, verbosity=1)
        async with app.run_test(size=(120, 40)) as pilot:
            await _submit(pilot, app, "@code first verbose turn")
            assert await _wait_for(lambda: len(calls) == 1 and not app._busy)
            await pilot.pause()

            first = list(app.query(ToolActivityGroup))[-1]
            assert first._thinking_chars == 2500
            assert str(first._body.content).count("x") > 800
            assert first.collapsed is False

            await _submit(pilot, app, "/quiet ")
            assert app.verbosity == 0
            assert first.collapsed is True

            await _submit(pilot, app, "@code second quiet turn")
            assert await _wait_for(lambda: len(calls) == 2 and not app._busy)
            await pilot.pause()

            groups = list(app.query(ToolActivityGroup))
            assert len(groups) == 2
            second = groups[-1]
            assert second._thinking_chars == 2500
            assert str(second._body.content).count("x") > 800
            assert second.collapsed is True

            await _submit(pilot, app, "/verbose 1")
            assert app.verbosity == 1
            assert first.collapsed is False
            assert second.collapsed is False

    asyncio.run(flow())


def _load_cli_module():
    bin_path = PROJECT_ROOT / "bin" / "nexus"
    loader = importlib.machinery.SourceFileLoader("nexus_cli_advanced_mod", str(bin_path))
    spec = importlib.util.spec_from_loader("nexus_cli_advanced_mod", loader)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_tui_flags_and_non_tty_fallback(capsys, monkeypatch) -> None:
    cli = _load_cli_module()

    # `nexus tui --help` exits 0 and documents the TUI flags.
    with pytest.raises(SystemExit) as tui_help:
        cli.main(["tui", "--help"])
    assert tui_help.value.code == 0
    tui_out = capsys.readouterr().out
    assert "usage: nexus tui" in tui_out
    assert "--theme" in tui_out
    assert "--no-sidebar" in tui_out

    # Top-level help mentions both the REPL and the TUI.
    with pytest.raises(SystemExit) as top_help:
        cli.main(["--help"])
    assert top_help.value.code == 0
    top_out = capsys.readouterr().out
    assert "--repl" in top_out
    assert "tui" in top_out

    # Non-TTY stdin/stdout must fall back to the line-oriented REPL.
    calls: list[dict] = []

    def fake_repl(**kwargs) -> int:
        calls.append(kwargs)
        return 0

    class _NotATty:
        def isatty(self) -> bool:
            return False

        def write(self, _text: str) -> int:  # pragma: no cover - safety only
            return 0

        def flush(self) -> None:  # pragma: no cover - safety only
            return None

    monkeypatch.setattr(cli, "run_repl", fake_repl)
    monkeypatch.setattr(cli.sys, "stdin", _NotATty())
    monkeypatch.setattr(cli.sys, "stdout", _NotATty())

    assert cli.main([]) == 0
    assert len(calls) == 1


def test_tui_save_extracts_largest_code_block_to_drop_dir(
    tmp_path: Path, monkeypatch
) -> None:
    drop_root = tmp_path / "drops"
    monkeypatch.setenv("NEXUS_DROP_DIR", str(drop_root))

    sample_reply = (
        "Generated tool:\n\n"
        "```python\ndef greet():\n    return 'hello from ollama'\n\nprint(greet())\n```\n\n"
        "Usage:\n```sh\npython3 demo.py\n```\n"
    )

    def fake_execute(trace_id, route_label, route_cfg, prompt, **kwargs):
        on_stream = kwargs.get("on_stream")
        if on_stream is not None:
            on_stream("text", sample_reply)
        return _route_result(trace_id, route_label, route_cfg, sample_reply)

    monkeypatch.setattr(tui_app, "execute_route", fake_execute)

    async def flow() -> None:
        app = _make_app(tmp_path, dry_run=True)
        async with app.run_test(size=(120, 40)) as pilot:
            await _submit(pilot, app, "@ollama generate demo.py")
            assert await _wait_for(lambda: not app._busy and bool(app.active_session.items))
            await _submit(pilot, app, "/save demo.py")
            await pilot.pause()

        saved = drop_root / "demo.py"
        assert saved.is_file()
        assert saved.read_text(encoding="utf-8") == (
            "def greet():\n    return 'hello from ollama'\n\nprint(greet())\n"
        )

    asyncio.run(flow())


def test_tui_entity_profile_pin_mode_stats_and_ensemble(
    tmp_path: Path, monkeypatch
) -> None:
    """Exercise /profile, /pin, /unpin, /mode, /stats, and unified ensemble turns."""
    captured_prompts: list[str] = []

    def fake_execute(trace_id, route_label, route_cfg, prompt, **kwargs):
        captured_prompts.append(prompt)
        on_stream = kwargs.get("on_stream")
        reply = f"unified reply from {route_label}"
        if on_stream is not None:
            on_stream("text", reply)
        return _route_result(trace_id, route_label, route_cfg, reply)

    monkeypatch.setattr(tui_app, "execute_route", fake_execute)

    async def flow() -> None:
        app = _make_app(tmp_path, dry_run=True)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _submit(pilot, app, "@code initial context turn")
            assert await _wait_for(lambda: not app._busy and bool(app.active_session.items))
            initial_count = len(app.active_session.items)

            # Hot-swap to free-only preserves transcript & tags subsequent spans.
            await _submit(pilot, app, "/profile free-only")
            assert app.active_session.profile == "free-only"
            assert len(app.active_session.items) == initial_count

            # /pin and /unpin toggle sticky override without clearing entity mode.
            await _submit(pilot, app, "/pin code")
            assert app.active_session.pinned_route == "code"
            assert app.active_session.active_route == "code"
            await _submit(pilot, app, "/unpin")
            assert app.active_session.pinned_route is None
            assert app.active_session.active_route is None

            # /mode sticky -> /mode entity toggles cleanly.
            await _submit(pilot, app, "/mode sticky")
            assert app.active_session.entity_mode is False
            await _submit(pilot, app, "/mode entity")
            assert app.active_session.entity_mode is True

            # /stats mounts a performance summary note without crashing.
            await _submit(pilot, app, "/stats route")
            assert any("PERFORMANCE STATS" in str(w.content) for w in app.query(".system-note"))

            # Switching to ensemble runs parallel seats -> ONE synthesized assistant card.
            await _submit(pilot, app, "/profile ensemble")
            assert app.active_session.profile == "ensemble"
            await _submit(pilot, app, "Compare event sourcing vs CRUD")
            assert await _wait_for(
                lambda: not app._busy
                and sum(1 for item in app.active_session.items if item.kind == "assistant") >= 2
            )
            assert len(app.query(CouncilGridCard)) == 0
            assert len(captured_prompts) >= 4
            assert "[Internal Drafts]" in captured_prompts[-1]

    asyncio.run(flow())
