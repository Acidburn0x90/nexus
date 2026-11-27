"""Headless regression tests for the TUI and event-bus performance fixes.

Each test isolates all persistent state under `tmp_path`, drives the app with
`app.run_test(...)` + `Pilot`, and enters the async driver through
`asyncio.run(...)` inside a plain sync test, so no `pytest-asyncio` plugin is
required.

Covered fixes:

* `read_events(limit=...)` tail-seeks instead of parsing the whole JSONL bus.
* `ComposerTextArea._autosize` skips redundant `styles.height` writes.
* The composer elapsed-time ticker only runs while a run is in flight.
* Submission feedback (user card + running timer) lands before the front-desk
  decision returns.
* `AssistantMessageCard` appends markdown deltas instead of re-rendering.
* `InspectorSidebar` refreshes its tabs lazily and never reads while hidden.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable

from textual import events
from textual.color import Color
from textual.widgets import Button, Markdown, OptionList, Static, TabbedContent

try:  # Textual 8 moved `Edges` out of `textual.css.styles`.
    from textual.css.styles import Edges  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - depends on the installed Textual
    from textual.css._style_properties import Edges

import nexus.tui.app as tui_app
import nexus.tui.widgets as tui_widgets
from nexus.events import emit_event, read_events
from nexus.square import post_square
from nexus.tui.app import NexusApp
from nexus.tui.widgets import (
    AssistantMessageCard,
    ComposerBox,
    HomeHero,
    InspectorSidebar,
    RoutingPill,
    SessionTabBar,
    ThinkingDots,
    ToolActivityGroup,
    UserMessageCard,
)

ROUTES_PATH = Path(__file__).resolve().parents[1] / "config" / "routes.json"


def make_app(tmp_path: Path, **overrides: object) -> NexusApp:
    """Build a fully isolated, dry-run Nexus app rooted at `tmp_path`."""
    kwargs: dict[str, object] = {
        "events_path": tmp_path / "events.jsonl",
        "sessions_path": tmp_path / "tui_sessions.json",
        "config_path": tmp_path / "tui.json",
        "square_path": tmp_path / "square.jsonl",
        "board_path": tmp_path / "BOARD.md",
        "history_path": tmp_path / "history",
        "routes_path": ROUTES_PATH,
        "dry_run": True,
        "ui_mode": "cli",
    }
    kwargs.update(overrides)
    return NexusApp(**kwargs)  # type: ignore[arg-type]


async def wait_until(
    pilot: object,
    predicate: Callable[[], bool],
    timeout: float = 5.0,
    interval: float = 0.05,
) -> bool:
    """Poll `predicate` while letting the Textual event loop advance."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await pilot.pause(interval)  # type: ignore[attr-defined]
    return predicate()


def static_text(widget: Static) -> str:
    """Return the markup a `Static` currently renders."""
    return str(widget.content)


def no_exception(app: NexusApp) -> bool:
    event = getattr(app, "_exception_event", None)
    return event is None or not event.is_set()


def content_tab(sidebar: InspectorSidebar, pane_id: str) -> object:
    """Return the clickable header tab for a `TabPane` id."""
    for tab in sidebar.query("ContentTab"):
        if tab.id and tab.id.endswith(pane_id):
            return tab
    raise AssertionError(f"no header tab for {pane_id}")


# -- 1. event bus tail seek ---------------------------------------------------


def _event_line(index: int) -> dict[str, Any]:
    """Build one valid span record with multi-byte UTF-8 text."""
    return {
        "ts": 1_700_000_000.0 + index * 0.25,
        "mono_ns": index,
        "trace_id": f"tr_perf{index % 7:02d}",
        "span_id": f"sp_{index:06d}",
        "parent_id": None,
        "route": "code",
        "intent": "perf",
        "backend": "opencode",
        "model": "deepseek-v4.1-flash",
        "quota_tier": "FREE",
        "status": "DONE" if index % 2 else "RUNNING",
        "step": "stream",
        "prompt": f"café résumé 🚀 日本語 line {index}",
        "tokens_in": index,
        "tokens_out": index // 2,
        "elapsed_ms": index % 1000,
        "error": "",
    }


def test_read_events_tail_seek_equivalence_and_speed(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    lines: list[str] = []
    for index in range(3000):
        if index % 25 == 0:
            lines.append("{bad json")
        elif index % 19 == 0:
            lines.append("")
        else:
            lines.append(json.dumps(_event_line(index), ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    full = read_events(path)
    assert len(full) > 500
    assert full[-1]["trace_id"] == f"tr_perf{(2999 % 7):02d}"

    # Tail-seek must reproduce the exact same newest windows as a full scan.
    assert read_events(path, limit=50) == full[-50:]
    assert read_events(path, limit=400) == full[-400:]

    started = time.perf_counter()
    tail = read_events(path, limit=400)
    elapsed = time.perf_counter() - started
    assert tail == full[-400:]
    assert elapsed < 0.15, f"tail seek took {elapsed:.3f}s for 400 events"

    assert read_events(path, limit=1) == full[-1:]
    assert read_events(tmp_path / "missing.jsonl", limit=10) == []
    assert read_events(tmp_path / "missing.jsonl") == []


# -- 2. composer autosize -----------------------------------------------------


def test_composer_autosize_skips_redundant_height_writes(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = make_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            assert await wait_until(pilot, lambda: bool(app.query(HomeHero)), 2.0)
            composer = app.query_one(ComposerBox)
            composer.focus_input()
            area = composer.input
            await pilot.pause()

            assert area._last_line_count == 1
            assert area.styles.height.value == 1

            # Count the actual `height` style-rule writes. `_autosize` assigns
            # `self.styles.height`, and the scalar descriptor goes through
            # `styles.set_rule("height", ...)`.
            height_writes: list[Any] = []
            styles = area.styles
            original_set_rule = styles.set_rule

            def spy_set_rule(name: str, value: Any, *args: Any, **kwargs: Any) -> bool:
                if name == "height":
                    height_writes.append(value)
                return original_set_rule(name, value, *args, **kwargs)

            styles.set_rule = spy_set_rule  # type: ignore[method-assign]
            try:
                prompt = "hello typing fast ok"
                assert len(prompt) == 20
                for character in prompt:
                    await pilot.press(character)
                await pilot.pause()

                assert area.text == prompt
                assert area._last_line_count == 1
                assert area.styles.height.value == 1
                assert height_writes == [], "redundant height writes while typing"

                # Two newlines -> three lines -> exactly two height writes.
                await pilot.press("shift+enter")
                await pilot.pause()
                await pilot.press("shift+enter")
                await pilot.pause()
                assert area.document.line_count == 3
                assert area._last_line_count == 3
                assert area.styles.height.value == 3
                assert len(height_writes) == 2

                # Each backspace removes one newline: 3 -> 2 -> 1 lines, so two
                # more height writes (2, then 1) and none while on one line.
                await pilot.press("backspace")
                await pilot.press("backspace")
                await pilot.pause()
                assert area.document.line_count == 1
                assert area._last_line_count == 1
                assert area.styles.height.value == 1
                assert [write.value for write in height_writes] == [2.0, 3.0, 2.0, 1.0]
            finally:
                styles.set_rule = original_set_rule  # type: ignore[method-assign]

            assert no_exception(app)

    asyncio.run(scenario())


# -- 3. composer ticker -------------------------------------------------------


def test_composer_tick_timer_only_runs_while_busy(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = make_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            assert await wait_until(pilot, lambda: bool(app.query(HomeHero)), 2.0)
            composer = app.query_one(ComposerBox)
            await pilot.pause()

            assert composer._tick_timer is None
            assert composer._busy_running is False

            composer.set_running(True, label="test")
            assert composer._busy_running is True
            assert composer._tick_timer is not None

            composer.set_running(False)
            assert composer._busy_running is False
            assert composer._tick_timer is None

            # A real dry-run turn starts and stops the ticker exactly once.
            composer.focus_input()
            composer.input.load_text("@code hello")
            await pilot.pause()
            await pilot.press("enter")

            assert await wait_until(
                pilot,
                lambda: not app._busy
                and any(
                    item.kind == "assistant"
                    for item in app.active_session.items
                ),
                8.0,
            )
            await pilot.pause()

            assert composer._busy_running is False
            assert composer._tick_timer is None
            assert no_exception(app)

    asyncio.run(scenario())


# -- 4. instant submission feedback ------------------------------------------


def test_instant_submit_feedback_before_front_desk_decide_returns(
    tmp_path: Path, monkeypatch: Any
) -> None:
    decide_started = threading.Event()
    release_decide = threading.Event()
    real_decide = tui_app.front_desk_decide
    calls: list[str] = []

    def fake_decide(*args: Any, **kwargs: Any) -> Any:
        calls.append(str(args[0]) if args else "")
        decide_started.set()
        release_decide.wait(timeout=10.0)
        kwargs = dict(kwargs)
        kwargs["use_llm"] = False
        return real_decide(*args, **kwargs)

    monkeypatch.setattr(tui_app, "front_desk_decide", fake_decide)

    async def scenario() -> None:
        app = make_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            assert await wait_until(pilot, lambda: bool(app.query(HomeHero)), 2.0)
            composer = app.query_one(ComposerBox)
            composer.focus_input()
            composer.input.load_text("explain why my build failed")
            await pilot.pause()
            await pilot.press("enter")

            assert await wait_until(pilot, decide_started.is_set, 5.0)
            # Let the queued mount/remove messages land; routing is still
            # blocked inside `front_desk_decide`, which has not returned yet.
            await pilot.pause()

            assert calls == ["explain why my build failed"]
            assert not app.query(HomeHero)
            assert len(app.query(UserMessageCard)) == 1
            assert app.active_id in app._busy
            assert composer._busy_running is True
            assert len(app.query(RoutingPill)) == 0

            release_decide.set()
            assert await wait_until(pilot, lambda: not app._busy, 8.0)
            await pilot.pause()

            # The instant card is reused, never duplicated.
            assert len(app.query(UserMessageCard)) == 1
            assert len(app.query(RoutingPill)) == 1
            assert len(app.query(AssistantMessageCard)) == 1
            assert app.query_one(ComposerBox)._tick_timer is None
            assert no_exception(app)

    try:
        asyncio.run(scenario())
    finally:
        release_decide.set()


# -- 5. assistant markdown ----------------------------------------------------


def test_assistant_card_incremental_markdown_and_set_text(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = make_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            assert await wait_until(pilot, lambda: bool(app.query(HomeHero)), 2.0)
            card = AssistantMessageCard()
            app.query_one("#transcript").mount(card)
            await pilot.pause()

            body = card.query_one(Markdown)
            updates: list[str] = []
            appends: list[str] = []
            real_update = body.update
            real_append = body.append

            def spy_update(text: str, *args: Any, **kwargs: Any) -> Any:
                updates.append(text)
                return real_update(text, *args, **kwargs)

            def spy_append(text: str, *args: Any, **kwargs: Any) -> Any:
                appends.append(text)
                return real_append(text, *args, **kwargs)

            body.update = spy_update  # type: ignore[method-assign]
            body.append = spy_append  # type: ignore[method-assign]
            try:
                first = "# Title\n\nFirst paragraph."
                card.append_text(first)
                card.flush_markdown()
                assert updates == [first]
                assert appends == []
                assert card._rendered_text == first

                delta = "\n\nSecond paragraph."
                card.append_text(delta)
                card.flush_markdown()
                assert appends == [delta]
                assert len(updates) == 1, "incremental append must not re-update"
                assert card._rendered_text == first + delta

                card.set_text("Replaced answer")
                assert updates == [first, "Replaced answer"]
                assert card.raw_text == "Replaced answer"

                await pilot.pause()
                assert no_exception(app)
            finally:
                body.update = real_update  # type: ignore[method-assign]
                body.append = real_append  # type: ignore[method-assign]

    asyncio.run(scenario())


# -- 6. lazy sidebar tabs -----------------------------------------------------


def test_inspector_sidebar_lazy_tabs_and_hidden_skip(
    tmp_path: Path, monkeypatch: Any
) -> None:
    events_path = tmp_path / "events.jsonl"
    square_path = tmp_path / "square.jsonl"
    emit_event(
        "tr_sidebar",
        "sp_sidebar",
        route="code",
        model="deepseek-v4.1-flash",
        status="DONE",
        prompt="café 🚀 日本語",
        tokens_in=10,
        tokens_out=4,
        elapsed_ms=21,
        path=events_path,
    )
    post_square(
        "hello lazy inspector",
        author="tester",
        tag="perf",
        square_path=square_path,
    )

    counts = {"events": 0, "routes": 0, "square": 0}
    real_read_events = tui_widgets.read_events
    real_load_routes = tui_widgets.load_routes
    real_read_square = tui_widgets.read_square

    def spy_read_events(*args: Any, **kwargs: Any) -> Any:
        counts["events"] += 1
        return real_read_events(*args, **kwargs)

    def spy_load_routes(*args: Any, **kwargs: Any) -> Any:
        counts["routes"] += 1
        return real_load_routes(*args, **kwargs)

    def spy_read_square(*args: Any, **kwargs: Any) -> Any:
        counts["square"] += 1
        return real_read_square(*args, **kwargs)

    app = make_app(tmp_path, show_sidebar=False)
    monkeypatch.setattr(tui_widgets, "read_events", spy_read_events)
    monkeypatch.setattr(tui_widgets, "load_routes", spy_load_routes)
    monkeypatch.setattr(tui_widgets, "read_square", spy_read_square)

    async def scenario() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            assert await wait_until(pilot, lambda: bool(app.query(HomeHero)), 2.0)
            sidebar = app.query_one(InspectorSidebar)
            assert sidebar.has_class("hidden")
            await pilot.pause()

            # Mounting a hidden inspector must not touch any data source.
            assert counts == {"events": 0, "routes": 0, "square": 0}
            tabbed = sidebar.query_one(TabbedContent)
            assert tabbed.active == "tab-tree"

            # Opening the sidebar refreshes only the active tab.
            await pilot.press("ctrl+b")
            await pilot.pause()
            assert not sidebar.has_class("hidden")
            assert await wait_until(pilot, lambda: counts["events"] >= 1, 2.0)
            assert counts["routes"] == 0
            assert counts["square"] == 0
            tree_text = static_text(sidebar.query_one("#tree-view", Static))
            assert "code" in tree_text
            assert "spans" in tree_text

            # Routes stay lazy until their header is clicked.
            await pilot.click(content_tab(sidebar, "tab-routes"))
            await pilot.pause()
            assert await wait_until(pilot, lambda: counts["routes"] >= 1, 2.0)
            routes = sidebar.query_one("#routes-list", OptionList)
            assert routes.option_count > 0
            assert counts["square"] == 0

            # Square stays lazy until its header is clicked.
            await pilot.click(content_tab(sidebar, "tab-square"))
            await pilot.pause()
            assert await wait_until(pilot, lambda: counts["square"] >= 1, 2.0)
            assert await wait_until(
                pilot,
                lambda: "hello lazy inspector"
                in static_text(sidebar.query_one("#square-posts", Static)),
                2.0,
            )
            assert no_exception(app)

    asyncio.run(scenario())


# -- 7. tab-bar hover ---------------------------------------------------------


def borderless(widget: Any) -> bool:
    """True when every border edge is cleared, i.e. Textual's empty `Edges`."""
    border = widget.styles.border
    assert isinstance(border, Edges)
    return not border


def test_tabbar_hover_preserves_active_and_avoids_black_bg_or_restyle_churn(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        app = make_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            assert await wait_until(pilot, lambda: bool(app.query(HomeHero)), 2.0)
            await pilot.pause()

            active_tab = app.query_one(".tab-pill.active", Button)
            new_btn = app.query_one("#btn-new-tab", Button)
            close_btn = app.query_one("#btn-close-tab", Button)

            for btn in (active_tab, new_btn, close_btn):
                assert btn.active_effect_duration == 0
                assert btn.styles.pointer == "default"
                assert borderless(btn), f"{btn.id} unexpectedly has a border"

            normal_active_bg = active_tab.styles.background

            await pilot.hover(".tab-pill.active")
            await pilot.pause()
            hovered_bg = active_tab.styles.background
            assert hovered_bg != Color(0, 0, 0)
            assert hovered_bg != Color(0x25, 0x25, 0x29)
            assert hovered_bg.r >= normal_active_bg.r
            assert hovered_bg.b > 100
            assert borderless(active_tab)

            for selector in ("#btn-new-tab", "#btn-close-tab"):
                await pilot.hover(selector)
                await pilot.pause()
                btn = app.query_one(selector, Button)
                assert btn.styles.background != Color(0, 0, 0)
                assert borderless(btn)

            # Re-entering the same pill at a new offset must not run Textual's
            # hover restyle pass again.
            await pilot.hover(".tab-pill.active", offset=(1, 0))
            await pilot.pause()

            restyles: list[Any] = []
            real_update_styles = app.update_styles

            def spy_update_styles(node: Any, animate: bool = True) -> None:
                restyles.append(node)
                real_update_styles(node, animate=animate)

            app.update_styles = spy_update_styles  # type: ignore[method-assign]
            try:
                for x in (3, 5, 7):
                    await pilot.hover(".tab-pill.active", offset=(x, 0))
                await pilot.pause()
                assert restyles == [], f"{len(restyles)} redundant restyle(s)"
            finally:
                app.update_styles = real_update_styles  # type: ignore[method-assign]

            assert no_exception(app)

    asyncio.run(scenario())


# -- 8. empty-tab hero reuse --------------------------------------------------


def test_empty_tab_new_and_close_keeps_home_hero_without_flash(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = make_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            assert await wait_until(pilot, lambda: bool(app.query(HomeHero)), 2.0)
            hero = app.query_one(HomeHero)

            await pilot.click("#btn-new-tab")
            await pilot.pause()
            assert len(app.sessions) == 2
            assert app.query_one(HomeHero) is hero

            await pilot.click("#btn-close-tab")
            await pilot.pause()
            assert len(app.sessions) == 1
            assert app.query_one(HomeHero) is hero
            assert no_exception(app)

    asyncio.run(scenario())


# -- 9. routing spinner lifecycle --------------------------------------------


def test_thinking_dots_animate_during_routing_and_stop_when_done(
    tmp_path: Path, monkeypatch: Any
) -> None:
    decide_started = threading.Event()
    release_decide = threading.Event()
    real_decide = tui_app.front_desk_decide

    def fake_decide(*args: Any, **kwargs: Any) -> Any:
        decide_started.set()
        release_decide.wait(timeout=10.0)
        kwargs = dict(kwargs)
        kwargs["use_llm"] = False
        return real_decide(*args, **kwargs)

    monkeypatch.setattr(tui_app, "front_desk_decide", fake_decide)

    async def scenario() -> None:
        app = make_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            assert await wait_until(pilot, lambda: bool(app.query(HomeHero)), 2.0)
            composer = app.query_one(ComposerBox)
            tab_bar = app.query_one(SessionTabBar)

            composer.focus_input()
            composer.input.load_text("analyze tradeoffs quickly")
            await pilot.pause()
            await pilot.press("enter")

            assert await wait_until(pilot, decide_started.is_set, 5.0)
            await pilot.pause()

            indicators = list(app.query(ThinkingDots))
            assert any(
                dots.display is True and dots._active is True for dots in indicators
            )
            assert tab_bar._spin_timer is not None
            assert composer._tick_timer is not None

            frames_before = [dots._frame for dots in indicators]
            spin_before = tab_bar._spin_frame
            await pilot.pause(0.35)
            indicators = list(app.query(ThinkingDots))
            assert (
                any(dots._frame > 0 for dots in indicators)
                or tab_bar._spin_frame > spin_before
            ), f"no animation (thinking frames {frames_before})"

            release_decide.set()
            assert await wait_until(pilot, lambda: not app._busy, 8.0)
            await pilot.pause()

            for dots in app.query(ThinkingDots):
                assert dots.display is False
                assert dots._timer is None
            assert composer._tick_timer is None
            assert tab_bar._spin_timer is None
            assert no_exception(app)

    try:
        asyncio.run(scenario())
    finally:
        release_decide.set()


# -- 10. assistant thinking dots ---------------------------------------------


def test_assistant_card_thinking_dots_hide_on_first_chunk(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = make_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            assert await wait_until(pilot, lambda: bool(app.query(HomeHero)), 2.0)
            card = AssistantMessageCard()
            app.query_one("#transcript").mount(card)
            await pilot.pause()

            dots = card.query_one(".assistant-thinking", ThinkingDots)
            assert dots.display is True
            assert dots._timer is not None
            assert dots._active is True

            card.append_text("Hello world")
            await pilot.pause()

            assert dots.display is False
            assert dots._timer is None
            assert dots._active is False
            assert no_exception(app)

    asyncio.run(scenario())


# -- 11. tool-group spinner ---------------------------------------------------


def test_tool_activity_group_spinner_skips_body_updates_during_thinking(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        app = make_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            assert await wait_until(pilot, lambda: bool(app.query(HomeHero)), 2.0)
            group = ToolActivityGroup()
            app.query_one("#transcript").mount(group)
            await pilot.pause()

            group.append_thinking("Compare the options")
            updates: list[str] = []
            real_update = group._body.update

            def spy_update(text: str, *args: Any, **kwargs: Any) -> Any:
                updates.append(text)
                return real_update(text, *args, **kwargs)

            group._body.update = spy_update  # type: ignore[method-assign]
            try:
                titles = []
                for _ in range(5):
                    group._advance_spinner()
                    titles.append(group.title)
                assert updates == []
                assert len(set(titles)) == 5
                assert all("thinking" in title for title in titles)
            finally:
                group._body.update = real_update  # type: ignore[method-assign]
            assert no_exception(app)

    asyncio.run(scenario())
