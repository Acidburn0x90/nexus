"""Focused headless tests for the Nexus TUI core.

Every app is built with `dry_run=True` (keyword routing, no backend binary) and
with all state paths redirected into `tmp_path`, so a run never touches
`~/.local/state/nexus` or `~/.config/nexus`.

The async Textual driver is entered with `asyncio.run(...)` inside a plain sync
test, so no `pytest-asyncio` plugin is required.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time
from typing import Callable

from textual.widgets import Input, OptionList, Static, TabbedContent

from nexus.events import read_events
from nexus.sessions import load_sessions
from nexus.tui.app import NexusApp
from nexus.tui.modals import RoutePickerModal
from nexus.tui.themes import resolve_theme_name
from nexus.tui.widgets import (
    AssistantMessageCard,
    ComposerBox,
    HintFooter,
    HomeHero,
    InspectorSidebar,
    RoutingPill,
    SessionTabBar,
    SuggestionPopup,
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


def test_session_loading_migrates_legacy_premium_span_count(tmp_path: Path) -> None:
    """A legacy session document loads its paid-span count into the new field."""
    path = tmp_path / "tui_sessions.json"
    path.write_text(
        json.dumps([
            {
                "id": "tui_legacy",
                "title": "Legacy session",
                "created_at": 1.0,
                "updated_at": 1.0,
                "argon_spans": 2,
                "items": [],
            }
        ]),
        encoding="utf-8",
    )

    loaded = load_sessions(path)

    assert len(loaded) == 1
    assert loaded[0].premium_spans == 2


def test_legacy_theme_alias_resolves_to_github_slate() -> None:
    assert resolve_theme_name("agy-slate") == "github-slate"


def test_tui_home_screen_mounts_cleanly(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = make_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            assert await wait_until(pilot, lambda: bool(app.query(HomeHero)), 2.0)
            await pilot.pause()
            assert await wait_until(
                pilot,
                lambda: app.query_one(HomeHero).region.width > 0,
                2.0,
            )

            hero = app.query_one(HomeHero)
            tab_bar = app.query_one(SessionTabBar)
            composer = app.query_one(ComposerBox)
            sidebar = app.query_one(InspectorSidebar)
            footer = app.query_one(HintFooter)

            for widget in (hero, tab_bar, composer, sidebar, footer):
                assert widget.is_mounted
                assert widget.visible
                assert widget.region.width > 0
                assert widget.region.height > 0

            # The inspector is open at 120 columns and no pane is hidden.
            assert not sidebar.has_class("hidden")
            assert sidebar.size.width > 0
            assert composer.input.is_mounted
            assert static_text(footer).strip()
            assert static_text(app.query_one("#composer-route", Static)).strip()

            await pilot.pause()
            assert no_exception(app)

    asyncio.run(scenario())


def test_tui_dry_run_prompt_submission_and_persistence(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = make_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            assert await wait_until(pilot, lambda: bool(app.query(HomeHero)), 2.0)
            composer = app.query_one(ComposerBox)
            composer.focus_input()

            prompt = "@code write hello world"
            for character in prompt:
                await pilot.press(character)
            await pilot.pause()
            assert composer.input.text == prompt

            await pilot.press("enter")

            session = app.active_session
            assert session is not None

            def finished() -> bool:
                return not app._busy and any(
                    item.kind == "assistant" for item in session.items
                )

            assert await wait_until(pilot, finished, 8.0)
            await pilot.pause()

            # The empty state is gone and the turn rendered three widgets.
            assert not app.query(HomeHero)
            user_cards = app.query(UserMessageCard)
            assert len(user_cards) == 1
            user_card = user_cards.first()
            assert user_card.text == prompt
            assert "hello world" in static_text(user_card.query_one(".user-body", Static))

            pills = app.query(RoutingPill)
            assert len(pills) == 1
            pill_text = static_text(pills.first())
            assert "front-desk" in pill_text
            assert "@code" in pill_text

            cards = app.query(AssistantMessageCard)
            assert len(cards) == 1
            card = cards.first()
            assert card.raw_text.strip()
            assert "[dry-run]" in card.raw_text
            assert "in-charge:code" in card.raw_text
            assert "write hello world" in card.raw_text
            assert card.status == "DONE"
            receipt = static_text(card.query_one(".assistant-receipt-text", Static))
            assert "DONE" in receipt
            assert "@code (in charge)" in receipt

            # Entity mode keeps active_route unpinned while recording last_route
            # and the dry-run token counters.
            assert session.active_route is None
            assert session.last_route == "code"
            assert session.total_tokens_in > 0
            assert session.total_tokens_out > 0
            route_status = static_text(app.query_one("#tab-status-route", Static))
            assert "via @code" in route_status
            token_status = static_text(app.query_one("#tab-status-tokens", Static))
            assert " tok" in token_status
            assert "0→0 tok" not in token_status
            assert "via @code" in static_text(app.query_one("#composer-route", Static))

            # The trace reached the append-only event bus.
            assistant_item = next(
                item for item in session.items if item.kind == "assistant"
            )
            events = read_events(
                path=app.events_path, trace_id=assistant_item.trace_id, limit=100
            )
            assert len(events) >= 2
            statuses = {event.get("status") for event in events}
            assert "RUNNING" in statuses
            assert "DONE" in statuses
            root_events = [event for event in events if event.get("route") == "entrypoint"]
            assert {event.get("status") for event in root_events} == {"RUNNING", "DONE"}
            root_ids = {event.get("span_id") for event in root_events}
            assert len(root_ids) == 1
            root_id = root_ids.pop()
            child_events = [
                event for event in events if event.get("route") == "in-charge:code"
            ]
            assert child_events
            assert all(event.get("parent_id") == root_id for event in child_events)

            # The turn was folded into the session document on disk.
            saved = load_sessions(tmp_path / "tui_sessions.json")
            assert len(saved) == 1
            persisted = saved[0]
            kinds = [item.kind for item in persisted.items]
            assert "user" in kinds
            assert "assistant" in kinds
            assert persisted.active_route is None
            assert persisted.last_route == "code"
            assert persisted.title != "New session"
            assert persisted.total_tokens_out > 0

            # Prompt history is written to the isolated history file.
            history = (tmp_path / "history").read_text(encoding="utf-8")
            assert prompt in history
            assert no_exception(app)

    asyncio.run(scenario())


def test_tui_inline_slash_and_at_autocomplete(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = make_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            composer = app.query_one(ComposerBox)
            popup = app.query_one(SuggestionPopup)
            composer.focus_input()

            # Slash alone opens the command palette.
            await pilot.press("/")
            assert await wait_until(pilot, lambda: popup.is_visible, 2.0)
            first = popup.selected_item()
            assert first is not None
            assert first.value.startswith("/")

            # Down moves the highlight, Tab accepts the completion.
            await pilot.press("down")
            await pilot.pause()
            chosen = popup.selected_item()
            assert chosen is not None
            assert chosen.value == "/pin"
            await pilot.press("tab")
            assert await wait_until(pilot, lambda: not popup.is_visible, 2.0)
            assert composer.input.text.strip() == chosen.value

            # Clear, then filter the @route palette.
            composer.input.load_text("")
            await pilot.pause()
            await pilot.press("@", "r", "e")
            assert await wait_until(pilot, lambda: popup.is_visible, 2.0)
            values = {item.value for item in popup._items}
            assert {"@reason", "@refactor"} <= values
            everything = {f"@{key}" for key in app.routes_data["routes"]}
            assert values < everything
            assert len(values) < len(everything)
            options = popup.query_one("#suggestion-options", OptionList)
            assert options.option_count == len(popup._items)

            # Tab accepts the highlighted route and closes the popup.
            item = popup.selected_item()
            assert item is not None
            await pilot.press("tab")
            assert await wait_until(pilot, lambda: not popup.is_visible, 2.0)
            assert composer.input.text == item.text_to_insert
            assert composer.input.text.startswith("@")
            assert no_exception(app)

    asyncio.run(scenario())


def test_tui_sidebar_toggle_and_tabs_and_square_post(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = make_app(tmp_path)
        square_path = tmp_path / "square.jsonl"
        async with app.run_test(size=(120, 40)) as pilot:
            sidebar = app.query_one(InspectorSidebar)
            assert not sidebar.has_class("hidden")

            await pilot.press("ctrl+b")
            await pilot.pause()
            assert sidebar.has_class("hidden")

            await pilot.press("ctrl+b")
            await pilot.pause()
            assert not sidebar.has_class("hidden")

            tabbed = sidebar.query_one(TabbedContent)
            assert tabbed.active == "tab-tree"

            # Activate the Routes tab by clicking its header.
            await pilot.click(content_tab(sidebar, "tab-routes"))
            await pilot.pause()
            assert tabbed.active == "tab-routes"
            routes = sidebar.query_one("#routes-list", OptionList)
            assert routes.option_count > 0

            # Activate the Square tab, then post through the quick-post field.
            await pilot.click(content_tab(sidebar, "tab-square"))
            await pilot.pause()
            assert tabbed.active == "tab-square"

            post_input = sidebar.query_one("#square-quick-post", Input)
            post_input.focus()
            await pilot.press(*"hello town square")
            await pilot.pause()
            assert post_input.value == "hello town square"
            await pilot.press("enter")

            assert await wait_until(pilot, lambda: square_path.is_file(), 3.0)
            lines = [
                json.loads(line)
                for line in square_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            assert len(lines) == 1
            assert lines[0]["body"] == "hello town square"
            assert lines[0]["author"] == "tui"

            # The feed refreshed with the new post.
            assert await wait_until(
                pilot,
                lambda: "hello town square"
                in static_text(sidebar.query_one("#square-posts", Static)),
                2.0,
            )
            assert post_input.value == ""
            assert no_exception(app)

    asyncio.run(scenario())


def test_tui_route_picker_and_release(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = make_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            composer = app.query_one(ComposerBox)

            # Ctrl+K opens the route picker.
            await pilot.press("ctrl+k")
            assert await wait_until(
                pilot, lambda: isinstance(app.screen, RoutePickerModal), 2.0
            )
            modal = app.screen
            assert isinstance(modal, RoutePickerModal)
            options = modal.query_one("#route-options", OptionList)
            option_ids = [
                options.get_option_at_index(index).id
                for index in range(options.option_count)
            ]
            assert "code" in option_ids

            options.highlighted = option_ids.index("code")
            await pilot.pause()
            await pilot.press("enter")

            assert await wait_until(
                pilot, lambda: not isinstance(app.screen, RoutePickerModal), 2.0
            )
            session = app.active_session
            assert session is not None
            assert session.active_route == "code"
            assert "@code" in static_text(app.query_one("#composer-route", Static))
            assert "@code" in static_text(app.query_one("#tab-status-route", Static))
            saved = load_sessions(tmp_path / "tui_sessions.json")
            assert saved and saved[0].active_route == "code"

            # /release hands control back to unified entity routing.
            app.prefill_composer("/release")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("enter")

            assert await wait_until(
                pilot, lambda: app.active_session.active_route is None, 2.0
            )
            await pilot.pause()
            released = app.active_session
            assert released is not None
            assert released.active_route is None
            assert released.pinned_route is None
            assert released.active_model == ""
            assert "entity" in static_text(
                app.query_one("#composer-route", Static)
            )
            assert "entity" in static_text(
                app.query_one("#tab-status-route", Static)
            )
            saved = load_sessions(tmp_path / "tui_sessions.json")
            assert saved[0].active_route is None

            # /mode sticky exposes the front-desk indicator when unpinned.
            app.prefill_composer("/mode sticky")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert "front-desk" in static_text(
                app.query_one("#composer-route", Static)
            )
            assert no_exception(app)

    asyncio.run(scenario())


def test_tool_activity_group_concise_titles_and_empty_body() -> None:
    empty = ToolActivityGroup()
    empty.finalize("DONE")
    assert empty.title == "✓ no tool calls"
    assert str(empty._body.content) == "[dim]no tool calls or thinking trace[/]"

    thinking = ToolActivityGroup()
    thinking.append_thinking("Compare the options")
    thinking.finalize("DONE")
    assert thinking.title == "✓ thinking"

    one_tool = ToolActivityGroup()
    one_tool.add_tool("read app.py")
    one_tool.finalize("DONE")
    assert one_tool.title == "✓ 1 tool"

    two_tools = ToolActivityGroup()
    two_tools.add_tool("read app.py")
    two_tools.add_tool("run pytest")
    two_tools.finalize("DONE")
    assert two_tools.title == "✓ 2 tools"

    running = ToolActivityGroup()
    running.add_tool("read app.py")
    running.add_tool("run pytest")
    running.append_thinking("Checking the failures")
    running._refresh()
    assert "thinking · 2 tools" in running.title


def test_webui_default_mode_ctrl_g_and_slash_ui(tmp_path: Path, monkeypatch) -> None:
    async def scenario() -> None:
        app = make_app(tmp_path, ui_mode=None)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.ui_mode == "webui"
            assert app.screen.has_class("ui-webui")
            assert app.theme == "nexus-charcoal"
            hero = app.query_one("#home-hero", HomeHero)
            assert hero._tip_timer is None
            banner_txt = str(hero.query_one(".hero-banner", Static).content)
            assert "Hello." in banner_txt
            assert "Ctrl+G" in str(app.query_one(HintFooter).content)

            # Ctrl+G: webui -> cli, then cli -> webui swaps nexus-charcoal -> gemini-dark.
            await pilot.press("ctrl+g")
            await pilot.pause()
            assert app.ui_mode == "cli"
            assert not app.screen.has_class("ui-webui")
            assert "N E X U S" in str(hero.query_one(".hero-banner", Static).content)

            await pilot.press("ctrl+g")
            await pilot.pause()
            assert app.ui_mode == "webui"
            assert app.theme == "gemini-dark"

            await pilot.press("ctrl+g")
            await pilot.pause()
            assert app.ui_mode == "cli"
            assert app.theme == "nexus-charcoal"

            app._handle_slash("/ui webui")
            await pilot.pause()
            assert app.ui_mode == "webui"
            saved = json.loads((tmp_path / "tui.json").read_text(encoding="utf-8"))
            assert saved["ui_mode"] == "webui"
            assert saved["theme"] == "gemini-dark"

            assert resolve_theme_name("gemini-dark") == "gemini-dark"

    asyncio.run(scenario())

    import importlib.machinery
    import importlib.util
    import sys

    bin_path = Path(__file__).resolve().parents[1] / "bin" / "nexus"
    loader = importlib.machinery.SourceFileLoader("nexus_cli_ui_test", str(bin_path))
    spec = importlib.util.spec_from_loader("nexus_cli_ui_test", loader)
    assert spec is not None
    cli = importlib.util.module_from_spec(spec)
    sys.modules["nexus_cli_ui_test"] = cli
    loader.exec_module(cli)

    seen_ui: list[object] = []
    import nexus.tui as tui_pkg

    monkeypatch.setattr(tui_pkg, "run_tui", lambda **kw: seen_ui.append(kw.get("ui_mode")) or 0)
    assert cli.main(["--dry-run", "tui", "--ui", "cli"]) == 0
    assert cli.main(["--dry-run", "--ui", "webui", "tui"]) == 0
    assert seen_ui == ["cli", "webui"]
