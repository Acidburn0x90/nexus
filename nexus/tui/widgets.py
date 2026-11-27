"""Textual widgets for the Nexus TUI.

The widgets here are deliberately self-contained: each one owns its markup,
keeps its own state, and only talks to the app through small `Message` classes.
That keeps Step 4's app layer thin and keeps every pane testable in isolation.

Two conventions worth knowing before editing:

* Every widget that can be fed streaming text stores the raw markdown and
  pushes it into `Markdown` through a short debounce. Unmounting mid-stream is
  normal (a session switch), so all queries are guarded with `NoMatches`.
* Rich markup is built from user or model text, so every interpolated string
  goes through `rich.markup.escape`; otherwise a stray `[` would swallow the
  rest of the line.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Optional, TypeVar

from rich.markup import escape
from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import (
    Button,
    Collapsible,
    Input,
    Markdown,
    OptionList,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)
from textual.widgets.option_list import Option

from nexus.backends import ExecutionResult
from nexus.events import DEFAULT_EVENTS_PATH, read_events, reconstruct_tree
from nexus.router import list_opencode_models, load_routes
from nexus.sessions import TuiSession
from nexus.square import read_board, read_square
from nexus.tui.themes import normalize_tier, quota_markup, tier_color
from nexus.visualizer import (
    format_duration,
    format_status_icon,
    format_tokens,
    render_delegation_tree,
    truncate_ansi,
)

# Narrower than this and the skeletal delegation tree wastes more columns on
# borders than it spends on data, so the sidebar draws a compact indented tree.
_TREE_MIN_WIDTH = 78

MessageT = TypeVar("MessageT", bound=type[Message])


def _handler(name: str) -> Callable[[MessageT], MessageT]:
    """Pin a public handler name on a nested message class.

    Textual derives `handler_name` from the qualified class name, so
    `SessionTabBar.TabSelected` would otherwise only answer to
    `on_session_tab_bar_tab_selected`. Nexus keeps the public names short:
    `on_tab_selected`, `on_composer_submitted`, and so on.
    """

    def decorator(message_class: MessageT) -> MessageT:
        message_class.handler_name = name
        return message_class

    return decorator


_STATUS_ICONS = {
    "SPAWN": "○",
    "RUNNING": "⠋",
    "DONE": "✓",
    "FAIL": "✗",
    "TIMEOUT": "⏱",
}


_SPINNER_FRAMES: tuple[str, ...] = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
_DOT_FRAMES: tuple[str, ...] = (
    "[#38bdf8]●[/][dim] ○ ○[/]",
    "[dim]○ [/][#38bdf8]●[/][dim] ○[/]",
    "[dim]○ ○ [/][#38bdf8]●[/]",
    "[dim]○ [/][#38bdf8]●[/][dim] ○[/]",
)


def _thinking_markup(frame: int, label: str = "thinking") -> str:
    """Render one frame of the braille spinner + pulsing dots strip."""
    spin = _SPINNER_FRAMES[frame % len(_SPINNER_FRAMES)]
    dots = _DOT_FRAMES[frame % len(_DOT_FRAMES)]
    if label:
        return f"[#38bdf8]{spin}[/] [dim]{escape(label)}[/]  {dots}"
    return f"[#38bdf8]{spin}[/]  {dots}"


def _status_icon(status: str) -> str:
    return _STATUS_ICONS.get((status or "").upper(), "•")


def _short_model(model: str, limit: int = 26) -> str:
    """Drop the provider prefix and clip the model id to `limit` characters."""
    name = model.split("/", 1)[1] if "/" in model else model
    if len(name) <= limit:
        return name
    return name[: max(1, limit - 1)] + "…"


def _short_title(title: str, limit: int = 18) -> str:
    clean = " ".join((title or "untitled").split())
    if len(clean) <= limit:
        return clean
    return clean[: max(1, limit - 1)] + "…"


class SessionTabBar(Horizontal):
    """Top bar: clickable session pills on the left, compact status on the right."""

    @_handler("on_tab_selected")
    class TabSelected(Message):
        def __init__(self, session_id: str) -> None:
            super().__init__()
            self.session_id = session_id

    @_handler("on_new_tab_requested")
    class NewTabRequested(Message):
        pass

    @_handler("on_tab_close_requested")
    class TabCloseRequested(Message):
        def __init__(self, session_id: str) -> None:
            super().__init__()
            self.session_id = session_id

    DEFAULT_CSS = """
    SessionTabBar {
        height: 2;
        align: left middle;
    }
    SessionTabBar #tab-pills {
        width: 1fr;
        height: 1;
        layout: horizontal;
        overflow: hidden;
    }
    SessionTabBar Button,
    SessionTabBar Button.-style-default {
        height: 1;
        min-width: 3;
        border: none;
        border-top: none;
        border-bottom: none;
        padding: 0 1;
        margin: 0;
        background: $panel;
        color: $text-muted;
        content-align: center middle;
        pointer: default;
    }
    SessionTabBar Button.-style-default:hover,
    SessionTabBar Button:hover {
        border: none;
        border-top: none;
        border-bottom: none;
        background: $panel-lighten-1;
        color: $text;
    }
    SessionTabBar Button.-style-default.-active,
    SessionTabBar Button.-active {
        border: none;
        border-top: none;
        border-bottom: none;
        tint: transparent;
    }
    SessionTabBar Button.tab-pill {
        min-width: 9;
        max-width: 24;
        margin: 0 1 0 0;
    }
    SessionTabBar Button.tab-pill:hover {
        background: $panel-lighten-1;
        color: $text;
    }
    SessionTabBar Button.tab-pill.active {
        background: $primary;
        color: $background;
        text-style: bold;
    }
    SessionTabBar Button.tab-pill.active:hover {
        background: $primary-lighten-1;
        color: $background;
        text-style: bold;
    }
    SessionTabBar Button#btn-new-tab {
        width: 3;
        min-width: 3;
        padding: 0;
        color: $primary;
        margin: 0 1 0 0;
    }
    SessionTabBar Button#btn-new-tab:hover {
        background: $panel-lighten-1;
        color: $primary;
    }
    SessionTabBar Button#btn-close-tab {
        width: 3;
        min-width: 3;
        padding: 0;
        color: $text-muted;
    }
    SessionTabBar Button#btn-close-tab:hover {
        background: $error;
        color: $background;
    }
    SessionTabBar #tab-spacer {
        display: none;
        width: 0;
        height: 1;
    }
    SessionTabBar Static.status-pill {
        width: auto;
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(self, *children: Any, **kwargs: Any) -> None:
        kwargs.setdefault("id", "tab-bar")
        super().__init__(*children, **kwargs)
        self._sessions: list[TuiSession] = []
        self._active_id: str = ""
        self._busy_ids: set[str] = set()
        self._active_route: Optional[str] = None
        self._active_model: str = ""
        self._quota_tier: str = "FREE"
        self._sidebar_open: bool = True
        self._tokens_in: int = 0
        self._tokens_out: int = 0
        self._premium_spans: int = 0
        self._entity_mode: bool = True
        self._profile_name: str = "default"
        self._pinned_route: Optional[str] = None
        self._last_route: Optional[str] = None
        self._tab_buttons: dict[str, Button] = {}
        self._tab_labels: dict[str, str] = {}
        self._last_status: dict[str, tuple[bool, str]] = {}
        self._new_tab_button: Optional[Button] = None
        self._close_tab_button: Optional[Button] = None
        self._ready: bool = False
        self._spin_frame: int = 0
        self._spin_timer: Any = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="tab-pills"):
            pass
        yield Static("", id="tab-spacer")
        yield Static("", id="tab-status-route", classes="status-pill")
        yield Static("", id="tab-status-tokens", classes="status-pill")
        yield Static("", id="tab-status-quota", classes="status-pill")
        yield Static("", id="tab-status-sidebar", classes="status-pill")

    def on_mount(self) -> None:
        if self._ready:
            self._sync_tabs()
            self._sync_status()
        self._sync_spin_timer()

    def on_unmount(self) -> None:
        if self._spin_timer is not None:
            self._spin_timer.stop()
            self._spin_timer = None

    def update_state(
        self,
        sessions: list[TuiSession],
        active_id: str,
        busy_ids: set[str],
        active_route: Optional[str],
        active_model: str,
        quota_tier: str,
        sidebar_open: bool,
        total_tokens_in: int,
        total_tokens_out: int,
        premium_spans: int,
        *,
        entity_mode: bool = True,
        profile_name: str = "default",
        pinned_route: Optional[str] = None,
        last_route: Optional[str] = None,
    ) -> None:
        """Replace the bar's view of the world; safe to call on every event."""
        self._sessions = list(sessions)
        self._active_id = active_id
        self._busy_ids = set(busy_ids)
        self._active_route = active_route
        self._active_model = active_model
        self._quota_tier = quota_tier or "FREE"
        self._sidebar_open = sidebar_open
        self._tokens_in = total_tokens_in
        self._tokens_out = total_tokens_out
        self._premium_spans = premium_spans
        self._entity_mode = bool(entity_mode)
        self._profile_name = profile_name or "default"
        self._pinned_route = pinned_route
        self._last_route = last_route
        self._ready = True
        if self.is_mounted:
            self._sync_tabs()
            self._sync_status()
            self._sync_spin_timer()

    # -- rendering ---------------------------------------------------------

    def _tab_label(self, session: TuiSession) -> str:
        if session.id in self._busy_ids:
            marker = _SPINNER_FRAMES[self._spin_frame % len(_SPINNER_FRAMES)]
        elif session.id == self._active_id:
            marker = "●"
        else:
            marker = "○"
        return f"{marker} {escape(_short_title(session.title))}"

    def _sync_spin_timer(self) -> None:
        """Animate busy tab spinners only while a session is actually busy."""
        if self._busy_ids and self.is_mounted:
            if self._spin_timer is None:
                self._spin_timer = self.set_interval(0.18, self._advance_spinner)
        elif self._spin_timer is not None:
            self._spin_timer.stop()
            self._spin_timer = None

    def _advance_spinner(self) -> None:
        if not self._busy_ids or not self.is_mounted:
            if self._spin_timer is not None:
                self._spin_timer.stop()
                self._spin_timer = None
            return
        self._spin_frame += 1
        for session in self._sessions:
            if session.id not in self._busy_ids:
                continue
            button = self._tab_buttons.get(session.id)
            if button is None:
                continue
            label = self._tab_label(session)
            button.label = label
            self._tab_labels[session.id] = label

    def _sync_tabs(self) -> None:
        """Reuse buttons per session id so a re-render never fights the DOM."""
        try:
            pills = self.query_one("#tab-pills", Horizontal)
        except NoMatches:
            return

        live = {session.id for session in self._sessions}
        for session_id, button in list(self._tab_buttons.items()):
            if session_id not in live:
                self._tab_buttons.pop(session_id, None)
                self._tab_labels.pop(session_id, None)
                button.remove()

        for session in self._sessions:
            button = self._tab_buttons.get(session.id)
            label = self._tab_label(session)
            if button is None:
                button = Button(label, id=f"tab-{session.id}", classes="tab-pill", compact=True)
                button.active_effect_duration = 0
                self._tab_buttons[session.id] = button
                self._tab_labels[session.id] = label
                # Keep new tabs left of the `+` / `×` controls, even when they
                # arrive after the first render.
                if self._new_tab_button is not None:
                    pills.mount(button, before=self._new_tab_button)
                else:
                    pills.mount(button)
            elif self._tab_labels.get(session.id) != label:
                button.label = label
                self._tab_labels[session.id] = label
            is_active = session.id == self._active_id
            if button.has_class("active") != is_active:
                button.set_class(is_active, "active")

        self._sync_controls()

    def _sync_controls(self) -> None:
        """Mount the static `+` / `×` buttons exactly once, then leave them alone."""
        try:
            pills = self.query_one("#tab-pills", Horizontal)
        except NoMatches:
            return
        if self._new_tab_button is None:
            self._new_tab_button = Button("+", id="btn-new-tab", compact=True)
            self._new_tab_button.active_effect_duration = 0
            pills.mount(self._new_tab_button)
        if self._close_tab_button is None:
            self._close_tab_button = Button("×", id="btn-close-tab", compact=True)
            self._close_tab_button.active_effect_duration = 0
            pills.mount(self._close_tab_button)

    def _sync_status(self) -> None:
        narrow = bool(self.size.width and self.size.width < 92)
        pinned = self._pinned_route or (self._active_route if not self._entity_mode else None)
        if pinned:
            model_short = _short_model(self._active_model or "cli", limit=14 if narrow else 24)
            route_text = (
                f"[{tier_color(self._quota_tier)}]@{escape(pinned)}[/] "
                f"[dim]{escape(model_short)}[/]"
            )
        elif self._entity_mode:
            route_text = f"[#22c55e]entity[/] [dim]· {escape(self._profile_name)}[/]"
            if self._last_route and not narrow:
                route_text += f" [dim]via @{escape(self._last_route)}[/]"
        else:
            route_text = "[dim]front-desk[/]"

        tokens_text = (
            ""
            if narrow
            else f"[dim]{format_tokens(self._tokens_in)}→{format_tokens(self._tokens_out)} tok[/]"
        )
        if self._premium_spans > 0:
            quota_text = f"[bold white on #ef4444] PREMIUM {self._premium_spans} [/]"
        else:
            quota_text = "[#22c55e]$0 premium[/]"
        sidebar_text = (
            ""
            if narrow
            else ("[dim]▤ sidebar[/]" if self._sidebar_open else "[dim]▢ sidebar[/]")
        )

        for selector, text in (
            ("#tab-status-route", route_text),
            ("#tab-status-tokens", tokens_text),
            ("#tab-status-quota", quota_text),
            ("#tab-status-sidebar", sidebar_text),
        ):
            try:
                widget = self.query_one(selector, Static)
                state = (bool(text), text)
                if self._last_status.get(selector) == state:
                    continue
                widget.display = bool(text)
                widget.update(text)
                self._last_status[selector] = state
            except NoMatches:
                continue

    def on_resize(self) -> None:
        if self._ready:
            self._sync_status()

    # -- events ------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "btn-new-tab":
            event.stop()
            self.post_message(self.NewTabRequested())
            return
        if button_id == "btn-close-tab":
            event.stop()
            if self._active_id:
                self.post_message(self.TabCloseRequested(self._active_id))
            return
        if button_id.startswith("tab-"):
            event.stop()
            session_id = button_id[len("tab-") :]
            if session_id in {session.id for session in self._sessions}:
                self.post_message(self.TabSelected(session_id))


class HomeHero(Vertical):
    """Empty-state panel: banner, active front-desk model and quick actions."""

    @_handler("on_quick_action_selected")
    class QuickActionSelected(Message):
        def __init__(self, command: str) -> None:
            super().__init__()
            self.command = command

    QUICK_ACTIONS = (
        ("@code", "hero-code", "@code"),
        ("@reason", "hero-reason", "@reason"),
        ("/council", "hero-council", "/council"),
        ("/routes", "hero-routes", "/routes"),
    )

    SHORTCUT_TIPS = (
        "Tab accepts the highlighted @route or /command suggestion",
        "Shift+Enter or Ctrl+J adds a newline without sending",
        "Up/Down at the edge of the composer walks prompt history",
        "@local and @ollama keep work offline at $0",
        "/profile switches model configurations mid-chat without losing context",
        "/council runs Pioneer, Red-Teamer and Synthesizer in parallel",
        "Ctrl+B toggles the inspector sidebar",
    )

    DEFAULT_CSS = """
    HomeHero {
        height: auto;
        align: center middle;
    }
    HomeHero .hero-banner {
        width: 1fr;
        height: auto;
        text-align: center;
        text-style: bold;
        color: $primary;
    }
    HomeHero .hero-model {
        width: 1fr;
        height: auto;
        text-align: center;
        color: $text-muted;
        margin-bottom: 1;
    }
    HomeHero #hero-actions {
        height: auto;
        align: center middle;
    }
    HomeHero Button.hero-action {
        height: 1;
        min-width: 10;
        margin: 0 1;
        border: none;
        border-top: none;
        border-bottom: none;
        background: $panel;
        color: $primary;
        content-align: center middle;
        pointer: default;
    }
    HomeHero Button.hero-action:hover {
        border: none;
        border-top: none;
        border-bottom: none;
        background: $primary;
        color: $background;
    }
    HomeHero Button.hero-action.-active {
        border: none;
        border-top: none;
        border-bottom: none;
        tint: transparent;
    }
    HomeHero .hero-tip {
        width: 1fr;
        height: auto;
        text-align: center;
        color: $text-muted;
        margin-top: 1;
    }
    """

    def __init__(self, *children: Any, **kwargs: Any) -> None:
        kwargs.setdefault("id", "home-hero")
        super().__init__(*children, **kwargs)
        self._front_desk_model = "opencode/nemotron-3.5-lightning-free"
        self._quota_tier = "FREE"
        self._tip_index = 0
        self._tip_timer: Any = None

    def compose(self) -> ComposeResult:
        yield Static(
            "[bold]N E X U S[/]\n[dim]multi-model delegation console[/]",
            classes="hero-banner",
        )
        yield Static(self._front_desk_text(), classes="hero-model")
        with Horizontal(id="hero-actions"):
            for label, widget_id, _command in self.QUICK_ACTIONS:
                btn = Button(label, id=widget_id, classes="hero-action")
                btn.active_effect_duration = 0
                yield btn
        yield Static(self._tip_text(), classes="hero-tip")

    def on_mount(self) -> None:
        self._tip_timer = self.set_interval(7.0, self._rotate_tip)

    def on_unmount(self) -> None:
        if self._tip_timer is not None:
            self._tip_timer.stop()
            self._tip_timer = None

    def _front_desk_text(self) -> str:
        return (
            "[dim]front desk ·[/] "
            f"[bold]{escape(self._front_desk_model)}[/] "
            f"{quota_markup(self._quota_tier)}"
        )

    def _tip_text(self) -> str:
        tip = self.SHORTCUT_TIPS[self._tip_index % len(self.SHORTCUT_TIPS)]
        return f"[dim]tip ·[/] {escape(tip)}"

    def _rotate_tip(self) -> None:
        self._tip_index = (self._tip_index + 1) % len(self.SHORTCUT_TIPS)
        try:
            self.query_one(".hero-tip", Static).update(self._tip_text())
        except NoMatches:
            return

    def set_front_desk(self, model: str, quota_tier: str = "FREE") -> None:
        self._front_desk_model = model or self._front_desk_model
        self._quota_tier = quota_tier or "FREE"
        if not self.is_mounted:
            return
        try:
            self.query_one(".hero-model", Static).update(self._front_desk_text())
        except NoMatches:
            return

    def on_button_pressed(self, event: Button.Pressed) -> None:
        for _label, widget_id, command in self.QUICK_ACTIONS:
            if event.button.id == widget_id:
                event.stop()
                self.post_message(self.QuickActionSelected(command))
                return


class UserMessageCard(Vertical):
    """A submitted prompt: timestamp, optional route badge, left-accent body."""

    DEFAULT_CSS = """
    UserMessageCard .user-meta {
        height: auto;
        color: $text-muted;
    }
    UserMessageCard .user-body {
        height: auto;
        color: $text;
    }
    """

    def __init__(
        self,
        text: str,
        *,
        route: str = "",
        quota_tier: str = "",
        timestamp: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("classes", "user-card")
        super().__init__(**kwargs)
        self.text = text
        self.route = route
        self.quota_tier = quota_tier
        self.timestamp = timestamp if timestamp is not None else time.time()

    def _meta_text(self) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        parts = [f"[dim]{stamp}[/]"]
        if self.route:
            parts.append(f"[bold]@{escape(self.route)}[/]")
        if self.quota_tier:
            parts.append(quota_markup(self.quota_tier))
        return "  ".join(parts)

    def compose(self) -> ComposeResult:
        yield Static(self._meta_text(), classes="user-meta")
        yield Static(escape(self.text), classes="user-body")

    def set_route(self, route: str, quota_tier: str = "") -> None:
        """Update the route/tier badge after front-desk classification resolves."""
        self.route = route or ""
        if quota_tier:
            self.quota_tier = quota_tier
        if not self.is_mounted:
            return
        try:
            self.query_one(".user-meta", Static).update(self._meta_text())
        except NoMatches:
            return


class RoutingPill(Static):
    """One-line summary of the front-desk routing decision."""

    DEFAULT_CSS = """
    RoutingPill {
        height: auto;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(self, text: str = "", **kwargs: Any) -> None:
        super().__init__(text, **kwargs)

    def set_decision(
        self,
        route: str,
        model: str,
        reason: str = "",
        quota_tier: str = "FREE",
    ) -> None:
        line = (
            "[dim]front-desk →[/] "
            f"[bold]@{escape(route)}[/] "
            f"[dim]{escape(model)}[/] "
            f"{quota_markup(quota_tier)}"
        )
        if reason:
            line += f" · [dim]{escape(reason)}[/]"
        self.update(line)


class ThinkingDots(Static):
    """Single-line animated braille spinner + bouncing dots while waiting."""

    DEFAULT_CSS = """
    ThinkingDots {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        label: str = "thinking",
        *,
        active: bool = True,
        **kwargs: Any,
    ) -> None:
        self._label: str = label
        self._frame: int = 0
        self._timer: Any = None
        self._active: bool = bool(active)
        super().__init__(_thinking_markup(0, label) if self._active else "", **kwargs)
        if not self._active:
            self.display = False

    def on_mount(self) -> None:
        if self._active and self._timer is None:
            self._timer = self.set_interval(0.14, self._advance)

    def on_unmount(self) -> None:
        self._stop_timer()

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _advance(self) -> None:
        if not self._active or not self.is_mounted:
            self._stop_timer()
            return
        self._frame = (self._frame + 1) % 20
        self.update(_thinking_markup(self._frame, self._label))

    def set_label(self, label: str) -> None:
        self._label = label
        if self._active and self.is_mounted:
            self.update(_thinking_markup(self._frame, self._label))

    def stop(self) -> None:
        self._active = False
        self._stop_timer()
        self.display = False


class ToolActivityGroup(Collapsible):
    """Collapsible thinking + numbered tool-call group for one assistant turn."""

    DEFAULT_CSS = """
    ToolActivityGroup {
        height: auto;
        border-top: none;
        padding-bottom: 0;
        margin: 0 0 0 1;
    }
    ToolActivityGroup .tool-body {
        height: auto;
        color: $text-muted;
        padding: 0 0 0 1;
    }
    """

    def __init__(
        self,
        *,
        collapsed: bool = True,
        verbosity: int = 1,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ) -> None:
        self._body = Static("", classes="tool-body")
        self._tools: list[str] = []
        self._thinking_parts: list[str] = []
        self._thinking_text: str = ""
        self._thinking_chars: int = 0
        self._thinking_dirty: bool = False
        self._tools_dirty: bool = False
        self._verbosity = max(0, min(2, int(verbosity)))
        self._status = "RUNNING"
        self._elapsed_ms = 0
        self._refresh_timer: Any = None
        self._last_body_markup: str = ""
        self._last_title: str = "⠋ tools (0)"
        self._spin_frame: int = 0
        self._spin_timer: Any = None
        super().__init__(
            self._body,
            title="⠋ tools (0)",
            collapsed=collapsed,
            id=id,
            classes=classes or "tool-group",
        )

    @property
    def has_activity(self) -> bool:
        return bool(self._thinking_chars or self._tools)

    def _cancel_timer(self) -> None:
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
            self._refresh_timer = None

    def on_unmount(self) -> None:
        self._cancel_timer()
        self._stop_spinner()

    def _stop_spinner(self) -> None:
        if self._spin_timer is not None:
            self._spin_timer.stop()
            self._spin_timer = None

    def _advance_spinner(self) -> None:
        if self._status != "RUNNING" or not self.is_mounted:
            self._stop_spinner()
            return
        self._spin_frame += 1
        self._refresh_title()
        if not self._thinking_chars and not self._tools:
            body_markup = _thinking_markup(self._spin_frame, "waiting")
            if body_markup != self._last_body_markup:
                self._body.update(body_markup)
                self._last_body_markup = body_markup

    def _refresh_title(self) -> None:
        icon = (
            _SPINNER_FRAMES[self._spin_frame % len(_SPINNER_FRAMES)]
            if self._status == "RUNNING"
            else _status_icon(self._status)
        )
        size_tag = (
            f" ({format_tokens(self._thinking_chars)}c)"
            if self._thinking_chars >= 1000
            else ""
        )
        tool_count_done = (
            f"{len(self._tools)} tool{'s' if len(self._tools) != 1 else ''}"
        )
        if self._tools and self._thinking_chars:
            title = f"{icon} thinking · {tool_count_done}"
            if self._thinking_chars >= 1000:
                title += f" · {format_tokens(self._thinking_chars)}c"
        elif self._thinking_chars:
            title = f"{icon} thinking{size_tag}"
        elif self._tools:
            title = (
                f"{icon} tools ({len(self._tools)})"
                if self._status == "RUNNING"
                else f"{icon} {tool_count_done}"
            )
        else:
            title = (
                f"{icon} tools (0)"
                if self._status == "RUNNING"
                else f"{icon} no tool calls"
            )
        if self._elapsed_ms:
            title += f" · {format_duration(self._elapsed_ms)}"
        if title != self._last_title:
            self.title = title
            self._last_title = title

    def _refresh_body(self, *, force: bool = False) -> None:
        if not (
            force
            or self._thinking_dirty
            or self._tools_dirty
            or not self._last_body_markup
        ):
            return
        if self._thinking_parts:
            self._thinking_text += "".join(self._thinking_parts)
            self._thinking_parts.clear()
        self._thinking_dirty = False
        self._tools_dirty = False

        lines: list[str] = []
        thinking = self._thinking_text.strip()
        if thinking:
            if self._status != "RUNNING":
                cap = 16_000
            elif self._verbosity >= 2:
                cap = 12_000
            else:
                cap = 8_000
            shown = thinking[-cap:] if len(thinking) > cap else thinking
            lines.append(f"[dim]thinking · {format_tokens(len(thinking))} chars[/]")
            if len(thinking) > cap:
                omitted = len(thinking) - cap
                lines.append(
                    f"[dim]… ({format_tokens(omitted)} earlier chars omitted)[/]"
                )
            lines.append(f"[dim italic]{escape(shown)}[/]")
        for index, tool in enumerate(self._tools, start=1):
            lines.append(f"[#38bdf8]{index}.[/] {escape(tool)}")
        if lines:
            body_markup = "\n".join(lines)
        elif self._status == "RUNNING":
            body_markup = _thinking_markup(self._spin_frame, "waiting")
        else:
            body_markup = "[dim]no tool calls or thinking trace[/]"
        if body_markup != self._last_body_markup:
            self._body.update(body_markup)
            self._last_body_markup = body_markup

    def _refresh(self) -> None:
        self._refresh_body(force=True)
        self._refresh_title()

    def add_tool(self, summary: str) -> None:
        self._tools.append(" ".join((summary or "").split()) or "tool")
        self._tools_dirty = True
        self._cancel_timer()
        self._refresh_body()
        self._refresh_title()

    def append_thinking(self, chunk: str) -> None:
        if not chunk:
            return
        self._thinking_parts.append(chunk)
        self._thinking_chars += len(chunk)
        self._thinking_dirty = True
        self._refresh_title()
        if self.is_mounted and self._refresh_timer is None:
            self._refresh_timer = self.set_timer(0.10, self._on_refresh_timer)

    def _on_refresh_timer(self) -> None:
        self._refresh_timer = None
        self._refresh_body()

    def finalize(self, status: str = "DONE", elapsed_ms: int = 0) -> None:
        self._status = (status or "DONE").upper()
        self._elapsed_ms = max(0, int(elapsed_ms))
        self._cancel_timer()
        self._stop_spinner()
        self._refresh()

    def set_verbosity(self, level: int) -> None:
        self._verbosity = max(0, min(2, int(level)))
        if self._thinking_chars > 8000:
            self._refresh_body(force=True)
        if self._verbosity >= 1 and self.has_activity:
            self.collapsed = False
        elif self._verbosity == 0:
            self.collapsed = True

    def set_expanded(self, expanded: bool) -> None:
        self.collapsed = not expanded

    def on_mount(self) -> None:
        self._refresh()
        if self._status == "RUNNING" and self._spin_timer is None:
            self._spin_timer = self.set_interval(0.18, self._advance_spinner)


class AssistantMessageCard(Vertical):
    """Streaming assistant answer with a receipt row and a copy button."""

    @_handler("on_copy_requested")
    class CopyRequested(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    DEFAULT_CSS = """
    AssistantMessageCard {
        height: auto;
        padding: 0 1 1 1;
    }
    AssistantMessageCard .assistant-body {
        height: auto;
        padding: 0;
        margin: 0;
    }
    AssistantMessageCard .assistant-thinking {
        height: 1;
        padding: 0;
        color: $text-muted;
    }
    AssistantMessageCard .assistant-receipt {
        height: auto;
        color: $text-muted;
        margin-top: 1;
    }
    AssistantMessageCard .assistant-receipt-text {
        width: 1fr;
        height: auto;
        color: $text-muted;
    }
    AssistantMessageCard Button.copy-btn {
        width: auto;
        min-width: 8;
        height: 1;
        border: none;
        border-top: none;
        border-bottom: none;
        padding: 0 1;
        background: $panel;
        color: $text-muted;
        content-align: center middle;
        pointer: default;
    }
    AssistantMessageCard Button.copy-btn:hover {
        border: none;
        border-top: none;
        border-bottom: none;
        background: $primary;
        color: $background;
    }
    AssistantMessageCard Button.copy-btn.-active {
        border: none;
        border-top: none;
        border-bottom: none;
        tint: transparent;
    }
    """

    def __init__(self, text: str = "", **kwargs: Any) -> None:
        kwargs.setdefault("classes", "assistant-card")
        super().__init__(**kwargs)
        self.raw_text: str = text
        self.status: str = "RUNNING"
        self.route: str = ""
        self.model: str = ""
        self.quota_tier: str = ""
        self.error: str = ""
        self._rendered_text: str = ""
        self._rendered_len: int = 0
        self._receipt_text: str = ""
        self._flush_timer: Any = None
        self._last_think_label: str = "thinking"

    def compose(self) -> ComposeResult:
        yield ThinkingDots(
            label="thinking",
            active=(self.status == "RUNNING" and not self.raw_text),
            classes="assistant-thinking",
        )
        yield Markdown("", classes="assistant-body")
        with Horizontal(classes="assistant-receipt"):
            yield Static("", classes="assistant-receipt-text")
            copy_btn = Button("copy", classes="copy-btn")
            copy_btn.active_effect_duration = 0
            yield copy_btn

    def on_mount(self) -> None:
        self.flush_markdown()
        self._apply_receipt()
        if self.raw_text or self.status != "RUNNING":
            self._stop_thinking()

    def on_unmount(self) -> None:
        if self._flush_timer is not None:
            self._flush_timer.stop()
            self._flush_timer = None
        self._stop_thinking()

    def _stop_thinking(self) -> None:
        try:
            self.query_one(".assistant-thinking", ThinkingDots).stop()
        except NoMatches:
            return

    def set_thinking_progress(self, chars: int) -> None:
        if self.status != "RUNNING" or self.raw_text or chars <= 0:
            return
        label = f"thinking · {format_tokens(chars)}c"
        if label == self._last_think_label:
            return
        try:
            dots = self.query_one(".assistant-thinking", ThinkingDots)
        except NoMatches:
            return
        dots.set_label(label)
        self._last_think_label = label

    # -- text --------------------------------------------------------------

    def append_text(self, chunk: str) -> None:
        if not chunk:
            return
        self.raw_text += chunk
        self._stop_thinking()
        self._schedule_flush()

    def set_text(self, full_text: str) -> None:
        self.raw_text = full_text
        self._stop_thinking()
        self.flush_markdown()

    def _schedule_flush(self) -> None:
        if self._flush_timer is not None or not self.is_mounted:
            return
        self._flush_timer = self.set_timer(0.10, self._on_flush_timer)

    def _on_flush_timer(self) -> None:
        self._flush_timer = None
        self.flush_markdown()

    def flush_markdown(self) -> None:
        """Push `raw_text` into the Markdown widget, ignoring an unmounted card."""
        if self._flush_timer is not None:
            self._flush_timer.stop()
            self._flush_timer = None
        if not self.is_mounted or self.raw_text == self._rendered_text:
            return
        try:
            body = self.query_one(Markdown)
        except NoMatches:
            return
        if self._rendered_len > 0 and self.raw_text.startswith(self._rendered_text):
            delta = self.raw_text[self._rendered_len :]
            if delta:
                body.append(delta)
        elif self.raw_text or self._rendered_text:
            body.update(self.raw_text or " ")
        self._rendered_text = self.raw_text
        self._rendered_len = len(self.raw_text)

    # -- receipt -----------------------------------------------------------

    def _build_receipt(
        self,
        status: str,
        route: str,
        model: str,
        quota_tier: str,
        elapsed_ms: int,
        tokens_in: int,
        tokens_out: int,
        tools_count: int,
        error: str,
        cached_tokens: int,
    ) -> str:
        parts = [f"[bold]{_status_icon(status)} {escape(status.upper())}[/]"]
        if route:
            if route.startswith("in-charge:"):
                route_label = f"@{route[len('in-charge:'):]} (in charge)"
            else:
                route_label = f"@{route}"
            parts.append(f"[bold]{escape(route_label)}[/]")
        if model:
            parts.append(f"[dim]{escape(model)}[/]")
        if quota_tier:
            parts.append(quota_markup(quota_tier))
        parts.append(format_duration(elapsed_ms))
        if tokens_in or tokens_out:
            parts.append(f"{format_tokens(tokens_in)}→{format_tokens(tokens_out)} tok")
        if cached_tokens:
            parts.append(f"{format_tokens(cached_tokens)} cached")
        if tools_count:
            parts.append(f"{tools_count} tools")
        receipt = " · ".join(parts)
        if error:
            receipt += f"\n[#ef4444]✗ {escape(' '.join(error.split()))}[/]"
        return receipt

    def finalize(
        self,
        status: str,
        route: str,
        model: str,
        quota_tier: str,
        elapsed_ms: int,
        tokens_in: int,
        tokens_out: int,
        tools_count: int,
        error: str = "",
        cached_tokens: int = 0,
    ) -> None:
        self.status = (status or "DONE").upper()
        self.route = route or ""
        self.model = model or ""
        self.quota_tier = quota_tier or ""
        self.error = error or ""
        self._receipt_text = self._build_receipt(
            self.status,
            self.route,
            self.model,
            self.quota_tier,
            elapsed_ms,
            tokens_in,
            tokens_out,
            tools_count,
            self.error,
            cached_tokens,
        )
        self.set_class(normalize_tier(self.quota_tier) == "PREMIUM", "tier-premium")
        self._stop_thinking()
        self.flush_markdown()
        self._apply_receipt()

    def _apply_receipt(self) -> None:
        if not self.is_mounted or not self._receipt_text:
            return
        try:
            self.query_one(".assistant-receipt-text", Static).update(self._receipt_text)
        except NoMatches:
            return

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if "copy-btn" in event.button.classes:
            event.stop()
            self.post_message(self.CopyRequested(self.raw_text))


class CouncilGridCard(Vertical):
    """Three-seat parallel council: Pioneer, Red-Teamer, Synthesizer."""

    SEATS: tuple[str, ...] = ("pioneer", "red_teamer", "synthesizer")
    SEAT_LABELS = {
        "pioneer": "PIONEER",
        "red_teamer": "RED-TEAMER",
        "synthesizer": "SYNTHESIZER",
    }

    DEFAULT_CSS = """
    CouncilGridCard {
        height: auto;
        padding: 0 1 1 1;
    }
    CouncilGridCard .council-grid {
        height: auto;
        layout: grid;
        grid-size: 3;
        grid-gutter: 1;
        margin: 1 0;
    }
    CouncilGridCard .council-grid.stacked {
        grid-size: 1;
    }
    CouncilGridCard .council-seat {
        height: 100%;
        background: $surface;
        border: solid $border;
        padding: 0 1;
    }
    CouncilGridCard .council-grid.stacked .council-seat {
        height: auto;
    }
    CouncilGridCard .council-seat-title {
        height: 1;
        text-style: bold;
        color: $primary;
    }
    CouncilGridCard .council-seat-body {
        height: auto;
        padding: 0;
        margin: 0;
    }
    CouncilGridCard .council-seat-thinking {
        height: 1;
        padding: 0;
        color: $text-muted;
    }
    CouncilGridCard .council-seat-receipt {
        height: auto;
        color: $text-muted;
        margin-top: 1;
    }
    """

    def __init__(self, *children: Any, **kwargs: Any) -> None:
        kwargs.setdefault("classes", "assistant-card")
        super().__init__(**kwargs)
        self._grid: Optional[Vertical] = None
        self._bodies: dict[str, Markdown] = {}
        self._receipts: dict[str, Static] = {}
        self._seat_text: dict[str, str] = {seat: "" for seat in self.SEATS}
        self._seat_receipt: dict[str, str] = {seat: "" for seat in self.SEATS}
        self._rendered_text: dict[str, str] = {seat: "" for seat in self.SEATS}
        self._rendered_receipt: dict[str, str] = {seat: "" for seat in self.SEATS}
        self._flush_timer: Any = None
        self._seat_thinking: dict[str, ThinkingDots] = {}
        self._seat_think_label: dict[str, str] = {
            seat: "thinking" for seat in self.SEATS
        }

    def compose(self) -> ComposeResult:
        with Vertical(classes="council-grid") as grid:
            self._grid = grid
            for seat in self.SEATS:
                with Vertical(classes="council-seat"):
                    yield Static(self.SEAT_LABELS[seat], classes="council-seat-title")
                    dots = ThinkingDots(
                        label="thinking",
                        active=(not self._seat_text[seat]),
                        classes="council-seat-thinking",
                    )
                    self._seat_thinking[seat] = dots
                    yield dots
                    body = Markdown("", classes="council-seat-body")
                    self._bodies[seat] = body
                    yield body
                    receipt = Static("", classes="council-seat-receipt")
                    self._receipts[seat] = receipt
                    yield receipt

    def _seat_key(self, seat: str) -> Optional[str]:
        key = (seat or "").strip().lower().replace("-", "_").replace(" ", "_")
        if key in self.SEATS:
            return key
        if key in ("redteam", "redteamer", "red_teamer"):
            return "red_teamer"
        if key in ("synth", "synthesis", "synthesizer"):
            return "synthesizer"
        if key in ("pioneer", "optimist"):
            return "pioneer"
        return None

    def _stop_seat_thinking(self, seat: str) -> None:
        if dots := self._seat_thinking.get(seat):
            dots.stop()

    def update_seat_thinking(self, seat: str, chars: int) -> None:
        key = self._seat_key(seat)
        if key is None or self._seat_text[key] or chars <= 0:
            return
        label = f"thinking · {format_tokens(chars)}c"
        if label == self._seat_think_label[key]:
            return
        dots = self._seat_thinking.get(key)
        if dots is None:
            return
        dots.set_label(label)
        self._seat_think_label[key] = label

    def update_seat(self, seat: str, chunk: str) -> None:
        key = self._seat_key(seat)
        if key is None or not chunk:
            return
        self._seat_text[key] += chunk
        self._stop_seat_thinking(key)
        self._schedule_flush()

    def set_seat_text(self, seat: str, text: str) -> None:
        key = self._seat_key(seat)
        if key is None:
            return
        self._seat_text[key] = text
        self._stop_seat_thinking(key)
        self._flush_seats()

    def _schedule_flush(self) -> None:
        if self._flush_timer is not None or not self.is_mounted:
            return
        self._flush_timer = self.set_timer(0.12, self._on_flush_timer)

    def _on_flush_timer(self) -> None:
        self._flush_timer = None
        self._flush_seats()

    def _flush_seats(self) -> None:
        if not self.is_mounted:
            return
        for seat in self.SEATS:
            text = self._seat_text[seat]
            prev = self._rendered_text[seat]
            if text == prev:
                continue
            body = self._bodies.get(seat)
            if body is None:
                continue
            if prev and text.startswith(prev):
                delta = text[len(prev) :]
                if delta:
                    body.append(delta)
            elif text or prev:
                body.update(text or " ")
            self._rendered_text[seat] = text
        self._apply_receipts()

    def _apply_receipts(self) -> None:
        if not self.is_mounted:
            return
        for seat in self.SEATS:
            text = self._seat_receipt[seat]
            if not text or text == self._rendered_receipt[seat]:
                continue
            receipt = self._receipts.get(seat)
            if receipt is not None:
                receipt.update(text)
                self._rendered_receipt[seat] = text

    def finalize_seat(self, seat: str, result: ExecutionResult) -> None:
        key = self._seat_key(seat)
        if key is None:
            return
        self._seat_text[key] = result.output or self._seat_text[key]
        status = (result.status or "DONE").upper()
        model_label = _short_model(result.model)
        if result.model.startswith(f"{result.backend}/"):
            model_disp = model_label
        else:
            model_disp = f"{result.backend}:{model_label}"
        parts = [
            f"[bold]{_status_icon(status)} {escape(status)}[/]",
            f"[dim]{escape(model_disp)}[/]",
            quota_markup(result.quota_tier),
            format_duration(result.elapsed_ms),
            f"{format_tokens(result.tokens_in)}→{format_tokens(result.tokens_out)} tok",
        ]
        if result.error:
            parts.append(f"[#ef4444]✗ {escape(' '.join(result.error.split()))}[/]")
        self._seat_receipt[key] = " · ".join(parts)
        self._stop_seat_thinking(key)
        self._flush_seats()

    def set_stacked(self, stacked: bool) -> None:
        if self._grid is not None:
            self._grid.set_class(stacked, "stacked")

    def on_mount(self) -> None:
        self._flush_seats()

    def on_unmount(self) -> None:
        if self._flush_timer is not None:
            self._flush_timer.stop()
            self._flush_timer = None
        for dots in self._seat_thinking.values():
            dots.stop()


@dataclass
class SuggestionItem:
    """One row in the composer suggestion popup."""

    value: str
    label: str
    description: str
    badge: str = ""
    insert_text: str = ""

    @property
    def text_to_insert(self) -> str:
        return self.insert_text or self.value


class SuggestionPopup(Vertical):
    """Slash-command / @route palette backed by an `OptionList`."""

    @_handler("on_suggestion_chosen")
    class SuggestionChosen(Message):
        def __init__(self, item: SuggestionItem) -> None:
            super().__init__()
            self.item = item

    DEFAULT_CSS = """
    SuggestionPopup {
        height: auto;
        max-height: 10;
        display: none;
    }
    SuggestionPopup.visible {
        display: block;
    }
    SuggestionPopup #suggestion-options {
        height: auto;
        max-height: 8;
        background: transparent;
        border: none;
        padding: 0;
    }
    """

    def __init__(self, *children: Any, **kwargs: Any) -> None:
        kwargs.setdefault("id", "suggestion-popup")
        super().__init__(*children, **kwargs)
        self._items: list[SuggestionItem] = []

    def compose(self) -> ComposeResult:
        yield OptionList(id="suggestion-options")

    def on_mount(self) -> None:
        self._render_items()

    def set_items(self, items: list[SuggestionItem]) -> None:
        incoming = list(items)
        if not incoming:
            if not self._items and not self.has_class("visible"):
                return
            self._items = []
            self.hide()
            if self.is_mounted:
                try:
                    self.query_one("#suggestion-options", OptionList).clear_options()
                except NoMatches:
                    pass
            return
        if incoming == self._items and self.has_class("visible"):
            return
        self._items = incoming
        if self.is_mounted:
            self._render_items()

    def _format_option(self, item: SuggestionItem) -> str:
        line = f"[bold]{escape(item.label or item.value)}[/]"
        if item.description:
            line += f"  [dim]{escape(item.description)}[/]"
        if item.badge:
            line += f"  {item.badge}"
        return line

    def _render_items(self) -> None:
        try:
            options = self.query_one("#suggestion-options", OptionList)
        except NoMatches:
            return
        options.clear_options()
        if not self._items:
            self.hide()
            return
        options.add_options(
            [
                Option(self._format_option(item), id=str(index))
                for index, item in enumerate(self._items)
            ]
        )
        options.highlighted = 0
        self.add_class("visible")

    def move(self, delta: int) -> None:
        if not self._items:
            return
        try:
            options = self.query_one("#suggestion-options", OptionList)
        except NoMatches:
            return
        count = len(self._items)
        current = options.highlighted
        if current is None:
            target = 0 if delta >= 0 else count - 1
        else:
            target = (current + delta) % count
        options.highlighted = target
        options.scroll_to_highlight()

    def selected_item(self) -> Optional[SuggestionItem]:
        if not self._items:
            return None
        try:
            options = self.query_one("#suggestion-options", OptionList)
        except NoMatches:
            return None
        index = options.highlighted
        if index is None or not 0 <= index < len(self._items):
            return None
        return self._items[index]

    def hide(self) -> None:
        if self.has_class("visible"):
            self.remove_class("visible")

    @property
    def is_visible(self) -> bool:
        return self.has_class("visible")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "suggestion-options":
            return
        event.stop()
        try:
            index = int(event.option.id or "")
        except ValueError:
            self.hide()
            return
        if 0 <= index < len(self._items):
            item = self._items[index]
            self.hide()
            self.post_message(self.SuggestionChosen(item))


class ComposerTextArea(TextArea):
    """Prompt input: popup-aware key handling, prompt history and auto-height."""

    @_handler("on_composer_submitted")
    class ComposerSubmitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    @_handler("on_popup_accept_requested")
    class PopupAcceptRequested(Message):
        pass

    @_handler("on_popup_navigate_requested")
    class PopupNavigateRequested(Message):
        def __init__(self, delta: int) -> None:
            super().__init__()
            self.delta = delta

    @_handler("on_popup_dismiss_requested")
    class PopupDismissRequested(Message):
        pass

    @_handler("on_cancel_requested")
    class CancelRequested(Message):
        pass

    popup_open: bool = False

    DEFAULT_CSS = """
    ComposerTextArea {
        height: auto;
        min-height: 1;
        max-height: 6;
        border: none;
        background: transparent;
        padding: 0;
    }
    """

    def __init__(
        self,
        text: str = "",
        *,
        history: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("compact", True)
        kwargs.setdefault("placeholder", "Ask Nexus — Enter to send · Shift+Enter for newline")
        super().__init__(text, **kwargs)
        self._history: list[str] = [entry for entry in (history or []) if entry.strip()]
        self._history_index: Optional[int] = None
        self._draft: str = ""
        self._last_line_count: int = 0

    # -- history -----------------------------------------------------------

    def set_history(self, entries: list[str]) -> None:
        self._history = [entry for entry in entries if entry.strip()]
        self._history_index = None
        self._draft = ""

    def _load_history(self, text: str) -> None:
        self.load_text(text)
        self.move_cursor(self.document.end)

    def _cycle_history(self, delta: int) -> bool:
        if not self._history:
            return False
        if delta < 0:
            if self._history_index is None:
                self._draft = self.text
                self._history_index = len(self._history) - 1
            else:
                self._history_index = max(0, self._history_index - 1)
            self._load_history(self._history[self._history_index])
            return True
        if self._history_index is None:
            return False
        if self._history_index >= len(self._history) - 1:
            self._history_index = None
            self._load_history(self._draft)
        else:
            self._history_index += 1
            self._load_history(self._history[self._history_index])
        return True

    # -- key handling ------------------------------------------------------

    async def _on_key(self, event: events.Key) -> None:
        key = event.key

        if self.popup_open:
            if key in ("tab", "enter"):
                event.stop()
                event.prevent_default()
                self.post_message(self.PopupAcceptRequested())
                return
            if key == "up":
                event.stop()
                event.prevent_default()
                self.post_message(self.PopupNavigateRequested(-1))
                return
            if key == "down":
                event.stop()
                event.prevent_default()
                self.post_message(self.PopupNavigateRequested(1))
                return
            if key == "escape":
                event.stop()
                event.prevent_default()
                self.post_message(self.PopupDismissRequested())
                return
            await super()._on_key(event)
            return

        if key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.ComposerSubmitted(self.text))
            return
        if key in ("shift+enter", "ctrl+j"):
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        if key == "escape":
            event.stop()
            event.prevent_default()
            self.post_message(self.CancelRequested())
            return
        if key == "up" and self.cursor_location[0] == 0:
            if self._cycle_history(-1):
                event.stop()
                event.prevent_default()
                return
        if key == "down" and self.cursor_location[0] == self.document.line_count - 1:
            if self._cycle_history(1):
                event.stop()
                event.prevent_default()
                return

        await super()._on_key(event)

    # -- sizing ------------------------------------------------------------

    def _autosize(self) -> None:
        lines = max(1, min(6, self.document.line_count))
        if lines != self._last_line_count:
            self._last_line_count = lines
            self.styles.height = lines

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        # Keep bubbling: NexusApp watches this message to drive the suggestion
        # popup, so stopping it here would make the palette deaf to typing.
        self._autosize()

    def on_mount(self) -> None:
        self._autosize()


class ComposerBox(Vertical):
    """Composer container: input plus the route/mode/cwd/timer meta strip."""

    DEFAULT_CSS = """
    ComposerBox {
        height: auto;
    }
    ComposerBox #composer-meta Static {
        width: auto;
        height: 1;
        padding: 0 1 0 0;
        color: $text-muted;
    }
    ComposerBox #composer-meta #composer-spacer {
        width: 1fr;
        padding: 0;
    }
    ComposerBox #composer-meta #composer-timer {
        padding: 0;
    }
    """

    def __init__(self, *children: Any, **kwargs: Any) -> None:
        kwargs.setdefault("id", "composer-container")
        super().__init__(*children, **kwargs)
        self._route: str = ""
        self._model: str = ""
        self._quota_tier: str = "FREE"
        self._use_llm: bool = True
        self._cwd_label: str = ""
        self._think_label: str = ""
        self._attach_name: str = ""
        self._busy_running: bool = False
        self._run_label: str = ""
        self._run_started: float = 0.0
        self._tick_timer: Any = None
        self._spin_frame: int = 0
        self._history_entries: list[str] = []
        self._popup_open: bool = False
        self._last_meta: dict[str, str] = {}
        self._last_tier: Optional[str] = None
        self._entity_mode: bool = True
        self._profile_name: str = "default"
        self._pinned_route: Optional[str] = None
        self._last_route: Optional[str] = None

    def compose(self) -> ComposeResult:
        yield ComposerTextArea(id="composer-input")
        with Horizontal(id="composer-meta"):
            yield Static("", id="composer-route")
            yield Static("", id="composer-mode")
            yield Static("", id="composer-cwd")
            yield Static("", id="composer-spacer")
            yield Static("", id="composer-timer")

    @property
    def input(self) -> ComposerTextArea:
        return self.query_one("#composer-input", ComposerTextArea)

    def on_mount(self) -> None:
        self._sync_timer()
        if self._history_entries:
            self.input.set_history(self._history_entries)
        self.input.popup_open = self._popup_open
        self._sync_meta()

    def on_unmount(self) -> None:
        if self._tick_timer is not None:
            self._tick_timer.stop()
            self._tick_timer = None

    def _sync_timer(self) -> None:
        """Run the elapsed-time ticker only while a run is actually in flight."""
        if self._busy_running and self.is_mounted:
            if self._tick_timer is None:
                self._tick_timer = self.set_interval(0.16, self._tick)
        elif self._tick_timer is not None:
            self._tick_timer.stop()
            self._tick_timer = None

    def _route_text(self) -> str:
        pinned = self._pinned_route or (self._route if not self._entity_mode else None)
        if pinned:
            return (
                f"[{tier_color(self._quota_tier)}]@{escape(pinned)}[/] "
                f"[dim]{escape(_short_model(self._model or 'cli', limit=20))}[/]"
            )
        if self._entity_mode:
            label = f"[#22c55e]entity[/] [dim]· {escape(self._profile_name)}[/]"
            if self._last_route:
                label += f" [dim]via @{escape(self._last_route)}[/]"
            return label
        return "[dim]front-desk[/]"

    def _mode_text(self) -> str:
        text = "[dim]·[/] [#38bdf8]LLM[/]" if self._use_llm else "[dim]· fast[/]"
        if self._think_label:
            text += f" [dim]· think:{escape(self._think_label)}[/]"
        if self._attach_name:
            text += f" [dim]· +{escape(self._attach_name)}[/]"
        return text

    def _timer_text(self) -> str:
        if not self._busy_running:
            return "[dim]idle[/]"
        elapsed = max(0.0, time.monotonic() - self._run_started)
        label = self._run_label or "running"
        spin = _SPINNER_FRAMES[self._spin_frame % len(_SPINNER_FRAMES)]
        dots = _DOT_FRAMES[self._spin_frame % len(_DOT_FRAMES)]
        return (
            f"[#38bdf8]{spin}[/] [bold]{escape(label)}[/] {dots} "
            f"[dim]{elapsed:0.1f}s[/]"
        )

    def _sync_meta(self) -> None:
        if not self.is_mounted:
            return
        cwd_txt = self._cwd_label or "~"
        if len(cwd_txt) > 18:
            cwd_txt = "…" + cwd_txt[-17:]
        values = {
            "#composer-route": self._route_text(),
            "#composer-mode": self._mode_text(),
            "#composer-cwd": f"[dim]· {escape(cwd_txt)}[/]",
            "#composer-timer": self._timer_text(),
        }
        for selector, text in values.items():
            if self._last_meta.get(selector) == text:
                continue
            try:
                self.query_one(selector, Static).update(text)
            except NoMatches:
                continue
            self._last_meta[selector] = text

    def _tick(self) -> None:
        if not self._busy_running:
            return
        self._spin_frame = (self._spin_frame + 1) % 20
        text = self._timer_text()
        try:
            self.query_one("#composer-timer", Static).update(text)
        except NoMatches:
            return
        self._last_meta["#composer-timer"] = text

    def set_route_state(
        self,
        route: str,
        model: str,
        quota_tier: str,
        use_llm_decision: bool,
        cwd_label: str,
        think_label: str = "",
        attach_name: str = "",
        *,
        entity_mode: bool = True,
        profile_name: str = "default",
        pinned_route: Optional[str] = None,
        last_route: Optional[str] = None,
    ) -> None:
        self._route = route or ""
        self._model = model or ""
        self._quota_tier = quota_tier or "FREE"
        self._use_llm = bool(use_llm_decision)
        self._cwd_label = cwd_label or ""
        self._think_label = think_label or ""
        self._attach_name = attach_name or ""
        self._entity_mode = bool(entity_mode)
        self._profile_name = profile_name or "default"
        self._pinned_route = pinned_route
        self._last_route = last_route
        self._sync_tier()
        self._sync_meta()

    def _sync_tier(self) -> None:
        tier = normalize_tier(self._quota_tier)
        if tier == self._last_tier:
            return
        self.remove_class("tier-go", "tier-local", "tier-premium")
        if tier in ("GO", "PLAN"):
            self.add_class("tier-go")
        elif tier == "LOCAL":
            self.add_class("tier-local")
        elif tier == "PREMIUM":
            self.add_class("tier-premium")
        self._last_tier = tier

    def set_running(self, running: bool, label: str = "", elapsed_s: float = 0.0) -> None:
        self._busy_running = bool(running)
        self._run_label = label or ""
        self._run_started = time.monotonic() - max(0.0, float(elapsed_s))
        self._sync_meta()
        self._sync_timer()

    def set_history(self, entries: list[str]) -> None:
        self._history_entries = list(entries)
        if self.is_mounted:
            self.input.set_history(self._history_entries)

    def focus_input(self) -> None:
        if self.is_mounted:
            self.input.focus()

    def set_popup_open(self, open_: bool) -> None:
        self._popup_open = bool(open_)
        if self.is_mounted:
            self.input.popup_open = self._popup_open


class InspectorSidebar(Vertical):
    """Right-hand inspector: delegation tree, route table and town square."""

    @_handler("on_route_picked")
    class RoutePicked(Message):
        def __init__(self, route_key: str) -> None:
            super().__init__()
            self.route_key = route_key

    @_handler("on_square_post_requested")
    class SquarePostRequested(Message):
        def __init__(self, body: str) -> None:
            super().__init__()
            self.body = body

    @_handler("on_refresh_requested")
    class RefreshRequested(Message):
        pass

    DEFAULT_CSS = """
    InspectorSidebar {
        width: 44;
        height: 1fr;
        padding: 0;
    }
    InspectorSidebar TabbedContent {
        height: 1fr;
    }
    InspectorSidebar ContentSwitcher {
        height: 1fr;
    }
    InspectorSidebar TabPane {
        height: 1fr;
        padding: 0;
    }
    InspectorSidebar #tree-scroll {
        height: 1fr;
        scrollbar-size: 1 1;
    }
    InspectorSidebar #tree-view {
        height: auto;
        width: 1fr;
        padding: 0;
        color: $text;
    }
    InspectorSidebar #routes-list {
        height: 1fr;
        background: transparent;
        border: none;
        padding: 0;
    }
    InspectorSidebar #square-scroll {
        height: 1fr;
        scrollbar-size: 1 1;
    }
    InspectorSidebar #square-board,
    InspectorSidebar #square-posts {
        height: auto;
        width: 1fr;
        padding: 0;
        color: $text;
    }
    InspectorSidebar #square-posts {
        margin-top: 1;
    }
    InspectorSidebar #square-quick-post {
        height: 3;
        border: tall $border-blurred;
        background: $panel;
    }
    """

    def __init__(
        self,
        *children: Any,
        events_path: Optional[Path] = None,
        square_path: Optional[Path] = None,
        board_path: Optional[Path] = None,
        routes_path: Optional[Path] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("id", "sidebar")
        super().__init__(**kwargs)
        # `None` means "use the module default", which keeps every reader
        # (`read_events`, `read_square`, `read_board`, `load_routes`) unchanged.
        self.events_path = Path(events_path) if events_path is not None else None
        self.square_path = Path(square_path) if square_path is not None else None
        self.board_path = Path(board_path) if board_path is not None else None
        self.routes_path = Path(routes_path) if routes_path is not None else None
        self._dirty_tree: bool = False
        self._dirty_routes: bool = True
        self._dirty_square: bool = True
        self._events_cache_key: Optional[tuple[int, int, int, Optional[str]]] = None
        self._last_tree_markup: str = ""
        self._last_board_markup: str = ""
        self._last_posts_markup: str = ""
        self._last_routes_sig: Optional[tuple] = None
        self._routes_override: Optional[dict] = None

    def set_routes_data(self, routes_data: Optional[dict]) -> None:
        """Replace (or clear) the route table shown by the sidebar lazily."""
        self._routes_override = dict(routes_data) if routes_data is not None else None
        self._dirty_routes = True
        self._last_routes_sig = None
        if self.is_mounted and not self.has_class("hidden") and self._active_tab_id() == "tab-routes":
            self.refresh_routes()

    def compose(self) -> ComposeResult:
        with TabbedContent(initial="tab-tree"):
            with TabPane("Tree", id="tab-tree"):
                with VerticalScroll(id="tree-scroll"):
                    yield Static("", id="tree-view")
            with TabPane("Routes", id="tab-routes"):
                yield OptionList(id="routes-list")
            with TabPane("Square", id="tab-square"):
                with VerticalScroll(id="square-scroll"):
                    yield Static("", id="square-board")
                    yield Static("", id="square-posts")
                yield Input(placeholder="Post to the town square…", id="square-quick-post")

    def on_mount(self) -> None:
        if not self.has_class("hidden"):
            self.refresh_tree()
        else:
            self._dirty_tree = True

    def _active_tab_id(self) -> str:
        try:
            return str(self.query_one(TabbedContent).active or "tab-tree")
        except NoMatches:
            return "tab-tree"

    # -- tree --------------------------------------------------------------

    def refresh_tree(self, trace_id: Optional[str] = None, *, force: bool = False) -> None:
        if not self.is_mounted:
            return
        if not force and (self.has_class("hidden") or self._active_tab_id() != "tab-tree"):
            self._dirty_tree = True
            return
        width = max(20, int(self.content_size.width or 42) - 2)
        target = self.events_path if self.events_path is not None else DEFAULT_EVENTS_PATH
        try:
            st = target.stat()
            file_sig = (st.st_mtime_ns, st.st_size)
        except OSError:
            file_sig = (0, 0)
        cache_key = (file_sig[0], file_sig[1], width, trace_id)
        if not force and not self._dirty_tree and cache_key == self._events_cache_key:
            return
        events = read_events(path=self.events_path, limit=400)
        if width >= _TREE_MIN_WIDTH:
            text = render_delegation_tree(events, trace_id=trace_id, color=False, width=width)
        else:
            text = _render_compact_tree(events, trace_id=trace_id, width=width)
        markup = escape(text)
        self._events_cache_key = cache_key
        self._dirty_tree = False
        if markup == self._last_tree_markup:
            return
        try:
            self.query_one("#tree-view", Static).update(markup)
        except NoMatches:
            return
        self._last_tree_markup = markup

    # -- routes ------------------------------------------------------------

    def refresh_routes(self, *, validate: bool = False) -> None:
        if not self.is_mounted:
            return
        data = self._routes_override if self._routes_override is not None else load_routes(self.routes_path)
        routes = data.get("routes", {})
        missing: set[str] = set()
        if validate:
            available = list_opencode_models()
            if available:
                missing = {
                    key
                    for key, cfg in routes.items()
                    if str(cfg.get("model", "")) and str(cfg.get("model", "")) not in available
                }
        try:
            options = self.query_one("#routes-list", OptionList)
        except NoMatches:
            return
        sig = tuple(
            (
                key,
                str(cfg.get("model", "cli")),
                str(cfg.get("backend", "opencode")),
                str(cfg.get("quota_tier", "FREE")),
                key in missing,
            )
            for key, cfg in routes.items()
        )
        if not validate and sig == self._last_routes_sig and options.option_count == len(routes):
            self._dirty_routes = False
            return
        options.clear_options()
        for key, cfg in routes.items():
            model = str(cfg.get("model", "cli"))
            backend = str(cfg.get("backend", "opencode"))
            badge = quota_markup(str(cfg.get("quota_tier", "FREE")))
            prompt = (
                f"[bold]@{escape(key)}[/]  [dim]{escape(model)}[/]  {badge}"
                f"  [dim]{escape(backend)}[/]"
            )
            if key in missing:
                prompt += "  [#ef4444]✗ missing[/]"
            options.add_option(Option(prompt, id=key))
        self._last_routes_sig = sig
        self._dirty_routes = False

    # -- square ------------------------------------------------------------

    def refresh_square(self) -> None:
        if not self.is_mounted:
            return
        board = read_board(self.board_path)
        posts = read_square(limit=8, square_path=self.square_path)
        if board:
            board_text = f"[bold]BOARD.md[/]\n{escape(board)}"
        else:
            board_text = "[dim]BOARD.md[/]\n[dim](empty — no pinned notes yet)[/]"

        if posts:
            lines = ["[bold]RECENT POSTS[/]"]
            for post in posts:
                stamp = time.strftime("%H:%M", time.localtime(float(post.get("ts", 0) or 0)))
                author = escape(str(post.get("author", "agent")))
                tag = escape(str(post.get("tag", "note")))
                body = escape(" ".join(str(post.get("body", "")).split())[:220])
                lines.append(f"[dim]{stamp}[/] [bold]{author}[/] [dim]#{tag}[/] {body}")
            posts_text = "\n".join(lines)
        else:
            posts_text = "[bold]RECENT POSTS[/]\n[dim](nothing posted yet)[/]"

        try:
            if board_text != self._last_board_markup:
                self.query_one("#square-board", Static).update(board_text)
                self._last_board_markup = board_text
            if posts_text != self._last_posts_markup:
                self.query_one("#square-posts", Static).update(posts_text)
                self._last_posts_markup = posts_text
        except NoMatches:
            return
        self._dirty_square = False

    # -- events ------------------------------------------------------------

    def set_open(self, open_: bool) -> None:
        was_hidden = self.has_class("hidden")
        self.set_class(not open_, "hidden")
        if open_ and was_hidden and self.is_mounted:
            tab = self._active_tab_id()
            if tab == "tab-tree":
                self.refresh_tree()
            elif tab == "tab-routes" and self._dirty_routes:
                self.refresh_routes()
            elif tab == "tab-square" and self._dirty_square:
                self.refresh_square()

    def select_tab(self, tab_id: str) -> None:
        """Switch the inspector to `tab-tree`, `tab-routes` or `tab-square`."""
        if not self.is_mounted:
            return
        try:
            self.query_one(TabbedContent).active = tab_id
        except NoMatches:
            return

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "routes-list":
            return
        event.stop()
        route_key = event.option.id or ""
        if route_key:
            self.post_message(self.RoutePicked(route_key))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "square-quick-post":
            return
        event.stop()
        body = event.value.strip()
        if body:
            self.post_message(self.SquarePostRequested(body))
        event.input.value = ""

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        pane_id = event.pane.id if event.pane is not None else ""
        if pane_id == "tab-tree":
            self.refresh_tree()
        elif pane_id == "tab-routes":
            self.refresh_routes()
        elif pane_id == "tab-square":
            self.refresh_square()


def _render_compact_tree(
    events: list[dict],
    *,
    trace_id: Optional[str] = None,
    width: int = 40,
) -> str:
    """Render a border-free indented tree that survives a 40-column sidebar."""
    tree = reconstruct_tree(events, trace_id=trace_id)
    spans: dict[str, dict] = tree["spans"]
    children_map: dict[Optional[str], list[str]] = tree["children"]
    roots: list[str] = tree["roots"]
    summary: dict = tree["summary"]

    if not spans:
        return "(no delegation spans recorded yet)"

    lines: list[str] = []

    def walk(span_id: str, prefix: str, is_last: bool) -> None:
        span = spans[span_id]
        status = str(span.get("status", "RUNNING")).upper()
        icon = format_status_icon(status, color=False) or _status_icon(status)
        route = str(span.get("route") or span.get("intent") or "root")
        model = _short_model(str(span.get("model") or "cli"), limit=22)
        elapsed_ms = int(span.get("elapsed_ms") or 0)
        connector = "●  " if not prefix else ("└─ " if is_last else "├─ ")
        line = f"{prefix}{connector}{icon} [{route}] {model} {format_duration(elapsed_ms)}"
        lines.append(truncate_ansi(line, max(8, width), "…"))
        child_prefix = prefix + ("   " if is_last else "│  ")
        children = children_map.get(span_id, [])
        for index, child_id in enumerate(children):
            walk(child_id, child_prefix, index == len(children) - 1)

    for root_index, root_id in enumerate(roots):
        walk(root_id, "", root_index == len(roots) - 1)

    premium_spans = int(summary.get("premium_spans", 0) or 0)
    footer = (
        f"{summary.get('total_spans', 0)} spans · "
        f"{format_tokens(int(summary.get('tokens_in', 0) or 0))}→"
        f"{format_tokens(int(summary.get('tokens_out', 0) or 0))} tok"
    )
    if premium_spans:
        footer += f" · PREMIUM {premium_spans}"
    else:
        footer += " · $0 premium"
    lines.append(truncate_ansi(footer, max(8, width), "…"))
    return "\n".join(lines)


class HintFooter(Static):
    """Single-line keybind hint strip."""

    DEFAULT_HINTS = (
        "[dim]Enter[/] send  [dim]⇧Enter[/] newline  [dim]Tab[/] accept  "
        "[dim]↑/↓[/] history  [dim]Ctrl+B[/] sidebar  [dim]Ctrl+P[/] commands  "
        "[dim]Ctrl+O[/] thinking/tools  [dim]Ctrl+Q[/] quit"
    )

    DEFAULT_CSS = """
    HintFooter {
        height: 1;
        color: $footer-foreground;
        padding: 0 2;
    }
    """

    def __init__(self, text: str = "", **kwargs: Any) -> None:
        kwargs.setdefault("id", "hint-footer")
        super().__init__(text or self.DEFAULT_HINTS, **kwargs)

    def set_hints(self, text: str) -> None:
        self.update(text or self.DEFAULT_HINTS)
