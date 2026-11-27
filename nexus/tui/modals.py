"""Modal screens for the Nexus TUI: route/theme/session pickers and help.

Each modal is a `ModalScreen` that owns its own markup and returns plain data
through `dismiss(...)`, so the app layer only reacts to a small result:

* `RoutePickerModal` -> route key, `"__release__"`, or `None`
* `ThemePickerModal` -> theme name or `None`
* `SessionPickerModal` -> `("switch"|"new"|"delete", session_id)` or `None`
* `HelpModal` -> `None`

Every dialog is a 0-gap skeletal panel that stays inside the terminal width,
matching the terminal styling used by the rest of the TUI.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from rich.markup import escape
from textual import events, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option

from nexus.router import list_opencode_models, validate_routes
from nexus.sessions import TuiSession
from nexus.tui.themes import NEXUS_THEMES, THEME_ORDER, quota_markup
from nexus.visualizer import format_tokens, render_panel

RELEASE_KEY = "__release__"
"""Sentinel route key that means "hand control back to the front desk"."""

THEME_DESCRIPTIONS = {
    "nexus-charcoal": "Monotone charcoal · default",
    "oled-black": "True black for OLED panels",
    "opencode-dark": "OpenCode violet accent",
    "github-slate": "GitHub-style slate",
    "catppuccin-mocha": "Catppuccin Mocha",
}


def _clip(text: str, limit: int) -> str:
    """Collapse whitespace and clip to `limit` characters, marking the cut."""
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: max(1, limit - 1)] + "…"


def _relative_ts(ts: float) -> str:
    """Format a timestamp as a short relative age, falling back to clock time."""
    stamp = float(ts or 0.0)
    if not stamp:
        return "never"
    delta = max(0.0, time.time() - stamp)
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86_400:
        return f"{int(delta // 3600)}h ago"
    return time.strftime("%b %d %H:%M", time.localtime(stamp))


class _ShortcutInput(Input):
    """Filter field that forwards single-letter modal shortcuts while empty.

    A focused `Input` swallows printable keys, so `r`/`v`/`n`/`d` shortcuts
    would never reach the screen bindings. For an empty field there is nothing
    to filter yet, so those keys run the screen action instead.
    """

    def __init__(
        self,
        *args: Any,
        shortcuts: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._shortcuts = dict(shortcuts or {})

    async def _on_key(self, event: events.Key) -> None:
        action = self._shortcuts.get(event.key) if not self.value else None
        if action is not None:
            event.stop()
            event.prevent_default()
            await self.screen.run_action(action)
            return
        await super()._on_key(event)


class RoutePickerModal(ModalScreen[Optional[str]]):
    """Fuzzy route picker with live model-pin validation."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("r", "release", "Front-desk"),
        Binding("v", "validate", "Validate"),
    ]

    DEFAULT_CSS = """
    RoutePickerModal .route-dialog {
        width: 88;
        max-width: 94%;
        height: auto;
        max-height: 84%;
    }
    RoutePickerModal #route-filter {
        height: 3;
        border: tall $border-blurred;
        background: $panel;
        margin-bottom: 1;
    }
    RoutePickerModal #route-options {
        height: auto;
        max-height: 22;
        background: transparent;
        border: none;
        padding: 0;
    }
    RoutePickerModal .modal-hint {
        height: auto;
        color: $text-muted;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        routes_data: dict,
        active_route: Optional[str] = None,
        available_models: Optional[set[str]] = None,
    ) -> None:
        super().__init__()
        self._routes_data = routes_data or {}
        self._active_route = active_route
        self._available_models = available_models
        self._validating = False

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-dialog route-dialog"):
            yield Static(
                "[bold][4:routes] SWITCH SPECIALIST ROUTE[/]",
                classes="modal-title",
            )
            yield _ShortcutInput(
                placeholder="fuzzy filter: route, model or purpose…",
                id="route-filter",
                shortcuts={"r": "release", "v": "validate"},
            )
            yield OptionList(id="route-options")
            yield Static(self._hint_text(), classes="modal-hint")

    def on_mount(self) -> None:
        self._rebuild_options()
        self.query_one("#route-filter", Input).focus()

    # -- data --------------------------------------------------------------

    def _hint_text(self) -> str:
        return (
            "[dim]Enter[/] switch  [dim]r[/] front-desk  "
            "[dim]v[/] validate pins  [dim]Esc[/] cancel"
        )

    def _routes(self) -> list[tuple[str, dict]]:
        routes = self._routes_data.get("routes", {})
        return [(str(key), dict(cfg or {})) for key, cfg in routes.items()]

    def _filtered(self, query: str) -> list[tuple[str, dict]]:
        needle = query.strip().lstrip("@").lower()
        routes = self._routes()
        if not needle:
            return routes
        return [
            (key, cfg)
            for key, cfg in routes
            if needle in key.lower()
            or needle in str(cfg.get("model", "")).lower()
            or needle in str(cfg.get("backend", "")).lower()
            or needle in str(cfg.get("description", "")).lower()
        ]

    def _validation_mark(self, cfg: dict) -> str:
        if self._available_models is None:
            return ""
        if str(cfg.get("backend", "opencode")) != "opencode":
            return ""
        model = str(cfg.get("model", ""))
        if model in self._available_models:
            return "  [#22c55e]✓[/]"
        return "  [#ef4444]✗ missing[/]"

    def _release_label(self) -> str:
        mark = "● " if self._active_route is None else "  "
        return (
            f"{mark}[bold]@front-desk[/]  [dim]Front-Desk Auto Router[/]  "
            f"{quota_markup('FREE')}  [dim]LLM or fast keyword decision[/]"
        )

    def _route_label(self, key: str, cfg: dict) -> str:
        backend = str(cfg.get("backend", "opencode"))
        model = _clip(str(cfg.get("model", "cli")), 34)
        badge = quota_markup(str(cfg.get("quota_tier", "FREE")))
        mark = "● " if key == self._active_route else "  "
        line = (
            f"{mark}[bold]@{escape(key)}[/]  "
            f"[dim]{escape(backend)}:{escape(model)}[/]  {badge}"
            f"{self._validation_mark(cfg)}"
        )
        desc = _clip(str(cfg.get("description", "")), 30)
        if desc:
            line += f"  [dim]{escape(desc)}[/]"
        return line

    def _rebuild_options(self, query: Optional[str] = None) -> None:
        try:
            options = self.query_one("#route-options", OptionList)
        except NoMatches:
            return
        if query is None:
            try:
                query = self.query_one("#route-filter", Input).value
            except NoMatches:
                query = ""
        previous = options.highlighted_option.id if options.highlighted is not None else None

        options.clear_options()
        if not query.strip():
            options.add_option(Option(self._release_label(), id=RELEASE_KEY))
        for key, cfg in self._filtered(query):
            options.add_option(Option(self._route_label(key, cfg), id=key))
        if options.option_count == 0:
            options.add_option(Option("[dim]no matching route[/]", id="no-match", disabled=True))
        restore = 0
        if previous is not None:
            for index in range(options.option_count):
                if options.get_option_at_index(index).id == previous:
                    restore = index
                    break
        options.highlighted = restore

    # -- events ------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "route-filter":
            return
        event.stop()
        self._rebuild_options(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "route-filter":
            return
        event.stop()
        self._confirm_highlighted()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "route-options":
            return
        event.stop()
        self.dismiss(str(event.option.id or RELEASE_KEY))

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        # Nothing to preview; the highlight is the pending choice.
        return

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "route-cancel":
            event.stop()
            self.dismiss(None)

    def _confirm_highlighted(self) -> None:
        try:
            options = self.query_one("#route-options", OptionList)
        except NoMatches:
            return
        option = options.highlighted_option
        if option is not None and option.id:
            self.dismiss(str(option.id))

    # -- actions -----------------------------------------------------------

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_release(self) -> None:
        self.dismiss(RELEASE_KEY)

    def action_validate(self) -> None:
        if self._validating:
            return
        self._validating = True
        self.notify("Validating route pins…", timeout=2)
        self._validate_worker()

    @work(thread=True, exclusive=True, group="route-validate")
    def _validate_worker(self) -> None:
        try:
            ok, errors = validate_routes(self._routes_data, refresh_cache=True)
            available = list_opencode_models()
        except Exception as exc:  # pragma: no cover - defensive: never kill the modal
            ok, errors, available = False, [str(exc)], set()
        self._ui(self._apply_validation, ok, errors, available)

    def _ui(self, callback: Callable[..., Any], *args: Any) -> None:
        """Marshal a worker result back to the UI thread."""
        if not self.app.is_running:
            return
        try:
            self.app.call_from_thread(callback, *args)
        except RuntimeError:
            return

    def _apply_validation(self, ok: bool, errors: list[str], available: set[str]) -> None:
        self._validating = False
        self._available_models = set(available)
        self._rebuild_options()
        if ok:
            self.notify(f"Routes validated · {len(available)} models available", timeout=3)
            return
        first = errors[0] if errors else "validation failed"
        self.notify(
            f"{len(errors)} route pin problem(s): {_clip(first, 60)}",
            severity="warning",
            timeout=5,
        )


class ThemePickerModal(ModalScreen[Optional[str]]):
    """Theme list that live-previews each theme while the highlight moves."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", priority=True)]

    DEFAULT_CSS = """
    ThemePickerModal .theme-dialog {
        width: 72;
        max-width: 94%;
        height: auto;
        max-height: 80%;
    }
    ThemePickerModal #theme-options {
        height: auto;
        max-height: 14;
        background: transparent;
        border: none;
        padding: 0;
    }
    ThemePickerModal .modal-hint {
        height: auto;
        color: $text-muted;
        margin-top: 1;
    }
    """

    def __init__(self, current_theme: str) -> None:
        super().__init__()
        self._current_theme = current_theme

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-dialog theme-dialog"):
            yield Static("[bold][5:theme] SWITCH THEME[/]", classes="modal-title")
            yield OptionList(id="theme-options")
            yield Static(
                "[dim]↑/↓[/] preview  [dim]Enter[/] apply  [dim]Esc[/] keep current",
                classes="modal-hint",
            )

    def on_mount(self) -> None:
        options = self.query_one("#theme-options", OptionList)
        options.clear_options()
        for name in THEME_ORDER:
            options.add_option(Option(self._label(name), id=name))
        if options.option_count:
            options.highlighted = (
                THEME_ORDER.index(self._current_theme)
                if self._current_theme in THEME_ORDER
                else 0
            )
        options.focus()

    def _label(self, name: str) -> str:
        theme = NEXUS_THEMES.get(name)
        if theme is None:
            return f"  {escape(name)}"
        swatch = f"[{theme.primary} on {theme.background}]███[/]"
        mark = "● " if name == self._current_theme else "  "
        desc = escape(THEME_DESCRIPTIONS.get(name, ""))
        return f"{mark}{swatch} [bold]{escape(name)}[/]  [dim]{desc}[/]"

    # -- events ------------------------------------------------------------

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_list.id != "theme-options":
            return
        name = str(event.option.id or "")
        if name in NEXUS_THEMES:
            self.app.theme = name

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "theme-options":
            return
        event.stop()
        name = str(event.option.id or "")
        self.dismiss(name or None)

    def action_cancel(self) -> None:
        if self._current_theme in NEXUS_THEMES:
            self.app.theme = self._current_theme
        self.dismiss(None)


class SessionPickerModal(ModalScreen[Optional[tuple[str, str]]]):
    """Filterable session list with switch / new / delete actions."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("n", "new_session", "New"),
        Binding("d", "delete_session", "Delete"),
    ]

    DEFAULT_CSS = """
    SessionPickerModal .session-dialog {
        width: 88;
        max-width: 94%;
        height: auto;
        max-height: 84%;
    }
    SessionPickerModal #session-filter {
        height: 3;
        border: tall $border-blurred;
        background: $panel;
        margin-bottom: 1;
    }
    SessionPickerModal #session-options {
        height: auto;
        max-height: 18;
        background: transparent;
        border: none;
        padding: 0;
    }
    SessionPickerModal #session-actions {
        height: 3;
        margin-top: 1;
    }
    SessionPickerModal #session-actions Button {
        height: 1;
        min-width: 9;
        border: none;
        margin-right: 1;
        background: $panel;
        color: $text-muted;
        content-align: center middle;
    }
    SessionPickerModal #session-actions Button:hover {
        background: $primary;
        color: $background;
    }
    SessionPickerModal .modal-hint {
        height: auto;
        color: $text-muted;
        margin-top: 1;
    }
    """

    def __init__(self, sessions: list[TuiSession], active_id: str) -> None:
        super().__init__()
        self._sessions = list(sessions)
        self._active_id = active_id

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-dialog session-dialog"):
            yield Static("[bold][6:sessions] SWITCH SESSION[/]", classes="modal-title")
            yield _ShortcutInput(
                placeholder="fuzzy filter: title, route or id…",
                id="session-filter",
                shortcuts={"n": "new_session", "d": "delete_session"},
            )
            yield OptionList(id="session-options")
            with Horizontal(id="session-actions"):
                yield Button("new", id="session-new")
                yield Button("delete", id="session-delete")
                yield Button("close", id="session-close")
            yield Static(
                "[dim]Enter[/] switch  [dim]n[/] new  [dim]d[/] delete  [dim]Esc[/] cancel",
                classes="modal-hint",
            )

    def on_mount(self) -> None:
        self._rebuild_options()
        self.query_one("#session-filter", Input).focus()

    # -- data --------------------------------------------------------------

    def _filtered(self, query: str) -> list[TuiSession]:
        needle = query.strip().lower()
        if not needle:
            return list(self._sessions)
        return [
            session
            for session in self._sessions
            if needle in session.title.lower()
            or needle in session.id.lower()
            or needle in str(session.active_route or "front-desk").lower()
        ]

    def _label(self, session: TuiSession) -> str:
        mark = "● " if session.id == self._active_id else "  "
        route = f"@{session.active_route}" if session.active_route else "front-desk"
        tokens = (
            f"{format_tokens(session.total_tokens_in)}→"
            f"{format_tokens(session.total_tokens_out)} tok"
        )
        return (
            f"{mark}[bold]{escape(_clip(session.title, 30))}[/]  "
            f"[dim]{escape(route)} · {len(session.items)} items · {tokens} · "
            f"{_relative_ts(session.updated_at)}[/]"
        )

    def _rebuild_options(self, query: Optional[str] = None) -> None:
        try:
            options = self.query_one("#session-options", OptionList)
        except NoMatches:
            return
        if query is None:
            try:
                query = self.query_one("#session-filter", Input).value
            except NoMatches:
                query = ""
        previous = options.highlighted_option.id if options.highlighted is not None else None

        options.clear_options()
        for session in self._filtered(query):
            options.add_option(Option(self._label(session), id=session.id))
        if options.option_count == 0:
            options.add_option(Option("[dim]no matching session[/]", id="no-match", disabled=True))
            options.highlighted = 0
            return

        restore = 0
        if previous is not None:
            for index in range(options.option_count):
                if options.get_option_at_index(index).id == previous:
                    restore = index
                    break
        else:
            for index in range(options.option_count):
                if options.get_option_at_index(index).id == self._active_id:
                    restore = index
                    break
        options.highlighted = restore

    def _highlighted_session_id(self) -> str:
        try:
            options = self.query_one("#session-options", OptionList)
        except NoMatches:
            return self._active_id
        option = options.highlighted_option
        candidate = str(option.id or "") if option is not None else ""
        if candidate in {session.id for session in self._sessions}:
            return candidate
        return self._active_id

    # -- events ------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "session-filter":
            return
        event.stop()
        self._rebuild_options(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "session-filter":
            return
        event.stop()
        self.action_switch()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "session-options":
            return
        event.stop()
        session_id = str(event.option.id or "")
        if session_id in {session.id for session in self._sessions}:
            self.dismiss(("switch", session_id))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "session-new":
            event.stop()
            self.action_new_session()
        elif button_id == "session-delete":
            event.stop()
            self.action_delete_session()
        elif button_id == "session-close":
            event.stop()
            self.action_cancel()

    # -- actions -----------------------------------------------------------

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_switch(self) -> None:
        session_id = self._highlighted_session_id()
        if session_id:
            self.dismiss(("switch", session_id))

    def action_new_session(self) -> None:
        self.dismiss(("new", ""))

    def action_delete_session(self) -> None:
        session_id = self._highlighted_session_id()
        if session_id:
            self.dismiss(("delete", session_id))


class HelpModal(ModalScreen[None]):
    """Two-column skeletal reference for every shortcut and slash command."""

    BINDINGS = [
        Binding("escape", "close", "Close", priority=True),
        Binding("q", "close", "Close"),
        Binding("enter", "close", "Close"),
    ]

    SHORTCUTS = (
        ("Ctrl+P", "Command palette"),
        ("Ctrl+K", "Route picker"),
        ("Ctrl+B", "Toggle sidebar"),
        ("Ctrl+N", "New session"),
        ("Ctrl+L", "Session picker"),
        ("Ctrl+T", "Theme picker"),
        ("Ctrl+G", "Toggle WebUI / CLI"),
        ("Ctrl+O", "Thinking / tools"),
        ("Ctrl+Tab", "Next session"),
        ("Ctrl+Shift+Tab", "Previous session"),
        ("Ctrl+Q", "Quit Nexus"),
        ("Ctrl+C", "Cancel / clear"),
        ("Esc", "Dismiss popup"),
        ("F1 / ?", "This help"),
        ("Enter", "Send prompt"),
        ("Shift+Enter", "Newline"),
        ("Up / Down", "Prompt history"),
        ("Tab", "Accept suggestion"),
    )

    COMMANDS = (
        ("/profile", "switch profile"),
        ("/pin", "pin specialist"),
        ("/unpin", "unified entity"),
        ("/mode", "entity / sticky"),
        ("/stats", "perf telemetry"),
        ("/switch", "lock a route"),
        ("/release", "front-desk"),
        ("/council", "3-seat council"),
        ("/routes", "route table"),
        ("/validate", "audit pins"),
        ("/tree", "delegation tree"),
        ("/square", "town square"),
        ("/post", "post a note"),
        ("/board", "pin the board"),
        ("/new", "new session"),
        ("/sessions", "session list"),
        ("/theme", "switch theme"),
        ("/ui", "webui / cli mode"),
        ("/sidebar", "toggle sidebar"),
        ("/verbose", "detail & thinking"),
        ("/quiet", "silent status"),
        ("/fast", "keyword routing"),
        ("/llm", "LLM routing"),
        ("/think", "thinking effort"),
        ("/attach", "attach next prompt"),
        ("/copy", "copy answer"),
        ("/save", "save code block"),
        ("/clear", "clear transcript"),
        ("/tui", "external TUI"),
        ("/help", "this help"),
        ("/quit", "quit Nexus"),
        ("/exit", "quit Nexus"),
    )

    DEFAULT_CSS = """
    HelpModal .help-dialog {
        width: 80;
        max-width: 94%;
        height: 84%;
    }
    HelpModal #help-scroll {
        height: 1fr;
        scrollbar-size: 1 1;
    }
    HelpModal #help-body {
        height: auto;
        width: 1fr;
        padding: 0;
    }
    HelpModal #help-close {
        width: 100%;
        height: 1;
        border: none;
        margin-top: 1;
        background: $panel;
        color: $text-muted;
        content-align: center middle;
    }
    HelpModal #help-close:hover {
        background: $primary;
        color: $background;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-dialog help-dialog"):
            with VerticalScroll(id="help-scroll"):
                yield Static(self._reference(), id="help-body")
            yield Button("close", id="help-close")

    def _reference(self) -> str:
        rows = [f"{'KEYBOARD':<36}│{'COMMANDS':<32}"]
        rows.append(f"{'─' * 36}┼{'─' * 32}")
        for index in range(max(len(self.SHORTCUTS), len(self.COMMANDS))):
            left = ""
            if index < len(self.SHORTCUTS):
                key, desc = self.SHORTCUTS[index]
                left = f"{key:<14}{_clip(desc, 21)}"
            right = ""
            if index < len(self.COMMANDS):
                command, desc = self.COMMANDS[index]
                right = f"{command:<14}{_clip(desc, 17)}"
            rows.append(f"{left:<36}│{right:<32}")
        panel = render_panel(
            rows,
            title="[4:help] NEXUS KEYMAP & SLASH COMMANDS",
            footer="Esc/q/Enter close",
            width=74,
        )
        return escape(panel)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "help-close":
            event.stop()
            self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)
