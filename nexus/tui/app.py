"""Textual application shell for the Nexus TUI.

This is the Step 4 wiring layer: it owns the session list, the composer, the
inspector sidebar and the background execution of specialist routes. Widgets
stay dumb (they own markup only) and modals return plain values, so everything
here is about state transitions:

* a prompt is submitted -> history is recorded -> a worker thread runs
  `front_desk_decide` + `execute_route`
* every stream/event/completion hop back to the UI happens through
  `_ui(...)` -> `call_from_thread`, because Textual is not thread-safe
* the finished turn is folded into a `TuiSession` and saved atomically

The dry-run path is a first-class citizen: with `dry_run=True` the route is
decided by keywords (never an LLM call) and no backend binary is spawned.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Optional

from rich.markup import escape
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import Hit, Hits, Provider
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Static, TextArea

from nexus.backends import ExecutionResult, execute_route
from nexus.drops import save_text_artifact
from nexus.entity import (
    DEFAULT_PROFILE_NAME,
    EntityProfile,
    build_handoff_context,
    build_synthesis_prompt,
    compute_performance_stats,
    list_profiles,
    load_profile,
    render_stats_table,
    resolve_profile_routes,
)
from nexus.events import (
    DEFAULT_EVENTS_PATH,
    emit_event,
    new_span_id,
    new_trace_id,
    read_events,
)
from nexus.router import (
    OrchestratorDecision,
    budget_allows,
    front_desk_decide,
    load_budgets,
    load_routes,
    validate_routes,
)
from nexus.sessions import (
    DEFAULT_SESSIONS_PATH,
    DEFAULT_TUI_CONFIG_PATH,
    TranscriptItem,
    TuiSession,
    load_sessions,
    load_tui_config,
    new_session,
    save_sessions,
    save_tui_config,
)
from nexus.sim import load_sim, render_sim_summary, resolve_sim_path
from nexus.square import (
    DEFAULT_BOARD_PATH,
    DEFAULT_SQUARE_PATH,
    post_square,
    render_square_view,
    write_board,
)
from nexus.tui.modals import (
    RELEASE_KEY,
    HelpModal,
    RoutePickerModal,
    SessionPickerModal,
    ThemePickerModal,
)
from nexus.tui.themes import (
    DEFAULT_THEME,
    NEXUS_THEMES,
    THEME_ALIASES,
    THEME_ORDER,
    normalize_tier,
    quota_markup,
    resolve_theme_name,
)
from nexus.tui.widgets import (
    AssistantMessageCard,
    ComposerBox,
    ComposerTextArea,
    CouncilGridCard,
    HintFooter,
    HomeHero,
    InspectorSidebar,
    RoutingPill,
    SessionTabBar,
    SuggestionItem,
    SuggestionPopup,
    ThinkingDots,
    ToolActivityGroup,
    UserMessageCard,
)
from nexus.visualizer import (
    render_delegation_tree,
    render_panel,
    render_routes_table,
)

DEFAULT_HISTORY_PATH = Path.home() / ".local/state/nexus/history"

# Slash commands shown by the composer palette, the help modal and /help.
# (command, args hint, description)
SLASH_COMMANDS: list[tuple[str, str, str]] = [
    ("/profile", "[name]", "List or hot-swap configuration profile"),
    ("/pin", "<route>", "Pin one specialist route (entity override)"),
    ("/unpin", "", "Return to unified entity auto-routing"),
    ("/mode", "[entity|sticky]", "Switch between unified entity and sticky routing"),
    ("/stats", "[profile|route|model]", "Show performance telemetry summary"),
    ("/switch", "<route>", "Lock a specialist route in charge"),
    ("/release", "", "Return control to the front desk"),
    ("/front", "", "Alias for /release"),
    ("/back", "", "Alias for /release"),
    ("/council", "<prompt>", "Run Pioneer, Red-Teamer and Synthesizer in parallel"),
    ("/routes", "", "Render the specialist route table"),
    ("/validate", "", "Audit every pinned model against `opencode models`"),
    ("/tree", "[trace]", "Render the delegation tree"),
    ("/square", "", "Show the Town Square board and feed"),
    ("/post", "<msg>", "Post a note to the Town Square"),
    ("/board", "<txt>", "Update the pinned BOARD.md (or 'clear')"),
    ("/new", "", "Open a fresh session"),
    ("/sessions", "", "Switch, create or delete a session"),
    ("/theme", "[name]", "Switch the TUI theme"),
    ("/ui", "[webui|cli]", "Switch between Gemini WebUI and CLI modes (Ctrl+G)"),
    ("/sidebar", "", "Toggle the inspector sidebar"),
    ("/verbose", "[0-2]", "Set detail and thinking visibility"),
    ("/quiet", "", "Status lines and tree off (verbosity 0)"),
    ("/fast", "", "Use fast keyword routing instead of the LLM"),
    ("/llm", "", "Use the free LLM front-desk decision model"),
    ("/think", "[off|low|medium|high|default]", "Set thinking effort override"),
    ("/attach", "<path|clear>", "Attach a file or image to the next prompt"),
    ("/copy", "", "Copy the last assistant response"),
    ("/save", "[filename]", "Save last code block to drops folder"),
    ("/clear", "", "Clear the current transcript"),
    ("/tui", "", "Launch the external harness TUI (opencode/cli)"),
    ("/help", "", "Show keyboard shortcuts and slash commands"),
    ("/quit", "", "Quit Nexus"),
    ("/exit", "", "Quit Nexus"),
]

_LLM_VERBOSITY_LABELS = {0: "quiet", 1: "default", 2: "verbose"}


def _clip(text: str, limit: int) -> str:
    """Collapse whitespace and clip to `limit` characters, marking the cut."""
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: max(1, limit - 1)] + "…"


def _premium_spans(session: Optional[TuiSession]) -> int:
    """Return the active session's premium-span counter."""
    return int(session.premium_spans or 0) if session is not None else 0


def _at_word(prefix: str) -> tuple[Optional[str], int]:
    """Return the `@token` ending at the cursor and its start column."""
    index = len(prefix)
    while index > 0 and not prefix[index - 1].isspace():
        index -= 1
    word = prefix[index:]
    if word.startswith("@"):
        return word, index
    return None, index


class NexusCommandProvider(Provider):
    """Command-palette entries for every Nexus action and slash command."""

    @property
    def nexus(self) -> "NexusApp":
        return self.app  # type: ignore[return-value]

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        app = self.nexus
        entries: list[tuple[str, str, Callable[[], Any]]] = [
            ("New session", "Ctrl+N · open a fresh conversation tab", app.action_new_session_tab),
            ("Switch / manage sessions", "Ctrl+L · list, create or delete sessions", app.action_pick_session),
            ("Clear current transcript", "/clear · empty the active session view", app.action_clear_transcript),
            ("Switch specialist route", "Ctrl+K · pick a route or return to the front desk", app.action_pick_route),
            ("Release to Front-Desk", "/release · hand control back to the auto router", app.action_release),
            ("Show / switch profile", "/profile [name] · hot-swap routes without losing memory", app.prefill_profile),
            ("Pin specialist route", "/pin <route> · stay on one route while keeping entity context", app.prefill_pin),
            ("Unpin specialist route", "/unpin · return to unified entity routing", app.action_release),
            (
                "Toggle entity vs sticky routing",
                f"/mode · current: {'entity' if app.entity_mode else 'sticky'}",
                app.action_toggle_entity_mode,
            ),
            ("Show performance telemetry", "/stats · latency, success rate, and zero-cost ratio", app.action_show_stats),
            ("Validate route pins", "/validate · audit pinned models against `opencode models`", app.action_validate_routes),
            ("Run 3-seat Tripartite Council", "/council <prompt> · Pioneer, Red-Teamer, Synthesizer", app.prefill_council),
            (
                "Toggle LLM front-desk vs fast keyword routing",
                f"current: {'LLM decision' if app.use_llm_decision else 'fast keywords'}",
                app.action_toggle_llm,
            ),
            ("Set thinking effort", "/think [off|low|medium|high|default]", app.prefill_think),
            ("Attach file to next prompt", "/attach <path|clear>", app.prefill_attach),
            ("Toggle inspector sidebar", "Ctrl+B · show or hide the right pane", app.action_toggle_sidebar),
            ("Sidebar: Tree view", "delegation spans for the active trace", app.show_tree_tab),
            ("Sidebar: Routes table", "specialist table with model pins", app.show_routes_tab),
            ("Sidebar: Town Square", "pinned board + recent agent posts", app.show_square_tab),
            ("Expand/collapse all tool groups", "Ctrl+O · fold or unfold tool activity", app.action_toggle_tools),
            ("Cycle verbosity", "0 quiet · 1 default · 2 verbose", app.action_cycle_verbosity),
            ("Switch theme", "Ctrl+T · live-preview every bundled theme", app.action_pick_theme),
            ("Toggle UI mode (WebUI / CLI)", f"Ctrl+G · current: {app.ui_mode}", app.action_toggle_ui_mode),
            ("Copy last assistant response", "/copy · clipboard via OSC 52 or wl-copy", app.action_copy_last),
            ("Save last code block to drop folder", "/save [filename] · writes to drops folder", app.prefill_save),
            ("Launch external harness TUI", "/tui · open external CLI TUI for this session", app.action_launch_tui),
            ("Show shortcuts & commands help", "F1 or ? · keyboard and slash reference", app.action_show_help),
            ("Quit Nexus", "Ctrl+Q · exit the full-screen console", app.action_quit_app),
        ]
        for label, help_text, callback in entries:
            match = matcher.match(label)
            if match:
                yield Hit(
                    match.score,
                    matcher.highlight(label),
                    callback,
                    help=help_text,
                )


class NexusApp(App[None]):
    """Full-screen Nexus console: sessions, delegation and the Town Square."""

    TITLE = "Nexus"
    CSS_PATH = "styles.tcss"
    COMMANDS = {NexusCommandProvider}

    BINDINGS = [
        Binding("ctrl+p", "command_palette", "Commands", priority=True),
        Binding("ctrl+k", "pick_route", "Route", priority=True),
        Binding("ctrl+b", "toggle_sidebar", "Sidebar", priority=True),
        Binding("ctrl+n", "new_session_tab", "New session", priority=True),
        Binding("ctrl+l", "pick_session", "Sessions", priority=True),
        Binding("ctrl+t", "pick_theme", "Theme", priority=True),
        Binding("ctrl+g", "toggle_ui_mode", "UI Mode", show=False),
        Binding("ctrl+o", "toggle_tools", "Tools", priority=True),
        Binding("ctrl+tab", "next_tab", "Next tab", show=False, priority=True),
        Binding("alt+right", "next_tab", "Next tab", show=False, priority=True),
        Binding("ctrl+shift+tab", "prev_tab", "Prev tab", show=False, priority=True),
        Binding("alt+left", "prev_tab", "Prev tab", show=False, priority=True),
        Binding("f1", "show_help", "Help", priority=True),
        Binding("question_mark", "show_help", "Help", show=False),
        Binding("ctrl+q", "quit_app", "Quit", priority=True),
        Binding("ctrl+c", "smart_ctrl_c", "Cancel", show=False, priority=True),
        Binding("escape", "handle_escape", "Dismiss", show=False),
    ]

    def __init__(
        self,
        *,
        events_path: Path = DEFAULT_EVENTS_PATH,
        sessions_path: Path = DEFAULT_SESSIONS_PATH,
        config_path: Path = DEFAULT_TUI_CONFIG_PATH,
        square_path: Path = DEFAULT_SQUARE_PATH,
        board_path: Path = DEFAULT_BOARD_PATH,
        routes_path: Optional[Path] = None,
        dry_run: bool = False,
        verbosity: Optional[int] = None,
        initial_theme: Optional[str] = None,
        show_sidebar: Optional[bool] = None,
        history_path: Optional[Path] = None,
        profile: Optional[str] = None,
        entity_mode: Optional[bool] = None,
        ui_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.events_path = Path(events_path)
        self.sessions_path = Path(sessions_path)
        self.config_path = Path(config_path)
        self.square_path = Path(square_path)
        self.board_path = Path(board_path)
        self.routes_path = Path(routes_path) if routes_path is not None else None
        self.history_path = Path(history_path) if history_path is not None else DEFAULT_HISTORY_PATH
        self.dry_run = bool(dry_run)

        for theme in NEXUS_THEMES.values():
            self.register_theme(theme)

        self.config = load_tui_config(self.config_path)
        self._base_routes_data = load_routes(self.routes_path)
        self._profile_cache: dict[str, EntityProfile] = {}

        requested_profile = (profile or "").strip() or str(self.config.get("profile") or DEFAULT_PROFILE_NAME)
        try:
            self._profile_obj = load_profile(requested_profile)
        except (OSError, ValueError):
            if profile is not None:
                raise
            self._profile_obj = load_profile(DEFAULT_PROFILE_NAME)
        self._profile_cache[self._profile_obj.name] = self._profile_obj
        self.routes_data = resolve_profile_routes(self._profile_obj, self._base_routes_data)

        raw_verbosity = self.config.get("verbosity", 1) if verbosity is None else verbosity
        self.verbosity = max(0, min(2, int(raw_verbosity)))
        self.use_llm_decision = bool(self.config.get("use_llm_decision", True))
        self.entity_mode = (
            bool(entity_mode)
            if entity_mode is not None
            else bool(self.config.get("entity_mode", True))
        )

        self.sessions: list[TuiSession] = load_sessions(self.sessions_path) or [
            new_session()
        ]
        if profile is not None:
            self.sessions[0].profile = self._profile_obj.name
        if entity_mode is not None:
            self.sessions[0].entity_mode = self.entity_mode
        elif self.sessions and not getattr(self.sessions[0], "profile", ""):
            self.sessions[0].profile = self._profile_obj.name
        self.active_id = self.sessions[0].id
        self.history: list[str] = self._read_history()
        self._route_sessions: dict[tuple[str, str], str] = {}

        configured_theme = str(self.config.get("theme") or DEFAULT_THEME)
        self._initial_theme = resolve_theme_name(initial_theme or configured_theme)

        if show_sidebar is None:
            show_sidebar = bool(self.config.get("sidebar_open", True))
        self._sidebar_open = bool(show_sidebar)
        self._sidebar_manual_narrow = False
        raw_ui = str(ui_mode if ui_mode is not None else self.config.get("ui_mode", "webui")).strip().lower()
        self._ui_mode = "cli" if raw_ui in ("cli", "console") else "webui"

        self._busy: set[str] = set()
        self._cancel_events: dict[str, threading.Event] = {}
        self._last_assistant_text = ""
        self._think_override: Optional[str | bool] = None
        self._attached_file: Optional[Path] = None
        self._ctrl_c_at = 0.0
        self._suppress_suggestions = False
        self._available_models: Optional[set[str]] = None
        # Monotonic guard so a slow transcript rebuild never mounts twice.
        self._render_generation = 0
        # Coalesces sidebar tree refreshes during a streaming turn.
        self._sidebar_tree_timer: Any = None
        # Last hint string pushed into HintFooter (skip identical re-renders).
        self._last_hints: str = ""
        # Nexus has no hover tooltips: skip the 0.5 s tooltip timer churn that
        # every mouse pixel would otherwise restart.
        self._disable_tooltips = True
        # Live "routing" spinners owned by `_begin_submission`, keyed by session.
        self._routing_indicators: dict[str, ThinkingDots] = {}

    # -- layout ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield SessionTabBar(id="tab-bar")
        with Horizontal(id="workspace"):
            with Vertical(id="main-pane"):
                yield VerticalScroll(id="transcript")
                yield SuggestionPopup(id="suggestion-popup")
                yield ComposerBox(id="composer-container")
            yield InspectorSidebar(
                events_path=self.events_path,
                square_path=self.square_path,
                board_path=self.board_path,
                routes_path=self.routes_path,
                id="sidebar",
                classes="" if self._sidebar_open else "hidden",
            )
        yield HintFooter(id="hint-footer")

    def on_mount(self) -> None:
        self.theme = self._initial_theme
        if self._profile_obj.replace_routes or self._profile_obj.routes:
            try:
                self.query_one("#sidebar", InspectorSidebar).set_routes_data(self.routes_data)
            except NoMatches:
                pass
        self._apply_sidebar(self._sidebar_open)
        self._apply_ui_mode(self._ui_mode, persist=False, swap_default_theme=False)
        self._render_active_session()
        self._sync_state()
        self._focus_composer()

    def on_unmount(self) -> None:
        if self._sidebar_tree_timer is not None:
            try:
                self._sidebar_tree_timer.stop()
            except Exception:  # pragma: no cover - timer already gone
                pass
            self._sidebar_tree_timer = None
        for session_id in list(self._routing_indicators):
            self._clear_routing_indicator(session_id)

    def _set_mouse_over(self, widget: Any, hover_widget: Any) -> None:
        """Skip Textual's hover restyle pass when the hovered widget has not changed."""
        if widget is self.mouse_over and hover_widget is self.hover_over:
            return
        super()._set_mouse_over(widget, hover_widget)

    # -- state helpers -----------------------------------------------------

    @property
    def active_session(self) -> Optional[TuiSession]:
        return self._session_by_id(self.active_id)

    def _session_by_id(self, session_id: str) -> Optional[TuiSession]:
        for session in self.sessions:
            if session.id == session_id:
                return session
        return None

    def _read_history(self) -> list[str]:
        try:
            raw = self.history_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return [line.rstrip("\n") for line in raw.splitlines() if line.strip()][-500:]

    def _append_history(self, text: str) -> None:
        entry = " ".join(text.split())
        if not entry:
            return
        self.history.append(entry)
        if len(self.history) > 500:
            self.history = self.history[-500:]
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_path, "a", encoding="utf-8") as handle:
                handle.write(entry + "\n")
        except OSError:
            pass
        try:
            self.query_one(ComposerBox).set_history(self.history)
        except NoMatches:
            pass

    def _cwd_label(self) -> str:
        cwd = str(Path.cwd())
        home = str(Path.home())
        if cwd == home:
            return "~"
        if cwd.startswith(home + os.sep):
            return "~" + cwd[len(home) :]
        return cwd

    def _active_profile_for(self, session: Optional[TuiSession]) -> EntityProfile:
        name = (getattr(session, "profile", "") or "").strip() if session else ""
        if not name or name == self._profile_obj.name:
            return self._profile_obj
        cached = self._profile_cache.get(name)
        if cached is not None:
            return cached
        try:
            prof = load_profile(name)
        except (OSError, ValueError):
            return self._profile_obj
        self._profile_cache[name] = prof
        return prof

    def _is_entity_session(self, session: Optional[TuiSession], prof: Optional[EntityProfile] = None) -> bool:
        active_prof = prof or self._active_profile_for(session)
        enabled = bool(getattr(session, "entity_mode", self.entity_mode)) if session else self.entity_mode
        return enabled and active_prof.mode != "single"

    def _tier_for(self, route: Optional[str]) -> str:
        if not route:
            return "FREE"
        cfg = (self.routes_data.get("routes") or {}).get(route) or {}
        return str(cfg.get("quota_tier", "FREE"))

    def _sync_state(self) -> None:
        """Push app state into the tab bar and composer meta strip."""
        session = self.active_session
        prof = self._active_profile_for(session)
        is_entity = self._is_entity_session(session, prof)
        pinned = getattr(session, "pinned_route", None) if session else None
        last_route = getattr(session, "last_route", None) if session else None
        route = pinned or (session.active_route if session and not is_entity else None)
        tier = self._tier_for(route or last_route)
        tokens_in = session.total_tokens_in if session else 0
        tokens_out = session.total_tokens_out if session else 0
        premium_spans = _premium_spans(session)
        try:
            self.query_one(SessionTabBar).update_state(
                self.sessions,
                self.active_id,
                self._busy,
                route,
                session.active_model if session else "",
                tier,
                self._sidebar_open,
                tokens_in,
                tokens_out,
                premium_spans,
                entity_mode=is_entity,
                profile_name=prof.name,
                pinned_route=pinned,
                last_route=last_route,
            )
        except NoMatches:
            pass
        try:
            self.query_one(ComposerBox).set_route_state(
                route or "",
                session.active_model if session else "",
                tier,
                self.use_llm_decision,
                self._cwd_label(),
                think_label=str(self._think_override or ""),
                attach_name=self._attached_file.name if self._attached_file else "",
                entity_mode=is_entity,
                profile_name=prof.name,
                pinned_route=pinned,
                last_route=last_route,
            )
        except NoMatches:
            pass
        self._update_hints()

    def _update_hints(self) -> None:
        if self._ui_mode == "webui" and not self._busy:
            hints = HintFooter.WEBUI_HINTS
        else:
            hints = (
                "[dim]Enter[/] send · [dim]⇧Enter[/] newline · [dim]Tab[/] accept · "
                "[dim]^P[/] cmd · [dim]^K[/] route · [dim]^B[/] sidebar · "
                "[dim]^O[/] thinking/tools · [dim]F1[/] help · [dim]^Q[/] quit"
            )
            if self.verbosity != 1:
                label = _LLM_VERBOSITY_LABELS.get(self.verbosity, "default")
                hints += f"  [dim]· verbosity {self.verbosity} ({label})[/]"
        if hints == self._last_hints:
            return
        self._last_hints = hints
        try:
            self.query_one(HintFooter).set_hints(hints)
        except NoMatches:
            pass

    def _save_sessions(self) -> None:
        try:
            save_sessions(self.sessions, self.sessions_path)
        except OSError:
            pass

    def _save_config(self) -> None:
        try:
            save_tui_config(self.config, self.config_path)
        except OSError:
            pass

    def _persist_toggle_defaults(self) -> None:
        self.config["theme"] = self.theme
        self.config["sidebar_open"] = self._sidebar_open
        self.config["use_llm_decision"] = self.use_llm_decision
        self.config["verbosity"] = self.verbosity
        self._save_config()

    # -- transcript --------------------------------------------------------

    def _transcript(self) -> Optional[VerticalScroll]:
        try:
            return self.query_one("#transcript", VerticalScroll)
        except NoMatches:
            return None

    def _mount(self, *widgets: Any) -> None:
        transcript = self._transcript()
        if transcript is None:
            return
        transcript.mount(*widgets)
        self._scroll_transcript()

    def _mount_note(self, title: str, body: str) -> None:
        note = Static(f"[bold]{escape(title)}[/]\n{escape(body)}", classes="system-note")
        self._mount(note)

    def _scroll_transcript(self) -> None:
        transcript = self._transcript()
        if transcript is None:
            return
        try:
            self.call_after_refresh(transcript.scroll_end, animate=False)
        except (NoMatches, RuntimeError):
            pass

    def _render_active_session(self) -> None:
        """Rebuild #transcript from the active session's stored items.

        `remove_children()` prunes asynchronously, so the mount has to wait one
        loop turn; otherwise a fixed-id widget such as `HomeHero` collides with
        the copy that is still being removed. A generation counter drops stale
        rebuilds when sessions are switched quickly.
        """
        transcript = self._transcript()
        if transcript is None:
            return
        session = self.active_session
        # An empty session already showing its hero needs no teardown/rebuild:
        # removing and remounting HomeHero flashes on every `+` / `×` / switch.
        if (
            (session is None or not session.items)
            and len(transcript.children) == 1
            and isinstance(transcript.children[0], HomeHero)
        ):
            self._render_generation += 1
            return
        self._render_generation += 1
        generation = self._render_generation
        transcript.remove_children()
        self.call_next(self._populate_transcript, generation)

    def _populate_transcript(self, generation: int) -> None:
        if generation != self._render_generation:
            return
        transcript = self._transcript()
        if transcript is None:
            return
        session = self.active_session
        if session is None or not session.items:
            transcript.mount(HomeHero(ui_mode=self._ui_mode))
            return
        widgets = []
        for item in session.items:
            widget = self._widget_for_item(item)
            if widget is not None:
                widgets.append(widget)
        if widgets:
            transcript.mount(*widgets)
        self.call_after_refresh(self._sync_council_layout)
        self._scroll_transcript()

    def _widget_for_item(self, item: TranscriptItem) -> Optional[Any]:
        kind = (item.kind or "assistant").lower()
        if kind == "user":
            return UserMessageCard(
                item.content,
                route=item.route,
                quota_tier=item.quota_tier,
                timestamp=item.ts,
            )
        if kind == "routing":
            pill = RoutingPill()
            pill.set_decision(item.route, item.model, item.content, item.quota_tier)
            return pill
        if kind == "tools":
            group = ToolActivityGroup(
                collapsed=(self._ui_mode == "webui") or not (
                    self.verbosity >= 1 and bool(item.thinking or item.tools)
                ),
                verbosity=self.verbosity,
            )
            for tool in item.tools:
                group.add_tool(tool)
            if item.thinking:
                group.append_thinking(item.thinking)
            group.finalize(item.status, item.elapsed_ms)
            return group
        if kind == "assistant":
            card = AssistantMessageCard(item.content)
            card.finalize(
                status=item.status,
                route=item.route,
                model=item.model,
                quota_tier=item.quota_tier,
                elapsed_ms=item.elapsed_ms,
                tokens_in=item.tokens_in,
                tokens_out=item.tokens_out,
                tools_count=len(item.tools),
            )
            return card
        if kind == "council":
            card = CouncilGridCard()
            for seat in item.council_seats:
                name = str(seat.get("seat", ""))
                result = self._execution_result_from(seat)
                if result is not None:
                    card.finalize_seat(name, result)
            return card
        if item.content:
            return Static(escape(item.content), classes="system-note")
        return None

    @staticmethod
    def _execution_result_from(payload: dict) -> Optional[ExecutionResult]:
        """Rebuild an `ExecutionResult` from a stored council-seat payload."""
        fields = {
            "trace_id": str(payload.get("trace_id", "")),
            "span_id": str(payload.get("span_id", "")),
            "route": str(payload.get("route", "")),
            "backend": str(payload.get("backend", "")),
            "model": str(payload.get("model", "")),
            "quota_tier": str(payload.get("quota_tier", "")),
            "status": str(payload.get("status", "DONE")),
            "output": str(payload.get("output", "")),
            "tokens_in": int(payload.get("tokens_in", 0) or 0),
            "tokens_out": int(payload.get("tokens_out", 0) or 0),
            "elapsed_ms": int(payload.get("elapsed_ms", 0) or 0),
            "error": str(payload.get("error", "")),
            "session_id": str(payload.get("session_id", "")),
            "tokens_cached": int(payload.get("tokens_cached", 0) or 0),
        }
        try:
            return ExecutionResult(**fields)
        except TypeError:  # pragma: no cover - defensive against schema drift
            return None

    def _sync_council_layout(self) -> None:
        stacked = (self.size.width or 0) < 110
        try:
            for card in self.query(CouncilGridCard):
                card.set_stacked(stacked)
        except NoMatches:
            return

    def _focus_composer(self) -> None:
        try:
            self.query_one(ComposerBox).focus_input()
        except NoMatches:
            return

    # -- sidebar -----------------------------------------------------------

    def _apply_sidebar(self, open_: bool, *, manual: bool = False) -> None:
        self._sidebar_open = bool(open_)
        try:
            self.query_one("#sidebar", InspectorSidebar).set_open(self._sidebar_open)
        except NoMatches:
            pass
        if manual:
            # A manual toggle at the current width wins over the auto-hide rule.
            self._sidebar_manual_narrow = (self.size.width or 0) < 100
            self.config["sidebar_open"] = self._sidebar_open
            self._save_config()
        self._sync_state()

    def _refresh_sidebar(self, *, tree: bool = False, routes: bool = False, square: bool = False) -> None:
        try:
            sidebar = self.query_one("#sidebar", InspectorSidebar)
        except NoMatches:
            return
        if tree:
            sidebar.refresh_tree()
        if routes:
            sidebar.refresh_routes()
        if square:
            sidebar.refresh_square()

    def _schedule_sidebar_tree_refresh(self) -> None:
        if not self._sidebar_open or not self.is_running:
            return
        if self._sidebar_tree_timer is not None:
            return
        self._sidebar_tree_timer = self.set_timer(0.25, self._on_sidebar_tree_timer)

    def _on_sidebar_tree_timer(self) -> None:
        self._sidebar_tree_timer = None
        self._refresh_sidebar(tree=True)

    # -- actions: bindings -------------------------------------------------

    @property
    def ui_mode(self) -> str:
        return self._ui_mode

    def _apply_ui_mode(
        self,
        mode: str,
        *,
        persist: bool = True,
        swap_default_theme: bool = False,
    ) -> None:
        target = "cli" if str(mode).strip().lower() in ("cli", "console") else "webui"
        self._ui_mode = target
        try:
            self.screen.set_class(target == "webui", "ui-webui")
        except Exception:
            pass
        if swap_default_theme:
            if target == "webui" and self.theme == "nexus-charcoal":
                self.theme = "gemini-dark"
                self.config["theme"] = "gemini-dark"
            elif target == "cli" and self.theme == "gemini-dark":
                self.theme = "nexus-charcoal"
                self.config["theme"] = "nexus-charcoal"
        if target == "cli":
            self._apply_sidebar(self._sidebar_open)
        try:
            self.query_one("#home-hero", HomeHero).set_ui_mode(target)
        except NoMatches:
            pass
        self._update_hints()
        if persist:
            self.config["ui_mode"] = target
            self._save_config()

    def action_toggle_ui_mode(self) -> None:
        next_mode = "cli" if self._ui_mode == "webui" else "webui"
        self._apply_ui_mode(next_mode, persist=True, swap_default_theme=True)
        self.notify(f"UI mode → {self._ui_mode} (Ctrl+G)", timeout=2)

    def action_toggle_sidebar(self) -> None:
        if self._ui_mode == "webui":
            self.notify("Sidebar is available in CLI mode (Ctrl+G)", timeout=2)
            return
        self._apply_sidebar(not self._sidebar_open, manual=True)

    def action_new_session_tab(self) -> None:
        session = new_session()
        session.profile = self._profile_obj.name
        session.entity_mode = self.entity_mode
        self.sessions.append(session)
        self.active_id = session.id
        self._save_sessions()
        self._render_active_session()
        self._sync_state()
        self._focus_composer()
        self.notify("New session", timeout=2)

    def action_pick_session(self) -> None:
        self.push_screen(
            SessionPickerModal(self.sessions, self.active_id),
            self._session_modal_result,
        )

    def _session_modal_result(self, result: Optional[tuple[str, str]]) -> None:
        if not result:
            return
        action, session_id = result
        if action == "switch":
            self.switch_session(session_id)
        elif action == "new":
            self.action_new_session_tab()
        elif action == "delete":
            self._delete_session(session_id)

    def action_pick_route(self) -> None:
        session = self.active_session
        current_route = (
            getattr(session, "pinned_route", None) or session.active_route
            if session
            else None
        )
        self.push_screen(
            RoutePickerModal(
                self.routes_data,
                active_route=current_route,
                available_models=self._available_models,
            ),
            self._route_modal_result,
        )

    def _route_modal_result(self, result: Optional[str]) -> None:
        if not result:
            return
        if result == RELEASE_KEY:
            self.action_release()
            return
        self._set_route(result)

    def action_pick_theme(self) -> None:
        self.push_screen(ThemePickerModal(self.theme), self._theme_modal_result)

    def _theme_modal_result(self, result: Optional[str]) -> None:
        if not result:
            return
        resolved = resolve_theme_name(result)
        self.config["theme"] = resolved
        self._save_config()
        self.notify(f"Theme → {resolved}", timeout=2)

    def action_show_help(self) -> None:
        self.push_screen(HelpModal())

    def action_quit_app(self) -> None:
        self.exit()

    def action_clear_transcript(self) -> None:
        session = self.active_session
        if session is None:
            return
        session.items.clear()
        session.updated_at = time.time()
        self._clear_routing_indicator(session.id)
        self._save_sessions()
        self._render_active_session()
        self.notify("Transcript cleared", timeout=2)

    def action_release(self) -> None:
        session = self.active_session
        if session is None:
            return
        released = getattr(session, "pinned_route", None) or session.active_route
        session.pinned_route = None
        session.active_route = None
        session.active_model = ""
        session.active_backend = ""
        session.backend_session_id = None
        self._save_sessions()
        self._sync_state()
        if released:
            self.notify(f"Unpinned @{released} — unified entity routing active", timeout=3)
        else:
            self.notify("Unified entity routing already active", timeout=2)

    def _set_route(self, route_key: str) -> None:
        route_key = (route_key or "").strip().lstrip("@")
        routes = self.routes_data.get("routes", {})
        if route_key not in routes:
            self.notify(f"Unknown route '@{route_key}'", severity="error", timeout=3)
            return
        session = self.active_session
        if session is None:
            return
        cfg = dict(routes.get(route_key) or {})
        session.pinned_route = route_key
        session.active_route = route_key
        session.active_model = str(cfg.get("model", ""))
        session.active_backend = str(cfg.get("backend", "opencode"))
        session.backend_session_id = None
        self._save_sessions()
        self._sync_state()
        self.notify(f"Pinned @{route_key} · {cfg.get('model', '')}", timeout=3)

    def _set_profile(self, argument: str) -> None:
        arg = argument.strip()
        if not arg:
            profiles = list_profiles()
            width = max(68, min(104, (self.size.width or 104) - 8))
            active_name = self._active_profile_for(self.active_session).name
            rows = [
                f"{'* ' if prof.name == active_name else '  '}{prof.name:<14} [{prof.mode:<8}] {prof.description}"
                for prof in profiles
            ]
            panel = render_panel(
                rows or ["(no profiles found)"],
                title="NEXUS PROFILES",
                footer="/profile <name> hot-swaps without clearing memory",
                width=width,
            )
            self._mount_note("CONFIGURATION PROFILES", panel)
            return
        try:
            prof = load_profile(arg)
        except (OSError, ValueError) as exc:
            self.notify(f"Profile error: {exc}", severity="error", timeout=4)
            return
        self._profile_obj = prof
        self._profile_cache[prof.name] = prof
        self.routes_data = resolve_profile_routes(prof, self._base_routes_data)
        self.config["profile"] = prof.name
        self._save_config()
        session = self.active_session
        if session is not None:
            session.profile = prof.name
            session.entity_mode = prof.mode != "single"
            if session.entity_mode and not getattr(session, "pinned_route", None):
                session.active_route = None
            self._save_sessions()
        try:
            sidebar_data = self.routes_data if (prof.replace_routes or prof.routes) else None
            self.query_one("#sidebar", InspectorSidebar).set_routes_data(sidebar_data)
        except NoMatches:
            pass
        self._sync_state()
        self.notify(f"Profile → {prof.name} ({prof.mode}) · context preserved", timeout=3)

    def _set_mode(self, argument: str) -> None:
        val = argument.strip().lower()
        session = self.active_session
        if not val:
            current = "entity" if self._is_entity_session(session) else "sticky"
            self.notify(f"Routing mode → {current} (usage: /mode [entity|sticky])", timeout=3)
            return
        if val not in ("entity", "sticky"):
            self.notify("Usage: /mode [entity|sticky]", severity="warning", timeout=3)
            return
        entity = val == "entity"
        self.entity_mode = entity
        self.config["entity_mode"] = entity
        self._save_config()
        if session is not None:
            session.entity_mode = entity
            if entity and not getattr(session, "pinned_route", None):
                session.active_route = None
            self._save_sessions()
        self._sync_state()
        self.notify(f"Routing mode → {'unified entity' if entity else 'sticky orchestrator'}", timeout=3)

    def action_toggle_entity_mode(self) -> None:
        self._set_mode("sticky" if self.entity_mode else "entity")

    def action_show_stats(self) -> None:
        self._show_stats("profile")

    def _show_stats(self, argument: str = "") -> None:
        group_by = argument.strip().lower() or "profile"
        if group_by not in ("profile", "route", "model"):
            self.notify("Usage: /stats [profile|route|model]", severity="warning", timeout=3)
            return
        events = read_events(path=self.events_path, limit=1000)
        stats = compute_performance_stats(events, group_by=group_by)
        width = max(72, min(108, (self.size.width or 108) - 8))
        table = render_stats_table(stats, group_by=group_by, color=False, width=width)
        try:
            from nexus.evolve import detect_improvement_opportunities

            opps = detect_improvement_opportunities(events, routes_data=self.routes_data)
            actionable = [o for o in opps if o.get("priority") in ("high", "medium")]
            if actionable:
                table += f"\nIMPROVE | {len(actionable)} opportunity(ies) — run 'nexus evolve suggest'"
        except Exception:
            pass
        self._mount_note(f"PERFORMANCE STATS ({group_by.upper()})", table)

    def show_tree_tab(self) -> None:
        self.show_sidebar_tab("tab-tree")

    def show_routes_tab(self) -> None:
        self.show_sidebar_tab("tab-routes")

    def show_square_tab(self) -> None:
        self.show_sidebar_tab("tab-square")

    def show_sidebar_tab(self, tab_id: str) -> None:
        self._apply_sidebar(True, manual=True)
        try:
            self.query_one("#sidebar", InspectorSidebar).select_tab(tab_id)
        except NoMatches:
            return

    def switch_session(self, session_id: str) -> None:
        if self._session_by_id(session_id) is None:
            return
        self.active_id = session_id
        self._render_active_session()
        self._sync_state()
        try:
            self.query_one(ComposerBox).set_running(session_id in self._busy)
        except NoMatches:
            pass
        self._focus_composer()

    def _delete_session(self, session_id: str) -> None:
        remaining = [session for session in self.sessions if session.id != session_id]
        if not remaining:
            remaining = [new_session()]
        self.sessions = remaining
        if self.active_id == session_id:
            self.active_id = remaining[0].id
        self._clear_routing_indicator(session_id)
        self._busy.discard(session_id)
        self._cancel_events.pop(session_id, None)
        self._save_sessions()
        self._render_active_session()
        self._sync_state()
        try:
            self.query_one(ComposerBox).set_running(self.active_id in self._busy)
        except NoMatches:
            pass

    def action_next_tab(self) -> None:
        self._cycle_session(1)

    def action_prev_tab(self) -> None:
        self._cycle_session(-1)

    def _cycle_session(self, delta: int) -> None:
        if len(self.sessions) < 2:
            return
        ids = [session.id for session in self.sessions]
        try:
            index = ids.index(self.active_id)
        except ValueError:
            index = 0
        self.switch_session(ids[(index + delta) % len(ids)])

    def action_toggle_tools(self) -> None:
        groups = list(self.query(ToolActivityGroup))
        if not groups:
            self.notify("No tool activity in this session", timeout=2)
            return
        expand = any(group.collapsed for group in groups)
        for group in groups:
            group.set_expanded(expand)
        self.notify(f"{'Expanded' if expand else 'Collapsed'} {len(groups)} tool group(s)", timeout=2)

    def action_cycle_verbosity(self) -> None:
        self._set_verbosity((self.verbosity + 1) % 3)

    def action_toggle_llm(self) -> None:
        self._set_llm(not self.use_llm_decision)

    def _set_llm(self, use_llm: bool) -> None:
        self.use_llm_decision = bool(use_llm)
        self.config["use_llm_decision"] = self.use_llm_decision
        self._save_config()
        self._sync_state()
        mode = "LLM front-desk decision" if self.use_llm_decision else "fast keyword routing"
        self.notify(f"Routing → {mode}", timeout=3)

    def _set_think(self, argument: str) -> None:
        value = argument.strip().lower()
        if not value:
            current = str(self._think_override) if self._think_override is not None else "default"
            self.notify(f"Thinking effort → {current}", timeout=3)
            return
        if value in ("default", "reset", "auto"):
            override: Optional[str | bool] = None
        elif value in ("off", "none"):
            override = "off"
        elif value in ("low", "medium", "high"):
            override = value
        else:
            self.notify(
                "Usage: /think [off|low|medium|high|default]",
                severity="warning",
                timeout=3,
            )
            return
        self._think_override = override
        self._sync_state()
        self.notify(f"Thinking effort → {override or 'default'}", timeout=3)

    def _set_attach(self, argument: str) -> None:
        value = argument.strip()
        if not value:
            if self._attached_file is not None:
                self.notify(f"Attachment → {self._attached_file}", timeout=3)
            else:
                self.notify("Usage: /attach <path|clear>", severity="warning", timeout=3)
            return
        if value.lower() in ("clear", "none"):
            self._attached_file = None
            self._sync_state()
            self.notify("Attachment cleared", timeout=2)
            return
        path = Path(value).expanduser()
        if not path.is_file():
            self.notify(f"Attachment not found: {path}", severity="error", timeout=4)
            return
        self._attached_file = path
        self._sync_state()
        self.notify(f"Attached {path.name} to the next prompt", timeout=3)

    def _set_verbosity(self, level: int) -> None:
        self.verbosity = max(0, min(2, int(level)))
        self.config["verbosity"] = self.verbosity
        self._save_config()
        transcript = self._transcript()
        if transcript is not None:
            for group in transcript.query(ToolActivityGroup):
                group.set_verbosity(self.verbosity)
        self._update_hints()
        label = _LLM_VERBOSITY_LABELS.get(self.verbosity, "default")
        thinking_label = {
            0: "thinking collapsed",
            1: "thinking visible",
            2: "full thinking & rationale",
        }[self.verbosity]
        self.notify(
            f"Verbosity → {self.verbosity} {label} · {thinking_label}",
            timeout=2,
        )

    def action_validate_routes(self) -> None:
        self.notify("Validating route pins…", timeout=2)
        self._validate_routes_worker()

    @work(thread=True, exclusive=True, group="validate-routes")
    def _validate_routes_worker(self) -> None:
        try:
            ok, errors = validate_routes(self.routes_data, refresh_cache=True)
        except Exception as exc:  # pragma: no cover - defensive
            ok, errors = False, [str(exc)]
        self._ui(self._routes_validated, ok, errors)

    def _routes_validated(self, ok: bool, errors: list[str]) -> None:
        count = len(self.routes_data.get("routes", {}))
        if ok:
            body = f"ROUTES_AUDIT | PASS ({count}/{count} specialist pins verified)"
            self.notify("Route pins validated", timeout=3)
        else:
            body = "ROUTES_AUDIT | FAIL\n" + "\n".join(f"✗ {error}" for error in errors)
            self.notify(
                f"{len(errors)} route pin problem(s)",
                severity="warning",
                timeout=4,
            )
        self._mount_note("ROUTE VALIDATION", body)
        self._refresh_sidebar(routes=True)

    def _last_session_text(self) -> str:
        session = self.active_session
        if session is not None:
            for item in reversed(session.items):
                if item.kind == "assistant" and item.content.strip():
                    return item.content
                if item.kind == "council":
                    council_text = self._council_text(item)
                    if council_text.strip():
                        return council_text
        return ""

    def action_copy_last(self) -> None:
        self.copy_text(self._last_session_text() or self._last_assistant_text)

    def _save_last(self, argument: str = "") -> None:
        text = self._last_session_text() or self._last_assistant_text
        if not text.strip():
            self.notify("Nothing to save yet", severity="warning", timeout=2)
            return
        try:
            saved = save_text_artifact(text, argument)
        except (ValueError, OSError) as exc:
            self.notify(f"Save failed: {exc}", severity="error", timeout=4)
            return
        exec_note = ", +x" if saved["executable"] else ""
        self.notify(
            f"Saved {saved['relative']} ({saved['bytes']} B{exec_note}) → {saved['path']}",
            timeout=4,
        )

    def action_launch_tui(self) -> None:
        session = self.active_session
        if session is not None and session.active_backend == "agy":
            argv = ["agy"]
        elif session is not None and session.backend_session_id:
            argv = ["opencode", "--session", session.backend_session_id]
        else:
            argv = ["opencode"]
        if shutil.which(argv[0]) is None:
            self.notify(f"`{argv[0]}` not found on PATH", severity="error", timeout=4)
            return
        try:
            with self.suspend():
                subprocess.run(argv, check=False)
        except Exception as exc:  # pragma: no cover - terminal/driver dependent
            self.notify(f"Could not launch {argv[0]}: {exc}", severity="error", timeout=4)
            return
        self.refresh()
        self._focus_composer()

    # -- actions: copy / ctrl-c / escape -----------------------------------

    def copy_text(self, text: str) -> None:
        if not text or not text.strip():
            self.notify("Nothing to copy yet", severity="warning", timeout=2)
            return
        copied = False
        try:
            self.copy_to_clipboard(text)
            copied = True
        except Exception:  # pragma: no cover - driver dependent
            copied = False
        if not copied and shutil.which("wl-copy"):
            try:
                subprocess.run(
                    ["wl-copy"],
                    input=text,
                    text=True,
                    timeout=5,
                    check=False,
                )
                copied = True
            except Exception:  # pragma: no cover - no compositor
                copied = False
        self.notify("Copied response to clipboard", timeout=2)

    def action_copy_from_card(self, text: str) -> None:
        self.copy_text(text)

    def action_smart_ctrl_c(self) -> None:
        session = self.active_session
        if session is not None and session.id in self._busy:
            self._cancel_active(session.id)
            return
        try:
            composer = self.query_one(ComposerBox)
        except NoMatches:
            composer = None
        if composer is not None and composer.input.text:
            composer.input.load_text("")
            composer.set_popup_open(False)
            return
        now = time.monotonic()
        if now - self._ctrl_c_at <= 2.0:
            self.exit()
            return
        self._ctrl_c_at = now
        self.notify("Press Ctrl+C again to quit", timeout=3)

    def action_handle_escape(self) -> None:
        try:
            popup = self.query_one(SuggestionPopup)
        except NoMatches:
            popup = None
        if popup is not None and popup.is_visible:
            self._dismiss_popup()
            return
        session = self.active_session
        if session is not None and session.id in self._busy:
            self._cancel_active(session.id)

    def _cancel_active(self, session_id: str) -> None:
        cancel_event = self._cancel_events.get(session_id)
        if cancel_event is not None:
            cancel_event.set()
            self.notify("Cancelling…", timeout=2)

    def _dismiss_popup(self) -> None:
        try:
            self.query_one(SuggestionPopup).hide()
        except NoMatches:
            pass
        try:
            self.query_one(ComposerBox).set_popup_open(False)
        except NoMatches:
            pass

    # -- composer suggestions ----------------------------------------------

    def prefill_composer(self, text: str) -> None:
        try:
            composer = self.query_one(ComposerBox)
        except NoMatches:
            return
        composer.input.load_text(text)
        try:
            composer.input.move_cursor(composer.input.document.end)
        except Exception:  # pragma: no cover - document edge cases
            pass
        composer.focus_input()
        self._update_suggestions()

    def prefill_profile(self) -> None:
        self.prefill_composer("/profile ")

    def prefill_pin(self) -> None:
        self.prefill_composer("/pin ")

    def prefill_council(self) -> None:
        self.prefill_composer("/council ")

    def prefill_think(self) -> None:
        self.prefill_composer("/think ")

    def prefill_attach(self) -> None:
        self.prefill_composer("/attach ")

    def prefill_save(self) -> None:
        self.prefill_composer("/save ")

    def _current_line(self) -> tuple[str, int, int]:
        try:
            area = self.query_one(ComposerBox).input
        except NoMatches:
            return "", 0, 0
        row, col = area.cursor_location
        document = area.document
        line = document.get_line(row) if 0 <= row < document.line_count else ""
        return line, row, col

    def _update_suggestions(self) -> None:
        try:
            composer = self.query_one(ComposerBox)
            popup = self.query_one(SuggestionPopup)
        except NoMatches:
            return

        if self._suppress_suggestions:
            self._suppress_suggestions = False
            if popup.is_visible or composer.input.popup_open:
                popup.set_items([])
                popup.hide()
                composer.set_popup_open(False)
            return

        area = composer.input
        row, col = area.cursor_location
        document = area.document
        line = document.get_line(row) if 0 <= row < document.line_count else ""
        prefix = line[:col]
        stripped = line.lstrip()

        # Fast path: nothing typed can open a palette, so skip all evaluation.
        if (
            not popup.is_visible
            and not composer.input.popup_open
            and not stripped.startswith("/")
            and "@" not in prefix
        ):
            return

        items: list[SuggestionItem] = []
        arg_builders: tuple[tuple[str, Callable[[str], list[SuggestionItem]], bool], ...] = (
            ("/switch", lambda q: self._route_suggestions(q, mode="switch"), True),
            ("/pin", lambda q: self._route_suggestions(q, mode="switch"), True),
            ("/profile", self._profile_suggestions, False),
            ("/theme", self._theme_suggestions, False),
        )
        matched_arg_cmd = False
        for cmd_prefix, builder, strip_at in arg_builders:
            if stripped.startswith(cmd_prefix) and (
                stripped == cmd_prefix or stripped[len(cmd_prefix)].isspace()
            ):
                matched_arg_cmd = True
                arg = stripped[len(cmd_prefix) :].strip()
                items = builder(arg)
                needle = arg.lstrip("@").lower() if strip_at else arg.lower()
                if needle and any(
                    needle == item.insert_text.strip().lower() for item in items
                ):
                    items = []
                break
        if not matched_arg_cmd:
            if stripped.startswith("/") and not any(char.isspace() for char in stripped):
                items = self._command_suggestions(stripped[1:])
                if any(stripped.lower() == item.value.lower() for item in items):
                    items = []
            else:
                token, _start = _at_word(prefix)
                if token is not None:
                    items = self._route_suggestions(token, mode="route")

        if items:
            popup.set_items(items)
            composer.set_popup_open(True)
        elif popup.is_visible or composer.input.popup_open:
            popup.set_items([])
            popup.hide()
            composer.set_popup_open(False)

    def _command_suggestions(self, query: str) -> list[SuggestionItem]:
        needle = query.strip().lower()
        items: list[SuggestionItem] = []
        for command, args, description in SLASH_COMMANDS:
            label = f"{command} {args}".strip()
            if needle and needle not in command.lstrip("/").lower() and needle not in description.lower():
                continue
            items.append(
                SuggestionItem(
                    value=command,
                    label=label,
                    description=description,
                    insert_text=command + (" " if args else ""),
                )
            )
        return items

    def _route_suggestions(self, query: str, *, mode: str = "route") -> list[SuggestionItem]:
        needle = query.strip().lstrip("@").lower()
        items: list[SuggestionItem] = []
        for key, cfg in (self.routes_data.get("routes") or {}).items():
            if needle and needle not in key.lower() and needle not in str(cfg.get("model", "")).lower():
                continue
            tier = str(cfg.get("quota_tier", "FREE"))
            items.append(
                SuggestionItem(
                    value=f"@{key}",
                    label=f"@{key}",
                    description=_clip(str(cfg.get("description", "")), 40),
                    badge=quota_markup(tier),
                    insert_text=key if mode == "switch" else f"@{key} ",
                )
            )
        return items

    def _profile_suggestions(self, query: str) -> list[SuggestionItem]:
        needle = query.strip().lower()
        items: list[SuggestionItem] = []
        for prof in list_profiles():
            if needle and needle not in prof.name.lower() and needle not in prof.description.lower():
                continue
            items.append(
                SuggestionItem(
                    value=prof.name,
                    label=prof.name,
                    description=_clip(f"[{prof.mode}] {prof.description}", 46),
                    insert_text=prof.name,
                )
            )
        return items

    def _theme_suggestions(self, query: str) -> list[SuggestionItem]:
        needle = query.strip().lower()
        items: list[SuggestionItem] = []
        for name in THEME_ORDER:
            if needle and needle not in name.lower():
                continue
            items.append(
                SuggestionItem(
                    value=name,
                    label=name,
                    description="live-preview theme",
                    insert_text=name,
                )
            )
        return items

    def on_suggestion_chosen(self, event: SuggestionPopup.SuggestionChosen) -> None:
        event.stop()
        self._apply_suggestion(event.item)

    def on_popup_navigate_requested(self, event: ComposerTextArea.PopupNavigateRequested) -> None:
        event.stop()
        try:
            self.query_one(SuggestionPopup).move(event.delta)
        except NoMatches:
            return

    def on_popup_accept_requested(self, event: ComposerTextArea.PopupAcceptRequested) -> None:
        event.stop()
        try:
            item = self.query_one(SuggestionPopup).selected_item()
        except NoMatches:
            item = None
        if item is not None:
            self._apply_suggestion(item)

    def on_popup_dismiss_requested(self, event: ComposerTextArea.PopupDismissRequested) -> None:
        event.stop()
        self._dismiss_popup()

    def on_cancel_requested(self, event: ComposerTextArea.CancelRequested) -> None:
        event.stop()
        self.action_handle_escape()

    def _apply_suggestion(self, item: SuggestionItem) -> None:
        try:
            composer = self.query_one(ComposerBox)
        except NoMatches:
            return
        area = composer.input
        line, row, col = self._current_line()
        stripped = line.lstrip()
        insert = item.text_to_insert
        end_col = col

        self._suppress_suggestions = True
        if stripped.startswith("/switch") and (
            stripped == "/switch" or stripped[len("/switch")].isspace()
        ):
            head = line[: len(line) - len(stripped)]
            replacement = f"{head}/switch {insert.strip()}"
            area.replace(replacement, (row, 0), (row, len(line)))
            end_col = len(replacement)
        elif stripped.startswith("/pin") and (
            stripped == "/pin" or stripped[len("/pin")].isspace()
        ):
            head = line[: len(line) - len(stripped)]
            replacement = f"{head}/pin {insert.strip()}"
            area.replace(replacement, (row, 0), (row, len(line)))
            end_col = len(replacement)
        elif stripped.startswith("/profile") and (
            stripped == "/profile" or stripped[len("/profile")].isspace()
        ):
            head = line[: len(line) - len(stripped)]
            replacement = f"{head}/profile {insert.strip()}"
            area.replace(replacement, (row, 0), (row, len(line)))
            end_col = len(replacement)
        elif stripped.startswith("/theme") and (
            stripped == "/theme" or stripped[len("/theme")].isspace()
        ):
            head = line[: len(line) - len(stripped)]
            replacement = f"{head}/theme {insert.strip()}"
            area.replace(replacement, (row, 0), (row, len(line)))
            end_col = len(replacement)
        elif stripped.startswith("/"):
            replacement = insert.strip()
            area.replace(replacement, (row, 0), (row, len(line)))
            end_col = len(replacement)
        else:
            token, start = _at_word(line[:col])
            start_col = start if token is not None else col
            area.replace(insert, (row, start_col), (row, col))
            end_col = start_col + len(insert)

        self._dismiss_popup()
        try:
            area.move_cursor((row, max(0, end_col)))
        except Exception:  # pragma: no cover - document edge cases
            pass
        composer.focus_input()

    # -- submission --------------------------------------------------------

    def on_composer_submitted(self, event: ComposerTextArea.ComposerSubmitted) -> None:
        event.stop()
        text = event.text
        if not text.strip():
            return
        self._append_history(text)
        try:
            composer = self.query_one(ComposerBox)
            composer.input.load_text("")
            composer.set_popup_open(False)
        except NoMatches:
            pass
        self._dismiss_popup()
        self._submit_text(text)

    def _begin_submission(
        self,
        session_id: str,
        text: str,
        *,
        force_council: bool = False,
    ) -> Optional[UserMessageCard]:
        """Mount the user card and flip the running state before routing.

        This gives instant feedback: `front_desk_decide` can take seconds, so the
        prompt must appear (and the composer timer must start) immediately.
        """
        session = self._session_by_id(session_id)
        if session is None:
            return None

        self._clear_routing_indicator(session_id)

        if force_council:
            route = "council"
            tier = "ORCHESTRATOR"
        else:
            stripped = text.lstrip()
            pinned = getattr(session, "pinned_route", None)
            effective_route = pinned or session.active_route
            route = ""
            if stripped.startswith("@"):
                token = stripped.split(None, 1)[0].lstrip("@")
                if token in (self.routes_data.get("routes") or {}):
                    route = token
                    tier = self._tier_for(route)
                else:
                    route = effective_route or ""
                    tier = self._tier_for(route) if route else "FREE"
            else:
                route = effective_route or ""
                tier = self._tier_for(route) if route else "FREE"

        user_card = UserMessageCard(
            text,
            route=route,
            quota_tier=tier,
            timestamp=time.time(),
        )
        transcript = self._transcript()
        if transcript is not None:
            try:
                transcript.query_one(HomeHero).remove()
            except NoMatches:
                pass
            transcript.mount(user_card)
            indicator = ThinkingDots(
                label="council · routing" if force_council else "routing",
                classes="routing-thinking",
            )
            self._routing_indicators[session_id] = indicator
            transcript.mount(indicator)

        if session.title == "New session":
            title = _clip(text, 28)
            if title:
                session.title = title
        session.updated_at = time.time()
        self._busy.add(session_id)
        try:
            if force_council:
                label = "council"
            elif route:
                label = f"@{route}"
            elif self._is_entity_session(session):
                label = f"entity ({self._active_profile_for(session).name})…"
            else:
                label = "routing…"
            self.query_one(ComposerBox).set_running(True, label=label)
        except NoMatches:
            pass
        self._sync_state()
        self._scroll_transcript()
        return user_card

    def _submit_text(self, text: str) -> None:
        stripped = text.strip()
        if not stripped:
            return
        session = self.active_session
        if session is not None and session.id in self._busy:
            self.notify("Task already running — press Esc to cancel", severity="warning", timeout=3)
            return
        if stripped.startswith("/"):
            self._handle_slash(stripped)
            return
        if session is None:
            return
        attached = self._attached_file
        self._attached_file = None
        user_card = self._begin_submission(session.id, text, force_council=False)
        self._run_prompt_worker(session.id, text, False, user_card, attached)

    def _handle_slash(self, stripped: str) -> None:
        command, _, argument = stripped.partition(" ")
        argument = argument.strip()
        command = command.lower()

        if command in ("/quit", "/exit"):
            self.exit()
        elif command == "/help":
            self.action_show_help()
        elif command in ("/release", "/front", "/back", "/unpin"):
            self.action_release()
        elif command in ("/switch", "/pin"):
            if not argument:
                self.notify(f"Usage: {command} <route>", severity="warning", timeout=3)
                return
            self._set_route(argument)
        elif command == "/profile":
            self._set_profile(argument)
        elif command == "/mode":
            self._set_mode(argument)
        elif command == "/stats":
            self._show_stats(argument)
        elif command == "/sim":
            self._show_sim()
        elif command == "/council":
            if not argument:
                self.notify("Usage: /council <prompt>", severity="warning", timeout=3)
                return
            session = self.active_session
            if session is None:
                return
            if session.id in self._busy:
                self.notify("Task already running — press Esc to cancel", severity="warning", timeout=3)
                return
            attached = self._attached_file
            self._attached_file = None
            user_card = self._begin_submission(session.id, argument, force_council=True)
            self._run_prompt_worker(session.id, argument, True, user_card, attached)
        elif command == "/routes":
            self._show_routes()
        elif command == "/validate":
            self.action_validate_routes()
        elif command == "/tree":
            self._show_tree(argument or None)
        elif command == "/square":
            self._show_square()
        elif command == "/post":
            self._post_square(argument)
        elif command == "/board":
            self._update_board(argument)
        elif command == "/new":
            self.action_new_session_tab()
        elif command == "/sessions":
            self.action_pick_session()
        elif command == "/theme":
            if not argument:
                self.action_pick_theme()
            else:
                self._apply_theme(argument)
        elif command == "/ui":
            target = argument.strip().lower()
            if not target:
                self.action_toggle_ui_mode()
            elif target in ("webui", "web", "gemini"):
                self._apply_ui_mode("webui", persist=True, swap_default_theme=True)
                self.notify("UI mode → webui (Ctrl+G)", timeout=2)
            elif target in ("cli", "console"):
                self._apply_ui_mode("cli", persist=True, swap_default_theme=True)
                self.notify("UI mode → cli (Ctrl+G)", timeout=2)
            else:
                self.notify("Usage: /ui [webui|cli]", severity="warning", timeout=3)
        elif command == "/sidebar":
            self.action_toggle_sidebar()
        elif command == "/verbose":
            if argument in ("0", "1", "2"):
                self._set_verbosity(int(argument))
            else:
                self.action_cycle_verbosity()
        elif command == "/quiet":
            self._set_verbosity(0)
        elif command == "/think":
            self._set_think(argument)
        elif command == "/attach":
            self._set_attach(argument)
        elif command == "/copy":
            self.action_copy_last()
        elif command == "/save":
            self._save_last(argument)
        elif command == "/clear":
            self.action_clear_transcript()
        elif command == "/tui":
            self.action_launch_tui()
        elif command == "/fast":
            self._set_llm(False)
        elif command == "/llm":
            self._set_llm(True)
        else:
            self.notify(f"Unknown command '{command}' — try /help", severity="warning", timeout=3)

    def _apply_theme(self, name: str) -> None:
        key = (name or "").strip().lower()
        if key not in NEXUS_THEMES and key not in THEME_ALIASES:
            self.notify(f"Unknown theme '{name}'", severity="warning", timeout=3)
            return
        resolved = resolve_theme_name(key)
        self.theme = resolved
        self.config["theme"] = resolved
        self._save_config()
        self.notify(f"Theme → {resolved}", timeout=2)

    def _show_routes(self) -> None:
        width = max(72, min(110, (self.size.width or 110) - 8))
        text = render_routes_table(self.routes_data, color=False, width=width)
        self._mount_note("SPECIALIST ROUTES", text)

    def _show_tree(self, trace_id: Optional[str]) -> None:
        events = read_events(path=self.events_path, limit=400, trace_id=trace_id)
        width = max(64, min(110, (self.size.width or 110) - 8))
        text = render_delegation_tree(events, trace_id=trace_id, color=False, width=width)
        self._mount_note("DELEGATION TREE", text)

    def _show_square(self) -> None:
        width = max(64, min(110, (self.size.width or 110) - 8))
        text = render_square_view(
            limit=10,
            square_path=self.square_path,
            board_path=self.board_path,
            width=width,
            color=False,
        )
        self._mount_note("TOWN SQUARE", text)
        self.show_sidebar_tab("tab-square")

    def _show_sim(self) -> None:
        sim_path = resolve_sim_path()
        try:
            ledger = load_sim(sim_path, strict=True)
        except FileNotFoundError:
            self._mount_note(
                "SIM LEDGER",
                f"No SIM ledger at {sim_path}\nRun 'nexus sim init' to create it.",
            )
            return
        except ValueError as exc:
            self.notify(f"SIM error: {exc}", severity="error", timeout=4)
            return
        width = max(64, min(110, (self.size.width or 110) - 8))
        summary = render_sim_summary(ledger, color=False, width=width)
        try:
            from nexus.evolve import detect_improvement_opportunities

            opps = detect_improvement_opportunities(
                read_events(path=self.events_path, limit=500),
                ledger=ledger,
                routes_data=self.routes_data,
            )
            actionable = [o for o in opps if o.get("priority") in ("high", "medium")]
            if actionable:
                summary += f"\nIMPROVE | {len(actionable)} opportunity(ies) — run 'nexus evolve suggest'"
        except Exception:
            pass
        self._mount_note("SIM LEDGER", summary)

    def _post_square(self, body: str) -> None:
        if not body:
            self.notify("Usage: /post <message>", severity="warning", timeout=3)
            return
        session = self.active_session
        try:
            record = post_square(
                body,
                author="tui",
                model=session.active_model if session else "nexus",
                tag="note",
                square_path=self.square_path,
            )
        except (OSError, ValueError) as exc:
            self.notify(f"Could not post: {exc}", severity="error", timeout=4)
            return
        self.notify(f"Posted {record['id']} to the Town Square", timeout=3)
        self._refresh_sidebar(square=True)

    def _update_board(self, content: str) -> None:
        if not content:
            self.notify("Usage: /board <text> (or /board clear)", severity="warning", timeout=3)
            return
        try:
            write_board(
                "" if content.lower() == "clear" else content,
                author="tui",
                board_path=self.board_path,
            )
        except OSError as exc:
            self.notify(f"Could not update the board: {exc}", severity="error", timeout=4)
            return
        self.notify("BOARD.md updated", timeout=3)
        self._refresh_sidebar(square=True)

    # -- worker: execution -------------------------------------------------

    def _ui(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Run `callback` on the UI thread from a worker thread."""
        if not self.is_running:
            return
        try:
            self.call_from_thread(callback, *args, **kwargs)
        except RuntimeError:
            return

    @work(thread=True, exclusive=False)
    def _run_prompt_worker(
        self,
        session_id: str,
        text: str,
        force_council: bool = False,
        user_card: Optional[UserMessageCard] = None,
        attach_file: Optional[Path] = None,
    ) -> None:
        try:
            self._run_prompt(session_id, text, force_council, user_card, attach_file)
        except Exception as exc:  # pragma: no cover - last-resort guard
            self._ui(self._task_failed, session_id, f"{type(exc).__name__}: {exc}")

    def _run_prompt(
        self,
        session_id: str,
        text: str,
        force_council: bool,
        user_card: Optional[UserMessageCard] = None,
        attach_file: Optional[Path] = None,
    ) -> None:
        session = self._session_by_id(session_id)
        if session is None:
            self._ui(self._release_busy, session_id)
            return

        trace_id = new_trace_id()
        started = time.time()
        cancel_event = threading.Event()
        self._cancel_events[session_id] = cancel_event

        profile = self._active_profile_for(session)
        is_entity = self._is_entity_session(session, profile)
        pinned = getattr(session, "pinned_route", None)
        locked_route = pinned if is_entity else (pinned or session.active_route)
        has_at_prefix = text.lstrip().startswith("@")

        if (
            not force_council
            and is_entity
            and profile.mode == "ensemble"
            and not has_at_prefix
            and locked_route is None
        ):
            self._run_unified_ensemble(
                session_id,
                trace_id,
                text,
                started,
                cancel_event,
                profile=profile,
                user_card=user_card,
                attach_file=attach_file,
            )
            return

        explicit_route = None if has_at_prefix else locked_route
        use_llm = (
            locked_route is None
            and not has_at_prefix
            and self.use_llm_decision
            and not self.dry_run
        )
        try:
            decision = front_desk_decide(
                text,
                explicit_route=explicit_route,
                routes_data=self.routes_data,
                use_llm=use_llm,
                timeout_s=15.0,
                cancel_event=cancel_event,
            )
        except KeyError as exc:
            self._ui(self._task_failed, session_id, str(exc))
            return

        if cancel_event.is_set():
            self._ui(self._release_busy, session_id)
            return

        self._ui(self._task_routed, session_id, decision, force_council, user_card)
        items: list[TranscriptItem] = [
            TranscriptItem(
                kind="user",
                content=text,
                route="council" if (force_council or decision.orchestrator == "council") else decision.route_key,
                quota_tier=str(decision.route_cfg.get("quota_tier", "FREE")),
                ts=started,
            )
        ]
        items.append(
            TranscriptItem(
                kind="routing",
                content=decision.reason,
                route=decision.route_key,
                model=str(decision.route_cfg.get("model", "")),
                quota_tier=str(decision.route_cfg.get("quota_tier", "FREE")),
                ts=started,
            )
        )

        if force_council or decision.orchestrator == "council":
            self._run_council(
                session_id,
                trace_id,
                decision.cleaned_prompt or text,
                items,
                cancel_event,
                attach_file=attach_file,
                profile=profile,
            )
            return

        self._run_single(
            session_id,
            trace_id,
            decision,
            items,
            cancel_event,
            use_llm=use_llm,
            attach_file=attach_file,
            profile=profile,
            is_entity=is_entity,
            pinned=locked_route is not None,
        )

    @staticmethod
    def _conversation_turns(session: Optional[TuiSession]) -> list[dict]:
        """Return clean alternating conversation turns for context handoff."""
        if session is None:
            return []
        return [
            {"role": item.kind, "content": item.content}
            for item in session.items
            if item.kind in ("user", "assistant")
            and item.content.strip()
            and not item.content.strip().startswith("[error]")
        ]

    def _make_stream_handlers(
        self,
        tools: ToolActivityGroup,
        assistant: AssistantMessageCard,
    ) -> tuple[Callable[[str, str], None], Callable[[], None], list[str], list[str], dict[str, str]]:
        """Create debounced stream callbacks shared by single and ensemble turns."""
        tool_log: list[str] = []
        thinking_log: list[str] = []
        backend_session = {"id": ""}
        buf_lock = threading.Lock()
        text_buf: list[str] = []
        think_buf: list[str] = []
        state = {
            "text_len": 0,
            "think_len": 0,
            "think_total": 0,
            "last_text": time.monotonic(),
            "last_think": time.monotonic(),
        }

        def _flush_text() -> None:
            with buf_lock:
                if not text_buf:
                    return
                payload = "".join(text_buf)
                text_buf.clear()
                state["text_len"] = 0
                state["last_text"] = time.monotonic()
            self._ui(assistant.append_text, payload)

        def _update_thinking(payload: str, total: int) -> None:
            tools.append_thinking(payload)
            assistant.set_thinking_progress(total)

        def _flush_think() -> None:
            with buf_lock:
                if not think_buf:
                    return
                payload = "".join(think_buf)
                think_buf.clear()
                state["think_len"] = 0
                state["last_think"] = time.monotonic()
                total = int(state["think_total"])
            self._ui(_update_thinking, payload, total)

        def _flush_all() -> None:
            _flush_text()
            _flush_think()

        def _on_stream(kind: str, chunk: str) -> None:
            if kind == "text":
                now = time.monotonic()
                with buf_lock:
                    text_buf.append(chunk)
                    state["text_len"] += len(chunk)
                    due = state["text_len"] >= 256 or (now - state["last_text"]) >= 0.06
                if due:
                    _flush_text()
            elif kind == "thinking":
                thinking_log.append(chunk)
                now = time.monotonic()
                with buf_lock:
                    state["think_total"] += len(chunk)
                    think_buf.append(chunk)
                    state["think_len"] += len(chunk)
                    due = state["think_len"] >= 256 or (now - state["last_think"]) >= 0.08
                if due:
                    _flush_think()
            elif kind == "tool":
                _flush_all()
                tool_log.append(chunk)
                self._ui(tools.add_tool, chunk)
            elif kind == "session":
                backend_session["id"] = chunk
            elif kind == "status":
                self._ui(self.notify, chunk, severity="warning", timeout=4)

        return _on_stream, _flush_all, tool_log, thinking_log, backend_session

    def _run_single(
        self,
        session_id: str,
        trace_id: str,
        decision: Any,
        items: list[TranscriptItem],
        cancel_event: threading.Event,
        *,
        use_llm: bool = False,
        attach_file: Optional[Path] = None,
        profile: Optional[EntityProfile] = None,
        is_entity: bool = False,
        pinned: bool = False,
    ) -> None:
        profile_name = profile.name if profile is not None else ""
        root_span = new_span_id()
        root_details = {
            "parent_id": None,
            "route": "front-desk" if use_llm else "entrypoint",
            "intent": "decision",
            "backend": "opencode" if use_llm else "nexus",
            "model": decision.decision_model if use_llm else "nexus-router",
            "quota_tier": "FREE" if use_llm else "ORCHESTRATOR",
            "prompt": decision.cleaned_prompt[:200],
            "tokens_in": decision.tokens_in,
            "tokens_out": decision.tokens_out,
            "profile": profile_name,
        }
        emit_event(
            trace_id,
            root_span,
            **root_details,
            status="RUNNING",
            step=f"handoff->@{decision.route_key} ({decision.reason})",
            elapsed_ms=decision.elapsed_ms,
            path=self.events_path,
        )

        tools = ToolActivityGroup(
            collapsed=(self._ui_mode == "webui") or (self.verbosity == 0),
            verbosity=self.verbosity,
        )
        assistant = AssistantMessageCard()
        self._ui(self._mount_single, tools, assistant, session_id)

        _on_stream, _flush_all, tool_log, thinking_log, backend_session = self._make_stream_handlers(
            tools, assistant
        )

        def _on_event(_event: dict) -> None:
            self._ui(self._schedule_sidebar_tree_refresh)

        route_cfg = dict(decision.route_cfg)
        route_key = decision.route_key
        label = f"in-charge:{route_key}" if decision.take_charge else route_key
        session = self._session_by_id(session_id)
        backend = str(route_cfg.get("backend", "opencode"))
        prior_backend_session: Optional[str] = None
        if session is not None:
            if is_entity:
                prior_backend_session = self._route_sessions.get(session_id, {}).get(
                    (route_key, backend)
                )
                if (
                    not prior_backend_session
                    and (getattr(session, "last_route", None) == route_key or session.active_route == route_key)
                    and backend == session.active_backend
                ):
                    prior_backend_session = session.backend_session_id
            elif route_key == session.active_route and backend == session.active_backend:
                prior_backend_session = session.backend_session_id

        turns = self._conversation_turns(session)
        history_msgs: Optional[list[dict]] = None
        handoff_ctx = ""
        if backend in ("ollama", "openai-compat", "colibri"):
            history_msgs = turns[-12:] if turns else None
        elif is_entity and turns and (
            prior_backend_session is None or getattr(session, "last_route", None) != route_key
        ):
            handoff_ctx = build_handoff_context(
                turns,
                max_turns=profile.handoff_max_turns if profile is not None else 8,
                max_chars=profile.handoff_max_chars if profile is not None else 6000,
            )

        result = execute_route(
            trace_id,
            label,
            route_cfg,
            decision.cleaned_prompt,
            parent_id=root_span,
            attach_file=attach_file,
            session_id=prior_backend_session,
            events_path=self.events_path,
            square_path=self.square_path,
            board_path=self.board_path,
            inject_square=True,
            on_event=_on_event,
            dry_run=self.dry_run,
            emit_stderr=False,
            on_stream=_on_stream,
            cancel_event=cancel_event,
            history_messages=history_msgs,
            think=self._think_override,
            system_persona=profile.system_persona if (is_entity and profile is not None) else "",
            handoff_context=handoff_ctx,
            profile=profile_name,
        )
        emit_event(
            trace_id,
            root_span,
            **root_details,
            status="DONE" if result.status == "DONE" else "FAIL",
            step=f"spawned @{route_key} [{result.status}] — {decision.reason}",
            elapsed_ms=result.elapsed_ms + decision.elapsed_ms,
            path=self.events_path,
        )
        # A turn can end with a partial buffer; flush it before finalizing.
        _flush_all()

        items.append(
            TranscriptItem(
                kind="tools",
                content="",
                status=result.status,
                elapsed_ms=result.elapsed_ms,
                tools=list(tool_log),
                thinking="".join(thinking_log),
                ts=time.time(),
            )
        )
        items.append(
            TranscriptItem(
                kind="assistant",
                content=result.output,
                route=result.route,
                model=result.model,
                backend=result.backend,
                quota_tier=result.quota_tier,
                status=result.status,
                elapsed_ms=result.elapsed_ms,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                tools=list(tool_log),
                thinking="".join(thinking_log),
                trace_id=result.trace_id,
                ts=time.time(),
            )
        )
        self._ui(
            self._finish_task,
            session_id,
            items,
            [result],
            tools,
            assistant,
            None,
            (route_key, route_cfg),
            backend_session["id"] or result.session_id or prior_backend_session,
            entity_turn=(is_entity and not pinned),
        )

    def _run_council(
        self,
        session_id: str,
        trace_id: str,
        prompt: str,
        items: list[TranscriptItem],
        cancel_event: threading.Event,
        *,
        attach_file: Optional[Path] = None,
        profile: Optional[EntityProfile] = None,
    ) -> None:
        council_started = time.monotonic()
        profile_name = profile.name if profile is not None else ""
        card = CouncilGridCard()
        self._ui(self._mount_council, card, session_id)

        routes = self.routes_data.get("routes", {})
        roles = self.routes_data.get(
            "council_roles",
            {"pioneer": "fast-code", "red_teamer": "free-reason", "synthesizer": "refactor"},
        )
        seat_plan = [
            ("pioneer", str(roles.get("pioneer", "fast-code")),
             f"[Pioneer / Thesis — concise 3 bullets]: {prompt}"),
            ("red_teamer", str(roles.get("red_teamer", "free-reason")),
             f"[Red-Teamer / Antithesis — risks & failure modes in 3 bullets]: {prompt}"),
            ("synthesizer", str(roles.get("synthesizer", "refactor")),
             f"[Synthesizer / Pragmatic Verdict in 3 bullets]: {prompt}"),
        ]
        budgets = load_budgets(self.routes_data)
        max_children = max(0, int(budgets.get("max_children", 8)))
        if len(seat_plan) > max_children:
            seat_plan = seat_plan[:max_children]
        allowed, reason = budget_allows(
            budgets,
            spans=1 + len(seat_plan),
            children=len(seat_plan),
            depth=1,
        )
        if not allowed:
            self._ui(self._task_failed, session_id, f"Council refused: {reason}")
            return
        if not seat_plan:
            self._ui(self._task_failed, session_id, "Council refused: max_children is 0")
            return

        root_span = new_span_id()
        root_details = {
            "parent_id": None,
            "route": "council",
            "intent": "tripartite-council",
            "backend": "nexus",
            "model": "nexus-council",
            "quota_tier": "ORCHESTRATOR",
            "prompt": prompt[:200],
            "profile": profile_name,
        }
        emit_event(
            trace_id,
            root_span,
            **root_details,
            status="RUNNING",
            step=f"fanout->{len(seat_plan)} specialist models",
            path=self.events_path,
        )

        results: dict[str, ExecutionResult] = {}
        lock = threading.Lock()

        def _stream_for(seat: str) -> tuple[Callable[[str, str], None], Callable[[], None]]:
            state: dict[str, Any] = {
                "buf": [],
                "len": 0,
                "last": time.monotonic(),
                "think_chars": 0,
                "think_last": 0.0,
            }
            seat_lock = threading.Lock()

            def _flush_seat_text() -> None:
                with seat_lock:
                    if not state["buf"]:
                        return
                    payload = "".join(state["buf"])
                    state["buf"] = []
                    state["len"] = 0
                    state["last"] = time.monotonic()
                self._ui(card.update_seat, seat, payload)

            def _on_stream(kind: str, chunk: str) -> None:
                now = time.monotonic()
                if kind == "thinking":
                    with seat_lock:
                        state["think_chars"] += len(chunk)
                        total = state["think_chars"]
                        due = (now - state["think_last"]) >= 0.12
                        if due:
                            state["think_last"] = now
                    if due:
                        self._ui(card.update_seat_thinking, seat, total)
                    return
                if kind != "text":
                    return
                with seat_lock:
                    state["buf"].append(chunk)
                    state["len"] += len(chunk)
                    due = state["len"] >= 256 or (now - state["last"]) >= 0.08
                if due:
                    _flush_seat_text()

            return _on_stream, _flush_seat_text

        def _on_event(_event: dict) -> None:
            self._ui(self._schedule_sidebar_tree_refresh)

        def _run_seat(seat: str, role_key: str, seat_prompt: str) -> None:
            cfg = dict(routes.get(role_key) or routes.get("code") or {})
            cfg["intent"] = f"council:{seat}"
            on_stream, flush_stream = _stream_for(seat)
            result = execute_route(
                trace_id,
                f"{seat}:{role_key}",
                cfg,
                seat_prompt,
                parent_id=root_span,
                attach_file=attach_file,
                events_path=self.events_path,
                square_path=self.square_path,
                board_path=self.board_path,
                inject_square=True,
                dry_run=self.dry_run,
                emit_stderr=False,
                on_event=_on_event,
                on_stream=on_stream,
                cancel_event=cancel_event,
                think=self._think_override,
                profile=profile_name,
            )
            # Flush any tail text before the seat is finalized.
            flush_stream()
            with lock:
                results[seat] = result
            self._ui(card.finalize_seat, seat, result)
            self._ui(self._schedule_sidebar_tree_refresh)

        pool = ThreadPoolExecutor(max_workers=3)
        try:
            futures = [
                pool.submit(_run_seat, seat, role_key, seat_prompt)
                for seat, role_key, seat_prompt in seat_plan
            ]
            for future in futures:
                future.result()
        finally:
            pool.shutdown(wait=True)

        ordered = [results[seat] for seat, _key, _prompt in seat_plan if seat in results]
        seats_payload = [
            {"seat": seat, **asdict(results[seat])}
            for seat, _key, _prompt in seat_plan
            if seat in results
        ]
        ok = (
            len(ordered) == len(seat_plan)
            and all(result.status == "DONE" for result in ordered)
        )
        emit_event(
            trace_id,
            root_span,
            **root_details,
            status="DONE" if ok else "FAIL",
            step=(
                f"{len(ordered)}/{len(seat_plan)} seats complete"
                if ok
                else "partial failure"
            ),
            elapsed_ms=int((time.monotonic() - council_started) * 1000),
            path=self.events_path,
        )
        items.append(
            TranscriptItem(
                kind="council",
                content=prompt,
                route="council",
                model="nexus-council",
                backend="nexus",
                quota_tier="ORCHESTRATOR",
                status="DONE" if ok else "FAIL",
                elapsed_ms=max((result.elapsed_ms for result in ordered), default=0),
                tokens_in=sum(result.tokens_in for result in ordered),
                tokens_out=sum(result.tokens_out for result in ordered),
                trace_id=trace_id,
                council_seats=seats_payload,
                ts=time.time(),
            )
        )
        self._ui(
            self._finish_task,
            session_id,
            items,
            ordered,
            None,
            None,
            card,
            None,
            None,
        )

    def _run_unified_ensemble(
        self,
        session_id: str,
        trace_id: str,
        prompt: str,
        started: float,
        cancel_event: threading.Event,
        *,
        profile: EntityProfile,
        user_card: Optional[UserMessageCard] = None,
        attach_file: Optional[Path] = None,
    ) -> None:
        ensemble_started = time.monotonic()
        routes = self.routes_data.get("routes", {})
        roles = self.routes_data.get(
            "council_roles",
            {"pioneer": "fast-code", "red_teamer": "free-reason", "synthesizer": "refactor"},
        )
        synth_key = str(
            roles.get("synthesizer")
            or profile.default_route
            or self.routes_data.get("default_route", "code")
        )
        synth_cfg = dict(routes.get(synth_key) or routes.get("code") or {})

        seat_plan = [
            ("pioneer", str(roles.get("pioneer", "fast-code")),
             f"[Pioneer / Thesis — concise 3 bullets]: {prompt}"),
            ("red_teamer", str(roles.get("red_teamer", "free-reason")),
             f"[Red-Teamer / Antithesis — risks & edge cases in 3 bullets]: {prompt}"),
            ("architect", synth_key,
             f"[Architect / Trade-offs — concise recommendation in 3 bullets]: {prompt}"),
        ]
        budgets = load_budgets(self.routes_data)
        max_children = max(0, int(budgets.get("max_children", 8)))
        if max_children < 1:
            self._ui(self._task_failed, session_id, "Ensemble refused: max_children is 0")
            return
        # Reserve one child slot for the final synthesis call.
        seat_plan = seat_plan[: max(0, max_children - 1)]
        allowed, reason = budget_allows(
            budgets,
            spans=2 + len(seat_plan),
            children=len(seat_plan) + 1,
            depth=1,
        )
        if not allowed:
            self._ui(self._task_failed, session_id, f"Ensemble refused: {reason}")
            return

        synth_decision = OrchestratorDecision(
            route_key=synth_key,
            route_cfg=synth_cfg,
            reason=f"ensemble ({profile.name}) -> synth @{synth_key}",
            orchestrator="ensemble",
            cleaned_prompt=prompt,
            take_charge=False,
        )
        self._ui(self._task_routed, session_id, synth_decision, False, user_card)
        items: list[TranscriptItem] = [
            TranscriptItem(
                kind="user",
                content=prompt,
                route=synth_key,
                quota_tier=str(synth_cfg.get("quota_tier", "FREE")),
                ts=started,
            ),
            TranscriptItem(
                kind="routing",
                content=synth_decision.reason,
                route=synth_key,
                model=str(synth_cfg.get("model", "")),
                quota_tier=str(synth_cfg.get("quota_tier", "FREE")),
                ts=started,
            ),
        ]

        root_span = new_span_id()
        root_details = {
            "parent_id": None,
            "route": "ensemble",
            "intent": "unified-ensemble",
            "backend": "nexus",
            "model": f"ensemble:{synth_key}",
            "quota_tier": "ORCHESTRATOR",
            "prompt": prompt[:200],
            "profile": profile.name,
        }
        emit_event(
            trace_id,
            root_span,
            **root_details,
            status="RUNNING",
            step=f"ensemble->{len(seat_plan)} seats + synth @{synth_key}",
            path=self.events_path,
        )

        tools = ToolActivityGroup(
            collapsed=(self._ui_mode == "webui") or (self.verbosity == 0),
            verbosity=self.verbosity,
        )
        assistant = AssistantMessageCard()
        self._ui(self._mount_single, tools, assistant, session_id)
        _on_stream, _flush_all, tool_log, thinking_log, backend_session = self._make_stream_handlers(
            tools, assistant
        )

        def _on_event(_event: dict) -> None:
            self._ui(self._schedule_sidebar_tree_refresh)

        results: dict[str, ExecutionResult] = {}
        lock = threading.Lock()

        def _run_seat(seat: str, role_key: str, seat_prompt: str) -> None:
            cfg = dict(routes.get(role_key) or routes.get("code") or {})
            cfg["intent"] = f"ensemble:{seat}"
            res = execute_route(
                trace_id,
                f"{seat}:{role_key}",
                cfg,
                seat_prompt,
                parent_id=root_span,
                attach_file=attach_file,
                events_path=self.events_path,
                square_path=self.square_path,
                board_path=self.board_path,
                inject_square=True,
                dry_run=self.dry_run,
                emit_stderr=False,
                on_event=_on_event,
                cancel_event=cancel_event,
                think=self._think_override,
                profile=profile.name,
            )
            with lock:
                results[seat] = res
            note = f"seat:{seat} @{role_key} [{res.status}]"
            tool_log.append(note)
            self._ui(tools.add_tool, note)
            self._ui(self._schedule_sidebar_tree_refresh)

        if seat_plan:
            pool = ThreadPoolExecutor(max_workers=max(1, len(seat_plan)))
            try:
                futures = [
                    pool.submit(_run_seat, seat, role_key, seat_prompt)
                    for seat, role_key, seat_prompt in seat_plan
                ]
                for future in futures:
                    future.result()
            finally:
                pool.shutdown(wait=True)

        ordered = [results[seat] for seat, _key, _prompt in seat_plan if seat in results]
        seat_outputs = [
            (seat, results[seat].output.strip())
            for seat, _key, _prompt in seat_plan
            if seat in results and results[seat].output.strip()
        ]
        synth_prompt = build_synthesis_prompt(
            prompt,
            seat_outputs or [("direct", prompt)],
            system_persona=profile.system_persona,
        )
        session = self._session_by_id(session_id)
        turns = self._conversation_turns(session)
        synth_backend = str(synth_cfg.get("backend", "opencode"))
        prior_synth_session = self._route_sessions.get(session_id, {}).get(
            (synth_key, synth_backend)
        )
        if (
            not prior_synth_session
            and session is not None
            and getattr(session, "last_route", None) == synth_key
            and session.active_backend == synth_backend
        ):
            prior_synth_session = session.backend_session_id

        history_msgs: Optional[list[dict]] = None
        handoff_ctx = ""
        if synth_backend in ("ollama", "openai-compat", "colibri"):
            history_msgs = turns[-12:] if turns else None
        elif turns and (
            prior_synth_session is None or getattr(session, "last_route", None) != synth_key
        ):
            handoff_ctx = build_handoff_context(
                turns,
                max_turns=profile.handoff_max_turns,
                max_chars=profile.handoff_max_chars,
            )

        synth_res = execute_route(
            trace_id,
            f"synth:{synth_key}",
            synth_cfg,
            synth_prompt,
            parent_id=root_span,
            attach_file=attach_file,
            session_id=prior_synth_session,
            events_path=self.events_path,
            square_path=self.square_path,
            board_path=self.board_path,
            inject_square=True,
            dry_run=self.dry_run,
            emit_stderr=False,
            on_event=_on_event,
            on_stream=_on_stream,
            cancel_event=cancel_event,
            history_messages=history_msgs,
            think=self._think_override,
            handoff_context=handoff_ctx,
            profile=profile.name,
        )
        _flush_all()

        all_results = [synth_res, *ordered]
        emit_event(
            trace_id,
            root_span,
            **root_details,
            status="DONE" if synth_res.status == "DONE" else synth_res.status,
            step=f"ensemble {len(seat_outputs)}/{len(seat_plan)} seats -> synth [{synth_res.status}]",
            elapsed_ms=int((time.monotonic() - ensemble_started) * 1000),
            path=self.events_path,
        )
        items.append(
            TranscriptItem(
                kind="tools",
                content="",
                status=synth_res.status,
                elapsed_ms=synth_res.elapsed_ms,
                tools=list(tool_log),
                thinking="".join(thinking_log),
                ts=time.time(),
            )
        )
        items.append(
            TranscriptItem(
                kind="assistant",
                content=synth_res.output,
                route=synth_res.route,
                model=synth_res.model,
                backend=synth_res.backend,
                quota_tier=synth_res.quota_tier,
                status=synth_res.status,
                elapsed_ms=synth_res.elapsed_ms,
                tokens_in=sum(r.tokens_in for r in all_results),
                tokens_out=sum(r.tokens_out for r in all_results),
                tools=list(tool_log),
                thinking="".join(thinking_log),
                trace_id=trace_id,
                ts=time.time(),
            )
        )
        self._ui(
            self._finish_task,
            session_id,
            items,
            all_results,
            tools,
            assistant,
            None,
            (synth_key, synth_cfg),
            backend_session["id"] or synth_res.session_id or prior_synth_session,
            entity_turn=True,
        )

    # -- worker -> UI callbacks -------------------------------------------

    def _task_routed(
        self,
        session_id: str,
        decision: Any,
        force_council: bool,
        user_card: Optional[UserMessageCard],
    ) -> None:
        final_route = (
            "council"
            if (force_council or decision.orchestrator == "council")
            else decision.route_key
        )
        tier = str(decision.route_cfg.get("quota_tier", "FREE"))
        if user_card is not None and user_card.is_mounted:
            user_card.set_route(final_route, tier)
        self._clear_routing_indicator(session_id)
        if self.active_id != session_id:
            return
        try:
            label = "council" if final_route == "council" else f"@{decision.route_key}"
            self.query_one(ComposerBox).set_running(True, label=label)
        except NoMatches:
            pass
        pill = RoutingPill()
        pill.set_decision(
            final_route,
            str(decision.route_cfg.get("model", "")),
            decision.reason,
            tier,
        )
        self._mount(pill)

    def _mount_single(
        self,
        tools: ToolActivityGroup,
        assistant: AssistantMessageCard,
        session_id: Optional[str] = None,
    ) -> None:
        if session_id is not None and self.active_id != session_id:
            return
        self._mount(tools, assistant)

    def _mount_council(
        self,
        card: CouncilGridCard,
        session_id: Optional[str] = None,
    ) -> None:
        if session_id is not None and self.active_id != session_id:
            return
        self._mount(card)
        try:
            self.call_after_refresh(card.set_stacked, (self.size.width or 0) < 110)
        except RuntimeError:  # pragma: no cover - app shutting down
            pass

    def _clear_routing_indicator(self, session_id: str) -> None:
        indicator = self._routing_indicators.pop(session_id, None)
        if indicator is None:
            return
        indicator.stop()
        if indicator.is_mounted:
            indicator.remove()

    def _release_busy(self, session_id: str) -> None:
        self._clear_routing_indicator(session_id)
        self._busy.discard(session_id)
        self._cancel_events.pop(session_id, None)
        try:
            self.query_one(ComposerBox).set_running(session_id in self._busy or self.active_id in self._busy)
        except NoMatches:
            pass
        self._sync_state()

    def _task_failed(self, session_id: str, message: str) -> None:
        self._release_busy(session_id)
        session = self._session_by_id(session_id)
        if session is not None:
            session.updated_at = time.time()
            item = TranscriptItem(
                kind="assistant",
                content=f"\n[error] {message}",
                status="FAIL",
                quota_tier="FREE",
                ts=time.time(),
            )
            session.items.append(item)
            self._save_sessions()
        card = AssistantMessageCard(f"\n[error] {message}")
        card.finalize(
            status="FAIL",
            route="",
            model="nexus",
            quota_tier="FREE",
            elapsed_ms=0,
            tokens_in=0,
            tokens_out=0,
            tools_count=0,
            error=message,
        )
        self._mount(card)
        self.notify(_clip(message, 90), severity="error", timeout=5)
        self._sync_state()

    def _finish_task(
        self,
        session_id: str,
        items: list[TranscriptItem],
        results: list[ExecutionResult],
        tools_widget: Optional[ToolActivityGroup],
        assistant_widget: Optional[AssistantMessageCard],
        council_widget: Optional[CouncilGridCard],
        lock_route: Optional[tuple[str, dict]],
        backend_session_id: Optional[str],
        *,
        entity_turn: bool = False,
    ) -> None:
        ok = bool(results) and all(result.status == "DONE" for result in results)

        self._clear_routing_indicator(session_id)

        if tools_widget is not None and tools_widget.is_mounted:
            primary = results[0] if results else None
            tools_widget.finalize(
                primary.status if primary is not None else "FAIL",
                primary.elapsed_ms if primary is not None else 0,
            )
            tools_widget.collapsed = not (
                (not ok) or (self.verbosity >= 1 and tools_widget.has_activity)
            )

        if assistant_widget is not None and assistant_widget.is_mounted and results:
            primary = results[0]
            tools_item = next(
                (item for item in reversed(items) if item.kind == "tools"),
                None,
            )
            assistant_widget.finalize(
                status=primary.status,
                route=primary.route,
                model=primary.model,
                quota_tier=primary.quota_tier,
                elapsed_ms=primary.elapsed_ms,
                tokens_in=primary.tokens_in,
                tokens_out=primary.tokens_out,
                tools_count=len(tools_item.tools) if tools_item is not None else 0,
                error=primary.error,
                cached_tokens=primary.tokens_cached,
            )
            self._last_assistant_text = assistant_widget.raw_text

        if council_widget is not None:
            self._last_assistant_text = self._council_text(items[-1])

        session = self._session_by_id(session_id)
        premium_spans = 0
        if session is not None:
            session.items.extend(items)
            session.updated_at = time.time()
            if lock_route is not None:
                key, cfg = lock_route
                backend_name = str(cfg.get("backend", "opencode"))
                session.last_route = key
                session.active_model = (
                    results[0].model
                    if results and results[0].model
                    else str(cfg.get("model", ""))
                )
                session.active_backend = backend_name
                if backend_session_id:
                    self._route_sessions.setdefault(session_id, {})[
                        (key, backend_name)
                    ] = backend_session_id
                if entity_turn and not getattr(session, "pinned_route", None):
                    session.active_route = None
                    session.backend_session_id = backend_session_id or None
                else:
                    session.active_route = key
                    session.backend_session_id = backend_session_id or None
            elif backend_session_id:
                session.backend_session_id = backend_session_id
            session.total_tokens_in += sum(result.tokens_in for result in results)
            session.total_tokens_out += sum(result.tokens_out for result in results)
            premium_spans = sum(
                1 for result in results if normalize_tier(result.quota_tier) == "PREMIUM"
            )
            session.premium_spans = _premium_spans(session) + premium_spans

        self._busy.discard(session_id)
        self._cancel_events.pop(session_id, None)
        try:
            self.query_one(ComposerBox).set_running(self.active_id in self._busy, label="")
        except NoMatches:
            pass

        if self._sidebar_tree_timer is not None:
            try:
                self._sidebar_tree_timer.stop()
            except Exception:  # pragma: no cover - timer already gone
                pass
            self._sidebar_tree_timer = None

        if premium_spans:
            self.notify(
                "PREMIUM QUOTA USED — this turn used a premium route",
                severity="error",
                timeout=6,
            )
        failures = [result for result in results if result.status != "DONE"]
        if failures:
            first = failures[0]
            detail = _clip(first.error or first.status, 80)
            self.notify(f"{first.status}: {detail}", severity="error", timeout=6)

        if self._sidebar_open:
            self._refresh_sidebar(tree=True, square=True)
        self._save_sessions()
        self._sync_state()
        self._scroll_transcript()
        self._focus_composer()

    @staticmethod
    def _council_text(item: TranscriptItem) -> str:
        parts = []
        for seat in item.council_seats:
            body = str(seat.get("output", "")).strip()
            if body:
                parts.append(
                    f"### {str(seat.get('seat', 'seat')).replace('_', '-').upper()}\n{body}"
                )
        return "\n\n".join(parts)

    # -- messages ----------------------------------------------------------

    def on_tab_selected(self, event: SessionTabBar.TabSelected) -> None:
        event.stop()
        self.switch_session(event.session_id)

    def on_new_tab_requested(self, event: SessionTabBar.NewTabRequested) -> None:
        event.stop()
        self.action_new_session_tab()

    def on_tab_close_requested(self, event: SessionTabBar.TabCloseRequested) -> None:
        event.stop()
        self._delete_session(event.session_id)

    def on_quick_action_selected(self, event: HomeHero.QuickActionSelected) -> None:
        event.stop()
        command = event.command
        if command.startswith("/"):
            self._handle_slash(command)
        else:
            self.prefill_composer(f"{command} ")

    def on_route_picked(self, event: InspectorSidebar.RoutePicked) -> None:
        event.stop()
        self._set_route(event.route_key)

    def on_square_post_requested(self, event: InspectorSidebar.SquarePostRequested) -> None:
        event.stop()
        self._post_square(event.body)

    def on_refresh_requested(self, event: InspectorSidebar.RefreshRequested) -> None:
        event.stop()
        self._refresh_sidebar(tree=True, routes=True, square=True)

    def on_copy_requested(self, event: AssistantMessageCard.CopyRequested) -> None:
        event.stop()
        self.copy_text(event.text)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "composer-input":
            return
        event.stop()
        self._update_suggestions()

    def on_resize(self, event: events.Resize) -> None:
        width = event.size.width or 0
        if self._ui_mode != "webui" and width < 100 and self._sidebar_open and not self._sidebar_manual_narrow:
            self._apply_sidebar(False)
        try:
            for card in self.query(CouncilGridCard):
                card.set_stacked(width < 110)
        except NoMatches:
            return


__all__ = ["NexusApp", "NexusCommandProvider", "SLASH_COMMANDS"]
