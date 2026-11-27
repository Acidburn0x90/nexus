"""Textual UI package for the Nexus interactive TUI.

`run_tui` is the only entrypoint `bin/nexus` needs: it constructs `NexusApp`
with the caller's paths, runs it, and returns a clean exit code. Keeping the
import of `NexusApp` at module level lets `bin/nexus` fall back to the
line-oriented REPL if Textual is unavailable (`ImportError`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from nexus.events import DEFAULT_EVENTS_PATH
from nexus.sessions import DEFAULT_SESSIONS_PATH, DEFAULT_TUI_CONFIG_PATH
from nexus.square import DEFAULT_BOARD_PATH, DEFAULT_SQUARE_PATH
from nexus.tui.app import NexusApp

__all__ = ["NexusApp", "run_tui"]


def run_tui(
    *,
    events_path: Path = DEFAULT_EVENTS_PATH,
    sessions_path: Path = DEFAULT_SESSIONS_PATH,
    config_path: Path = DEFAULT_TUI_CONFIG_PATH,
    square_path: Path = DEFAULT_SQUARE_PATH,
    board_path: Path = DEFAULT_BOARD_PATH,
    routes_path: Optional[Path] = None,
    dry_run: bool = False,
    verbosity: Optional[int] = None,
    theme: Optional[str] = None,
    show_sidebar: Optional[bool] = None,
    profile: Optional[str] = None,
    entity_mode: Optional[bool] = None,
) -> int:
    """Run the full-screen Nexus TUI and return a process exit code."""
    NexusApp(
        events_path=events_path,
        sessions_path=sessions_path,
        config_path=config_path,
        square_path=square_path,
        board_path=board_path,
        routes_path=routes_path,
        dry_run=dry_run,
        verbosity=verbosity,
        initial_theme=theme,
        show_sidebar=show_sidebar,
        profile=profile,
        entity_mode=entity_mode,
    ).run()
    return 0
