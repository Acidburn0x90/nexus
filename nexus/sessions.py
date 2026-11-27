"""Persistent TUI sessions and configuration for Nexus.

The TUI keeps a short rolling list of conversations in one JSON document and a
small user config next to the other Nexus state. Both writers are atomic
(`tempfile` + `os.replace`) so a crash or a concurrent reader never observes a
half-written file, and both readers tolerate missing or corrupt files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Optional
import uuid

DEFAULT_SESSIONS_PATH = Path.home() / ".local/state/nexus/tui_sessions.json"
DEFAULT_TUI_CONFIG_PATH = Path.home() / ".config/nexus/tui.json"

DEFAULT_TUI_CONFIG: dict[str, Any] = {
    "theme": "nexus-charcoal",
    "sidebar_open": True,
    "use_llm_decision": True,
    "verbosity": 1,
    "profile": "default",
    "entity_mode": True,
    "ui_mode": "webui",
}


@dataclass
class TranscriptItem:
    kind: str
    content: str
    route: str = ""
    model: str = ""
    backend: str = ""
    quota_tier: str = ""
    status: str = "DONE"
    elapsed_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tools: list[str] = field(default_factory=list)
    thinking: str = ""
    trace_id: str = ""
    council_seats: list[dict] = field(default_factory=list)
    ts: float = field(default_factory=time.time)


@dataclass
class TuiSession:
    id: str
    title: str
    created_at: float
    updated_at: float
    active_route: Optional[str] = None
    active_model: str = ""
    active_backend: str = ""
    backend_session_id: Optional[str] = None
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    premium_spans: int = 0
    profile: str = "default"
    entity_mode: bool = True
    pinned_route: Optional[str] = None
    last_route: Optional[str] = None
    items: list[TranscriptItem] = field(default_factory=list)


def new_session(title: str = "New session") -> TuiSession:
    """Create an unsaved session with a short unique id."""
    now = time.time()
    return TuiSession(
        id="tui_" + uuid.uuid4().hex[:8],
        title=title,
        created_at=now,
        updated_at=now,
    )


_ITEM_FIELDS = {f.name for f in fields(TranscriptItem)}
_SESSION_FIELDS = {f.name for f in fields(TuiSession)}


def _item_from_dict(data: Any) -> Optional[TranscriptItem]:
    if not isinstance(data, dict):
        return None
    values = {key: value for key, value in data.items() if key in _ITEM_FIELDS}
    try:
        return TranscriptItem(**values)
    except TypeError:
        return None


def _session_from_dict(data: Any) -> Optional[TuiSession]:
    if not isinstance(data, dict):
        return None
    values = {
        key: value
        for key, value in data.items()
        if key in _SESSION_FIELDS and key != "items"
    }
    if "premium_spans" not in values and "argon_spans" in data:
        values["premium_spans"] = int(data.get("argon_spans") or 0)
    raw_items = data.get("items")
    items = []
    if isinstance(raw_items, list):
        items = [item for item in (_item_from_dict(entry) for entry in raw_items) if item is not None]
    try:
        return TuiSession(items=items, **values)
    except TypeError:
        return None


def load_sessions(path: Optional[Path] = None) -> list[TuiSession]:
    """Load saved sessions; a missing or corrupt file yields an empty list."""
    target = Path(path) if path is not None else DEFAULT_SESSIONS_PATH
    try:
        raw = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return []
    if isinstance(data, dict):
        data = data.get("sessions")
    if not isinstance(data, list):
        return []
    sessions = [session for session in (_session_from_dict(entry) for entry in data) if session is not None]
    return sessions


def _atomic_write(target: Path, text: str) -> None:
    """Write `text` to `target` via a temp file and `os.replace`."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=target.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def save_sessions(
    sessions: list[TuiSession],
    path: Optional[Path] = None,
    max_sessions: int = 30,
) -> None:
    """Atomically save sessions, keeping only the most recent `max_sessions`."""
    target = Path(path) if path is not None else DEFAULT_SESSIONS_PATH
    keep = list(sessions)
    if max_sessions is not None and len(keep) > max_sessions:
        ordered = sorted(
            range(len(keep)),
            key=lambda index: float(getattr(keep[index], "updated_at", 0.0) or 0.0),
        )
        retained = set(ordered[-max_sessions:]) if max_sessions > 0 else set()
        keep = [session for index, session in enumerate(keep) if index in retained]
    payload = json.dumps([asdict(session) for session in keep], indent=2, ensure_ascii=False)
    _atomic_write(target, payload + "\n")


def load_tui_config(path: Optional[Path] = None) -> dict:
    """Load the TUI config, merging saved keys over the defaults."""
    target = Path(path) if path is not None else DEFAULT_TUI_CONFIG_PATH
    config = dict(DEFAULT_TUI_CONFIG)
    try:
        raw = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return config
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return config
    if isinstance(data, dict):
        config.update(data)
    return config


def save_tui_config(cfg: dict, path: Optional[Path] = None) -> None:
    """Atomically save the TUI config."""
    target = Path(path) if path is not None else DEFAULT_TUI_CONFIG_PATH
    _atomic_write(target, json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
