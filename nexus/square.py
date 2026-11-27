"""Single-machine Town Square shared memory (`square.jsonl` + `BOARD.md`) for all agents and models."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import textwrap
import time
from typing import Optional
import uuid

from nexus.visualizer import pad_ansi, visible_len

DEFAULT_SQUARE_PATH = Path.home() / ".local/state/nexus/square.jsonl"
DEFAULT_BOARD_PATH = Path.home() / ".local/state/nexus/BOARD.md"
MAX_BOARD_BYTES = 4096


def post_square(
    body: str,
    *,
    author: str = "agent",
    model: str = "unknown",
    tag: str = "note",
    verified: bool = False,
    square_path: Optional[Path] = None,
) -> dict:
    """Append an atomic <4 KiB post to `~/.local/state/nexus/square.jsonl`."""
    target = Path(square_path) if square_path is not None else DEFAULT_SQUARE_PATH
    clean_body = body.strip()
    if not clean_body:
        raise ValueError("cannot post empty message to town square")

    record = {
        "ts": time.time(),
        "mono_ns": time.monotonic_ns(),
        "id": "sq_" + uuid.uuid4().hex[:6],
        "author": author[:48],
        "model": model[:80],
        "tag": tag[:32],
        "verified": bool(verified),
        "body": clean_body[:1500],
    }
    payload = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, payload.encode("utf-8"))
    finally:
        os.close(fd)
    return record


def read_square(
    *,
    tag: Optional[str] = None,
    limit: int = 10,
    square_path: Optional[Path] = None,
) -> list[dict]:
    """Read the newest `limit` Town Square posts from `square.jsonl`."""
    target = Path(square_path) if square_path is not None else DEFAULT_SQUARE_PATH
    try:
        raw = target.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return []

    posts: list[dict] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if tag is not None and obj.get("tag") != tag:
            continue
        posts.append(obj)

    if limit > 0:
        return posts[-limit:]
    return posts


def read_board(board_path: Optional[Path] = None) -> str:
    """Read the pinned Town Square whiteboard (`BOARD.md`), capped at 4 KiB."""
    target = Path(board_path) if board_path is not None else DEFAULT_BOARD_PATH
    try:
        return target.read_text(encoding="utf-8", errors="replace")[:MAX_BOARD_BYTES].strip()
    except FileNotFoundError:
        return ""


def write_board(
    content: str,
    *,
    author: str = "agent",
    board_path: Optional[Path] = None,
) -> Path:
    """Atomically write `BOARD.md` under an exclusive `flock` lock (capped at 4 KiB)."""
    target = Path(board_path) if board_path is not None else DEFAULT_BOARD_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_suffix(".lock")
    trimmed = content.strip()[:MAX_BOARD_BYTES]
    if trimmed:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        body = f"<!-- updated: {stamp} by {author} -->\n{trimmed}\n"
    else:
        body = ""

    lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, target)
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
    return target


def format_square_context(
    *,
    limit: int = 5,
    square_path: Optional[Path] = None,
    board_path: Optional[Path] = None,
) -> str:
    """Build a compact context block (<1.5 KB) for injection into spawned models."""
    board = read_board(board_path=board_path)
    posts = read_square(limit=limit, square_path=square_path)
    if not board and not posts:
        return ""

    lines = ["[NEXUS TOWN SQUARE — SHARED MULTI-AGENT CONTEXT]"]
    if board:
        lines.append("## Pinned Board (BOARD.md)")
        lines.append(board[:800])
    if posts:
        lines.append("## Recent Agent Posts (square.jsonl)")
        for p in posts:
            ts_str = time.strftime("%H:%M:%S", time.localtime(float(p.get("ts", 0))))
            v_badge = "✓verified" if p.get("verified") else "note"
            author = p.get("author", "agent")
            model = p.get("model", "")
            tag = p.get("tag", "note")
            body = " ".join(str(p.get("body", "")).splitlines())[:220]
            lines.append(f"- [{ts_str}] ({author} · {model} · #{tag} · {v_badge}): {body}")
    lines.append("[END TOWN SQUARE CONTEXT]\n")
    return "\n".join(lines)


def render_square_view(
    *,
    limit: int = 10,
    tag: Optional[str] = None,
    square_path: Optional[Path] = None,
    board_path: Optional[Path] = None,
    width: Optional[int] = None,
    color: bool = True,
) -> str:
    """Render a terminal-friendly view of the Town Square (`BOARD.md` + `square.jsonl`)."""
    board = read_board(board_path=board_path)
    posts = read_square(tag=tag, limit=limit, square_path=square_path)

    if width is None:
        term_cols = shutil.get_terminal_size(fallback=(88, 30)).columns
        width = max(80, min(104, term_cols))

    # Strip HTML comment and extract update info
    m = re.search(r"<!--\s*updated:\s*([0-9\-]+ [0-9:]+ by [^>]+?)\s*-->", board)
    update_str = f" (updated {m.group(1).strip()})" if m else ""
    clean_board = re.sub(r"<!--\s*updated:[^>]*-->", "", board).strip()

    lines: list[str] = []

    # Top header
    top_title = "[3:square] NEXUS TOWN SQUARE // ~/.local/state/nexus/"
    top_rem = max(0, width - visible_len(top_title) - 6)
    lines.append(f"┌─ {top_title} {'─' * top_rem}─┐")

    # Board section header
    board_header = f"BOARD.md // PINNED WHITEBOARD{update_str}"
    lines.append(f"│ {pad_ansi(board_header, width - 4)} │")

    # Board body
    if clean_board:
        for bl in clean_board.splitlines():
            s = bl.rstrip()
            if not s:
                lines.append(f"│ {' ' * (width - 4)} │")
            else:
                wrapped = textwrap.wrap(s, width=width - 6) or [""]
                for wl in wrapped:
                    lines.append(f"│   {pad_ansi(wl, width - 6)} │")
    else:
        lines.append(f"│   {pad_ansi('(empty — set with `nexus square board \"<markdown>\"`)', width - 6)} │")

    # Divider between board and feed
    lines.append(f"├{'─' * (width - 2)}┤")

    # Feed section header
    feed_header = f"SQUARE.JSONL // RECENT FEED (last {len(posts)})"
    lines.append(f"│ {pad_ansi(feed_header, width - 4)} │")

    # Feed posts
    if not posts:
        lines.append(f"│   {pad_ansi('(no posts yet — add with `nexus square post \"<message>\"`)', width - 6)} │")
    else:
        for p in posts:
            ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(p.get("ts", 0))))
            v_mark = "✓" if p.get("verified") else "•"
            author = p.get("author", "agent")
            model = p.get("model", "manual")
            ptag = p.get("tag", "note")
            body = str(p.get("body", ""))
            entry_str = f"{v_mark} [{ts_str}] #{ptag} ({author} · {model}): {body}"
            wrapped = textwrap.wrap(entry_str, width=width - 6) or [""]
            for wl in wrapped:
                lines.append(f"│   {pad_ansi(wl, width - 6)} │")

    # Bottom border
    lines.append(f"└{'─' * (width - 2)}┘")
    return "\n".join(lines)
