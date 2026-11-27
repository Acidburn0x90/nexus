"""Sandboxed drop-folder delivery and artifact saving for Nexus."""

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Optional

DEFAULT_DROP_DIR = Path.home() / ".local" / "share" / "nexus" / "drops"
MAX_DROP_BYTES = 2 * 1024 * 1024  # 2 MiB safety cap per dropped file

_BLOCKED_SEGMENTS = frozenset({".stfolder", ".stignore", ".stversions"})

_FENCE_OPEN_RE = re.compile(r"^\s*(`{3,})([A-Za-z0-9_+.-]*)\s*$")
_LANG_EXTS = {
    "python": ".py",
    "py": ".py",
    "bash": ".sh",
    "sh": ".sh",
    "zsh": ".sh",
    "javascript": ".js",
    "js": ".js",
    "typescript": ".ts",
    "ts": ".ts",
    "json": ".json",
    "yaml": ".yaml",
    "yml": ".yml",
    "rust": ".rs",
    "rs": ".rs",
    "c": ".c",
    "cpp": ".cpp",
    "cxx": ".cpp",
    "java": ".java",
    "go": ".go",
    "html": ".html",
    "css": ".css",
    "sql": ".sql",
    "markdown": ".md",
    "md": ".md",
    "toml": ".toml",
}

SEND_FILE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "send_file",
        "description": (
            "Write a generated script or text file into the configured Nexus drops "
            "directory (~/.local/share/nexus/drops or NEXUS_DROP_DIR)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": (
                        "Safe relative filename or subpath inside the Nexus drops directory "
                        "(e.g. 'backup.sh' or 'notes/summary.md')."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "Complete file contents as UTF-8 text.",
                },
                "executable": {
                    "type": "boolean",
                    "description": "Set chmod +x (0755) for executable scripts.",
                },
            },
            "required": ["filename", "content"],
        },
    },
}


def get_protected_paths(extra: Optional[list[Path]] = None) -> list[Path]:
    """Return resolved protected directory paths from `NEXUS_PROTECTED_PATHS` + `extra`."""
    roots: list[Path] = []
    raw = os.environ.get("NEXUS_PROTECTED_PATHS", "").strip()
    if raw:
        for item in raw.split(os.pathsep):
            entry = item.strip()
            if entry:
                roots.append(Path(entry).expanduser().resolve())
    if extra:
        for item in extra:
            roots.append(Path(item).expanduser().resolve())
    return roots


def resolve_drop_dir(drop_dir: Optional[Path] = None) -> Path:
    """Resolve the active drop folder (`NEXUS_DROP_DIR` overrides the default)."""
    if drop_dir is not None:
        return Path(drop_dir).expanduser()
    env = os.environ.get("NEXUS_DROP_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    return DEFAULT_DROP_DIR


def _should_exec(content: str, dest: Path, executable: Optional[bool]) -> bool:
    if executable is not None:
        return bool(executable)
    return content.startswith("#!") or dest.suffix.lower() == ".sh"


def _atomic_write_text(dest: Path, content: str, *, executable: Optional[bool] = None) -> dict[str, Any]:
    """Atomically write UTF-8 text (<= 2 MiB) to `dest`, returning size and mode metadata."""
    data = content.encode("utf-8")
    if len(data) > MAX_DROP_BYTES:
        raise ValueError(f"file exceeds {MAX_DROP_BYTES} byte limit ({len(data)} bytes)")

    is_exec = _should_exec(content, dest, executable)
    mode = 0o755 if is_exec else 0o644
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=dest.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), mode)
        os.replace(tmp_name, dest)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return {
        "ok": True,
        "path": str(dest),
        "bytes": len(data),
        "executable": is_exec,
    }


def _validated_relative_parts(filename: str) -> list[str]:
    """Validate and split a sandboxed relative filename into clean path segments."""
    if not isinstance(filename, str):
        raise ValueError("filename must be a string")
    raw = filename.strip()
    if not raw:
        raise ValueError("filename is required")
    if "\x00" in raw or "\\" in raw or ":" in raw:
        raise ValueError(f"invalid characters in filename: {filename!r}")
    if raw.startswith(("/", "~")):
        raise ValueError(f"absolute or home paths are not allowed: {filename!r}")

    parts = raw.split("/")
    for part in parts:
        if part in ("", ".", ".."):
            raise ValueError(f"invalid path segment in filename: {filename!r}")
        if part.lower() in _BLOCKED_SEGMENTS:
            raise ValueError(f"protected sync metadata entry rejected: {filename!r}")
    return parts


def write_drop_file(
    filename: str,
    content: str,
    *,
    executable: Optional[bool] = None,
    drop_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Atomically write `content` under the sandboxed drop folder with path-escape guards."""
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    parts = _validated_relative_parts(filename)

    root_raw = resolve_drop_dir(drop_dir)
    if root_raw.is_symlink():
        raise ValueError(f"drop directory must not be a symlink: {root_raw}")
    root_raw.mkdir(parents=True, exist_ok=True)
    root = root_raw.resolve()

    curr = root
    for part in parts[:-1]:
        curr = curr / part
        if curr.is_symlink():
            raise ValueError(f"symlink path segment rejected: {part}")
        if curr.exists() and not curr.is_dir():
            raise ValueError(f"parent component is not a directory: {part}")

    rel = Path(*parts)
    dest_raw = root / rel
    if dest_raw.is_symlink():
        raise ValueError(f"destination must not be a symlink: {filename!r}")
    if dest_raw.exists() and dest_raw.is_dir():
        raise ValueError(f"destination is a directory: {filename!r}")

    dest_resolved = dest_raw.resolve()
    if dest_resolved == root or not dest_resolved.is_relative_to(root):
        raise ValueError(f"path escapes drop directory: {filename!r}")

    info = _atomic_write_text(dest_resolved, content, executable=executable)
    info["relative"] = rel.as_posix()
    return info


def extract_code_block(text: str) -> tuple[str, str]:
    """Return `(body, lang)` for the largest fenced code block, else `(text.strip(), '')`."""
    lines = (text or "").splitlines()
    blocks: list[tuple[str, str]] = []
    fence: str = ""
    lang: str = ""
    buf: list[str] = []

    for line in lines:
        if not fence:
            match = _FENCE_OPEN_RE.match(line)
            if match:
                fence = match.group(1)
                lang = match.group(2).strip().lower()
                buf = []
            continue

        stripped = line.strip()
        if stripped.startswith("`") and len(stripped) >= len(fence) and set(stripped) == {"`"}:
            body = "\n".join(buf)
            if body.strip():
                blocks.append((body, lang))
            fence = ""
            lang = ""
            buf = []
            continue
        buf.append(line)

    if blocks:
        best_body, best_lang = max(blocks, key=lambda item: len(item[0]))
        return best_body.strip("\n"), best_lang
    return (text or "").strip(), ""


def save_text_artifact(
    text: str,
    target: str = "",
    *,
    drop_dir: Optional[Path] = None,
    extract_code: bool = True,
    raw_path: bool = False,
    executable: Optional[bool] = None,
    protected_paths: Optional[list[Path]] = None,
) -> dict[str, Any]:
    """Save extracted code (or raw text) to `target` or an auto-named drop file."""
    body, lang = extract_code_block(text) if extract_code else ((text or "").strip(), "")
    if not body.strip():
        raise ValueError("nothing to save")
    if not body.endswith("\n"):
        body += "\n"

    target_str = (target or "").strip()
    is_explicit_fs = raw_path or target_str.startswith(("/", "~", "./", "../"))
    if is_explicit_fs:
        if not target_str:
            raise ValueError("output file path is required")
        dest = Path(target_str).expanduser().resolve()
        for root in get_protected_paths(protected_paths):
            if dest == root or dest.is_relative_to(root):
                raise ValueError(f"refusing to write inside protected path: {root}")
        if dest.exists() and dest.is_dir():
            raise ValueError(f"destination is a directory: {dest}")
        info = _atomic_write_text(dest, body, executable=executable)
        info["relative"] = dest.name
        info["lang"] = lang
        return info

    if not target_str:
        ext = _LANG_EXTS.get(lang, ".txt")
        target_str = f"drop-{time.strftime('%Y%m%d-%H%M%S')}{ext}"

    info = write_drop_file(target_str, body, executable=executable, drop_dir=drop_dir)
    info["lang"] = lang
    return info
