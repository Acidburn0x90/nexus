"""Machine-readable CLI contract, trust-root validation, and VersionDrift checks."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Optional

CONTRACT_SCHEMA = "nexus-cli/v1"

EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_USAGE = 2
EXIT_NOT_FOUND = 3
EXIT_INTEGRITY = 4
EXIT_REFUSED = 5


def make_envelope(
    command: str,
    *,
    ok: bool,
    data: Optional[dict[str, Any]] = None,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
) -> dict[str, Any]:
    """Build a strict 5-key CLI response envelope matching the LifeOS Cortex contract."""
    if ok:
        err = None
        payload_data: Optional[dict[str, Any]] = dict(data) if data is not None else {}
    else:
        payload_data = None
        err = {
            "code": str(error_code or "internal_error"),
            "message": str(error_message or "command failed"),
        }
    return {
        "schema": CONTRACT_SCHEMA,
        "ok": bool(ok),
        "command": str(command or "nexus"),
        "data": payload_data,
        "error": err,
    }


def validate_trust_root(root: Path) -> tuple[bool, list[str]]:
    """Verify `root` is a valid directory with no nested symlinks or path escapes."""
    raw_root = Path(root).expanduser()
    try:
        resolved_root = raw_root.resolve(strict=True)
    except OSError as exc:
        return False, [f"trust root missing or unreadable: {raw_root} ({exc})"]
    if not resolved_root.is_dir():
        return False, [f"trust root is not a directory: {resolved_root}"]

    errors: list[str] = []
    for dirpath, dirnames, filenames in os.walk(resolved_root, followlinks=False):
        current = Path(dirpath)
        for name in list(dirnames) + list(filenames):
            candidate = current / name
            if candidate.is_symlink():
                errors.append(f"integrity_error: nested symlink not permitted under trust root: {candidate}")
                continue
            try:
                candidate.resolve().relative_to(resolved_root)
            except (OSError, ValueError):
                errors.append(f"integrity_error: path escapes trust root: {candidate}")
    return (len(errors) == 0), errors


def check_version_drift(repo_root: Path, *, threshold: int = 10) -> dict[str, Any]:
    """Flag core Python edits accumulating ahead of AGENTS.md/README.md/docs updates."""
    root = Path(repo_root)
    result: dict[str, Any] = {
        "git_available": False,
        "core_changed": 0,
        "docs_changed": 0,
        "threshold": max(1, int(threshold)),
        "drifted": False,
        "core_files": [],
        "doc_files": [],
    }
    if not (root / ".git").exists():
        return result

    changed: set[str] = set()
    try:
        status_proc = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "-uall"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if status_proc.returncode != 0:
            return result
        result["git_available"] = True
        for line in status_proc.stdout.splitlines():
            if len(line) > 3:
                path_part = line[3:].strip()
                if " -> " in path_part:
                    path_part = path_part.split(" -> ", 1)[1].strip()
                if path_part:
                    changed.add(path_part)

        # Also examine the working diff vs HEAD~5 when working tree is clean so recent drift surfaces.
        if not changed:
            log_proc = subprocess.run(
                ["git", "-C", str(root), "diff", "--name-only", "HEAD~5..HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if log_proc.returncode == 0:
                for line in log_proc.stdout.splitlines():
                    if line.strip():
                        changed.add(line.strip())
    except (OSError, subprocess.SubprocessError):
        return result

    core_files = sorted(
        p for p in changed if p.startswith("nexus/") and p.endswith(".py")
    )
    doc_files = sorted(
        p for p in changed if p in ("AGENTS.md", "README.md") or p.startswith("docs/")
    )
    result["core_files"] = core_files
    result["doc_files"] = doc_files
    result["core_changed"] = len(core_files)
    result["docs_changed"] = len(doc_files)
    result["drifted"] = bool(len(core_files) >= result["threshold"] and len(doc_files) == 0)
    return result
