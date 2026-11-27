"""Optional ctypes bridge to the compiled ``libnexus_fast.so`` hot path.

The shared object is built from ``native/nexus_fast.c`` with
``make -C native`` and is a build artifact (git-ignored). Nothing in this
module is required for correct behaviour: every entry point degrades to the
pure-Python implementation when the library is missing, built for another
ABI, or explicitly disabled.

Disabling
---------
``NEXUS_DISABLE_NATIVE`` is read on *every* call, not at import time, so
``monkeypatch.setenv("NEXUS_DISABLE_NATIVE", "1")`` takes effect immediately
and re-enabling the variable restores the native path without a reload.
Accepted truthy values: ``1``, ``true``, ``yes`` (case-insensitive).
"""

from __future__ import annotations

import ctypes
import os
import re
import threading
from pathlib import Path
from typing import Optional

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

__all__ = [
    "native_available",
    "native_status",
    "strip_ansi_fast",
    "find_tail_offset_fast",
]

LIBRARY_PATH = Path(__file__).resolve().parent.parent / "native" / "libnexus_fast.so"

_DISABLE_ENV = "NEXUS_DISABLE_NATIVE"
_TRUTHY = frozenset({"1", "true", "yes"})

_STRIP_FUNCTIONS = ["nexus_strip_ansi", "nexus_find_tail_offset"]

_lib_lock = threading.Lock()
_lib: Optional[ctypes.CDLL] = None
_lib_failed = False


def _native_disabled() -> bool:
    """Return True when ``NEXUS_DISABLE_NATIVE`` requests the Python fallback."""
    value = os.environ.get(_DISABLE_ENV, "")
    return value.strip().lower() in _TRUTHY


def _bind(library: ctypes.CDLL) -> Optional[ctypes.CDLL]:
    """Attach signatures to every exported symbol, or reject the library."""

    strip = library.nexus_strip_ansi
    strip.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    strip.restype = ctypes.c_size_t

    tail = library.nexus_find_tail_offset
    tail.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    tail.restype = ctypes.c_longlong

    return library


def _load_library() -> Optional[ctypes.CDLL]:
    """Load and bind the shared object once, guarding with a lock.

    A missing or broken library is remembered as a failure so the hot path
    never stats the file twice, but a later successful build is picked up
    after the cached object is cleared by :func:`_reset_cache` (used in tests).
    """
    global _lib, _lib_failed

    with _lib_lock:
        if _lib is not None:
            return _lib
        if _lib_failed or not LIBRARY_PATH.is_file():
            return None
        try:
            candidate = ctypes.CDLL(str(LIBRARY_PATH))
            _lib = _bind(candidate)
        except (OSError, AttributeError):
            _lib = None
            _lib_failed = True
        return _lib


def _reset_cache() -> None:
    """Forget any previous load attempt (test helper, not part of the API)."""
    global _lib, _lib_failed

    with _lib_lock:
        _lib = None
        _lib_failed = False


def native_available() -> bool:
    """Return True when the native library can serve calls right now."""
    if _native_disabled():
        return False
    return _load_library() is not None


def native_status() -> dict:
    """Describe the native library and the functions it is expected to export."""
    return {
        "available": native_available(),
        "library_path": str(LIBRARY_PATH),
        "functions": list(_STRIP_FUNCTIONS),
    }


def strip_ansi_fast(s: str) -> str:
    """Strip ANSI CSI escapes, using the native library when it is available."""
    if "\x1b" not in s:
        return s

    library = None if _native_disabled() else _load_library()
    if library is not None:
        try:
            raw = s.encode("utf-8", "surrogatepass")
            buf = ctypes.create_string_buffer(len(raw) + 1)
            written = int(
                library.nexus_strip_ansi(
                    raw,
                    len(raw),
                    ctypes.cast(buf, ctypes.c_void_p),
                    len(raw) + 1,
                )
            )
            if 0 <= written <= len(raw):
                return bytes(buf.raw[:written]).decode("utf-8", "surrogatepass")
        except (UnicodeEncodeError, UnicodeDecodeError, ctypes.ArgumentError, OSError, ValueError):
            pass

    return _ANSI_RE.sub("", s)


def find_tail_offset_fast(path: Path, keep_lines: int) -> Optional[int]:
    """Return the byte offset of the last ``keep_lines`` lines, or None.

    ``None`` means "no native answer": the caller should use its own Python
    tail-seek. A native offset of 0 is a real answer (the whole file).
    """
    if keep_lines < 0:
        return None

    library = None if _native_disabled() else _load_library()
    if library is None:
        return None

    try:
        encoded = os.fsencode(str(path))
        offset = int(library.nexus_find_tail_offset(encoded, keep_lines))
    except (OSError, ValueError, ctypes.ArgumentError):
        return None

    return None if offset < 0 else offset
