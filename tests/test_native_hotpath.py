"""Tests for the compiled C99 hot path (native/nexus_fast.c) and its ctypes bridge.

The library is an optional accelerator, so every behavioural test asserts
byte-for-byte parity with the pure-Python implementation in
``nexus.visualizer``. Build it first:

    make -C native
    python3 -m pytest tests/test_native_hotpath.py -q
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nexus import native_bridge
from nexus.native_bridge import (
    find_tail_offset_fast,
    native_available,
    native_status,
    strip_ansi_fast,
)
from nexus.visualizer import _ANSI_RE, strip_ansi

BUILT = pytest.mark.skipif(
    not native_bridge.LIBRARY_PATH.is_file(),
    reason="libnexus_fast.so is not built; run `make -C native`",
)


def _python_strip(s: str) -> str:
    """Reference implementation, identical to nexus.visualizer.strip_ansi."""
    return _ANSI_RE.sub("", s)


def _python_tail_offset(data: bytes, keep_lines: int) -> int:
    """Reference tail seek: ignore one trailing newline, count backwards."""
    if keep_lines == 0:
        return len(data)
    if not data:
        return 0
    end = len(data)
    if data[end - 1 : end] == b"\n":
        end -= 1
    count = 0
    for offset in range(end - 1, -1, -1):
        if data[offset] == 0x0A:
            count += 1
            if count == keep_lines:
                return offset + 1
    return 0


# --------------------------------------------------------------------------
# Availability & status
# --------------------------------------------------------------------------


@BUILT
def test_native_available_when_built(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEXUS_DISABLE_NATIVE", raising=False)
    assert native_available() is True
    status = native_status()
    assert status["available"] is True
    assert status["library_path"] == str(native_bridge.LIBRARY_PATH)
    assert status["functions"] == ["nexus_strip_ansi", "nexus_find_tail_offset"]


@BUILT
@pytest.mark.parametrize("value", ["1", "true", "yes", " TRUE "])
def test_native_disabled_by_env(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("NEXUS_DISABLE_NATIVE", value)
    assert native_available() is False
    assert native_status()["available"] is False
    # The fallback still produces correct output.
    assert strip_ansi_fast("\x1b[31mred\x1b[0m") == "red"
    assert find_tail_offset_fast(Path("/nonexistent"), 1) is None  # disabled short-circuit


@BUILT
def test_native_reenabled_after_env_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEXUS_DISABLE_NATIVE", "1")
    assert native_available() is False
    monkeypatch.delenv("NEXUS_DISABLE_NATIVE")
    assert native_available() is True


@pytest.mark.parametrize("value", ["0", "false", "no", ""])
def test_native_env_falsy_values_keep_native(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("NEXUS_DISABLE_NATIVE", value)
    assert native_available() is (native_bridge.LIBRARY_PATH.is_file())


# --------------------------------------------------------------------------
# strip_ansi parity
# --------------------------------------------------------------------------

ANSI_CASES = [
    "",
    "plain ascii text",
    "\x1b[1mbold\x1b[0m",
    "\x1b[38;5;196m256-color\x1b[0m",
    "\x1b[38;2;10;20;30mtruecolor\x1b[0m",
    "\x1b[1;4;31mcombined\x1b[0m",
    "✓ ┆ 🚀 ─ box drawing and emoji \x1b[32m✓\x1b[0m 🚀",
    "\x1b[31m\x1b[42m\x1b[1mconsecutive\x1b[0m\x1b[0m",
    "\x1b[1m┆ thinking ─────────────\x1b[0m 1.2k→340 tok",
    # malformed / incomplete sequences: ESC must be copied literally
    "\x1b",
    "\x1b[",
    "\x1b[31",
    "\x1b[31;",
    "\x1b[12x text",
    "\x1bXnot-csi",
    "start\x1b[0m end\x1b",
    "\x1b[\x1b[31mreal\x1b[0m",
    "\x1b[0m\x1b",
    "🚀\x1b[1m",
]


@BUILT
@pytest.mark.parametrize("case", ANSI_CASES)
def test_strip_ansi_parity(case: str) -> None:
    assert strip_ansi_fast(case) == _python_strip(case)


@BUILT
def test_strip_ansi_parity_over_rendered_tree_lines() -> None:
    """Renderer-style coloured lines, including box characters and arrows."""
    from nexus.visualizer import BOLD, GREEN, RESET

    rendered = (
        f"{BOLD}{GREEN}@code{RESET}  ✓ 1.2s  ┆ thinking ── 1.2k→340 tok\n"
        f"{BOLD}[PREMIUM]{RESET} @reason · 3 tools · thought 1.5kc\n"
        "├─ └─ ✓ ┆ 🚀 ─────────────"
    )
    assert "\x1b" in rendered
    assert strip_ansi_fast(rendered) == strip_ansi(rendered)


@BUILT
def test_strip_ansi_matches_public_wrapper() -> None:
    line = "\x1b[1;38;5;196m[PREMIUM]\x1b[0m @reason · 3 tools · thought 1.5kc"
    assert strip_ansi_fast(line) == strip_ansi(line)


def test_strip_ansi_fast_without_escape_returns_same_object() -> None:
    text = "no escapes here ✓"
    assert strip_ansi_fast(text) is text


# --------------------------------------------------------------------------
# find_tail_offset parity
# --------------------------------------------------------------------------


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


@BUILT
def test_tail_offset_empty_file(tmp_path: Path) -> None:
    target = _write(tmp_path / "empty.jsonl", b"")
    for keep in (0, 1, 5):
        assert find_tail_offset_fast(target, keep) == 0


@BUILT
def test_tail_offset_missing_file(tmp_path: Path) -> None:
    assert find_tail_offset_fast(tmp_path / "missing.jsonl", 3) is None


@BUILT
def test_tail_offset_negative_keep_lines(tmp_path: Path) -> None:
    target = _write(tmp_path / "one.jsonl", b'{"a": 1}\n')
    assert find_tail_offset_fast(target, -1) is None


@BUILT
@pytest.mark.parametrize(
    "data, keep",
    [
        (b"one line no newline", 1),
        (b"one line no newline", 2),
        (b"only\n", 1),
        (b"only\n", 2),
        (b"a\nb\nc\n", 1),
        (b"a\nb\nc\n", 2),
        (b"a\nb\nc\n", 3),
        (b"a\nb\nc\n", 4),
        (b"a\nb\nc", 2),
        (b"a\nb\nc", 3),
        (b"a\n\nb\n", 2),
        (b"\n\n\n", 2),
    ],
)
def test_tail_offset_parity(tmp_path: Path, data: bytes, keep: int) -> None:
    target = _write(tmp_path / "case.jsonl", data)
    expected = _python_tail_offset(data, keep)
    assert find_tail_offset_fast(target, keep) == expected
    # Semantic check: the slice must hold the trailing lines.
    sliced = data[expected:].splitlines()
    assert sliced == (data.splitlines()[-keep:] if keep else [])


@BUILT
@pytest.mark.parametrize("keep", [50, 200, 2000])
def test_tail_offset_large_jsonl(tmp_path: Path, keep: int) -> None:
    lines = [
        json.dumps({"span_id": f"s{i}", "trace_id": f"t{i:06d}", "model": "x" * 40, "n": i})
        for i in range(4000)
    ]
    data = ("\n".join(lines) + "\n").encode("utf-8")
    assert len(data) > 150 * 1024

    target = _write(tmp_path / "events.jsonl", data)
    expected = _python_tail_offset(data, keep)

    assert find_tail_offset_fast(target, keep) == expected
    assert len(data[expected:].splitlines()) == keep
    assert data[expected:].splitlines() == [line.encode("utf-8") for line in lines[-keep:]]


@BUILT
def test_tail_offset_matches_keep_zero_is_file_size(tmp_path: Path) -> None:
    data = b'{"a": 1}\n{"b": 2}\n'
    target = _write(tmp_path / "events.jsonl", data)
    assert find_tail_offset_fast(target, 0) == len(data)


# --------------------------------------------------------------------------
# Microbenchmark (correctness is asserted; timings are informational only)
# --------------------------------------------------------------------------


@BUILT
def test_strip_ansi_microbenchmark_matches_and_logs(capsys: pytest.CaptureFixture[str]) -> None:
    palette = [f"\x1b[38;5;{n}m" for n in range(16, 256, 7)]
    lines = [
        f"{palette[i % len(palette)]}@code\x1b[0m ✓ {i} ┆ thinking ── "
        f"\x1b[1;32m{i * 3} tok\x1b[0m 🚀\r\n"
        for i in range(10_000)
    ]
    payload = "".join(lines)

    begin = time.perf_counter()
    expected = [strip_ansi(line) for line in lines]
    python_seconds = time.perf_counter() - begin

    begin = time.perf_counter()
    actual = [strip_ansi_fast(line) for line in lines]
    native_seconds = time.perf_counter() - begin

    with capsys.disabled():
        print(
            f"\n[strip_ansi 10k lines] python={python_seconds:.4f}s "
            f"native={native_seconds:.4f}s "
            f"ratio={python_seconds / max(native_seconds, 1e-9):.2f}x"
        )
        print(f"[strip_ansi 10k lines] payload={len(payload)} bytes")

    assert actual == expected

    # Per-call ctypes overhead dominates for short lines; on a batched payload
    # the native loop wins. Report both, assert the bytes, never the clock.
    begin = time.perf_counter()
    batch_expected = strip_ansi(payload)
    batch_python_seconds = time.perf_counter() - begin

    begin = time.perf_counter()
    batch_actual = strip_ansi_fast(payload)
    batch_native_seconds = time.perf_counter() - begin

    with capsys.disabled():
        print(
            f"[strip_ansi 1 batch] python={batch_python_seconds:.4f}s "
            f"native={batch_native_seconds:.4f}s "
            f"ratio={batch_python_seconds / max(batch_native_seconds, 1e-9):.2f}x"
        )

    assert batch_actual == batch_expected
