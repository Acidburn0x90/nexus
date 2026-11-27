"""Tests for Nexus delegation tree visualizer and CLI entrypoint."""

from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nexus.events import emit_event, read_events
from nexus.square import post_square, render_square_view, write_board
from nexus.visualizer import (
    BOLD,
    GREEN,
    RED,
    RESET,
    format_quota_badge,
    pad_ansi,
    render_delegation_tree,
    render_panel,
    render_routes_table,
    strip_ansi,
    truncate_ansi,
    visible_len,
)


def test_visualizer_renders_hierarchy_and_premium_badge(tmp_path: Path) -> None:
    events_file = tmp_path / "events.jsonl"
    trace_id = "tr_test0001"

    emit_event(
        trace_id,
        "sp_root",
        parent_id=None,
        route="entrypoint",
        intent="dispatch",
        backend="nexus",
        model="nexus-router",
        quota_tier="ORCHESTRATOR",
        status="DONE",
        step="completed",
        prompt="Audit and refactor parser",
        elapsed_ms=1250,
        path=events_file,
    )
    emit_event(
        trace_id,
        "sp_child1",
        parent_id="sp_root",
        route="reason",
        intent="reason",
        backend="opencode",
        model="opencode-go/deepseek-v4-pro",
        quota_tier="GO",
        status="DONE",
        step="tool:read",
        tokens_in=420,
        tokens_out=110,
        elapsed_ms=800,
        path=events_file,
    )
    emit_event(
        trace_id,
        "sp_child2",
        parent_id="sp_root",
        route="code",
        intent="code",
        backend="opencode",
        model="opencode/deepseek-v4.1-flash-free",
        quota_tier="FREE",
        status="DONE",
        step="completed",
        tokens_in=200,
        tokens_out=95,
        elapsed_ms=450,
        path=events_file,
    )

    events = read_events(path=events_file, trace_id=trace_id)
    out = render_delegation_tree(events, trace_id=trace_id, color=False)
    assert "trace: tr_test0001" in out
    assert "Zero-Cost: 2 spans ($0)" in out
    assert "[reason] opencode:opencode-go/deepseek-v4-pro [GO]" in out
    assert "[code] opencode:opencode/deepseek-v4.1-flash-free [FREE $0]" in out

    # Add a premium span and verify the red warning badge appears.
    emit_event(
        trace_id,
        "sp_premium",
        parent_id="sp_root",
        route="premium",
        intent="premium",
        backend="agy",
        model="premium-cli",
        quota_tier="PREMIUM",
        status="DONE",
        step="external CLI complete",
        tokens_in=500,
        tokens_out=120,
        elapsed_ms=3200,
        path=events_file,
    )
    events2 = read_events(path=events_file, trace_id=trace_id)
    out2 = render_delegation_tree(events2, trace_id=trace_id, color=False)
    assert "PREMIUM: 1 span" in out2
    assert "Quota: PREMIUM (1)" in out2
    assert "[PREMIUM]" in out2


def test_format_quota_badge_supports_legacy_premium_alias() -> None:
    assert format_quota_badge("ARGON-QUOTA", color=False) == "[PREMIUM]"


def test_cli_dry_run_single_and_council(tmp_path: Path) -> None:
    events_file = tmp_path / "events.jsonl"
    bin_path = PROJECT_ROOT / "bin" / "nexus"

    proc = subprocess.run(
        [
            sys.executable,
            str(bin_path),
            "--events-file",
            str(events_file),
            "--no-color",
            "run",
            "--dry-run",
            "@reason",
            "Verify security invariants",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[dry-run] in-charge:reason -> opencode:opencode-go/glm-5.3" in proc.stdout
    assert "NEXUS DELEGATION TREE" in proc.stderr
    assert "Zero-Cost: 1 span ($0)" in proc.stderr

    proc_council = subprocess.run(
        [
            sys.executable,
            str(bin_path),
            "--events-file",
            str(events_file),
            "--no-color",
            "council",
            "--dry-run",
            "Compare JSONL vs SQLite",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "PIONEER" in proc_council.stdout
    assert "RED-TEAMER" in proc_council.stdout
    assert "SYNTHESIZER" in proc_council.stdout
    assert "Zero-Cost: 3 spans ($0)" in proc_council.stderr

    # Verify bare `nexus` REPL default Unified Entity Mode + dynamic re-routing
    proc_repl = subprocess.run(
        [
            sys.executable,
            str(bin_path),
            "--events-file",
            str(events_file),
            "--no-color",
            "--dry-run",
        ],
        input="please refactor and modularize parser.py\nwrite a bash helper\n/quit\n",
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Profile: default [entity] (entity)" in proc_repl.stdout
    assert "nexus [entity: default" in proc_repl.stdout
    assert "via @refactor]" in proc_repl.stdout
    assert "via @code]" in proc_repl.stdout
    assert "nexus [in-charge:" not in proc_repl.stdout

    # Verify --sticky preserves classic orchestrator takeover -> /release flow
    proc_sticky = subprocess.run(
        [
            sys.executable,
            str(bin_path),
            "--events-file",
            str(events_file),
            "--no-color",
            "--sticky",
            "--dry-run",
        ],
        input="please refactor and modularize parser.py\nfollow up step\n/release\n/quit\n",
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Orchestrator @refactor (opencode-go/kimi-k3.1-pro) has taken charge" in proc_sticky.stdout
    assert "nexus [in-charge: @refactor" in proc_sticky.stdout
    assert "Unpinned @refactor" in proc_sticky.stdout


def test_town_square_board_and_feed_cli(tmp_path: Path) -> None:
    sq_file = tmp_path / "square.jsonl"
    bd_file = tmp_path / "BOARD.md"
    bin_path = PROJECT_ROOT / "bin" / "nexus"

    subprocess.run(
        [
            sys.executable,
            str(bin_path),
            "square",
            "--square-file",
            str(sq_file),
            "--board-file",
            str(bd_file),
            "board",
            "-a",
            "agy",
            "Active Sprint: Nexus Town Square live",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(bin_path),
            "square",
            "--square-file",
            str(sq_file),
            "--board-file",
            str(bd_file),
            "post",
            "-a",
            "opencode",
            "-m",
            "deepseek-v4.1-flash-free",
            "-t",
            "finding",
            "--verified",
            "All 11 routes validated against opencode models",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    view_proc = subprocess.run(
        [
            sys.executable,
            str(bin_path),
            "square",
            "--square-file",
            str(sq_file),
            "--board-file",
            str(bd_file),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    import re as _re

    view_out = _re.sub(r"[┌─┐└┘├┤│\s]+", " ", view_proc.stdout)
    assert "Active Sprint: Nexus Town Square live" in view_out
    assert "All 11 routes validated against opencode models" in view_out
    assert "#finding" in view_out

    digest_proc = subprocess.run(
        [
            sys.executable,
            str(bin_path),
            "square",
            "--square-file",
            str(sq_file),
            "--board-file",
            str(bd_file),
            "--digest",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "SQUARE | board: Active Sprint: Nexus Town Square live" in digest_proc.stdout


def test_visualizer_box_alignment_and_no_rounded_corners(tmp_path: Path) -> None:
    events_file = tmp_path / "events.jsonl"
    trace_id = "tr_align_test"

    # Test empty tree
    empty_out = render_delegation_tree([], trace_id=trace_id, color=False, width=88)
    for corner in ("╭", "╮", "╰", "╯"):
        assert corner not in empty_out
    empty_lines = empty_out.splitlines()
    assert len(empty_lines) >= 3
    for line in empty_lines:
        assert visible_len(line) == 88
        assert line.startswith(("┌", "│", "└"))
        assert line.endswith(("┐", "│", "┘"))

    # Test tree with multiple spans and colors
    emit_event(
        trace_id,
        "sp_root",
        parent_id=None,
        route="front-desk",
        model="nemotron-3.5-lightning-free",
        quota_tier="FREE",
        status="DONE",
        elapsed_ms=120,
        path=events_file,
    )
    emit_event(
        trace_id,
        "sp_child",
        parent_id="sp_root",
        route="code",
        model="deepseek-v4.1-flash-free",
        quota_tier="FREE",
        status="DONE",
        elapsed_ms=540,
        tokens_in=300,
        tokens_out=150,
        path=events_file,
    )
    events = read_events(path=events_file, trace_id=trace_id)
    tree_out = render_delegation_tree(events, trace_id=trace_id, color=True, width=96)
    for corner in ("╭", "╮", "╰", "╯"):
        assert corner not in tree_out

    lines = tree_out.splitlines()
    for line in lines:
        assert visible_len(line) == 96
        assert line.endswith(("┐", "│", "┘"))
    assert "[1:tree] NEXUS DELEGATION TREE" in lines[0]
    assert "Spans: 2" in lines[-1]
    assert "Quota: $0 premium" in lines[-1]


def test_render_routes_table_alignment_and_columns() -> None:
    table_plain = render_routes_table(color=False, width=108)
    table_color = render_routes_table(color=True, width=108)

    for corner in ("╭", "╮", "╰", "╯"):
        assert corner not in table_plain
        assert corner not in table_color

    for tbl in (table_plain, table_color):
        lines = tbl.splitlines()
        for idx, line in enumerate(lines):
            assert visible_len(line) == 108, f"line {idx} width {visible_len(line)} != 108: {line!r}"
            assert line.endswith(("┐", "│", "┤", "┘"))

    # Verify column structure and legend
    assert "[2:routes] NEXUS FRONT-DESK & SPECIALIST ORCHESTRATOR TABLE" in table_plain
    assert "★ [front-desk]" in table_plain
    separator = next(line for line in table_plain.splitlines() if line.startswith("├"))
    assert separator.count("┼") == 4, f"expected 5 columns, got {separator!r}"
    assert separator.startswith("├───────────────┼────────┼")
    assert "★ front-desk router   * default specialist" in table_plain
    assert "--validate to audit pins" in table_plain


def test_render_square_view_strips_html_comment_and_aligns(tmp_path: Path) -> None:
    sq_file = tmp_path / "square.jsonl"
    bd_file = tmp_path / "BOARD.md"

    write_board(
        "Active Project Goals:\n1. Revamp terminal UI to 0-gap skeletal wireframe.\n2. Ensure zero ragged right borders.",
        author="premium-subagent",
        board_path=bd_file,
    )
    post_square(
        "Unit test verifying that long square post bodies word-wrap cleanly inside wireframe borders without overflowing width.",
        author="subagent",
        model="deepseek",
        tag="test-finding",
        verified=True,
        square_path=sq_file,
    )

    out = render_square_view(square_path=sq_file, board_path=bd_file, width=90, color=False)

    # HTML comment must NOT appear in body
    assert "<!-- updated:" not in out
    # Update stamp must be present in header
    assert "BOARD.md // PINNED WHITEBOARD (updated " in out
    assert "by premium-subagent" in out
    # No emojis or rounded corners
    for emoji in ("📌", "📣"):
        assert emoji not in out
    for corner in ("╭", "╮", "╰", "╯"):
        assert corner not in out

    lines = out.splitlines()
    for idx, line in enumerate(lines):
        assert visible_len(line) == 90, f"square line {idx} width {visible_len(line)} != 90: {line!r}"
        assert line.endswith(("┐", "│", "┤", "┘"))

    assert "[3:square] NEXUS TOWN SQUARE" in lines[0]
    assert "SQUARE.JSONL // RECENT FEED (last 1)" in out
    assert "#test-finding" in out


def test_ansi_helpers_pad_truncate_render_panel() -> None:
    assert visible_len("hello") == 5
    assert visible_len(f"{BOLD}hello{RESET}") == 5

    # Pad
    padded = pad_ansi(f"{GREEN}ok{RESET}", 10)
    assert visible_len(padded) == 10
    assert padded.startswith(GREEN)

    # Truncate
    long_str = f"{BOLD}abcdefghijklmnopqrstuvwxyz{RESET}"
    trunc = truncate_ansi(long_str, 10)
    assert visible_len(trunc) == 10

    # Panel
    panel = render_panel(["line 1", f"{BOLD}line 2 bold{RESET}"], title="Test Panel", width=50)
    for corner in ("╭", "╮", "╰", "╯"):
        assert corner not in panel
    for line in panel.splitlines():
        assert visible_len(line) == 50
        assert line.startswith(("┌", "│", "└"))
        assert line.endswith(("┐", "│", "┘"))


def test_render_panel_various_widths_titles_footers_ansi() -> None:
    """render_panel must keep 0-gap wireframe, consistent width, no rounded corners, across widths/titles/footers/ANSI."""
    for width in (20, 40, 80, 120):
        panel = render_panel(["line one", "line two"], title="T", footer="F", width=width)
        lines = panel.splitlines()
        assert len(lines) >= 3
        for line in lines:
            assert visible_len(line) == width, f"width={width}: {visible_len(line)} != {width}: {line!r}"
            assert line.startswith(("┌", "│", "└"))
            assert line.endswith(("┐", "│", "┘"))
        for corner in ("╭", "╮", "╰", "╯"):
            assert corner not in panel, f"rounded corner in width={width} panel"

    # No title / no footer
    panel_plain = render_panel(["a", "b"], width=30)
    lines = panel_plain.splitlines()
    assert lines[0].startswith("┌") and lines[0].endswith("┐")
    assert lines[-1].startswith("└") and lines[-1].endswith("┘")
    for line in lines:
        assert visible_len(line) == 30

    # ANSI content preserved, visible width still exact
    panel_ansi = render_panel([f"{BOLD}bold{RESET}", f"{RED}red{RESET}"], title="ANSI Panel", width=40)
    for line in panel_ansi.splitlines():
        assert visible_len(line) == 40

    # Empty content
    panel_empty = render_panel([], width=30)
    assert visible_len(panel_empty.splitlines()[0]) == 30


def test_render_delegation_tree_running_status_spinner(tmp_path: Path) -> None:
    """RUNNING status spans render the spinner icon (⠋) and report status without crashing."""
    events_file = tmp_path / "events.jsonl"
    trace_id = "tr_running01"

    emit_event(
        trace_id, "sp_root", parent_id=None, route="entrypoint",
        backend="nexus", model="nexus-router", quota_tier="ORCHESTRATOR",
        status="RUNNING", step="thinking", path=events_file,
    )
    emit_event(
        trace_id, "sp_child", parent_id="sp_root", route="code",
        backend="opencode", model="opencode/deepseek-v4.1-flash-free",
        quota_tier="FREE", status="RUNNING", step="generating", path=events_file,
    )

    events = read_events(path=events_file, trace_id=trace_id)
    out = render_delegation_tree(events, trace_id=trace_id, color=False, width=100)
    assert "trace: tr_running01" in out
    # Spinner icon for RUNNING status
    assert "⠋" in out
    assert "Zero-Cost: 1 span ($0)" in out

    # Color mode also keeps the spinner icon inside ANSI
    out_color = render_delegation_tree(events, trace_id=trace_id, color=True, width=100)
    assert "⠋" in out_color
    assert "│" in out_color  # wireframe borders intact


def test_render_routes_table_quota_badge_rendering() -> None:
    """Every quota tier badge (FREE/GO/LOCAL/ORCHESTRATOR/PREMIUM) appears in the routes table."""
    table_plain = render_routes_table(color=False, width=130)
    assert "[FREE $0]" in table_plain
    assert "[GO]" in table_plain
    assert "[LOCAL $0]" in table_plain
    assert "[PREMIUM]" in table_plain
    # ORCHESTRATOR tier badge ([ROUTER]) is not used by any route in the table

    # Color mode: badges present after stripping ANSI, alignment preserved
    table_color = render_routes_table(color=True, width=130)
    stripped = strip_ansi(table_color)
    assert "[FREE $0]" in stripped
    assert "[PREMIUM]" in stripped
    for line in table_color.splitlines():
        assert visible_len(line) == 130, f"{visible_len(line)} != 130: {line!r}"
        assert line.endswith(("┐", "│", "┤", "┘"))


def test_render_square_view_panel_consistency(tmp_path: Path) -> None:
    """render_square_view keeps exact width, wireframe borders, no rounded corners, across widths."""
    sq_file = tmp_path / "square.jsonl"
    bd_file = tmp_path / "BOARD.md"

    write_board("Test pinned board content for consistency checks", author="tester", board_path=bd_file)
    post_square("Test post body verifying panel consistency across widths", author="tester",
                model="test-model", tag="test", verified=True, square_path=sq_file)

    for width in (80, 90, 100, 120):
        out = render_square_view(square_path=sq_file, board_path=bd_file, width=width, color=False)
        lines = out.splitlines()
        assert len(lines) >= 3
        for idx, line in enumerate(lines):
            assert visible_len(line) == width, f"width={width} line {idx}: {visible_len(line)} != {width}: {line!r}"
            assert line.endswith(("┐", "│", "┤", "┘"))
        for corner in ("╭", "╮", "╰", "╯"):
            assert corner not in out
        assert "[3:square] NEXUS TOWN SQUARE" in lines[0]


def test_cli_exit_codes_help_no_color_dry_run(tmp_path: Path) -> None:
    """CLI exit codes and no-color discipline across help, routes, square, run, and council."""
    bin_path = PROJECT_ROOT / "bin" / "nexus"

    # --help exits 0
    proc = subprocess.run([sys.executable, str(bin_path), "--help"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "usage" in proc.stdout.lower() or "nexus" in proc.stdout.lower()

    # --no-color routes exits 0
    proc = subprocess.run(
        [sys.executable, str(bin_path), "--no-color", "routes"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"

    # --no-color square --digest exits 0
    proc = subprocess.run(
        [sys.executable, str(bin_path), "--no-color", "square", "--digest"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"

    # --no-color run --dry-run exits 0 and emits zero ANSI on stderr
    proc = subprocess.run(
        [sys.executable, str(bin_path), "--no-color", "run", "--dry-run", "-r", "code", "hello"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    assert "\x1b[" not in proc.stderr, f"ANSI leaked with --no-color: {proc.stderr!r}"
    assert "spawn @code" in proc.stderr
    assert "front-desk → @code" in proc.stderr

    # --no-color council --dry-run exits 0 and emits zero ANSI on stderr
    proc = subprocess.run(
        [sys.executable, str(bin_path), "--no-color", "council", "--dry-run", "hello council"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    assert "\x1b[" not in proc.stderr, f"ANSI leaked with --no-color: {proc.stderr!r}"


def test_status_line_and_formatters() -> None:
    """Status lines carry icons per kind; duration/token formatters stay compact."""
    from nexus.visualizer import format_duration, format_tokens, status_line

    assert status_line("ok", "done", color=False) == "NEXUS | ✓ done"
    assert status_line("fail", "boom", color=False) == "NEXUS | ✗ boom"
    assert status_line("timeout", "slow", color=False) == "NEXUS | ⏱ slow"
    assert status_line("stop", "halt", color=False) == "NEXUS | ⛔ halt"
    assert status_line("step", "tool", color=False) == "NEXUS | ▸ tool"
    assert status_line("info", "plan", color=False) == "NEXUS | ◆ plan"
    assert status_line("ok", "done", color=True).startswith("NEXUS | \x1b[32m✓\x1b[0m")
    assert status_line("ok", "done", color=False, prefix="X | ") == "X | ✓ done"

    assert format_duration(0) == "0ms"
    assert format_duration(999) == "999ms"
    assert format_duration(1000) == "1.0s"
    assert format_duration(12_400) == "12.4s"
    assert format_duration(64_000) == "1m 04s"

    assert format_tokens(0) == "0"
    assert format_tokens(812) == "812"
    assert format_tokens(1234) == "1.2k"
    assert format_tokens(41_300) == "41.3k"
    assert format_tokens(2_500_000) == "2.5M"


def _fake_tty_stderr(writes: list[str]):
    class _FakeStderr:
        def isatty(self) -> bool:
            return True

        def write(self, s: str) -> int:
            writes.append(s)
            return len(s)

        def flush(self) -> None:
            return None

    return _FakeStderr()


def test_spin_stderr_clears_line_and_stops(monkeypatch) -> None:
    """The spinner clears exactly the line it drew and honours its stop event."""
    import threading
    import time

    import nexus.backends as backends

    writes: list[str] = []
    monkeypatch.setattr(sys, "stderr", _fake_tty_stderr(writes))

    stop = threading.Event()
    spin = threading.Thread(
        target=backends._spin_stderr, args=("code", stop), kwargs={"color": False}, daemon=True
    )
    spin.start()

    deadline = time.monotonic() + 2.0
    while not any("thinking" in w for w in writes) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert any("thinking" in w for w in writes), "spinner never drew a frame"
    assert not any("\x1b[" in w for w in writes), "color=False must emit no ANSI"

    backends._stop_spin(stop, spin)
    assert not spin.is_alive()

    clear = writes[-1]
    assert set(clear) <= {"\r", " "}, f"final write is not a blanking clear: {clear!r}"
    assert clear.startswith("\r") and clear.endswith("\r")
    widest = max(len(w.lstrip("\r")) for w in writes if "thinking" in w)
    assert len(clear) - 2 >= widest, "clear shorter than the widest spinner frame"


def test_execute_route_stops_spinner_before_done_announcement(monkeypatch, tmp_path: Path) -> None:
    """The done receipt is printed only after the spinner line is cleared (no gluing)."""
    import nexus.backends as backends

    writes: list[str] = []
    monkeypatch.setattr(sys, "stderr", _fake_tty_stderr(writes))
    monkeypatch.setattr(backends, "_run_subprocess_bounded", lambda *a, **k: (0, "", "", False))
    monkeypatch.setattr(
        backends, "_parse_opencode_json_stream", lambda raw: backends.OpenCodeStream("ok", 3, 1, 0, "complete", "")
    )

    def _stub_spin(route_key: str, stop, *, color=None, verbosity: int = 1) -> None:
        writes.append("\r" + f"NEXUS | @{route_key} thinking…⠋ ")
        stop.wait(5.0)
        writes.append("\r" + " " * 32 + "\r")

    monkeypatch.setattr(backends, "_spin_stderr", _stub_spin)

    cfg = {"backend": "opencode", "model": "opencode/mock-free", "quota_tier": "FREE", "timeout_s": 5}
    res = backends.execute_route(
        "tr_spin01",
        "code",
        cfg,
        "hi",
        events_path=tmp_path / "events.jsonl",
        dry_run=False,
        color=True,
    )
    assert res.status == "DONE"

    frame_idx = next(i for i, w in enumerate(writes) if "thinking" in w)
    done_idx = next(i for i, w in enumerate(writes) if "✓" in w)
    clear_idx = max(i for i, w in enumerate(writes[:done_idx]) if set(w) <= {"\r", " "})
    assert frame_idx < clear_idx < done_idx, "done receipt printed before the spinner was cleared"

def test_cli_quiet_and_verbose_verbosity(tmp_path: Path) -> None:
    """--quiet keeps stderr silent; --verbose adds rationale, intent, prompt and trace."""
    bin_path = PROJECT_ROOT / "bin" / "nexus"
    base = [sys.executable, str(bin_path), "--no-color"]

    quiet = subprocess.run(
        base + ["--quiet", "run", "--dry-run", "-r", "code", "hello"],
        capture_output=True, text=True,
    )
    assert quiet.returncode == 0, quiet.stderr
    assert quiet.stderr.strip() == "", f"quiet mode leaked status lines: {quiet.stderr!r}"
    assert "[dry-run]" in quiet.stdout

    verbose = subprocess.run(
        base + ["--verbose", "run", "--dry-run", "-r", "code", "hello"],
        capture_output=True, text=True,
    )
    assert verbose.returncode == 0, verbose.stderr
    for needle in ("front-desk → @code", "reason:", "spawn @code", "intent:", "prompt:", "trace: tr_"):
        assert needle in verbose.stderr, f"verbose output missing {needle!r}: {verbose.stderr!r}"
    assert "total" in verbose.stderr


def test_cli_quiet_suppresses_tree(tmp_path: Path) -> None:
    """--quiet without --dry-run keeps the delegation tree off stderr."""
    bin_path = PROJECT_ROOT / "bin" / "nexus"
    proc = subprocess.run(
        [sys.executable, str(bin_path), "--no-color", "--quiet", "run", "--dry-run", "-r", "code", "hi"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "NEXUS DELEGATION TREE" not in proc.stderr


def test_cli_council_verbose_header_and_footer(tmp_path: Path) -> None:
    """Council prints the seat plan up front and a seat/token footer after the tree."""
    bin_path = PROJECT_ROOT / "bin" / "nexus"
    proc = subprocess.run(
        [sys.executable, str(bin_path), "--no-color", "council", "--dry-run", "compare stores"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "council · 3 seats · pioneer→fast-code" in proc.stderr
    assert "red_teamer→free-reason" in proc.stderr
    assert "synthesizer→refactor" in proc.stderr
    assert "council · 3/3 seats" in proc.stderr
    assert "$0 premium" in proc.stderr


def test_execute_route_fail_and_timeout_receipts(monkeypatch, tmp_path: Path) -> None:
    """Failures and timeouts get the same one-line receipt as successes."""
    import nexus.backends as backends

    writes: list[str] = []
    monkeypatch.setattr(sys, "stderr", _fake_tty_stderr(writes))

    cfg = {"backend": "opencode", "model": "opencode/mock-free", "quota_tier": "FREE", "timeout_s": 5}

    monkeypatch.setattr(backends, "_run_subprocess_bounded", lambda *a, **k: (2, "", "boom: bad pin", False))
    monkeypatch.setattr(
        backends, "_parse_opencode_json_stream", lambda raw: backends.OpenCodeStream("", 1, 0, 0, "complete", "")
    )
    res = backends.execute_route(
        "tr_fail01", "code", cfg, "hi", events_path=tmp_path / "e1.jsonl", color=True
    )
    assert res.status == "FAIL"
    joined = "".join(writes)
    assert "✗" in joined and "@code" in joined and "boom: bad pin" in joined

    writes.clear()
    monkeypatch.setattr(backends, "_run_subprocess_bounded", lambda *a, **k: (0, "", "", True))
    res = backends.execute_route(
        "tr_timeout01", "code", cfg, "hi", events_path=tmp_path / "e2.jsonl", color=True
    )
    assert res.status == "TIMEOUT"
    joined = "".join(writes)
    assert "⏱" in joined and "Timed out after 5s" in joined


def test_humanize_error_extracts_json_messages() -> None:
    """Backend error blobs collapse to one readable line."""
    from nexus.backends import _humanize_error

    assert _humanize_error("") == ""
    assert _humanize_error("plain failure") == "plain failure"
    assert _humanize_error("line one\nline two") == "line one"
    nested = '{"type":"error","error":{"type":"provider.no-route","data":{"message":"Model unavailable"}}}'
    assert _humanize_error(nested) == "Model unavailable"
    flat = '{"type":"error","message":"rate limited"}'
    assert _humanize_error(flat) == "rate limited"
    assert _humanize_error('{"type":"error","error":{"type":"provider.x"}}') == "provider.x"
    assert _humanize_error("x" * 400, limit=10) == "x" * 10


def test_execute_route_quiet_emits_no_receipt(monkeypatch, tmp_path: Path) -> None:
    """verbosity=0 suppresses the receipt entirely."""
    import nexus.backends as backends

    writes: list[str] = []
    monkeypatch.setattr(sys, "stderr", _fake_tty_stderr(writes))
    monkeypatch.setattr(backends, "_run_subprocess_bounded", lambda *a, **k: (0, "", "", False))
    monkeypatch.setattr(
        backends, "_parse_opencode_json_stream", lambda raw: backends.OpenCodeStream("ok", 3, 1, 0, "complete", "")
    )

    cfg = {"backend": "opencode", "model": "opencode/mock-free", "quota_tier": "FREE", "timeout_s": 5}
    res = backends.execute_route(
        "tr_quiet01", "code", cfg, "hi", events_path=tmp_path / "eq.jsonl", color=True, verbosity=0
    )
    assert res.status == "DONE"
    assert "".join(writes) == ""


def _load_cli_module():
    import importlib.machinery
    import importlib.util

    bin_path = PROJECT_ROOT / "bin" / "nexus"
    loader = importlib.machinery.SourceFileLoader("nexus_cli_mod", str(bin_path))
    spec = importlib.util.spec_from_loader("nexus_cli_mod", loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repl_banner_is_rectangular(capsys) -> None:
    """Every banner line is the same visible width and nothing is truncated mid-word."""
    cli = _load_cli_module()
    for level in (0, 1, 2):
        capsys.readouterr()
        cli.print_repl_banner("opencode/mock-model-free", color=False, verbosity=level)
        out = capsys.readouterr().out.strip("\n")
        lines = out.splitlines()
        widths = {visible_len(line) for line in lines}
        assert len(widths) == 1, f"ragged banner at verbosity={level}: {widths}"
        assert all(line.endswith(("┐", "│", "┤", "┘")) for line in lines)
        assert "..." not in out
    capsys.readouterr()
    cli.print_repl_banner("opencode/mock-model-free", color=False, verbosity=2)
    assert "Status output: verbose" in capsys.readouterr().out


def test_repl_quiet_mode_suppresses_tree(tmp_path: Path) -> None:
    """`/quiet` in the REPL stops status lines and the tree from being printed."""
    bin_path = PROJECT_ROOT / "bin" / "nexus"
    transcript = "/quiet\n@code hello\n/quit\n"
    proc = subprocess.run(
        [sys.executable, str(bin_path), "--no-color", "--dry-run"],
        input=transcript,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    after_banner = proc.stdout.split("┘", 1)[1]
    assert "Verbosity -> 0" in after_banner
    assert "NEXUS DELEGATION TREE" not in after_banner
    assert "front-desk → @code" not in after_banner
    assert "[dry-run] in-charge:code" in after_banner


def test_parse_opencode_stream_sums_steps_and_includes_cache() -> None:
    """Token accounting sums step-finish usage and folds cache reads into the prompt size."""
    import json as _json

    from nexus.backends import _parse_opencode_json_stream

    lines = [
        {"type": "text", "part": {"type": "text", "text": "hello "}},
        {"type": "text", "part": {"type": "text", "text": "world"}},
        {"type": "step-finish", "part": {"type": "step-finish", "tokens": {
            "input": 100, "output": 10, "reasoning": 5, "cache": {"read": 900}}}},
        {"type": "step-finish", "part": {"type": "step-finish", "tokens": {
            "input": 50, "output": 20, "cache": {"read": 1000}}}},
    ]
    stream = _parse_opencode_json_stream("\n".join(_json.dumps(x) for x in lines))
    assert stream.text == "hello world"
    assert stream.tokens_in == 100 + 900 + 50 + 1000, "cache reads must count toward prompt size"
    assert stream.tokens_out == 10 + 5 + 20
    assert stream.tokens_cached == 1900


def test_parse_opencode_stream_falls_back_without_step_finish() -> None:
    """Older streams without step-finish still report the peak usage they do carry."""
    import json as _json

    from nexus.backends import _parse_opencode_json_stream

    lines = [
        {"type": "message", "part": {"type": "message", "tokens": {"input": 10, "output": 2}}},
        {"type": "message", "part": {"type": "message", "tokens": {"input": 40, "output": 3, "cache": {"read": 60}}}},
    ]
    stream = _parse_opencode_json_stream("\n".join(_json.dumps(x) for x in lines))
    assert stream.tokens_in == 100
    assert stream.tokens_out == 3
    assert stream.tokens_cached == 60


def test_render_delegation_tree_fits_width_and_keeps_errors(tmp_path: Path) -> None:
    """Tree rows never exceed the frame, and long errors move to a continuation row."""
    events_file = tmp_path / "events.jsonl"
    trace_id = "tr_width01"
    long_err = "Model unavailable: opencode/this-model-id-is-far-too-long-to-fit-inline-anywhere"

    emit_event(
        trace_id, "sp_root", parent_id=None, route="entrypoint", backend="nexus", model="nexus-router",
        quota_tier="ORCHESTRATOR", status="DONE",
        prompt="audit the token accounting path for cache reads and multi-step finishes end to end",
        tokens_in=14443, tokens_out=11, elapsed_ms=908, path=events_file,
    )
    emit_event(
        trace_id, "sp_child", parent_id="sp_root", route="in-charge:code", backend="opencode",
        model="opencode/deepseek-v4.1-flash-free", quota_tier="FREE", status="DONE",
        step="tool:shell", tokens_in=4820, tokens_out=227, elapsed_ms=4400, path=events_file,
    )
    emit_event(
        trace_id, "sp_fail", parent_id="sp_root", route="red_teamer:free-reason", backend="opencode",
        model="opencode/flint-40b-free", quota_tier="FREE", status="FAIL", error=long_err,
        elapsed_ms=1840, path=events_file,
    )

    events = read_events(path=events_file, trace_id=trace_id)
    for width in (96, 104, 120):
        out = render_delegation_tree(events, trace_id=trace_id, color=False, width=width)
        for idx, line in enumerate(out.splitlines()):
            assert visible_len(line) == width, f"width={width} line {idx}: {visible_len(line)}: {line!r}"
        assert "…" in out, "a cut prompt/detail must be marked with an ellipsis"
        assert "err: Model unavailable" in out, "the error must survive width pressure"
        assert "4.8k→227 tok" in out, "token counts must stay compact"
        assert "4.4s" in out

    # The error continuation row sits directly under the failing span.
    plain = render_delegation_tree(events, trace_id=trace_id, color=False, width=104).splitlines()
    fail_idx = next(i for i, l in enumerate(plain) if "red_teamer" in l)
    assert "err: Model unavailable" in plain[fail_idx + 1]


def test_execute_route_preflight_missing_binary(monkeypatch, tmp_path: Path) -> None:
    """A missing harness binary fails fast with an actionable message, not a traceback."""
    import nexus.backends as backends

    writes: list[str] = []
    monkeypatch.setattr(sys, "stderr", _fake_tty_stderr(writes))
    monkeypatch.setattr(backends.shutil, "which", lambda name: None)

    def _explode(*_a, **_k):
        raise AssertionError("subprocess must not run when the binary is missing")

    monkeypatch.setattr(backends, "_run_subprocess_bounded", _explode)

    cfg = {"backend": "opencode", "model": "opencode/mock-free", "quota_tier": "FREE", "timeout_s": 5}
    res = backends.execute_route("tr_pre01", "code", cfg, "hi", events_path=tmp_path / "e.jsonl", color=True)
    assert res.status == "FAIL"
    assert "not found on PATH" in res.error
    joined = "".join(writes)
    assert "✗" in joined and "not found on PATH" in joined

    events = read_events(path=tmp_path / "e.jsonl")
    assert any(ev.get("status") == "FAIL" and "preflight" in str(ev.get("step")) for ev in events)


def test_council_cards_use_panels_with_status_footer(tmp_path: Path) -> None:
    """Council verdict cards are width-safe panels with status/token footers."""
    bin_path = PROJECT_ROOT / "bin" / "nexus"
    proc = subprocess.run(
        [sys.executable, str(bin_path), "--no-color", "council", "--dry-run", "compare stores"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    cards = [line for line in proc.stdout.splitlines() if line.startswith("└─ ✓ DONE")]
    assert len(cards) == 3, f"expected 3 seat footers, got {cards}"
    assert all("tok" in card for card in cards)
    widths = {visible_len(line) for line in proc.stdout.splitlines() if line.startswith(("┌─", "│", "└─"))}
    assert len(widths) == 1, f"council cards are ragged: {widths}"


def test_run_subprocess_bounded_times_out_a_silent_child() -> None:
    """A child that prints nothing must still be killed at the deadline (no hang)."""
    import time as _time

    from nexus.backends import _run_subprocess_bounded

    started = _time.monotonic()
    rc, out, err, timed_out = _run_subprocess_bounded(
        [sys.executable, "-c", "import time; time.sleep(10)"], 0.5
    )
    elapsed = _time.monotonic() - started
    assert timed_out is True
    assert rc != 0
    assert out == ""
    assert elapsed < 4.0, f"timeout took {elapsed:.1f}s"


def test_run_subprocess_bounded_drains_large_stderr() -> None:
    """A child flooding stderr past the pipe buffer must not deadlock the reader."""
    from nexus.backends import _run_subprocess_bounded

    script = "import sys; sys.stderr.write('x' * 600000); sys.stdout.write('done')"
    rc, out, err, timed_out = _run_subprocess_bounded([sys.executable, "-c", script], 20)
    assert timed_out is False
    assert rc == 0
    assert "done" in out
    assert len(err) >= 600000


def test_run_subprocess_bounded_streams_stdout_lines() -> None:
    """on_stdout_line fires per line while the child runs."""
    from nexus.backends import _run_subprocess_bounded

    seen: list[str] = []
    script = "print('a'); print('b'); print('c')"
    rc, out, err, timed_out = _run_subprocess_bounded([sys.executable, "-c", script], 20, seen.append)
    assert rc == 0 and not timed_out
    assert [line.strip() for line in seen] == ["a", "b", "c"]
    assert "a" in out and "c" in out


def test_cli_global_flags_survive_subcommand_parsing(tmp_path: Path) -> None:
    """`nexus --dry-run run ...` must stay a dry run — subparsers cannot clobber globals."""
    bin_path = PROJECT_ROOT / "bin" / "nexus"

    proc = subprocess.run(
        [sys.executable, str(bin_path), "--no-color", "--dry-run", "run", "-r", "code", "hi"],
        capture_output=True, text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "[dry-run]" in proc.stdout, "global --dry-run was ignored; the backend was invoked"

    proc = subprocess.run(
        [sys.executable, str(bin_path), "--no-color", "-r", "triage", "--dry-run", "run", "hi"],
        capture_output=True, text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "front-desk → @triage" in proc.stderr, "global -r was ignored"


def test_cli_rejects_empty_prompt() -> None:
    """An empty prompt must not dispatch a real backend call."""
    bin_path = PROJECT_ROOT / "bin" / "nexus"
    for sub in ("run", "council"):
        proc = subprocess.run(
            [sys.executable, str(bin_path), "--no-color", "--dry-run", sub, "   "],
            capture_output=True, text=True,
        )
        assert proc.returncode == 2, f"{sub}: {proc.returncode}"
        assert "needs a prompt" in proc.stderr
        assert proc.stdout.strip() == ""


def test_repl_autocomplete_and_ctrl_c_handling(tmp_path: Path, monkeypatch) -> None:
    import importlib.machinery
    import importlib.util

    bin_path = PROJECT_ROOT / "bin" / "nexus"
    loader = importlib.machinery.SourceFileLoader("nexus_cli_mod", str(bin_path))
    spec = importlib.util.spec_from_loader("nexus_cli_mod", loader)
    assert spec is not None and spec.loader is not None
    nexus_cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nexus_cli)

    routes = ["triage", "code", "fast-code", "refactor", "reason", "free-reason", "vision"]

    # 1. Slash command autocomplete at start of line
    sw_matches = nexus_cli.complete_repl_candidates("/sw", "/sw", 0, routes)
    assert sw_matches == ["/switch "]

    rel_matches = nexus_cli.complete_repl_candidates("/re", "/re", 0, routes)
    assert "/release" in rel_matches

    # 2. @route autocomplete at start of line or mid-line
    at_matches = nexus_cli.complete_repl_candidates("@fr", "@fr", 0, routes)
    assert at_matches == ["@free-reason "]

    mid_matches = nexus_cli.complete_repl_candidates("please ask @vi", "@vi", 11, routes)
    assert mid_matches == ["@vision "]

    # 3. Route argument autocomplete after `/switch `
    arg_matches = nexus_cli.complete_repl_candidates("/switch ref", "ref", 8, routes)
    assert arg_matches == ["refactor"]

    # 4. Single Ctrl-C at REPL prompt clears line and stays in REPL; second consecutive Ctrl-C exits
    inputs = iter([KeyboardInterrupt(), "/routes", KeyboardInterrupt(), KeyboardInterrupt()])

    def _fake_input(_prompt: str = "") -> str:
        val = next(inputs)
        if isinstance(val, BaseException):
            raise val
        return str(val)

    monkeypatch.setattr("builtins.input", _fake_input)
    events_file = tmp_path / "events.jsonl"
    rc = nexus_cli.run_repl(events_path=events_file, color=False, dry_run=True)
    assert rc == 0

    # 5. Ctrl-C inside execute_route records FAIL / interrupted span instead of crashing
    from nexus.backends import execute_route
    import nexus.backends as backends_mod

    def _interrupt_subproc(*_args, **_kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(backends_mod, "_run_subprocess_bounded", _interrupt_subproc)
    res = execute_route(
        "tr_sigint01",
        "code",
        {"backend": "opencode", "model": "opencode/deepseek-v4.1-flash-free", "quota_tier": "FREE"},
        "test prompt",
        events_path=events_file,
        dry_run=False,
    )
    assert res.status == "INTERRUPTED"
    evs = read_events(path=events_file, trace_id="tr_sigint01")
    assert evs[-1]["status"] == "FAIL"
    assert "interrupted (Ctrl-C)" in evs[-1]["step"]


def test_cli_post_subcommand_quiet_and_no_color(tmp_path: Path) -> None:
    """-q and --no-color work after the subcommand, not only before it."""
    bin_path = PROJECT_ROOT / "bin" / "nexus"

    proc = subprocess.run(
        [sys.executable, str(bin_path), "run", "-q", "--no-color", "--dry-run", "hello"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr.strip() == "", f"post-subcommand -q leaked status lines: {proc.stderr!r}"
    assert "[dry-run]" in proc.stdout

    proc = subprocess.run(
        [sys.executable, str(bin_path), "council", "--no-color", "--dry-run", "compare stores"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "\x1b[" not in proc.stderr, f"post-subcommand --no-color leaked ANSI: {proc.stderr!r}"
    assert "PIONEER" in proc.stdout


def test_cli_think_and_no_fallback_plumb_to_execute_route(monkeypatch, tmp_path: Path) -> None:
    """`--think off` and `--no-fallback` reach execute_route through main()."""
    cli = _load_cli_module()
    from nexus.backends import ExecutionResult

    captured: dict = {}

    def _fake_execute_route(*_args, **kwargs):
        captured.update(kwargs)
        return ExecutionResult("tr_think01", "sp", "code", "opencode", "mock", "FREE", "DONE", "ok")

    monkeypatch.setattr(cli, "execute_route", _fake_execute_route)

    rc = cli.main(
        [
            "--events-file", str(tmp_path / "events.jsonl"),
            "--no-color",
            "run", "--think", "off", "--no-fallback", "--dry-run",
            "-r", "code", "hello",
        ]
    )
    assert rc == 0
    assert captured["think"] == "off"
    assert captured["allow_fallback"] is False


def test_cli_doctor_reports_routes_local_endpoints_and_state(monkeypatch, capsys, tmp_path: Path) -> None:
    """`nexus doctor` reports binaries, pins, local endpoints, and state file stats."""
    cli = _load_cli_module()

    events_file = tmp_path / "events.jsonl"
    events_file.write_text("\n".join(f'{{"i":{i}}}' for i in range(5)) + "\n", encoding="utf-8")
    sessions_file = tmp_path / "tui_sessions.json"
    sessions_file.write_text(
        json.dumps({"sessions": [
            {"id": "tui_a", "title": "one", "created_at": 1.0, "updated_at": 1.0},
            {"id": "tui_b", "title": "two", "created_at": 2.0, "updated_at": 2.0},
        ]}),
        encoding="utf-8",
    )
    board_file = tmp_path / "BOARD.md"
    board_file.write_text("line one\n", encoding="utf-8")
    square_file = tmp_path / "square.jsonl"
    square_file.write_text("a\nb\n", encoding="utf-8")

    monkeypatch.setattr(cli, "DEFAULT_SESSIONS_PATH", sessions_file)
    monkeypatch.setattr(cli, "DEFAULT_BOARD_PATH", board_file)
    monkeypatch.setattr(cli, "DEFAULT_SQUARE_PATH", square_file)
    monkeypatch.setattr(
        cli.shutil, "which",
        lambda name: f"/usr/bin/{name}" if name in ("opencode", "agy") else None,
    )
    monkeypatch.setattr(cli, "validate_routes", lambda *a, **k: (True, []))
    monkeypatch.setattr(
        cli, "probe_local_endpoints",
        lambda *a, **k: {
            "local": {
                "backend": "openai-compat", "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
                "model": "local-model", "reachable": True, "model_present": True, "detail": "",
            },
            "ollama": {
                "backend": "ollama", "endpoint": "http://127.0.0.1:11434/api/chat",
                "model": "llama3.2:latest", "reachable": False,
                "model_present": False, "detail": "connection refused",
            },
        },
    )

    rc = cli.main(["--events-file", str(events_file), "--no-color", "doctor"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "opencode: /usr/bin/opencode" in out
    assert "verified" in out
    assert "@local [ONLINE]" in out and "model_present=yes" in out
    assert "@ollama [OFFLINE]" in out and "model_present=no" in out
    assert "2 session(s)" in out
    assert "5 line(s)" in out
    assert "doctor PASS" in out


def test_cli_doctor_prune_events_trims_to_keep_lines(monkeypatch, capsys, tmp_path: Path) -> None:
    """`nexus doctor --prune-events` trims events.jsonl and reports before -> after."""
    cli = _load_cli_module()
    events_file = tmp_path / "events.jsonl"
    events_file.write_text("\n".join(f'{{"i":{i}}}' for i in range(10)) + "\n", encoding="utf-8")

    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(cli, "validate_routes", lambda *a, **k: (True, []))
    monkeypatch.setattr(cli, "probe_local_endpoints", lambda *a, **k: {})

    rc = cli.main(
        [
            "--events-file", str(events_file), "--no-color",
            "doctor", "--prune-events", "--keep-lines", "3",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "events pruned: 10 -> 3 lines" in out
    assert len(events_file.read_text(encoding="utf-8").splitlines()) == 3


def test_cli_routes_validate_local_passes_check_local(monkeypatch, capsys) -> None:
    """`nexus routes --validate --local` asks validate_routes to probe local endpoints."""
    cli = _load_cli_module()
    seen: dict = {}

    def _fake_validate(_data, **kwargs):
        seen.update(kwargs)
        return True, []

    monkeypatch.setattr(cli, "validate_routes", _fake_validate)
    rc = cli.main(["--no-color", "routes", "--validate", "--local"])
    assert rc == 0
    assert seen["check_local"] is True
    assert seen["refresh_cache"] is True


def test_complete_repl_candidates_think() -> None:
    """`/think` completes the command and its effort levels."""
    cli = _load_cli_module()
    routes = ["code", "reason"]
    assert cli.complete_repl_candidates("/thi", "/thi", 0, routes) == ["/think"]
    assert cli.complete_repl_candidates("/think ", "", 7, routes) == ["off", "low", "medium", "high"]
    assert cli.complete_repl_candidates("/think h", "h", 8, routes) == ["high"]


def test_repl_think_command_reports_and_resets(monkeypatch, capsys, tmp_path: Path) -> None:
    """`/think` with no argument reports the current effort; `default` clears it."""
    cli = _load_cli_module()
    inputs = iter(["/think", "/think high", "/think", "/think default", "/think", "/quit"])
    prompts: list[str] = []

    def _fake_input(prompt: str = "") -> str:
        prompts.append(prompt)
        return next(inputs)

    monkeypatch.setattr("builtins.input", _fake_input)
    monkeypatch.setattr(cli, "_setup_readline", lambda keys: None)

    rc = cli.run_repl(events_path=tmp_path / "events.jsonl", color=False, dry_run=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "Thinking effort -> default" in out
    assert "Thinking effort -> high" in out
    assert "Thinking effort -> default (backend decides)" in out
    assert any("think:high" in prompt for prompt in prompts)


def test_repl_forwards_history_and_updates_fallback_model(monkeypatch, capsys, tmp_path: Path) -> None:
    """Local routes get bounded history; a fallback model updates the prompt."""
    cli = _load_cli_module()
    from nexus.backends import ExecutionResult

    inputs = iter(["/think high", "/switch local", "hello", "again", "/quit"])
    prompts: list[str] = []

    def _fake_input(prompt: str = "") -> str:
        prompts.append(prompt)
        return next(inputs)

    recorded: list[tuple[str, object, list[dict]]] = []

    def _fake_dispatch(prompt: str, **kwargs):
        recorded.append((prompt, kwargs.get("think"), list(kwargs.get("history_messages") or [])))
        model = "local-model-fallback" if len(recorded) == 2 else "local-model"
        return 0, None, ExecutionResult(
            "tr_repl01", "sp", "local", "openai-compat", model, "LOCAL", "DONE", f"answer {len(recorded)}"
        )

    monkeypatch.setattr("builtins.input", _fake_input)
    monkeypatch.setattr(cli, "_setup_readline", lambda keys: None)
    monkeypatch.setattr(cli, "dispatch_with_front_desk", _fake_dispatch)

    rc = cli.run_repl(events_path=tmp_path / "events.jsonl", color=False, dry_run=True)
    assert rc == 0
    assert len(recorded) == 2
    assert recorded[0][1] == "high" and recorded[1][1] == "high"
    assert recorded[0][2] == []
    assert recorded[1][2] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "answer 1"},
    ]
    assert any("think:high" in prompt for prompt in prompts)
    assert any("local-model-fallback" in prompt for prompt in prompts)


def test_cli_run_out_saves_dry_run_artifact(tmp_path: Path) -> None:
    """`nexus run -o <path>` writes the run response cleanly to the target path."""
    bin_path = PROJECT_ROOT / "bin" / "nexus"
    out_file = tmp_path / "sub" / "result.txt"
    proc = subprocess.run(
        [
            sys.executable,
            str(bin_path),
            "--no-color",
            "run",
            "--dry-run",
            "-r",
            "code",
            "-o",
            str(out_file),
            "hello",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_file.is_file()
    assert "[dry-run]" in out_file.read_text(encoding="utf-8")
    assert "saved " in proc.stderr


def test_repl_save_command_and_completion(monkeypatch, capsys, tmp_path: Path) -> None:
    """REPL `/save` extracts the largest fenced code block into NEXUS_DROP_DIR with +x."""
    cli = _load_cli_module()
    from nexus.backends import ExecutionResult

    drop_root = tmp_path / "drops"
    monkeypatch.setenv("NEXUS_DROP_DIR", str(drop_root))
    assert "/save" in cli.complete_repl_candidates("/sa", "/sa", 0, ["code", "ollama"])

    inputs = iter(["write a script", "/save backup.sh", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
    monkeypatch.setattr(cli, "_setup_readline", lambda keys: None)

    sample_reply = (
        "Here is your script:\n```bash\n#!/bin/sh\necho synced\n```\n"
        "Run:\n```bash\n./backup.sh\n```"
    )
    monkeypatch.setattr(
        cli,
        "dispatch_with_front_desk",
        lambda *_a, **_k: (
            0,
            None,
            ExecutionResult("tr_save01", "sp", "ollama", "ollama", "llama3.2", "LOCAL", "DONE", sample_reply),
        ),
    )

    rc = cli.run_repl(events_path=tmp_path / "events.jsonl", color=False, dry_run=True)
    out = capsys.readouterr().out
    assert rc == 0
    saved = drop_root / "backup.sh"
    assert saved.read_text(encoding="utf-8") == "#!/bin/sh\necho synced\n"
    assert (saved.stat().st_mode & 0o777) == 0o755
    assert "Saved " in out and "(+x)" in out


def test_cli_profile_stats_bench_unified_council_and_sim(tmp_path: Path, monkeypatch) -> None:
    """Exercise profile, stats, bench, unified council, sim, and evolve CLI subcommands."""
    bin_path = PROJECT_ROOT / "bin" / "nexus"
    events_file = tmp_path / "events.jsonl"
    sim_file = tmp_path / "sim.json"
    env = {"HOME": str(tmp_path), "PATH": os.environ.get("PATH", ""), "NEXUS_SIM_PATH": str(sim_file)}

    prof_list = subprocess.run(
        [sys.executable, str(bin_path), "--no-color", "profile", "list", "--json"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    names = {item["name"] for item in json.loads(prof_list.stdout)}
    assert {"default", "free-only", "local-only", "ensemble"} <= names

    prof_show = subprocess.run(
        [sys.executable, str(bin_path), "--no-color", "profile", "show", "free-only"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert "PROFILE | name=free-only" in prof_show.stdout

    prof_init = subprocess.run(
        [
            sys.executable,
            str(bin_path),
            "--no-color",
            "profile",
            "init",
            "smoke-prof",
            "--from",
            "free-only",
            "--mode",
            "entity",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert "initialized smoke-prof" in prof_init.stdout
    assert (tmp_path / ".config" / "nexus" / "profiles" / "smoke-prof.json").is_file()

    subprocess.run(
        [
            sys.executable,
            str(bin_path),
            "--events-file",
            str(events_file),
            "--profile",
            "free-only",
            "--dry-run",
            "--no-color",
            "Write a Python helper",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    stats_proc = subprocess.run(
        [sys.executable, str(bin_path), "--events-file", str(events_file), "--no-color", "stats", "--by", "profile", "--json"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    stats_payload = json.loads(stats_proc.stdout)
    assert stats_payload.get("excluded_dry_run", 0) >= 1
    assert "free-only" not in stats_payload.get("groups", {})

    stats_all_proc = subprocess.run(
        [sys.executable, str(bin_path), "--events-file", str(events_file), "--no-color", "stats", "--by", "profile", "--include-dry-run", "--json"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    stats_all_payload = json.loads(stats_all_proc.stdout)
    assert "free-only" in stats_all_payload.get("groups", {})

    bench_proc = subprocess.run(
        [
            sys.executable,
            str(bin_path),
            "--no-color",
            "bench",
            "--profiles",
            "default,free-only",
            "--split",
            "held-out",
            "--json",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    bench_rows = json.loads(bench_proc.stdout)
    assert {r["profile"] for r in bench_rows} == {"default", "free-only"}

    unified_proc = subprocess.run(
        [
            sys.executable,
            str(bin_path),
            "--events-file",
            str(events_file),
            "--no-color",
            "council",
            "--unified",
            "--dry-run",
            "Compare SQLite WAL vs JSONL",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert "[dry-run]" in unified_proc.stdout
    assert "[council // PIONEER]" not in unified_proc.stdout

    subprocess.run(
        [sys.executable, str(bin_path), "--no-color", "sim", "init"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    val_proc = subprocess.run(
        [sys.executable, str(bin_path), "--no-color", "sim", "validate"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert "SIM_AUDIT | PASS" in val_proc.stdout
    ev_proc = subprocess.run(
        [sys.executable, str(bin_path), "--no-color", "evolve", "--dry-run", "--rounds", "1"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert "EVOLVE | round=" in ev_proc.stdout

