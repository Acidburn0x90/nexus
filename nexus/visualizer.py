"""Hierarchical ANSI delegation tree visualizer for Nexus."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import sys
import time
from typing import Optional

from nexus.events import DEFAULT_EVENTS_PATH, normalize_quota_tier, read_events, reconstruct_tree

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
CYAN = "\033[36m"
BLUE = "\033[34m"
YELLOW = "\033[33m"
RED = "\033[31m"
BG_RED = "\033[41;97;1m"

STATUS_ICONS = {
    "SPAWN": f"{YELLOW}○{RESET}",
    "RUNNING": f"{CYAN}⠋{RESET}",
    "DONE": f"{GREEN}✓{RESET}",
    "FAIL": f"{RED}✗{RESET}",
    "TIMEOUT": f"{RED}⏱{RESET}",
}

QUOTA_BADGES = {
    "FREE": f"{GREEN}[FREE $0]{RESET}",
    "GO": f"{CYAN}[GO]{RESET}",
    "PLAN": f"{CYAN}[PLAN]{RESET}",
    "LOCAL": f"{BLUE}[LOCAL $0]{RESET}",
    "ORCHESTRATOR": f"{DIM}[ROUTER]{RESET}",
    "PREMIUM": f"{BG_RED}[PREMIUM]{RESET}",
}

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def strip_ansi(s: str) -> str:
    """Strip ANSI escape sequences from a string."""
    return _ANSI_RE.sub("", s)


def visible_len(s: str) -> int:
    """Return the visible character width of a string disregarding ANSI escape sequences."""
    return len(strip_ansi(s))


def truncate_ansi(s: str, max_width: int, ellipsis: str = "") -> str:
    """Truncate a string to max visible length while preserving ANSI style sequences."""
    if max_width <= 0:
        return ""
    if visible_len(s) <= max_width:
        return s

    el_len = visible_len(ellipsis)
    target_len = max_width - el_len
    if target_len <= 0:
        return ellipsis[:max_width]

    out: list[str] = []
    curr = 0
    styled = False
    tokens = re.split(r"(\x1b\[[0-9;]*[a-zA-Z])", s)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("\x1b["):
            out.append(tok)
            styled = True
        else:
            rem = target_len - curr
            if len(tok) <= rem:
                out.append(tok)
                curr += len(tok)
            else:
                out.append(tok[:rem])
                curr += rem
                break
    if ellipsis:
        out.append(ellipsis)
    if styled:
        out.append(RESET)
    return "".join(out)


def pad_ansi(s: str, width: int, align: str = "left") -> str:
    """Pad or truncate a string to an exact visible character width, ANSI-aware."""
    vlen = visible_len(s)
    if vlen > width:
        return truncate_ansi(s, width)
    if vlen == width:
        return s
    pad = " " * (width - vlen)
    if align == "right":
        return pad + s
    if align == "center":
        left_pad = (width - vlen) // 2
        right_pad = width - vlen - left_pad
        return (" " * left_pad) + s + (" " * right_pad)
    return s + pad


def render_panel(
    lines: list[str],
    *,
    title: Optional[str] = None,
    footer: Optional[str] = None,
    width: int = 80,
    border_color: str = "",
) -> str:
    """Render a sharp 0-gap skeletal wireframe box with aligned right borders."""
    reset = RESET if border_color else ""
    out: list[str] = []

    # Top border
    if title:
        v_title = visible_len(title)
        rem = max(0, width - v_title - 6)
        if v_title > width - 6:
            title = truncate_ansi(title, width - 6, "…")
            rem = 0
        out.append(f"{border_color}┌─{reset} {title} {border_color}{'─' * rem}─┐{reset}")
    else:
        out.append(f"{border_color}┌{'─' * (width - 2)}┐{reset}")

    # Content rows
    inner_width = max(1, width - 4)
    for line in lines:
        c = pad_ansi(line, inner_width)
        out.append(f"{border_color}│{reset} {c} {border_color}│{reset}")

    # Bottom border
    if footer:
        v_footer = visible_len(footer)
        rem = max(0, width - v_footer - 6)
        if v_footer > width - 6:
            footer = truncate_ansi(footer, width - 6, "…")
            rem = 0
        out.append(f"{border_color}└─{reset} {footer} {border_color}{'─' * rem}─┘{reset}")
    else:
        out.append(f"{border_color}└{'─' * (width - 2)}┘{reset}")

    return "\n".join(out)


def _strip_ansi(s: str) -> str:
    return strip_ansi(s)


def format_quota_badge(tier: str, color: bool = True) -> str:
    tier = normalize_quota_tier(tier)
    badge = QUOTA_BADGES.get(tier, f"[{tier}]")
    return badge if color else strip_ansi(badge)


def format_status_icon(status: str, color: bool = True) -> str:
    icon = STATUS_ICONS.get(status.upper(), "•")
    return icon if color else strip_ansi(icon)


STATUS_LINE_ICONS = {
    "info": ("◆", CYAN),
    "step": ("▸", YELLOW),
    "ok": ("✓", GREEN),
    "warn": ("!", YELLOW),
    "fail": ("✗", RED),
    "timeout": ("⏱", RED),
    "stop": ("⛔", YELLOW),
    "spin": ("⠋", CYAN),
    "cont": (" ", RESET),
}


def status_line(kind: str, text: str, *, color: bool = True, prefix: str = "NEXUS | ") -> str:
    """Render one `NEXUS | <icon> text` status line for stderr."""
    icon, code = STATUS_LINE_ICONS.get(kind, STATUS_LINE_ICONS["info"])
    if color:
        return f"{prefix}{code}{icon}{RESET} {text}"
    return f"{prefix}{icon} {text}"


def format_duration(ms: int) -> str:
    """Human-friendly latency: 412ms, 1.2s, 1m 04s."""
    ms = max(0, int(ms))
    if ms < 1000:
        return f"{ms}ms"
    if ms < 60_000:
        return f"{ms / 1000:.1f}s"
    minutes, seconds = divmod(ms // 1000, 60)
    return f"{minutes}m {seconds:02d}s"


def format_tokens(count: int) -> str:
    """Compact token counts: 812, 1.2k, 41.3k."""
    count = max(0, int(count))
    if count < 1000:
        return str(count)
    if count < 1_000_000:
        return f"{count / 1000:.1f}k"
    return f"{count / 1_000_000:.1f}M"


def render_delegation_tree(
    events: list[dict],
    trace_id: Optional[str] = None,
    *,
    color: bool = True,
    width: Optional[int] = None,
) -> str:
    """Render a hierarchical box-drawing delegation tree from span events."""
    tree = reconstruct_tree(events, trace_id=trace_id)
    spans: dict[str, dict] = tree["spans"]
    children_map: dict[Optional[str], list[str]] = tree["children"]
    roots: list[str] = tree["roots"]
    summary: dict = tree["summary"]
    active_trace = tree["trace_id"] or "none"

    if width is None:
        term_cols = shutil.get_terminal_size(fallback=(104, 30)).columns
        width = max(96, min(120, term_cols))

    premium_spans = summary.get("premium_spans", 0)
    non_premium = summary.get("non_premium_spans", 0)
    if premium_spans > 0:
        label = "span" if premium_spans == 1 else "spans"
        quota_summary = f"PREMIUM: {premium_spans} {label}"
        quota_colored = f"{BG_RED} {quota_summary} {RESET}" if color else quota_summary
        quota_footer = f"{BG_RED}Quota: PREMIUM ({premium_spans}){RESET}" if color else f"Quota: PREMIUM ({premium_spans})"
    else:
        label = "span" if non_premium == 1 else "spans"
        quota_summary = f"Zero-Cost: {non_premium} {label} ($0)"
        quota_colored = f"{GREEN}{quota_summary}{RESET}" if color else quota_summary
        quota_footer = f"{GREEN}Quota: $0 premium{RESET}" if color else "Quota: $0 premium"

    header_left = f"[1:tree] NEXUS DELEGATION TREE ── trace: {active_trace}"
    v_quota = visible_len(quota_colored)
    # Header, quota badge and frame share `width` columns. Narrow panes drop the
    # badge first, then clip the title, so the top line never outruns the box.
    if width - 9 - v_quota < 1:
        quota_colored = ""
        v_quota = 0
    max_left = max(1, width - (9 if v_quota else 8) - v_quota)
    if visible_len(header_left) > max_left:
        header_left = truncate_ansi(header_left, max_left, "…")
    v_left = visible_len(header_left)
    if v_quota:
        dashes_count = max(1, width - 8 - v_left - v_quota)
        top_line = f"┌─ {header_left} {'─' * dashes_count} {quota_colored} ─┐"
    else:
        dashes_count = max(1, width - 7 - v_left)
        top_line = f"┌─ {header_left} {'─' * dashes_count} ─┐"
    lines: list[str] = [top_line]

    if not spans:
        lines.append(f"│ {pad_ansi('(no delegation spans recorded yet)', width - 4)} │")
        lines.append(f"└{'─' * (width - 2)}┘")
        return "\n".join(lines)

    def _paint(code: str, text: str) -> str:
        return f"{code}{text}{RESET}" if color else text

    def _fit(text: str, budget: int) -> str:
        return truncate_ansi(text, budget, "…") if visible_len(text) > budget else text

    def _render_node(span_id: str, prefix: str, is_last: bool, is_root: bool = False) -> None:
        sp = spans[span_id]
        status = str(sp.get("status", "RUNNING")).upper()
        icon = format_status_icon(status, color=color)
        route = sp.get("route") or sp.get("intent") or "root"
        backend = sp.get("backend") or "nexus"
        model = sp.get("model") or "cli"
        tier = sp.get("quota_tier") or ("ORCHESTRATOR" if is_root else "FREE")
        badge = format_quota_badge(tier, color=color)
        elapsed_ms = int(sp.get("elapsed_ms", 0) or 0)
        tok_in = int(sp.get("tokens_in", 0) or 0)
        tok_out = int(sp.get("tokens_out", 0) or 0)
        step = sp.get("step") or ""
        prompt = sp.get("prompt") or ""
        error = sp.get("error") or ""

        connector = "● " if is_root else ("└─ " if is_last else "├─ ")
        route_label = f"{BOLD}[{route}]{RESET}" if color else f"[{route}]"
        target_label = f"{backend}:{model}" if backend != "nexus" else model
        time_str = format_duration(elapsed_ms)
        tok_str = f"{format_tokens(tok_in)}→{format_tokens(tok_out)} tok" if (tok_in or tok_out) else ""

        left = f"{prefix}{connector}{icon} {route_label} {target_label} {badge} {time_str}"
        if tok_str:
            left += f" {tok_str}"

        avail = width - 4
        if visible_len(left) > avail:
            left = truncate_ansi(left, avail, "…")

        detail = ""
        extra_row = ""
        if error and status in ("FAIL", "TIMEOUT"):
            # Errors need room to be actionable, so they get their own continuation
            # row instead of being crammed behind a long model id.
            first_err = " ".join(error.split())
            indent = " " * (visible_len(prefix) + 5)
            extra_row = f"{indent}{_paint(RED, 'err: ')}{_fit(first_err, max(12, avail - visible_len(indent) - 5))}"
        elif is_root and prompt:
            room = avail - visible_len(left) - 3
            if room >= 12:
                detail = f' "{_fit(" ".join(prompt.split()), room - 2)}"'
        elif step:
            room = avail - visible_len(left) - 4
            if room >= 10:
                detail = f" ── {_paint(DIM, _fit(step, room - 4))}"

        lines.append(f"│ {pad_ansi(left + detail, avail)} │")
        if extra_row:
            lines.append(f"│ {pad_ansi(extra_row, avail)} │")

        kids = children_map.get(span_id, [])
        child_prefix = prefix + ("" if is_root else ("   " if is_last else "│  "))
        for idx, kid_id in enumerate(kids):
            _render_node(kid_id, child_prefix, idx == len(kids) - 1, is_root=False)

    for r_idx, root_id in enumerate(roots):
        _render_node(root_id, "", r_idx == len(roots) - 1, is_root=True)

    tot_in = summary.get("tokens_in", 0)
    tot_out = summary.get("tokens_out", 0)
    tot_ms = summary.get("elapsed_ms", 0)
    tot_spans = summary.get("total_spans", 0)

    footer_content = (
        f"Spans: {tot_spans} │ Tokens: {format_tokens(tot_in)} in → {format_tokens(tot_out)} out"
        f" │ Total Elapsed: {format_duration(tot_ms)} │ {quota_footer}"
    )
    v_footer = visible_len(footer_content)
    rem = width - v_footer - 6
    if rem >= 0:
        footer_line = f"└─ {footer_content} {'─' * rem}─┘"
    else:
        footer_content_cut = truncate_ansi(footer_content, max(0, width - 6), "…")
        footer_line = f"└─ {footer_content_cut} ─┘"

    lines.append(footer_line)
    return "\n".join(lines)


def render_routes_table(
    routes_data: Optional[dict] = None,
    *,
    color: bool = True,
    width: Optional[int] = None,
) -> str:
    """Render an aligned 0-gap skeletal wireframe table of Nexus front-desk and specialist routes."""
    if routes_data is None:
        from nexus.router import load_routes
        routes_data = load_routes()

    routes = routes_data.get("routes", {})
    default_key = routes_data.get("default_route", "code")
    decision_model = routes_data.get("decision_model", "opencode/nemotron-3.5-lightning-free")

    if width is None:
        term_cols = shutil.get_terminal_size(fallback=(116, 30)).columns
        width = max(80, min(140, term_cols))

    route_w, backend_w, quota_w = 15, 8, 19
    # Below 100 columns the PURPOSE prose is dropped rather than every column being
    # squeezed into unreadable stubs; the model id is what people scan this table for.
    if width >= 100:
        model_w = 40
        purpose_w = max(12, width - (route_w + backend_w + model_w + quota_w + 6))
    else:
        model_w = max(24, min(40, width - (route_w + backend_w + quota_w + 5)))
        purpose_w = 0
    width = route_w + backend_w + model_w + quota_w + (6 if purpose_w else 5) + purpose_w

    def _cell(text: str, cell_width: int, ellipsis: str = "…") -> str:
        cut = truncate_ansi(text, cell_width, ellipsis) if visible_len(text) > cell_width else text
        return pad_ansi(cut, cell_width)

    def _row(cells: list[str]) -> str:
        return "│" + "│".join(cells) + "│"

    lines: list[str] = []

    # Top header bar
    title = "[2:routes] NEXUS FRONT-DESK & SPECIALIST ORCHESTRATOR TABLE"
    top_rem = max(0, width - visible_len(title) - 6)
    if visible_len(title) > width - 6:
        title = truncate_ansi(title, width - 6, "…")
        top_rem = 0
    lines.append(f"┌─ {title} {'─' * top_rem}─┐")

    def _seat(marker: str, key: str, backend: str, model: str, badge: str, desc: str) -> str:
        cells = [
            _cell(f" {marker} @{key}", route_w),
            _cell(backend, backend_w),
            _cell(model, model_w),
            _cell(badge, quota_w),
        ]
        if purpose_w:
            cells.append(_cell(f" {desc}", purpose_w))
        return _row(cells)

    # Front-Desk row
    fd_marker = f"{YELLOW}★{RESET}" if color else "★"
    fd_route_cell = _cell(f" {fd_marker} [front-desk]", route_w)
    fd_cells = [
        fd_route_cell,
        _cell("opencode", backend_w),
        _cell(decision_model, model_w),
        _cell(format_quota_badge("FREE", color=color), quota_w),
    ]
    if purpose_w:
        fd_cells.append(_cell(" Front-facing decision router", purpose_w))
    if color:
        fd_cells[1] = f"{DIM}{fd_cells[1]}{RESET}"
        fd_cells[2] = f"{BOLD}{fd_cells[2]}{RESET}"
    lines.append(_row(fd_cells))

    # Separator
    sep = [route_w, backend_w, model_w, quota_w] + ([purpose_w] if purpose_w else [])
    lines.append("├" + "┼".join("─" * w for w in sep) + "┤")

    for key, cfg in routes.items():
        marker = (f"{GREEN}*{RESET}" if color else "*") if key == default_key else " "
        lines.append(
            _seat(
                marker,
                key,
                cfg.get("backend", "opencode"),
                cfg.get("model", ""),
                format_quota_badge(cfg.get("quota_tier", "FREE"), color=color),
                cfg.get("description", ""),
            )
        )

    # Footer legend
    legend = (
        f"★ front-desk router   * default specialist (@{default_key})   --validate to audit pins"
        + ("" if purpose_w else "   (widen the terminal to see route purposes)")
    )
    bot_rem = max(0, width - visible_len(legend) - 6)
    if visible_len(legend) > width - 6:
        legend = truncate_ansi(legend, width - 6, "…")
        bot_rem = 0
    lines.append(f"└─ {legend} {'─' * bot_rem}─┘")

    return "\n".join(lines)


def watch_events(
    events_path: Path = DEFAULT_EVENTS_PATH,
    trace_id: Optional[str] = None,
    *,
    interval_s: float = 0.25,
    max_cycles: Optional[int] = None,
    color: bool = True,
) -> int:
    """Continuously render the latest delegation tree from events.jsonl."""
    cycles = 0
    last_rendered = ""
    is_tty = sys.stdout.isatty()
    keybind_strip = (
        f"{DIM}  [Ctrl-C] exit watch · [--once] snapshot · [--trace <id>] filter · [--interval <s>] poll{RESET}"
        if color
        else "  [Ctrl-C] exit watch · [--once] snapshot · [--trace <id>] filter · [--interval <s>] poll"
    )
    try:
        while True:
            evs = read_events(path=events_path, trace_id=trace_id, limit=250)
            rendered = render_delegation_tree(evs, trace_id=trace_id, color=color)
            if rendered != last_rendered:
                if is_tty and max_cycles is None:
                    sys.stdout.write("\033[2J\033[H")
                if max_cycles is None:
                    sys.stdout.write(rendered + "\n" + keybind_strip + "\n")
                else:
                    sys.stdout.write(rendered + "\n")
                sys.stdout.flush()
                last_rendered = rendered
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break
            time.sleep(interval_s)
    except KeyboardInterrupt:
        return 0
    return 0

