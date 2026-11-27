"""Curated dark Textual themes and quota-tier colour helpers for the Nexus TUI.

Five themes ship with the TUI. `nexus-charcoal` is the default: a monotone
charcoal palette. The others are drop-in alternatives (`oled-black`,
`opencode-dark`, `github-slate`, `catppuccin-mocha`) so the TUI can match
whatever the terminal is already wearing. Every theme is dark and defines the
same variable set, which keeps the skeletal `styles.tcss` valid under all five.
"""

from __future__ import annotations

from typing import Optional

from textual.theme import Theme

THEME_ORDER = [
    "nexus-charcoal",
    "oled-black",
    "opencode-dark",
    "github-slate",
    "catppuccin-mocha",
]

THEME_ALIASES: dict[str, str] = {"agy-slate": "github-slate"}

NEXUS_THEMES: dict[str, Theme] = {
    "nexus-charcoal": Theme(
        name="nexus-charcoal",
        background="#0b0b0d",
        surface="#131316",
        panel="#16161a",
        foreground="#f4f4f5",
        primary="#38bdf8",
        secondary="#71717a",
        accent="#22c55e",
        success="#22c55e",
        warning="#f59e0b",
        error="#ef4444",
        dark=True,
        variables={
            "border": "#26262b",
            "border-blurred": "#1f1f24",
            "footer-background": "#0b0b0d",
            "footer-foreground": "#71717a",
        },
    ),
    "oled-black": Theme(
        name="oled-black",
        background="#000000",
        surface="#09090b",
        panel="#101014",
        foreground="#f4f4f5",
        primary="#38bdf8",
        secondary="#71717a",
        accent="#22c55e",
        success="#22c55e",
        warning="#f59e0b",
        error="#ef4444",
        dark=True,
        variables={
            "border": "#222228",
            "border-blurred": "#18181c",
            "footer-background": "#000000",
            "footer-foreground": "#71717a",
        },
    ),
    "opencode-dark": Theme(
        name="opencode-dark",
        background="#0f0f0f",
        surface="#15141b",
        panel="#1c1b24",
        foreground="#edecee",
        primary="#a277ff",
        secondary="#82e2ff",
        accent="#61ffca",
        success="#61ffca",
        warning="#ffca85",
        error="#ff6767",
        dark=True,
        variables={
            "border": "#2d2d2d",
            "border-blurred": "#22212c",
            "footer-background": "#0f0f0f",
            "footer-foreground": "#6d6d6d",
        },
    ),
    "github-slate": Theme(
        name="github-slate",
        background="#0d1117",
        surface="#161b22",
        panel="#1c2128",
        foreground="#e6edf3",
        primary="#58a6ff",
        secondary="#79c0ff",
        accent="#3fb950",
        success="#3fb950",
        warning="#d29922",
        error="#f85149",
        dark=True,
        variables={
            "border": "#30363d",
            "border-blurred": "#21262d",
            "footer-background": "#0d1117",
            "footer-foreground": "#8b949e",
        },
    ),
    "catppuccin-mocha": Theme(
        name="catppuccin-mocha",
        background="#1e1e2e",
        surface="#181825",
        panel="#313244",
        foreground="#cdd6f4",
        primary="#89b4fa",
        secondary="#cba6f7",
        accent="#a6e3a1",
        success="#a6e3a1",
        warning="#f9e2af",
        error="#f38ba8",
        dark=True,
        variables={
            "border": "#45475a",
            "border-blurred": "#313244",
            "footer-background": "#1e1e2e",
            "footer-foreground": "#a6adc8",
        },
    ),
}

DEFAULT_THEME = THEME_ORDER[0]

TIER_COLORS: dict[str, str] = {
    "FREE": "#22c55e",
    "GO": "#38bdf8",
    "PLAN": "#38bdf8",
    "LOCAL": "#60a5fa",
    "ORCHESTRATOR": "#a1a1aa",
    "PREMIUM": "#ef4444",
}

TIER_LABELS: dict[str, str] = {
    "FREE": "FREE $0",
    "GO": "GO",
    "PLAN": "PLAN",
    "LOCAL": "LOCAL $0",
    "ORCHESTRATOR": "ROUTER",
    "PREMIUM": "PREMIUM",
}

_UNKNOWN_TIER_COLOR = "#a1a1aa"


def resolve_theme_name(name: Optional[str]) -> str:
    """Map a stored or requested theme name onto a bundled theme.

    Legacy aliases resolve to their renamed equivalents. Unknown or empty
    names fall back to `DEFAULT_THEME`.
    """
    key = (name or "").strip().lower()
    key = THEME_ALIASES.get(key, key)
    return key if key in NEXUS_THEMES else DEFAULT_THEME


def normalize_tier(tier: str) -> str:
    """Fold legacy paid/quota aliases onto the canonical tiers."""
    key = (tier or "").strip().upper().replace("_", "-")
    if key in ("ARGON", "ARGON-QUOTA", "PAID", "PREMIUM"):
        return "PREMIUM"
    return key


def tier_color(tier: str) -> str:
    """Return the hex colour used for a quota tier."""
    return TIER_COLORS.get(normalize_tier(tier), _UNKNOWN_TIER_COLOR)


def quota_markup(tier: str) -> str:
    """Return a Rich-markup quota badge with the brackets escaped as literals.

    The badge text itself starts with `[`, so the opening bracket is escaped:
    without that Rich would parse `[FREE $0]` as an unknown style tag and print
    nothing but the surrounding text.
    """
    key = normalize_tier(tier)
    label = TIER_LABELS.get(key, key or "UNKNOWN")
    if key == "PREMIUM":
        return f"[bold white on {TIER_COLORS[key]}]\\[{label}][/]"
    return f"[{tier_color(key)}]\\[{label}][/]"


def quota_plain(tier: str) -> str:
    """Return the badge text without markup, for logs and plain renderers."""
    key = normalize_tier(tier)
    return f"[{TIER_LABELS.get(key, key or 'UNKNOWN')}]"
