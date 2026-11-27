"""Shared pytest fixtures keeping tests isolated from host user config."""

from __future__ import annotations

from pathlib import Path

import pytest

import nexus.router as _router


@pytest.fixture(autouse=True)
def _isolate_user_routes_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Prevent ~/.config/nexus/routes.json on the developer host from altering unit tests."""
    monkeypatch.setattr(_router, "USER_ROUTES_PATH", tmp_path / "no-user-routes.json")
