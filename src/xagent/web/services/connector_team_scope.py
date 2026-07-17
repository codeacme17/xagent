"""Optional application hooks for team-owned MCP and Custom API connectors.

Standalone xagent keeps connectors user-owned. A multi-tenant application can
install these hooks to overlay team visibility without teaching xagent about
the application's team tables.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

ConnectorType = Literal["mcp", "custom_api"]
ConnectorRenamedHook = Callable[[Any, int, ConnectorType, int, str, str], None]


@dataclass(frozen=True)
class ConnectorDeleteDecision:
    team_owned: bool = False
    authorized: bool = False
    delete_definition: bool = False
    # Set when the delete is refused because the connector is still selected by a
    # team agent. The endpoint surfaces this as a 403 before any mutation, mirroring
    # the unshare path's "still used by a team agent" guard.
    blocked_reason: str | None = None


ConnectorDeletedHook = Callable[[Any, int, ConnectorType, int], ConnectorDeleteDecision]

ConnectorVisibilityHook = Callable[[Any, int], dict[str, set[int]]]

_connector_deleted_hook: ConnectorDeletedHook | None = None
_connector_renamed_hook: ConnectorRenamedHook | None = None
_connector_visibility_hook: ConnectorVisibilityHook | None = None


def set_connector_team_hooks(
    *,
    deleted: ConnectorDeletedHook | None = None,
    renamed: ConnectorRenamedHook | None = None,
    visibility: ConnectorVisibilityHook | None = None,
) -> None:
    """Install application-owned connector lifecycle hooks."""

    global _connector_deleted_hook, _connector_renamed_hook
    global _connector_visibility_hook
    _connector_deleted_hook = deleted
    _connector_renamed_hook = renamed
    _connector_visibility_hook = visibility


def visible_team_connector_ids(db: Any, user_id: int) -> dict[str, set[int]]:
    """Team-shared connector ids visible to user; empty when no hook/standalone."""
    if _connector_visibility_hook is None:
        return {"mcp": set(), "custom_api": set()}
    return _connector_visibility_hook(db, int(user_id))


def delete_team_connector(
    db: Any, user_id: int, connector_type: ConnectorType, connector_id: int
) -> ConnectorDeleteDecision:
    """Remove team ownership before a global delete.

    Returns whether the application recognized the connector as team-owned.
    Hooks must use the passed session and must not commit independently, so a
    refused endpoint request can discard all hook-side mutations atomically.
    """

    if _connector_deleted_hook is None:
        return ConnectorDeleteDecision()
    return _connector_deleted_hook(db, user_id, connector_type, connector_id)


def rename_team_connector(
    db: Any,
    user_id: int,
    connector_type: ConnectorType,
    connector_id: int,
    old_name: str,
    new_name: str,
) -> None:
    """Keep application-owned connector selectors aligned after a rename."""

    if _connector_renamed_hook is not None and old_name != new_name:
        _connector_renamed_hook(
            db, user_id, connector_type, connector_id, old_name, new_name
        )
