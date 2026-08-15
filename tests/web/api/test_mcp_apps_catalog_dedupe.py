"""`/api/mcp/apps`'s local branch must not re-emit a connected catalog app
(#1346).

The catalog branch matches a stored server row to its app on *both* the app_id
and the display name (`_connected_non_oauth_server_for_app` ->
`_app_lookup_keys`), but the local branch's skip compared the row's name against
display names only. Catalog connects name the shared row after the app_id
(`_ensure_catalog_app_server`, `_ensure_catalog_mcp_oauth_server`), so every app
whose app_id and name differ escaped the skip and was emitted a second time as
an `is_custom: true` "Local" entry.

Two app shapes appear below. Built-in ids (`google-maps`) take their name,
transport and launch_config from `builtin_mcp_registry` regardless of what the
row stores, so they pin the duplicate a user reaches on a stock deployment.
Fictional ids (`acme*`) are admin-created apps, whose stored fields are used
as-is — the only way to vary the shape under test.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.api.mcp import list_mcp_apps
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.services import connector_team_scope

_MCP_OAUTH_LAUNCH: dict[str, Any] = {
    "url": "https://mcp.example.com/mcp",
    "auth": {
        "type": "mcp_oauth",
        "resource": "https://mcp.example.com/mcp",
        "issuer": "https://auth.example.com",
    },
}


@pytest.fixture()
def db_session(tmp_path):
    db_path = tmp_path / "mcp-apps-dedupe.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = SessionLocal()

    user = User(username="alice", password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)

    yield db, user
    db.close()
    engine.dispose()


@pytest.fixture(autouse=True)
def _reset_connector_team_hooks():
    """The hooks are process-global; never leak one into a sibling test."""
    yield
    connector_team_scope.set_connector_team_hooks()


def _add_key_based_app(db, app_id: str, name: str) -> PublicMCPApp:
    """A catalog app classified `api_key` — the shape connected through
    `_ensure_catalog_app_server`, which names the shared row after the app_id."""
    return _add_app(
        db,
        app_id,
        name,
        transport="stdio",
        launch_config={
            "command": "npx",
            "args": ["-y", f"{app_id}-mcp"],
            "required_env": ["API_KEY"],
        },
    )


def _add_mcp_oauth_app(db, app_id: str, name: str) -> PublicMCPApp:
    return _add_app(
        db, app_id, name, transport="streamable_http", launch_config=_MCP_OAUTH_LAUNCH
    )


def _add_app(
    db, app_id: str, name: str, *, transport: str, launch_config: dict
) -> PublicMCPApp:
    app = PublicMCPApp(
        app_id=app_id,
        name=name,
        description="A catalog app",
        icon="",
        category="Productivity",
        transport=transport,
        launch_config=launch_config,
        is_visible_in_connector=True,
    )
    db.add(app)
    db.commit()
    db.refresh(app)
    return app


def _add_catalog_server(db, app_id: str) -> MCPServer:
    """The shared stdio row a key-based/keyless connect writes: named after the
    app_id, carrying no `auth.app_id` back-reference."""
    return _add_server_row(
        db,
        {
            "name": app_id,
            "description": "A catalog app",
            "managed": "external",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", f"{app_id}-mcp"],
        },
    )


def _add_mcp_oauth_server(db, app_id: str) -> MCPServer:
    """The shared remote row `_ensure_catalog_mcp_oauth_server` writes: also
    named after the app_id, and its `auth` carries the connector's OAuth
    metadata rather than an app_id back-reference."""
    return _add_server_row(
        db,
        {
            "name": app_id,
            "managed": "external",
            "transport": "streamable_http",
            "url": _MCP_OAUTH_LAUNCH["url"],
            "auth": dict(_MCP_OAUTH_LAUNCH["auth"]),
        },
    )


def _add_custom_server(db, name: str) -> MCPServer:
    return _add_server_row(
        db,
        {
            "name": name,
            "description": f"{name} MCP server",
            "managed": "external",
            "transport": "stdio",
            "command": f"{name}-mcp",
        },
    )


def _add_server_row(db, config: dict) -> MCPServer:
    server = MCPServer.from_config(config)
    db.add(server)
    db.commit()
    db.refresh(server)
    return server


def _associate(db, user: User, server: MCPServer, *, is_owner: bool = False) -> None:
    """The association a catalog connect writes: the connecting user never owns
    the shared row (`is_owner=False`)."""
    db.add(
        UserMCPServer(
            user_id=user.id,
            mcpserver_id=server.id,
            is_owner=is_owner,
            can_edit=is_owner,
            can_delete=True,
            is_active=True,
        )
    )
    db.commit()


def _install_visibility(user: User, server_ids: set[int]) -> None:
    def visibility(_db, user_id: int) -> dict[str, set[int]]:
        if int(user_id) != int(user.id):
            return {"mcp": set(), "custom_api": set()}
        return {"mcp": set(server_ids), "custom_api": set()}

    connector_team_scope.set_connector_team_hooks(visibility=visibility)


def test_a_connected_catalog_app_with_a_mismatched_id_is_listed_once(db_session):
    """AC1: `location=all` emits the app from the catalog branch only.

    `google-maps`/"Google Maps" is a shipped built-in pair, so this is the
    duplicate a user reaches today with no admin action at all."""
    db, user = db_session
    _add_key_based_app(db, "google-maps", "Google Maps")
    server = _add_catalog_server(db, "google-maps")
    _associate(db, user, server)

    entries = [
        a
        for a in list_mcp_apps(location="all", current_user=user, db=db)
        if a["id"] == "google-maps"
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["name"] == "Google Maps"
    assert entry.get("is_custom") is not True
    assert entry.get("is_local") is not True
    assert entry["is_connected"] is True


def test_a_connected_catalog_app_with_a_mismatched_id_is_absent_from_local(db_session):
    """AC2: the app belongs to the Remote tab, so `location=local` has no entry
    for it at all — not even a correctly-shaped one."""
    db, user = db_session
    _add_key_based_app(db, "google-maps", "Google Maps")
    server = _add_catalog_server(db, "google-maps")
    _associate(db, user, server)

    assert list_mcp_apps(location="local", current_user=user, db=db) == []


def test_an_mcp_oauth_catalog_app_produces_no_ungranted_local_twin(db_session):
    """AC4: the local branch tags an active association `auth_type:
    "mcp_oauth"`, and the picker treats `is_custom && mcp_oauth` as attachable
    (#1332). A duplicate therefore made a catalog OAuth app attachable with no
    completed grant behind it; the fix removes the duplicate, so no entry
    carries that pair.

    Admin-created (non-built-in) app on purpose: the two shipped `mcp_oauth`
    entries have matching id/name, and renaming one is refused —
    `_BUILTIN_PROTECTED_FIELDS` 409s a `name` change for a built-in app, and
    `_app_to_dict` would take the registry's name regardless. An admin-created
    connector is where an id/name mismatch is actually authored."""
    db, user = db_session
    _add_mcp_oauth_app(db, "acme-notes", "Acme Notes")
    server = _add_mcp_oauth_server(db, "acme-notes")
    _associate(db, user, server)

    results = list_mcp_apps(location="all", current_user=user, db=db)
    assert not [
        a
        for a in results
        if a.get("is_custom") is True and a.get("auth_type") == "mcp_oauth"
    ]
    notes = [a for a in results if a["id"] == "acme-notes"]
    assert len(notes) == 1
    # No grant was issued, so the catalog entry stays disconnected — the
    # duplicate was the only route to attaching it.
    assert notes[0]["is_connected"] is False


def test_a_team_overlaid_catalog_row_is_skipped_too(db_session):
    """The overlay feeds the same loop as `(server, user_mcp)` pairs, so a
    catalog row reaching the branch through team visibility (#1321) — with no
    personal association at all — must obey the same skip."""
    db, user = db_session
    _add_key_based_app(db, "google-maps", "Google Maps")
    server = _add_catalog_server(db, "google-maps")
    _install_visibility(user, {int(server.id)})

    assert list_mcp_apps(location="local", current_user=user, db=db) == []


def test_a_catalog_app_whose_id_matches_its_name_is_listed_as_before(db_session):
    """AC5: the already-skipped convention keeps working. This differs from the
    mismatch cases above only in that the app's two keys agree."""
    db, user = db_session
    _add_key_based_app(db, "acme", "Acme")
    server = _add_catalog_server(db, "acme")
    _associate(db, user, server)

    assert list_mcp_apps(location="local", current_user=user, db=db) == []
    acme = [
        a
        for a in list_mcp_apps(location="all", current_user=user, db=db)
        if a["id"] == "acme"
    ]
    assert len(acme) == 1
    assert acme[0]["is_connected"] is True


def test_a_server_row_named_after_the_display_name_is_still_skipped(db_session):
    """The built-in OAuth convention (`_ensure_user_mcp_server` names the row
    after the display name) must keep matching — widening the skip to app_ids
    must not narrow it away from names."""
    db, user = db_session
    _add_key_based_app(db, "acme-tools", "Acme Tools")
    server = _add_custom_server(db, "Acme Tools")
    _associate(db, user, server, is_owner=True)

    assert list_mcp_apps(location="local", current_user=user, db=db) == []


def test_an_unrelated_custom_server_is_still_listed(db_session):
    """The skip must stay keyed on the catalog: a custom server that resembles
    no catalog app keeps its Local entry."""
    db, user = db_session
    _add_key_based_app(db, "google-maps", "Google Maps")
    server = _add_custom_server(db, "records")
    _associate(db, user, server, is_owner=True)

    entries = list_mcp_apps(location="local", current_user=user, db=db)
    assert [a["id"] for a in entries] == ["records"]
    assert entries[0]["is_custom"] is True


def test_remote_results_are_unchanged(db_session):
    """AC6: the skip lives in the local branch; the catalog branch's own
    connection state is untouched by it."""
    db, user = db_session
    _add_key_based_app(db, "google-maps", "Google Maps")
    server = _add_catalog_server(db, "google-maps")
    _associate(db, user, server)

    remote = list_mcp_apps(location="remote", current_user=user, db=db)
    assert [a["id"] for a in remote] == ["google-maps"]
    assert remote[0]["is_connected"] is True
    assert remote[0]["server_id"] == server.id
