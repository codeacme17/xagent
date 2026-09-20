"""seed built-in Miro (remote MCP, DCR OAuth) connector

Revision ID: 20260920_seed_miro_mcp_app
Revises: 20260920_seed_atlassian_mcp_app
Create Date: 2026-09-20 00:00:00.000000

"""

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.builtin_identity import (
    builtin_provenance_identity,
    canonicalize_builtin_identity,
)

logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision: str = "20260920_seed_miro_mcp_app"
down_revision: Union[str, None] = "20260920_seed_atlassian_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("name", sa.String),
    sa.column("description", sa.Text),
    sa.column("icon", sa.String),
    sa.column("transport", sa.String),
    sa.column("provider_name", sa.String),
    sa.column("category", sa.String),
    sa.column("oauth_scopes", sa.JSON),
    sa.column("is_visible_in_connector", sa.Boolean),
    sa.column("launch_config", sa.JSON),
)
MCP_SERVERS_TABLE = sa.table(
    "mcp_servers",
    sa.column("id", sa.Integer),
    sa.column("name", sa.String),
    sa.column("transport", sa.String),
    sa.column("url", sa.String),
    sa.column("auth", sa.JSON),
    sa.column("command", sa.String),
    sa.column("env", sa.JSON),
    sa.column("headers", sa.JSON),
    sa.column("runtime_input_schema", sa.JSON),
    sa.column("runtime_bindings", sa.JSON),
)
# Fields a fresh connect-created shared row never carries; a colliding row
# holding any of them is a custom server's own policy (or an admin edit), not
# our row, even when name/transport/URL/auth.type all match -- mirrors
# _server_has_policy_beyond_catalog_identity on the stdio connect path.
MCP_SERVER_POLICY_COLUMNS = (
    "command",
    "env",
    "headers",
    "runtime_input_schema",
    "runtime_bindings",
)
USER_MCPSERVERS_TABLE = sa.table(
    "user_mcpservers",
    sa.column("mcpserver_id", sa.Integer),
    sa.column("is_owner", sa.Boolean),
)

APP_ID = "miro"
BUILTIN_PROVENANCE = {
    "registry": "xagent",
    "app_id": APP_ID,
    "version": 1,
}

# Frozen copy of the builtin_mcp_registry.py row at the time this migration
# was written (tests/alembic pins the two against each other). Same connector
# shape as the granola/notion seeds: a vendor-hosted remote MCP server reached
# over streamable_http, authenticated per user through OAuth 2.1 + PKCE with
# Dynamic Client Registration, so there is no oauth_providers row and no
# secret in launch_config. Unlike those two seeds it carries the
# builtin_provenance ownership marker (whatsapp/shopify pattern) so that a
# pre-existing operator row under the same app_id is never adopted on upgrade
# nor deleted on downgrade.
# Miro serves MCP at the host root: its protected-resource metadata names
# "https://mcp.miro.com/" as the resource and as its own authorization server.
ROW = {
    "app_id": APP_ID,
    "name": "Miro",
    "description": "Connect to Miro to find boards and read, create and update board content through Miro's hosted MCP server.",
    "icon": "https://www.google.com/s2/favicons?domain=miro.com&sz=128",
    "transport": "streamable_http",
    "provider_name": None,
    "category": "Productivity",
    "oauth_scopes": None,
    "is_visible_in_connector": True,
    "launch_config": {
        "url": "https://mcp.miro.com/",
        "auth": {"type": "mcp_oauth"},
        "builtin_provenance": BUILTIN_PROVENANCE,
    },
}

# mcp_servers is reconciled too (see _reconcile_mcp_servers): the connector
# listing resolves a catalog app's shared server row by normalized transport +
# app_id/display name, not by URL, auth or owner, so a custom server that
# predates this identity would be presented as the official card while the
# runtime kept using its URL/auth. The connect-time guards in
# _ensure_catalog_mcp_oauth_server (transport/URL match and
# _reject_user_owned_catalog_squat) only run when someone connects and cannot
# repair an already-misattributed listing, so the check has to happen here,
# before the identity is claimed. Unlike the shopify seed, the trusted
# round-trip row is recognised by ownership rather than a marker in
# MCPServer.auth: create_mcp_server does not reject a caller-supplied
# builtin_provenance key inside auth, so such a marker is forgeable, whereas
# the shared row our connect path creates never has an is_owner=true link.


def _filter_row(row: dict[str, object], allowed_columns: set[str]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key in allowed_columns}


def _has_provenance(launch_config: object) -> bool:
    return isinstance(launch_config, dict) and builtin_provenance_identity(
        launch_config.get("builtin_provenance")
    ) == builtin_provenance_identity(BUILTIN_PROVENANCE)


def _collides_with_miro_identity(value: object) -> bool:
    return canonicalize_builtin_identity(value) in {
        canonicalize_builtin_identity(APP_ID),
        canonicalize_builtin_identity(ROW["name"]),
    }


def _reconcile_mcp_servers(bind: sa.engine.Connection, inspector: sa.Inspector) -> None:
    """Fail closed when an mcp_servers row already carries this identity.

    The single accepted collision is the shared row our own mcp_oauth connect
    path creates: name == app_id, catalog transport and URL, auth.type ==
    "mcp_oauth", and no user_mcpservers link with is_owner=true. That row
    exists legitimately on a downgrade -> upgrade round trip. Anything else
    (a user-owned row, a foreign URL or transport, a row carrying its own
    headers/env/bindings, a row named after the display name, or a schema
    that cannot prove ownership) aborts the upgrade so the operator
    renames or deletes the row before the catalog identity is claimed.
    """
    tables = set(inspector.get_table_names())
    if "mcp_servers" not in tables:
        return
    server_columns = {column["name"] for column in inspector.get_columns("mcp_servers")}
    colliding = [
        row
        for row in bind.execute(
            sa.select(MCP_SERVERS_TABLE.c.id, MCP_SERVERS_TABLE.c.name)
        ).mappings()
        if _collides_with_miro_identity(row["name"])
    ]
    if not colliding:
        return
    described = sorted((int(row["id"]), str(row["name"])) for row in colliding)
    if (
        not {"transport", "url", "auth"} <= server_columns
        or "user_mcpservers" not in tables
    ):
        raise RuntimeError(
            "Cannot seed builtin Miro connector: mcp_servers row(s) "
            f"{described} collide with '{APP_ID}' and the schema cannot prove "
            "which one our connect path created"
        )
    if len(colliding) == 1:
        policy_columns = [c for c in MCP_SERVER_POLICY_COLUMNS if c in server_columns]
        server = (
            bind.execute(
                sa.select(
                    MCP_SERVERS_TABLE.c.name,
                    MCP_SERVERS_TABLE.c.transport,
                    MCP_SERVERS_TABLE.c.url,
                    MCP_SERVERS_TABLE.c.auth,
                    *(MCP_SERVERS_TABLE.c[c] for c in policy_columns),
                ).where(MCP_SERVERS_TABLE.c.id == described[0][0])
            )
            .mappings()
            .one()
        )
        auth = server["auth"] if isinstance(server["auth"], dict) else {}
        carries_policy = any(server[c] for c in policy_columns)
        owned = bind.execute(
            sa.select(USER_MCPSERVERS_TABLE.c.mcpserver_id).where(
                USER_MCPSERVERS_TABLE.c.mcpserver_id == described[0][0],
                USER_MCPSERVERS_TABLE.c.is_owner.is_(True),
            )
        ).first()
        if (
            server["name"] == APP_ID
            and str(server["transport"] or "").lower() == ROW["transport"]
            and server["url"] == ROW["launch_config"]["url"]
            and auth.get("type") == "mcp_oauth"
            and not carries_policy
            and owned is None
        ):
            return
    raise RuntimeError(
        "Cannot seed builtin Miro connector: mcp_servers row(s) "
        f"{described} collide with '{APP_ID}' and could not all be proven to "
        "come from the catalog connect path; rename or delete them before upgrading"
    )


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "public_mcp_apps" not in tables:
        return

    columns = {column["name"] for column in inspector.get_columns("public_mcp_apps")}
    if "launch_config" not in columns:
        raise RuntimeError(
            "Cannot seed builtin Miro identity: public_mcp_apps.launch_config "
            "is required for provenance"
        )

    catalog_rows = list(
        bind.execute(
            sa.select(
                PUBLIC_MCP_APPS_TABLE.c.app_id,
                PUBLIC_MCP_APPS_TABLE.c.name,
                PUBLIC_MCP_APPS_TABLE.c.launch_config,
            )
        ).mappings()
    )
    exact_app_rows = [row for row in catalog_rows if row["app_id"] == APP_ID]
    if exact_app_rows:
        existing = exact_app_rows[0]
        # Idempotent re-run over a row this migration (or a prior version of
        # it) already owns: accept and stop. An unrelated row created *after*
        # this one was seeded is policed at that row's own creation time
        # (POST /admin/mcp/apps), not by failing every later
        # `alembic upgrade head` against an already-correctly-owned row.
        if _has_provenance(existing["launch_config"]):
            return
        # app_id is what the builtin execution overlay keys off of
        # (get_builtin_execution_fields / _matches_builtin_provenance look up
        # by exact app_id, never by name), so an unprovenanced row squatting
        # on it is a genuine misidentification risk: fail closed rather than
        # seed alongside it or silently skip.
        raise RuntimeError(
            "Cannot seed builtin Miro connector: an existing "
            "public_mcp_apps row with app_id='miro' has no matching "
            "builtin_provenance"
        )

    # No row claims app_id "miro". A *different* app_id whose display name
    # (or, via typo/casing, its app_id) normalizes to the same identity is a
    # one-time cosmetic collision in the connector picker, not a
    # misidentification risk, so it does not warrant aborting the whole
    # `alembic upgrade head` run. Skip with a warning instead.
    #
    # This skip is permanent: Alembic stamps this revision as applied whether
    # or not the insert below ran, so a re-run will NOT retry seeding even if
    # the colliding row is later renamed away. Recovering the builtin row
    # afterwards needs a manual INSERT (or a follow-up migration) using this
    # file's ROW/BUILTIN_PROVENANCE as the template.
    colliding_rows = [
        row
        for row in catalog_rows
        if _collides_with_miro_identity(row["app_id"])
        or _collides_with_miro_identity(row["name"])
    ]
    if colliding_rows:
        logger.error(
            "Permanently skipping builtin Miro seed: public_mcp_apps "
            "row(s) with app_id %s share its identity under a different "
            "app_id. Re-running `alembic upgrade head` will NOT retry this "
            "-- the revision is already stamped applied. Seed the row "
            "manually (see this migration's ROW/BUILTIN_PROVENANCE) once "
            "the collision is resolved.",
            sorted({row["app_id"] for row in colliding_rows}),
        )
        return

    _reconcile_mcp_servers(bind, inspector)

    dropped_keys = sorted(set(ROW) - columns)
    if dropped_keys:
        logger.warning(
            "public_mcp_apps is missing columns %s; seeding %r without them",
            dropped_keys,
            APP_ID,
        )
    bind.execute(sa.insert(PUBLIC_MCP_APPS_TABLE), [_filter_row(ROW, columns)])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return
    columns = {column["name"] for column in inspector.get_columns("public_mcp_apps")}
    if "launch_config" not in columns:
        return
    # Only a provenance-owned catalog entry is removed. No oauth_providers row
    # exists for Miro (auth is per-user Dynamic Client Registration, not a
    # shared static client), and any MCPServer/UserMCPServer/MCPOAuth* rows
    # created by users who already connected are intentionally left in place
    # -- connect-driven rows are not owned by this migration and are cleaned
    # up through the normal disconnect path.
    existing = bind.execute(
        sa.select(PUBLIC_MCP_APPS_TABLE.c.launch_config).where(
            PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID
        )
    ).scalar_one_or_none()
    if not _has_provenance(existing):
        return
    bind.execute(
        sa.delete(PUBLIC_MCP_APPS_TABLE).where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
    )
