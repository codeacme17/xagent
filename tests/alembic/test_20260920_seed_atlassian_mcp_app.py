"""Tests for the provenance-safe Atlassian remote-MCP connector seed."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, select, text

APP_ID = "atlassian"


def _load_migration():
    path = (
        Path(__file__).parents[2]
        / "src/xagent/migrations/versions/20260920_seed_atlassian_mcp_app.py"
    )
    spec = importlib.util.spec_from_file_location("seed_atlassian_migration", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _create_apps_table(connection):
    connection.execute(
        text(
            """
            CREATE TABLE public_mcp_apps (
                id INTEGER PRIMARY KEY,
                app_id VARCHAR(100) NOT NULL UNIQUE,
                name VARCHAR(200) NOT NULL,
                description TEXT,
                icon VARCHAR(1000),
                transport VARCHAR(50) NOT NULL DEFAULT 'oauth',
                provider_name VARCHAR(50),
                category VARCHAR(100),
                oauth_scopes JSON,
                is_visible_in_connector BOOLEAN NOT NULL DEFAULT 1,
                launch_config JSON
            )
            """
        )
    )


def _app_ids(connection):
    return set(connection.execute(text("SELECT app_id FROM public_mcp_apps")).scalars())


def _run(connection, migration, *steps):
    with patch.object(migration, "op", _operations(connection)):
        for step in steps:
            getattr(migration, step)()


def test_upgrade_inserts_row_matching_seed_snapshot(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _run(connection, migration, "upgrade")
        # Full-row comparison through the migration's own typed table object:
        # JSON columns deserialize and booleans come back as real bools, so
        # the persisted row compares to ROW directly (including the
        # builtin_provenance marker inside launch_config).
        row = (
            connection.execute(
                select(migration.PUBLIC_MCP_APPS_TABLE).where(
                    migration.PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID
                )
            )
            .mappings()
            .first()
        )
        assert row is not None
        assert dict(row) == migration.ROW


def test_upgrade_is_idempotent_for_provenance_owned_row(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _run(connection, migration, "upgrade", "upgrade")
        count = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id=:app_id"),
            {"app_id": APP_ID},
        ).scalar_one()
    assert count == 1


def test_upgrade_is_idempotent_even_with_an_unrelated_later_collision(tmp_path):
    """A later, unrelated custom app that happens to normalize to the same
    display name must not retroactively block every future `alembic upgrade
    head` over the already-seeded, correctly-owned row."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _run(connection, migration, "upgrade")
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, transport, is_visible_in_connector) "
                "VALUES ('unrelated-app', :name, 'stdio', 1)"
            ),
            {"name": migration.ROW["name"]},
        )
        _run(connection, migration, "upgrade")
        count = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id=:app_id"),
            {"app_id": APP_ID},
        ).scalar_one()
    assert count == 1


def test_upgrade_accepts_owned_row_from_an_older_provenance_version(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        launch_config = dict(migration.ROW["launch_config"])
        marker = dict(migration.BUILTIN_PROVENANCE)
        marker["version"] = 0
        launch_config["builtin_provenance"] = marker
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, transport, launch_config) "
                "VALUES (:app_id, :name, 'streamable_http', :launch_config)"
            ),
            {
                "app_id": APP_ID,
                "name": migration.ROW["name"],
                "launch_config": json.dumps(launch_config),
            },
        )
        _run(connection, migration, "upgrade")
        count = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id=:app_id"),
            {"app_id": APP_ID},
        ).scalar_one()
    assert count == 1


def test_upgrade_refuses_custom_catalog_collision(tmp_path):
    """A pre-existing custom row under this app_id (e.g. hand-created by an
    operator via POST /admin/mcp/apps before this migration deployed) has no
    provenance marker, so this must fail closed rather than adopt the row for
    the builtin execution overlay to misidentify later."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, description, transport, is_visible_in_connector) "
                "VALUES (:app_id, 'Operator Row', 'hand-made', 'streamable_http', 0)"
            ),
            {"app_id": APP_ID},
        )
        with pytest.raises(RuntimeError, match="public_mcp_apps"):
            _run(connection, migration, "upgrade")
        row = connection.execute(
            text("SELECT name FROM public_mcp_apps WHERE app_id=:app_id"),
            {"app_id": APP_ID},
        ).scalar_one()
    assert row == "Operator Row"


@pytest.mark.parametrize(
    "app_id,name",
    [
        (f" {APP_ID.upper()} ", "Unrelated"),
        ("other", None),  # None -> the seeded display name, padded
        (APP_ID.capitalize(), "Unrelated"),
    ],
)
def test_upgrade_skips_seeding_on_a_name_only_collision(tmp_path, app_id, name):
    """None of these rows claim the exact app_id the overlay keys off of, so
    a display-name (or look-alike app_id) collision under a *different*
    app_id must not abort the whole `alembic upgrade head` run. It only means
    the builtin row is not seeded this run."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    if name is None:
        name = f" {migration.ROW['name'].lower()} "
    with engine.begin() as connection:
        _create_apps_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, transport, is_visible_in_connector) "
                "VALUES (:app_id, :name, 'oauth', 0)"
            ),
            {"app_id": app_id, "name": name},
        )
        _run(connection, migration, "upgrade")  # must not raise
        assert APP_ID not in _app_ids(connection)
        row = connection.execute(
            text("SELECT app_id, name FROM public_mcp_apps WHERE app_id = :app_id"),
            {"app_id": app_id},
        ).one()
        assert row == (app_id, name)


def test_upgrade_skips_columns_missing_from_a_reduced_schema(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE public_mcp_apps (
                    id INTEGER PRIMARY KEY,
                    app_id VARCHAR(100) NOT NULL UNIQUE,
                    name VARCHAR(200) NOT NULL,
                    transport VARCHAR(50) NOT NULL DEFAULT 'oauth',
                    launch_config JSON
                )
                """
            )
        )
        _run(connection, migration, "upgrade")
        assert APP_ID in _app_ids(connection)


def test_upgrade_raises_when_launch_config_column_is_missing(tmp_path):
    """launch_config is where the provenance marker lives; without the
    column there is no way to prove ownership of any existing row."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE public_mcp_apps (
                    id INTEGER PRIMARY KEY,
                    app_id VARCHAR(100) NOT NULL UNIQUE,
                    name VARCHAR(200) NOT NULL,
                    transport VARCHAR(50) NOT NULL DEFAULT 'oauth'
                )
                """
            )
        )
        with pytest.raises(RuntimeError, match="launch_config"):
            _run(connection, migration, "upgrade")
        assert APP_ID not in _app_ids(connection)


def test_seed_row_matches_registry():
    """The migration snapshot and the runtime registry must define the same
    row (the migration is a frozen copy; this catches drift)."""
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration()
    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == APP_ID
    )
    assert migration.ROW == registry_row
    assert registry_row["is_visible_in_connector"] is True


def test_seed_row_classifies_as_mcp_oauth():
    """The seeded shape must classify as a remote-MCP OAuth connector -- an
    "unconnectable" classification would make the catalog entry dead on
    arrival (no connect endpoint accepts it). The provenance marker inside
    launch_config must not disturb that classification."""
    from xagent.web.mcp_apps import classify_app_auth

    migration = _load_migration()
    assert (
        classify_app_auth(migration.ROW["transport"], migration.ROW["launch_config"])
        == "mcp_oauth"
    )


def test_downgrade_only_deletes_provenance_owned_row(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _run(connection, migration, "upgrade")
        # A sentinel row unrelated to this migration must survive the
        # downgrade -- otherwise the seeded app_id missing from _app_ids could
        # equally mean the whole table was wiped.
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps (app_id, name, transport)"
                " VALUES ('sentinel', 'Sentinel', 'oauth')"
            )
        )
        _run(connection, migration, "downgrade")
        assert _app_ids(connection) == {"sentinel"}


def test_downgrade_preserves_a_preexisting_operator_row(tmp_path):
    """A pre-existing operator row makes upgrade() raise (it never gets
    adopted), so downgrade() must never delete it either."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, description, transport) "
                "VALUES (:app_id, :name, 'hand-made', 'streamable_http')"
            ),
            {"app_id": APP_ID, "name": migration.ROW["name"]},
        )
        _run(connection, migration, "downgrade")
        row = connection.execute(
            text("SELECT name, description FROM public_mcp_apps WHERE app_id=:app_id"),
            {"app_id": APP_ID},
        ).first()
        assert row == (migration.ROW["name"], "hand-made")


def test_downgrade_is_a_noop_when_launch_config_column_is_missing(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE public_mcp_apps (
                    id INTEGER PRIMARY KEY,
                    app_id VARCHAR(100) NOT NULL UNIQUE,
                    name VARCHAR(200) NOT NULL,
                    transport VARCHAR(50) NOT NULL DEFAULT 'oauth'
                )
                """
            )
        )
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps (app_id, name, transport) "
                "VALUES (:app_id, :name, 'streamable_http')"
            ),
            {"app_id": APP_ID, "name": migration.ROW["name"]},
        )
        _run(connection, migration, "downgrade")
        assert APP_ID in _app_ids(connection)


def test_upgrade_and_downgrade_no_op_without_table(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _run(connection, migration, "upgrade", "downgrade")
        table_names = set(
            connection.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).scalars()
        )
        assert "public_mcp_apps" not in table_names


# --- mcp_servers reconciliation (rollout collisions; #2506 review G3) ---


def _create_server_tables(connection, *, with_identity_columns=True):
    extra = (
        ", transport VARCHAR(50), url VARCHAR(500), auth JSON, command VARCHAR(500),"
        " env JSON, headers JSON, runtime_input_schema JSON, runtime_bindings JSON"
        if with_identity_columns
        else ""
    )

    connection.execute(
        text(
            "CREATE TABLE mcp_servers ("
            "id INTEGER PRIMARY KEY, name VARCHAR(100) NOT NULL UNIQUE" + extra + ")"
        )
    )
    connection.execute(
        text(
            """
            CREATE TABLE user_mcpservers (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                mcpserver_id INTEGER NOT NULL,
                is_owner BOOLEAN NOT NULL DEFAULT 0,
                is_active BOOLEAN NOT NULL DEFAULT 1
            )
            """
        )
    )


def _insert_server(
    connection,
    *,
    name,
    url,
    auth,
    transport="streamable_http",
    owner_user_id=None,
    server_id=1,
    headers=None,
):
    connection.execute(
        text(
            "INSERT INTO mcp_servers (id, name, transport, url, auth, headers) "
            "VALUES (:id, :name, :transport, :url, :auth, :headers)"
        ),
        {
            "id": server_id,
            "name": name,
            "transport": transport,
            "url": url,
            "auth": json.dumps(auth) if auth is not None else None,
            "headers": json.dumps(headers) if headers is not None else None,
        },
    )
    if owner_user_id is not None:
        connection.execute(
            text(
                "INSERT INTO user_mcpservers (user_id, mcpserver_id, is_owner, is_active) "
                "VALUES (:user_id, :server_id, 1, 1)"
            ),
            {"user_id": owner_user_id, "server_id": server_id},
        )


def test_upgrade_refuses_user_owned_server_with_foreign_url(tmp_path):
    """A custom streamable_http server a user created under this app_id before
    the identity existed points at a foreign URL. Seeding the catalog row would
    let the listing present that row as the official card (it resolves by
    normalized transport + name) while the runtime keeps using the foreign
    URL/auth, so the migration must fail closed and leave the row alone."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _create_server_tables(connection)
        _insert_server(
            connection,
            name=APP_ID,
            url="https://evil.example/mcp",
            auth={"type": "mcp_oauth"},
            owner_user_id=7,
        )
        with pytest.raises(RuntimeError, match="mcp_servers"):
            _run(connection, migration, "upgrade")
        assert APP_ID not in _app_ids(connection)
        assert connection.execute(text("SELECT url FROM mcp_servers")).scalar_one() == (
            "https://evil.example/mcp"
        )


def test_upgrade_refuses_user_owned_server_even_with_matching_url(tmp_path):
    """A matching URL is not enough: the owner keeps edit rights and could swap
    in a foreign URL later that every user of the catalog card would then hit
    (the same reasoning as _reject_user_owned_catalog_squat at connect time)."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _create_server_tables(connection)
        _insert_server(
            connection,
            name=APP_ID,
            url=migration.ROW["launch_config"]["url"],
            auth={"type": "mcp_oauth"},
            owner_user_id=7,
        )
        with pytest.raises(RuntimeError, match="mcp_servers"):
            _run(connection, migration, "upgrade")
        assert APP_ID not in _app_ids(connection)


def test_upgrade_refuses_server_named_after_display_name(tmp_path):
    """The listing also claims rows named after the display name (see
    _catalog_app_keys), so an unowned custom row under that spelling is a
    collision too, not just an exact app_id match."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _create_server_tables(connection)
        _insert_server(
            connection,
            name=migration.ROW["name"],
            url="https://other.example/mcp",
            auth={"type": "bearer", "token": "x"},
        )
        with pytest.raises(RuntimeError, match="mcp_servers"):
            _run(connection, migration, "upgrade")
        assert APP_ID not in _app_ids(connection)


def test_upgrade_accepts_unowned_connect_created_row_round_trip(tmp_path):
    """upgrade -> a user connects (our connect path creates the shared row:
    name == app_id, catalog transport/URL, auth.type mcp_oauth, no owner) ->
    downgrade -> upgrade must succeed: that row is ours, not a squatter."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _create_server_tables(connection)
        _run(connection, migration, "upgrade")
        _insert_server(
            connection,
            name=APP_ID,
            url=migration.ROW["launch_config"]["url"],
            auth={"type": "mcp_oauth"},
        )
        connection.execute(
            text(
                "INSERT INTO user_mcpservers (user_id, mcpserver_id, is_owner, is_active) "
                "VALUES (7, 1, 0, 1)"
            )
        )
        _run(connection, migration, "downgrade")
        assert APP_ID not in _app_ids(connection)
        _run(connection, migration, "upgrade")
        count = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id=:app_id"),
            {"app_id": APP_ID},
        ).scalar_one()
    assert count == 1


def test_upgrade_raises_when_mcp_servers_cannot_prove_ownership(tmp_path):
    """Without url/auth/transport columns a colliding row's provenance cannot
    be established, so the migration must fail loudly rather than guess."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _create_server_tables(connection, with_identity_columns=False)
        connection.execute(
            text("INSERT INTO mcp_servers (id, name) VALUES (1, :name)"),
            {"name": APP_ID},
        )
        with pytest.raises(RuntimeError, match="mcp_servers"):
            _run(connection, migration, "upgrade")
        assert APP_ID not in _app_ids(connection)


def test_upgrade_ignores_unrelated_servers(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _create_server_tables(connection)
        _insert_server(
            connection,
            name="my-other-server",
            url="https://other.example/mcp",
            auth=None,
            owner_user_id=7,
        )
        _run(connection, migration, "upgrade")
        assert APP_ID in _app_ids(connection)


def test_upgrade_refuses_unowned_row_on_a_foreign_transport(tmp_path):
    """Same name and URL but a different HTTP transport is not the row our
    connect path creates (it mirrors the transport check in
    _ensure_catalog_mcp_oauth_server)."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _create_server_tables(connection)
        _insert_server(
            connection,
            name=APP_ID,
            url=migration.ROW["launch_config"]["url"],
            auth={"type": "mcp_oauth"},
            transport="sse",
        )
        with pytest.raises(RuntimeError, match="mcp_servers"):
            _run(connection, migration, "upgrade")
        assert APP_ID not in _app_ids(connection)


def test_upgrade_refuses_unowned_row_carrying_its_own_policy(tmp_path):
    """An ownerless row (its owner account was deleted, or an admin edited it
    after a downgrade) whose name/transport/URL/auth.type all match but that
    carries static headers is a custom server's policy, not our shared row:
    adopting it would send every user's calls with those headers."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _create_server_tables(connection)
        _insert_server(
            connection,
            name=APP_ID,
            url=migration.ROW["launch_config"]["url"],
            auth={"type": "mcp_oauth"},
            headers={"X-Proxy": "attacker"},
        )
        with pytest.raises(RuntimeError, match="mcp_servers"):
            _run(connection, migration, "upgrade")
        assert APP_ID not in _app_ids(connection)
