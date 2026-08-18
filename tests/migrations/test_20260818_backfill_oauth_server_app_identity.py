"""Tests for migration 20260818_backfill_oauth_server_app_identity (#1429).

Following this repo's migration-test convention: the two tables are built
in their pre-migration shape with SQLAlchemy Core, seeded, and only the
migration under test is run against them (via MigrationContext/Operations,
so no full alembic history replay is needed).

What must hold:

- an auth-less oauth row named exactly after an OAuth app is stamped with
  that app's ``app_id`` (and ``provider`` when the app has one);
- a provider-only row (blank/absent ``app_id``) is stamped through its
  provider when exactly one OAuth app uses that provider, and left alone
  when the provider is ambiguous — providers are non-unique across apps;
- rows already carrying a nonblank ``app_id`` are untouched (idempotence);
- rows resolving to nothing (orphans, non-oauth-app name matches) are left
  exactly as they were, preserving the name-fallback shim's behavior;
- non-oauth-transport server rows are never candidates.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine

TARGET_REVISION = "20260818_backfill_oauth_server_app_identity"


def _migration_module() -> ModuleType:
    import xagent.migrations as migrations_pkg

    migrations_dir = Path(next(iter(migrations_pkg.__path__)))
    path = migrations_dir / "versions" / f"{TARGET_REVISION}.py"
    spec = importlib.util.spec_from_file_location(TARGET_REVISION, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pre_migration_metadata() -> sa.MetaData:
    """The two tables reduced to the columns the migration reads/writes."""
    metadata = sa.MetaData()
    sa.Table(
        "mcp_servers",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("transport", sa.String(50), nullable=False),
        sa.Column("auth", sa.JSON, nullable=True),
    )
    sa.Table(
        "public_mcp_apps",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("app_id", sa.String(100), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("transport", sa.String(50), nullable=False),
        sa.Column("provider_name", sa.String(50), nullable=True),
    )
    return metadata


@pytest.fixture()
def seeded_engine():
    engine = create_engine("sqlite:///:memory:")
    metadata = _pre_migration_metadata()
    metadata.create_all(engine)
    try:
        yield engine, metadata
    finally:
        engine.dispose()


def _run_upgrade(engine) -> None:
    module = _migration_module()
    with engine.begin() as conn:
        migration_context = MigrationContext.configure(conn)
        with Operations.context(migration_context):
            module.upgrade()


def _auth_by_name(engine, metadata) -> dict[str, object]:
    servers = metadata.tables["mcp_servers"]
    with engine.connect() as conn:
        return {row.name: row.auth for row in conn.execute(sa.select(servers)).all()}


def _seed(engine, metadata, apps: list[dict], servers: list[dict]) -> None:
    with engine.begin() as conn:
        if apps:
            conn.execute(metadata.tables["public_mcp_apps"].insert(), apps)
        if servers:
            conn.execute(metadata.tables["mcp_servers"].insert(), servers)


def test_an_auth_less_row_is_stamped_by_exact_display_name(seeded_engine):
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            }
        ],
        servers=[{"name": "Acme Drive", "transport": "oauth", "auth": None}],
    )

    _run_upgrade(engine)

    auth = _auth_by_name(engine, metadata)["Acme Drive"]
    assert auth == {"app_id": "acme-drive", "provider": "acme"}


def test_a_provider_only_row_is_stamped_when_the_provider_is_unambiguous(
    seeded_engine,
):
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            }
        ],
        servers=[
            {
                "name": "Old Acme Name",
                "transport": "oauth",
                "auth": {"app_id": "   ", "provider": "acme"},
            }
        ],
    )

    _run_upgrade(engine)

    auth = _auth_by_name(engine, metadata)["Old Acme Name"]
    assert auth["app_id"] == "acme-drive"
    # The row's own provider spelling is kept, not overwritten.
    assert auth["provider"] == "acme"


def test_an_ambiguous_provider_resolves_nothing(seeded_engine):
    """The meta/Instagram case: two apps on one provider. Guessing would
    attribute another connector's identity, so the row stays unstamped and
    keeps the name-fallback shim's behavior."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            },
            {
                "app_id": "acme-mail",
                "name": "Acme Mail",
                "transport": "oauth",
                "provider_name": "acme",
            },
        ],
        servers=[
            {
                "name": "Old Acme Name",
                "transport": "oauth",
                "auth": {"provider": "acme"},
            }
        ],
    )

    _run_upgrade(engine)

    assert _auth_by_name(engine, metadata)["Old Acme Name"] == {"provider": "acme"}


def test_rows_already_stamped_and_orphans_are_untouched(seeded_engine):
    engine, metadata = seeded_engine
    stamped = {"app_id": "acme-drive", "provider": "acme", "extra": "kept"}
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            }
        ],
        servers=[
            {"name": "Acme Drive", "transport": "oauth", "auth": dict(stamped)},
            {"name": "no-such-app", "transport": "oauth", "auth": None},
        ],
    )

    _run_upgrade(engine)
    # Idempotence: a second run changes nothing either.
    _run_upgrade(engine)

    auth = _auth_by_name(engine, metadata)
    assert auth["Acme Drive"] == stamped
    assert auth["no-such-app"] is None


def test_a_provider_conflict_refuses_the_name_match(seeded_engine):
    """A row named after app A whose own auth.provider names a different
    provider is the conflict _ensure_server_matches_oauth_app refuses with a
    ValueError. Stamping A's app_id would create a row *no* app claims — it
    fails A's provider gate and the other provider's app_id gate — so the
    migration refuses too, leaving the read-time provider fallback exactly as
    it was."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            },
            {
                "app_id": "other-app",
                "name": "Other App",
                "transport": "oauth",
                "provider_name": "other",
            },
        ],
        servers=[
            {
                "name": "Acme Drive",
                "transport": "oauth",
                "auth": {"provider": "other"},
            }
        ],
    )

    _run_upgrade(engine)

    # Untouched: the conflicting evidence is left for a human (or the
    # provisioning writer's own refusal path) to resolve. NOTE the provider
    # fallback did not fire either — the name match short-circuits before it,
    # and resurrecting it here would guess between two contradictory signals.
    assert _auth_by_name(engine, metadata)["Acme Drive"] == {"provider": "other"}


def test_a_non_dict_auth_payload_is_replaced_wholesale(seeded_engine):
    """Garbage (a scalar/list auth on an oauth row) is replaced by the stamped
    identity, matching the provisioning writer, which discards non-dict auth
    the same way."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            }
        ],
        servers=[
            {"name": "Acme Drive", "transport": "oauth", "auth": "garbage-string"}
        ],
    )

    _run_upgrade(engine)

    auth = _auth_by_name(engine, metadata)["Acme Drive"]
    assert auth == {"app_id": "acme-drive", "provider": "acme"}


def test_offline_sql_mode_emits_nothing(seeded_engine):
    """A data migration cannot read rows through a MockConnection; the offline
    branch is a documented no-op for both dialects."""
    from io import StringIO

    module = _migration_module()
    for dialect in ("sqlite", "postgresql"):
        output = StringIO()
        migration_context = MigrationContext.configure(
            dialect_name=dialect,
            opts={"as_sql": True, "output_buffer": output},
        )
        with Operations.context(migration_context):
            module.upgrade()
        assert output.getvalue() == ""


def test_non_oauth_shapes_are_never_candidates(seeded_engine):
    """A stdio server row named like an app, and an oauth row named after a
    *non-oauth* app, are both left alone: the first is a different transport,
    the second would stamp a cross-shape identity."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-notes",
                "name": "Acme Notes",
                "transport": "stdio",
                "provider_name": None,
            }
        ],
        servers=[
            {"name": "Acme Notes", "transport": "stdio", "auth": None},
            {"name": "Acme Notes", "transport": "oauth", "auth": None},
        ],
    )

    _run_upgrade(engine)

    servers = metadata.tables["mcp_servers"]
    with engine.connect() as conn:
        rows = conn.execute(sa.select(servers)).all()
    assert all(row.auth is None for row in rows)
