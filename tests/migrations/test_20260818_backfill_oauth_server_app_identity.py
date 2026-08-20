"""Tests for migration 20260818_backfill_oauth_server_app_identity (#1429).

Following this repo's migration-test convention: the two tables are built
in their pre-migration shape with SQLAlchemy Core, seeded, and only the
migration under test is run against them (via MigrationContext/Operations,
so no full alembic history replay is needed).

What must hold:

- an auth-less oauth row named exactly after an OAuth app is stamped with
  that app's ``app_id`` -- and only that: no ``provider`` is written;
- a provider-only row (blank/absent ``app_id``) is stamped through its
  provider when exactly one OAuth app uses that provider, and left alone
  when the provider is ambiguous — providers are non-unique across apps;
- rows already carrying a nonblank ``app_id`` are untouched (idempotence);
- rows resolving to nothing (orphans, non-oauth-app name matches) are left
  exactly as they were, preserving the name-fallback shim's behavior;
- non-oauth-transport server rows are never candidates, and neither are
  rows whose whole ``auth`` payload is a non-dict or whose stored
  ``provider`` is present but unusable;
- one ``app_id`` never lands on two rows -- including when the colliding
  partner was stamped before the run, or by an earlier run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text

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
    # Uniqueness mirrors production (mcp_servers.name, public_mcp_apps.app_id)
    # so a fixture cannot seed a state the real schema would reject -- which
    # is also what makes the same-name-server cases below honest: they must use
    # distinct names, exactly like production data.
    sa.Table(
        "mcp_servers",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
        sa.Column("transport", sa.String(50), nullable=False),
        sa.Column("auth", sa.JSON, nullable=True),
    )
    sa.Table(
        "public_mcp_apps",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("app_id", sa.String(100), nullable=False, unique=True),
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
    assert auth == {"app_id": "acme-drive"}


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


def test_a_non_dict_auth_payload_is_left_alone(seeded_engine):
    """Garbage (a scalar/list auth on an oauth row) is not a candidate: such a
    row resolves fine today through the name fallback, and this migration is
    irreversible, so destroying a value it does not have to touch is the wrong
    default."""
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

    assert _auth_by_name(engine, metadata)["Acme Drive"] == "garbage-string"


def test_offline_sql_mode_raises_instead_of_stamping():
    """Alembic emits alembic_version bookkeeping even for an empty migration
    body, so a silent offline no-op would advance the version while touching
    no row — permanently skipping the backfill on the next online upgrade.
    The offline branch must fail loudly instead, for both dialects.

    Scope: this pins the raise, which is the whole guard. The bookkeeping
    hazard itself lives in Alembic's env.py/CLI path, which this harness does
    not drive — an end-to-end `alembic upgrade --sql` test would need a real
    config and script directory, and the raise here is what makes that path
    unreachable in the first place."""
    module = _migration_module()
    for dialect in ("sqlite", "postgresql"):
        migration_context = MigrationContext.configure(
            dialect_name=dialect, opts={"as_sql": True}
        )
        with (
            Operations.context(migration_context),
            pytest.raises(RuntimeError, match="offline"),
        ):
            module.upgrade()


def test_duplicate_exact_names_are_ambiguous_and_resolve_nothing(seeded_engine):
    """Two OAuth apps sharing one exact display name poison that name key: an
    auth-less row under it stays untouched rather than being stamped with
    whichever app happened to be seeded first."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": None,
            },
            {
                "app_id": "acme-drive-eu",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": None,
            },
        ],
        servers=[{"name": "Acme Drive", "transport": "oauth", "auth": None}],
    )

    _run_upgrade(engine)

    assert _auth_by_name(engine, metadata)["Acme Drive"] is None


def test_a_non_string_app_id_is_malformed_and_restamped(seeded_engine):
    """get_app_for_mcp_server rejects a non-string auth.app_id outright, so a
    row carrying one is permanently unresolvable — it must stay a candidate
    and a successful name match overwrites the malformed value."""
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
        servers=[{"name": "Acme Drive", "transport": "oauth", "auth": {"app_id": 123}}],
    )

    _run_upgrade(engine)

    auth = _auth_by_name(engine, metadata)["Acme Drive"]
    assert auth == {"app_id": "acme-drive"}


def test_identities_are_matched_raw_never_trimmed(seeded_engine):
    """Identities are opaque and every read path compares them exactly, so the
    migration must not trim: a padded catalog app_id is stamped verbatim
    (that raw value is what get_app_by_id can resolve), and a
    whitespace-variant row name does not match at all — under-matching leaves
    the name-fallback shim in charge."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": " acme-drive ",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            }
        ],
        servers=[
            {"name": "Acme Drive", "transport": "oauth", "auth": None},
            {"name": "Acme Drive ", "transport": "oauth", "auth": None},
        ],
    )

    _run_upgrade(engine)

    auth = _auth_by_name(engine, metadata)
    assert auth["Acme Drive"] == {"app_id": " acme-drive "}
    assert auth["Acme Drive "] is None


def test_one_app_is_never_stamped_onto_two_rows(seeded_engine):
    """_ensure_user_mcp_server creates a *new* row when a rename makes the old
    one unfindable, so two rows for one app are this migration's own target
    population. Stamping both would give them one identity, and
    _lookup_oauth_server_for_app reads an unordered query — configure and
    disconnect would then pick between them nondeterministically. Name
    evidence wins; the provider-matched row keeps its pre-migration
    behavior."""
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
            # Matches by name.
            {"name": "Acme Drive", "transport": "oauth", "auth": None},
            # Would match the same app by provider — the orphan left behind by
            # a rename.
            {
                "name": "Acme Drive (old)",
                "transport": "oauth",
                "auth": {"provider": "acme"},
            },
        ],
    )

    _run_upgrade(engine)

    auth = _auth_by_name(engine, metadata)
    assert auth["Acme Drive"] == {"app_id": "acme-drive"}
    assert auth["Acme Drive (old)"] == {"provider": "acme"}


def test_name_evidence_wins_regardless_of_row_order(seeded_engine):
    """The same collision with the provider-matched row seeded *first*: which
    row wins must follow the evidence, not insertion order."""
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
                "name": "Acme Drive (old)",
                "transport": "oauth",
                "auth": {"provider": "acme"},
            },
            {"name": "Acme Drive", "transport": "oauth", "auth": None},
        ],
    )

    _run_upgrade(engine)

    auth = _auth_by_name(engine, metadata)
    assert auth["Acme Drive"] == {"app_id": "acme-drive"}
    assert auth["Acme Drive (old)"] == {"provider": "acme"}


def test_an_ambiguous_name_still_gets_the_provider_fallback(seeded_engine):
    """Poisoning is per *key*, not per row: an ambiguous name carries no
    information, so a row under it still resolves through an unambiguous
    provider. The two same-named apps differ in provider, so only one is
    reachable that way."""
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
                "app_id": "other-drive",
                "name": "Acme Drive",
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

    assert _auth_by_name(engine, metadata)["Acme Drive"] == {
        "app_id": "other-drive",
        "provider": "other",
    }


def test_lookup_map_construction_skips_and_folds_the_right_fields(seeded_engine):
    """Three map-building branches at once: an app with a blank app_id is
    skipped entirely, an app with a blank name contributes to the provider map
    only, and app-side `transport` is case-folded (it is a shape enum, not an
    identity)."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            # Blank app_id: unusable as an identity, contributes nothing.
            {
                "app_id": "   ",
                "name": "Blank Id App",
                "transport": "oauth",
                "provider_name": "blank",
            },
            # Blank name: reachable by provider only.
            {
                "app_id": "nameless-app",
                "name": "   ",
                "transport": "oauth",
                "provider_name": "nameless",
            },
            # Mixed-case transport still counts as oauth.
            {
                "app_id": "cased-app",
                "name": "Cased App",
                "transport": "OAuth",
                "provider_name": None,
            },
        ],
        servers=[
            {"name": "Blank Id App", "transport": "oauth", "auth": None},
            {
                "name": "Whatever",
                "transport": "oauth",
                "auth": {"provider": "nameless"},
            },
            {"name": "Cased App", "transport": "oauth", "auth": None},
        ],
    )

    _run_upgrade(engine)

    auth = _auth_by_name(engine, metadata)
    assert auth["Blank Id App"] is None
    assert auth["Whatever"] == {"provider": "nameless", "app_id": "nameless-app"}
    # No provider on the app, so only app_id is written.
    assert auth["Cased App"] == {"app_id": "cased-app"}


def test_unrelated_auth_keys_survive_the_stamp(seeded_engine):
    """Only app_id is added: everything already in the auth dict -- including
    a blank provider and an encrypted secret -- is carried through
    untouched."""
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
                "name": "Acme Drive",
                "transport": "oauth",
                "auth": {"provider": "  ", "access_token": "encrypted-blob"},
            }
        ],
    )

    _run_upgrade(engine)

    assert _auth_by_name(engine, metadata)["Acme Drive"] == {
        "app_id": "acme-drive",
        "provider": "  ",
        "access_token": "encrypted-blob",
    }


def test_a_pre_stamped_row_blocks_a_second_row_claiming_its_app(seeded_engine):
    """The collision the two-pass fix alone did not close: the partner row was
    stamped *before* this run (by the writer, or by an earlier run), so it
    never enters the candidate set. Its identity must still be claimed, or the
    orphan gets the same app_id and _lookup_oauth_server_for_app picks between
    them nondeterministically."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive v2",
                "transport": "oauth",
                "provider_name": "acme",
            }
        ],
        servers=[
            # Already carries the identity — the row the writer created on
            # rename, and not a candidate.
            {
                "name": "Acme Drive v2",
                "transport": "oauth",
                "auth": {"app_id": "acme-drive", "provider": "acme"},
            },
            # The orphan left under the old name; only its provider resolves.
            {
                "name": "Acme Drive",
                "transport": "oauth",
                "auth": {"provider": "acme"},
            },
        ],
    )

    _run_upgrade(engine)

    auth = _auth_by_name(engine, metadata)
    assert auth["Acme Drive v2"] == {"app_id": "acme-drive", "provider": "acme"}
    assert auth["Acme Drive"] == {"provider": "acme"}


def test_a_stamp_survives_a_rerun_unchanged(seeded_engine):
    """Idempotence across runs, not just within one: run 1 stamps the name
    match, and run 2 -- which is what a downgrade (a no-op) plus upgrade
    produces -- must neither restamp it nor hand its identity to the orphan
    that lost the first time."""
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
            {"name": "Acme Drive", "transport": "oauth", "auth": None},
            {
                "name": "Acme Drive (old)",
                "transport": "oauth",
                "auth": {"provider": "acme"},
            },
        ],
    )

    _run_upgrade(engine)
    after_first = _auth_by_name(engine, metadata)

    module = _migration_module()
    module.downgrade()
    _run_upgrade(engine)

    assert _auth_by_name(engine, metadata) == after_first
    assert after_first["Acme Drive"] == {"app_id": "acme-drive"}
    assert after_first["Acme Drive (old)"] == {"provider": "acme"}


def test_a_name_owned_by_a_non_oauth_app_refuses_the_provider_fallback(
    seeded_engine,
):
    """A name matching a *non-OAuth* app is evidence about this row, not an
    absence of evidence. Treating the two alike would let the row fall through
    to the provider fallback and be stamped with an unrelated app's id."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-notes",
                "name": "Acme Notes",
                "transport": "stdio",
                "provider_name": "acme",
            },
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            },
        ],
        servers=[
            {
                "name": "Acme Notes",
                "transport": "oauth",
                "auth": {"provider": "acme"},
            }
        ],
    )

    _run_upgrade(engine)

    assert _auth_by_name(engine, metadata)["Acme Notes"] == {"provider": "acme"}


def test_a_malformed_non_string_provider_refuses_the_row(seeded_engine):
    """A provider that is not a string at all cannot be compared by the
    conflict gate, and _is_oauth_server_for_app would reject the row against
    any app once stamped — so a stamp buys nothing and the row is left
    alone."""
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
                "name": "Acme Drive",
                "transport": "oauth",
                "auth": {"provider": ["acme"]},
            }
        ],
    )

    _run_upgrade(engine)

    assert _auth_by_name(engine, metadata)["Acme Drive"] == {"provider": ["acme"]}


def test_two_provider_only_rows_resolve_deterministically(seeded_engine):
    """A tie the evidence cannot break: two unstamped rows whose only signal
    is the same provider. Exactly one may be stamped, and candidates are read
    ordered by id so it is the same one on every run and every backend."""
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
            {"name": "orphan-a", "transport": "oauth", "auth": {"provider": "acme"}},
            {"name": "orphan-b", "transport": "oauth", "auth": {"provider": "acme"}},
        ],
    )

    _run_upgrade(engine)

    auth = _auth_by_name(engine, metadata)
    assert auth["orphan-a"] == {"app_id": "acme-drive", "provider": "acme"}
    assert auth["orphan-b"] == {"provider": "acme"}


def test_downgrade_is_a_no_op(seeded_engine):
    """Documented as irreversible: the downgrade must not attempt to strip
    stamps it cannot tell apart from the writer's own."""
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
    before = _auth_by_name(engine, metadata)

    _migration_module().downgrade()

    assert _auth_by_name(engine, metadata) == before


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
            # mcp_servers.name is unique in production, so the two shapes must
            # be seeded under distinct names; the stdio row is a candidate by
            # neither transport nor name, and the oauth row's name resolves
            # only a stdio app, which is the cross-shape refusal.
            {"name": "Acme Notes Local", "transport": "stdio", "auth": None},
            {"name": "Acme Notes", "transport": "oauth", "auth": None},
        ],
    )

    _run_upgrade(engine)

    servers = metadata.tables["mcp_servers"]
    with engine.connect() as conn:
        rows = conn.execute(sa.select(servers)).all()
    assert all(row.auth is None for row in rows)


def _postgres_url() -> str | None:
    import os

    return os.environ.get("XAGENT_TEST_POSTGRES_URL")


@pytest.fixture
def postgres_seeded_engine():
    """The same two tables on a real PostgreSQL server.

    The chain-level CI job upgrades an *empty* database, so without this the
    backfill's data path would never run on PostgreSQL at all -- and JSON
    round-tripping and the ordered read are exactly the parts that could
    differ from SQLite.
    """
    url = _postgres_url()
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    engine = create_engine(url)
    metadata = _pre_migration_metadata()
    with engine.begin() as conn:
        for table in ("mcp_servers", "public_mcp_apps"):
            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
    metadata.create_all(bind=engine)
    try:
        yield engine, metadata
    finally:
        with engine.begin() as conn:
            for table in ("mcp_servers", "public_mcp_apps"):
                conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
        engine.dispose()


@pytest.mark.postgresql
def test_the_backfill_runs_on_postgresql(postgres_seeded_engine):
    """Stamp, refusal and collision behavior on the real backend: the JSON
    write round-trips, and the pre-stamped partner still blocks the orphan."""
    engine, metadata = postgres_seeded_engine
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
                "provider_name": "mail",
            },
        ],
        servers=[
            {"name": "Acme Mail", "transport": "oauth", "auth": None},
            {
                "name": "Acme Drive v2",
                "transport": "oauth",
                "auth": {"app_id": "acme-drive", "provider": "acme"},
            },
            {
                "name": "Acme Drive",
                "transport": "oauth",
                "auth": {"provider": "acme"},
            },
            {"name": "orphan", "transport": "oauth", "auth": None},
        ],
    )

    _run_upgrade(engine)

    auth = _auth_by_name(engine, metadata)
    assert auth["Acme Mail"] == {"app_id": "acme-mail"}
    assert auth["Acme Drive v2"] == {"app_id": "acme-drive", "provider": "acme"}
    # Its app_id is already carried by the row above.
    assert auth["Acme Drive"] == {"provider": "acme"}
    assert auth["orphan"] is None
