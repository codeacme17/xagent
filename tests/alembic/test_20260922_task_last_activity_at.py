"""The retention anchor's migration: backfill correctness and idempotence.

Runs on SQLite and on a disposable PostgreSQL through the shared ``engine``
fixture. Both matter: the backfill's correlated ``MAX()`` subquery and its
``UPDATE ... WHERE id IN (SELECT ... LIMIT)`` batching are the two constructs
most likely to behave differently across the pair.
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa
from alembic import op
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.web.services.task_database_shared import engine as engine_fixture

engine = engine_fixture

MODULE = "xagent.migrations.versions.20260922_task_last_activity_at"

NOW = datetime(2026, 9, 22, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def migration():
    return importlib.import_module(MODULE)


@contextmanager
def _migration_env(engine):
    """Drive the revision the way ``env.py`` drives it.

    Not ``engine.begin()``. A connection already inside a transaction when
    ``MigrationContext.configure`` runs is recorded as an *external*
    transaction, and ``begin_transaction()`` then returns a nullcontext
    without creating ``_transaction`` -- so ``autocommit_block()`` asserts on
    a transaction it was never given. ``env.py`` commits the reflection
    transaction before ``configure()`` for exactly this reason, and this
    revision's PostgreSQL path depends on that block, so the harness has to
    mirror it or it tests a shape production never runs.
    """
    with engine.connect() as connection:
        context = MigrationContext.configure(connection)
        with Operations.context(context), context.begin_transaction():
            yield connection


def _schema(*, with_messages: bool) -> tuple[sa.MetaData, sa.Table, sa.Table | None]:
    """The pre-migration shape of the two tables the backfill reads.

    Deliberately minimal rather than the real metadata: the migration must work
    against a database whose ``tasks`` table predates every column this PR
    adds, which is exactly what an upgrade encounters.
    """
    metadata = sa.MetaData()
    tasks = sa.Table(
        "tasks",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    messages = None
    if with_messages:
        messages = sa.Table(
            "task_chat_messages",
            metadata,
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column(
                "task_id", sa.Integer, sa.ForeignKey("tasks.id", ondelete="CASCADE")
            ),
            sa.Column("created_at", sa.DateTime(timezone=True)),
        )
    return metadata, tasks, messages


def _anchors(connection: sa.Connection) -> dict[int, datetime | None]:
    # Typed columns, not a bare ``text()``: SQLite hands back the stored
    # string unless SQLAlchemy is told the column is a DateTime, and the
    # whole point of these assertions is to compare instants.
    rows = connection.execute(
        sa.select(
            sa.column("id", sa.Integer),
            sa.column("last_activity_at", sa.DateTime(timezone=True)),
        )
        .select_from(sa.table("tasks"))
        .order_by(sa.column("id"))
    ).all()
    return {
        row[0]: (
            row[1].replace(tzinfo=timezone.utc)
            if row[1] is not None and row[1].tzinfo is None
            else row[1]
        )
        for row in rows
    }


def test_harness_supplies_what_an_autocommit_block_needs(engine):
    """Pin the harness contract, not just the migration's.

    This revision's PostgreSQL path runs the backfill inside
    ``autocommit_block()``, and that block commits the transaction Alembic
    owns. A harness that hands ``MigrationContext`` a connection already in
    someone else's transaction gets a nullcontext from
    ``begin_transaction()`` instead, and the block then asserts on a
    ``_transaction`` it never received.

    SQLite cannot catch that through ``upgrade()``, because the dialect
    branch skips the block there -- which is how a harness bug reached CI as
    eight PostgreSQL failures while every local run was green. The assertion
    itself is dialect-independent, so entering a block directly pins it here.
    """
    with _migration_env(engine) as connection:
        with op.get_context().autocommit_block():
            connection.execute(sa.text("SELECT 1"))


def _has_anchor_column(connection: sa.Connection) -> bool:
    return any(
        c["name"] == "last_activity_at"
        for c in sa.inspect(connection).get_columns("tasks")
    )


def test_backfill_uses_last_message_and_falls_back_to_created_at(engine, migration):
    """Anchor = last message, or the task's own creation time when it has none.

    The fallback is not cosmetic: a NULL anchor compares as NULL against any
    cutoff, so a message-less task left NULL would never expire.
    """
    metadata, tasks, messages = _schema(with_messages=True)
    metadata.create_all(engine)
    with _migration_env(engine) as connection:
        connection.execute(
            tasks.insert(),
            [
                {"id": 1, "created_at": NOW - timedelta(days=500)},
                {"id": 2, "created_at": NOW - timedelta(days=300)},
            ],
        )
        connection.execute(
            messages.insert(),
            [
                {"id": 10, "task_id": 1, "created_at": NOW - timedelta(days=400)},
                {"id": 11, "task_id": 1, "created_at": NOW - timedelta(days=120)},
            ],
        )

        migration.upgrade()

        anchors = _anchors(connection)
        # Task 1 has messages: the most recent one wins, not the first.
        assert anchors[1] == NOW - timedelta(days=120)
        # Task 2 never carried a message: it ages from its own creation.
        assert anchors[2] == NOW - timedelta(days=300)


def test_upgrade_is_idempotent_and_finishes_a_partial_backfill(engine, migration):
    """Re-running must not re-stamp finished rows, but must finish unfinished ones.

    A run that added the column and died leaves NULLs behind, so the backfill
    is unconditional rather than gated on the column being new. That makes the
    "already done" case a no-op instead of a rewrite.
    """
    metadata, tasks, messages = _schema(with_messages=True)
    metadata.create_all(engine)
    with _migration_env(engine) as connection:
        connection.execute(
            tasks.insert(), {"id": 1, "created_at": NOW - timedelta(days=9)}
        )
        connection.execute(
            messages.insert(),
            {"id": 10, "task_id": 1, "created_at": NOW - timedelta(days=5)},
        )
        migration.upgrade()
        after_first = _anchors(connection)

        # A row that arrives (or is reset) after the first run.
        connection.execute(
            tasks.insert(), {"id": 2, "created_at": NOW - timedelta(days=3)}
        )
        migration.upgrade()
        after_second = _anchors(connection)

        assert after_second[1] == after_first[1]
        assert after_second[2] == NOW - timedelta(days=3)

        # A third run changes nothing at all.
        migration.upgrade()
        assert _anchors(connection) == after_second


def test_backfill_pages_past_one_batch(engine, migration, monkeypatch):
    """The loop must converge on a set larger than one batch, not stop at it."""
    monkeypatch.setattr(migration, "BACKFILL_BATCH_SIZE", 3)
    metadata, tasks, _ = _schema(with_messages=False)
    metadata.create_all(engine)
    with _migration_env(engine) as connection:
        connection.execute(
            tasks.insert(),
            [
                {"id": index, "created_at": NOW - timedelta(days=index)}
                for index in range(1, 11)
            ],
        )
        migration.upgrade()
        anchors = _anchors(connection)
        assert len(anchors) == 10
        assert all(value is not None for value in anchors.values())


def test_column_is_added_without_a_server_default(engine, migration):
    """A default would stamp existing rows with the migration's own clock.

    That is the value the backfill exists to avoid: it would push every
    historical task's expiry out by a full retention period.
    """
    metadata, _, _ = _schema(with_messages=False)
    metadata.create_all(engine)
    with _migration_env(engine) as connection:
        migration.upgrade()
        column = next(
            c
            for c in sa.inspect(connection).get_columns("tasks")
            if c["name"] == "last_activity_at"
        )
        assert column["default"] is None
        assert column["nullable"] is True


def test_downgrade_removes_the_column_and_is_repeatable(engine, migration):
    metadata, _, _ = _schema(with_messages=False)
    metadata.create_all(engine)
    with _migration_env(engine) as connection:
        migration.upgrade()
        assert any(
            c["name"] == "last_activity_at"
            for c in sa.inspect(connection).get_columns("tasks")
        )
        migration.downgrade()
        migration.downgrade()
        assert not any(
            c["name"] == "last_activity_at"
            for c in sa.inspect(connection).get_columns("tasks")
        )
        migration.upgrade()
        assert any(
            c["name"] == "last_activity_at"
            for c in sa.inspect(connection).get_columns("tasks")
        )


def test_alembic_only_missing_tables_are_left_to_metadata(engine, migration):
    """No ``tasks`` table means an Alembic-only install; metadata owns it."""
    with _migration_env(engine) as connection:
        migration.upgrade()
        assert not sa.inspect(connection).has_table("tasks")
        migration.downgrade()


def test_backfill_without_the_messages_table_falls_back_to_created_at(
    engine, migration
):
    """``tasks`` can exist before ``task_chat_messages`` in an Alembic-only run."""
    metadata, tasks, _ = _schema(with_messages=False)
    metadata.create_all(engine)
    with _migration_env(engine) as connection:
        connection.execute(
            tasks.insert(), {"id": 1, "created_at": NOW - timedelta(days=7)}
        )
        migration.upgrade()
        assert _anchors(connection)[1] == NOW - timedelta(days=7)


def test_backfill_converges_when_no_source_can_supply_an_instant(engine, migration):
    """A row with nothing to anchor on must terminate the loop, not spin on it.

    Selecting the remaining NULL set on every pass looks equivalent to paging
    and is not: a row written to NULL stays in that set and is handed back
    forever, so the revision dies at the batch ceiling and rolls back --
    deterministically, on every retry. The row is left NULL, which the
    predicate reads as "no anchor" and refuses to expire.
    """
    metadata, tasks, messages = _schema(with_messages=True)
    metadata.create_all(engine)
    with _migration_env(engine) as connection:
        connection.execute(
            tasks.insert(),
            [
                {"id": 1, "created_at": None},
                {"id": 2, "created_at": NOW - timedelta(days=3)},
            ],
        )
        migration.upgrade()

        anchors = _anchors(connection)
        assert anchors[1] is None
        assert anchors[2] == NOW - timedelta(days=3)

        # And a message arriving later still anchors the unanchored row.
        connection.execute(
            messages.insert(),
            {"id": 10, "task_id": 1, "created_at": NOW - timedelta(days=1)},
        )
        migration.upgrade()
        assert _anchors(connection)[1] == NOW - timedelta(days=1)


def test_upgrade_survives_a_tasks_table_with_no_created_at(engine, migration):
    """A legacy ``tasks`` shape must not take the whole upgrade chain down.

    ``tasks`` predates most of its columns, and tests/migration/test_migration.py
    builds exactly this shape. Naming ``tasks.created_at`` unconditionally
    fails the statement and every later revision with it.
    """
    metadata = sa.MetaData()
    sa.Table(
        "tasks",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("source", sa.String(20)),
    )
    metadata.create_all(engine)
    with _migration_env(engine) as connection:
        connection.execute(sa.text("INSERT INTO tasks (id, source) VALUES (1, 'sdk')"))
        migration.upgrade()
        assert _has_anchor_column(connection)
        # Nothing to derive an anchor from, so the row stays unanchored
        # rather than being stamped with a guess.
        assert _anchors(connection)[1] is None


def test_backfill_uses_messages_when_tasks_has_no_created_at(engine, migration):
    """Missing ``tasks.created_at`` must not disable the message source too."""
    metadata = sa.MetaData()
    sa.Table("tasks", metadata, sa.Column("id", sa.Integer, primary_key=True))
    sa.Table(
        "task_chat_messages",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("task_id", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    metadata.create_all(engine)
    with _migration_env(engine) as connection:
        connection.execute(sa.text("INSERT INTO tasks (id) VALUES (1)"))
        connection.execute(
            sa.text(
                "INSERT INTO task_chat_messages (id, task_id, created_at) "
                "VALUES (10, 1, :ts)"
            ),
            {"ts": NOW - timedelta(days=2)},
        )
        migration.upgrade()
        assert _anchors(connection)[1] == NOW - timedelta(days=2)


def test_postgresql_backfills_outside_the_add_column_transaction(
    migration, monkeypatch
):
    """The ALTER's ACCESS EXCLUSIVE lock must not span the backfill.

    ``env.py`` gives PostgreSQL one transaction per migration, so without the
    autocommit block ``ALTER TABLE ... ADD COLUMN`` holds ACCESS EXCLUSIVE on
    ``tasks`` until the backfill finishes -- every task read and write in the
    deployment blocks behind it, for as long as the table is large. There is
    no assertion a SQLite run can make about that, so this pins the dialect
    branch instead: the one thing between the fix and a silent regression.
    """
    entered: list[str] = []

    class _Block:
        def __enter__(self):
            entered.append("in")

        def __exit__(self, *exc):
            entered.append("out")
            return False

    class _Context:
        as_sql = False

        def autocommit_block(self):
            return _Block()

    class _Dialect:
        name = "postgresql"

    class _Bind:
        dialect = _Dialect()

    class _Inspector:
        def has_table(self, name):
            return name == "tasks"

        def get_columns(self, name):
            return [
                {"name": "id"},
                {"name": "created_at"},
                {"name": "last_activity_at"},
            ]

    monkeypatch.setattr(migration.op, "get_context", _Context)
    monkeypatch.setattr(migration.op, "get_bind", _Bind)
    monkeypatch.setattr(migration.sa, "inspect", lambda _bind: _Inspector())
    monkeypatch.setattr(
        migration, "_backfill", lambda *a, **k: entered.append("backfill") or 0
    )

    migration.upgrade()
    assert entered == ["in", "backfill", "out"], entered


def test_sqlite_backfills_inside_the_migration_transaction(migration, monkeypatch):
    """SQLite shares one transaction across the whole chain; do not break it.

    Entering an autocommit block here would commit the chain mid-way, trading
    a lock problem this dialect does not have for a partial-upgrade one it
    does.
    """
    calls: list[str] = []

    class _Context:
        as_sql = False

        def autocommit_block(self):  # pragma: no cover - must not be reached
            raise AssertionError("SQLite must not break the chain transaction")

    class _Dialect:
        name = "sqlite"

    class _Bind:
        dialect = _Dialect()

    class _Inspector:
        def has_table(self, name):
            return name == "tasks"

        def get_columns(self, name):
            return [
                {"name": "id"},
                {"name": "created_at"},
                {"name": "last_activity_at"},
            ]

    monkeypatch.setattr(migration.op, "get_context", _Context)
    monkeypatch.setattr(migration.op, "get_bind", _Bind)
    monkeypatch.setattr(migration.sa, "inspect", lambda _bind: _Inspector())
    monkeypatch.setattr(
        migration, "_backfill", lambda *a, **k: calls.append("backfill") or 0
    )

    migration.upgrade()
    assert calls == ["backfill"]


def test_offline_mode_emits_ddl_and_skips_the_backfill(migration, monkeypatch):
    """``--sql`` cannot read rows back, so it must emit DDL and stop.

    Offline SQL generation is a supported path in this repo, and a
    MockConnection cannot be inspected -- reaching the inspector at all would
    raise rather than render.
    """
    calls: list[str] = []

    class _Context:
        as_sql = True

    monkeypatch.setattr(migration.op, "get_context", _Context)
    monkeypatch.setattr(
        migration.op, "add_column", lambda *a, **k: calls.append("add_column")
    )
    monkeypatch.setattr(
        migration.sa,
        "inspect",
        lambda _bind: pytest.fail("offline mode must not inspect"),
    )
    monkeypatch.setattr(
        migration, "_backfill", lambda *a, **k: pytest.fail("offline must not backfill")
    )

    migration.upgrade()
    assert calls == ["add_column"]


def test_revision_chains_onto_the_previous_head(migration):
    """A second head would break the repository's single-head invariant."""
    assert migration.revision == "20260922_task_last_activity_at"
    assert migration.down_revision == "20260922_seed_rocketlane_mcp_app"
