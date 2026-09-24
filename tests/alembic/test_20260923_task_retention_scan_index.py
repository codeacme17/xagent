"""Tests for the retention candidate scan index migration (#2563).

Structure follows ``test_20260725_add_task_lease_recovery_index.py``, whose
migration solved the same problem for ``ix_tasks_status_lease_expires_at``.
Three things here are specific to this one and are where the tests concentrate:

* the index carries an **expression**, so the model contract cannot be checked
  by comparing column names, and SQLAlchemy's SQLite reflection skips such an
  index entirely -- which is why the online SQLite path drops before creating
  rather than comparing first;
* the offline rendering must put ``CONCURRENTLY`` *outside* the migration's
  transaction on PostgreSQL and must not emit it at all elsewhere. Both were
  wrong in an earlier draft of this migration and are pinned here;
* an index of this name may exist with the wrong shape. No installation is
  expected to hold one -- the name can only come from this revision, and a
  database that ran an earlier draft is already stamped here -- so this is a
  defensive path, pinned because it is cheap to pin.
"""

import importlib.util
from contextlib import nullcontext
from io import StringIO
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from xagent.web.models.task import Task

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20260923_task_retention_scan_index.py"
)
REVISION = "20260923_task_retention_scan_index"
DOWN_REVISION = "20260924_hide_google_drive_until_picker"
TABLE = "tasks"
INDEX = "ix_tasks_retention_scan"

#: The shape an earlier draft of this migration built. Used to exercise the
#: replace path, not because an installation is expected to hold it.
SUPERSEDED_INDEX_SQL = (
    f"CREATE INDEX {INDEX} ON {TABLE} (status, last_activity_at, lease_expires_at)"
)


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "task_retention_scan_index_migration", MIGRATION_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection) -> Operations:
    return Operations(MigrationContext.configure(connection))


def _transactional_offline_sql(migration, dialect_name: str, operation: str) -> str:
    """Render offline SQL the way Alembic's own offline run does.

    ``begin_transaction()`` is the part that matters: it is what wraps the
    migration in ``BEGIN``/``COMMIT`` on a transactional-DDL dialect, and
    therefore what a bare ``CREATE INDEX CONCURRENTLY`` would land inside.
    """
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect_name,
        opts={"as_sql": True, "output_buffer": output},
    )
    with Operations.context(context), context.begin_transaction():
        getattr(migration, operation)()
    return output.getvalue()


def _create_tasks_table(connection) -> None:
    connection.execute(
        sa.text(
            "CREATE TABLE tasks ("
            "id INTEGER PRIMARY KEY, "
            "status VARCHAR(32), "
            "last_activity_at DATETIME, "
            "created_at DATETIME, "
            "lease_expires_at DATETIME"
            ")"
        )
    )


def _index_sql(connection, name: str = INDEX) -> str | None:
    """Read the index's DDL back from SQLite.

    Reflection cannot be used: SQLAlchemy skips expression-based indexes on
    this dialect, which is the same fact the migration's online path is built
    around. ``sqlite_master`` still holds the statement.
    """
    return connection.execute(
        sa.text("SELECT sql FROM sqlite_master WHERE type='index' AND name=:name"),
        {"name": name},
    ).scalar_one_or_none()


def test_revision_metadata() -> None:
    migration = _load_migration_module()

    assert migration.revision == REVISION
    assert migration.down_revision == DOWN_REVISION


def test_task_model_and_migration_share_the_same_index_contract() -> None:
    """The model must declare it, and declare the same thing.

    Both halves matter. A fresh install never runs migrations -- it stamps
    head and builds the schema from this metadata -- so an index only the
    migration knows about would exist on upgraded deployments and nowhere
    else. And the expression has to match the migration's text exactly, or
    the two installations end up with indexes the planner treats differently.
    """
    migration = _load_migration_module()
    model_index = next(
        (index for index in Task.__table__.indexes if index.name == INDEX), None
    )

    assert model_index is not None, "a fresh install would never get this index"
    # The column element renders qualified (``tasks.status``) while the text
    # element renders as written; the prefix is stripped so the comparison is
    # about what is indexed rather than how SQLAlchemy spells it.
    rendered = [
        str(expression).strip().lower().replace("tasks.", "")
        for expression in model_index.expressions
    ]
    assert rendered == ["status", migration.ANCHOR_SQL]


def test_the_indexed_expression_matches_the_predicate_it_serves() -> None:
    """The index is pointless unless it spells the anchor the scan filters on.

    ``retention_anchor()`` is the authority; this asserts against its rendered
    SQL rather than against a repeated string literal, so changing the anchor
    fails here instead of silently leaving an index the planner ignores.
    """
    migration = _load_migration_module()
    from xagent.web.services.task_retention import retention_anchor

    rendered = str(
        retention_anchor().compile(compile_kwargs={"literal_binds": True})
    ).lower()

    assert rendered.replace("tasks.", "") == migration.ANCHOR_SQL


def test_online_upgrade_creates_the_index_idempotently() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_tasks_table(connection)
        operations = _operations(connection)
        with Operations.context(operations.get_context()):
            migration.upgrade()
            migration.upgrade()

        assert "coalesce(last_activity_at, created_at)" in (
            _index_sql(connection) or ""
        )


def test_online_upgrade_replaces_the_superseded_index_shape() -> None:
    """An early adopter holds a same-named index over the raw column.

    Leaving it would be worse than having none: the name is taken, so a bare
    create fails, and the planner cannot use it for the COALESCE predicate.
    """
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_tasks_table(connection)
        connection.execute(sa.text(SUPERSEDED_INDEX_SQL))
        operations = _operations(connection)
        with Operations.context(operations.get_context()):
            migration.upgrade()

        assert _index_sql(connection) == (
            f"CREATE INDEX {INDEX} ON {TABLE} "
            "(status, coalesce(last_activity_at, created_at))"
        )


@pytest.mark.parametrize(
    "schema",
    [
        None,
        "CREATE TABLE tasks (id INTEGER PRIMARY KEY, status VARCHAR(32))",
        "CREATE TABLE tasks (id INTEGER PRIMARY KEY, last_activity_at DATETIME)",
    ],
)
def test_online_upgrade_noops_without_tasks_or_required_columns(
    schema: str | None,
) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        if schema is not None:
            connection.execute(sa.text(schema))
        operations = _operations(connection)
        with Operations.context(operations.get_context()):
            migration.upgrade()

        if schema is not None:
            assert _index_sql(connection) is None


def test_online_downgrade_drops_the_index() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        _create_tasks_table(connection)
        operations = _operations(connection)
        with Operations.context(operations.get_context()):
            migration.upgrade()
            migration.downgrade()
            migration.downgrade()

        assert _index_sql(connection) is None


@pytest.mark.parametrize("operation", ["upgrade", "downgrade"])
def test_postgresql_offline_puts_concurrent_ddl_outside_the_transaction(
    operation: str,
) -> None:
    """The defect an earlier draft of this migration shipped.

    Rendered inside ``BEGIN``/``COMMIT``, PostgreSQL rejects the statement
    outright -- and because the rejection happens inside the transaction, the
    whole script applies nothing. The COMMIT before the statement is what the
    ``autocommit_block`` exists to emit.
    """
    migration = _load_migration_module()

    sql = _transactional_offline_sql(migration, "postgresql", operation)

    concurrent_line = next(
        line for line in sql.splitlines() if "CONCURRENTLY" in line.upper()
    )
    before = sql[: sql.index(concurrent_line)]
    assert before.rstrip().endswith("COMMIT;"), sql
    assert "BEGIN;" in sql[sql.index(concurrent_line) :], sql


@pytest.mark.parametrize("dialect", ["sqlite", "mysql"])
@pytest.mark.parametrize("operation", ["upgrade", "downgrade"])
def test_non_postgresql_offline_emits_portable_sql(
    dialect: str, operation: str
) -> None:
    """``CONCURRENTLY`` is not a keyword these dialects have."""
    migration = _load_migration_module()

    sql = _transactional_offline_sql(migration, dialect, operation)

    assert "CONCURRENTLY" not in sql.upper(), sql
    assert INDEX in sql


def test_postgresql_online_upgrade_rebuilds_an_invalid_concurrent_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed concurrent build leaves a corpse that IF NOT EXISTS would keep.

    Pinned as a statement sequence rather than against a live server: what
    matters is that an invalid index is dropped before the rebuild, and that
    both statements stay concurrent.
    """
    migration = _load_migration_module()
    context = MigrationContext.configure(dialect_name="postgresql")
    operations = Operations(context)
    calls: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        migration, "_online_columns", lambda: set(migration.REQUIRED_COLUMNS)
    )
    monkeypatch.setattr(migration, "_postgres_index_validity", lambda: False)
    monkeypatch.setattr(
        migration,
        "_online_index_definition",
        lambda _name: ("status", migration.ANCHOR_SQL),
    )
    monkeypatch.setattr(context, "autocommit_block", nullcontext)
    monkeypatch.setattr(
        operations, "drop_index", lambda *a, **kw: calls.append(("drop", kw))
    )
    monkeypatch.setattr(
        operations, "create_index", lambda *a, **kw: calls.append(("create", kw))
    )

    with Operations.context(context):
        monkeypatch.setattr(migration, "op", operations)
        migration.upgrade()

    assert calls == [
        (
            "drop",
            {"table_name": TABLE, "if_exists": True, "postgresql_concurrently": True},
        ),
        ("create", {"if_not_exists": True, "postgresql_concurrently": True}),
    ]


def test_postgresql_online_upgrade_leaves_a_valid_matching_index_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-running must not rebuild: a concurrent build is two table passes."""
    migration = _load_migration_module()
    context = MigrationContext.configure(dialect_name="postgresql")
    operations = Operations(context)
    calls: list[str] = []

    monkeypatch.setattr(
        migration, "_online_columns", lambda: set(migration.REQUIRED_COLUMNS)
    )
    monkeypatch.setattr(migration, "_postgres_index_validity", lambda: True)
    monkeypatch.setattr(
        migration,
        "_online_index_definition",
        lambda _name: ("status", migration.ANCHOR_SQL),
    )
    monkeypatch.setattr(context, "autocommit_block", nullcontext)
    monkeypatch.setattr(operations, "drop_index", lambda *a, **kw: calls.append("drop"))
    monkeypatch.setattr(
        operations, "create_index", lambda *a, **kw: calls.append("create")
    )

    with Operations.context(context):
        monkeypatch.setattr(migration, "op", operations)
        migration.upgrade()

    assert calls == []


def test_postgresql_online_upgrade_replaces_a_valid_wrong_shaped_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid, present, and covering the superseded columns: still rebuilt."""
    migration = _load_migration_module()
    context = MigrationContext.configure(dialect_name="postgresql")
    operations = Operations(context)
    calls: list[str] = []

    monkeypatch.setattr(
        migration, "_online_columns", lambda: set(migration.REQUIRED_COLUMNS)
    )
    monkeypatch.setattr(migration, "_postgres_index_validity", lambda: True)
    monkeypatch.setattr(
        migration,
        "_online_index_definition",
        lambda _name: ("status", "last_activity_at", "lease_expires_at"),
    )
    monkeypatch.setattr(context, "autocommit_block", nullcontext)
    monkeypatch.setattr(operations, "drop_index", lambda *a, **kw: calls.append("drop"))
    monkeypatch.setattr(
        operations, "create_index", lambda *a, **kw: calls.append("create")
    )

    with Operations.context(context):
        monkeypatch.setattr(migration, "op", operations)
        migration.upgrade()

    assert calls == ["drop", "create"]


# ---------------------------------------------------------------------------
# Against a real PostgreSQL. Everything above either mocks the reflection or
# runs on SQLite, which cannot reflect an expression index at all -- so the
# comparison that decides between "leave alone" and "rebuild", and the
# question of whether the planner uses this index, are only answerable here.
# ---------------------------------------------------------------------------


@pytest.fixture
def postgres_engine():
    from tests.shared.postgres_disposable import disposable_database_factory

    with disposable_database_factory("retention_scan_index") as make:
        yield make("index")


def _apply_migration(engine, migration, operation: str = "upgrade") -> list[str]:
    """Run one direction against a live database, returning the DDL it issued.

    Statements are captured rather than inferred, because the property under
    test is "the second run issues none" -- which no amount of reading the
    final schema can distinguish from "it rebuilt the same thing".
    """
    issued: list[str] = []

    @sa.event.listens_for(engine, "before_cursor_execute")
    def record(_conn, _cursor, statement, _params, _context, _executemany):
        if statement.strip().upper().startswith(("CREATE INDEX", "DROP INDEX")):
            issued.append(" ".join(statement.split()))

    try:
        with engine.connect() as connection:
            # Mirrors env.py's online run, including its own comment's reason:
            # SQLAlchemy 2.0 autobegins a transaction on first use, and
            # ``autocommit_block`` refuses to run inside a transaction Alembic
            # did not open. So close any implicit one, then let Alembic own
            # the boundary through ``begin_transaction()`` -- which is what
            # lets the block commit it, switch to AUTOCOMMIT for the concurrent
            # build, and start a fresh one after. The first version of this
            # helper skipped both steps and failed inside ``autocommit_block``
            # before the migration issued anything.
            if connection.in_transaction():
                connection.commit()
            context = MigrationContext.configure(connection)
            with Operations.context(context), context.begin_transaction():
                getattr(migration, operation)()
    finally:
        sa.event.remove(engine, "before_cursor_execute", record)
    return issued


@pytest.mark.postgresql
def test_postgresql_upgrade_is_idempotent_against_a_real_catalog(
    postgres_engine,
) -> None:
    """The reflection and normalisation path, unmocked.

    The three mocked tests above pin the decision *given* a definition; this
    pins the definition itself. A mismatch between what `get_indexes` reports
    and what `EXPECTED_DEFINITION` spells -- a different quoting, a case
    difference, an extra space -- would make every run rebuild the index
    concurrently, or never replace a wrong one, and no mocked test could tell.
    """
    from xagent.web.models.database import Base

    migration = _load_migration_module()
    Base.metadata.drop_all(postgres_engine)
    Base.metadata.create_all(postgres_engine)
    with postgres_engine.connect() as connection:
        connection.execute(sa.text(f"DROP INDEX IF EXISTS {INDEX}"))
        connection.commit()

    first = _apply_migration(postgres_engine, migration)
    second = _apply_migration(postgres_engine, migration)

    assert any("CREATE INDEX" in statement for statement in first), first
    assert second == [], (
        "the second upgrade issued DDL, so the reflected definition does not "
        f"match EXPECTED_DEFINITION: {second}"
    )


@pytest.mark.postgresql
def test_the_purge_scan_uses_this_index(postgres_engine) -> None:
    """The question this index exists to answer, asked of the planner.

    Everything else about the shape was derived from the predicate. Whether
    PostgreSQL actually prefers it over the primary key is not derivable --
    the scan also carries ``id > :cursor`` and ``ORDER BY id``, which the PK
    alone can satisfy. If this fails, the index is not earning the non-HOT
    anchor updates it costs and should be dropped rather than kept.

    The query is captured from ``select_purge_candidates`` rather than
    rewritten here, so what is explained is what the purge runs.
    """
    from datetime import datetime, timedelta, timezone

    from xagent.web.models.database import Base
    from xagent.web.models.task import Task, TaskStatus
    from xagent.web.models.user import User
    from xagent.web.services.task_retention_purge import select_purge_candidates

    now = datetime.now(timezone.utc)
    Base.metadata.drop_all(postgres_engine)
    # ``create_all`` builds the index from the model declaration, which is the
    # shape a fresh install gets -- so this also checks that the declaration
    # and the migration describe the same thing well enough to be planned on.
    Base.metadata.create_all(postgres_engine)

    # Enough rows, and enough of them ineligible, that a sequential scan is
    # not simply the cheaper plan. A planner choice on ten rows says nothing.
    with sa.orm.Session(postgres_engine) as db:
        user = User(username="scan-plan", password_hash="unused")
        db.add(user)
        db.flush()
        db.add_all(
            Task(
                user_id=int(user.id),
                title=f"task-{index}",
                status=TaskStatus.COMPLETED if index % 2 else TaskStatus.RUNNING,
                last_activity_at=now - timedelta(days=400 if index % 4 == 1 else 1),
            )
            for index in range(5000)
        )
        db.commit()
    with postgres_engine.connect() as connection:
        connection.execute(sa.text("ANALYZE tasks"))
        connection.commit()

    captured: list[tuple[str, object]] = []

    @sa.event.listens_for(postgres_engine, "before_cursor_execute")
    def record(_conn, _cursor, statement, params, _context, _executemany):
        if "FROM tasks" in statement and "LIMIT" in statement.upper():
            captured.append((statement, params))

    try:
        with sa.orm.Session(postgres_engine) as db:
            select_purge_candidates(
                db, now=now, conversation_days=365, trace_days=90, limit=100
            )
    finally:
        sa.event.remove(postgres_engine, "before_cursor_execute", record)

    assert captured, "the scan query was not captured"
    statement, params = captured[-1]
    with postgres_engine.connect() as connection:
        # ``exec_driver_sql``, not ``text()``: the captured statement is in the
        # driver's own paramstyle (``%(name)s`` for psycopg2), which ``text()``
        # does not parse -- it expects ``:name`` and passes the rest through,
        # so the first version of this test sent a literal ``%`` to the server.
        plan = "\n".join(
            row[0] for row in connection.exec_driver_sql(f"EXPLAIN {statement}", params)
        )

    assert INDEX in plan, (
        "the planner did not choose ix_tasks_retention_scan for the purge's "
        "candidate scan, so the index is not earning its write cost:\n" + plan
    )
