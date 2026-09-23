"""Index the retention candidate scan (#2563).

#2562 added ``tasks.last_activity_at`` and deliberately shipped no index for
it: its only consumer was a hand-run preview, and the scan that needs an index
is the purge's. This is that purge, so this is that index.

What it covers
--------------
``select_purge_candidates`` filters on three things and orders by primary key:
terminal ``status``, a clear ``lease_expires_at``, and an aged anchor. The
column order below is chosen for how PostgreSQL can use it: ``status`` is the
only equality-shaped leg (an ``IN`` over two labels), so it leads and leaves
``last_activity_at`` as the ranged one. ``lease_expires_at`` follows as a
filter on the rows that survive. The "no live command" leg is a ``NOT EXISTS``
against another table and is served by ``ix_task_commands_task_order``, which
already exists.

The index is deliberately **not** partial. A ``WHERE status IN (...)``
predicate would shrink it, but the labels ``tasks.status`` stores are a
deployment-visible contract (``task_status_predicate`` exists because that
storage form has been got wrong before), and an index predicate that stops
matching the query's own predicate silently stops being used. A plain
composite index has no such failure mode.

Why CONCURRENTLY, and what that costs
-------------------------------------
A plain ``CREATE INDEX`` takes SHARE on ``tasks`` for the duration of the
build, blocking every INSERT and UPDATE in the deployment -- on a
multi-million-row table that is a write outage measured in minutes.
``CONCURRENTLY`` trades that for two table passes and the inability to run
inside a transaction, which is why this migration uses
``autocommit_block()``: Alembic otherwise wraps each migration in one
(``env.py`` sets ``transaction_per_migration`` for PostgreSQL).

The consequence an operator must know: a ``CONCURRENTLY`` build that fails --
a deadlock, a cancelled session, a crash -- leaves an **invalid** index behind
rather than nothing. It is not used by the planner and cannot be reused by a
retry; it has to be dropped first. ``IF NOT EXISTS`` would happily skip such a
corpse forever, so this migration checks the catalog for validity rather than
for existence, and drops an invalid leftover before rebuilding. See
``docs/deployment.md`` for the operator-facing procedure.

SQLite gets the ordinary indexed build: it has no concurrent variant, no
autocommit block is available mid-chain, and no deployment runs retention on
it (the purge refuses to start there).
"""

import sqlalchemy as sa
from alembic import op

revision = "20260923_task_retention_scan_index"
down_revision = "20260922_task_last_activity_at"
branch_labels = None
depends_on = None

INDEX_NAME = "ix_tasks_retention_scan"
TABLE = "tasks"
COLUMNS = ("status", "last_activity_at", "lease_expires_at")

#: An index row in ``pg_index`` carries ``indisvalid=false`` when a
#: CONCURRENTLY build did not finish. Such an index is invisible to the
#: planner but very much visible to ``CREATE INDEX``, which is why existence
#: alone cannot decide whether to build.
_INVALID_INDEX_SQL = sa.text(
    "SELECT NOT i.indisvalid FROM pg_class c "
    "JOIN pg_index i ON i.indexrelid = c.oid "
    "WHERE c.relname = :name"
)


def _has_column(inspector: sa.Inspector, table: str, column: str) -> bool:
    if not inspector.has_table(table):
        return False
    return any(c["name"] == column for c in inspector.get_columns(table))


def _columns_present(inspector: sa.Inspector) -> bool:
    return all(_has_column(inspector, TABLE, column) for column in COLUMNS)


def upgrade() -> None:
    if op.get_context().as_sql:
        # Offline cannot inspect a catalog, so it renders the concurrent build
        # unconditionally and leaves the invalid-leftover check to the
        # operator running the script -- who is, by construction, reading it.
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {INDEX_NAME} "
            f"ON {TABLE} ({', '.join(COLUMNS)})"
        )
        return

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not _columns_present(inspector):
        # An Alembic-only or partially-migrated schema. The purge cannot run
        # against it either, so there is nothing to index.
        return

    if bind.dialect.name != "postgresql":
        if INDEX_NAME not in {ix["name"] for ix in inspector.get_indexes(TABLE)}:
            op.create_index(INDEX_NAME, TABLE, list(COLUMNS))
        return

    invalid = bind.execute(_INVALID_INDEX_SQL, {"name": INDEX_NAME}).scalar()
    if invalid is False:
        return  # a finished index; nothing to do
    with op.get_context().autocommit_block():
        if invalid is True:
            # A previous CONCURRENTLY build died. Dropping concurrently too,
            # so the retry does not take the lock the concurrent build exists
            # to avoid.
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {INDEX_NAME} "
            f"ON {TABLE} ({', '.join(COLUMNS)})"
        )


def downgrade() -> None:
    if op.get_context().as_sql:
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
        return

    bind = op.get_bind()
    if not sa.inspect(bind).has_table(TABLE):
        return
    if bind.dialect.name == "postgresql":
        with op.get_context().autocommit_block():
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
        return
    if INDEX_NAME in {ix["name"] for ix in sa.inspect(bind).get_indexes(TABLE)}:
        op.drop_index(INDEX_NAME, TABLE)
