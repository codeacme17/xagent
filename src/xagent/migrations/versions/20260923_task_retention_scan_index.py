"""add the retention candidate scan index (#2563)

#2571 shipped ``tasks.last_activity_at`` with no index, on the stated grounds
that the set-scanning consumer is this purge. This is that index.

Why it indexes an expression
----------------------------
The scan does not filter on ``last_activity_at``. It filters on
``retention_anchor()``, which is ``COALESCE(last_activity_at, created_at)``
(``services/task_retention.py``) -- the column is nullable, and a NULL anchor
compared against a cutoff yields NULL, which every WHERE clause reads as "not
eligible", so the COALESCE is what stops un-backfilled rows being immortal.

A btree on the raw column cannot range-bound a COALESCE of it, so an index on
``(status, last_activity_at, ...)`` would reduce to its ``status`` prefix --
which ``ix_tasks_status_lease_expires_at`` already provides -- while adding
write cost on a column every persisted message touches. The first version of
this migration made exactly that mistake; it was caught in review on #2599.

Column order follows the query's shape: ``status`` is the equality-shaped leg
(an ``IN`` over two labels), so it leads and leaves the expression as the
ranged one. The lease leg is left out: it is an ``OR`` over NULL and a
comparison, which a btree cannot serve as a trailing column anyway, and it
eliminates almost nothing after the other two.

Not partial. A ``WHERE status IN (...)`` predicate would shrink it, but the
labels ``tasks.status`` stores are a deployment-visible contract, and an index
predicate that stops matching the query's own silently stops being used.

**Unverified:** this shape was derived from the predicate, not from an EXPLAIN
against a populated table. No PostgreSQL was available when it was written.
The open question an EXPLAIN would settle is not whether the expression is
range-bounded -- it is -- but whether the planner prefers this index at all
over the primary key, since the scan also carries ``id > :cursor`` and
``ORDER BY id`` and can satisfy both from the PK alone. Whoever first runs
this against real data should check that, and drop the index rather than keep
a version of it that is never chosen.

CONCURRENTLY, and what a failed build leaves
--------------------------------------------
A plain ``CREATE INDEX`` takes SHARE on ``tasks`` for the whole build,
blocking every write in the deployment. ``CONCURRENTLY`` trades that for two
table passes and the inability to run inside a transaction, hence
``autocommit_block()`` -- in the offline path too, where Alembic otherwise
emits ``BEGIN; ... COMMIT;`` around the migration and PostgreSQL rejects the
statement outright. Non-PostgreSQL dialects get a plain indexed build, because
``CONCURRENTLY`` is not their syntax at all.

A ``CONCURRENTLY`` build that does not finish leaves an **invalid** index
rather than nothing: unusable by the planner, and not replaced by
``CREATE INDEX ... IF NOT EXISTS``. So the online path asks
``pg_index.indisvalid`` rather than asking whether the name exists, and drops
the leftover before rebuilding. ``docs/deployment.md`` carries the operator
procedure.

Structure, the validity query and the drop-then-create ordering all follow
``20260725_add_task_lease_recovery_index.py``, which solved this same problem
for ``ix_tasks_status_lease_expires_at``.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260923_task_retention_scan_index"
down_revision: Union[str, None] = "20260924_hide_google_drive_until_picker"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "tasks"
INDEX = "ix_tasks_retention_scan"

#: Columns the index reads. Every one must exist before the index can be
#: built -- an Alembic-only or partially-migrated schema simply gets nothing.
REQUIRED_COLUMNS = ("status", "last_activity_at", "created_at")

#: The anchor expression, spelled exactly as ``retention_anchor()`` renders it
#: and as ``Task.__table_args__`` declares it. All three must agree or the
#: planner will not use this index for that query; the model contract test
#: pins the pair that can be compared programmatically.
ANCHOR_SQL = "coalesce(last_activity_at, created_at)"

#: ``to_regclass`` resolves through ``search_path`` exactly as the DDL that
#: created the index did, so this cannot match a same-named index in an
#: unrelated schema. Returns NULL when nothing resolves, so a missing index
#: and an invalid one are distinguishable.
POSTGRES_INDEX_VALIDITY_SQL = sa.text(
    """
    SELECT i.indisvalid
    FROM pg_catalog.pg_index AS i
    WHERE i.indexrelid = pg_catalog.to_regclass(:index_name)
    """
)


def _index_elements() -> list:
    """The index's elements: the status column, then the anchor expression."""
    return [sa.column("status"), sa.text(ANCHOR_SQL)]


def _online_columns() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names():
        return set()
    return {item["name"] for item in inspector.get_columns(TABLE)}


def _online_index_definition(index_name: str) -> tuple[str, ...] | None:
    """What an existing index actually covers, or ``None`` if it is absent.

    Reflected expression indexes report ``None`` in ``column_names`` and carry
    the rendered text in ``expressions``; a plain column index reports the
    name in both. Normalizing through ``expressions`` therefore compares like
    with like whichever form the database holds -- which is the point, because
    the shape this migration replaces was a plain-column index of the same
    name.
    """
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names():
        return None
    for item in inspector.get_indexes(TABLE):
        if item.get("name") != index_name:
            continue
        elements = item.get("expressions") or item.get("column_names") or []
        return tuple(str(element).strip().lower() for element in elements)
    return None


def _expected_definition() -> tuple[str, ...]:
    """The definition an existing index must match to be left alone.

    Consumed only by the PostgreSQL path: it is the one dialect whose
    reflection reports an expression index at all.
    """
    return ("status", ANCHOR_SQL)


def _postgres_index_validity() -> bool | None:
    """Whether the current-schema index exists and is usable."""
    return (
        op.get_bind()
        .execute(POSTGRES_INDEX_VALIDITY_SQL, {"index_name": INDEX})
        .scalar_one_or_none()
    )


def _upgrade_postgresql() -> None:
    if not set(REQUIRED_COLUMNS).issubset(_online_columns()):
        return

    validity = _postgres_index_validity()
    existing = _online_index_definition(INDEX)
    if validity is True and existing == _expected_definition():
        return

    with op.get_context().autocommit_block():
        # Drops both leftovers this can meet: an invalid corpse from a failed
        # concurrent build, and a valid index of the same name covering the
        # wrong thing -- which is what an installation that ran the first
        # version of this revision holds.
        if validity is not None or existing is not None:
            op.drop_index(
                INDEX,
                table_name=TABLE,
                if_exists=True,
                postgresql_concurrently=True,
            )
        op.create_index(
            INDEX,
            TABLE,
            _index_elements(),
            if_not_exists=True,
            postgresql_concurrently=True,
        )


def upgrade() -> None:
    context = op.get_context()
    if context.as_sql:
        if context.dialect.name == "postgresql":
            # autocommit_block renders as COMMIT; <ddl>; BEGIN; so the
            # concurrent build lands outside the migration's transaction.
            with context.autocommit_block():
                op.create_index(
                    INDEX,
                    TABLE,
                    _index_elements(),
                    postgresql_concurrently=True,
                )
        else:
            op.create_index(INDEX, TABLE, _index_elements())
        return

    if context.dialect.name == "postgresql":
        _upgrade_postgresql()
        return

    if not set(REQUIRED_COLUMNS).issubset(_online_columns()):
        return
    # Drop-then-create rather than compare-then-act, because on SQLite the
    # comparison is impossible: SQLAlchemy skips reflection of expression
    # indexes outright ("Skipped unsupported reflection of expression-based
    # index"), so _online_index_definition reports this index as absent even
    # when it exists -- and a bare create then fails with "already exists".
    # Dropping by name first is idempotent, and it also replaces the
    # wrong-shaped index an installation of this revision's first version
    # holds. Affordable here in a way it would not be on PostgreSQL: this
    # branch serves development databases, where the table is small and no
    # concurrent writer is waiting on the lock.
    op.drop_index(INDEX, table_name=TABLE, if_exists=True)
    op.create_index(INDEX, TABLE, _index_elements(), if_not_exists=True)


def downgrade() -> None:
    context = op.get_context()
    is_postgresql = context.dialect.name == "postgresql"
    if context.as_sql:
        if is_postgresql:
            with context.autocommit_block():
                op.drop_index(
                    INDEX,
                    table_name=TABLE,
                    postgresql_concurrently=True,
                )
        else:
            op.drop_index(INDEX, table_name=TABLE)
        return

    if TABLE not in sa.inspect(op.get_bind()).get_table_names():
        return
    if is_postgresql:
        with op.get_context().autocommit_block():
            op.drop_index(
                INDEX,
                table_name=TABLE,
                if_exists=True,
                postgresql_concurrently=True,
            )
        return
    # if_exists rather than a reflected check, for the reason in upgrade().
    op.drop_index(INDEX, table_name=TABLE, if_exists=True)
