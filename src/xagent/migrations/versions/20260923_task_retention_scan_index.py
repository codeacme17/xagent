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
``(status, last_activity_at, ...)`` would reduce to its ``status`` prefix,
which ``ix_tasks_status_lease_expires_at`` already provides. An earlier draft
of this migration indexed the raw column and was caught in review on #2599.

What this index costs, and why the raw-column argument did not distinguish it
-----------------------------------------------------------------------------
That earlier draft was also rejected here for "adding write cost on a column
every persisted message touches". **This index pays exactly that cost**, and
saying otherwise was the weaker half of the argument: the expression
references ``last_activity_at`` too.

Concretely, no index on ``tasks`` referenced that column before this
migration, so the ``touch_task_last_activity`` UPDATE behind every persisted
chat message was HOT-eligible -- the new tuple needed no index entry at all.
With this index it is not: each of those UPDATEs now maintains *every* index
on ``tasks``, not just this one, and leaves a dead tuple the same page-level
cleanup has to reclaim.

So the trade is not "cheaper than the raw-column index". It is: non-HOT anchor
updates on the hottest write path in this table, in exchange for the only
index shape that can range-bound the scan's predicate. Whether that trade pays
depends entirely on the planner actually choosing it, which
``test_the_purge_scan_uses_this_index`` now checks against a real server
rather than leaving to reasoning.

Column order follows the query's shape: ``status`` is the equality-shaped leg
(an ``IN`` over two labels), so it leads and leaves the expression as the
ranged one. The lease leg is left out: it is an ``OR`` over NULL and a
comparison, which a btree cannot serve as a trailing column anyway, and it
eliminates almost nothing after the other two.

Not partial. A ``WHERE status IN (...)`` predicate would shrink it, but the
labels ``tasks.status`` stores are a deployment-visible contract, and an index
predicate that stops matching the query's own silently stops being used.

The shape was derived from the predicate, not from an EXPLAIN -- no
PostgreSQL was available when it was written -- so the question it left open
is whether the planner prefers this index over the primary key at all, since
the scan also carries ``id > :cursor`` and ``ORDER BY id`` and can satisfy
both from the PK alone. That question is now asked in CI. If the answer ever
becomes "no", this index should be dropped rather than kept in a shape nothing
chooses, because the write cost above is paid either way.

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

#: What a reflected index must match to be left alone. Consumed only by the
#: PostgreSQL path: it is the one dialect whose reflection reports an
#: expression index at all.
EXPECTED_DEFINITION: tuple[str, ...] = ("status", ANCHOR_SQL)

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
    if validity is True and existing == EXPECTED_DEFINITION:
        return

    with op.get_context().autocommit_block():
        # Drops both leftovers this can meet: an invalid corpse from a failed
        # concurrent build, and a valid index of this name covering something
        # else. The second is defensive rather than observed -- an index of
        # this name can only come from this revision, and a database that ran
        # an earlier form of it is already stamped here, so upgrade() would
        # not run again. It costs one comparison and removes a silent
        # divergence if one is ever created by hand.
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
    # Dropping by name first is idempotent, and replaces any index of this
    # name whose shape differs. Affordable here in a way it would not be on
    # PostgreSQL: this
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
