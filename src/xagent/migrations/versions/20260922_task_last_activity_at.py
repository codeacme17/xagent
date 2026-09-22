"""Add the conversation retention anchor ``tasks.last_activity_at`` (#2562).

The column is added without a server default on purpose. A default would make
``ALTER TABLE`` stamp every pre-existing row with this migration's own clock --
the exact value the backfill below exists to avoid, and one that would push
every historical task's expiry out by the full retention period.

Three properties the backfill is shaped around:

* **It derives the anchor only from columns that exist.** ``tasks`` predates
  most of its current columns, and a partially-migrated or Alembic-only
  database can present a ``tasks`` table with no ``created_at`` at all (see
  the legacy shape built in tests/migration/test_migration.py). Naming a
  column that is not there fails the statement and takes the whole upgrade
  chain down with it, so each source is probed before it is used.

* **It leaves a row NULL rather than inventing a timestamp**, and NULL is
  much weaker protection than it looks. ``retention_anchor()`` is
  ``COALESCE(last_activity_at, created_at)``, so a NULL anchor beside a live
  ``created_at`` is not refused -- it expires the task from its creation date,
  which for a long conversation is far too early. NULL is only harmless where
  ``tasks.created_at`` does not exist either, and there it is harmless for a
  blunt reason rather than a designed one: the predicate cannot run against
  such a schema at all, because the mapped column it selects is not there.

  So the invariant this revision must hold is not "NULL is safe" but "NULL
  must never mean the backfill has not run". Wherever an instant exists, the
  upgrade writes it before the revision is stamped; where no source can
  supply one, the row stays NULL on a schema the predicate cannot read. The
  offline path refuses outright rather than stamping without filling.

* **It pages forward by primary key.** Selecting the remaining NULL set on
  every pass looks equivalent and is not: a row written to NULL stays in that
  set, so the loop re-selects it forever and the revision dies at the batch
  ceiling. The cursor advances whatever value was written, which also keeps
  the walk linear instead of re-scanning the rows already done.

On PostgreSQL the backfill runs inside ``autocommit_block()``. Without it
the ``ADD COLUMN`` and every batch share one transaction (``env.py`` sets
``transaction_per_migration`` for this dialect), and ``ALTER TABLE ... ADD
COLUMN`` holds ACCESS EXCLUSIVE on ``tasks`` until that transaction commits --
so every task read and write in the deployment, API calls and lease
heartbeats alike, would block for the length of the backfill rather than for
the length of the ALTER. The block commits the ALTER first and then commits
each batch, which is also what makes a re-run meaningful: an interrupted
backfill leaves committed progress and NULLs behind, and the next upgrade
finishes them. SQLite keeps the single-transaction behaviour, because there
the whole migration chain shares one transaction and breaking it mid-chain
would trade a lock problem this dialect does not have for a partial-upgrade
problem it does.

The backfill runs on every upgrade rather than only when the column is
created, because a database whose column arrived from ``create_all`` still
needs filling, and because an interrupted PostgreSQL run must be able to
resume.
"""

import sqlalchemy as sa
from alembic import op

revision = "20260922_task_last_activity_at"
down_revision = "20260921_word_contract"
branch_labels = None
depends_on = None

#: Rows per UPDATE. Bounded so one statement cannot take an unbounded lock
#: footprint over a multi-million-row ``tasks`` table.
BACKFILL_BATCH_SIZE = 5000


def _has_column(inspector: sa.Inspector, table: str, column: str) -> bool:
    if not inspector.has_table(table):
        return False
    return any(c["name"] == column for c in inspector.get_columns(table))


def _anchor_expression(*, with_messages: bool, with_created_at: bool) -> str | None:
    """The SQL for one row's anchor, using only sources that exist.

    ``None`` means no source is available, which is not an error. Such a
    database has no instant to offer, and it is also one the retention
    predicate cannot query: ``retention_anchor()`` selects ``tasks.created_at``
    through the mapped column, so a schema without it raises rather than
    returning an anchor. Leaving these rows NULL is therefore moot, not a
    safety property to rely on elsewhere.
    """
    message_max = "(SELECT MAX(m.created_at) FROM task_chat_messages m WHERE m.task_id = tasks.id)"
    if with_messages and with_created_at:
        return f"COALESCE({message_max}, tasks.created_at)"
    if with_messages:
        return message_max
    if with_created_at:
        return "tasks.created_at"
    return None


def _backfill(connection: sa.Connection, *, anchor_sql: str) -> None:
    """Fill every NULL anchor, paging forward by primary key.

    Terminates because the cursor only moves forward: a batch that writes
    NULL still advances it, where a re-select of the NULL set would hand the
    same row back on every pass. Deliberately returns nothing -- a count
    would have to say whether it meant rows selected or rows written, and
    since the guard below can skip a row those differ.
    """
    select_batch = sa.text(
        "SELECT id FROM tasks "
        "WHERE id > :after AND last_activity_at IS NULL "
        "ORDER BY id LIMIT :batch"
    )
    # ``AND last_activity_at IS NULL`` repeats the SELECT's predicate on
    # purpose; it is not redundant. The ids were chosen by an earlier
    # statement, and on PostgreSQL each batch commits separately, so a
    # transcript writer can advance a selected task's anchor in between. A
    # blocked UPDATE re-checks its search condition against the *target* row
    # once the lock is released, but the ``MAX()`` subquery reads
    # ``task_chat_messages`` -- other rows -- and keeps the original command
    # snapshot, so without this guard the backfill would overwrite the newer
    # anchor with a value computed before that message existed. With it, a row
    # someone else has already anchored is simply skipped: touches only ever
    # move forward, so their value is the better one.
    #
    # noqa: S608 is on the UPDATE below: `anchor_sql` comes only from
    # _anchor_expression above, which returns one of three literals. No
    # caller-supplied text reaches it, and every value is bound.
    update_batch = sa.text(
        f"UPDATE tasks SET last_activity_at = {anchor_sql} "  # noqa: S608
        "WHERE id IN :ids AND last_activity_at IS NULL"
    ).bindparams(sa.bindparam("ids", expanding=True))

    after = -1
    while True:
        ids = [
            row[0]
            for row in connection.execute(
                select_batch, {"after": after, "batch": BACKFILL_BATCH_SIZE}
            )
        ]
        if not ids:
            return
        connection.execute(update_batch, {"ids": ids})
        after = ids[-1]


def upgrade() -> None:
    # ``op.get_context().as_sql`` rather than ``context.is_offline_mode()``:
    # the latter needs the EnvironmentContext proxy, which only an env.py run
    # establishes, so it cannot be reached when a test drives upgrade()
    # through Operations.context(). Both read the same flag.
    if op.get_context().as_sql:
        # Refuse, rather than render. This revision cannot be applied offline
        # safely, and every alternative was worse:
        #
        # * Emitting only the ADD COLUMN stamps ``alembic_version`` too, so a
        #   later online ``upgrade head`` considers the revision applied and
        #   never backfills. Every historical task keeps a NULL anchor, which
        #   ``retention_anchor()`` resolves to ``created_at`` -- reporting a
        #   task created 400 days ago with a message from last week as expired
        #   at 90 days. That is the data-loss path this refusal closes.
        #
        # * Emitting the backfill too puts a full-table correlated UPDATE
        #   inside the same transaction as the ALTER, because offline mode has
        #   no per-migration boundaries and no ``autocommit_block``. On
        #   PostgreSQL the rendered script is BEGIN / ALTER / UPDATE / COMMIT,
        #   so ACCESS EXCLUSIVE on ``tasks`` is held for the whole scan -- the
        #   outage the online path takes care to avoid, reintroduced in a form
        #   the operator cannot split. It also has to name
        #   ``task_chat_messages`` and ``tasks.created_at`` unconditionally,
        #   since offline cannot inspect, so against the legacy shapes the
        #   online path handles it fails and rolls back the whole chain.
        #
        # Refusing costs little: offline generation across the full chain
        # already fails earlier in this repository (20260317 reflects through
        # a MockConnection), so this is not a working path being withdrawn.
        raise RuntimeError(
            "20260922_task_last_activity_at cannot be generated with --sql. "
            "It backfills tasks.last_activity_at row by row, and an offline "
            "script would stamp the revision without running that backfill, "
            "leaving historical tasks with an anchor that reads as their "
            "creation date and expires them early. Run this revision online "
            "(alembic upgrade head against the database)."
        )

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    # ``tasks`` is metadata-owned and can be absent in Alembic-only runs.
    if not inspector.has_table("tasks"):
        return
    if not _has_column(inspector, "tasks", "last_activity_at"):
        op.add_column(
            "tasks",
            sa.Column("last_activity_at", sa.DateTime(timezone=True), nullable=True),
        )
        inspector = sa.inspect(bind)

    anchor_sql = _anchor_expression(
        with_messages=_has_column(inspector, "task_chat_messages", "created_at"),
        with_created_at=_has_column(inspector, "tasks", "created_at"),
    )
    if anchor_sql is None:
        return

    if bind.dialect.name == "postgresql":
        # Commits the ADD COLUMN, and with it the ACCESS EXCLUSIVE lock,
        # before the batches start. See the module docstring.
        with op.get_context().autocommit_block():
            _backfill(bind, anchor_sql=anchor_sql)
    else:
        _backfill(bind, anchor_sql=anchor_sql)


def downgrade() -> None:
    # Unlike upgrade(), this renders offline: dropping the column needs no
    # row work, so the emitted statement is the whole of the revision.
    if op.get_context().as_sql:
        op.drop_column("tasks", "last_activity_at")
        return

    inspector = sa.inspect(op.get_bind())
    if _has_column(inspector, "tasks", "last_activity_at"):
        op.drop_column("tasks", "last_activity_at")
