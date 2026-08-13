"""convert trace payload columns from json to jsonb on PostgreSQL

``trace_events.data`` was a ``json`` column, which validates syntax at
write time and nothing else: it happily stores the NUL escape and either
half of an unpaired UTF-16 surrogate, both of which ``->>`` then refuses
to convert back to text -- one such row failed three monitoring endpoints
whole-query (#1149). ``jsonb`` decodes escapes to native text at INSERT,
so it rejects those payloads at the source (#1248). The two checkpoint
blob tables carry the same payloads (their rows are hashed slices of the
same trace data), so all three columns convert together:

- ``trace_events.data``
- ``trace_message_blobs.message_data``
- ``trace_checkpoint_blobs.blob_data``

``ALTER ... TYPE jsonb USING <col>::jsonb`` fails on any stored row that
carries one of those escapes, so the online upgrade first rewrites such
rows with the offending code points replaced by U+FFFD -- the same
substitution the write-side sanitizer
(``web/utils/json_payload_sanitizer.py``) applies to new payloads. Rows
are found with the same normalize-then-match steps as the read guard in
``web/api/monitor.py`` (``PG_ESCAPED_BACKSLASH`` and friends -- that file
is the patterns' source of truth and documents why each step is
load-bearing); the actual rewrite happens in Python, where escape
semantics are exact, not with SQL regex surgery on the JSON text.

This migration is PostgreSQL-only. SQLite stores JSON as TEXT and has no
jsonb; both branches are no-ops there. On PostgreSQL the online branch
guards on the table existing (a bare database has none of the three until
``create_all`` runs) and on the column not already being jsonb (a
create_all-first startup builds jsonb directly from the model, and this
keeps the upgrade idempotent on rerun).

The offline (--sql) branch emits only the three ALTERs: the cleanup step
must read rows, which a MockConnection cannot. An offline script applied
to a database still holding an unconvertible row fails on that ALTER --
PostgreSQL offline scripts run inside the BEGIN/COMMIT wrapper Alembic
emits, so the failure aborts the whole script and converts nothing.
Clean such rows first (or run the migration online, which does it for
you).

``jsonb`` does not preserve key order or duplicate keys. Consumers of
these columns deserialize into Python dicts (checkpoint restore, the
monitoring endpoints), where key order already carries no meaning, and
duplicate keys cannot be produced by ``json.dumps`` from a dict in the
first place -- the write path only ever serializes dicts.

``jsonb`` does, however, re-render numbers: a float stored as ``1e+16``
reads back as the int ``10000000000000000``. The checkpoint blob path
re-hashes payloads it reads and compares them against the write-time
hash, so a *pre-existing* row carrying such a float becomes undecodable
after this migration. New writes are covered -- the write-side sanitizer
normalizes those floats before the hash is taken -- and rows this
migration rewrites are normalized here too. Rows it does not touch are
deliberately left alone: catching them would mean rewriting every row in
the three tables to defend against a payload shape that requires a float
above 2**53 in trace output, and the existing failure mode for an
unreadable checkpoint row is already a logged skip that falls back to an
older checkpoint (``web/api/trace_handlers.py``), not a lost task.

The detection patterns assume a UTF8 server encoding, which is what the
shipped compose file runs. On a server in another encoding the cast also
rejects any escape naming a character outside that charset -- including a
valid surrogate pair, which the pair-stripping step deliberately removes
before matching -- so the cleanup would miss such a row and the ALTER
would fail on it. That is a safe failure (the transaction rolls back and
nothing is converted), but it needs the row cleaned by hand before the
upgrade can proceed. The read guard in ``web/api/monitor.py`` documents
the same limitation for the same patterns.

Operationally this is the expensive kind of migration, and on a large
deployment it should be scheduled rather than slipped into a routine
restart. ``ALTER ... TYPE`` takes an ACCESS EXCLUSIVE lock and rewrites
every row of the table -- there is no in-place path from ``json`` to
``jsonb``, because the on-disk representation genuinely differs -- and the
cleanup scan that precedes it is an unindexed full-table regex pass over
the same rows. Both run inside one transaction per table, so the lock is
held for the whole of it, and ``trace_events`` is the table the monitoring
dashboard already scans. Pruning checkpoint history first shortens both
steps.

Revision ID: 20260813_trace_json_columns_to_jsonb
Revises: 20260809_add_task_interaction_requests
Create Date: 2026-08-13

"""

import json
import re
from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260813_trace_json_columns_to_jsonb"
down_revision: Union[str, None] = "20260809_add_task_interaction_requests"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# (table, column) pairs that hold trace payloads. All three receive slices
# of the same event data, so a payload rejected by one would be rejected by
# any -- they must convert together or the write path splits behaviour.
TRACE_JSON_COLUMNS: tuple[tuple[str, str], ...] = (
    ("trace_events", "data"),
    ("trace_message_blobs", "message_data"),
    ("trace_checkpoint_blobs", "blob_data"),
)

# Detection patterns, copied verbatim from web/api/monitor.py
# (PG_ESCAPED_BACKSLASH / PG_SURROGATE_PAIR_PATTERN /
# PG_UNSAFE_ESCAPE_PATTERN) -- that file is the source of truth and
# documents why each normalization step is load-bearing. Bound as
# parameters, never interpolated, so SQL literal escaping cannot distort
# them. The migration inlines the values instead of importing them: a
# migration must stay runnable after the application module moves or
# changes.
ESCAPED_BACKSLASH = "\\\\"
ESCAPED_BACKSLASH_STANDIN = "__"
SURROGATE_PAIR_PATTERN = (
    "\\\\u[dD][89abAB][0-9a-fA-F]{2}\\\\u[dD][c-fC-F][0-9a-fA-F]{2}"
)
UNSAFE_ESCAPE_PATTERN = (
    "\\\\u0000|\\\\u[dD][89abAB][0-9a-fA-F]{2}|\\\\u[dD][c-fC-F][0-9a-fA-F]{2}"
)

# The Python-side rewrite: NUL and every surrogate code point become
# U+FFFD, and floats jsonb would hand back as ints are converted up front.
# Mirrors web/utils/json_payload_sanitizer.py, which documents both rules;
# duplicated here because a migration must not import application code that
# can drift.
_UNSTORABLE_CODE_POINTS = re.compile("[\x00\ud800-\udfff]")
_REPLACEMENT_CHARACTER = "�"
_EXPONENT_NOTATION_THRESHOLD = 1e16

# Rows per cleanup batch. Small enough that one batch of checkpoint blobs
# stays comfortably in memory, large enough that the scan is not dominated
# by round trips on the usual case of nothing to fix.
REWRITE_BATCH_SIZE = 100


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return _UNSTORABLE_CODE_POINTS.sub(_REPLACEMENT_CHARACTER, value)
    # bool is an int subclass, and True/False are not numbers to normalize.
    if isinstance(value, float) and not isinstance(value, bool):
        if abs(value) >= _EXPONENT_NOTATION_THRESHOLD and value.is_integer():
            return int(value)
        return value
    if isinstance(value, dict):
        return {_sanitize(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def _table_exists(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def _column_is_jsonb(table: str, column: str) -> bool:
    for col in sa.inspect(op.get_bind()).get_columns(table):
        if col["name"] == column:
            return isinstance(col["type"], postgresql.JSONB)
    return False


def _rewrite_unconvertible_rows(table: str, column: str) -> None:
    """Replace unstorable escapes in rows the jsonb cast would reject.

    Rows carrying such payloads exist only where something slipped past the
    (younger) write-side sanitizer, so the scan usually matches nothing;
    the full-table regex pass runs once, here, not on any query path.

    Matching rows are walked in id order, one bounded batch at a time,
    rather than collected up front. These payloads are whole trace events
    and checkpoint blobs -- the blob tables exist precisely because they
    get large -- so a deployment that emitted bad output for a while could
    otherwise materialize every poisoned payload in the migration process
    at once. Keyset pagination, not OFFSET: the loop rewrites the rows it
    just read, and an OFFSET walk would skip past rows as the result set
    shifts under it.
    """
    bind = op.get_bind()
    # noqa: S608 on both statements -- table/column come from
    # TRACE_JSON_COLUMNS above, never from user input, and every payload
    # pattern is bound as a parameter rather than interpolated.
    select_batch = sa.text(
        f"SELECT id, CAST({column} AS text) AS payload FROM {table} "  # noqa: S608
        f"WHERE id > :after "
        f"AND regexp_replace("
        f"replace(CAST({column} AS text), :bs, :standin), "
        f":pair, '', 'g') ~ :unsafe "
        f"ORDER BY id LIMIT :limit"
    )
    # The column is still json at this point -- the ALTER runs after this
    # returns -- so the rewritten payload is cast back to json, not jsonb.
    update = sa.text(
        f"UPDATE {table} SET {column} = CAST(:payload AS json) "  # noqa: S608
        f"WHERE id = :id"
    )

    after = 0
    while True:
        rows = bind.execute(
            select_batch,
            {
                "after": after,
                "bs": ESCAPED_BACKSLASH,
                "standin": ESCAPED_BACKSLASH_STANDIN,
                "pair": SURROGATE_PAIR_PATTERN,
                "unsafe": UNSAFE_ESCAPE_PATTERN,
                "limit": REWRITE_BATCH_SIZE,
            },
        ).fetchall()
        if not rows:
            return
        for row_id, payload_text in rows:
            cleaned = _sanitize(json.loads(payload_text))
            bind.execute(update, {"payload": json.dumps(cleaned), "id": row_id})
        # A rewritten row no longer matches the predicate, so the next
        # batch would find these again only by id -- advance past them.
        after = rows[-1][0]


def upgrade() -> None:
    context = op.get_context()
    if context.dialect.name != "postgresql":
        return

    # Offline (--sql) generation runs against a MockConnection: reflection
    # and the row-cleanup SELECT are both impossible, so emit the plain
    # ALTERs a migration-built database needs (see the module docstring for
    # what happens if unconvertible rows are still present).
    if context.as_sql:
        for table, column in TRACE_JSON_COLUMNS:
            op.alter_column(
                table,
                column,
                type_=postgresql.JSONB(),
                existing_nullable=False,
                postgresql_using=f"{column}::jsonb",
            )
        return

    for table, column in TRACE_JSON_COLUMNS:
        # trace_events is created by Base.metadata.create_all(), not by a
        # migration; on a bare database none of the three tables exist yet.
        if not _table_exists(table):
            continue
        # A create_all-first startup already built the column as jsonb from
        # the model; converting again is a rerun no-op.
        if _column_is_jsonb(table, column):
            continue
        _rewrite_unconvertible_rows(table, column)
        op.alter_column(
            table,
            column,
            type_=postgresql.JSONB(),
            existing_nullable=False,
            postgresql_using=f"{column}::jsonb",
        )


def downgrade() -> None:
    context = op.get_context()
    if context.dialect.name != "postgresql":
        return

    if context.as_sql:
        for table, column in TRACE_JSON_COLUMNS:
            op.alter_column(
                table,
                column,
                type_=sa.JSON(),
                existing_nullable=False,
                postgresql_using=f"{column}::json",
            )
        return

    # jsonb -> json always casts cleanly; the U+FFFD rewrites are not (and
    # cannot be) undone.
    for table, column in TRACE_JSON_COLUMNS:
        if not _table_exists(table):
            continue
        if not _column_is_jsonb(table, column):
            continue
        op.alter_column(
            table,
            column,
            type_=sa.JSON(),
            existing_nullable=False,
            postgresql_using=f"{column}::json",
        )
