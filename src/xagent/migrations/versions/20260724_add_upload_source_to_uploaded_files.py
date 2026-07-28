"""add nullable upload_source marker to uploaded_files

Public share-channel hardening (#973): task-less public-share uploads are
created before any task/owner binding, so they can never be reaped by a
plain ``task_id IS NULL`` sweep without also catching logged-in users'
un-sent draft attachments. This adds a provenance marker so orphan GC can
scope its predicate to exactly those task-less public uploads. NULL for all
existing rows and for every other upload path.

Also adds a composite index serving the hourly GC predicate
(``upload_source + task_id + created_at``): marked rows keep their marker
after binding (provenance), so without an index the sweep's scan grows with
total upload history. On PostgreSQL the index is built concurrently so the
hot uploaded_files table is not write-locked.

Revision ID: 20260724_add_upload_source_to_uploaded_files
Revises: 20260725_add_uploaded_file_recovery_index
Create Date: 2026-07-24

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

revision: str = "20260724_add_upload_source_to_uploaded_files"
down_revision: Union[str, None] = "20260725_add_uploaded_file_recovery_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "uploaded_files"
COLUMN = "upload_source"
INDEX = "ix_uploaded_files_orphan_gc"
INDEX_COLUMNS = (COLUMN, "task_id", "created_at")

# uploaded_files is migration-created (20260225) with FKs to users/tasks,
# which are create_all()-owned and may not exist in a migrations-only
# database (SQLite tolerates dangling FK targets). SQLite batch recreate
# reflects the table; without this, resolving those FKs raises
# NoSuchTableError. The FK DDL itself is still carried over by name.
BATCH_REFLECT_KWARGS = {"resolve_fks": False}


def _existing_columns(inspector: Inspector, table: str) -> list[str]:
    return [col["name"] for col in inspector.get_columns(table)]


def _existing_indexes(inspector: Inspector, table: str) -> set[str]:
    return {ix["name"] for ix in inspector.get_indexes(table) if ix.get("name")}


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    if TABLE not in inspector.get_table_names():
        return
    # Guarded so the migration is re-runnable on a partially-applied DB.
    if COLUMN not in _existing_columns(inspector, TABLE):
        with op.batch_alter_table(
            TABLE, reflect_kwargs=BATCH_REFLECT_KWARGS
        ) as batch_op:
            batch_op.add_column(sa.Column(COLUMN, sa.String(length=64), nullable=True))
    if INDEX not in _existing_indexes(Inspector.from_engine(bind), TABLE):
        if op.get_context().dialect.name == "postgresql":
            # Build online: uploaded_files takes constant writes and a plain
            # CREATE INDEX would hold a write lock for the whole build.
            with op.get_context().autocommit_block():
                op.create_index(
                    INDEX,
                    TABLE,
                    list(INDEX_COLUMNS),
                    if_not_exists=True,
                    postgresql_concurrently=True,
                )
        else:
            op.create_index(INDEX, TABLE, list(INDEX_COLUMNS))


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    if TABLE not in inspector.get_table_names():
        return
    if INDEX in _existing_indexes(inspector, TABLE):
        op.drop_index(INDEX, table_name=TABLE)
    if COLUMN in _existing_columns(inspector, TABLE):
        with op.batch_alter_table(
            TABLE, reflect_kwargs=BATCH_REFLECT_KWARGS
        ) as batch_op:
            batch_op.drop_column(COLUMN)
