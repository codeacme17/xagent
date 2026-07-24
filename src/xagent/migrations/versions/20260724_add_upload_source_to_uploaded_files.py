"""add nullable upload_source marker to uploaded_files

Public share-channel hardening (#973): task-less public-share uploads are
created before any task/owner binding, so they can never be reaped by a
plain ``task_id IS NULL`` sweep without also catching logged-in users'
un-sent draft attachments. This adds a provenance marker so orphan GC can
scope its predicate to exactly those task-less public uploads. NULL for all
existing rows and for every other upload path.

Revision ID: 20260724_add_upload_source_to_uploaded_files
Revises: 20260722_add_workforce_id_to_agent_api_keys
Create Date: 2026-07-24

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

revision: str = "20260724_add_upload_source_to_uploaded_files"
down_revision: Union[str, None] = "20260722_add_workforce_id_to_agent_api_keys"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "uploaded_files"
COLUMN = "upload_source"


def _existing_columns(inspector: Inspector, table: str) -> list[str]:
    return [col["name"] for col in inspector.get_columns(table)]


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    if TABLE not in inspector.get_table_names():
        return
    # Guarded so the migration is re-runnable on a partially-applied DB.
    if COLUMN not in _existing_columns(inspector, TABLE):
        with op.batch_alter_table(TABLE) as batch_op:
            batch_op.add_column(sa.Column(COLUMN, sa.String(length=64), nullable=True))


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    if TABLE not in inspector.get_table_names():
        return
    if COLUMN in _existing_columns(inspector, TABLE):
        with op.batch_alter_table(TABLE) as batch_op:
            batch_op.drop_column(COLUMN)
