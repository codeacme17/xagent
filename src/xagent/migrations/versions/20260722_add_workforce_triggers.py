"""add workforce trigger support to agent_triggers

Workforce webhook/trigger deployment channel (#950 / #805):
- ``agent_triggers.workforce_id``: nullable FK to ``workforces`` for
  workforce-owned triggers, which fire via ``create_workforce_run`` and are
  never bound to the workforce's generated manager agent.
- ``agent_triggers.agent_id`` becomes nullable; exactly one of ``agent_id`` /
  ``workforce_id`` is set (enforced by the service layer).
- Data cleanup (from the #951 review): disable pre-existing trigger rows
  bound to workforce-generated manager agents. Those rows were invisible to
  every management API yet kept firing through the dispatcher and webhook
  receivers ("orphaned but still live").

Revision ID: 20260722_add_workforce_triggers
Revises: 20260721_add_workforce_deployment_foundation
Create Date: 2026-07-22

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

revision: str = "20260722_add_workforce_triggers"
down_revision: Union[str, None] = "20260721_add_workforce_deployment_foundation"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TRIGGERS_TABLE = "agent_triggers"
WORKFORCES_TABLE = "workforces"
AGENTS_TABLE = "agents"
WORKFORCE_ID_COLUMN = "workforce_id"
WORKFORCE_ID_INDEX = "ix_agent_triggers_workforce_id"
WORKFORCE_ID_FK = "fk_agent_triggers_workforce_id"

WORKFORCE_GENERATED_MANAGER_ORIGIN = "workforce_generated_manager"


def _columns(inspector: Inspector, table: str) -> dict[str, dict]:
    return {col["name"]: col for col in inspector.get_columns(table)}


def _index_names(inspector: Inspector, table: str) -> list[str]:
    return [str(index["name"]) for index in inspector.get_indexes(table)]


def _disable_orphaned_manager_triggers(bind: sa.engine.Connection) -> None:
    """Disable triggers bound to workforce-generated manager agents.

    Disabled (not deleted) so run/audit history stays reachable if a later
    cleanup wants it; the firing path additionally refuses these rows at
    execution time as defense in depth.
    """
    agent_triggers = sa.table(
        TRIGGERS_TABLE,
        sa.column("agent_id", sa.Integer),
        sa.column("enabled", sa.Boolean),
    )
    agents = sa.table(
        AGENTS_TABLE,
        sa.column("id", sa.Integer),
        sa.column("origin", sa.String),
    )
    manager_agent_ids = sa.select(agents.c.id).where(
        agents.c.origin == WORKFORCE_GENERATED_MANAGER_ORIGIN
    )
    bind.execute(
        agent_triggers.update()
        .where(agent_triggers.c.agent_id.in_(manager_agent_ids))
        .values(enabled=False)
    )


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_names = inspector.get_table_names()
    if TRIGGERS_TABLE not in table_names:
        return

    columns = _columns(inspector, TRIGGERS_TABLE)
    needs_workforce_column = WORKFORCE_ID_COLUMN not in columns
    agent_id_not_nullable = "agent_id" in columns and not bool(
        columns["agent_id"].get("nullable", True)
    )

    if needs_workforce_column or agent_id_not_nullable:
        with op.batch_alter_table(TRIGGERS_TABLE) as batch:
            if needs_workforce_column:
                batch.add_column(
                    sa.Column(WORKFORCE_ID_COLUMN, sa.Integer(), nullable=True)
                )
                if WORKFORCES_TABLE in table_names:
                    batch.create_foreign_key(
                        WORKFORCE_ID_FK,
                        WORKFORCES_TABLE,
                        [WORKFORCE_ID_COLUMN],
                        ["id"],
                        ondelete="CASCADE",
                    )
            if agent_id_not_nullable:
                batch.alter_column(
                    "agent_id", existing_type=sa.Integer(), nullable=True
                )

    # Re-reflect: the batch rebuild above may have replaced the table.
    inspector = sa.inspect(bind)
    if WORKFORCE_ID_INDEX not in _index_names(inspector, TRIGGERS_TABLE):
        op.create_index(WORKFORCE_ID_INDEX, TRIGGERS_TABLE, [WORKFORCE_ID_COLUMN])

    if AGENTS_TABLE in table_names:
        _disable_orphaned_manager_triggers(bind)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_names = inspector.get_table_names()
    if TRIGGERS_TABLE not in table_names:
        return

    columns = _columns(inspector, TRIGGERS_TABLE)
    if WORKFORCE_ID_COLUMN in columns:
        # Workforce triggers cannot exist without the column; drop their rows
        # so agent_id can be tightened back to NOT NULL. Rows disabled by the
        # upgrade's data cleanup stay disabled (not restored).
        agent_triggers = sa.table(
            TRIGGERS_TABLE,
            sa.column("agent_id", sa.Integer),
            sa.column(WORKFORCE_ID_COLUMN, sa.Integer),
        )
        bind.execute(agent_triggers.delete().where(agent_triggers.c.agent_id.is_(None)))

        if WORKFORCE_ID_INDEX in _index_names(inspector, TRIGGERS_TABLE):
            op.drop_index(WORKFORCE_ID_INDEX, table_name=TRIGGERS_TABLE)
        with op.batch_alter_table(TRIGGERS_TABLE) as batch:
            batch.drop_column(WORKFORCE_ID_COLUMN)
            batch.alter_column("agent_id", existing_type=sa.Integer(), nullable=False)
