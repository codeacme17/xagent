"""Merge the Atlassian/Miro seed branch with the SharePoint merge head.

Revision ID: 20260920_merge_miro_sharepoint
Revises: 20260920_seed_miro_mcp_app, 91899e1d97d3
Create Date: 2026-09-20 00:00:00.000000

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "20260920_merge_miro_sharepoint"
down_revision: Union[str, None] = (
    "20260920_seed_miro_mcp_app",
    "91899e1d97d3",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
