"""The cleanup-obligation table is created without any task table present."""

import importlib

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.web.services.task_database_shared import engine as engine_fixture
from xagent.web.models.task_cleanup_obligation import TaskCleanupObligation

engine = engine_fixture

MIGRATION = "xagent.migrations.versions.20260925_task_cleanup_obligations"


def test_migration_matches_the_model_and_is_idempotent(engine):
    migration = importlib.import_module(MIGRATION)
    with (
        engine.begin() as connection,
        Operations.context(MigrationContext.configure(connection)),
    ):
        # No ``tasks`` table: the obligations reference their task by value
        # only, so an Alembic-only database gets the table regardless.
        migration.upgrade()
        migration.upgrade()
        inspector = sa.inspect(connection)
        reflected = {
            column["name"] for column in inspector.get_columns(migration.TABLE)
        }
        assert reflected == set(TaskCleanupObligation.__table__.columns.keys())
        assert inspector.get_foreign_keys(migration.TABLE) == []
        # No uniqueness: a reused task id must be able to owe twice.
        assert inspector.get_unique_constraints(migration.TABLE) == []
        indexed = {
            tuple(index["column_names"])
            for index in inspector.get_indexes(migration.TABLE)
        }
        assert {("task_id",), ("status", "next_attempt_at")} <= indexed

        migration.downgrade()
        migration.downgrade()
        assert not sa.inspect(connection).has_table(migration.TABLE)
