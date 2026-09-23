"""Admission schema can be added to an existing task database without rewriting it."""

from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from tests.shared.postgres_disposable import load_migration_module
from tests.web.services.task_database_shared import engine as engine_fixture
from tests.web.services.task_database_shared import task_id as task_id_fixture
from xagent.web.models.task_admission import TaskAdmissionBucket, TaskAdmissionTicket

engine = engine_fixture
task_id = task_id_fixture


def test_upgrade_preserves_tasks_and_matches_metadata_then_downgrades(engine, task_id):
    migration = load_migration_module(
        Path(__file__).parents[2]
        / "src/xagent/migrations/versions/20260923_task_admission.py",
        "task_admission_migration",
    )
    from xagent.web.models.task_admission_pacing import TaskAdmissionPacing

    TaskAdmissionPacing.__table__.drop(engine)
    TaskAdmissionTicket.__table__.drop(engine)
    TaskAdmissionBucket.__table__.drop(engine)
    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
            migration.upgrade()
        for model in (TaskAdmissionBucket, TaskAdmissionTicket):
            actual = sa.inspect(connection).get_columns(model.__tablename__)
            expected = model.__table__.columns
            assert {c["name"]: c["nullable"] for c in actual} == {
                c.name: c.nullable for c in expected
            }
        assert (
            connection.scalar(
                sa.text("SELECT id FROM tasks WHERE id = :id"), {"id": task_id}
            )
            == task_id
        )
        with Operations.context(MigrationContext.configure(connection)):
            migration.downgrade()
        assert "task_admission_tickets" not in sa.inspect(connection).get_table_names()
        assert (
            connection.scalar(
                sa.text("SELECT id FROM tasks WHERE id = :id"), {"id": task_id}
            )
            == task_id
        )
