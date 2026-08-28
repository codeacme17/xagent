from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text

from tests.shared.postgres_disposable import (
    disposable_database_factory,
    load_migration_module,
)
from xagent.db.config import create_alembic_config

REVISION = "20260828_terminal_cmd_events"
DOWN_REVISION = "20260821_actor_oauth_flow_states"
MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20260828_add_task_command_terminal_events.py"
)


def _offline_sql(migration, dialect_name: str, operation: str) -> str:
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect_name,
        opts={"as_sql": True, "output_buffer": output},
    )
    with Operations.context(context):
        getattr(migration, operation)()
    return output.getvalue()


@pytest.fixture
def postgresql_engine_factory():
    with disposable_database_factory("xagent_terminal_events") as make:
        yield make


def test_upgrade_adds_terminal_task_command_event_log() -> None:
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)

    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
        )
        connection.execute(
            text("INSERT INTO alembic_version (version_num) VALUES (:revision)"),
            {"revision": DOWN_REVISION},
        )
        connection.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY)"))
        connection.execute(
            text(
                "CREATE TABLE tasks ("
                "id INTEGER PRIMARY KEY, state_version INTEGER NOT NULL DEFAULT 0)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE task_execution_commands ("
                "id INTEGER PRIMARY KEY, task_id INTEGER NOT NULL, "
                "target_run_id VARCHAR(64))"
            )
        )
        connection.execute(
            text(
                "INSERT INTO task_execution_commands "
                "(id, task_id, target_run_id) VALUES (1, 1, 'legacy-run')"
            )
        )
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)

        columns = {
            column["name"]
            for column in inspect(connection).get_columns(
                "task_command_terminal_events"
            )
        }
        indexes = {
            index["name"]
            for index in inspect(connection).get_indexes("task_command_terminal_events")
        }
        assert {
            "event_id",
            "task_command_id",
            "task_id",
            "task_run_id",
            "task_state_version",
            "command_id",
            "command_kind",
            "actor_user_id",
            "task_owner_user_id",
            "outcome_version",
            "outcome",
            "message_code",
            "resend_safe",
            "include_command_identity",
            "created_at",
        } <= columns
        assert "ix_task_command_terminal_events_task_cursor" in indexes
        command_columns = {
            column["name"]
            for column in inspect(connection).get_columns("task_execution_commands")
        }
        assert "target_state_version" in command_columns
        legacy_version = connection.execute(
            text(
                "SELECT target_state_version FROM task_execution_commands WHERE id = 1"
            )
        ).scalar_one()
        assert legacy_version is None

        command.downgrade(config, DOWN_REVISION)
        assert (
            "task_command_terminal_events" not in inspect(connection).get_table_names()
        )
        command_columns = {
            column["name"]
            for column in inspect(connection).get_columns("task_execution_commands")
        }
        assert "target_state_version" not in command_columns


def test_upgrade_skips_without_command_table() -> None:
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)

    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
        )
        connection.execute(
            text("INSERT INTO alembic_version (version_num) VALUES (:revision)"),
            {"revision": DOWN_REVISION},
        )
        connection.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY)"))
        connection.execute(text("CREATE TABLE tasks (id INTEGER PRIMARY KEY)"))
        config.attributes["connection"] = connection

        command.upgrade(config, REVISION)

        assert (
            "task_command_terminal_events" not in inspect(connection).get_table_names()
        )


@pytest.mark.parametrize("dialect_name", ["sqlite", "postgresql"])
def test_offline_upgrade_emits_terminal_event_schema(dialect_name: str) -> None:
    migration = load_migration_module(
        MIGRATION_PATH, f"terminal_command_events_offline_upgrade_{dialect_name}"
    )

    with patch.object(
        migration.sa,
        "inspect",
        side_effect=AssertionError("offline branch must not reflect"),
    ):
        sql = _offline_sql(migration, dialect_name, "upgrade")

    assert "ALTER TABLE task_execution_commands ADD COLUMN target_state_version" in sql
    assert "CREATE TABLE task_command_terminal_events" in sql
    assert "CREATE INDEX ix_task_command_terminal_events_task_cursor" in sql
    assert "%(" not in sql


@pytest.mark.parametrize("dialect_name", ["sqlite", "postgresql"])
def test_offline_downgrade_emits_terminal_event_cleanup(dialect_name: str) -> None:
    migration = load_migration_module(
        MIGRATION_PATH, f"terminal_command_events_offline_downgrade_{dialect_name}"
    )

    with patch.object(
        migration.sa,
        "inspect",
        side_effect=AssertionError("offline branch must not reflect"),
    ):
        sql = _offline_sql(migration, dialect_name, "downgrade")

    assert "DROP TABLE task_command_terminal_events" in sql
    assert "ALTER TABLE task_execution_commands DROP COLUMN target_state_version" in sql
    assert "%(" not in sql


@pytest.mark.postgresql
def test_postgresql_upgrade_and_downgrade_preserve_unknown_legacy_version(
    postgresql_engine_factory,
) -> None:
    migration = load_migration_module(
        MIGRATION_PATH, "terminal_command_events_migration"
    )
    engine = postgresql_engine_factory("upgrade")

    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY)"))
        connection.execute(
            text(
                "CREATE TABLE tasks ("
                "id INTEGER PRIMARY KEY, state_version INTEGER NOT NULL DEFAULT 0)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE task_execution_commands ("
                "id INTEGER PRIMARY KEY, task_id INTEGER NOT NULL, "
                "target_run_id VARCHAR(64))"
            )
        )
        connection.execute(
            text(
                "INSERT INTO task_execution_commands "
                "(id, task_id, target_run_id) VALUES (1, 1, 'legacy-run')"
            )
        )
        context = MigrationContext.configure(connection)
        with Operations.context(context):
            migration.upgrade()

        inspector = sa.inspect(connection)
        assert "task_command_terminal_events" in inspector.get_table_names()
        legacy_version = connection.execute(
            text(
                "SELECT target_state_version FROM task_execution_commands WHERE id = 1"
            )
        ).scalar_one()
        assert legacy_version is None

        with Operations.context(context):
            migration.downgrade()
        assert (
            "task_command_terminal_events"
            not in sa.inspect(connection).get_table_names()
        )
