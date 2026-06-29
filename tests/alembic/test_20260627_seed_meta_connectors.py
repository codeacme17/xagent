"""Tests for the Meta connector registry seed migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260627_seed_meta_connectors.py"
    )
    spec = importlib.util.spec_from_file_location(
        "seed_meta_connectors_migration", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    context = MigrationContext.configure(connection)
    return Operations(context)


def test_upgrade_inserts_meta_provider_and_public_apps(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE oauth_providers (
                    id INTEGER PRIMARY KEY,
                    provider_name VARCHAR(50) UNIQUE NOT NULL,
                    name VARCHAR(100) NOT NULL,
                    client_id VARCHAR(500) NOT NULL,
                    client_secret VARCHAR(500) NOT NULL,
                    auth_url VARCHAR(1000) NOT NULL,
                    token_url VARCHAR(1000) NOT NULL,
                    redirect_uri VARCHAR(1000),
                    userinfo_url VARCHAR(1000),
                    user_id_path VARCHAR(100),
                    email_path VARCHAR(100),
                    default_scopes JSON
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE public_mcp_apps (
                    id INTEGER PRIMARY KEY,
                    app_id VARCHAR(100) UNIQUE NOT NULL,
                    name VARCHAR(200) NOT NULL,
                    description TEXT,
                    icon VARCHAR(1000),
                    transport VARCHAR(50) NOT NULL,
                    provider_name VARCHAR(50),
                    category VARCHAR(100),
                    oauth_scopes JSON,
                    is_visible_in_connector BOOLEAN NOT NULL DEFAULT 1,
                    launch_config JSON
                )
                """
            )
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()

        providers = connection.execute(
            text("SELECT provider_name, name FROM oauth_providers")
        ).fetchall()
        apps = connection.execute(
            text("SELECT app_id, provider_name, category FROM public_mcp_apps")
        ).fetchall()

    assert providers == [("meta", "Meta")]
    assert {
        (row.app_id, row.provider_name, row.category)
        for row in apps
        if row.app_id in {"facebook", "instagram"}
    } == {
        ("facebook", "meta", "Marketing"),
        ("instagram", "meta", "Marketing"),
    }
