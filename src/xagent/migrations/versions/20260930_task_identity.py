"""Reserve retired SQLite task identities before enabling AUTOINCREMENT.

Revision ID: 20260930_task_identity
Revises: 20260929_task_attachment_detachment

Run online with writers stopped and the deployment's upload roots mounted.
PostgreSQL already allocates task IDs from a sequence and is unchanged.
"""

import os
import re
from pathlib import Path

import sqlalchemy as sa
from alembic import op

from xagent.config import get_external_upload_dirs, get_uploads_dir

revision = "20260930_task_identity"
down_revision = "20260929_task_attachment_detachment"
branch_labels = None
depends_on = None

WORKSPACE_NAME = re.compile(r"(?:web_task_|task_)([0-9]+)\Z")
MAX_ID = (1 << 63) - 1


def _workspace_id(name):
    match = WORKSPACE_NAME.fullmatch(name)
    if match is None:
        return 0
    digits = match[1].lstrip("0") or "0"
    if len(digits) > 19 or int(digits) >= MAX_ID:
        raise RuntimeError("Historical workspace identity exhausts SQLite task IDs")
    return int(digits)


def _path_floor(value):
    # Only directory components count: a file named web_task_123 is not a
    # workspace. Both separators support metadata from relocated databases.
    return max(
        (_workspace_id(part) for part in re.split(r"[/\\]", value or "")[:-1]),
        default=0,
    )


def _directory_floor():
    pending = [get_uploads_dir(), *get_external_upload_dirs()]
    visited = set()
    floor = 0
    while pending:
        directory = pending.pop()
        try:
            stat = directory.stat()
        except FileNotFoundError:
            # An unused root is valid on an installation with no uploads.
            continue
        floor = max(floor, _workspace_id(directory.name))
        identity = (stat.st_dev, stat.st_ino)
        if identity in visited:
            continue
        visited.add(identity)
        # Follow scope-directory symlinks, but visit each physical directory
        # once. Scope names can themselves look like workspace names, so keep
        # walking. Read directory entries only, never file contents. Permission
        # and I/O errors abort rather than hiding identities.
        with os.scandir(directory) as entries:
            pending.extend(Path(entry.path) for entry in entries if entry.is_dir())
    return floor


def _historical_floor(connection, inspector):
    floor = int(connection.exec_driver_sql("SELECT max(id) FROM tasks").scalar() or 0)
    tables = set(inspector.get_table_names())
    if (
        "sqlite_sequence" in tables
        or connection.exec_driver_sql(
            "SELECT 1 FROM sqlite_master WHERE name='sqlite_sequence'"
        ).first()
    ):
        floor = max(
            floor,
            int(
                connection.exec_driver_sql(
                    "SELECT max(seq) FROM sqlite_sequence WHERE name='tasks'"
                ).scalar()
                or 0
            ),
        )
    # These rows intentionally survive deletion, without a task FK.
    for table in ("task_cleanup_obligations", "expired_task_tombstones"):
        if table in tables:
            floor = max(
                floor,
                int(
                    connection.exec_driver_sql(
                        f"SELECT max(task_id) FROM {table}"
                    ).scalar()
                    or 0
                ),
            )
    if "uploaded_files" in tables:
        columns = {column["name"] for column in inspector.get_columns("uploaded_files")}
        paths = sorted(columns & {"storage_path", "storage_key", "storage_uri"})
        if paths:
            candidate = " OR ".join(f"instr({path}, 'task_') > 0" for path in paths)
            rows = connection.exec_driver_sql(
                f"SELECT {', '.join(paths)} FROM uploaded_files WHERE {candidate}"
            )
            for row in rows:
                floor = max(floor, *(_path_floor(value) for value in row))
    floor = max(floor, _directory_floor())
    if floor >= MAX_ID:
        raise RuntimeError("Historical workspace identity exhausts SQLite task IDs")
    return floor


def _task_table(connection):
    table = sa.Table(
        "tasks", sa.MetaData(), autoload_with=connection, resolve_fks=False
    )
    # Reflection promotes inline CHECKs to table constraints. These two must
    # stay on their columns: older event migrations drop the columns directly.
    for column in ("conversation_storage_version", "conversation_event_sequence"):
        for constraint in list(table.constraints):
            if (
                isinstance(constraint, sa.CheckConstraint)
                and constraint.name == f"ck_tasks_{column}"
            ):
                table.constraints.remove(constraint)
                table.c[column].constraints.add(constraint)
    # Recover referential actions and compact-DDL uniqueness directly from
    # SQLite's catalog, which is more complete than SQLAlchemy's DDL parser.
    foreign_keys = {}
    for row in connection.exec_driver_sql("PRAGMA foreign_key_list('tasks')"):
        foreign_keys.setdefault(row[0], []).append(row)
    for rows in foreign_keys.values():
        rows.sort(key=lambda row: row[1])
        columns = tuple(row[3] for row in rows)
        constraint = next(
            fk
            for fk in table.foreign_key_constraints
            if tuple(fk.columns.keys()) == columns
        )
        constraint.onupdate = rows[0][5]
        constraint.ondelete = rows[0][6]
    known = {
        tuple(c.columns.keys())
        for c in table.constraints
        if isinstance(c, sa.UniqueConstraint)
    }
    for row in connection.exec_driver_sql("PRAGMA index_list('tasks')"):
        if row[2] and row[3] == "u":
            columns = tuple(
                item[2]
                for item in connection.exec_driver_sql(
                    "SELECT * FROM pragma_index_info(?)", (row[1],)
                )
            )
            if columns not in known:
                table.append_constraint(sa.UniqueConstraint(*columns))
                known.add(columns)
    # Recreate explicit indexes from their original SQL, including expression
    # and partial indexes that reflection may omit. Triggers are saved too.
    table.indexes.clear()
    return table


def upgrade():
    context = op.get_context()
    if context.dialect.name == "postgresql":
        return
    if context.dialect.name != "sqlite":
        raise RuntimeError(
            "Task identity migration supports SQLite and PostgreSQL only"
        )
    if context.as_sql:
        raise RuntimeError(
            "Task identity migration requires an online SQLite connection"
        )
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    if not inspector.has_table("tasks"):
        return  # Fresh installations stamp head, then use model metadata.
    if connection.exec_driver_sql("PRAGMA foreign_keys").scalar():
        raise RuntimeError("Run task identity migration through _migration_connection")
    # Start a real SQLite write transaction before any DDL, even under the
    # sqlite3 driver's legacy transaction mode. A failure rolls back the entire
    # table replacement and sequence update, and concurrent writers wait.
    connection.exec_driver_sql("UPDATE tasks SET id=id WHERE 0")
    floor = _historical_floor(connection, inspector)
    objects = (
        connection.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE tbl_name='tasks' "
            "AND type IN ('index', 'trigger') AND sql IS NOT NULL ORDER BY type, name"
        )
        .scalars()
        .all()
    )
    with op.batch_alter_table(
        "tasks",
        recreate="always",
        copy_from=_task_table(connection),
        table_kwargs={"sqlite_autoincrement": True},
    ):
        pass
    for sql in objects:
        connection.exec_driver_sql(sql)
    connection.exec_driver_sql("DELETE FROM sqlite_sequence WHERE name='tasks'")
    connection.exec_driver_sql(
        "INSERT INTO sqlite_sequence(name, seq) VALUES ('tasks', ?)", (floor,)
    )


def downgrade():
    # Older application versions work with AUTOINCREMENT. Removing it would
    # discard the only record of retired IDs after retained files are collected.
    pass
