# SQLite task and workspace identity

[Issue #2793](https://github.com/xorbitsai/xagent/issues/2793) prevents a newly
created task from receiving a deleted task's ID and `web_task_<id>` directory.
Fresh SQLite databases use `INTEGER PRIMARY KEY AUTOINCREMENT` for `tasks.id`.
Migration `20260930_task_identity` establishes the same rule for existing databases.
PostgreSQL keeps its existing task sequence and schema.

## Upgrade

1. Stop task/file writers and cleanup workers. Back up the database and uploads.
2. Use the deployment's normal `XAGENT_UPLOADS_DIR` and
   `XAGENT_EXTERNAL_UPLOAD_DIRS` configuration. Mount all roots containing
   historical workspaces; include an old root as an external upload directory
   if it remains in use after relocation. The database migration must see the
   same filesystem as the application.
3. Run the normal application database upgrade (`try_upgrade_db`). Its SQLite
   migration connection disables FK actions during table replacement, rejects
   newly introduced FK violations, and restores the original FK setting.
   This revision requires an online connection. A direct invocation with FKs
   enabled refuses to rebuild the parent table.
4. Restart writers only after the migration commits. Keep the retired-ID
   sequence when copying or restoring the database.

The initial sequence floor is the maximum of:

- Current task IDs and the existing `tasks` entry in `sqlite_sequence`.
- Task IDs in cleanup obligations and expired-task tombstones.
- `web_task_<id>` and legacy `task_<id>` directory components in uploaded-file
  `storage_path`, `storage_key`, and `storage_uri`, including detached and
  unmarked historical rows. No file row is reclassified or removed.
- Directory names under the configured upload roots, including owner/scope
  subdirectories and symlinked directories. Physical directories are visited
  once to break symlink cycles. File contents are never read or modified.

The scan is conservative: a directory component that looks like a task identity
reserves that number globally, regardless of owner or scope. Filename-only
matches do not count. Filesystem permission/I/O failures abort the upgrade;
uncreated roots are allowed. An unavailable old mount with no surviving metadata
cannot be inventoried: mount it before upgrading. History with no remaining
database record or directory cannot be reconstructed by this migration.

SQLite locks writers for the inventory and rebuild. Plan the maintenance window
for the number of upload directories, uploaded-file rows, and task rows. SQLite
scans the uploaded-file table, but only rows containing `task_` in a supported
path column are decoded in Python. The revision rebuilds the tasks table while
preserving task IDs, rows, constraints, explicit indexes (including
expression/partial indexes), task-table triggers, and inbound relationships. A
real write transaction starts before DDL, so interrupted rebuilds roll back; a
retry retains any already consumed sequence value. A historical ID at SQLite's
signed 64-bit limit fails before changing the schema rather than exhausting the
allocator silently.

## Contract and rollback

The invariant covers automatically allocated IDs of committed tasks. As with
SQLite's native contract, explicit ID insertion, sequence resets, and rolled-back
allocations are not protected. Production task creation uses automatic IDs.
Future migrations that rebuild `tasks` must preserve both `AUTOINCREMENT` and
the sequence floor, including when no task rows remain.

Application rollback can retain this schema. This revision's downgrade keeps
`AUTOINCREMENT` and its sequence deliberately: removing them would discard
retired identities after their retained files have been collected. Downgrading
further through older table-rebuild revisions requires separate review.

Explicit file-ID reattachment and the retained-path registration guard remain
required. The migration prevents future identity collisions; it cannot restore
bytes overwritten before upgrade. Attachment cleanup and ambiguous historical
orphan disposition remain tracked in [#1086](https://github.com/xorbitsai/xagent/issues/1086).
