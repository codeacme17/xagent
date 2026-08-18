"""backfill auth.app_id onto legacy builtin-OAuth server rows

Rows provisioned before ``_ensure_user_mcp_server`` wrote OAuth metadata
carry no ``auth`` payload at all (and a small malformed population can
carry a blank or provider-only ``auth``). Every read path that resolves a
server row back to its catalog app prefers the stable ``auth.app_id`` and
falls back to the row's *exact current display name*
(``get_app_for_mcp_server``) -- so the moment an admin renames the app,
an unstamped row loses its identity: ``/api/mcp/servers`` enrichment
reports no ``app_id``, the connector picker persists the app's new
display name, and the runtime -- which knows the row only under its old
name -- resolves the selection to zero tools, silently (#1429, surfaced
reviewing #1403).

This migration stamps the identity while the name still matches, which
is the only moment it can be derived safely:

- Candidates are ``mcp_servers`` rows with ``transport = 'oauth'`` whose
  ``auth`` is missing, not a dict, or lacks a nonblank *string* ``app_id``
  — a non-string value (e.g. an integer) is malformed metadata the read
  path rejects outright, so such a row stays a candidate and a successful
  match overwrites the malformed value.
- Identities are opaque and matched/stored as **raw strings**, never
  trimmed or coerced: every read path compares them exactly, and stamping
  a trimmed variant of a padded catalog id would write a value the exact
  lookup can never resolve. Whitespace-variant names therefore do not
  match — under-matching leaves the name-fallback shim in charge, which
  loses nothing.
- Each is matched against ``public_mcp_apps`` (builtin apps are seeded
  into that table too) by exact display name, the same value
  ``_ensure_user_mcp_server`` named the row after. If the name resolves
  nothing and the row's ``auth.provider`` names exactly one OAuth app,
  that app is used instead -- providers are non-unique across apps (the
  meta/Instagram siblings), so an ambiguous provider is never resolved.
- Only apps whose own ``transport`` is ``oauth`` are eligible: a
  same-named non-OAuth app is a different connector shape, and stamping
  its id onto an oauth-transport row would manufacture exactly the
  cross-shape identity confusion this migration exists to remove.
- Rows that resolve to nothing are left untouched. The exact-name
  fallback in ``get_app_for_mcp_server`` remains as the compatibility
  shim for them, unchanged in behavior: stamping nothing loses nothing.
- A row whose own ``auth.provider`` contradicts the name-resolved app's
  provider is refused — the same conflict the provisioning writer
  (``_ensure_server_matches_oauth_app``) refuses with a ValueError.
  Stamping the name's id would produce a row *no* app claims.
- ``app_id`` is written into a *copy* of the existing auth dict;
  ``provider`` is added only when absent. Nothing else in ``auth`` is
  touched — except a non-dict ``auth`` payload (garbage on an
  oauth-transport row), which is replaced wholesale, matching the
  provisioning writer, which discards non-dict auth the same way. Rows
  already carrying a nonblank ``app_id`` are not candidates at all, so
  re-running the migration is a no-op (idempotent).

Offline (``--sql``) mode **raises** instead of no-opping: Alembic emits
the ``alembic_version`` bookkeeping even for an empty migration body, so
a silently-skipped offline run would advance the version while touching
no row — and a later online upgrade would then skip this revision
forever. Failing loudly forces the one correct path: run this revision
online. The schema is untouched either way.

The downgrade is a deliberate no-op. Removing ``auth.app_id`` would need
to distinguish stamped rows from rows the post-metadata writer created,
which the data cannot express -- and an extra stable identity is harmless
to every pre-migration reader (they all prefer ``app_id`` and only fall
back to names when it is absent).

Revision ID: 20260818_backfill_oauth_server_app_identity
Revises: 20260813_trace_json_columns_to_jsonb
Create Date: 2026-08-18
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260818_backfill_oauth_server_app_identity"
down_revision: str | None = "20260813_trace_json_columns_to_jsonb"
branch_labels = None
depends_on = None


def _nonblank_str(value: object) -> str | None:
    """The value itself when it is a str with non-whitespace content, else None.

    Raw, never trimmed or coerced: identities are opaque and every downstream
    comparison is exact (``get_app_by_id`` compares ``PublicMCPApp.app_id``
    directly; ``get_app_for_mcp_server`` rejects a non-string ``auth.app_id``).
    Stamping a trimmed variant of a padded catalog id would write a value the
    exact lookup can never resolve — strictly worse than not stamping. This
    helper decides only *whether* a usable string exists; the string used for
    matching and for storage is always the raw one.
    """
    if isinstance(value, str) and value.strip():
        return value
    return None


def upgrade() -> None:
    if op.get_context().as_sql:
        # Fail loudly rather than no-op: returning here would still let
        # Alembic emit the alembic_version bookkeeping, so an operator who
        # applies the generated --sql script advances the version while no
        # row was read or updated — and a later online upgrade then skips
        # this revision forever, leaving the legacy rows unstamped.
        raise RuntimeError(
            "20260818_backfill_oauth_server_app_identity is a data migration "
            "and cannot run in offline (--sql) mode: applying generated SQL "
            "would advance alembic_version without performing the backfill. "
            "Run `alembic upgrade` online for this revision."
        )

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    # A bare database has neither table until create_all runs; there is
    # nothing to backfill there.
    if "mcp_servers" not in tables or "public_mcp_apps" not in tables:
        return

    mcp_servers = sa.table(
        "mcp_servers",
        sa.column("id", sa.Integer),
        sa.column("name", sa.String),
        sa.column("transport", sa.String),
        sa.column("auth", sa.JSON),
    )
    public_mcp_apps = sa.table(
        "public_mcp_apps",
        sa.column("app_id", sa.String),
        sa.column("name", sa.String),
        sa.column("transport", sa.String),
        sa.column("provider_name", sa.String),
    )

    # Exact display name -> app, and provider -> apps, for OAuth apps only.
    # Both maps refuse ambiguity rather than guessing: two same-named OAuth
    # apps drop the name key entirely, and a provider shared by more than one
    # app resolves nothing (the meta/Instagram case). Keys and stored values
    # are the raw strings — identities are opaque, and every read path
    # compares them exactly. Only ``transport`` is folded, because it is a
    # shape enum, not an identity (mirroring classify_app_auth).
    apps_by_name: dict[str, tuple[str, str | None] | None] = {}
    apps_by_provider: dict[str, tuple[str, str | None] | None] = {}
    for app_row in bind.execute(sa.select(public_mcp_apps)).mappings():
        if str(app_row["transport"] or "").strip().lower() != "oauth":
            continue
        app_id = _nonblank_str(app_row["app_id"])
        if not app_id:
            continue
        provider = _nonblank_str(app_row["provider_name"])
        entry = (app_id, provider)
        name = _nonblank_str(app_row["name"])
        if name:
            apps_by_name[name] = None if name in apps_by_name else entry
        if provider:
            apps_by_provider[provider] = None if provider in apps_by_provider else entry

    # Materialized before the loop: the loop UPDATEs the same table, and
    # stepping a live pysqlite cursor across a mutating table has formally
    # unspecified row visibility. The table is small (one row per connector).
    candidates = (
        bind.execute(sa.select(mcp_servers).where(mcp_servers.c.transport == "oauth"))
        .mappings()
        .all()
    )

    for row in candidates:
        auth = row["auth"] if isinstance(row["auth"], dict) else None
        # Already-stamped means a nonblank *string* app_id — the only shape
        # the read path accepts (get_app_for_mcp_server rejects a non-string
        # app_id outright). A non-string or blank value is malformed metadata
        # that leaves the row permanently unresolvable, so such a row stays a
        # candidate and a successful match overwrites the malformed value.
        if auth is not None and _nonblank_str(auth.get("app_id")):
            continue
        row_provider = _nonblank_str(auth.get("provider")) if auth is not None else None

        resolved = apps_by_name.get(str(row["name"]) if row["name"] else "")
        if (
            resolved is not None
            and row_provider
            and resolved[1]
            and row_provider != resolved[1]
        ):
            # The row's own provider contradicts the name-resolved app — the
            # same conflict _ensure_server_matches_oauth_app refuses with a
            # ValueError. Stamping the name's app_id would create a row *no*
            # app claims (it fails this app's provider gate and the provider's
            # app_id gate), so refuse and leave the read-time provider
            # fallback exactly as it was.
            continue
        if resolved is None and row_provider:
            resolved = apps_by_provider.get(row_provider)
        if resolved is None:
            continue

        app_id, provider = resolved
        new_auth = dict(auth or {})
        new_auth["app_id"] = app_id
        if provider and not _nonblank_str(new_auth.get("provider")):
            new_auth["provider"] = provider
        bind.execute(
            sa.update(mcp_servers)
            .where(mcp_servers.c.id == row["id"])
            .values(auth=new_auth)
        )


def downgrade() -> None:
    # Deliberate no-op: stamped rows are indistinguishable from rows the
    # post-metadata writer created, and the extra stable identity is harmless
    # to every pre-migration reader (see module docstring).
    pass
