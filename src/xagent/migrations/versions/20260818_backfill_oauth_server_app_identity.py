"""backfill auth.app_id onto unstamped OAuth-transport server rows

Rows provisioned before ``_ensure_user_mcp_server`` wrote OAuth metadata
carry no ``auth`` payload at all (and a small malformed population can
carry a blank or provider-only ``auth``). Every read path that resolves a
server row back to its catalog app prefers the stable ``auth.app_id`` and
falls back to the row's *exact current display name*
(``get_app_for_mcp_server``) -- so an unstamped row's identity lives and
dies by that name: once it stops matching, ``/api/mcp/servers``
enrichment reports no ``app_id``, the connector picker persists the app's
current display name, and the runtime -- which knows the row only under
its stored name -- resolves the selection to zero tools, silently (#1429,
surfaced reviewing #1403).

Which apps can drift that way is narrower than "any OAuth app", and
worth stating precisely, because it is the whole reason this is a
hardening pass rather than a repair of a known-broken population:

- Genuine **builtin** apps cannot be renamed at all: ``name``,
  ``transport`` and ``provider_name`` are in
  ``_BUILTIN_PROTECTED_FIELDS`` (``admin_mcp.py``) and a PATCH touching
  them 409s, while ``_app_to_dict`` serves the registry's name
  regardless of what the row stores.
- **Admin-created** OAuth catalog apps *are* freely renameable, and
  ``classify_app_auth`` labels them ``builtin_oauth`` too (purely on
  ``transport == "oauth"``). But admin app CRUD and the unconditional
  ``app_id`` writer arrived in the same commit, so every row provisioned
  for such an app already carries the stamp.
- What is left unstamped-and-renameable is therefore reachable only
  outside the API surface: direct database edits, or a code-level
  registry rename across versions. This migration exists to make that
  residue resolvable rather than to fix a live API-reachable break.

The stamp is derived while the name still matches, the only moment it
can be derived safely:

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
  nothing -- including when it is *ambiguous*, i.e. shared by two OAuth
  apps -- and the row's ``auth.provider`` names exactly one OAuth app,
  that app is used instead. Providers are non-unique across apps (the
  meta/Instagram siblings), so an ambiguous provider is never resolved.
  Names are resolved for *every* candidate before any provider fallback
  runs, and an app already claimed by a name match is never handed to a
  second row: ``_ensure_user_mcp_server`` creates a *new* row when a
  rename makes the old one unfindable, so orphan pairs for one app are
  this migration's own target population, and stamping both would give
  two rows the same identity -- which ``_lookup_oauth_server_for_app``
  resolves from an unordered query, making configure/disconnect pick
  nondeterministically between them. Name evidence wins; the loser keeps
  the behavior it had before this migration.
- Only apps whose own ``transport`` is ``oauth`` are eligible: a
  same-named non-OAuth app is a different connector shape, and stamping
  its id onto an oauth-transport row would manufacture exactly the
  cross-shape identity confusion this migration exists to remove.
- Rows that resolve to nothing are left untouched, keeping the exact-name
  fallback in ``get_app_for_mcp_server`` as their compatibility shim. That
  is a deliberate trade rather than a free win: ``get_app_for_mcp_server``
  never falls back to the name once ``app_id`` is present, so a stamp that
  later goes dangling (an admin deletes a custom OAuth app and recreates
  it under the same name with a new id) resolves to nothing where an
  unstamped row would still have matched by name. Stamping only
  unambiguous matches is what keeps that trade narrow.
- A drifted builtin row is classified here from its *stored* columns,
  while runtime reads take ``transport``/``provider_name``/``name`` from
  the code registry (``_app_to_dict``) and only warn about drift. The two
  can therefore disagree about a drifted row; the blast radius is one
  unambiguous stamp on a row whose stored shape says ``oauth``.
- Only rows whose stored name still matches can be stamped by name. In a
  deployment where the drift already happened, only the provider fallback
  can rescue the row -- and not when the provider is shared or absent. So
  this hardens the population that is still resolvable today; it cannot
  retroactively repair a row whose every identity signal is already gone.
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
Revises: 20260818_seed_jira_mcp_app
Create Date: 2026-08-18
"""

from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision: str = "20260818_backfill_oauth_server_app_identity"
down_revision: str | None = "20260818_seed_jira_mcp_app"
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
    # A key shared by two OAuth apps is poisoned to None: that *key* then
    # resolves nothing rather than picking whichever app was seeded first.
    # Poisoning is per key, not per row -- a row whose name is ambiguous
    # still gets the provider fallback below, which is the point (the name
    # carries no information there, while an unambiguous provider does).
    # Keys and stored values are the raw strings — identities are opaque, and
    # every read path compares them exactly. Only ``transport`` is folded,
    # because it is a shape enum, not an identity (mirroring
    # classify_app_auth).
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

    # (row, auth-or-None, row provider) for every row still needing a stamp.
    pending: list[tuple[sa.RowMapping, dict | None, str | None]] = []
    for row in candidates:
        auth = row["auth"] if isinstance(row["auth"], dict) else None
        # Already-stamped means a nonblank *string* app_id — the only shape
        # the read path accepts (get_app_for_mcp_server rejects a non-string
        # app_id outright). A non-string or blank value is malformed metadata
        # that leaves the row permanently unresolvable, so such a row stays a
        # candidate and a successful match overwrites the malformed value.
        if auth is not None and _nonblank_str(auth.get("app_id")):
            continue
        provider = _nonblank_str(auth.get("provider")) if auth is not None else None
        pending.append((row, auth, provider))

    # Two passes so one app is never stamped onto two rows, and so which row
    # wins is decided by evidence rather than by row order. Name evidence is
    # strictly stronger than a provider (it names one app; a provider can be
    # shared), so every name match is resolved first and claims its app; the
    # provider fallback then only fills apps nobody claimed by name.
    #
    # This is not hypothetical: _ensure_user_mcp_server creates a *new* row
    # when a rename makes the old one unfindable, so two rows for one app are
    # exactly this migration's target population. Stamping both would make
    # _lookup_oauth_server_for_app -- which reads an unordered query -- pick
    # nondeterministically between them for configure/disconnect.
    resolutions: dict[int, tuple[str, str | None]] = {}
    claimed: set[str] = set()
    deferred: list[tuple[sa.RowMapping, dict | None, str]] = []

    for row, auth, provider in pending:
        resolved = apps_by_name.get(str(row["name"]) if row["name"] else "")
        if resolved is None:
            if provider:
                deferred.append((row, auth, provider))
            continue
        if provider and resolved[1] and provider != resolved[1]:
            # The row's own provider contradicts the name-resolved app — the
            # same conflict _ensure_server_matches_oauth_app refuses with a
            # ValueError. Stamping the name's app_id would create a row *no*
            # app claims (it fails this app's provider gate and the provider's
            # app_id gate), so refuse and leave the read-time provider
            # fallback exactly as it was.
            continue
        if resolved[0] in claimed:
            # Two rows named after one app: only the unique-name namespace on
            # mcp_servers.name makes this impossible in practice, so refuse
            # rather than assume it.
            logger.warning(
                "%s: app_id %r already claimed; leaving mcp_servers row %s unstamped",
                revision,
                resolved[0],
                row["id"],
            )
            continue
        claimed.add(resolved[0])
        resolutions[int(row["id"])] = resolved

    for row, auth, provider in deferred:
        resolved = apps_by_provider.get(provider)
        if resolved is None:
            continue
        if resolved[0] in claimed:
            logger.warning(
                "%s: app_id %r already claimed by a name match; leaving "
                "mcp_servers row %s to its pre-migration name fallback",
                revision,
                resolved[0],
                row["id"],
            )
            continue
        claimed.add(resolved[0])
        resolutions[int(row["id"])] = resolved

    stamped = 0
    for row, auth, _provider in pending:
        resolved = resolutions.get(int(row["id"]))
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
        stamped += 1

    logger.info(
        "%s: stamped %s of %s unstamped oauth row(s); %s left to the name fallback",
        revision,
        stamped,
        len(pending),
        len(pending) - stamped,
    )


def downgrade() -> None:
    # Deliberate no-op: stamped rows are indistinguishable from rows the
    # post-metadata writer created, and the extra stable identity is harmless
    # to every pre-migration reader (see module docstring).
    pass
