# Bounded upload cleanup discovery (2C-1)

Refs #1086. This extends [2B complete cleanup](complete-upload-cleanup.md)
without enabling detached-file collection. The second 2C PR supplies the
seven-day retention policy, detached scan index and independent fair scheduler.

## Work and completion contract

Claim capture reads only the fixed source and checksum-specific materialization
locators. It no longer enumerates directories under the KB reference gate.
The exact compensation claim, manifest and retired-ID fence still commit before
any destructive operation. All later discovery, checksums and disposal run
under the existing execution lock, with SQL connections and reference locks
released. Task-before-upload SQL locking and binder/reference errors are unchanged.

A version **2** manifest retains the original configuration, generation,
resource/parent identities and quarantine names. It also records local and
preview discovery jobs, acknowledged page revisions, phase completion and
per-resource disposal progress. Partial discovery is an outstanding obligation;
it cannot authorize a local/preview receipt or settlement. Local discovery
finishes before durable deletion. Preview discovery finishes before preview
disposal, and a separate bounded verification pass must prove no owned previews
remain before the preview receipt. Known uncertainty preserves the durable object
and handle; new uncertainty discovered after a durable receipt still retains every
remaining obligation. Neither budget exhaustion nor an EOF on a changed directory
means the resources are gone.

Each `run_uploaded_file_cleanup` activation shares these fixed budgets:

| Work | Limit |
| --- | ---: |
| Directory entries, EOF probes and directory-job attempts | 256 |
| Resource disposal attempts, child unlinks and directory removals | 128 |
| Process-local discovery streams | 8 |
| Converter quarantine traversal depth | 64 |

The limits include unrelated entries, missing-directory attempts and the final
preview verification, rather than only matching resources. Converter trees are
removed incrementally through no-follow directory descriptors; no unbounded
`rmtree` or `listdir` materialization remains in this protocol. A scan closes
before its directory is modified. Quarantine names and top-level ownership
evidence persist before partial disposal, so restarting at an already reduced
tree is safe. Files deeper than the supported converter limit remain pending
with their manifest for reconciliation.

These are operation-count bounds, not wall-clock guarantees. One provider call,
filesystem operation or whole-file checksum can still be slow. Captured manifests
grow with the number of owned resources; this PR does not introduce a separate
resource registry or a byte quota on the existing JSON recovery handle.

## Continuation, failures and overlapping workers

SQL stores captured ownership and the discovery obligation, never a portable
filesystem directory offset. A process can retain a stream between activations,
but only continue it against the acknowledged manifest revision. A page consumed
before a failed progress commit cannot be silently skipped. A restart, a different
worker or invalidated optimization re-enumerates the current unfinished directory
from its beginning, retaining earlier captured resource identities. Replacements
are conflicts; they cannot overwrite the original evidence.

Directory identity and modification evidence guard each page and EOF. A changed
listing restarts its scan; a replaced directory remains pending for reconciliation.
An actively changing shared directory may require a quiescent pass to finish.
Previously completed directory jobs and resource receipts remain committed across
restart. The per-file execution lock serializes workers, token takeover, discovery,
resource publication and settlement. A stale worker cannot apply progress or do
storage work on its replacement's token.

At capacity, new discovery work yields without deleting anything or losing its
claim. Active streams make progress through the existing rotating recovery scan;
failed operations release their optimization. Streams unused for one hour are
closed on a subsequent discovery activation, allowing another process's completed
claims to relinquish local optimization slots. Closing streams or exiting a
process does not erase persistent obligations. Cancellation drains the current
bounded activation through the existing worker-draining boundary, then propagates
cancellation with unfinished work still recoverable.

Budget exhaustion returns `yielded`. Recovery reports it as `deferred_budget`,
separately from actual `pending` uncertainty failures. This does not reclassify
ownership uncertainty or implement #2851's reconciliation/reporting work.
The recovery scan, stale-claim delay, poll interval and cursor remain unchanged;
2C-2 owns collector cadence and per-tick scheduling budgets.

## Ownership and matching axes

- **Stable ID:** preserve row/owner/file/key identity, exact claim token and
  manifest generation. Retired IDs remain fenced after settlement.
- **Name:** literal file-ID preview prefixes and producer temp prefixes locate
  resources; names alone do not replace captured parent/inode/checksum evidence.
- **Transport:** preserve captured backend, URI and provider routing validation.
- **Configuration:** use unified canonical roots and the effective shared
  coordination directory; no-follow walks reject replaced or escaping parents.
- **Authentication:** use current credentials on retry; manifests contain no
  authentication material. Credential rotation does not change ownership.
- **Ownership:** preserve external/shared sources, unproven historical temporaries
  and replacements. Capture discovered owned resources before deleting them.
- **Scope:** preserve exact normalized owner/workspace storage keys and 2A's
  cross-owner document/active-target protection.

## Activation and rollback

No SQL schema migration or environment-variable change is required for 2C-1.
Fresh installations and populated SQLite/PostgreSQL deployments use the existing
nullable JSON manifest column. Existing version 1 claims adopt discovery state
without replacing resource evidence or repeating their committed phase receipts.
Complete version 1 receipts remain valid.

Stop all upload/local/preview producers, reference writers and cleanup/recovery
workers before deployment; restart compatible versions together with the same
resource roots and effective coordination locks. A version 2 manifest intentionally
fails old version 1 workers' validation rather than allowing them to ignore its
unfinished discovery. Mixed versions are unsupported.

Before rolling back application code, stop new claims and drain or reconcile all
version 2 claims with the new recovery worker. Verify no version 2 manifests remain,
then quiesce every writer/collector and restart the prior version together. Never
strip discovery metadata to make a pending manifest appear compatible. Preserve
the SQL database, resource roots and LanceDB coordination directory together.
Do not erase SQL fences or `.claimed` markers, or replace live reference/execution
lock inodes. Safe compaction is separate work.

## Remaining delivery

This completes 2C's per-manifest discovery/disposal bound, not all of 2C or #1086.
Detached rows remain excluded from the task-less collector until 2C-2. Historical
inventory, ambiguous unmarked rows and managed legacy/local-only backfill remain
later delivery. Files uploaded before rollout are not permanently excluded from
the future detached policy merely because of their creation date.

Existing scoped follow-ups remain #2835 (canonical-path republishing), #2836
(compensation-batch reference error isolation), #2848 (immediate direct-delete
transaction scope), #2849 (preview locator consolidation), and #2851 (uncertainty
classification). Account/team erasure, backup replay and recovery UI remain outside
this PR. No new deletion entry point or unrelated producer layout is introduced.
