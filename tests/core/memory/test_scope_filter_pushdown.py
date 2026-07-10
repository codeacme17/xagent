"""Recall regression + scope-filter semantics for the where pushdown (#822, 822-02).

The centerpiece is the population-independence test: a single ``user_id`` shared
by ``N`` principals (isolated only by scope dimensions), the target principal
holding ``R`` notes, searched at ``k = R``. With Python post-filtering the target
is crowded out of the top-k before the filter runs and recall collapses to
``~1/N``; with the ``where`` prefilter it stays ``1.0`` regardless of ``N``.

All notes share one identical vector so the vector distance cannot itself
separate principals — top-k selection is decided purely by whether the scope
filter is applied before (prefilter) or after (post-filter) the limit.
"""

from __future__ import annotations

import json
import shutil
import tempfile

import lancedb  # type: ignore
import pytest

from xagent.core.execution_scope import MEMORY_DIMENSION_METADATA_PREFIX
from xagent.core.memory.core import MemoryNote
from xagent.core.memory.lancedb import LanceDBMemoryStore
from xagent.core.model.embedding import BaseEmbedding
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table

P = MEMORY_DIMENSION_METADATA_PREFIX
DIM = 8
_VECTOR = [0.1] * DIM


class FixedEmbedding(BaseEmbedding):
    """Every text embeds to the same vector, so distance cannot rank principals."""

    def encode(self, text, dimension=None, instruct=None):
        if isinstance(text, str):
            return list(_VECTOR)
        return [list(_VECTOR) for _ in text]

    def get_dimension(self):
        return DIM

    @property
    def abilities(self):
        return ["embed"]


@pytest.fixture
def temp_db_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def _scope_filter(tenant, user_id=1):
    return {"metadata": {"user_id": user_id, f"{P}tenant": tenant}}


def _bulk_build(db_dir, n_principals, r=10, name="mem"):
    """Create a shared collection: one user_id, N principals, R notes each.

    Notes are inserted round-robin (principal-major within each note index) so
    the target principal's notes are spread across the population rather than
    clustered — without a prefilter, an insertion-order top-k would pick at most
    ~1 of them.
    """
    db = lancedb.connect(db_dir)
    ts = "2026-01-01T00:00:00"
    rows = []
    for j in range(r):
        for p in range(n_principals):
            tenant = f"t{p}"
            note_id = f"{tenant}-{j}"
            metadata = {
                "content": note_id,
                "timestamp": ts,
                "user_id": 1,
                f"{P}tenant": tenant,
            }
            rows.append(
                {
                    "id": note_id,
                    "text": note_id,
                    "metadata": json.dumps(metadata),
                    "user_id": 1,
                    "scope_dims": [f"tenant={tenant}"],
                    "vector": list(_VECTOR),
                }
            )
    table = db.create_table(name, data=rows)
    _safe_close_table(table)


def _store(db_dir, name="mem"):
    return LanceDBMemoryStore(
        db_dir=db_dir, collection_name=name, embedding_model=FixedEmbedding()
    )


@pytest.mark.parametrize("n_principals", [10, 100, 1000])
def test_scoped_recall_is_population_independent(temp_db_dir, n_principals):
    r = 10
    _bulk_build(temp_db_dir, n_principals, r=r)
    store = _store(temp_db_dir)

    # Search the target principal (t0) at k = R over a collection of N*R notes.
    results = store.search("anything", k=r, filters=_scope_filter("t0"))

    returned = {n.content for n in results}
    expected = {f"t0-{j}" for j in range(r)}
    recall = len(returned & expected) / r
    assert recall == 1.0, (
        f"N={n_principals}: recall collapsed to {recall} "
        f"(returned {len(results)} notes)"
    )
    # And nothing from another principal leaked in.
    assert returned <= expected


def test_unscoped_search_sees_all_principals(temp_db_dir):
    _bulk_build(temp_db_dir, n_principals=5, r=4)
    store = _store(temp_db_dir)

    # Unscoped: user_id only, no dimension predicate -> sees every principal.
    results = store.search("anything", k=1000, filters={"metadata": {"user_id": 1}})
    tenants = {n.content.split("-")[0] for n in results}
    assert tenants == {f"t{p}" for p in range(5)}


def test_dimension_value_matches_exactly_through_search(temp_db_dir):
    """End-to-end: a value with a would-be LIKE wildcard (`_`) matches only its
    exact sibling, driven through store.search (guards the array_contains term
    against any accidental widening)."""
    store = _store(temp_db_dir)
    for tenant in ("a_b", "axb", "a-b", "ab"):
        store.add(
            MemoryNote(
                id=tenant, content=tenant, metadata={"user_id": 1, f"{P}tenant": tenant}
            )
        )

    for tenant in ("a_b", "axb", "a-b", "ab"):
        got = {
            n.content for n in store.search("x", k=10, filters=_scope_filter(tenant))
        }
        assert got == {tenant}, f"{tenant!r} matched {got!r}"


def test_pushed_dims_combine_with_residual_category_filter(temp_db_dir):
    """user_id + dimension go into the `where` prefilter; a category filter is
    not pushable and stays a Python post-filter over the already-scoped rows."""
    store = _store(temp_db_dir)
    store.add(
        MemoryNote(
            id="w",
            content="w",
            category="work",
            metadata={"user_id": 1, f"{P}tenant": "acme"},
        )
    )
    store.add(
        MemoryNote(
            id="p",
            content="p",
            category="personal",
            metadata={"user_id": 1, f"{P}tenant": "acme"},
        )
    )

    got = {
        n.content
        for n in store.search(
            "x",
            k=10,
            filters={
                "category": "work",
                "metadata": {"user_id": 1, f"{P}tenant": "acme"},
            },
        )
    }
    assert got == {"w"}


def test_subset_match_superset_returned_missing_excluded(temp_db_dir):
    store = _store(temp_db_dir)
    # A carries a superset {tenant, agent}; B carries only {tenant}; C differs.
    store.add(
        MemoryNote(
            id="A",
            content="A",
            metadata={"user_id": 1, f"{P}tenant": "acme", f"{P}agent": "x"},
        )
    )
    store.add(
        MemoryNote(id="B", content="B", metadata={"user_id": 1, f"{P}tenant": "acme"})
    )
    store.add(
        MemoryNote(id="C", content="C", metadata={"user_id": 1, f"{P}tenant": "other"})
    )

    # Query {tenant=acme}: A (superset) and B match; C does not.
    got = {n.content for n in store.search("x", k=10, filters=_scope_filter("acme"))}
    assert got == {"A", "B"}

    # Query {tenant=acme, agent=x}: only A carries both.
    got = {
        n.content
        for n in store.search(
            "x",
            k=10,
            filters={"metadata": {"user_id": 1, f"{P}tenant": "acme", f"{P}agent": "x"}},
        )
    }
    assert got == {"A"}


def test_scoped_search_excludes_other_users(temp_db_dir):
    store = _store(temp_db_dir)
    store.add(
        MemoryNote(id="u1", content="u1", metadata={"user_id": 1, f"{P}tenant": "acme"})
    )
    store.add(
        MemoryNote(id="u2", content="u2", metadata={"user_id": 2, f"{P}tenant": "acme"})
    )

    got = {n.content for n in store.search("x", k=10, filters=_scope_filter("acme", 1))}
    assert got == {"u1"}
