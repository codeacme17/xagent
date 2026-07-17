import json
import logging
import shutil
import tempfile

import pytest

from xagent.core.memory.core import MemoryNote
from xagent.core.memory.lancedb import LanceDBMemoryStore, _safe_close_table


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for the database."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model for testing."""
    from xagent.core.model.embedding import BaseEmbedding

    class MockEmbedding(BaseEmbedding):
        def __init__(self):
            self._dimension = 64

        def encode(self, text, dimension=None, instruct=None):
            if isinstance(text, str):
                return [0.1] * self._dimension
            else:
                return [[0.1] * self._dimension] * len(text)

        def get_dimension(self):
            return self._dimension

        @property
        def abilities(self):
            return ["embed"]

    return MockEmbedding()


@pytest.fixture
def memory_store(temp_db_dir, mock_embedding_model):
    """Create a LanceDB memory store for testing."""
    return LanceDBMemoryStore(
        db_dir=temp_db_dir,
        collection_name="test_memories",
        embedding_model=mock_embedding_model,
    )


def test_add_and_get_memory(memory_store):
    """Test adding and retrieving a memory."""
    note = MemoryNote(
        content="Test memory content",
        keywords=["test", "memory"],
        tags=["important"],
        category="test",
        metadata={"user": "alice", "priority": "high"},
    )

    # Add memory
    response = memory_store.add(note)
    assert response.success
    assert response.memory_id is not None

    # Get memory
    get_response = memory_store.get(response.memory_id)
    assert get_response.success
    retrieved_note = get_response.content
    assert isinstance(retrieved_note, MemoryNote)
    assert retrieved_note.content == "Test memory content"
    assert retrieved_note.keywords == ["test", "memory"]
    assert retrieved_note.tags == ["important"]
    assert retrieved_note.category == "test"
    assert retrieved_note.metadata["user"] == "alice"
    assert retrieved_note.metadata["priority"] == "high"


def test_update_memory(memory_store):
    """Test updating an existing memory."""
    note = MemoryNote(content="Original content", metadata={"version": 1})

    # Add memory
    response = memory_store.add(note)
    memory_id = response.memory_id

    # Update memory
    updated_note = MemoryNote(
        id=memory_id,
        content="Updated content",
        metadata={"version": 2, "updated": True},
    )

    update_response = memory_store.update(updated_note)
    assert update_response.success

    # Get updated memory
    get_response = memory_store.get(memory_id)
    assert get_response.success
    retrieved_note = get_response.content
    assert retrieved_note.content == "Updated content"
    assert retrieved_note.metadata["version"] == 2
    assert retrieved_note.metadata["updated"] is True


def test_delete_memory(memory_store):
    """Test deleting a memory."""
    note = MemoryNote(content="To be deleted")

    # Add memory
    response = memory_store.add(note)
    memory_id = response.memory_id

    # Delete memory
    delete_response = memory_store.delete(memory_id)
    assert delete_response.success

    # Try to get deleted memory
    get_response = memory_store.get(memory_id)
    assert not get_response.success
    assert "not found" in get_response.error.lower()


def test_search_memories(memory_store):
    """Test searching memories."""
    # Add multiple memories
    memories = [
        MemoryNote(content="Cats are cute pets", keywords=["pets", "cats"]),
        MemoryNote(content="Dogs are loyal companions", keywords=["pets", "dogs"]),
        MemoryNote(
            content="Python is a programming language",
            keywords=["programming", "python"],
        ),
        MemoryNote(content="Machine learning is a subset of AI", keywords=["ai", "ml"]),
    ]

    for memory in memories:
        memory_store.add(memory)

    # Search for "pets"
    results = memory_store.search("pets", k=5)
    assert len(results) >= 2
    pet_contents = [r.content for r in results]
    assert any("cats" in content.lower() for content in pet_contents)
    assert any("dogs" in content.lower() for content in pet_contents)

    # Search with category filter
    results = memory_store.search("programming", k=5, filters={"category": "general"})
    assert len(results) >= 1
    assert any("python" in r.content.lower() for r in results)


def test_search_with_metadata_filters(memory_store):
    """Test searching with metadata filters."""
    # Add memories with different metadata
    memories = [
        MemoryNote(
            content="Important work task", metadata={"type": "work", "priority": "high"}
        ),
        MemoryNote(
            content="Personal reminder",
            metadata={"type": "personal", "priority": "low"},
        ),
        MemoryNote(
            content="Another work item", metadata={"type": "work", "priority": "medium"}
        ),
    ]

    added_memories = []
    for memory in memories:
        response = memory_store.add(memory)
        if response.success:
            added_memories.append(memory)

    # Search with type filter
    results = memory_store.search("work", k=5, filters={"type": "work"})
    assert len(results) >= 1  # At least one work item should match
    for result in results:
        assert result.metadata.get("type") == "work"

    # Search with priority filter
    results = memory_store.search("", k=5, filters={"priority": "high"})
    assert len(results) >= 1
    for result in results:
        assert result.metadata.get("priority") == "high"


def test_clear_memories(memory_store):
    """Test clearing all memories."""
    # Add some memories
    memory_store.add(MemoryNote(content="Memory 1"))
    memory_store.add(MemoryNote(content="Memory 2"))

    # Verify memories exist
    results = memory_store.search("Memory", k=10)
    assert len(results) >= 2

    # Clear all memories
    memory_store.clear()

    # Verify memories are cleared
    results = memory_store.search("Memory", k=10)
    assert len(results) == 0


def test_auto_id_generation(memory_store):
    """Test that IDs are automatically generated when not provided."""
    from datetime import datetime

    # Create a new MemoryNote without specifying ID
    note = MemoryNote(
        content="Test without ID",
        keywords=[],
        tags=[],
        category="general",
        timestamp=datetime.now(),
        mime_type="text/plain",
        metadata={},
    )

    # The note should have an auto-generated ID
    original_id = note.id
    assert original_id is not None

    response = memory_store.add(note)
    assert response.success
    assert response.memory_id is not None

    # Get the memory to verify it has the same ID
    get_response = memory_store.get(response.memory_id)
    assert get_response.success
    assert get_response.content.id == response.memory_id


def test_list_all_memories(memory_store):
    """Test listing all memories with new list_all method."""
    # Add multiple memories
    memories = [
        MemoryNote(content="Memory 1", category="general"),
        MemoryNote(content="Memory 2", category="system"),
        MemoryNote(content="Memory 3", category="general"),
    ]

    for memory in memories:
        memory_store.add(memory)

    # List all memories
    results = memory_store.list_all()
    assert len(results) >= 3

    # Test with category filter
    general_results = memory_store.list_all(filters={"category": "general"})
    assert len(general_results) >= 2
    assert all(r.category == "general" for r in general_results)

    system_results = memory_store.list_all(filters={"category": "system"})
    assert len(system_results) >= 1
    assert all(r.category == "system" for r in system_results)


def test_get_stats(memory_store):
    """Test getting memory store statistics."""
    # Add some memories
    memories = [
        MemoryNote(content="General memory 1", category="general", tags=["important"]),
        MemoryNote(content="General memory 2", category="general", tags=["normal"]),
        MemoryNote(
            content="System memory", category="system", tags=["system", "config"]
        ),
    ]

    for memory in memories:
        memory_store.add(memory)

    # Get stats
    stats = memory_store.get_stats()

    assert stats["total_count"] >= 3
    assert stats["category_counts"]["general"] >= 2
    assert stats["category_counts"]["system"] >= 1
    assert stats["tag_counts"]["important"] >= 1
    assert stats["tag_counts"]["normal"] >= 1
    assert stats["tag_counts"]["system"] >= 1
    assert stats["memory_store_type"] == "lancedb"


def test_list_all_with_date_filters(memory_store):
    """Test listing memories with date range filters."""
    from datetime import datetime, timedelta

    # Add memories with different timestamps
    now = datetime.now()
    old_time = now - timedelta(days=1)

    old_memory = MemoryNote(content="Old memory", category="test", timestamp=old_time)
    recent_memory = MemoryNote(content="Recent memory", category="test", timestamp=now)

    assert memory_store.add(old_memory).success
    assert memory_store.add(recent_memory).success

    # Test date_from filter
    results = memory_store.list_all(filters={"date_from": now})
    assert len(results) >= 1
    assert all(r.timestamp >= now for r in results)

    # Test date_to filter
    results = memory_store.list_all(filters={"date_to": now})
    assert len(results) >= 1
    assert all(r.timestamp <= now for r in results)


def test_list_all_with_nested_metadata_filter(memory_store):
    """list_all must honor the nested {"metadata": {...}} shape (as used by
    the user-isolation layer), matching search()'s filter semantics."""
    assert memory_store.add(
        MemoryNote(content="alice note", metadata={"user_id": 1})
    ).success
    assert memory_store.add(
        MemoryNote(content="bob note", metadata={"user_id": 2})
    ).success

    nested = {"metadata": {"user_id": 1}}
    # list_all must agree with search on the nested-metadata filter.
    assert len(memory_store.search("", k=100, filters=nested)) == 1
    results = memory_store.list_all(filters=nested)
    assert len(results) == 1
    assert results[0].metadata.get("user_id") == 1
    assert results[0].content == "alice note"


def test_non_dict_nested_metadata_filter_matches_nothing(memory_store):
    """A malformed (non-dict) filters["metadata"] value must not crash —
    it can't match anything."""
    assert memory_store.add(
        MemoryNote(content="hello world", metadata={"user_id": 1})
    ).success

    for bad in (None, "abc", 42):
        assert memory_store.search("hello", k=10, filters={"metadata": bad}) == []
        assert memory_store.list_all(filters={"metadata": bad}) == []


def test_list_all_with_tag_filters(memory_store):
    """Test listing memories with tag filters."""
    assert memory_store.add(
        MemoryNote(content="Work task", tags=["work", "urgent"])
    ).success
    assert memory_store.add(
        MemoryNote(content="Personal note", tags=["personal"])
    ).success
    assert memory_store.add(
        MemoryNote(content="Another work item", tags=["work"])
    ).success

    # Notes carrying the "work" tag
    results = memory_store.list_all(filters={"tags": ["work"]})
    assert len(results) == 2
    assert all("work" in r.tags for r in results)

    # A note must carry all requested tags
    results = memory_store.list_all(filters={"tags": ["work", "urgent"]})
    assert len(results) == 1
    assert all({"work", "urgent"} <= set(r.tags) for r in results)


def test_list_all_with_keyword_filters(memory_store):
    """Test listing memories with keyword filters."""
    assert memory_store.add(
        MemoryNote(content="AI in Python", keywords=["python", "ai"])
    ).success
    assert memory_store.add(
        MemoryNote(content="Cooking recipe", keywords=["cooking"])
    ).success
    assert memory_store.add(
        MemoryNote(content="Python tips", keywords=["python"])
    ).success

    # Notes carrying the "python" keyword
    results = memory_store.list_all(filters={"keywords": ["python"]})
    assert len(results) == 2
    assert all("python" in r.keywords for r in results)

    # A note must carry all requested keywords
    results = memory_store.list_all(filters={"keywords": ["python", "ai"]})
    assert len(results) == 1
    assert all({"python", "ai"} <= set(r.keywords) for r in results)


def test_embedding_fallback(memory_store):
    """Test fallback to text search when embedding fails."""
    # Create a memory store with failing embedding model
    temp_dir = tempfile.mkdtemp()
    try:
        from xagent.core.model.embedding import BaseEmbedding

        class FailingEmbedding(BaseEmbedding):
            def __init__(self):
                self._dimension = 64

            def encode(self, text, dimension=None, instruct=None):
                raise Exception("Embedding failed")

            def get_dimension(self):
                return self._dimension

            @property
            def abilities(self):
                return ["embed"]

        failing_model = FailingEmbedding()

        fallback_store = LanceDBMemoryStore(
            db_dir=temp_dir,
            collection_name="test_fallback",
            embedding_model=failing_model,
        )

        # Add memory without embedding
        note = MemoryNote(content="Test content for fallback")
        response = fallback_store.add(note)
        assert response.success

        # Search should still work with text matching
        results = fallback_store.search("Test content", k=5)
        assert len(results) >= 1
        assert results[0].content == "Test content for fallback"

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_large_content_handling(memory_store):
    """Test handling of large content."""
    large_content = "This is a test sentence. " * 100  # Create large content

    note = MemoryNote(content=large_content, metadata={"size": "large"})
    response = memory_store.add(note)
    assert response.success

    # Retrieve and verify
    get_response = memory_store.get(response.memory_id)
    assert get_response.success
    assert get_response.content.content == large_content
    assert get_response.content.metadata["size"] == "large"


def test_unicode_content(memory_store):
    """Test handling of unicode content."""
    unicode_content = "测试中文内容 🚀 Python & AI"

    note = MemoryNote(content=unicode_content, keywords=["中文", "test"])
    response = memory_store.add(note)
    assert response.success

    # Retrieve and verify
    get_response = memory_store.get(response.memory_id)
    assert get_response.success
    assert get_response.content.content == unicode_content
    assert get_response.content.keywords == ["中文", "test"]


def test_invalid_memory_id(memory_store):
    """Test handling of invalid memory IDs."""
    # Try to get non-existent memory
    response = memory_store.get("non-existent-id")
    assert not response.success
    assert "not found" in response.error.lower()

    # Try to update non-existent memory
    note = MemoryNote(id="non-existent-id", content="Updated")
    response = memory_store.update(note)
    assert not response.success
    assert "not found" in response.error.lower()

    # Try to delete non-existent memory
    response = memory_store.delete("non-existent-id")
    # This might succeed or fail depending on implementation
    # The important thing is that it doesn't crash


# --- #847: search() must not silently truncate on a mid-loop row failure ---


def _insert_raw_row(store, collection_name, record):
    """Insert a row directly into the LanceDB table, bypassing add()."""
    conn = store._vector_store.get_raw_connection()
    table = conn.open_table(collection_name)
    try:
        table.add([record])
    finally:
        _safe_close_table(table)


def test_dict_to_memory_note_missing_timestamp(memory_store):
    """A legacy row whose metadata lacks `timestamp` must convert using the
    model default instead of raising ValidationError (#847 root cause)."""
    note = memory_store._dict_to_memory_note(
        {
            "id": "legacy-1",
            "text": "legacy content",
            "metadata": json.dumps({"content": "legacy content"}),
        }
    )
    assert note.content == "legacy content"
    assert note.timestamp is not None
    assert "timestamp" not in note.metadata


def test_get_returns_legacy_row_missing_timestamp(memory_store):
    """get() uses the same converter, so a timestamp-less legacy row must load
    with the default timestamp instead of surfacing a generic failure (#847)."""
    # Settle the table schema (vector column dimension) with a regular add
    # before bypassing it with a raw legacy row.
    assert memory_store.add(MemoryNote(content="schema-settling note")).success

    _insert_raw_row(
        memory_store,
        "test_memories",
        {
            "id": "legacy-get-no-ts",
            "vector": [0.1] * 64,
            "text": "legacy get target",
            "metadata": json.dumps({"content": "legacy get target"}),
            "user_id": None,
            "scope_dims": [],
        },
    )

    response = memory_store.get("legacy-get-no-ts")
    assert response.success
    note = response.content
    assert isinstance(note, MemoryNote)
    assert note.content == "legacy get target"
    assert note.timestamp is not None


def test_search_returns_legacy_rows_missing_timestamp(memory_store):
    """A vector search whose top-k contains a timestamp-less legacy row must
    return all matching rows, including the legacy one (#847)."""
    for i in range(3):
        assert memory_store.add(MemoryNote(content=f"well-formed note {i}")).success

    _insert_raw_row(
        memory_store,
        "test_memories",
        {
            "id": "legacy-no-ts",
            "vector": [0.1] * 64,
            "text": "legacy note without timestamp",
            "metadata": json.dumps({"content": "legacy note without timestamp"}),
            "user_id": None,
            "scope_dims": [],
        },
    )

    results = memory_store.search("note", k=10)
    ids = {r.id for r in results}
    contents = {r.content for r in results}
    assert {f"well-formed note {i}" for i in range(3)} <= contents
    assert "legacy-no-ts" in ids


def test_search_skips_malformed_row_without_truncation(memory_store, caplog):
    """A row whose conversion genuinely fails (unparsable timestamp) must be
    skipped *and logged*, not abort the vector branch after earlier appends —
    which used to suppress the text fallback and silently truncate the
    results (#847).

    The malformed row is seeded between well-formed rows so that, pre-fix, the
    mid-loop escape would drop whichever rows the scan happened to visit after
    it (LanceDB does not guarantee tied-distance iteration order, so exactly
    which rows those are is an implementation detail). The assertions below
    are order-independent — set membership plus log presence — so the test
    stays valid regardless of how tied rows are iterated."""
    for i in range(2):
        assert memory_store.add(MemoryNote(content=f"well-formed note {i}")).success

    _insert_raw_row(
        memory_store,
        "test_memories",
        {
            "id": "malformed-ts",
            "vector": [0.1] * 64,
            "text": "malformed note",
            "metadata": json.dumps(
                {"content": "malformed note", "timestamp": "not-a-datetime"}
            ),
            "user_id": None,
            "scope_dims": [],
        },
    )

    for i in range(2, 4):
        assert memory_store.add(MemoryNote(content=f"well-formed note {i}")).success

    with caplog.at_level(logging.WARNING, logger="xagent.core.memory.lancedb"):
        results = memory_store.search("note", k=10)

    contents = {r.content for r in results}
    assert {f"well-formed note {i}" for i in range(4)} <= contents
    assert all(r.id != "malformed-ts" for r in results)
    assert any(
        "Skipping malformed memory row" in record.getMessage()
        and "malformed-ts" in record.getMessage()
        for record in caplog.records
    )


def test_text_fallback_skips_malformed_row(temp_db_dir, caplog):
    """The text-search fallback must also skip (and log) a malformed row
    instead of escaping to the outer except and returning an empty result
    set (#847)."""
    from xagent.core.model.embedding import BaseEmbedding

    class FailingEmbedding(BaseEmbedding):
        def __init__(self):
            self._dimension = 64

        def encode(self, text, dimension=None, instruct=None):
            raise Exception("Embedding failed")

        def get_dimension(self):
            return self._dimension

        @property
        def abilities(self):
            return ["embed"]

    store = LanceDBMemoryStore(
        db_dir=temp_db_dir,
        collection_name="test_text_fallback_847",
        embedding_model=FailingEmbedding(),
    )

    for i in range(3):
        assert store.add(MemoryNote(content=f"well-formed note {i}")).success

    _insert_raw_row(
        store,
        "test_text_fallback_847",
        {
            "id": "malformed-ts",
            "text": "malformed note",
            "metadata": json.dumps(
                {"content": "malformed note", "timestamp": "not-a-datetime"}
            ),
            "user_id": None,
            "scope_dims": [],
        },
    )

    with caplog.at_level(logging.WARNING, logger="xagent.core.memory.lancedb"):
        results = store.search("note", k=10)

    contents = {r.content for r in results}
    assert {f"well-formed note {i}" for i in range(3)} <= contents
    assert all(r.id != "malformed-ts" for r in results)
    assert any(
        "Skipping malformed memory row" in record.getMessage()
        and "malformed-ts" in record.getMessage()
        and "text search" in record.getMessage()
        for record in caplog.records
    )


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
