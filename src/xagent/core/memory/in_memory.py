from __future__ import annotations

import uuid
from typing import Any, List, Optional

from .base import MemoryStore
from .core import MemoryNote, MemoryResponse
from .scope_columns import SCOPE_EXCLUSIVE_FILTER_KEY, encode_scope_dims


class InMemoryMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._store: dict[str, MemoryNote] = {}

    @staticmethod
    def _is_scope_excluded(note: MemoryNote, filters: dict[str, Any]) -> bool:
        """#822: strict dimension-less exclusion (the ``__scope_exclusive__``
        directive, ``SCOPE_EXCLUSIVE_FILTER_KEY``) — a note carrying any scope
        dimension is excluded."""
        return bool(filters.get(SCOPE_EXCLUSIVE_FILTER_KEY)) and bool(
            encode_scope_dims(note.metadata)
        )

    @staticmethod
    def _matches_metadata_filters(
        metadata: dict[str, Any], metadata_filters: dict[str, Any]
    ) -> bool:
        """Metadata equality for both the nested ``filters["metadata"]`` dict
        and flat metadata keys, matching the string-coerced semantics of
        ``LanceDBMemoryStore`` so ``UserIsolatedMemoryStore`` isolation and
        ad-hoc filters behave the same on both stores (#842)."""
        return all(
            str(metadata.get(key, "")) == str(value)
            for key, value in metadata_filters.items()
        )

    def add(self, note: MemoryNote) -> MemoryResponse:
        note_id = note.id or str(uuid.uuid4())
        note.id = note_id
        self._store[note_id] = note
        return MemoryResponse(success=True, memory_id=note_id)

    def get(self, note_id: str) -> MemoryResponse:
        note = self._store.get(note_id)
        if note:
            return MemoryResponse(success=True, memory_id=note_id, content=note)
        else:
            return MemoryResponse(
                success=False, error="Note not found", memory_id=note_id
            )

    def update(self, note: MemoryNote) -> MemoryResponse:
        if note.id is None or note.id not in self._store:
            return MemoryResponse(
                success=False, error="Note not found or ID missing", memory_id=note.id
            )
        self._store[note.id] = note
        return MemoryResponse(success=True, memory_id=note.id)

    def delete(self, note_id: str) -> MemoryResponse:
        if note_id in self._store:
            del self._store[note_id]
            return MemoryResponse(success=True, memory_id=note_id)
        else:
            return MemoryResponse(
                success=False, error="Note not found", memory_id=note_id
            )

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> list[MemoryNote]:
        # Flat metadata filters outside the known keys (string-coerced, like
        # LanceDBMemoryStore) — note-independent, so computed once.
        other_filters = {
            key: value
            for key, value in (filters or {}).items()
            if key not in ("category", "metadata", SCOPE_EXCLUSIVE_FILTER_KEY)
        }
        results = []
        for note in self._store.values():
            if query.lower() in note.content.lower():
                if filters:
                    match = True

                    if self._is_scope_excluded(note, filters):
                        match = False

                    # Category filter
                    if "category" in filters and note.category != filters["category"]:
                        match = False

                    # Nested metadata filters (the shape UserIsolatedMemoryStore
                    # emits for user_id/scope isolation)
                    if "metadata" in filters and not self._matches_metadata_filters(
                        note.metadata, filters["metadata"]
                    ):
                        match = False

                    if other_filters and not self._matches_metadata_filters(
                        note.metadata, other_filters
                    ):
                        match = False

                    if match:
                        results.append(note)
                else:
                    results.append(note)
        return results[:k]

    def clear(self) -> None:
        self._store.clear()

    def list_all(self, filters: Optional[dict[str, Any]] = None) -> List[MemoryNote]:
        results = list(self._store.values())

        if filters:
            # Flat metadata filters outside the known keys — same string-coerced
            # equality as search() and LanceDBMemoryStore.list_all (previously
            # ignored entirely, the same fail-open shape as the nested metadata
            # case). Note-independent, so computed once.
            other_filters = {
                key: value
                for key, value in filters.items()
                if key
                not in (
                    "category",
                    "metadata",
                    "date_from",
                    "date_to",
                    "tags",
                    "keywords",
                    SCOPE_EXCLUSIVE_FILTER_KEY,
                )
            }
            filtered_results = []
            for note in results:
                match = True

                if self._is_scope_excluded(note, filters):
                    match = False

                # Category filter
                if "category" in filters and note.category != filters["category"]:
                    match = False

                # Nested metadata filters (the shape UserIsolatedMemoryStore
                # emits for user_id/scope isolation — previously ignored here,
                # which made user_id isolation fail-open, #842)
                if "metadata" in filters and not self._matches_metadata_filters(
                    note.metadata, filters["metadata"]
                ):
                    match = False

                # Date range filters
                if "date_from" in filters and note.timestamp < filters["date_from"]:
                    match = False
                if "date_to" in filters and note.timestamp > filters["date_to"]:
                    match = False

                # Tag filter
                if "tags" in filters:
                    required_tags = filters["tags"]
                    if not all(tag in note.tags for tag in required_tags):
                        match = False

                # Keyword filter
                if "keywords" in filters:
                    required_keywords = filters["keywords"]
                    if not all(
                        keyword in note.keywords for keyword in required_keywords
                    ):
                        match = False

                if other_filters and not self._matches_metadata_filters(
                    note.metadata, other_filters
                ):
                    match = False

                if match:
                    filtered_results.append(note)

            results = filtered_results

        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x.timestamp, reverse=True)

        return results

    def get_stats(self) -> dict[str, Any]:
        total_count = len(self._store)
        category_counts: dict[str, int] = {}
        tag_counts: dict[str, int] = {}

        for note in self._store.values():
            # Count by category
            category_counts[note.category] = category_counts.get(note.category, 0) + 1

            # Count tags
            for tag in note.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            "total_count": total_count,
            "category_counts": category_counts,
            "tag_counts": tag_counts,
            "memory_store_type": "in_memory",
        }
