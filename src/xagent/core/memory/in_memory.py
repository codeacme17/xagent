from __future__ import annotations

import uuid
from typing import Any, Collection, List, Optional

from .base import MemoryStore, comparable_timestamp
from .core import MemoryNote, MemoryResponse
from .scope_columns import SCOPE_EXCLUSIVE_FILTER_KEY, encode_scope_dims


class InMemoryMemoryStore(MemoryStore):
    # Filter keys with dedicated handling in _matches_filters; anything else
    # is treated as a flat metadata-equality filter.
    _KNOWN_FILTER_KEYS = frozenset(
        {
            "category",
            "metadata",
            "date_from",
            "date_to",
            "tags",
            "keywords",
            SCOPE_EXCLUSIVE_FILTER_KEY,
        }
    )

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
        ad-hoc filters behave the same on both stores (#842). A non-dict
        filter value cannot match anything (rather than crashing on a
        malformed caller-supplied ``filters["metadata"]``)."""
        if not isinstance(metadata_filters, dict):
            return False
        return all(
            str(metadata.get(key, "")) == str(value)
            for key, value in metadata_filters.items()
        )

    @staticmethod
    def _required_items(value: Any) -> Optional[Collection[Any]]:
        """Normalize a ``tags``/``keywords`` filter value: a plain string is a
        single required item (not iterated per character), an iterable is many,
        and anything else can't match (rather than raising ``TypeError`` on
        malformed caller-supplied filters — the same no-match policy as
        ``_matches_metadata_filters``)."""
        if isinstance(value, str):
            return (value,)
        if isinstance(value, (list, tuple, set, frozenset)):
            return value
        return None

    @classmethod
    def _flat_other_filters(cls, filters: Optional[dict[str, Any]]) -> dict[str, Any]:
        """Filter keys without dedicated handling, applied as flat metadata
        equality. Note-independent — compute once per call, not per note."""
        return {
            key: value
            for key, value in (filters or {}).items()
            if key not in cls._KNOWN_FILTER_KEYS
        }

    @classmethod
    def _matches_filters(
        cls,
        note: MemoryNote,
        filters: dict[str, Any],
        other_filters: dict[str, Any],
    ) -> bool:
        """Single filter dispatch shared by ``search()`` and ``list_all()``,
        so the two methods cannot drift apart on filter semantics (#842).

        ``other_filters`` is ``_flat_other_filters(filters)``, precomputed by
        the caller so the per-note check does not rebuild it.
        """
        if cls._is_scope_excluded(note, filters):
            return False

        if "category" in filters and note.category != filters["category"]:
            return False

        # Nested metadata filters (the shape UserIsolatedMemoryStore emits
        # for user_id/scope isolation — before #842 search() never matched
        # them and list_all() ignored them, fail-open)
        if "metadata" in filters and not cls._matches_metadata_filters(
            note.metadata, filters["metadata"]
        ):
            return False

        # Date range filters (both sides tz-normalized so an aware filter
        # against the naive default timestamps cannot raise TypeError)
        if "date_from" in filters or "date_to" in filters:
            note_ts = comparable_timestamp(note.timestamp)
            if "date_from" in filters and note_ts < comparable_timestamp(
                filters["date_from"]
            ):
                return False
            if "date_to" in filters and note_ts > comparable_timestamp(
                filters["date_to"]
            ):
                return False

        # Tag filter (all required tags present)
        if "tags" in filters:
            required_tags = cls._required_items(filters["tags"])
            if required_tags is None or not all(
                tag in note.tags for tag in required_tags
            ):
                return False

        # Keyword filter (all required keywords present)
        if "keywords" in filters:
            required_keywords = cls._required_items(filters["keywords"])
            if required_keywords is None or not all(
                keyword in note.keywords for keyword in required_keywords
            ):
                return False

        # Other flat metadata filters (string-coerced, like LanceDBMemoryStore)
        if other_filters and not cls._matches_metadata_filters(
            note.metadata, other_filters
        ):
            return False

        return True

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
        other_filters = self._flat_other_filters(filters)
        results = []
        for note in self._store.values():
            if query.lower() not in note.content.lower():
                continue
            if filters and not self._matches_filters(note, filters, other_filters):
                continue
            results.append(note)
        return results[:k]

    def clear(self) -> None:
        self._store.clear()

    def list_all(self, filters: Optional[dict[str, Any]] = None) -> List[MemoryNote]:
        if filters:
            other_filters = self._flat_other_filters(filters)
            results = [
                note
                for note in self._store.values()
                if self._matches_filters(note, filters, other_filters)
            ]
        else:
            results = list(self._store.values())

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
