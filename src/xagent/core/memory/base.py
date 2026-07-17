from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, List, Optional

from .core import MemoryNote, MemoryResponse


def comparable_timestamp(value: Any) -> Any:
    """Normalize a datetime for cross-comparison in date-range filters.

    Stored note timestamps default to naive local time
    (``MemoryNote.timestamp``'s ``datetime.now`` factory), while caller-supplied
    ``date_from``/``date_to`` filter values may be timezone-aware (FastAPI
    parses ISO date query params with an offset into aware datetimes).
    Comparing the two directly raises ``TypeError``, which the stores' outer
    exception handlers would swallow into a silently empty result. Aware
    datetimes are therefore converted to naive local time before comparison;
    naive datetimes and non-datetime values pass through unchanged.

    Shared by every ``MemoryStore`` implementation's filter dispatch so the
    stores cannot drift apart on this policy.
    """
    if isinstance(value, datetime) and value.tzinfo is not None:
        return value.astimezone().replace(tzinfo=None)
    return value


class MemoryStore(ABC):
    """
    Abstract base class defining the interface for a memory storage backend.

    Any concrete implementation (e.g., in-memory store, ChromaDB, Redis, etc.)
    should implement all the following methods to manage MemoryNote objects.
    """

    @abstractmethod
    def add(self, note: "MemoryNote") -> "MemoryResponse":
        """
        Add a memory note to the store.

        Args:
            note (MemoryNote): The memory note to be added.

        Returns:
            MemoryResponse: Response indicating success and the note ID.
        """
        pass

    @abstractmethod
    def get(self, note_id: str) -> "MemoryResponse":
        """
        Retrieve a memory note by its ID.

        Args:
            note_id (str): The unique identifier of the memory note.

        Returns:
            MemoryResponse: Response containing the memory note or an error.
        """
        pass

    @abstractmethod
    def update(self, note: "MemoryNote") -> "MemoryResponse":
        """
        Update an existing memory note.

        Args:
            note (MemoryNote): The memory note with updated data.

        Returns:
            MemoryResponse: Response indicating success or failure.
        """
        pass

    @abstractmethod
    def delete(self, note_id: str) -> "MemoryResponse":
        """
        Delete a memory note by its ID.

        Args:
            note_id (str): The unique identifier of the memory note.

        Returns:
            MemoryResponse: Response indicating success or failure.
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> list["MemoryNote"]:
        """
        Search memory notes by query text with optional filters.

        Args:
            query (str): The query string to search for.
            k (int, optional): Number of top results to return. Defaults to 5.
            filters (Dict[str, Any], optional): Additional filter criteria. Defaults to None.

        Returns:
            List[MemoryNote]: List of matching memory notes.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all memory notes from the store.
        """
        pass

    @abstractmethod
    def list_all(self, filters: Optional[dict[str, Any]] = None) -> List["MemoryNote"]:
        """
        List all memory notes with optional filtering.

        Args:
            filters (Dict[str, Any], optional): Filter criteria like category, date range, etc.

        Returns:
            List[MemoryNote]: List of memory notes matching the filters.
        """
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the memory store.

        Returns:
            Dict[str, Any]: Statistics including total count, counts by category, etc.
        """
        pass
