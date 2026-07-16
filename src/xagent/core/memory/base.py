from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from .core import MemoryNote, MemoryResponse
from .scope_columns import encode_scope_dims, scope_dim_element


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

    def delete_by_scope_dimension(self, dim_key: str, value: Any) -> MemoryResponse:
        """
        Delete every note stamped with the execution-scope dimension
        ``dim_key=value``.

        Maintenance operation for reaping notes whose scope dimension no longer
        maps to a live principal (e.g. all memories of a revoked client
        application). Matching is exact per-dimension string equality — a note
        carrying ``client_application_id=42`` is never touched by a delete for
        ``client_application_id=4`` — and notes carrying no scope dimensions are
        never candidates. Idempotent: re-running deletes nothing further.

        This default walks ``list_all()`` and deletes note-by-note; backends
        with a native bulk predicate delete should override it.

        Returns:
            MemoryResponse: ``success`` plus ``metadata["deleted_count"]``.
        """
        element = scope_dim_element(dim_key, value)
        deleted = 0
        failures = 0
        for note in self.list_all():
            if note.id and element in encode_scope_dims(note.metadata):
                if self.delete(note.id).success:
                    deleted += 1
                else:
                    failures += 1
        if failures:
            return MemoryResponse(
                success=False,
                error=f"Failed to delete {failures} matching note(s)",
                metadata={"deleted_count": deleted},
            )
        return MemoryResponse(success=True, metadata={"deleted_count": deleted})

    def list_scope_dimension_values(self, dim_key: str) -> set[str]:
        """
        Distinct values stamped for one execution-scope dimension, store-wide.

        Lets a control plane reconcile a dimension against its own records —
        e.g. find the ``client_application_id`` values still present in memory
        whose rows have since been hard-deleted from the database, which no
        query on the database side can recover. Values are returned in their
        stamped string form.

        This default walks ``list_all()``; backends able to project the
        dimension column should override it. Raises on backend failure rather
        than returning a partial set, so a reconciler never mistakes an error
        for "no values".
        """
        prefix = f"{dim_key}="
        values: set[str] = set()
        for note in self.list_all():
            for element in encode_scope_dims(note.metadata):
                if element.startswith(prefix):
                    values.add(element[len(prefix) :])
        return values
