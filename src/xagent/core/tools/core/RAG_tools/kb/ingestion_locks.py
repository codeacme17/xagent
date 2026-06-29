"""Process-local locks for RAG ingestion write boundaries."""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import AsyncIterator, Iterator


_HELD_LOCK_COUNTS: ContextVar[dict[str, int]] = ContextVar(
    "rag_ingestion_held_lock_counts",
    default={},
)


class RAGIngestionLockRegistry:
    """Provide re-entrant ingestion locks keyed by logical RAG resources."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, tuple[threading.Lock, int]] = {}

    @staticmethod
    def collection_key(collection: str) -> str:
        normalized = str(collection or "").strip()
        return f"collection:{normalized or '<empty>'}"

    def _retain_lock(self, key: str) -> threading.Lock:
        with self._guard:
            entry = self._locks.get(key)
            if entry is None:
                lock = threading.Lock()
                self._locks[key] = (lock, 1)
                return lock

            lock, ref_count = entry
            self._locks[key] = (lock, ref_count + 1)
            return lock

    def _release_lock_entry(self, key: str) -> None:
        with self._guard:
            entry = self._locks.get(key)
            if entry is None:
                return

            lock, ref_count = entry
            if ref_count <= 1:
                self._locks.pop(key, None)
            else:
                self._locks[key] = (lock, ref_count - 1)

    @staticmethod
    def _enter_reentrant_context(key: str) -> object:
        held_counts = dict(_HELD_LOCK_COUNTS.get())
        held_counts[key] = held_counts.get(key, 0) + 1
        return _HELD_LOCK_COUNTS.set(held_counts)

    @staticmethod
    def _holds_lock(key: str) -> bool:
        return _HELD_LOCK_COUNTS.get().get(key, 0) > 0

    @contextmanager
    def guard_collection(self, collection: str) -> Iterator[str]:
        key = self.collection_key(collection)
        if self._holds_lock(key):
            token = self._enter_reentrant_context(key)
            try:
                yield key
            finally:
                _HELD_LOCK_COUNTS.reset(token)
            return

        lock = self._retain_lock(key)
        lock.acquire()
        token = self._enter_reentrant_context(key)
        try:
            yield key
        finally:
            _HELD_LOCK_COUNTS.reset(token)
            lock.release()
            self._release_lock_entry(key)

    @asynccontextmanager
    async def async_guard_collection(self, collection: str) -> AsyncIterator[str]:
        key = self.collection_key(collection)
        if self._holds_lock(key):
            token = self._enter_reentrant_context(key)
            try:
                yield key
            finally:
                _HELD_LOCK_COUNTS.reset(token)
            return

        lock = self._retain_lock(key)
        try:
            while not lock.acquire(blocking=False):
                await asyncio.sleep(0.01)
            token = self._enter_reentrant_context(key)
            try:
                yield key
            finally:
                _HELD_LOCK_COUNTS.reset(token)
                lock.release()
        finally:
            self._release_lock_entry(key)

    def reset_for_tests(self) -> None:
        with self._guard:
            self._locks.clear()


RAG_INGESTION_LOCKS = RAGIngestionLockRegistry()
