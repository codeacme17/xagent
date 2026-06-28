"""Shared normalized-path mutation locks."""

from __future__ import annotations

import threading
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator


class PathMutationLockRegistry:
    """Provide re-entrant locks keyed by normalized absolute filesystem path."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, tuple[threading.RLock, int]] = {}

    @staticmethod
    def normalize_path(path: str | Path) -> Path:
        return Path(path).expanduser().resolve()

    def _acquire_lock_for_key(self, key: str) -> threading.RLock:
        with self._guard:
            lock_entry = self._locks.get(key)
            if lock_entry is None:
                lock = threading.RLock()
                self._locks[key] = (lock, 1)
                return lock

            lock, ref_count = lock_entry
            self._locks[key] = (lock, ref_count + 1)
            return lock

    def _release_lock_for_key(self, key: str) -> None:
        with self._guard:
            lock_entry = self._locks.get(key)
            if lock_entry is None:
                return

            lock, ref_count = lock_entry
            if ref_count <= 1:
                self._locks.pop(key, None)
            else:
                self._locks[key] = (lock, ref_count - 1)

    @contextmanager
    def guard_path(self, path: str | Path) -> Iterator[Path]:
        normalized_path = self.normalize_path(path)
        key = str(normalized_path)
        lock = self._acquire_lock_for_key(key)
        try:
            with lock:
                yield normalized_path
        finally:
            self._release_lock_for_key(key)

    @contextmanager
    def guard_paths(self, paths: list[str | Path]) -> Iterator[tuple[Path, ...]]:
        normalized_paths = tuple(self.normalize_path(path) for path in paths)
        unique_paths = {str(path): path for path in normalized_paths}
        ordered_paths = [unique_paths[key] for key in sorted(unique_paths)]

        with ExitStack() as stack:
            for normalized_path in ordered_paths:
                stack.enter_context(self.guard_path(normalized_path))
            yield normalized_paths


GLOBAL_PATH_MUTATION_LOCKS = PathMutationLockRegistry()
