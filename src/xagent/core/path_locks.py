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
        self._locks: dict[str, threading.RLock] = {}

    @staticmethod
    def normalize_path(path: str | Path) -> Path:
        return Path(path).expanduser().resolve()

    def _lock_for_normalized_path(self, normalized_path: Path) -> threading.RLock:
        key = str(normalized_path)
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.RLock()
                self._locks[key] = lock
            return lock

    @contextmanager
    def guard_path(self, path: str | Path) -> Iterator[Path]:
        normalized_path = self.normalize_path(path)
        lock = self._lock_for_normalized_path(normalized_path)
        with lock:
            yield normalized_path

    @contextmanager
    def guard_paths(self, paths: list[str | Path]) -> Iterator[tuple[Path, ...]]:
        normalized_paths = tuple(self.normalize_path(path) for path in paths)
        unique_paths = {str(path): path for path in normalized_paths}
        ordered_paths = [unique_paths[key] for key in sorted(unique_paths)]

        with ExitStack() as stack:
            for normalized_path in ordered_paths:
                stack.enter_context(self._lock_for_normalized_path(normalized_path))
            yield normalized_paths


GLOBAL_PATH_MUTATION_LOCKS = PathMutationLockRegistry()
