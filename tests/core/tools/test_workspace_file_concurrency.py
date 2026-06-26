from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Lock
from typing import Iterator

import pytest

from xagent.core.tools.core.workspace_file_tool import WorkspaceFileOperations
from xagent.core.workspace import TaskWorkspace


class _RegisterTracker:
    def __init__(self) -> None:
        self._lock = Lock()
        self.active = 0
        self.peak = 0

    @contextmanager
    def auto_register_files(self) -> Iterator[TaskWorkspace]:
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(0.05)
            yield self.workspace
        finally:
            with self._lock:
                self.active -= 1

    def bind(self, workspace: TaskWorkspace) -> "_RegisterTracker":
        self.workspace = workspace
        return self


class _CallTracker:
    def __init__(self) -> None:
        self._lock = Lock()
        self.active = 0
        self.peak = 0

    def enter(self) -> None:
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)

    def leave(self) -> None:
        with self._lock:
            self.active -= 1


def _install_register_tracker(workspace: TaskWorkspace) -> _RegisterTracker:
    tracker = _RegisterTracker().bind(workspace)
    workspace.auto_register_files = tracker.auto_register_files  # type: ignore[method-assign]
    return tracker


def test_same_normalized_workspace_write_path_serializes_registration(
    tmp_path,
) -> None:
    workspace = TaskWorkspace("same_path_serial", str(tmp_path))
    ops = WorkspaceFileOperations(workspace)
    tracker = _install_register_tracker(workspace)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(ops.write_file, "same.txt", "first"),
            pool.submit(ops.write_file, "output/same.txt", "second"),
        ]
        results = [future.result() for future in futures]

    assert tracker.peak == 1
    assert all(result["success"] for result in results)
    assert (workspace.output_dir / "same.txt").exists()


def test_different_workspace_write_paths_can_overlap_registration(tmp_path) -> None:
    workspace = TaskWorkspace("different_path_overlap", str(tmp_path))
    ops = WorkspaceFileOperations(workspace)
    tracker = _install_register_tracker(workspace)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(ops.write_file, "one.txt", "one"),
            pool.submit(ops.write_file, "two.txt", "two"),
        ]
        results = [future.result() for future in futures]

    assert tracker.peak == 2
    assert all(result["success"] for result in results)
    assert (workspace.output_dir / "one.txt").read_text(encoding="utf-8") == "one"
    assert (workspace.output_dir / "two.txt").read_text(encoding="utf-8") == "two"


def test_concurrent_workspace_writes_return_valid_file_refs(tmp_path) -> None:
    workspace = TaskWorkspace("file_refs", str(tmp_path))
    ops = WorkspaceFileOperations(workspace)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(ops.write_file, "reports/one.txt", "one"),
            pool.submit(ops.write_file, "reports/two.txt", "two"),
        ]
        results = [future.result() for future in futures]

    for result in results:
        assert result["success"] is True
        assert result["file_id"]
        assert result["preview_url"].endswith(result["file_id"])
        assert result["download_url"].endswith(result["file_id"])
        assert result["markdown_link"] == (
            f"[{result['filename']}](file:{result['file_id']})"
        )
        assert result["file_ref"]["file_id"] == result["file_id"]


def test_workspace_mutation_rejects_paths_outside_workspace(tmp_path) -> None:
    workspace = TaskWorkspace("outside_boundary", str(tmp_path))
    ops = WorkspaceFileOperations(workspace)

    with pytest.raises(ValueError, match="outside"):
        ops.write_file("../../outside.txt", "outside")

    assert not (tmp_path / "outside.txt").exists()


def test_prepare_html_asset_serializes_unique_names_in_same_asset_dir(
    tmp_path,
) -> None:
    workspace = TaskWorkspace("html_asset_names", str(tmp_path))
    ops = WorkspaceFileOperations(workspace)
    source = ops.write_file("input/logo.png", "fake image")

    existing_asset = workspace.output_dir / "assets" / "logo.png"
    existing_asset.parent.mkdir(parents=True, exist_ok=True)
    existing_asset.write_text("existing", encoding="utf-8")

    tracker = _CallTracker()
    original_build_unique_asset_path = ops._build_unique_asset_path

    def tracked_build_unique_asset_path(path):
        tracker.enter()
        try:
            time.sleep(0.05)
            return original_build_unique_asset_path(path)
        finally:
            tracker.leave()

    ops._build_unique_asset_path = tracked_build_unique_asset_path  # type: ignore[method-assign]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                ops.prepare_html_asset,
                source["file_id"],
                "index.html",
                "logo.png",
            ),
            pool.submit(
                ops.prepare_html_asset,
                source["file_id"],
                "index.html",
                "logo_1.png",
            ),
        ]
        results = [future.result() for future in futures]

    assert tracker.peak == 1
    assert len({result["html_src"] for result in results}) == 2
