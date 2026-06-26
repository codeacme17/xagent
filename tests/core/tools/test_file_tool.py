import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pytest

from xagent.core.tools.core import file_tool as file_tool_module
from xagent.core.tools.core import workspace_file_tool as workspace_file_tool_module
from xagent.core.tools.core.workspace_file_tool import WorkspaceFileOperations
from xagent.core.tools.adapters.vibe.file_tool import (
    FILE_TOOLS,
    append_file,
    create_directory,
    delete_file,
    edit_file,
    file_exists,
    find_and_replace,
    get_file_info,
    list_files,
    read_csv_file,
    read_file,
    read_json_file,
    write_csv_file,
    write_file,
    write_json_file,
)
from xagent.core.workspace import TaskWorkspace


class _ConcurrencyTracker:
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


class _TrackedFile:
    def __init__(self, handle, tracker: _ConcurrencyTracker) -> None:
        self._handle = handle
        self._tracker = tracker

    def __enter__(self):
        entered = self._handle.__enter__()
        self._tracker.enter()
        time.sleep(0.05)
        return entered

    def __exit__(self, *args):
        try:
            return self._handle.__exit__(*args)
        finally:
            self._tracker.leave()

    def __getattr__(self, name):
        return getattr(self._handle, name)


def _install_tracked_open(
    monkeypatch: pytest.MonkeyPatch,
    tracker: _ConcurrencyTracker,
    module=file_tool_module,
) -> None:
    real_open = open

    def tracked_open(*args, **kwargs):
        return _TrackedFile(real_open(*args, **kwargs), tracker)

    monkeypatch.setattr(module, "open", tracked_open, raising=False)


def _run_concurrently(*calls) -> None:
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        futures = [pool.submit(call) for call in calls]
        for future in futures:
            future.result()


def test_basic_file_operations():
    """Test basic file operations."""
    # Create a temporary file.
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        temp_file = f.name
        f.write("Hello, World!")

    try:
        # Read file content.
        content = read_file(temp_file)
        assert content == "Hello, World!"

        # Check file existence.
        assert file_exists(temp_file)

        # Get file metadata.
        info = get_file_info(temp_file)
        assert info.name.endswith(".txt")
        assert info.is_file
        assert not info.is_dir

        # Write file content.
        write_file(temp_file, "New content")
        content = read_file(temp_file)
        assert content == "New content"

        # Append file content.
        append_file(temp_file, " Appended content")
        content = read_file(temp_file)
        assert content == "New content Appended content"

    finally:
        # Clean up the temporary file.
        if os.path.exists(temp_file):
            delete_file(temp_file)


def test_read_file_line_range():
    """Test reading a specific line range."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        temp_file = f.name
        f.write("line 1\nline 2\nline 3\nline 4\n")

    try:
        content = read_file(temp_file, start_line=2, end_line=3)
        assert content == "line 2\nline 3\n"

        with pytest.raises(ValueError, match="start_line must be >= 1"):
            read_file(temp_file, start_line=0)
        with pytest.raises(ValueError, match="end_line must be >= 1"):
            read_file(temp_file, end_line=0)
        with pytest.raises(ValueError, match="start_line must be <= end_line"):
            read_file(temp_file, start_line=3, end_line=2)
    finally:
        if os.path.exists(temp_file):
            delete_file(temp_file)


def test_directory_operations():
    """Test directory operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a directory.
        new_dir = os.path.join(temp_dir, "test_dir", "sub_dir")
        create_directory(new_dir)
        assert os.path.exists(new_dir)

        # List files.
        files = list_files(temp_dir)
        assert files.total_count > 0
        assert any(f.name == "test_dir" for f in files.files)

        # List files recursively.
        recursive_files = list_files(temp_dir, recursive=True)
        assert recursive_files.total_count >= 2  # Includes test_dir and sub_dir.


def test_json_operations():
    """Test JSON file operations."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        temp_file = f.name

    try:
        # Write JSON data.
        test_data = {"name": "Test", "age": 25, "hobbies": ["reading", "coding"]}
        write_json_file(temp_file, test_data)

        # Read JSON data.
        loaded_data = read_json_file(temp_file)
        assert loaded_data == test_data

    finally:
        if os.path.exists(temp_file):
            delete_file(temp_file)


def test_csv_operations():
    """Test CSV file operations."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        temp_file = f.name

    try:
        # Write CSV data.
        test_data = [
            {"name": "Alice", "age": "25", "city": "New York"},
            {"name": "Bob", "age": "30", "city": "San Francisco"},
        ]
        write_csv_file(temp_file, test_data)

        # Read CSV data.
        loaded_data = read_csv_file(temp_file)
        assert len(loaded_data) == 2
        assert loaded_data[0]["name"] == "Alice"
        assert loaded_data[1]["name"] == "Bob"

    finally:
        if os.path.exists(temp_file):
            delete_file(temp_file)


def test_error_handling():
    """Test error handling."""
    # Reading a missing file raises FileNotFoundError.
    with pytest.raises(FileNotFoundError):
        read_file("/non/existent/file.txt")

    # A missing file does not exist.
    assert not file_exists("/non/existent/file.txt")

    # Getting metadata for a missing file raises FileNotFoundError.
    with pytest.raises(FileNotFoundError):
        get_file_info("/non/existent/file.txt")


def test_file_tools_integration():
    """Test FileTool integration."""
    # Verify all tools can be imported and instantiated.
    assert len(FILE_TOOLS) == 14

    # Verify each tool has the expected attributes.
    for tool in FILE_TOOLS:
        assert hasattr(tool, "metadata")
        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert callable(tool.run_json_sync)
        assert callable(tool.run_json_async)

    # Verify tool names are unique.
    tool_names = [tool.name for tool in FILE_TOOLS]
    assert len(tool_names) == len(set(tool_names)), "Tool names should be unique"

    # Verify edit tools are present.
    assert "edit_file" in tool_names
    assert "find_and_replace" in tool_names


def test_specific_tool_functionality():
    """Test specific tool functionality."""
    # Test read_file_tool.
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        temp_file = f.name
        f.write("Test content")

    try:
        # Test through the tool instance.
        read_tool = next(t for t in FILE_TOOLS if t.name == "read_file")
        result = read_tool.run_json_sync({"file_path": temp_file})
        assert "Test content" in str(result)

        # Test write_file_tool.
        write_tool = next(t for t in FILE_TOOLS if t.name == "write_file")
        write_tool.run_json_sync({"file_path": temp_file, "content": "New content"})

        # Verify write succeeded.
        result = read_tool.run_json_sync({"file_path": temp_file})
        assert "New content" in str(result)

    finally:
        if os.path.exists(temp_file):
            delete_file(temp_file)


def test_image_file_info():
    """Test image file information retrieval"""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("PIL not installed")

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".png") as f:
        temp_file = f.name

    try:
        # Create a test image (800x600, RGB)
        test_image = Image.new("RGB", (800, 600), color="red")
        test_image.save(temp_file)

        # Get file information
        info = get_file_info(temp_file)

        # Verify basic information
        assert info.name.endswith(".png")
        assert info.is_file
        assert not info.is_dir
        assert info.size > 0

        # Verify image metadata
        assert info.image_width == 800, f"Expected width 800, got {info.image_width}"
        assert info.image_height == 600, f"Expected height 600, got {info.image_height}"
        assert info.image_format == "PNG", (
            f"Expected format PNG, got {info.image_format}"
        )
        assert info.image_mode == "RGB", f"Expected mode RGB, got {info.image_mode}"

    finally:
        if os.path.exists(temp_file):
            delete_file(temp_file)


def test_non_image_file_info():
    """Test non-image file information retrieval (should not include image metadata)"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        temp_file = f.name
        f.write("Test content")

    try:
        # Get file information
        info = get_file_info(temp_file)

        # Verify basic information
        assert info.name.endswith(".txt")
        assert info.is_file
        assert not info.is_dir

        # Verify image metadata is None
        assert info.image_width is None
        assert info.image_height is None
        assert info.image_format is None
        assert info.image_mode is None

    finally:
        if os.path.exists(temp_file):
            delete_file(temp_file)


@pytest.mark.parametrize(
    "case_name",
    [
        "write_file",
        "append_file",
        "write_json_file",
        "write_csv_file",
        "edit_file",
        "find_and_replace",
        "delete_file",
        "create_directory",
    ],
)
def test_same_normalized_basic_file_mutations_are_serialized(
    monkeypatch: pytest.MonkeyPatch, tmp_path, case_name: str
) -> None:
    tracker = _ConcurrencyTracker()
    path = tmp_path / "same.txt"
    same_path_alias = tmp_path / "." / "same.txt"

    if case_name in {"delete_file", "create_directory"}:

        def tracked_call(*args, **kwargs):
            tracker.enter()
            try:
                time.sleep(0.05)
            finally:
                tracker.leave()
            return None

        if case_name == "delete_file":
            monkeypatch.setattr(file_tool_module.os, "remove", tracked_call)
            calls = [
                lambda: delete_file(str(path)),
                lambda: delete_file(str(same_path_alias)),
            ]
        else:
            monkeypatch.setattr(file_tool_module.os, "makedirs", tracked_call)
            directory_path = tmp_path / "same_dir"
            same_directory_alias = tmp_path / "." / "same_dir"
            calls = [
                lambda: create_directory(str(directory_path)),
                lambda: create_directory(str(same_directory_alias)),
            ]
    else:
        _install_tracked_open(monkeypatch, tracker)

        if case_name in {"edit_file", "find_and_replace"}:
            path.write_text("needle\n", encoding="utf-8")

        if case_name == "write_file":
            calls = [
                lambda: write_file(str(path), "first"),
                lambda: write_file(str(same_path_alias), "second"),
            ]
        elif case_name == "append_file":
            calls = [
                lambda: append_file(str(path), "first"),
                lambda: append_file(str(same_path_alias), "second"),
            ]
        elif case_name == "write_json_file":
            calls = [
                lambda: write_json_file(str(path), {"value": "first"}),
                lambda: write_json_file(str(same_path_alias), {"value": "second"}),
            ]
        elif case_name == "write_csv_file":
            calls = [
                lambda: write_csv_file(str(path), [{"value": "first"}]),
                lambda: write_csv_file(str(same_path_alias), [{"value": "second"}]),
            ]
        elif case_name == "edit_file":
            operations = [
                {
                    "operation_type": "replace",
                    "line_number": 1,
                    "content": "needle",
                }
            ]
            calls = [
                lambda: edit_file(str(path), operations),
                lambda: edit_file(str(same_path_alias), operations),
            ]
        else:
            calls = [
                lambda: find_and_replace(
                    str(path), "needle", "needle", use_regex=False
                ),
                lambda: find_and_replace(
                    str(same_path_alias), "needle", "needle", use_regex=False
                ),
            ]

    _run_concurrently(*calls)

    assert tracker.peak == 1


def test_different_basic_file_mutation_paths_can_overlap(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    tracker = _ConcurrencyTracker()
    _install_tracked_open(monkeypatch, tracker)

    _run_concurrently(
        lambda: write_file(str(tmp_path / "one.txt"), "one"),
        lambda: write_file(str(tmp_path / "two.txt"), "two"),
    )

    assert tracker.peak == 2


def test_basic_and_workspace_file_mutations_share_normalized_path_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    tracker = _ConcurrencyTracker()
    _install_tracked_open(monkeypatch, tracker, file_tool_module)
    _install_tracked_open(monkeypatch, tracker, workspace_file_tool_module)

    workspace = TaskWorkspace("shared_basic_workspace_lock", str(tmp_path))
    ops = WorkspaceFileOperations(workspace)
    target_path = workspace.output_dir / "same.txt"

    _run_concurrently(
        lambda: ops.write_file("same.txt", "workspace"),
        lambda: write_file(str(target_path), "basic"),
    )

    assert tracker.peak == 1


def test_image_file_info_without_pil():
    """Test that get_file_info handles PIL unavailability gracefully."""
    from unittest.mock import patch

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".png") as f:
        temp_file = f.name
        f.write(b"fake png data")

    try:
        with patch("xagent.core.tools.core.file_tool.PIL_AVAILABLE", False):
            info = get_file_info(temp_file)

            assert info.is_file
            # When PIL is not available, image metadata should all be None
            assert info.image_width is None
            assert info.image_height is None
            assert info.image_format is None
            assert info.image_mode is None

    finally:
        if os.path.exists(temp_file):
            delete_file(temp_file)
