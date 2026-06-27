from __future__ import annotations

import asyncio
from contextlib import contextmanager
import inspect

import pytest

from xagent.core.tools.adapters.vibe.browser_use import (
    BrowserPdfTool,
    BrowserScreenshotTool,
    BrowserTaskSessionMixin,
    create_browser_tools,
)
from xagent.core.workspace import TaskWorkspace


@pytest.mark.asyncio
async def test_browser_task_session_mixin_defaults_to_no_task() -> None:
    tool = BrowserTaskSessionMixin()

    assert tool._task_id is None

    await tool.setup(task_id=None)

    assert tool._task_id is None


@pytest.mark.asyncio
async def test_browser_tools_share_runtime_task_session_after_setup() -> None:
    tools = create_browser_tools(task_id="workspace-task")

    for tool in tools:
        setup = getattr(tool, "setup", None)
        if not callable(setup):
            continue
        result = setup(task_id="runtime-task")
        if inspect.isawaitable(result):
            await result

    session_tools = [
        tool
        for tool in tools
        if getattr(tool, "name", "") != "browser_list_sessions"
        and hasattr(tool, "_task_id")
    ]

    assert session_tools
    assert {tool._task_id for tool in session_tools} == {"runtime-task"}


def test_browser_task_session_mixin_defaults_to_step_scoped_session() -> None:
    tool = BrowserTaskSessionMixin()
    tool._task_id = "task-412"

    args = tool._with_default_session(
        {"url": "poster.html", "_xagent_step_id": "render english"}
    )

    assert args["session_id"] == "task-412:render_english"
    assert "_xagent_step_id" not in args


def test_browser_task_session_mixin_keeps_explicit_session() -> None:
    tool = BrowserTaskSessionMixin()
    tool._task_id = "task-412"

    args = tool._with_default_session(
        {
            "url": "poster.html",
            "session_id": "custom-session",
            "_xagent_step_id": "render_english",
        }
    )

    assert args["session_id"] == "custom-session"
    assert "_xagent_step_id" not in args


def test_browser_session_tools_expose_concurrency_safe_metadata() -> None:
    tools = create_browser_tools(task_id="task-412")
    safe_tool_names = {
        "browser_navigate",
        "browser_click",
        "browser_fill",
        "browser_screenshot",
        "browser_extract_text",
        "browser_pdf",
        "browser_evaluate",
        "browser_select_option",
        "browser_wait_for_selector",
    }

    metadata_by_name = {tool.name: tool.metadata for tool in tools}

    assert {
        name for name, metadata in metadata_by_name.items() if metadata.concurrency_safe
    } >= safe_tool_names
    assert metadata_by_name["browser_close"].concurrency_safe is False
    assert metadata_by_name["browser_list_sessions"].concurrency_safe is False


@pytest.mark.asyncio
async def test_browser_task_session_teardown_closes_derived_task_sessions(
    monkeypatch,
) -> None:
    closed_session_ids = []
    manager = _FakeBrowserManager(
        {
            "task-412": _ClosableBrowserSession("task-412", closed_session_ids),
            "task-412:render_english": _ClosableBrowserSession(
                "task-412:render_english", closed_session_ids
            ),
            "task-412-preview": _ClosableBrowserSession(
                "task-412-preview", closed_session_ids
            ),
            "other-task": _ClosableBrowserSession("other-task", closed_session_ids),
        }
    )
    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.browser_use.get_browser_manager",
        lambda: manager,
    )
    tool = BrowserTaskSessionMixin()
    tool._task_id = "task-412"

    await tool.teardown()

    assert closed_session_ids == ["task-412", "task-412:render_english"]
    assert set(manager._sessions) == {"task-412-preview", "other-task"}


@pytest.mark.asyncio
async def test_browser_screenshot_returns_registered_file_ref(
    tmp_path, monkeypatch
) -> None:
    def mock_create_record(self, file_id, file_path, db_session=None):
        path_str = str(file_path)
        resolved_str = str(file_path.resolve())
        self._recently_registered_files[path_str] = file_id
        self._recently_registered_files[resolved_str] = file_id
        self._file_id_to_path[file_id] = file_path

    monkeypatch.setattr(TaskWorkspace, "_create_file_record", mock_create_record)

    async def fake_browser_screenshot(**kwargs):
        return {
            "success": True,
            "session_id": kwargs["session_id"],
            "screenshot": (
                "data:image/png;base64,"
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA"
                "DElEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            ),
            "format": "png",
            "full_page": True,
            "wait_for_lazy_load": False,
            "message": "ok",
            "error": "",
        }

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.browser_use.browser_screenshot",
        fake_browser_screenshot,
    )

    workspace = TaskWorkspace("test_task", str(tmp_path))
    tool = BrowserScreenshotTool(task_id="task-412", workspace=workspace)

    result = await tool.run_json_async(
        {
            "full_page": True,
            "output_filename": "poster_en.png",
            "_xagent_step_id": "render_english",
        }
    )

    assert result["success"] is True
    assert result["screenshot"] == "output/poster_en.png"
    assert result["file_id"]
    assert result["file_ref"]["file_id"] == result["file_id"]
    assert result["file_ref"]["relative_path"] == "output/poster_en.png"
    assert result["markdown_link"] == (f"[poster_en.png](file:{result['file_id']})")


@pytest.mark.asyncio
async def test_browser_screenshot_save_uses_workspace_path_guard(
    tmp_path, monkeypatch
) -> None:
    def mock_create_record(self, file_id, file_path, db_session=None):
        self._file_id_to_path[file_id] = file_path

    monkeypatch.setattr(TaskWorkspace, "_create_file_record", mock_create_record)

    async def fake_browser_screenshot(**kwargs):
        return {
            "success": True,
            "session_id": kwargs["session_id"],
            "screenshot": (
                "data:image/png;base64,"
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA"
                "DElEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            ),
            "format": "png",
            "full_page": False,
            "wait_for_lazy_load": False,
            "message": "ok",
            "error": "",
        }

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.browser_use.browser_screenshot",
        fake_browser_screenshot,
    )

    workspace = TaskWorkspace("test_task", str(tmp_path))
    guard_calls = []
    original_guard = workspace.guard_workspace_mutation_path

    @contextmanager
    def tracked_guard(file_path, default_dir="output"):
        guard_calls.append(file_path)
        with original_guard(file_path, default_dir=default_dir) as guarded_path:
            yield guarded_path

    workspace.guard_workspace_mutation_path = tracked_guard
    tool = BrowserScreenshotTool(task_id="task-412", workspace=workspace)

    result = await tool.run_json_async(
        {
            "output_filename": "poster_en.png",
            "_xagent_step_id": "render_english",
        }
    )

    assert result["success"] is True
    assert guard_calls == [workspace.output_dir / "poster_en.png"]


@pytest.mark.asyncio
async def test_browser_pdf_save_uses_workspace_path_guard(
    tmp_path, monkeypatch
) -> None:
    def mock_create_record(self, file_id, file_path, db_session=None):
        self._file_id_to_path[file_id] = file_path

    monkeypatch.setattr(TaskWorkspace, "_create_file_record", mock_create_record)

    async def fake_browser_pdf(**kwargs):
        return {
            "success": True,
            "session_id": kwargs["session_id"],
            "pdf": "JVBERi0xLjQK",
            "format": "base64",
            "size": 8,
            "message": "ok",
            "error": "",
        }

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.browser_use.browser_pdf",
        fake_browser_pdf,
    )

    workspace = TaskWorkspace("test_task", str(tmp_path))
    guard_calls = []
    original_guard = workspace.guard_workspace_mutation_path

    @contextmanager
    def tracked_guard(file_path, default_dir="output"):
        guard_calls.append(file_path)
        with original_guard(file_path, default_dir=default_dir) as guarded_path:
            yield guarded_path

    workspace.guard_workspace_mutation_path = tracked_guard
    tool = BrowserPdfTool(task_id="task-412", workspace=workspace)

    result = await tool.run_json_async(
        {
            "output_filename": "page.pdf",
            "_xagent_step_id": "render_english",
        }
    )

    assert result["success"] is True
    assert guard_calls == [workspace.output_dir / "page.pdf"]


@pytest.mark.asyncio
async def test_browser_screenshot_auto_filenames_are_unique(
    tmp_path, monkeypatch
) -> None:
    def mock_create_record(self, file_id, file_path, db_session=None):
        self._file_id_to_path[file_id] = file_path

    monkeypatch.setattr(TaskWorkspace, "_create_file_record", mock_create_record)

    async def fake_browser_screenshot(**kwargs):
        return {
            "success": True,
            "session_id": kwargs["session_id"],
            "screenshot": (
                "data:image/png;base64,"
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA"
                "DElEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            ),
            "format": "png",
            "full_page": False,
            "wait_for_lazy_load": False,
            "message": "ok",
            "error": "",
        }

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.browser_use.browser_screenshot",
        fake_browser_screenshot,
    )

    workspace = TaskWorkspace("test_task", str(tmp_path))
    tool = BrowserScreenshotTool(task_id="task-412", workspace=workspace)

    first, second = await asyncio.gather(
        tool.run_json_async({}), tool.run_json_async({})
    )

    assert first["success"] is True
    assert second["success"] is True
    assert first["screenshot"] != second["screenshot"]


@pytest.mark.asyncio
async def test_browser_pdf_auto_filenames_are_unique(tmp_path, monkeypatch) -> None:
    def mock_create_record(self, file_id, file_path, db_session=None):
        self._file_id_to_path[file_id] = file_path

    monkeypatch.setattr(TaskWorkspace, "_create_file_record", mock_create_record)

    async def fake_browser_pdf(**kwargs):
        return {
            "success": True,
            "session_id": kwargs["session_id"],
            "pdf": "JVBERi0xLjQK",
            "format": "base64",
            "size": 8,
            "message": "ok",
            "error": "",
        }

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.browser_use.browser_pdf",
        fake_browser_pdf,
    )

    workspace = TaskWorkspace("test_task", str(tmp_path))
    tool = BrowserPdfTool(task_id="task-412", workspace=workspace)

    first, second = await asyncio.gather(
        tool.run_json_async({}), tool.run_json_async({})
    )

    assert first["success"] is True
    assert second["success"] is True
    assert first["output_path"] != second["output_path"]


class _ClosableBrowserSession:
    def __init__(self, session_id, closed_session_ids):
        self._session_id = session_id
        self._closed_session_ids = closed_session_ids

    async def close(self) -> None:
        self._closed_session_ids.append(self._session_id)


class _FakeBrowserManager:
    def __init__(self, sessions):
        import asyncio

        self._lock = asyncio.Lock()
        self._sessions = sessions
