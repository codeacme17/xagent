"""Inc.1 — tool concurrency-safety metadata (design §3).

Pins the contract that drives the ReAct concurrent scheduler:
- ``ToolMetadata`` carries ``read_only`` / ``concurrency_safe`` (default False).
- ``read_only=True`` implies ``concurrency_safe=True``.
- A tool may opt into ``concurrency_safe`` without being ``read_only``.
- The fields flow through the tool wrappers unchanged.
- The v1 minimal safe set (read-only web/file tools) is concurrency-safe while
  writing / process-stateful tools are not.

``calculator`` from the original design table is intentionally NOT sampled here:
the core ``calculator`` function is not wired into the vibe tool layer in this
codebase, so there is no tool object to classify. The realized v1 safe set is
web search + read-only web fetch + read-only file tools.
"""

from __future__ import annotations

from typing import Any, Mapping, Type
from unittest.mock import MagicMock

from pydantic import BaseModel

from tests.core.tools.adapters.sandboxed_tool.conftest import FakeBaseTool
from xagent.core.tools.adapters.vibe.base import ToolMetadata
from xagent.core.tools.adapters.vibe.document_parser import (
    DocumentParseTool,
    DocumentParseWithOutputTool,
)
from xagent.core.tools.adapters.vibe.fetch_web_content import FetchWebContentTool
from xagent.core.tools.adapters.vibe.file_tool import (
    append_file_tool,
    create_directory_tool,
    delete_file_tool,
    edit_file_tool,
    find_and_replace_tool,
    read_file_tool,
    write_csv_file_tool,
    write_file_tool,
    write_json_file_tool,
)
from xagent.core.tools.adapters.vibe.function import FunctionTool
from xagent.core.tools.adapters.vibe.image_tool import ImageGenerationTool
from xagent.core.tools.adapters.vibe.output_filter_wrapper import (
    OutputFilteredToolWrapper,
)
from xagent.core.tools.adapters.vibe.python_executor import PythonExecutorTool
from xagent.core.tools.adapters.vibe.python_executor import PythonExecutorToolForBasic
from xagent.core.tools.adapters.vibe.python_executor import get_python_executor_tool
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandbox_config import (
    sandbox_config,
)
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
    SandboxedToolWrapper,
)
from xagent.core.tools.adapters.vibe.web_search import WebSearchTool
from xagent.core.tools.adapters.vibe.workspace_file_tool import WorkspaceFileTools
from xagent.core.workspace import TaskWorkspace


def _noop(x: int = 0) -> dict[str, Any]:
    """A trivial function for FunctionTool-based tests."""
    return {"x": x}


def _image_model() -> MagicMock:
    model = MagicMock()
    model.has_ability.side_effect = lambda ability: ability in {"generate", "edit"}
    return model


def test_tool_metadata_defaults_are_not_concurrency_safe() -> None:
    metadata = ToolMetadata(name="anything")
    assert metadata.read_only is False
    assert metadata.concurrency_safe is False


def test_read_only_implies_concurrency_safe_on_function_tool() -> None:
    tool = FunctionTool(_noop, name="ro", read_only=True)
    assert tool.metadata.read_only is True
    assert tool.metadata.concurrency_safe is True


def test_read_only_implies_concurrency_safe_on_class_attribute() -> None:
    class ReadOnlyTool(FakeBaseTool):
        read_only = True

        @property
        def name(self) -> str:
            return "class_attr_ro"

    assert ReadOnlyTool().metadata.concurrency_safe is True


def test_concurrency_safe_can_be_declared_without_read_only() -> None:
    tool = FunctionTool(_noop, name="cs", concurrency_safe=True)
    assert tool.metadata.read_only is False
    assert tool.metadata.concurrency_safe is True


def test_default_function_tool_is_not_concurrency_safe() -> None:
    tool = FunctionTool(_noop, name="plain")
    assert tool.metadata.read_only is False
    assert tool.metadata.concurrency_safe is False


def test_output_filter_wrapper_passthrough() -> None:
    safe = FunctionTool(_noop, name="ro", read_only=True)
    unsafe = FunctionTool(_noop, name="plain")

    wrapped_safe = OutputFilteredToolWrapper(safe, 1000, 50, 5)
    wrapped_unsafe = OutputFilteredToolWrapper(unsafe, 1000, 50, 5)

    assert wrapped_safe.metadata.concurrency_safe is True
    assert wrapped_unsafe.metadata.concurrency_safe is False


def test_sandboxed_wrapper_passthrough() -> None:
    @sandbox_config()
    class SandboxReadOnlyTool(FakeBaseTool):
        read_only = True

        def args_type(self) -> Type[BaseModel]:
            return BaseModel

        def run_json_sync(self, args: Mapping[str, Any]) -> Any:
            return {}

        async def run_json_async(self, args: Mapping[str, Any]) -> Any:
            return {}

        @property
        def name(self) -> str:
            return "sandbox_ro"

    # The sandbox is irrelevant to metadata delegation; a stub suffices.
    wrapped = SandboxedToolWrapper(SandboxReadOnlyTool(), MagicMock())
    assert wrapped.metadata.concurrency_safe is True


def test_builtin_read_only_tools_are_concurrency_safe() -> None:
    assert WebSearchTool().metadata.concurrency_safe is True
    assert FetchWebContentTool().metadata.concurrency_safe is True
    assert read_file_tool.metadata.concurrency_safe is True


def test_basic_file_mutation_tools_match_guarded_concurrency() -> None:
    for tool in {
        write_file_tool,
        append_file_tool,
        delete_file_tool,
        create_directory_tool,
        write_json_file_tool,
        write_csv_file_tool,
        edit_file_tool,
        find_and_replace_tool,
    }:
        assert tool.metadata.read_only is False
        assert tool.metadata.concurrency_safe is True


def test_python_executor_tools_are_concurrency_safe_after_call_isolation() -> None:
    assert PythonExecutorTool().metadata.concurrency_safe is True
    assert PythonExecutorToolForBasic().metadata.concurrency_safe is True
    assert get_python_executor_tool().metadata.concurrency_safe is True


def test_workspace_file_tool_metadata_matches_guarded_concurrency(tmp_path) -> None:
    workspace = TaskWorkspace("metadata_workspace_files", str(tmp_path))
    tools = {tool.name: tool for tool in WorkspaceFileTools(workspace).get_tools()}

    for name in {
        "read_file",
        "list_files",
        "file_exists",
        "get_file_info",
        "read_json_file",
        "read_csv_file",
        "get_workspace_output_files",
        "list_all_user_files",
    }:
        assert tools[name].metadata.read_only is True
        assert tools[name].metadata.concurrency_safe is True

    for name in {
        "write_file",
        "prepare_html_asset",
        "append_file",
        "delete_file",
        "create_directory",
        "write_json_file",
        "write_csv_file",
        "edit_file",
        "find_and_replace",
    }:
        assert tools[name].metadata.read_only is False
        assert tools[name].metadata.concurrency_safe is True


def test_image_tool_metadata_matches_artifact_safety(tmp_path) -> None:
    workspace = TaskWorkspace("metadata_image_artifacts", str(tmp_path))
    image_tool = ImageGenerationTool({"model": _image_model()}, workspace=workspace)
    tools = {tool.name: tool for tool in image_tool.get_tools()}

    assert tools["list_image_models"].metadata.read_only is True
    assert tools["list_image_models"].metadata.concurrency_safe is True

    for name in {"generate_image", "edit_image"}:
        assert tools[name].metadata.read_only is False
        assert tools[name].metadata.concurrency_safe is True


def test_document_parser_metadata_matches_artifact_safety() -> None:
    parse_tool = DocumentParseTool()
    parse_with_output_tool = DocumentParseWithOutputTool()

    assert parse_tool.metadata.read_only is True
    assert parse_tool.metadata.concurrency_safe is True
    assert parse_with_output_tool.metadata.read_only is False
    assert parse_with_output_tool.metadata.concurrency_safe is True


def test_abstract_base_tool_metadata_exposes_concurrency_fields() -> None:
    # The metadata object must always carry the fields (even when False) so the
    # scheduler can read them uniformly across every tool.
    assert "read_only" in PythonExecutorTool().metadata.model_dump()
    assert "concurrency_safe" in PythonExecutorTool().metadata.model_dump()
