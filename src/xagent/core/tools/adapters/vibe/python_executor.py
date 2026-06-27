"""
Python Code Execution Tool for xagent
Framework wrapper around the pure Python executor tool
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Type

from pydantic import BaseModel, Field

from ....workspace import TaskWorkspace
from ...artifacts import (
    build_generated_file_metadata,
)
from ...core.python_executor import _INTERNAL_WRITTEN_FILES_KEY, PythonExecutorCore
from .base import AbstractBaseTool, ToolCategory, ToolVisibility
from .function import FunctionTool
from .sandboxed_tool.sandbox_config import sandbox_config

logger = logging.getLogger(__name__)


class PythonExecutorFunctionTool(FunctionTool):
    """Python executor tool with BASIC category."""

    category = ToolCategory.BASIC


class PythonExecutorArgs(BaseModel):
    code: str = Field(description="Python code to execute")
    capture_output: bool = Field(
        default=True, description="Whether to capture stdout/stderr"
    )


class PythonExecutorResult(BaseModel):
    success: bool = Field(description="Whether the code executed successfully")
    output: str = Field(description="Output from the code execution")
    error: str = Field(default="", description="Error message if execution failed")
    generated_files: list[str] = Field(default_factory=list)
    file_refs: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


class PythonExecutorTool(AbstractBaseTool):
    """Framework wrapper for the pure Python executor tool"""

    concurrency_safe = True

    def __init__(self, workspace: Optional[TaskWorkspace] = None) -> None:
        self._visibility = ToolVisibility.PUBLIC
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "python_executor"

    @property
    def description(self) -> str:
        return """Execute Python code safely and return the output.
        Supports most Python operations including calculations, data processing, and visualization.
        Captures stdout and stderr from the execution."""

    @property
    def tags(self) -> list[str]:
        return ["python", "code", "execution", "computation"]

    def args_type(self) -> Type[BaseModel]:
        return PythonExecutorArgs

    def return_type(self) -> Type[BaseModel]:
        return PythonExecutorResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        exec_args = PythonExecutorArgs.model_validate(args)

        # Determine working directory
        working_directory = self._get_working_directory()

        # Add workspace environment variables if workspace is available
        workspace_env = self._get_workspace_env()

        # Create core executor instance
        executor = PythonExecutorCore(
            working_directory=working_directory,
            environment=workspace_env,
        )

        # Add workspace variables to the executor's globals if available
        if workspace_env:
            # Inject workspace variables into the code execution environment
            env_code = "\n".join([f"{k} = {repr(v)}" for k, v in workspace_env.items()])
            full_code = f"{env_code}\n\n{exec_args.code}"
        else:
            full_code = exec_args.code

        # Execute code in an isolated subprocess. The child reports only files
        # written by this execution, avoiding concurrent snapshot bleed-through.
        if self._workspace and working_directory:
            result = executor.execute_code(full_code, exec_args.capture_output)
            written_files = result.pop(_INTERNAL_WRITTEN_FILES_KEY, [])
            if result.get("success"):
                workspace_files = self._existing_workspace_files(written_files)
                self._register_workspace_files(workspace_files)
                result.update(
                    build_generated_file_metadata(
                        workspace=self._workspace,
                        file_paths=workspace_files,
                    )
                )
        else:
            result = executor.execute_code(full_code, exec_args.capture_output)
            result.pop(_INTERNAL_WRITTEN_FILES_KEY, None)

        return PythonExecutorResult(**result).model_dump()

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        return await asyncio.to_thread(self.run_json_sync, args)

    def _get_working_directory(self) -> Optional[str]:
        """Determine the working directory based on workspace settings"""
        if self._workspace:
            # Use workspace output directory as working directory
            return str(self._workspace.resolve_path(""))
        return None

    def _get_workspace_env(self) -> Optional[Dict[str, str]]:
        """Get workspace environment variables"""
        if not self._workspace:
            return None

        return {
            "WORKSPACE_OUTPUT_DIR": str(self._workspace.resolve_path("")),
            "WORKSPACE_INPUT_DIR": str(self._workspace.resolve_path("", "input")),
            "WORKSPACE_TEMP_DIR": str(self._workspace.resolve_path("", "temp")),
            "WORKSPACE_DIR": str(self._workspace.workspace_dir.resolve()),
        }

    def _existing_workspace_files(self, file_paths: Any) -> list[Path]:
        """Return existing files from this workspace for this execution only."""
        if not self._workspace or not isinstance(file_paths, list):
            return []

        workspace_root = self._workspace.workspace_dir.resolve()
        result: list[Path] = []
        seen: set[str] = set()

        for file_path in file_paths:
            try:
                resolved = Path(str(file_path)).resolve()
                resolved.relative_to(workspace_root)
            except (OSError, ValueError):
                continue
            if not resolved.exists() or not resolved.is_file():
                continue
            if any(
                part.startswith(".") or part == "__pycache__"
                for part in resolved.relative_to(workspace_root).parts
            ):
                continue
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            result.append(resolved)

        return result

    def _register_workspace_files(self, file_paths: list[Path]) -> None:
        if not self._workspace:
            return
        for file_path in file_paths:
            try:
                self._workspace.register_file(str(file_path))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to register Python executor generated file %s: %s",
                    file_path,
                    exc,
                )


@sandbox_config(
    packages=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "openpyxl>=3.1.0",  # required by the xlsx-financial-report skill
    ]
)
class PythonExecutorToolForBasic(PythonExecutorTool):
    """Python executor tool with BASIC category."""

    category = ToolCategory.BASIC

    @property
    def name(self) -> str:
        return "execute_python_code"


def get_python_executor_tool(info: Optional[dict[str, Any]] = None) -> FunctionTool:
    """
    Create a workspace-bound Python executor tool.

    Args:
        info: Dictionary containing workspace information

    Returns:
        A Python executor tool bound to the specified workspace
    """
    # Extract workspace from info if provided
    workspace = None
    if info and "workspace" in info:
        workspace = (
            info["workspace"] if isinstance(info["workspace"], TaskWorkspace) else None
        )

    # Create workspace-bound Python executor
    executor = PythonExecutorTool(workspace=workspace)

    # Wrap as LangChain tool
    def execute_python_code(code: str, capture_output: bool = True) -> Dict[str, Any]:
        result: Dict[str, Any] = executor.run_json_sync(
            {"code": code, "capture_output": capture_output}
        )
        return result

    return PythonExecutorFunctionTool(
        execute_python_code,
        description=executor.description,
        concurrency_safe=True,
    )


def create_python_executor_tool(workspace: TaskWorkspace) -> AbstractBaseTool:
    """Create Python executor tool bound to workspace"""
    return PythonExecutorTool(workspace)
