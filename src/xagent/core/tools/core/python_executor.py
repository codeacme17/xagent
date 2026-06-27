"""
Pure Python Code Execution Tool
Standalone Python execution functionality without framework dependencies
"""

import ast
import contextlib
import importlib
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_INTERNAL_WRITTEN_FILES_KEY = "_xagent_written_files"


class _LazyModule:
    """Import an optional module only when executed code first touches it."""

    def __init__(self, module_name: str, *, matplotlib_agg: bool = False) -> None:
        self._module_name = module_name
        self._matplotlib_agg = matplotlib_agg
        self._module: Any = None

    def _load(self) -> Any:
        if self._module is None:
            if self._matplotlib_agg:
                import matplotlib

                matplotlib.use("Agg")
            self._module = importlib.import_module(self._module_name)
        return self._module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)

    def __repr__(self) -> str:
        return f"<lazy module {self._module_name}>"


class PythonExecutorCore:
    """Pure Python executor without framework dependencies"""

    def __init__(
        self,
        working_directory: Optional[str] = None,
        environment: Optional[dict[str, str]] = None,
    ):
        """
        Initialize the Python executor.

        Args:
            working_directory: Directory to use as working directory during execution
            environment: Environment variables to expose to executed code
        """
        self.working_directory = working_directory
        self.environment = environment or {}

    def execute_code(self, code: str, capture_output: bool = True) -> Dict[str, Any]:
        """
        Execute Python code and return result.

        Args:
            code: Python code to execute
            capture_output: Whether to capture stdout/stderr

        Returns:
            Dictionary with success status, output, and error information
        """
        try:
            # Validate syntax first
            ast.parse(code)

            return self._execute_code_in_subprocess(code, capture_output)

        except SyntaxError as e:
            return {"success": False, "output": "", "error": f"Syntax Error: {str(e)}"}
        except Exception as e:
            return {"success": False, "output": "", "error": f"Error: {str(e)}"}

    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create a safe globals environment with common imports"""
        safe_globals = {
            "__builtins__": __builtins__,
            "print": print,
            "len": len,
            "range": range,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "abs": abs,
            "max": max,
            "min": min,
            "sum": sum,
            "sorted": sorted,
            "reversed": reversed,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "any": any,
            "all": all,
        }

        # Add common modules
        try:
            import datetime
            import json
            import math
            import re

            # Set matplotlib backend to non-interactive
            os.environ["MPLBACKEND"] = "Agg"

            safe_globals.update(
                {
                    "math": math,
                    "json": json,
                    "datetime": datetime,
                    "re": re,
                    "os": os,
                }
            )
        except ImportError:
            pass

        safe_globals["np"] = _LazyModule("numpy")
        safe_globals["numpy"] = safe_globals["np"]
        safe_globals["pd"] = _LazyModule("pandas")
        safe_globals["pandas"] = safe_globals["pd"]
        safe_globals["matplotlib"] = _LazyModule("matplotlib", matplotlib_agg=True)
        safe_globals["plt"] = _LazyModule("matplotlib.pyplot", matplotlib_agg=True)

        return safe_globals

    def _execute_code_in_subprocess(
        self, code: str, capture_output: bool
    ) -> Dict[str, Any]:
        """Execute code in a child process to isolate process-global state."""
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in self.environment.items()})

        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            payload = {
                "code": code,
                "capture_output": capture_output,
                "result_path": str(result_path),
            }

            try:
                process = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--xagent-python-executor-child",
                    ],
                    input=json.dumps(payload),
                    text=True,
                    capture_output=True,
                    cwd=self.working_directory,
                    env=env,
                )
            except Exception as exc:
                return {
                    "success": False,
                    "output": "",
                    "error": f"Error: {str(exc)}",
                    _INTERNAL_WRITTEN_FILES_KEY: [],
                }

            if result_path.exists():
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    if isinstance(result, dict):
                        return result
                except Exception as exc:
                    logger.warning(
                        "Failed to read Python executor child result: %s", exc
                    )

            error = process.stderr.strip() or process.stdout.strip()
            if not error:
                error = f"Python executor child failed with code {process.returncode}"
            return {
                "success": False,
                "output": process.stdout if capture_output else "",
                "error": error,
                _INTERNAL_WRITTEN_FILES_KEY: [],
            }


def _recorded_path(path: Any) -> str | None:
    if isinstance(path, int):
        return None
    try:
        raw_path = Path(os.fsdecode(path)).expanduser()
    except (TypeError, ValueError):
        return None
    if not raw_path.is_absolute():
        raw_path = Path.cwd() / raw_path
    return str(raw_path.resolve(strict=False))


def _open_event_is_write(mode: Any, flags: Any) -> bool:
    if isinstance(mode, str):
        return any(marker in mode for marker in ("w", "a", "x", "+"))
    if isinstance(flags, int):
        return bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC)
        )
    return False


def _execute_child_payload(payload: dict[str, Any]) -> dict[str, Any]:
    code = str(payload.get("code") or "")
    capture_output = bool(payload.get("capture_output", True))

    output_buffer = io.StringIO()
    error_buffer = io.StringIO()
    written_files: set[str] = set()

    def audit_hook(event: str, args: tuple[Any, ...]) -> None:
        try:
            if event == "open" and len(args) >= 3:
                path, mode, flags = args[:3]
                if _open_event_is_write(mode, flags):
                    recorded = _recorded_path(path)
                    if recorded:
                        written_files.add(recorded)
            elif event == "os.rename" and len(args) >= 2:
                recorded = _recorded_path(args[1])
                if recorded:
                    written_files.add(recorded)
        except Exception:
            return

    sys.addaudithook(audit_hook)

    exec_namespace = PythonExecutorCore()._create_safe_globals()
    initial_names = set(exec_namespace)

    try:
        if capture_output:
            with (
                contextlib.redirect_stdout(output_buffer),
                contextlib.redirect_stderr(error_buffer),
            ):
                exec(code, exec_namespace, exec_namespace)
        else:
            exec(code, exec_namespace, exec_namespace)

        output = output_buffer.getvalue() if capture_output else ""

        if not output and exec_namespace:
            visible_vars = {
                k: v
                for k, v in exec_namespace.items()
                if k not in initial_names and not k.startswith("_")
            }
            if visible_vars:
                output = "Variables created:\n"
                for name, value in visible_vars.items():
                    output += f"{name} = {repr(value)}\n"

        return {
            "success": True,
            "output": output or "Code executed successfully (no output)",
            "error": "",
            _INTERNAL_WRITTEN_FILES_KEY: sorted(written_files),
        }

    except Exception:
        error_msg = traceback.format_exc()
        stderr_content = error_buffer.getvalue() if capture_output else ""
        return {
            "success": False,
            "output": output_buffer.getvalue() if capture_output else "",
            "error": f"{error_msg}\n{stderr_content}".strip(),
            _INTERNAL_WRITTEN_FILES_KEY: sorted(written_files),
        }


def _child_main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, dict):
            raise ValueError("Child payload must be a JSON object")
        result_path = Path(str(payload["result_path"]))
        result = _execute_child_payload(payload)
        result_path.write_text(json.dumps(result), encoding="utf-8")
        return 0
    except Exception as exc:
        fallback = {
            "success": False,
            "output": "",
            "error": f"Python executor child error: {exc}",
            _INTERNAL_WRITTEN_FILES_KEY: [],
        }
        result_path_value = locals().get("payload", {}).get("result_path")
        if result_path_value:
            try:
                Path(str(result_path_value)).write_text(
                    json.dumps(fallback), encoding="utf-8"
                )
            except Exception:
                pass
        print(fallback["error"], file=sys.stderr)
        return 1


# Convenience function for direct usage
def execute_python_code(
    code: str, capture_output: bool = True, working_directory: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute Python code and return result.

    Args:
        code: Python code to execute
        capture_output: Whether to capture stdout/stderr
        working_directory: Directory to use as working directory

    Returns:
        Dictionary with execution results
    """
    executor = PythonExecutorCore(working_directory)
    result = executor.execute_code(code, capture_output)
    result.pop(_INTERNAL_WRITTEN_FILES_KEY, None)
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--xagent-python-executor-child":
        raise SystemExit(_child_main())
