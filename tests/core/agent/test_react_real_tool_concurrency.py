"""Inc.1 follow-up — the real safe-set tools actually parallelize.

The unit tests elsewhere prove the scheduler's ordering/isolation invariants
with ``FakeTool``. This module instead drives the *real* production tool classes
that are marked ``read_only`` / ``concurrency_safe`` through the real
``ReActPattern`` concurrent path, to guard against two regressions the fakes
cannot catch:

1. A real tool loses its ``concurrency_safe`` metadata (so the scheduler would
   silently fall back to serial) — covered by the segmentation assertions.
2. A real tool's ``run_json_async`` wrapper serializes the batch even though the
   scheduler offered it concurrency (e.g. a future refactor adds blocking work
   around the awaited I/O seam) — covered by the barrier-overlap assertions.

Scope boundary (intentional): the barrier is patched in at each tool's awaited
I/O seam (``fetch_web_content`` / ``*Core.search``), so these tests verify the
scheduler plus the tool's own async wrapper overlap. They do *not* exercise the
production HTTP client itself (which is replaced) — keeping each tool's network
layer non-blocking remains that tool's responsibility, and is asserted only
indirectly via ``run_json_async`` being a coroutine function.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterator

import pytest

from tests.core.agent.concurrency_harness import (
    ConcurrencyTracker,
    FakeRuntime,
    RecordingContext,
    make_react,
    make_tool_call,
)
from xagent.core.tools.adapters.vibe.exa_web_search import ExaWebSearchTool
from xagent.core.tools.adapters.vibe.browser_use import BrowserEvaluateTool
from xagent.core.tools.adapters.vibe.fetch_web_content import FetchWebContentTool
from xagent.core.tools.adapters.vibe.image_tool import ImageGenerationTool
from xagent.core.tools.adapters.vibe.python_executor import PythonExecutorTool
from xagent.core.tools.adapters.vibe.tavily_web_search import TavilyWebSearchTool
from xagent.core.tools.adapters.vibe.web_search import WebSearchTool
from xagent.core.tools.adapters.vibe.workspace_file_tool import WorkspaceFileTools
from xagent.core.tools.adapters.vibe.zhipu_web_search import ZhipuWebSearchTool
from xagent.core.tools.core import browser_use as browser_core
from xagent.core.workspace import TaskWorkspace

# How long a batched I/O seam waits to rendezvous with its sibling before giving
# up. A serialized batch never rendezvouses, so it trips this instead of hanging.
_RENDEZVOUS_TIMEOUT = 3.0


class _FetchStub:
    """Stand-in for ``WebContentFetchResult`` (only ``as_dict`` is consumed)."""

    def as_dict(self) -> dict[str, Any]:
        return {"success": True, "url": "https://example.com", "content": "ok"}


class _WorkspaceRegisterTracker:
    def __init__(self, workspace: TaskWorkspace) -> None:
        self.workspace = workspace
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


class _TrackedBrowserPage:
    def __init__(
        self,
        session_id: str,
        tracker: ConcurrencyTracker,
        barrier: asyncio.Barrier | None = None,
    ) -> None:
        self._session_id = session_id
        self._tracker = tracker
        self._barrier = barrier

    async def evaluate(self, javascript: str) -> str:
        self._tracker.enter(self._session_id)
        try:
            if self._barrier is not None:
                await asyncio.wait_for(
                    self._barrier.wait(), timeout=_RENDEZVOUS_TIMEOUT
                )
            else:
                await asyncio.sleep(0.05)
            return javascript
        finally:
            self._tracker.leave(self._session_id)


class _TrackedBrowserSession:
    def __init__(
        self,
        session_id: str,
        tracker: ConcurrencyTracker,
        barrier: asyncio.Barrier | None = None,
    ) -> None:
        self._page = _TrackedBrowserPage(session_id, tracker, barrier)
        self._operation_lock = asyncio.Lock()

    @asynccontextmanager
    async def operation_guard(self) -> Iterator[None]:
        async with self._operation_lock:
            yield

    async def get_page(self) -> _TrackedBrowserPage:
        return self._page


class _TrackedBrowserManager:
    def __init__(
        self,
        tracker: ConcurrencyTracker,
        barrier: asyncio.Barrier | None = None,
    ) -> None:
        self._tracker = tracker
        self._barrier = barrier
        self._sessions: dict[str, _TrackedBrowserSession] = {}

    async def get_or_create(
        self, session_id: str, headless: bool = False
    ) -> _TrackedBrowserSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = _TrackedBrowserSession(
                session_id, self._tracker, self._barrier
            )
        return self._sessions[session_id]


class _TrackedImageModel:
    model_id = "tracked-image"

    def __init__(
        self,
        tracker: ConcurrencyTracker,
        barrier: asyncio.Barrier,
    ) -> None:
        self._tracker = tracker
        self._barrier = barrier

    def has_ability(self, ability: str) -> bool:
        return ability == "generate"

    async def generate_image(self, **kwargs: Any) -> dict[str, Any]:
        prompt = str(kwargs["prompt"])
        self._tracker.enter(prompt)
        try:
            await asyncio.wait_for(self._barrier.wait(), timeout=_RENDEZVOUS_TIMEOUT)
        finally:
            self._tracker.leave(prompt)

        encoded = base64.b64encode(f"image:{prompt}".encode()).decode()
        return {
            "image_url": f"data:image/png;base64,{encoded}",
            "request_id": f"request-{prompt}",
        }


@dataclass
class _RealToolCase:
    label: str
    tool_factory: Callable[[], Any]
    # Dotted path of the awaited I/O seam to replace with the barrier.
    seam: str
    # Value the patched seam returns once both callers rendezvous.
    seam_return: Any
    args: dict[str, Any]


_CASES = [
    _RealToolCase(
        label="fetch_web_content",
        tool_factory=FetchWebContentTool,
        seam="xagent.core.tools.adapters.vibe.fetch_web_content.fetch_web_content",
        seam_return=_FetchStub(),
        args={"url": "https://example.com"},
    ),
    _RealToolCase(
        label="web_search",
        tool_factory=WebSearchTool,
        seam="xagent.core.tools.adapters.vibe.web_search.WebSearchCore.search",
        seam_return=[],
        args={"query": "x"},
    ),
    _RealToolCase(
        label="zhipu_web_search",
        tool_factory=ZhipuWebSearchTool,
        seam="xagent.core.tools.adapters.vibe.zhipu_web_search.ZhipuWebSearchCore.search",
        # Zhipu wraps the response through ``normalize_results``; an empty dict
        # normalizes to an empty result list.
        seam_return={},
        args={"query": "x"},
    ),
    _RealToolCase(
        label="exa_web_search",
        tool_factory=ExaWebSearchTool,
        seam="xagent.core.tools.adapters.vibe.exa_web_search.ExaWebSearchCore.search",
        seam_return=[],
        args={"query": "x"},
    ),
    _RealToolCase(
        label="tavily_web_search",
        tool_factory=TavilyWebSearchTool,
        seam="xagent.core.tools.adapters.vibe.tavily_web_search.TavilyWebSearchCore.search",
        seam_return=[],
        args={"query": "x"},
    ),
]

_CASE_IDS = [case.label for case in _CASES]


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
def test_real_safe_tool_is_batched_as_concurrent(case: _RealToolCase) -> None:
    """Two calls to a real safe tool are recognized as a concurrent segment."""
    tool = case.tool_factory()
    pattern = make_react(parallel=True, max_concurrency=3)
    # The registered tool name can differ from the test label (e.g. the Google
    # and Tavily backends both expose "web_search"); drive the scheduler with
    # the real name so ``_find_tool`` resolves the metadata.
    batch = [make_tool_call(tool.name, case.args) for _ in range(2)]

    segment, kind = pattern._next_segment(batch, [tool])

    assert kind == "concurrent"
    assert len(segment) == 2


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
def test_real_safe_tool_exposes_async_executor(case: _RealToolCase) -> None:
    """The real tool's executor is a coroutine function (yields to the loop)."""
    tool = case.tool_factory()
    assert inspect.iscoroutinefunction(tool.run_json_async)


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
async def test_real_safe_tool_runs_concurrently(
    case: _RealToolCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real tool's async path overlaps under the concurrent scheduler.

    A two-party barrier sits at the tool's awaited I/O seam: it only releases
    once both calls are in flight, so a passing run proves real overlap (peak
    concurrency == 2) and ordered, one-result-per-call backfill. A serialized
    run never reaches the barrier together and trips ``_RENDEZVOUS_TIMEOUT``,
    surfacing as timed-out (failed) results rather than a hang.
    """
    tracker = ConcurrencyTracker()
    barrier = asyncio.Barrier(2)

    async def _seam(*_args: Any, **_kwargs: Any) -> Any:
        tracker.enter(case.label)
        try:
            await asyncio.wait_for(barrier.wait(), timeout=_RENDEZVOUS_TIMEOUT)
        finally:
            tracker.leave(case.label)
        return case.seam_return

    monkeypatch.setattr(case.seam, _seam)

    tool = case.tool_factory()
    pattern = make_react(parallel=True, max_concurrency=3)
    context = RecordingContext()
    batch = [make_tool_call(tool.name, case.args) for _ in range(2)]

    results = await asyncio.wait_for(
        pattern._run_concurrent_batch(batch, [tool], FakeRuntime(), context),
        timeout=_RENDEZVOUS_TIMEOUT * 2,
    )

    # Real overlap: both calls were in flight at once.
    assert tracker.peak == 2
    # I1/I2: one result per call, in input order, none of them a seam timeout.
    assert len(results) == 2
    assert all(pattern._tool_result_success(result) for result in results)
    assert [r["tool_call_id"] for r in context.tool_results] == [
        tc["id"] for tc in batch
    ]


def _workspace_write_tool(
    tmp_path: Any, workspace_id: str
) -> tuple[TaskWorkspace, _WorkspaceRegisterTracker, Any]:
    workspace = TaskWorkspace(workspace_id, str(tmp_path))
    tracker = _WorkspaceRegisterTracker(workspace)
    workspace.auto_register_files = tracker.auto_register_files  # type: ignore[method-assign]
    tool = next(
        tool
        for tool in WorkspaceFileTools(workspace).get_tools()
        if tool.name == "write_file"
    )
    return workspace, tracker, tool


def test_real_workspace_write_tool_is_batched_as_concurrent(tmp_path: Any) -> None:
    """The real workspace write tool opts into the ReAct concurrent segment."""
    _workspace, _tracker, tool = _workspace_write_tool(
        tmp_path, "react_workspace_segment"
    )
    pattern = make_react(parallel=True, max_concurrency=2)
    batch = [
        make_tool_call(tool.name, {"file_path": "one.txt", "content": "one"}),
        make_tool_call(tool.name, {"file_path": "two.txt", "content": "two"}),
    ]

    segment, kind = pattern._next_segment(batch, [tool])

    assert kind == "concurrent"
    assert segment == batch


def test_real_browser_evaluate_tool_is_batched_as_concurrent() -> None:
    """Browser evaluate opts into the ReAct concurrent segment."""
    tool = BrowserEvaluateTool()
    pattern = make_react(parallel=True, max_concurrency=2)
    batch = [
        make_tool_call(tool.name, {"session_id": "session-a", "javascript": "'first'"}),
        make_tool_call(
            tool.name, {"session_id": "session-b", "javascript": "'second'"}
        ),
    ]

    segment, kind = pattern._next_segment(batch, [tool])

    assert kind == "concurrent"
    assert segment == batch


def test_real_python_executor_tool_is_batched_as_concurrent(tmp_path: Any) -> None:
    """Python executor opts into the ReAct concurrent segment after isolation."""
    workspace = TaskWorkspace("react_python_segment", str(tmp_path))
    tool = PythonExecutorTool(workspace)
    pattern = make_react(parallel=True, max_concurrency=2)
    batch = [
        make_tool_call(tool.name, {"code": "print('first')"}),
        make_tool_call(tool.name, {"code": "print('second')"}),
    ]

    segment, kind = pattern._next_segment(batch, [tool])

    assert kind == "concurrent"
    assert segment == batch


async def test_real_browser_evaluate_tool_serializes_same_session_in_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same-session browser calls stay serial inside a ReAct batch."""
    tracker = ConcurrencyTracker()
    manager = _TrackedBrowserManager(tracker)
    monkeypatch.setattr(browser_core, "PLAYWRIGHT_AVAILABLE", True)
    monkeypatch.setattr(browser_core, "get_browser_manager", lambda: manager)

    tool = BrowserEvaluateTool()
    pattern = make_react(parallel=True, max_concurrency=2)
    context = RecordingContext()
    batch = [
        make_tool_call(
            tool.name, {"session_id": "same-session", "javascript": "'first'"}
        ),
        make_tool_call(
            tool.name, {"session_id": "same-session", "javascript": "'second'"}
        ),
    ]

    results = await asyncio.wait_for(
        pattern._run_concurrent_batch(batch, [tool], FakeRuntime(), context),
        timeout=_RENDEZVOUS_TIMEOUT * 2,
    )

    assert tracker.peak == 1
    assert len(results) == 2
    assert all(pattern._tool_result_success(result) for result in results)


async def test_real_browser_evaluate_tool_overlaps_different_sessions_in_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different browser sessions can overlap inside a ReAct batch."""
    tracker = ConcurrencyTracker()
    barrier = asyncio.Barrier(2)
    manager = _TrackedBrowserManager(tracker, barrier)
    monkeypatch.setattr(browser_core, "PLAYWRIGHT_AVAILABLE", True)
    monkeypatch.setattr(browser_core, "get_browser_manager", lambda: manager)

    tool = BrowserEvaluateTool()
    pattern = make_react(parallel=True, max_concurrency=2)
    context = RecordingContext()
    batch = [
        make_tool_call(tool.name, {"session_id": "session-a", "javascript": "'first'"}),
        make_tool_call(
            tool.name, {"session_id": "session-b", "javascript": "'second'"}
        ),
    ]

    results = await asyncio.wait_for(
        pattern._run_concurrent_batch(batch, [tool], FakeRuntime(), context),
        timeout=_RENDEZVOUS_TIMEOUT * 2,
    )

    assert tracker.peak == 2
    assert len(results) == 2
    assert all(pattern._tool_result_success(result) for result in results)


async def test_real_workspace_write_tool_serializes_same_path_in_batch(
    tmp_path: Any,
) -> None:
    """Same normalized workspace path stays serial inside a ReAct batch."""
    workspace, tracker, tool = _workspace_write_tool(
        tmp_path, "react_workspace_same_path"
    )
    pattern = make_react(parallel=True, max_concurrency=2)
    context = RecordingContext()
    batch = [
        make_tool_call(tool.name, {"file_path": "same.txt", "content": "first"}),
        make_tool_call(
            tool.name, {"file_path": "output/same.txt", "content": "second"}
        ),
    ]

    results = await asyncio.wait_for(
        pattern._run_concurrent_batch(batch, [tool], FakeRuntime(), context),
        timeout=_RENDEZVOUS_TIMEOUT * 2,
    )

    assert tracker.peak == 1
    assert len(results) == 2
    assert all(pattern._tool_result_success(result) for result in results)
    assert (workspace.output_dir / "same.txt").exists()


async def test_real_workspace_write_tool_overlaps_different_paths_in_batch(
    tmp_path: Any,
) -> None:
    """Different normalized workspace paths can overlap in a ReAct batch."""
    workspace, tracker, tool = _workspace_write_tool(
        tmp_path, "react_workspace_different_paths"
    )
    pattern = make_react(parallel=True, max_concurrency=2)
    context = RecordingContext()
    batch = [
        make_tool_call(tool.name, {"file_path": "one.txt", "content": "one"}),
        make_tool_call(tool.name, {"file_path": "two.txt", "content": "two"}),
    ]

    results = await asyncio.wait_for(
        pattern._run_concurrent_batch(batch, [tool], FakeRuntime(), context),
        timeout=_RENDEZVOUS_TIMEOUT * 2,
    )

    assert tracker.peak == 2
    assert len(results) == 2
    assert all(pattern._tool_result_success(result) for result in results)
    assert (workspace.output_dir / "one.txt").read_text(encoding="utf-8") == "one"
    assert (workspace.output_dir / "two.txt").read_text(encoding="utf-8") == "two"


async def test_real_python_executor_tool_overlaps_and_isolates_batch_results(
    tmp_path: Any,
) -> None:
    """ReAct batch execution overlaps Python calls without output/artifact bleed."""
    workspace = TaskWorkspace("react_python_overlap", str(tmp_path))
    tool = PythonExecutorTool(workspace)
    pattern = make_react(parallel=True, max_concurrency=2)
    context = RecordingContext()

    def code_for(label: str) -> str:
        dirname = f"{label}_dir"
        artifact = f"{label}.docx"
        return f"""
import os
import time
from pathlib import Path

Path({dirname!r}).mkdir(exist_ok=True)
os.chdir({dirname!r})
print({f"{label}:entered:"!r} + Path.cwd().name)
time.sleep(1.0)
print({f"{label}:final:"!r} + Path.cwd().name)
Path("..", {artifact!r}).write_bytes({label.encode()!r})
"""

    batch = [
        make_tool_call(tool.name, {"code": code_for("alpha")}),
        make_tool_call(tool.name, {"code": code_for("beta")}),
    ]

    started = time.perf_counter()
    results = await asyncio.wait_for(
        pattern._run_concurrent_batch(batch, [tool], FakeRuntime(), context),
        timeout=_RENDEZVOUS_TIMEOUT * 2,
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 2.0
    assert len(results) == 2
    assert all(pattern._tool_result_success(result) for result in results)
    assert "alpha:final:alpha_dir" in results[0]["output"]
    assert "beta:" not in results[0]["output"]
    assert "beta:final:beta_dir" in results[1]["output"]
    assert "alpha:" not in results[1]["output"]
    assert results[0]["generated_files"] == ["alpha.docx"]
    assert results[1]["generated_files"] == ["beta.docx"]
    assert [r["tool_call_id"] for r in context.tool_results] == [
        tc["id"] for tc in batch
    ]


async def test_real_image_generation_tool_overlaps_and_registers_unique_artifacts(
    tmp_path: Any,
) -> None:
    """Image generation runs through ReAct concurrency with isolated artifacts."""
    tracker = ConcurrencyTracker()
    barrier = asyncio.Barrier(2)
    workspace = TaskWorkspace("react_image_generation", str(tmp_path))
    tool = ImageGenerationTool(
        {"tracked-image": _TrackedImageModel(tracker, barrier)},
        workspace=workspace,
    ).get_tools()[0]
    pattern = make_react(parallel=True, max_concurrency=2)
    context = RecordingContext()
    batch = [
        make_tool_call(tool.name, {"prompt": "alpha"}),
        make_tool_call(tool.name, {"prompt": "beta"}),
    ]

    segment, kind = pattern._next_segment(batch, [tool])
    assert kind == "concurrent"
    assert segment == batch

    results = await asyncio.wait_for(
        pattern._run_concurrent_batch(batch, [tool], FakeRuntime(), context),
        timeout=_RENDEZVOUS_TIMEOUT * 2,
    )

    assert tracker.peak == 2
    assert len(results) == 2
    assert all(pattern._tool_result_success(result) for result in results)
    assert results[0]["request_id"] == "request-alpha"
    assert results[1]["request_id"] == "request-beta"

    image_paths = [result["image_path"] for result in results]
    assert len(set(image_paths)) == 2
    assert all(
        path and (workspace.output_dir / Path(path).name).exists()
        for path in image_paths
    )

    for result in results:
        assert result["file_id"]
        assert result["file_ref"]["file_id"] == result["file_id"]
        assert result["artifacts"] == [
            {
                "type": "image",
                "file_id": result["file_id"],
                "filename": Path(result["image_path"]).name,
                "mime_type": "image/png",
                "display": "inline",
            }
        ]

    assert [r["tool_call_id"] for r in context.tool_results] == [
        tc["id"] for tc in batch
    ]
