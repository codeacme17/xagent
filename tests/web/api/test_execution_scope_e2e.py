"""#757 cross-cutting acceptance: every subsystem consumes the same scope.

Drives the real ``get_agent_for_task`` build through the resolver path (and
the persisted-snapshot path for delegated tasks) and checks each
consumption point in one pass, so no execution path reaches one subsystem
scoped and another unscoped:

* sandbox lifecycle key (container family),
* sandbox mount workspace-config (base dir + segments),
* AgentService workspace base dir + carried scope segments,
* the recorded scope fingerprint,

plus disjointness between two scopes under one platform user and
byte-for-byte unchanged unscoped behavior. Memory-result-set disjointness
through the same contextvar mechanism is pinned in
``tests/web/test_execution_scope_memory.py``.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.config import get_uploads_dir
from xagent.core.execution_scope import (
    ExecutionScope,
    scope_fingerprint,
    set_execution_scope_resolver,
    set_execution_scope_snapshot_loader,
)
from xagent.core.workspace import scoped_user_root
from xagent.web.api.chat import AgentServiceManager
from xagent.web.models.agent import AgentStatus
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.llm_utils import AgentRuntimeFields
from xagent.web.services.task_setup_snapshot import (
    TaskSetupSnapshot,
    _TaskFields,
)

SCOPE_A = ExecutionScope(
    sandbox_key_suffix="tenant-a",
    workspace_segments=("tenant-a",),
    memory_dimensions={"tenant": "a"},
)
SCOPE_B = ExecutionScope(
    sandbox_key_suffix="tenant-b",
    workspace_segments=("tenant-b",),
    memory_dimensions={"tenant": "b"},
)


@pytest.fixture(autouse=True)
def _clear_hooks():
    set_execution_scope_resolver(None)
    set_execution_scope_snapshot_loader(None)
    yield
    set_execution_scope_resolver(None)
    set_execution_scope_snapshot_loader(None)


def _make_user() -> User:
    return User(id=1, username="e2e-user", password_hash="hash", is_admin=False)


def _make_task_row(task_id: int) -> Task:
    return Task(
        id=task_id,
        user_id=1,
        title="e2e task",
        description="x",
        status=TaskStatus.PENDING,
        agent_id=7,
        agent_type="standard",
    )


def _build_snapshot(task_id: int) -> TaskSetupSnapshot:
    return TaskSetupSnapshot(
        task=_TaskFields(
            id=task_id,
            user_id=1,
            status=TaskStatus.PENDING,
            agent_id=7,
            agent_config=None,
            model_name=None,
            compact_model_name=None,
            execution_mode="flash",
            agent_type="standard",
        ),
        task_pattern="single_call",
        task_llm=None,
        task_fast_llm=None,
        task_vision_llm=None,
        task_compact_llm=None,
        agent=AgentRuntimeFields(
            id=7,
            name="e2e-agent",
            status=AgentStatus.PUBLISHED,
            instructions="be terse",
        ),
        agent_config={
            "llms": (None, None, None, None),
            "execution_mode": "flash",
            "instructions": "be terse",
            "skills": [],
            "knowledge_bases": [],
            "tool_categories": ["basic"],
        },
        excluded_agent_id=7,
    )


def _build_db_mock(task_row: Task) -> MagicMock:
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = task_row
    return db


class _Build:
    """Everything one ``get_agent_for_task`` build touched, per subsystem."""

    def __init__(self) -> None:
        self.sandbox_lifecycle: tuple[str, str] | None = None
        self.sandbox_workspace_config: dict[str, Any] | None = None
        self.agent_service_kwargs: dict[str, Any] | None = None
        self.recorded_sandbox_key: str | None = None
        self.recorded_fingerprint: Any = None


async def _run_build(manager: AgentServiceManager, task_id: int) -> _Build:
    build = _Build()
    manager._default_llm = MagicMock()

    fake_sandbox_manager = MagicMock()

    async def _lease_provider(lifecycle_type, lifecycle_id, *, workspace_config=None):
        build.sandbox_lifecycle = (lifecycle_type, lifecycle_id)
        build.sandbox_workspace_config = dict(workspace_config or {})
        return AsyncMock()

    fake_sandbox_manager.get_or_create_lease_provider = AsyncMock(
        side_effect=_lease_provider
    )

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(manager, "_load_persisted_conversation_history")
        )
        stack.enter_context(
            patch.object(manager, "_load_persisted_execution_context", new=AsyncMock())
        )
        stack.enter_context(
            patch("xagent.web.api.chat.create_task_tracer", return_value=MagicMock())
        )
        stack.enter_context(
            patch(
                "xagent.web.api.chat.create_default_tools",
                new=AsyncMock(return_value=([], MagicMock())),
            )
        )
        stack.enter_context(
            patch(
                "xagent.web.sandbox_manager.get_sandbox_manager",
                return_value=fake_sandbox_manager,
            )
        )
        agent_service_mock = stack.enter_context(
            patch("xagent.web.api.chat.AgentService")
        )
        try:
            await manager.get_agent_for_task(
                task_id=task_id,
                db=_build_db_mock(_make_task_row(task_id)),
                user=_make_user(),
                task_setup_snapshot=_build_snapshot(task_id),
            )
        except Exception:
            pass

    if agent_service_mock.call_args is not None:
        build.agent_service_kwargs = agent_service_mock.call_args.kwargs
    build.recorded_sandbox_key = manager._agent_sandbox_keys.get(task_id)
    build.recorded_fingerprint = manager._agent_scope_fingerprints.get(task_id)
    return build


@pytest.mark.asyncio
async def test_scoped_build_applies_the_scope_to_every_subsystem() -> None:
    """No partially-applied scope: one resolved scope reaches the sandbox
    lifecycle key, the sandbox mount config, the workspace base dir, the
    carried segments, and the cache fingerprint — in the same build."""
    set_execution_scope_resolver(lambda task_id: SCOPE_A)
    build = await _run_build(AgentServiceManager(), 42)

    scoped_base = str(scoped_user_root(get_uploads_dir(), 1, ("tenant-a",)))
    assert build.sandbox_lifecycle == ("user", "1:tenant-a")
    assert build.sandbox_workspace_config["base_dir"] == scoped_base
    assert build.sandbox_workspace_config["scope_segments"] == ("tenant-a",)
    assert build.recorded_sandbox_key == "user:1:tenant-a"
    assert build.agent_service_kwargs["workspace_base_dir"] == scoped_base
    assert build.agent_service_kwargs["scope_segments"] == ("tenant-a",)
    assert build.recorded_fingerprint == scope_fingerprint(SCOPE_A)


@pytest.mark.asyncio
async def test_two_scopes_under_one_user_are_disjoint_everywhere() -> None:
    def resolver(task_id: str):
        return {"42": SCOPE_A, "43": SCOPE_B}.get(task_id)

    set_execution_scope_resolver(resolver)
    manager = AgentServiceManager()
    build_a = await _run_build(manager, 42)
    build_b = await _run_build(manager, 43)
    set_execution_scope_resolver(None)
    build_unscoped = await _run_build(AgentServiceManager(), 44)

    keys = {
        build_a.recorded_sandbox_key,
        build_b.recorded_sandbox_key,
        build_unscoped.recorded_sandbox_key,
    }
    assert keys == {"user:1:tenant-a", "user:1:tenant-b", "user:1"}
    base_dirs = {
        build.agent_service_kwargs["workspace_base_dir"]
        for build in (build_a, build_b, build_unscoped)
    }
    assert len(base_dirs) == 3
    fingerprints = {
        build.recorded_fingerprint for build in (build_a, build_b, build_unscoped)
    }
    assert len(fingerprints) == 3 and None in fingerprints


@pytest.mark.asyncio
async def test_delegated_task_builds_scoped_from_persisted_snapshot() -> None:
    """A delegated (workforce) task id is unknown to the resolver; the
    persisted snapshot drives the whole build instead."""
    set_execution_scope_resolver(lambda task_id: None)  # embedder can't map it
    set_execution_scope_snapshot_loader(
        lambda task_id: SCOPE_A if task_id == "42" else None
    )
    build = await _run_build(AgentServiceManager(), 42)

    assert build.sandbox_lifecycle == ("user", "1:tenant-a")
    assert build.recorded_sandbox_key == "user:1:tenant-a"
    assert build.agent_service_kwargs["scope_segments"] == ("tenant-a",)
    assert build.recorded_fingerprint == scope_fingerprint(SCOPE_A)


@pytest.mark.asyncio
async def test_unscoped_build_is_byte_identical_to_pre_757_behavior() -> None:
    build = await _run_build(AgentServiceManager(), 42)

    legacy_base = str(get_uploads_dir() / "user_1")
    assert build.sandbox_lifecycle == ("user", "1")
    assert build.sandbox_workspace_config["base_dir"] == legacy_base
    assert build.recorded_sandbox_key == "user:1"
    assert build.agent_service_kwargs["workspace_base_dir"] == legacy_base
    assert build.agent_service_kwargs["scope_segments"] == ()
    assert build.recorded_fingerprint is None
