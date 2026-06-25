"""Tests for task-scoped sandbox worker cleanup in AgentServiceManager."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from xagent.web.api.chat import AgentServiceManager


class _FakeAgentService:
    def __init__(
        self,
        *,
        result: dict | None = None,
        started: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
    ) -> None:
        self.result = result or {"success": True}
        self.started = started
        self.release = release

    async def execute_task(self, **_kwargs):
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            await self.release.wait()
        return self.result


@pytest.mark.asyncio
async def test_worker_cleanup_does_not_block_other_users() -> None:
    """Slow worker cleanup for one user must not serialize other users."""
    manager = AgentServiceManager()
    provider = AsyncMock()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()

    async def slow_cleanup() -> None:
        cleanup_started.set()
        await cleanup_release.wait()

    provider.cleanup_worker_sandboxes.side_effect = slow_cleanup
    manager._sandboxes["user:7"] = provider
    manager._sandbox_active_tasks["user:7"] = 1
    manager._sandboxes["user:8"] = AsyncMock()
    manager._agent_owner_ids[2] = 8
    manager._agent_sandbox_keys[2] = "user:8"

    release_task = asyncio.create_task(manager._release_sandbox_task("user:7"))
    await cleanup_started.wait()

    try:
        sandbox_key = await asyncio.wait_for(
            manager._acquire_sandbox_task("2"),
            timeout=0.05,
        )
    finally:
        cleanup_release.set()
        await release_task

    assert sandbox_key == "user:8"
    assert manager._sandbox_active_tasks["user:8"] == 1


@pytest.mark.asyncio
async def test_worker_cleanup_removes_sandbox_lifecycle_lock_entry() -> None:
    """Lifecycle lock entries should not leak after a provider is removed."""
    manager = AgentServiceManager()
    provider = AsyncMock()
    manager._sandboxes["user:7"] = provider
    manager._sandbox_active_tasks["user:7"] = 1

    await manager._release_sandbox_task("user:7")

    assert "user:7" not in manager._sandbox_lifecycle_locks


@pytest.mark.asyncio
async def test_worker_cleanup_blocks_same_user_provider_recreate(monkeypatch) -> None:
    """The same user must not recreate same-named workers during cleanup."""
    manager = AgentServiceManager()
    old_provider = AsyncMock()
    new_provider = AsyncMock()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    create_called = asyncio.Event()

    async def slow_cleanup() -> None:
        cleanup_started.set()
        await cleanup_release.wait()

    async def create_provider(*_args, **_kwargs):
        create_called.set()
        return new_provider

    sandbox_mgr = AsyncMock()
    sandbox_mgr.create_lease_provider.side_effect = create_provider
    monkeypatch.setattr(
        "xagent.web.sandbox_manager.get_sandbox_manager",
        lambda: sandbox_mgr,
    )

    old_provider.cleanup_worker_sandboxes.side_effect = slow_cleanup
    manager._sandboxes["user:7"] = old_provider
    manager._sandbox_active_tasks["user:7"] = 1

    release_task = asyncio.create_task(manager._release_sandbox_task("user:7"))
    await cleanup_started.wait()

    create_task = asyncio.create_task(
        manager._get_or_create_task_sandbox(
            task_id=3,
            workspace_owner_id=7,
            workspace_config={},
        )
    )

    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(create_called.wait(), timeout=0.05)

        cleanup_release.set()
        sandbox = await asyncio.wait_for(create_task, timeout=0.5)
        await release_task
    finally:
        cleanup_release.set()
        await asyncio.gather(release_task, create_task, return_exceptions=True)

    assert sandbox is new_provider
    sandbox_mgr.create_lease_provider.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_task_releases_sandbox_workers_after_task() -> None:
    """Terminal task execution should release cached worker sandboxes."""
    manager = AgentServiceManager()
    provider = AsyncMock()
    agent_service = _FakeAgentService()
    manager._sandboxes["user:7"] = provider
    manager._agents[1] = agent_service
    manager._agent_owner_ids[1] = 7
    manager._agent_sandbox_keys[1] = "user:7"

    result = await manager.execute_task(
        agent_service=agent_service,
        task="run",
        task_id="1",
    )

    assert result == {"success": True}
    assert "user:7" not in manager._sandboxes
    provider.cleanup_worker_sandboxes.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_task_evicts_agents_for_released_sandbox_provider() -> None:
    """Cached agents must not retain a provider after its sandbox is released."""
    manager = AgentServiceManager()
    provider = AsyncMock()
    agent_service = _FakeAgentService()
    other_agent_service = _FakeAgentService()
    manager._sandboxes["user:7"] = provider
    manager._agents[1] = agent_service
    manager._agents[2] = other_agent_service
    manager._agent_owner_ids[1] = 7
    manager._agent_owner_ids[2] = 7
    manager._agent_sandbox_keys[1] = "user:7"
    manager._agent_sandbox_keys[2] = "user:7"

    await manager.execute_task(
        agent_service=agent_service,
        task="run",
        task_id="1",
    )

    assert "user:7" not in manager._sandboxes
    assert 1 not in manager._agents
    assert 2 not in manager._agents
    assert 1 not in manager._agent_owner_ids
    assert 2 not in manager._agent_owner_ids
    assert 1 not in manager._agent_sandbox_keys
    assert 2 not in manager._agent_sandbox_keys


@pytest.mark.asyncio
async def test_execute_task_keeps_workers_until_last_same_user_task_finishes() -> None:
    """One task finishing must not delete workers still shared by another task."""
    manager = AgentServiceManager()
    provider = AsyncMock()
    first_agent_service = _FakeAgentService()
    second_agent_service = _FakeAgentService()
    manager._sandboxes["user:7"] = provider
    manager._agents[1] = first_agent_service
    manager._agents[2] = second_agent_service
    manager._agent_owner_ids[1] = 7
    manager._agent_owner_ids[2] = 7
    manager._agent_sandbox_keys[1] = "user:7"
    manager._agent_sandbox_keys[2] = "user:7"

    first_started = asyncio.Event()
    first_release = asyncio.Event()
    second_started = asyncio.Event()
    second_release = asyncio.Event()
    first_agent_service.started = first_started
    first_agent_service.release = first_release
    second_agent_service.started = second_started
    second_agent_service.release = second_release

    first_task = asyncio.create_task(
        manager.execute_task(
            agent_service=first_agent_service,
            task="first",
            task_id="1",
        )
    )
    await first_started.wait()

    second_task = asyncio.create_task(
        manager.execute_task(
            agent_service=second_agent_service,
            task="second",
            task_id="2",
        )
    )
    await second_started.wait()

    first_release.set()
    await first_task
    assert "user:7" in manager._sandboxes
    assert 1 in manager._agents
    assert 2 in manager._agents
    provider.cleanup_worker_sandboxes.assert_not_awaited()

    second_release.set()
    await second_task
    assert "user:7" not in manager._sandboxes
    assert 1 not in manager._agents
    assert 2 not in manager._agents
    provider.cleanup_worker_sandboxes.assert_awaited_once()
