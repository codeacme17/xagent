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
async def test_execute_task_releases_sandbox_workers_after_task() -> None:
    """Terminal task execution should release cached worker sandboxes."""
    manager = AgentServiceManager()
    provider = AsyncMock()
    manager._sandboxes["user:7"] = provider
    manager._agent_owner_ids[1] = 7

    result = await manager.execute_task(
        agent_service=_FakeAgentService(),
        task="run",
        task_id="1",
    )

    assert result == {"success": True}
    provider.cleanup_worker_sandboxes.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_task_keeps_workers_until_last_same_user_task_finishes() -> None:
    """One task finishing must not delete workers still shared by another task."""
    manager = AgentServiceManager()
    provider = AsyncMock()
    manager._sandboxes["user:7"] = provider
    manager._agent_owner_ids[1] = 7
    manager._agent_owner_ids[2] = 7

    first_started = asyncio.Event()
    first_release = asyncio.Event()
    second_started = asyncio.Event()
    second_release = asyncio.Event()

    first_task = asyncio.create_task(
        manager.execute_task(
            agent_service=_FakeAgentService(
                started=first_started, release=first_release
            ),
            task="first",
            task_id="1",
        )
    )
    await first_started.wait()

    second_task = asyncio.create_task(
        manager.execute_task(
            agent_service=_FakeAgentService(
                started=second_started, release=second_release
            ),
            task="second",
            task_id="2",
        )
    )
    await second_started.wait()

    first_release.set()
    await first_task
    provider.cleanup_worker_sandboxes.assert_not_awaited()

    second_release.set()
    await second_task
    provider.cleanup_worker_sandboxes.assert_awaited_once()
