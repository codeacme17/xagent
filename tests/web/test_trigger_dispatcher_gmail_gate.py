from __future__ import annotations

import asyncio

import pytest

from xagent.web import app as app_module


@pytest.mark.asyncio
async def test_trigger_dispatcher_skips_gmail_scan_when_watch_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeSession:
        def close(self) -> None:
            return None

    def fake_get_session_local():
        return FakeSession

    def fake_scan_due_gmail_watch_renewals(_db) -> int:
        return 0

    async def fake_dispatch_pending_trigger_runs(_db, *, limit: int) -> int:
        raise asyncio.CancelledError

    async def fake_to_thread(func, /, *args, **kwargs):
        calls.append(func.__name__)
        return func(*args, **kwargs)

    monkeypatch.setattr(
        app_module, "get_gmail_watch_enabled", lambda: False, raising=False
    )
    monkeypatch.setattr(app_module.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local",
        fake_get_session_local,
    )
    monkeypatch.setattr(
        "xagent.web.services.gmail_triggers.scan_due_gmail_watch_renewals",
        fake_scan_due_gmail_watch_renewals,
    )
    monkeypatch.setattr(
        "xagent.web.services.triggers.dispatch_pending_trigger_runs",
        fake_dispatch_pending_trigger_runs,
    )

    with pytest.raises(asyncio.CancelledError):
        await app_module._run_trigger_dispatcher(
            poll_interval_seconds=60,
            batch_size=25,
        )

    assert calls == []
