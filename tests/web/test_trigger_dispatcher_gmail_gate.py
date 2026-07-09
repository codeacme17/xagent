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

    def fake_scan_due_scheduled_triggers(_db):
        return []

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
        "xagent.web.services.triggers.scan_due_scheduled_triggers",
        fake_scan_due_scheduled_triggers,
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

    # Scheduled triggers are scanned in-process every tick (no Celery needed),
    # but the Gmail watch-renewal scan stays gated off when watch is disabled.
    assert "_scan_due_scheduled_triggers_tick" in calls
    assert "_scan_due_gmail_watch_renewals_tick" not in calls


@pytest.mark.asyncio
async def test_trigger_dispatcher_survives_scheduled_scan_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scheduled-scan tick that raises must not kill the loop: it is caught
    at the loop level, logged, and the loop survives to the next tick."""

    class FakeSession:
        def close(self) -> None:
            return None

    scan_calls = {"n": 0}
    dispatch_calls = {"n": 0}

    def flaky_scan(_db):
        scan_calls["n"] += 1
        if scan_calls["n"] == 1:
            raise RuntimeError("scan blew up")
        return []

    async def fake_dispatch(_db, *, limit: int) -> int:
        dispatch_calls["n"] += 1
        # Only reached on a surviving tick; stop the loop here.
        raise asyncio.CancelledError

    monkeypatch.setattr(
        app_module, "get_gmail_watch_enabled", lambda: False, raising=False
    )
    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local",
        lambda: FakeSession,
    )
    monkeypatch.setattr(
        "xagent.web.services.triggers.scan_due_scheduled_triggers",
        flaky_scan,
    )
    monkeypatch.setattr(
        "xagent.web.services.triggers.dispatch_pending_trigger_runs",
        fake_dispatch,
    )

    with pytest.raises(asyncio.CancelledError):
        await app_module._run_trigger_dispatcher(
            poll_interval_seconds=0,
            batch_size=25,
        )

    # First tick's scan raised; the loop caught it and ran a second tick where
    # dispatch was finally reached.
    assert scan_calls["n"] == 2
    assert dispatch_calls["n"] == 1
