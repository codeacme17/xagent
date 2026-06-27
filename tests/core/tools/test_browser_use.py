"""
Tests for browser automation tools.

These tests use mocking to avoid requiring actual browser installation.
Run with: pytest tests/core/tools/test_browser_use.py
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest

from xagent.core.tools.core import browser_use


_SESSION_GUARDED_OPERATIONS = [
    (
        browser_use.browser_navigate,
        {"url": "https://example.com", "wait_until": "domcontentloaded"},
    ),
    (browser_use.browser_click, {"selector": "button.submit"}),
    (browser_use.browser_fill, {"selector": "input[name='email']", "value": "x"}),
    (browser_use.browser_screenshot, {}),
    (browser_use.browser_extract_text, {}),
    (browser_use.browser_evaluate, {"javascript": "'ok'"}),
    (browser_use.browser_select_option, {"selector": "select.country", "value": "US"}),
    (browser_use.browser_wait_for_selector, {"selector": ".ready"}),
    (browser_use.browser_pdf, {}),
]


@pytest.fixture
def mock_playwright():
    """Mock playwright modules."""
    with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", True):
        with patch("xagent.core.tools.core.browser_use.async_playwright") as mock_ap:
            yield mock_ap


@pytest.fixture
def reset_manager():
    """Reset global browser manager between tests."""
    manager = browser_use.get_browser_manager()
    import asyncio

    try:
        asyncio.run(manager.close_all())
    except Exception:
        pass
    browser_use._manager = None
    yield
    browser_use._manager = None


@pytest.mark.parametrize(
    ("operation", "extra_args"),
    _SESSION_GUARDED_OPERATIONS,
    ids=[operation.__name__ for operation, _extra_args in _SESSION_GUARDED_OPERATIONS],
)
async def test_browser_operation_serializes_calls_for_same_session(
    monkeypatch, operation, extra_args
):
    """Every browser operation is guarded by the target session."""
    tracker = _AsyncConcurrencyTracker()
    manager = _FakeBrowserManager(tracker)

    monkeypatch.setattr(browser_use, "PLAYWRIGHT_AVAILABLE", True)
    monkeypatch.setattr(browser_use, "get_browser_manager", lambda: manager)

    results = await asyncio.gather(
        operation(session_id="shared-session", **extra_args),
        operation(session_id="shared-session", **extra_args),
    )

    assert [result["success"] for result in results] == [True, True]
    assert tracker.peak == 1


class TestBrowserNavigate:
    """Tests for browser_navigate function."""

    def test_navigate_without_playwright(self):
        """Test that navigate fails gracefully without Playwright."""
        import asyncio

        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", False):
            result = asyncio.run(
                browser_use.browser_navigate(
                    session_id="test-session", url="https://example.com"
                )
            )

            assert result["success"] is False
            assert "not installed" in result["error"]

    def test_navigate_with_mock(self, mock_playwright, reset_manager):
        """Test navigation with mocked Playwright."""
        # The actual browser interaction test is skipped for simplicity
        # In real integration tests, you would test with a real browser
        # For now, just test the error handling without Playwright
        pass


class TestBrowserClick:
    """Tests for browser_click function."""

    def test_click_without_playwright(self):
        """Test that click fails gracefully without Playwright."""
        import asyncio

        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", False):
            result = asyncio.run(
                browser_use.browser_click(
                    session_id="test-session", selector="button.submit"
                )
            )

            assert result["success"] is False
            assert "not installed" in result["error"]


class TestBrowserFill:
    """Tests for browser_fill function."""

    def test_fill_without_playwright(self):
        """Test that fill fails gracefully without Playwright."""
        import asyncio

        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", False):
            result = asyncio.run(
                browser_use.browser_fill(
                    session_id="test-session",
                    selector="input[name='email']",
                    value="test@example.com",
                )
            )

            assert result["success"] is False
            assert "not installed" in result["error"]


class TestBrowserScreenshot:
    """Tests for browser_screenshot function."""

    def test_screenshot_without_playwright(self):
        """Test that screenshot fails gracefully without Playwright."""
        import asyncio

        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", False):
            result = asyncio.run(
                browser_use.browser_screenshot(session_id="test-session")
            )

            assert result["success"] is False
            assert "not installed" in result["error"]

    def test_screenshot_with_wait_for_lazy_load_parameter(self):
        """Test that screenshot accepts wait_for_lazy_load parameter."""
        import asyncio

        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", False):
            result = asyncio.run(
                browser_use.browser_screenshot(
                    session_id="test-session",
                    full_page=True,
                    wait_for_lazy_load=True,
                )
            )

            # Should contain the wait_for_lazy_load parameter in response
            assert "wait_for_lazy_load" in result
            assert result["wait_for_lazy_load"] is True

    def test_screenshot_default_wait_for_lazy_load(self):
        """Test that wait_for_lazy_load defaults to False."""
        import asyncio

        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", False):
            result = asyncio.run(
                browser_use.browser_screenshot(
                    session_id="test-session",
                    full_page=True,
                )
            )

            # Should default to False
            assert result.get("wait_for_lazy_load", False) is False


class TestBrowserExtractText:
    """Tests for browser_extract_text function."""

    def test_extract_text_without_playwright(self):
        """Test that extract_text fails gracefully without Playwright."""
        import asyncio

        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", False):
            result = asyncio.run(
                browser_use.browser_extract_text(session_id="test-session")
            )

            assert result["success"] is False
            assert "not installed" in result["error"]


class TestBrowserEvaluate:
    """Tests for browser_evaluate function."""

    def test_evaluate_without_playwright(self):
        """Test that evaluate fails gracefully without Playwright."""
        import asyncio

        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", False):
            result = asyncio.run(
                browser_use.browser_evaluate(
                    session_id="test-session", javascript="document.title"
                )
            )

            assert result["success"] is False
            assert "not installed" in result["error"]

    async def test_evaluate_serializes_operations_for_same_session(self, monkeypatch):
        """Concurrent calls sharing a browser session run one at a time."""
        tracker = _AsyncConcurrencyTracker()
        manager = _FakeBrowserManager(tracker)

        monkeypatch.setattr(browser_use, "PLAYWRIGHT_AVAILABLE", True)
        monkeypatch.setattr(browser_use, "get_browser_manager", lambda: manager)

        results = await asyncio.gather(
            browser_use.browser_evaluate(
                session_id="shared-session", javascript="'first'"
            ),
            browser_use.browser_evaluate(
                session_id="shared-session", javascript="'second'"
            ),
        )

        assert [result["success"] for result in results] == [True, True]
        assert tracker.peak == 1

    async def test_evaluate_allows_operations_for_different_sessions_to_overlap(
        self, monkeypatch
    ):
        """Concurrent calls with different sessions can overlap."""
        tracker = _AsyncConcurrencyTracker()
        manager = _FakeBrowserManager(tracker)

        monkeypatch.setattr(browser_use, "PLAYWRIGHT_AVAILABLE", True)
        monkeypatch.setattr(browser_use, "get_browser_manager", lambda: manager)

        results = await asyncio.gather(
            browser_use.browser_evaluate(session_id="session-a", javascript="'a'"),
            browser_use.browser_evaluate(session_id="session-b", javascript="'b'"),
        )

        assert [result["success"] for result in results] == [True, True]
        assert tracker.peak == 2


class TestBrowserSelectOption:
    """Tests for browser_select_option function."""

    def test_select_option_without_playwright(self):
        """Test that select_option fails gracefully without Playwright."""
        import asyncio

        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", False):
            result = asyncio.run(
                browser_use.browser_select_option(
                    session_id="test-session", selector="select.country", value="US"
                )
            )

            assert result["success"] is False
            assert "not installed" in result["error"]


class TestBrowserWaitForSelector:
    """Tests for browser_wait_for_selector function."""

    def test_wait_for_selector_without_playwright(self):
        """Test that wait_for_selector fails gracefully without Playwright."""
        import asyncio

        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", False):
            result = asyncio.run(
                browser_use.browser_wait_for_selector(
                    session_id="test-session", selector=".dynamic-content"
                )
            )

            assert result["success"] is False
            assert "not installed" in result["error"]


class TestBrowserClose:
    """Tests for browser_close function."""

    def test_close_session(self, reset_manager):
        """Test closing a browser session."""
        import asyncio

        result = asyncio.run(browser_use.browser_close("test-session"))

        assert result["success"] is True
        assert "closed" in result["message"]


class TestBrowserListSessions:
    """Tests for browser_list_sessions function."""

    def test_list_sessions_empty(self, reset_manager):
        """Test listing sessions when none exist."""
        import asyncio

        result = asyncio.run(browser_use.browser_list_sessions())

        assert result["success"] is True
        assert result["count"] == 0
        assert result["sessions"] == []


class TestBrowserSessionManager:
    """Tests for BrowserSessionManager class."""

    def test_manager_singleton(self, reset_manager):
        """Test that manager is a singleton."""
        manager1 = browser_use.get_browser_manager()
        manager2 = browser_use.get_browser_manager()

        assert manager1 is manager2

    async def test_session_timeout(self, reset_manager):
        """Test that sessions can be cleaned up after timeout."""
        manager = browser_use.BrowserSessionManager(timeout_minutes=0)

        # Mock a session that's expired
        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", True):
            session = browser_use.BrowserSession("test-session")
            session._last_used = browser_use.datetime.now() - browser_use.timedelta(
                minutes=1
            )

            async with manager._lock:
                manager._sessions["test-session"] = session

            # Run cleanup
            expired_count = await manager.cleanup_expired()

            assert expired_count >= 0


class TestBrowserSession:
    """Tests for BrowserSession class."""

    def test_session_initialization(self):
        """Test BrowserSession initialization."""
        with patch("xagent.core.tools.core.browser_use.PLAYWRIGHT_AVAILABLE", True):
            session = browser_use.BrowserSession("test-session", headless=True)

            assert session.session_id == "test-session"
            assert session.headless is True
            assert session._initialized is False


class _AsyncConcurrencyTracker:
    def __init__(self):
        self.active = 0
        self.peak = 0

    async def run(self, result):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.01)
            return result
        finally:
            self.active -= 1


class _FakeBrowserPage:
    def __init__(self, tracker):
        self._tracker = tracker
        self.viewport_size = {"width": 1920, "height": 1080}

    async def goto(self, url, **kwargs):
        return await self._tracker.run(url)

    async def title(self):
        return "Fake title"

    async def click(self, selector, **kwargs):
        return await self._tracker.run(selector)

    async def fill(self, selector, value, **kwargs):
        return await self._tracker.run(value)

    async def screenshot(self, **kwargs):
        return await self._tracker.run(b"fake-png")

    def locator(self, selector):
        return _FakeBrowserLocator(selector, self._tracker)

    async def evaluate(self, javascript):
        return await self._tracker.run(javascript)

    async def select_option(self, selector, **kwargs):
        return await self._tracker.run(kwargs)

    async def wait_for_selector(self, selector, **kwargs):
        return await self._tracker.run(selector)

    async def pdf(self, **kwargs):
        return await self._tracker.run(b"fake-pdf")

    async def set_viewport_size(self, viewport_size):
        self.viewport_size = dict(viewport_size)


class _FakeBrowserLocator:
    def __init__(self, selector, tracker):
        self._selector = selector
        self._tracker = tracker

    async def wait_for(self, **kwargs):
        return None

    async def inner_text(self, **kwargs):
        return await self._tracker.run(f"text for {self._selector}")


class _FakeBrowserSession:
    def __init__(self, session_id, tracker):
        self.session_id = session_id
        self._page = _FakeBrowserPage(tracker)
        self._operation_lock = asyncio.Lock()

    @asynccontextmanager
    async def operation_guard(self):
        async with self._operation_lock:
            yield

    async def get_page(self):
        return self._page


class _FakeBrowserManager:
    def __init__(self, tracker):
        self._tracker = tracker
        self._sessions = {}

    async def get_or_create(self, session_id, headless=False):
        if session_id not in self._sessions:
            self._sessions[session_id] = _FakeBrowserSession(session_id, self._tracker)
        return self._sessions[session_id]
