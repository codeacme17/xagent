"""Env and default fallback paths must carry the shared retry layer too.

``create_base_llm`` installs ``RetryWrapper``, but three supported paths build
a provider client directly and never reach it. Their only retry cover used to
be the SDK's own budget, which #2605 sets to zero -- so without a wrapper a
single transient fault fails the task on its first response.
"""

from unittest.mock import AsyncMock

import httpx
import openai
import pytest

REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


@pytest.fixture
def openai_env(monkeypatch):
    """An env-only deployment: an OpenAI key, no DB model row."""
    for name in (
        "ZHIPU_API_KEY",
        "DEEPSEEK_API_KEY",
        "ANTHROPIC_API_KEY",
        "CLAUDE_API_KEY",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")


def _budget_of(llm):
    wrapper = getattr(llm, "_retry_wrapper", None)
    return None if wrapper is None else wrapper.budget


class TestDefaultLlmCarriesTheRetryLayer:
    def test_agent_service_manager_default_is_wrapped(self, openai_env):
        """``AgentServiceManager.__init__`` builds this unconditionally."""
        from xagent.web.services.agent_service_manager import create_default_llm

        llm = create_default_llm()

        assert llm is not None
        budget = _budget_of(llm)
        assert budget is not None
        assert budget.deadline_seconds is not None

    def test_env_fallback_llm_is_wrapped(self, openai_env):
        """Reached from ``get_configured_defaults``' final fallback."""
        from xagent.web.services.llm_utils import create_llm_from_env

        llm = create_llm_from_env()

        assert llm is not None
        assert _budget_of(llm) is not None

    def test_default_vision_model_is_wrapped(self, openai_env, monkeypatch):
        """The vision branch needs its own model name to select a model."""
        monkeypatch.setenv("OPENAI_VISION_MODEL_NAME", "gpt-4o")
        from xagent.core.tools.adapters.vibe.vision_tool import (
            get_default_vision_model,
        )

        llm = get_default_vision_model()

        assert llm is not None
        assert _budget_of(llm) is not None

    async def test_a_transient_fault_is_retried_on_the_default_llm(
        self, openai_env, monkeypatch
    ):
        """The behaviour the wrapper exists for, on the real fallback path.

        Asserted on the number of provider requests rather than on a parsed
        response, so the test does not depend on a hand-built response shape.
        """
        from xagent.web.services.agent_service_manager import create_default_llm

        client = AsyncMock()
        client.chat.completions.create.side_effect = openai.APIConnectionError(
            request=REQUEST
        )
        monkeypatch.setattr(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            lambda **kwargs: client,
        )

        llm = create_default_llm()
        assert llm is not None

        with pytest.raises(Exception):
            await llm.chat([{"role": "user", "content": "hi"}])

        assert client.chat.completions.create.await_count > 1
