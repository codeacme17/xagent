"""Tests for model resolver utilities."""

from __future__ import annotations

import sqlite3
from typing import Dict

import pytest
from sqlalchemy.exc import OperationalError as SAOperationalError

from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import (
    ChatModelConfig,
    EmbeddingModelConfig,
    RerankModelConfig,
)
from xagent.core.model.rerank.base import BaseRerank
from xagent.core.tools.core.RAG_tools.core.exceptions import (
    EmbeddingAdapterError,
    RagCoreException,
)
from xagent.core.tools.core.RAG_tools.utils import model_resolver


class _StubHub:
    """Stub hub for testing."""

    def __init__(self, models: Dict[str, object]) -> None:
        self._models = models

    def list(self) -> Dict[str, object]:
        return self._models

    def load(self, model_id: str) -> object:
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} not found")
        return self._models[model_id]


class TestHubInitFailureClassification:
    """Tests for _hub_init_failure_is_benign_optional_sqlite."""

    def test_sqlite_missing_file_is_benign(self) -> None:
        exc = sqlite3.OperationalError("unable to open database file")
        assert model_resolver._hub_init_failure_is_benign_optional_sqlite(exc) is True

    def test_sqlalchemy_wrapped_sqlite_missing_is_benign(self) -> None:
        inner = sqlite3.OperationalError("unable to open database file")
        exc = SAOperationalError("SELECT 1", {}, inner)
        assert model_resolver._hub_init_failure_is_benign_optional_sqlite(exc) is True

    def test_database_locked_not_benign(self) -> None:
        exc = sqlite3.OperationalError("database is locked")
        assert model_resolver._hub_init_failure_is_benign_optional_sqlite(exc) is False

    def test_other_errors_not_benign(self) -> None:
        assert (
            model_resolver._hub_init_failure_is_benign_optional_sqlite(
                RuntimeError("connection refused")
            )
            is False
        )


class TestGetOrInitModelHub:
    """Test _get_or_init_model_hub helper function."""

    def test_get_existing_hub(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test getting existing hub."""
        stub_hub = _StubHub({})
        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", lambda: stub_hub)
        result = model_resolver._get_or_init_model_hub()
        assert result == stub_hub


class TestResolveEmbeddingAdapter:
    """Test resolve_embedding_adapter function with strict priority."""

    def test_resolve_embedding_explicit_model_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test resolving embedding with explicit model_id (highest priority)."""
        stub_hub = _StubHub(
            {
                "hub-model": EmbeddingModelConfig(
                    id="hub-model",
                    model_name="hub-model",
                    model_provider="dashscope",
                    api_key="hub-key",
                    abilities=["embedding"],
                )
            }
        )
        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", lambda: stub_hub)

        # Set env vars (should be ignored when model_id is explicit)
        monkeypatch.setenv("DASHSCOPE_EMBEDDING_MODEL", "env-model")
        monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")

        cfg, adapter = model_resolver.resolve_embedding_adapter(model_id="hub-model")
        assert cfg.id == "hub-model"
        assert isinstance(adapter, BaseEmbedding)

    def test_resolve_embedding_default_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test resolving embedding for placeholder (None) uses 'default' model in hub."""
        stub_hub = _StubHub(
            {
                "default": EmbeddingModelConfig(
                    id="default",
                    model_name="hub-embedding",
                    model_provider="dashscope",
                    api_key="hub-key",
                    abilities=["embedding"],
                )
            }
        )
        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", lambda: stub_hub)

        # Clear env vars
        for key in [
            "DASHSCOPE_EMBEDDING_MODEL",
            "DASHSCOPE_API_KEY",
            "DASHSCOPE_EMBEDDING_BASE_URL",
        ]:
            monkeypatch.delenv(key, raising=False)

        cfg, adapter = model_resolver.resolve_embedding_adapter(model_id=None)
        assert cfg.id == "default"
        assert isinstance(adapter, BaseEmbedding)

    def test_resolve_embedding_env_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test resolving embedding from env when hub fails (fallback)."""

        # Mock hub to raise exception
        def failing_hub():
            raise Exception("Hub not available")

        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", failing_hub)

        # Set env vars for fallback
        monkeypatch.setenv("DASHSCOPE_EMBEDDING_MODEL", "env-model")
        monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
        monkeypatch.setenv(
            "DASHSCOPE_EMBEDDING_BASE_URL",
            "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
        )
        monkeypatch.setenv("DASHSCOPE_EMBEDDING_DIMENSION", "1536")

        cfg, adapter = model_resolver.resolve_embedding_adapter(model_id=None)
        assert cfg.id == "env-model"
        assert cfg.dimension == 1536
        assert isinstance(adapter, BaseEmbedding)

    def test_resolve_embedding_both_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that error is raised when both hub and env fail."""

        # Mock hub to raise exception
        def failing_hub():
            raise Exception("Hub not available")

        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", failing_hub)

        # Clear env vars
        for key in [
            "DASHSCOPE_EMBEDDING_MODEL",
            "DASHSCOPE_API_KEY",
            "DASHSCOPE_EMBEDDING_BASE_URL",
        ]:
            monkeypatch.delenv(key, raising=False)

        with pytest.raises(EmbeddingAdapterError):
            model_resolver.resolve_embedding_adapter(model_id=None)


class TestResolveRerankAdapter:
    """Test resolve_rerank_adapter function with strict priority."""

    def test_resolve_rerank_explicit_model_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test resolving rerank with explicit model_id (highest priority)."""
        stub_hub = _StubHub(
            {
                "hub-rerank": RerankModelConfig(
                    id="hub-rerank",
                    model_name="hub-rerank",
                    model_provider="dashscope",
                    api_key="hub-key",
                    abilities=["rerank"],
                )
            }
        )
        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", lambda: stub_hub)

        # Set env vars (should be ignored when model_id is explicit)
        monkeypatch.setenv("DASHSCOPE_RERANK_MODEL", "env-rerank")
        monkeypatch.setenv("DASHSCOPE_RERANK_API_KEY", "env-key")

        cfg, adapter = model_resolver.resolve_rerank_adapter(model_id="hub-rerank")
        assert cfg.id == "hub-rerank"
        assert isinstance(adapter, BaseRerank)

    def test_resolve_rerank_default_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test resolving rerank for placeholder (None) uses 'default' model in hub."""
        stub_hub = _StubHub(
            {
                "default": RerankModelConfig(
                    id="default",
                    model_name="hub-rerank",
                    model_provider="dashscope",
                    api_key="hub-key",
                    abilities=["rerank"],
                )
            }
        )
        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", lambda: stub_hub)

        # Clear env vars
        for key in [
            "DASHSCOPE_RERANK_MODEL",
            "DASHSCOPE_RERANK_API_KEY",
            "DASHSCOPE_RERANK_BASE_URL",
        ]:
            monkeypatch.delenv(key, raising=False)

        cfg, adapter = model_resolver.resolve_rerank_adapter(model_id=None)
        assert cfg.id == "default"
        assert isinstance(adapter, BaseRerank)

    def test_resolve_rerank_env_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolving rerank from env when hub fails (fallback)."""

        # Mock hub to raise exception
        def failing_hub():
            raise Exception("Hub not available")

        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", failing_hub)

        # Set env vars for fallback
        monkeypatch.setenv("DASHSCOPE_RERANK_MODEL", "env-rerank")
        monkeypatch.setenv("DASHSCOPE_RERANK_API_KEY", "env-key")
        monkeypatch.setenv(
            "DASHSCOPE_RERANK_BASE_URL",
            "https://dashscope.aliyuncs.com/rerank",
        )
        monkeypatch.setenv("DASHSCOPE_RERANK_TIMEOUT", "30")

        cfg, adapter = model_resolver.resolve_rerank_adapter(model_id=None)
        assert cfg.id == "env-rerank"
        assert isinstance(adapter, BaseRerank)

    def test_resolve_rerank_both_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that error is raised when both hub and env fail."""

        # Mock hub to raise exception
        def failing_hub():
            raise Exception("Hub not available")

        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", failing_hub)

        # Clear env vars
        for key in [
            "DASHSCOPE_RERANK_MODEL",
            "DASHSCOPE_RERANK_API_KEY",
            "DASHSCOPE_RERANK_BASE_URL",
        ]:
            monkeypatch.delenv(key, raising=False)

        with pytest.raises(RagCoreException):
            model_resolver.resolve_rerank_adapter(model_id=None)


class TestResolveLLMAdapter:
    """Test resolve_llm_adapter function."""

    def test_resolve_llm_explicit_model_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test resolving LLM with explicit model_id (highest priority)."""
        stub_hub = _StubHub(
            {
                "hub-llm": ChatModelConfig(
                    id="hub-llm",
                    model_name="hub-llm",
                    model_provider="openai",
                    api_key="hub-key",
                    abilities=["chat"],
                )
            }
        )
        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", lambda: stub_hub)

        cfg, adapter = model_resolver.resolve_llm_adapter(
            model_id="hub-llm", use_langchain_adapter=False
        )
        assert cfg.id == "hub-llm"
        assert isinstance(adapter, BaseLLM)

    def test_resolve_llm_default_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test resolving LLM for placeholder (None) uses 'default' model in hub."""
        stub_hub = _StubHub(
            {
                "default": ChatModelConfig(
                    id="default",
                    model_name="hub-llm",
                    model_provider="openai",
                    api_key="hub-key",
                    abilities=["chat"],
                )
            }
        )
        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", lambda: stub_hub)

        # Clear env vars to ensure hub is used
        for key in [
            "OPENAI_API_KEY",
            "OPENAI_MODEL_NAME",
            "ZHIPU_API_KEY",
            "DEEPSEEK_API_KEY",
        ]:
            monkeypatch.delenv(key, raising=False)

        cfg, adapter = model_resolver.resolve_llm_adapter(
            model_id=None, use_langchain_adapter=False
        )
        assert cfg.id == "default"
        assert isinstance(adapter, BaseLLM)

    def test_resolve_llm_env_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolving LLM from env when hub fails (fallback)."""

        # Mock hub to raise exception
        def failing_hub():
            raise Exception("Hub not available")

        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", failing_hub)

        # Set env vars for fallback (OpenAI)
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        monkeypatch.setenv("OPENAI_MODEL_NAME", "gpt-4")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        cfg, adapter = model_resolver.resolve_llm_adapter(
            model_id=None, use_langchain_adapter=False
        )
        assert cfg.id == "gpt-4"
        assert cfg.model_provider == "openai"
        assert isinstance(adapter, BaseLLM)

    def test_resolve_llm_both_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that error is raised when both hub and env fail."""

        # Mock hub to raise exception
        def failing_hub():
            raise Exception("Hub not available")

        monkeypatch.setattr(model_resolver, "_get_or_init_model_hub", failing_hub)

        # Clear env vars
        for key in ["OPENAI_API_KEY", "ZHIPU_API_KEY", "DEEPSEEK_API_KEY"]:
            monkeypatch.delenv(key, raising=False)

        with pytest.raises(RagCoreException):
            model_resolver.resolve_llm_adapter(
                model_id=None, use_langchain_adapter=False
            )
