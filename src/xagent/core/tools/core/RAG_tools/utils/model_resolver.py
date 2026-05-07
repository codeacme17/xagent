"""Helpers to resolve embedding/rerank/llm configs with hub > env fallback priority."""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, Type, TypeVar, Union

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import Runnable

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError as SAOperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from xagent.core.model.chat.basic.adapter import create_base_llm
from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.langchain import create_base_chat_model_with_retry
from xagent.core.model.embedding.adapter import create_embedding_adapter
from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import (
    ChatModelConfig,
    EmbeddingModelConfig,
    ModelConfig,
    RerankModelConfig,
)
from xagent.core.model.rerank.adapter import create_rerank_adapter
from xagent.core.model.rerank.base import BaseRerank
from xagent.core.model.storage.db.adapter import SQLAlchemyModelHub
from xagent.core.model.storage.db.db_models import create_model_table
from xagent.core.storage.manager import get_default_db_url

from ..core.exceptions import EmbeddingAdapterError, RagCoreException

logger = logging.getLogger(__name__)

# Type variables for generic helper functions
ConfigType = TypeVar("ConfigType", bound=ModelConfig)
AdapterType = TypeVar("AdapterType")
ExceptionType = TypeVar("ExceptionType", bound=RagCoreException)

# Special placeholder values
_PLACEHOLDER_NONE = {"none", ""}


def _hub_init_failure_is_benign_optional_sqlite(exc: BaseException) -> bool:
    """Return True when the hub DB file is missing or not yet creatable.

    In those cases the model hub is an optional component and env-based config
    may still work; logging at DEBUG is enough. Permission errors and other DB
    failures should surface at WARNING with traceback.

    Args:
        exc: Exception raised while initializing SQLAlchemy / SQLite.

    Returns:
        True if failure matches a typical \"no sqlite file yet\" operational error.
    """
    msg = str(exc).lower()
    if "unable to open database file" not in msg:
        return False
    return isinstance(exc, (SAOperationalError, sqlite3.OperationalError))


def _is_placeholder_default(model_id: Optional[str]) -> bool:
    """Check if model_id is "default" (case-insensitive).

    Args:
        model_id: Model ID string to check

    Returns:
        True if model_id is "default" (case-insensitive), False otherwise
    """
    if model_id is None:
        return False
    return model_id.strip().lower() == "default"


def _is_placeholder_none(model_id: Optional[str]) -> bool:
    """Check if model_id is "none" or empty (case-insensitive).

    Args:
        model_id: Model ID string to check

    Returns:
        True if model_id is "none" or empty, False otherwise
    """
    if model_id is None:
        return True
    normalized = model_id.strip().lower()
    return normalized in _PLACEHOLDER_NONE


def _get_or_init_model_hub() -> Any:
    """Get or create model hub instance directly.

    Returns:
        Initialized model hub instance or None if database is not available
    """
    try:
        # Create database engine
        database_url = get_default_db_url()
        engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False}
            if "sqlite" in database_url
            else {},
        )
        # Create session factory
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        # Create base model class
        Base = declarative_base()
        Model = create_model_table(Base)
        db = SessionLocal()
        Base.metadata.create_all(engine)
        return SQLAlchemyModelHub(db, Model)
    except Exception as e:
        if _hub_init_failure_is_benign_optional_sqlite(e):
            logger.debug(
                "Model hub SQLite not available yet (optional component): %s",
                e,
            )
        else:
            logger.warning(
                "Model hub database initialization failed; hub-backed model "
                "resolution is disabled until this is fixed. "
                "If you rely on env-only configuration, you can ignore this. "
                "Otherwise check DB URL, permissions, and connectivity: %s",
                e,
                exc_info=True,
            )
        return None


def _load_model_from_hub(
    model_id: str,
    config_type: Type[ConfigType],
    exception_cls: Type[ExceptionType],
) -> ConfigType:
    """Load model configuration from hub with consistent error handling.

    Args:
        model_id: Model identifier to load
        config_type: Expected configuration type class
        exception_cls: Exception class to raise on errors

    Returns:
        Loaded model configuration of the specified type

    Raises:
        exception_cls: If model loading or type validation fails
    """
    # Get or initialize hub
    hub = _get_or_init_model_hub()
    if hub is None:
        raise exception_cls(
            f"Failed to load {config_type.__name__}: model hub database not available",
            details={"model_id": model_id, "error": "Database not available"},
        )

    # Load model configuration first
    try:
        cfg = hub.load(model_id)
    except ValueError as exc:
        # SQLAlchemyModelHub may raise ValueError (model not found or unknown category)
        raise exception_cls(
            f"Failed to load {config_type.__name__} from model hub",
            details={
                "model_id": model_id,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        ) from exc

    # Validate configuration type
    if not isinstance(cfg, config_type):
        raise exception_cls(
            f"Model '{model_id}' is not a {config_type.__name__}",
            details={"model_id": model_id, "actual_type": type(cfg).__name__},
        )

    return cfg


def _create_adapter_safe(
    cfg: ConfigType,
    adapter_factory: Callable[..., AdapterType],
    exception_cls: Type[ExceptionType],
    context: str = "",
    **adapter_kwargs: Any,
) -> AdapterType:
    """Create adapter with consistent error handling.

    Args:
        cfg: Model configuration
        adapter_factory: Function to create adapter from config (may accept additional kwargs)
        exception_cls: Exception class to raise on errors
        context: Additional context for error messages
        **adapter_kwargs: Additional keyword arguments passed to adapter_factory

    Returns:
        Created adapter instance

    Raises:
        exception_cls: If adapter creation fails
    """
    try:
        return adapter_factory(cfg, **adapter_kwargs)
    except (ImportError, ValueError, TypeError) as exc:
        # Adapter creation failed (dependency/configuration issue)
        raise exception_cls(
            f"Failed to create adapter{context}",
            details={"error": str(exc), "error_type": type(exc).__name__},
        ) from exc


def _create_llm_adapter_factory(
    use_langchain_adapter: bool,
) -> Callable[[ChatModelConfig], Union[BaseLLM, "Runnable"]]:
    """Create LLM adapter factory function based on adapter type preference.

    Args:
        use_langchain_adapter: Whether to use LangChain adapter

    Returns:
        Adapter factory function that takes ChatModelConfig and returns adapter
    """

    def adapter_factory(cfg: ChatModelConfig) -> Union[BaseLLM, "Runnable"]:
        if use_langchain_adapter:
            return create_base_chat_model_with_retry(cfg, None)
        else:
            return create_base_llm(cfg)

    return adapter_factory


def resolve_embedding_from_env(
    model_id: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    dimension: Optional[int] = None,
) -> Optional[EmbeddingModelConfig]:
    """Build embedding config from env (DashScope-compatible). Parameters have priority over env vars."""
    # Model name must be specific to embedding service, no fallback to generic DASHSCOPE_MODEL
    # Priority: parameter > env var
    model = model_id or os.getenv("DASHSCOPE_EMBEDDING_MODEL")
    key = (
        api_key
        or os.getenv("DASHSCOPE_EMBEDDING_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
    )
    # URL must be specific to embedding service, no fallback to generic DASHSCOPE_BASE_URL
    # Priority: parameter > env var
    base = base_url or os.getenv("DASHSCOPE_EMBEDDING_BASE_URL")
    timeout_val = (
        (timeout_sec if timeout_sec is not None else None)
        or os.getenv("DASHSCOPE_EMBEDDING_TIMEOUT")
        or os.getenv("DASHSCOPE_TIMEOUT")
    )
    timeout = float(timeout_val) if timeout_val else 180.0
    # Priority: parameter > env var
    dim_val = os.getenv("DASHSCOPE_EMBEDDING_DIMENSION")
    dim = dimension if dimension is not None else (int(dim_val) if dim_val else None)

    if model and key:
        return EmbeddingModelConfig(
            id=model,
            model_name=model,
            api_key=key,
            base_url=base,
            timeout=timeout,
            dimension=dim,
            abilities=["embedding"],
        )
    return None


def resolve_rerank_from_env(
    model_id: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_sec: Optional[float] = None,
) -> Optional[RerankModelConfig]:
    """Build rerank config from env (DashScope-compatible). Parameters have priority over env vars."""
    # Model name must be specific to rerank service, no fallback to generic DASHSCOPE_MODEL
    # Priority: parameter > env var
    model = model_id or os.getenv("DASHSCOPE_RERANK_MODEL")
    key = (
        api_key
        or os.getenv("DASHSCOPE_RERANK_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
    )
    # URL must be specific to rerank service, no fallback to generic DASHSCOPE_BASE_URL
    # Priority: parameter > env var
    base = base_url or os.getenv("DASHSCOPE_RERANK_BASE_URL")
    timeout_val = (
        (timeout_sec if timeout_sec is not None else None)
        or os.getenv("DASHSCOPE_RERANK_TIMEOUT")
        or os.getenv("DASHSCOPE_TIMEOUT")
    )
    timeout = float(timeout_val) if timeout_val else 180.0

    if model and key:
        return RerankModelConfig(
            id=model,
            model_name=model,
            api_key=key,
            base_url=base,
            timeout=timeout,
            abilities=["rerank"],
        )
    return None


def _resolve_adapter_generic(
    model_id: Optional[str],
    config_type: Type[ConfigType],
    exception_type: Type[ExceptionType],
    env_prefix: str,
    model_type_name: str,
    adapter_factory: Callable[..., AdapterType],
    env_resolver: Callable[..., Optional[ConfigType]],
    env_kwargs: dict[str, Any],
    adapter_kwargs: Optional[dict[str, Any]] = None,
) -> Tuple[ConfigType, AdapterType]:
    """Generic helper function for resolving model config/adapter with hub > env fallback priority.

    Args:
        model_id: Specific model ID to load from hub, or None for auto-selection
        config_type: Expected configuration type class
        exception_type: Exception class to raise on errors
        env_prefix: Environment variable prefix for hub auto-selection error messages
        model_type_name: Model type name for logging (e.g., "embedding", "rerank", "LLM")
        adapter_factory: Function to create adapter from config (may accept additional kwargs)
        env_resolver: Function to resolve config from environment variables
        env_kwargs: Keyword arguments passed to env_resolver
        adapter_kwargs: Optional keyword arguments passed to adapter_factory

    Returns:
        Tuple of (model config, adapter instance)

    Raises:
        exception_type: If no model available from hub or environment
    """
    adapter_kwargs = adapter_kwargs or {}

    # Try to get hub, but fallback to env if hub is not available
    hub = None
    try:
        hub = _get_or_init_model_hub()
    except Exception as hub_error:
        # Hub not available, will fallback to environment
        logger.warning(
            "Model hub not available for %s: %s. Falling back to environment configuration.",
            model_type_name,
            hub_error,
        )
        hub = None

    # Strategy 1: "default" placeholder
    # Priority: "default" ID in Hub -> Environment (no auto-selection)
    if _is_placeholder_default(model_id):
        if hub is not None:
            try:
                cfg = hub.load("default")
                if isinstance(cfg, config_type):
                    adapter = _create_adapter_safe(
                        cfg, adapter_factory, exception_type, **adapter_kwargs
                    )
                    return cfg, adapter
                else:
                    # Found "default" but wrong type
                    raise exception_type(
                        f"Model 'default' exists but is not a {config_type.__name__}",
                        details={
                            "model_id": "default",
                            "actual_type": type(cfg).__name__,
                        },
                    )
            except ValueError:
                # "default" model not found in hub, fallback to env
                logger.warning(
                    "Model 'default' not found in hub for %s, falling back to environment",
                    model_type_name,
                )

        # Fallback to Environment
        env_cfg = env_resolver(**env_kwargs)
        if env_cfg:
            adapter = _create_adapter_safe(
                env_cfg,
                adapter_factory,
                exception_type,
                " from environment configuration",
                **adapter_kwargs,
            )
            return env_cfg, adapter

        # All failed
        raise exception_type(
            f"No {model_type_name} model available: 'default' not found in hub and no environment configuration"
        )

    # Strategy 2: "none" or empty placeholder
    # Priority: "default" ID in Hub -> Environment (same as "default", no auto-selection)
    if _is_placeholder_none(model_id):
        if hub is not None:
            try:
                cfg = hub.load("default")
                if isinstance(cfg, config_type):
                    adapter = _create_adapter_safe(
                        cfg, adapter_factory, exception_type, **adapter_kwargs
                    )
                    return cfg, adapter
                else:
                    # Found "default" but wrong type
                    raise exception_type(
                        f"Model 'default' exists but is not a {config_type.__name__}",
                        details={
                            "model_id": "default",
                            "actual_type": type(cfg).__name__,
                        },
                    )
            except ValueError:
                # "default" model not found in hub, fallback to env
                logger.warning(
                    "Model 'default' not found in hub for %s, falling back to environment",
                    model_type_name,
                )

        # Fallback to Environment
        env_cfg = env_resolver(**env_kwargs)
        if env_cfg:
            adapter = _create_adapter_safe(
                env_cfg,
                adapter_factory,
                exception_type,
                " from environment configuration",
                **adapter_kwargs,
            )
            return env_cfg, adapter

        # All failed
        raise exception_type(
            f"No {model_type_name} model available: 'default' not found in hub and no environment configuration"
        )

    # Strategy 3: Explicit ID Intent
    # Priority: Explicit ID in Hub -> Environment (fallback)
    # Note: We do NOT auto-select here. If user asks for "my-model", we don't give them "other-model".
    else:
        # 3.1 Try Explicit ID in Hub
        if hub is not None:
            try:
                # We use _load_model_from_hub here mainly for its error wrapping/handling logic,
                # but we can also just use hub.load directly since we have hub.
                # Using try-except block to catch load errors.
                cfg = hub.load(model_id)
                if isinstance(cfg, config_type):
                    adapter = _create_adapter_safe(
                        cfg, adapter_factory, exception_type, **adapter_kwargs
                    )
                    return cfg, adapter
                else:
                    # Found model but wrong type
                    raise exception_type(
                        f"Model '{model_id}' exists but is not a {config_type.__name__}",
                        details={
                            "model_id": model_id,
                            "actual_type": type(cfg).__name__,
                        },
                    )
            except ValueError as hub_error:
                # Explicit ID not found in Hub
                logger.warning(
                    "Model '%s' not found in hub for %s: %s. Falling back to environment configuration.",
                    model_id,
                    model_type_name,
                    hub_error,
                )

        # 3.2 Fallback to Environment
        # Logic: If the explicit ID isn't in the hub, maybe the env vars are set up
        # to provide a model that the user *intends* to use, or maybe the code that called this
        # passed a model_id that is actually meant to be used with env vars (though less likely in this design).
        # We allow env fallback to be safe, but we don't auto-select a random model from Hub.
        env_cfg = env_resolver(**env_kwargs)
        if env_cfg:
            adapter = _create_adapter_safe(
                env_cfg,
                adapter_factory,
                exception_type,
                " from environment configuration",
                **adapter_kwargs,
            )
            return env_cfg, adapter

        # All failed
        raise exception_type(
            f"Model '{model_id}' not found in hub and no environment configuration available for {model_type_name}."
        )


def resolve_embedding_adapter(
    model_id: Optional[str],
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    dimension: Optional[int] = None,
) -> Tuple[EmbeddingModelConfig, BaseEmbedding]:
    """Resolve embedding config/adapter with priority: explicit model_id > hub single > env fallback."""
    return _resolve_adapter_generic(
        model_id=model_id,
        config_type=EmbeddingModelConfig,
        exception_type=EmbeddingAdapterError,
        env_prefix="DASHSCOPE_EMBEDDING_",
        model_type_name="embedding",
        adapter_factory=create_embedding_adapter,
        env_resolver=resolve_embedding_from_env,
        env_kwargs={
            "api_key": api_key,
            "base_url": base_url,
            "timeout_sec": timeout_sec,
            "dimension": dimension,
        },
        adapter_kwargs={},
    )


def resolve_rerank_adapter(
    model_id: Optional[str],
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_sec: Optional[float] = None,
) -> Tuple[RerankModelConfig, BaseRerank]:
    """Resolve rerank config/adapter with priority: explicit model_id > hub single > env fallback."""
    return _resolve_adapter_generic(
        model_id=model_id,
        config_type=RerankModelConfig,
        exception_type=RagCoreException,
        env_prefix="DASHSCOPE_RERANK_",
        model_type_name="rerank",
        adapter_factory=create_rerank_adapter,
        env_resolver=resolve_rerank_from_env,
        env_kwargs={
            "api_key": api_key,
            "base_url": base_url,
            "timeout_sec": timeout_sec,
        },
        adapter_kwargs={},
    )


def _create_llm_config_from_provider_env(
    env_prefix: str,
    provider_name: str,
    default_model: str,
    *,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Optional[ChatModelConfig]:
    """Create LLM config from environment variables for a specific provider.

    Args:
        env_prefix: Environment variable prefix (e.g., "OPENAI", "ZHIPU")
        provider_name: Provider name for ChatModelConfig (e.g., "openai", "zhipu")
        default_model: Default model name if not specified
        model_name: Optional model name override
        api_key: Optional API key override
        base_url: Optional base URL override
        timeout_sec: Optional timeout override
        temperature: Optional temperature override
        max_tokens: Optional max tokens override

    Returns:
        ChatModelConfig if provider is configured, None otherwise
    """
    provider_key = os.getenv(f"{env_prefix}_API_KEY")
    if not provider_key:
        return None

    try:
        final_model_name = model_name or os.getenv(
            f"{env_prefix}_MODEL_NAME", default_model
        )
        final_base_url = base_url or os.getenv(f"{env_prefix}_BASE_URL")
        final_temperature = (
            temperature
            if temperature is not None
            else float(os.getenv(f"{env_prefix}_TEMPERATURE", "0.7"))
        )
        final_max_tokens = (
            max_tokens
            if max_tokens is not None
            else int(os.getenv(f"{env_prefix}_MAX_TOKENS", "4096"))
        )
        final_timeout = (
            timeout_sec
            if timeout_sec is not None
            else float(os.getenv(f"{env_prefix}_TIMEOUT", "180.0"))
        )

        return ChatModelConfig(
            id=final_model_name,
            model_name=final_model_name,
            model_provider=provider_name,
            api_key=api_key or provider_key,
            base_url=final_base_url,
            default_temperature=final_temperature,
            default_max_tokens=final_max_tokens,
            timeout=final_timeout,
            abilities=["chat"],
        )
    except (ValueError, TypeError) as e:
        logger.warning("Failed to create %s config from env: %s", provider_name, e)
        return None


def _create_llm_from_env(
    model_name: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Optional[ChatModelConfig]:
    """Build LLM config from environment variables as fallback."""
    # Try OpenAI first
    openai_config = _create_llm_config_from_provider_env(
        "OPENAI",
        "openai",
        "gpt-4",
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        timeout_sec=timeout_sec,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if openai_config:
        return openai_config

    # Try Zhipu
    zhipu_config = _create_llm_config_from_provider_env(
        "ZHIPU",
        "zhipu",
        "glm-4",
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        timeout_sec=timeout_sec,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if zhipu_config:
        return zhipu_config

    return None


def resolve_llm_adapter(
    model_id: Optional[str] = None,
    *,
    use_langchain_adapter: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Tuple[ChatModelConfig, Union[BaseLLM, "BaseChatModel", "Runnable"]]:
    """Resolve LLM config/adapter with priority: explicit model_id > hub single > env fallback.

    Priority order:
        1. If model_id is provided: load from hub directly
        2. If no model_id: try hub auto-selection (hub priority)
        3. If hub fails: fallback to environment variables

    Args:
        model_id: Specific model ID to load from hub
        use_langchain_adapter: Whether to use LangChain adapter (for LangGraph) or BaseLLM adapter
        api_key: API key override
        base_url: Base URL override
        timeout_sec: Timeout override
        temperature: Temperature override
        max_tokens: Max tokens override

    Returns:
        Tuple of (ChatModelConfig, LLM adapter instance)
    """
    return _resolve_adapter_generic(
        model_id=model_id,
        config_type=ChatModelConfig,
        exception_type=RagCoreException,
        env_prefix="OPENAI_",
        model_type_name="LLM",
        adapter_factory=_create_llm_adapter_factory(use_langchain_adapter),
        env_resolver=_create_llm_from_env,
        env_kwargs={
            "model_name": model_id,
            "api_key": api_key,
            "base_url": base_url,
            "timeout_sec": timeout_sec,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        adapter_kwargs={},
    )
