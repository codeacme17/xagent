import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional

from ...providers import is_placeholder_api_key
from .base import StreamChunk
from .openai import OpenAILLM

logger = logging.getLogger(__name__)

DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_SUPPORTED_MODELS = (
    "deepseek-v4-flash",
    "deepseek-v4-pro",
)


def resolve_deepseek_api_key(api_key: Optional[str] = None) -> str:
    """Resolve DeepSeek API key with OpenAI-compatible fallback."""

    resolved_api_key = api_key.strip() if isinstance(api_key, str) else api_key
    if is_placeholder_api_key(resolved_api_key):
        resolved_api_key = None

    if resolved_api_key is None:
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        deepseek_api_key = (
            deepseek_api_key.strip()
            if isinstance(deepseek_api_key, str)
            else deepseek_api_key
        )
        if not is_placeholder_api_key(deepseek_api_key):
            resolved_api_key = deepseek_api_key

    if resolved_api_key is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_api_key = (
            openai_api_key.strip()
            if isinstance(openai_api_key, str)
            else openai_api_key
        )
        if not is_placeholder_api_key(openai_api_key):
            resolved_api_key = openai_api_key

    return resolved_api_key or ""


class DeepSeekLLM(OpenAILLM):
    """DeepSeek v4 client using the OpenAI SDK with DeepSeek-specific options."""

    def __init__(
        self,
        model_name: str = "deepseek-v4-flash",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
        timeout: float = 180.0,
        abilities: Optional[List[str]] = None,
        timeout_config: Optional[Any] = None,
    ):
        self._validate_model_name(model_name)
        resolved_abilities = abilities or ["chat", "tool_calling", "thinking_mode"]
        resolved_api_key = resolve_deepseek_api_key(api_key)

        super().__init__(
            model_name=model_name,
            base_url=(
                base_url or os.getenv("DEEPSEEK_BASE_URL") or DEEPSEEK_DEFAULT_BASE_URL
            ),
            api_key=resolved_api_key,
            default_temperature=default_temperature,
            default_max_tokens=default_max_tokens,
            timeout=timeout,
            abilities=resolved_abilities,
            timeout_config=timeout_config,
        )

    @staticmethod
    def _validate_model_name(model_name: str) -> None:
        if model_name not in DEEPSEEK_SUPPORTED_MODELS:
            supported = ", ".join(DEEPSEEK_SUPPORTED_MODELS)
            raise ValueError(
                f"Unsupported DeepSeek model '{model_name}'. Supported models: {supported}"
            )

    @property
    def supports_enable_thinking_param(self) -> bool:
        """DeepSeek uses a `thinking` payload instead of `enable_thinking`."""
        return False

    @property
    def supports_json_schema_response_format(self) -> bool:
        """DeepSeek supports JSON object mode, not OpenAI json_schema mode."""
        return False

    @property
    def supports_json_object_response_format(self) -> bool:
        """DeepSeek supports response_format={"type": "json_object"}."""
        return True

    def _build_deepseek_extra_body(
        self,
        *,
        tools: Optional[List[Dict[str, Any]]],
        response_format: Optional[Dict[str, Any]],
        output_config: Optional[Dict[str, Any]],
        thinking: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        extra_body: Dict[str, Any] = {}

        if thinking is not None:
            if thinking.get("type") == "enabled" or thinking.get("enable", False):
                extra_body["thinking"] = {"type": "enabled"}
            else:
                extra_body["thinking"] = {"type": "disabled"}
        elif tools or response_format or output_config:
            extra_body["thinking"] = {"type": "disabled"}

        return extra_body

    def _prepare_deepseek_kwargs(
        self,
        *,
        tools: Optional[List[Dict[str, Any]]],
        response_format: Optional[Dict[str, Any]],
        output_config: Optional[Dict[str, Any]],
        thinking: Optional[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        updated_kwargs = dict(kwargs)
        caller_extra_body = dict(updated_kwargs.pop("extra_body", {}) or {})
        provider_extra_body = self._build_deepseek_extra_body(
            tools=tools,
            response_format=response_format,
            output_config=output_config,
            thinking=thinking,
        )
        extra_body = {**caller_extra_body, **provider_extra_body}

        if "reasoning_effort" not in updated_kwargs:
            reasoning_effort = os.getenv("DEEPSEEK_REASONING_EFFORT")
            if reasoning_effort:
                updated_kwargs["reasoning_effort"] = reasoning_effort

        return extra_body, updated_kwargs

    def _normalize_response_format(
        self,
        response_format: Optional[Dict[str, Any]],
        output_config: Optional[Dict[str, Any]],
    ) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if response_format and response_format.get("type") == "json_schema":
            logger.warning(
                "DeepSeek does not support json_schema response_format; using json_object instead."
            )
            return {"type": "json_object"}, output_config

        format_config = (output_config or {}).get("format") or {}
        if not response_format and format_config.get("type") == "json_schema":
            logger.warning(
                "DeepSeek does not support json_schema output_config; using json_object instead."
            )
            return {"type": "json_object"}, None

        return response_format, output_config

    def _disable_thinking_extra_body(
        self, extra_body: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        updated_extra_body = dict(extra_body or {})
        updated_extra_body["thinking"] = {"type": "disabled"}
        updated_extra_body.pop("enable_thinking", None)
        return updated_extra_body

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        output_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        response_format, output_config = self._normalize_response_format(
            response_format=response_format,
            output_config=output_config,
        )
        extra_body, kwargs = self._prepare_deepseek_kwargs(
            tools=tools,
            response_format=response_format,
            output_config=output_config,
            thinking=thinking,
            kwargs=kwargs,
        )
        return await super().chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking=None,
            output_config=output_config,
            extra_body=extra_body,
            **kwargs,
        )

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        output_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        response_format, output_config = self._normalize_response_format(
            response_format=response_format,
            output_config=output_config,
        )
        extra_body, kwargs = self._prepare_deepseek_kwargs(
            tools=tools,
            response_format=response_format,
            output_config=output_config,
            thinking=thinking,
            kwargs=kwargs,
        )
        async for chunk in super().stream_chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking=None,
            output_config=output_config,
            extra_body=extra_body,
            **kwargs,
        ):
            yield chunk

    @staticmethod
    async def list_available_models(
        api_key: str, base_url: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        _ = api_key, base_url
        return [
            {
                "id": "deepseek-v4-flash",
                "created": 0,
                "owned_by": "deepseek",
                "abilities": ["chat", "tool_calling", "thinking_mode"],
            },
            {
                "id": "deepseek-v4-pro",
                "created": 0,
                "owned_by": "deepseek",
                "abilities": ["chat", "tool_calling", "thinking_mode"],
            },
        ]
