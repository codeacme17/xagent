"""Fixed client-visible fallbacks for incidental server failures."""

from enum import StrEnum

from ...core.tools.adapters.vibe.config import RequiredMCPUnavailableError

CLIENT_SAFE_VALIDATION_ERROR = "The message could not be processed. Please try again."

# Task audiences did not necessarily initiate the failing operation, so a
# task-level failure uses neutral wording instead of the validation fallback.
CLIENT_SAFE_TASK_FAILURE = "Task execution failed."
CLIENT_SAFE_GUIDANCE_IN_PROGRESS = (
    "A previous guidance message is still being applied. Please wait for it to finish."
)


class ClientErrorCode(StrEnum):
    """Stable identifiers clients may localize without trusting server prose."""

    MESSAGE_PROCESSING_FAILED = "message_processing_failed"
    TASK_EXECUTION_FAILED = "task_execution_failed"
    GUIDANCE_IN_PROGRESS = "guidance_in_progress"


def client_error_message(code: ClientErrorCode) -> str:
    """Return the fixed safe fallback for a stable client error code."""

    return {
        ClientErrorCode.MESSAGE_PROCESSING_FAILED: CLIENT_SAFE_VALIDATION_ERROR,
        ClientErrorCode.TASK_EXECUTION_FAILED: CLIENT_SAFE_TASK_FAILURE,
        ClientErrorCode.GUIDANCE_IN_PROGRESS: CLIENT_SAFE_GUIDANCE_IN_PROGRESS,
    }[code]


def required_mcp_unavailable_client_message(
    error: BaseException,
    *,
    fallback: str = CLIENT_SAFE_VALIDATION_ERROR,
) -> str:
    """Adapt the curated required-MCP failure without opening a generic escape.

    The runtime check keeps this boundary fail-closed even if a future caller
    passes an incidental exception despite the function's specific name.
    """

    if not isinstance(error, RequiredMCPUnavailableError):
        return fallback
    message = str(error)
    if message.strip():
        return message
    return fallback
