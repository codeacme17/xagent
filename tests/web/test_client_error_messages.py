from xagent.core.tools.adapters.vibe.config import RequiredMCPUnavailableError
from xagent.web.services.client_error_messages import (
    CLIENT_SAFE_GUIDANCE_IN_PROGRESS,
    CLIENT_SAFE_TASK_FAILURE,
    CLIENT_SAFE_VALIDATION_ERROR,
    ClientErrorCode,
    client_error_message,
    required_mcp_unavailable_client_message,
)


def test_client_error_codes_have_fixed_safe_fallbacks() -> None:
    assert (
        client_error_message(ClientErrorCode.MESSAGE_PROCESSING_FAILED)
        == CLIENT_SAFE_VALIDATION_ERROR
    )
    assert (
        client_error_message(ClientErrorCode.TASK_EXECUTION_FAILED)
        == CLIENT_SAFE_TASK_FAILURE
    )
    assert (
        client_error_message(ClientErrorCode.GUIDANCE_IN_PROGRESS)
        == CLIENT_SAFE_GUIDANCE_IN_PROGRESS
    )


def test_required_mcp_error_preserves_its_curated_client_message() -> None:
    error = RequiredMCPUnavailableError([])

    assert required_mcp_unavailable_client_message(error) == str(error)


def test_required_mcp_adapter_rejects_incidental_exceptions() -> None:
    error = RuntimeError("provider token=secret")

    assert (
        required_mcp_unavailable_client_message(
            error,
            fallback=CLIENT_SAFE_TASK_FAILURE,
        )
        == CLIENT_SAFE_TASK_FAILURE
    )
