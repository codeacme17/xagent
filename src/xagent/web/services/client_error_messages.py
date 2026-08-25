"""Fixed client-visible fallbacks for incidental server failures."""

CLIENT_SAFE_VALIDATION_ERROR = "The message could not be processed. Please try again."

# Task audiences did not necessarily initiate the failing operation, so a
# task-level failure uses neutral wording instead of the validation fallback.
CLIENT_SAFE_TASK_FAILURE = "Task execution failed."


def client_safe_error_message(
    error: BaseException,
    *,
    safe_for_display: bool = False,
    fallback: str = CLIENT_SAFE_VALIDATION_ERROR,
) -> str:
    """Return exception text only when the caller proves it is public-safe."""

    return str(error) if safe_for_display else fallback
