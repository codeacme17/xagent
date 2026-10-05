"""Local-file guard and response handling for the Jira attachment upload tool.

Kept apart from jira.py so the upload-specific logic does not grow that
module. Nothing here depends on jira.py, so jira.py can import it without a
cycle.
"""

import logging
import os
from pathlib import Path
from typing import Any

from .utils import allowed_dirs_from_env

logger = logging.getLogger("jira-mcp")

UPLOAD_ALLOWED_DIRS_ENV_VAR = "XAGENT_JIRA_FILE_ALLOWED_DIRS"

# A memory-safety bound for the connector subprocess, which holds the whole
# file in memory and builds a multipart body from it (a 429 retry builds a
# second body). It is NOT Jira's own limit: that is a per-site admin setting
# (Jira Cloud's default is 1 GB), and Jira's 413 gets its own message below.
MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024

_ATTACHMENT_STATUS_HINTS = {
    403: (
        "Jira denied the upload -- the connected account needs the Browse "
        "Projects and Create attachments permissions on this issue's project"
    ),
    404: (
        "Jira could not find the issue or the site (check issue_key and "
        "cloud_id), or the connected account cannot view the issue"
    ),
    413: (
        "Jira rejected the upload as too large -- the file exceeds the site's "
        "attachment size limit, or the issue has reached its attachment limit"
    ),
}

_ATTACHMENT_UNCONFIRMED = (
    "Jira did not confirm the attachment (no attachment id in the response); "
    "check the issue before retrying"
)


def allowed_file_dirs() -> list[Path]:
    try:
        return allowed_dirs_from_env(UPLOAD_ALLOWED_DIRS_ENV_VAR)
    except ValueError as exc:
        logger.warning("Invalid Jira upload directory configuration: %s", exc)
        raise ValueError("Upload directory configuration is invalid") from None


def _resolve(file_path: str) -> Path:
    try:
        candidate = Path(file_path).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        resolved = candidate.resolve()
        # An explicit symlink is resolved strictly, so a cyclic or dangling
        # link raises here instead of passing on as an unresolved path (the
        # same handling allowed_dirs_from_env gives its configured roots).
        if candidate.is_symlink():
            resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Could not resolve Jira attachment path %r: %s", file_path, exc)
        raise ValueError("Could not resolve file_path") from exc
    return resolved


def _require_allowed(local_path: Path) -> None:
    allowed_dirs = allowed_file_dirs()
    if any(local_path.is_relative_to(allowed_dir) for allowed_dir in allowed_dirs):
        return
    # The absolute host path is deliberately kept out of the raised message:
    # it reaches the caller/LLM unfiltered through the tool's error payload,
    # and host filesystem layout has no business in a model transcript. Full
    # detail (including the allowed directories) is logged server-side.
    logger.warning(
        "Rejected Jira attachment path %s outside allowed directories: %s",
        local_path,
        ", ".join(str(path) for path in allowed_dirs),
    )
    raise PermissionError(
        "file_path is outside the allowed upload directories; ask the user "
        "for a file inside the task workspace or another allowed location"
    )


def read_allowed_file(file_path: str) -> tuple[str, bytes]:
    """Return ``(filename, content)`` for a file under an allowlisted directory.

    Restricts jira_add_attachment to the task workspace (and any other
    configured root), the same defense the Slack, Gmail and OneDrive upload
    tools use -- without it an agent could be tricked into exfiltrating
    arbitrary host files to Jira.

    Containment is checked before existence, so a path outside the allowlist
    fails the same way whether or not it exists: reporting "not found" for
    one and "outside" for the other would let a caller probe the host
    filesystem. The size and the bytes come from one opened handle, so the
    file cannot grow past the checked size between the check and the read.

    The allowlist check runs on the resolved path before the open, as in the
    sibling connectors, so it does not by itself bind the opened handle to
    that path: a path swapped for a symlink inside that window is not
    detected.
    """
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("file_path must not be blank")
    if "\0" in file_path:
        raise ValueError("file_path must not contain NUL bytes")

    local_path = _resolve(file_path)
    _require_allowed(local_path)

    if local_path.exists() and not local_path.is_file():
        raise ValueError("The given path is not a regular file")
    if not local_path.is_file():
        raise FileNotFoundError("File not found at the given path")

    try:
        with local_path.open("rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            if size == 0:
                raise ValueError("File is empty")
            if size > MAX_ATTACHMENT_BYTES:
                raise ValueError(
                    f"File is {size} bytes, over the local upload limit of "
                    f"{MAX_ATTACHMENT_BYTES} bytes"
                )
            data = fh.read(size)
    except OSError as exc:
        # str(OSError) embeds the host path, so it cannot be re-raised as-is.
        logger.warning("Could not read Jira attachment %s: %s", local_path, exc)
        raise ValueError("Could not read the file") from exc
    if len(data) != size:
        raise ValueError("File changed while it was being read")
    return local_path.name, data


def attachment_error_hint(exc: Exception) -> str | None:
    """An actionable hint for a failed upload, or None.

    _request_absolute raises RuntimeError(...) from the HTTPError, which is
    where the failing response lives. The hint is only given when that
    response is the upload request's own: a 403/404 from the site lookup made
    when cloud_id is empty says nothing about attachment permissions.
    """
    response = getattr(exc.__cause__, "response", None)
    status = getattr(response, "status_code", None)
    url = getattr(response, "url", None)
    if not isinstance(status, int) or not isinstance(url, str):
        return None
    if not url.endswith("/attachments"):
        return None
    return _ATTACHMENT_STATUS_HINTS.get(status)


def confirmed_attachment(result: Any, sent_bytes: int) -> dict[str, Any]:
    """Summarize the attachment Jira created, or raise if the response does
    not confirm one.

    Jira answers with an array of the created attachments. A 200 that carries
    no attachment id is not a success, and a size that differs from what was
    sent means something other than the bytes that were read was stored: both
    raise rather than reporting success. The size message names the
    attachment because it does exist on the issue -- an agent that retried
    blindly would add a duplicate.
    """
    entry = result[0] if isinstance(result, list) and result else None
    if not isinstance(entry, dict):
        raise ValueError(_ATTACHMENT_UNCONFIRMED)
    attachment_id = entry.get("id")
    # Presence, not truthiness: see jira_transition_issue's id check.
    if attachment_id is None or (
        isinstance(attachment_id, str)
        and (not attachment_id or attachment_id.strip() != attachment_id)
    ):
        raise ValueError(_ATTACHMENT_UNCONFIRMED)
    size = entry.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size != sent_bytes:
        raise ValueError(
            f"Jira created attachment {attachment_id} but reported size "
            f"{size!r} instead of {sent_bytes} bytes; verify the attachment "
            "on the issue before retrying"
        )
    return {
        "id": attachment_id,
        "filename": entry.get("filename"),
        "size": size,
        "mime_type": entry.get("mimeType"),
    }
