import json
import logging
import os
from typing import Any
from urllib.parse import quote, urlparse

import requests
from mcp.server.fastmcp import FastMCP

from .utils import setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("instagram-mcp")

setup_proxy_env()

mcp = FastMCP("instagram-mcp")

GRAPH_BASE_URL = "https://graph.facebook.com/v25.0"
DEFAULT_TIMEOUT_SECONDS = 30

LINKED_ACCOUNT_FIELDS = (
    "id,name,category,tasks,access_token,"
    "instagram_business_account{id,username,name,profile_picture_url}"
)
PROFILE_FIELDS = (
    "id,username,name,biography,profile_picture_url,followers_count,"
    "follows_count,media_count,website"
)
MEDIA_FIELDS = (
    "id,caption,media_type,media_url,permalink,timestamp,username,thumbnail_url"
)


class GraphAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        details: Any = None,
        sensitive_values: set[str] | None = None,
    ):
        super().__init__(message)
        self.details = details
        self.sensitive_values = sensitive_values or set()


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _redact_secrets(value: Any, sensitive_values: set[str] | None = None) -> Any:
    tokens = {token for token in sensitive_values or set() if token}
    env_token = os.environ.get("META_ACCESS_TOKEN")
    if env_token:
        tokens.add(env_token)

    if isinstance(value, dict):
        return {
            key: _redact_secrets(item, sensitive_values=tokens)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_secrets(item, sensitive_values=tokens) for item in value]
    if isinstance(value, str):
        redacted = value
        for token in tokens:
            redacted = redacted.replace(token, "[redacted]")
        return redacted
    return value


def _error(
    message: str,
    *,
    details: Any = None,
    sensitive_values: set[str] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "status": "error",
        "message": _redact_secrets(message, sensitive_values=sensitive_values),
    }
    if details is not None:
        payload["details"] = _redact_secrets(details, sensitive_values=sensitive_values)
    return json.dumps(payload, ensure_ascii=False)


def _graph_error(e: GraphAPIError) -> str:
    return _error(str(e), details=e.details, sensitive_values=e.sensitive_values)


def _user_token() -> str:
    token = os.environ.get("META_ACCESS_TOKEN")
    if not token:
        raise ValueError("META_ACCESS_TOKEN environment variable is missing")
    return token


def _graph_headers(token: str, *, form: bool = False) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    if form:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    return headers


def _graph_request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> Any:
    request_token = _user_token()
    response = requests.request(
        method=method,
        url=f"{GRAPH_BASE_URL}{path}",
        headers=_graph_headers(request_token, form=method.upper() != "GET"),
        params=params,
        data=data,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        details = _response_json(response)
        message = str(exc)
        response_text = response.text.strip()
        if response_text:
            message = f"{message} - {response_text}"
        raise GraphAPIError(
            message,
            details=details,
            sensitive_values={request_token},
        ) from exc

    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def _response_json(response: Any) -> Any:
    try:
        return response.json()
    except Exception:
        return None


def _graph_path(object_id: str, suffix: str | None = None) -> str:
    if not object_id.strip():
        raise ValueError("object id is required")
    path = f"/{quote(object_id.strip(), safe='')}"
    if suffix:
        path = f"{path}/{suffix}"
    return path


def _bounded_limit(limit: int, maximum: int = 100) -> int:
    return max(1, min(int(limit), maximum))


def _is_public_image_url(image_url: str) -> bool:
    parsed = urlparse(image_url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _page_summary(page: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": page.get("id"),
        "name": page.get("name"),
        "category": page.get("category"),
        "tasks": page.get("tasks", []),
        "has_access_token": bool(page.get("access_token")),
    }


def _linked_account(page: dict[str, Any]) -> dict[str, Any] | None:
    instagram_account = page.get("instagram_business_account")
    if not isinstance(instagram_account, dict):
        return None
    return {
        "page": _page_summary(page),
        "instagram_account": {
            "id": instagram_account.get("id"),
            "username": instagram_account.get("username"),
            "name": instagram_account.get("name"),
            "profile_picture_url": instagram_account.get("profile_picture_url"),
        },
    }


def _list_pages_with_instagram_accounts() -> list[dict[str, Any]]:
    result = _graph_request(
        "GET",
        "/me/accounts",
        params={"fields": LINKED_ACCOUNT_FIELDS},
    )
    pages = result.get("data") if isinstance(result, dict) else None
    if not isinstance(pages, list):
        return []
    accounts: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        account = _linked_account(page)
        if account:
            accounts.append(account)
    return accounts


@mcp.tool()
def instagram_auth_status() -> str:
    """Check whether the injected Meta access token is usable."""
    try:
        me = _graph_request("GET", "/me", params={"fields": "id,name,email"})
        return _success(
            authenticated=True,
            user={
                "id": me.get("id"),
                "name": me.get("name"),
                "email": me.get("email"),
            },
        )
    except GraphAPIError as e:
        logger.error("Error checking Instagram auth status: %s", e)
        return _graph_error(e)
    except Exception as e:
        logger.error("Error checking Instagram auth status: %s", e)
        return _error(str(e))


@mcp.tool()
def instagram_list_linked_accounts() -> str:
    """List Facebook Pages that have linked Instagram professional accounts."""
    try:
        return _success(accounts=_list_pages_with_instagram_accounts())
    except GraphAPIError as e:
        logger.error("Error listing Instagram linked accounts: %s", e)
        return _graph_error(e)
    except Exception as e:
        logger.error("Error listing Instagram linked accounts: %s", e)
        return _error(str(e))


@mcp.tool()
def instagram_get_profile(instagram_account_id: str) -> str:
    """Get profile information for an Instagram professional account."""
    try:
        profile = _graph_request(
            "GET",
            _graph_path(instagram_account_id),
            params={"fields": PROFILE_FIELDS},
        )
        return _success(profile=profile)
    except GraphAPIError as e:
        logger.error(
            "Error getting Instagram profile for %s: %s", instagram_account_id, e
        )
        return _graph_error(e)
    except Exception as e:
        logger.error(
            "Error getting Instagram profile for %s: %s", instagram_account_id, e
        )
        return _error(str(e))


@mcp.tool()
def instagram_list_media(instagram_account_id: str, limit: int = 10) -> str:
    """List recent media for an Instagram professional account."""
    try:
        result = _graph_request(
            "GET",
            _graph_path(instagram_account_id, "media"),
            params={"fields": MEDIA_FIELDS, "limit": _bounded_limit(limit)},
        )
        return _success(
            media=result.get("data", []),
            next_link=result.get("paging", {}).get("next"),
        )
    except GraphAPIError as e:
        logger.error(
            "Error listing Instagram media for %s: %s", instagram_account_id, e
        )
        return _graph_error(e)
    except Exception as e:
        logger.error(
            "Error listing Instagram media for %s: %s", instagram_account_id, e
        )
        return _error(str(e))


@mcp.tool()
def instagram_publish_image(
    instagram_account_id: str,
    image_url: str,
    caption: str | None = None,
) -> str:
    """Create and publish an Instagram image media container from a public image URL."""
    try:
        if not _is_public_image_url(image_url):
            raise ValueError("image_url must be a public http or https URL")
        data = {"image_url": image_url}
        if caption:
            data["caption"] = caption

        container = _graph_request(
            "POST",
            _graph_path(instagram_account_id, "media"),
            data=data,
        )
        container_id = container.get("id")
        if not container_id:
            raise ValueError("Instagram media container response did not include id")

        published = _graph_request(
            "POST",
            _graph_path(instagram_account_id, "media_publish"),
            data={"creation_id": str(container_id)},
        )
        return _success(container_id=container_id, media_id=published.get("id"))
    except GraphAPIError as e:
        logger.error(
            "Error publishing Instagram image for %s: %s", instagram_account_id, e
        )
        return _graph_error(e)
    except Exception as e:
        logger.error(
            "Error publishing Instagram image for %s: %s", instagram_account_id, e
        )
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
