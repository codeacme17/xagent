"""Freshdesk MCP connector, backed by the Freshdesk REST API.

Scope note: this module currently carries the configuration, URL-composition
and error-translation layer. The ticket/contact/agent tools land once the REST
probe against a real tenant confirms the response envelopes -- see
xorbitsai/xagent-saas#1409.

Why REST rather than Freshdesk's own remote MCP endpoint: that endpoint is
metered separately and generously little (100 actions per account per month on
Growth, against 100 REST calls per minute), and bridging to it would put the
per-tenant URL and a static credential on a shared catalog row, which the
catalog has no per-user shape for. Going through REST keeps this connector on
the existing stdio/api_key path, where the subdomain and key are ordinary
per-user encrypted env.
"""

import json
import logging
import os
import re
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

from ....core.utils.security import redact_sensitive_text
from ...utils.graphql_errors import truncate_error_text
from .utils import clamp_limit, setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("freshdesk-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("freshdesk-mcp")

# The only host this connector ever talks to. Freshdesk serves each tenant at
# <subdomain>.freshdesk.com and, per the vendor's MCP/API documentation, does
# not support custom domains for programmatic access -- so the host is composed
# from a validated label here rather than accepted as a URL from the user.
# That is the whole SSRF story for this connector: there is no user-supplied
# host, port, scheme or path to validate, unlike magento.py, whose store URL is
# genuinely customer-controlled and therefore needs DNS pinning and
# private-network rejection.
FRESHDESK_DOMAIN = "freshdesk.com"

# A DNS label: 1-63 chars, alphanumeric, internal hyphens only. Deliberately
# stricter than DNS itself (no underscores, ASCII only) -- a Freshdesk
# subdomain is chosen from their signup form, which is narrower still, and
# anything this rejects is a user error rather than a tenant we are locking
# out. Anchored with \A/\Z rather than ^/$ because $ also matches before a
# trailing newline, which would let "acme\n" through into the Host header.
SUBDOMAIN_PATTERN = re.compile(r"\A[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")

DEFAULT_TIMEOUT_SECONDS = 30
# Freshdesk's own list page-size cap (developers.freshdesk.com/api: per_page
# defaults to 30 and maxes out at 100).
MAX_PER_PAGE = 100
DEFAULT_PER_PAGE = 30


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _subdomain() -> str:
    """Return the validated tenant subdomain label.

    Normalizes case and surrounding whitespace first: DNS labels are
    case-insensitive and a copy-pasted value routinely carries both, so
    normalizing keeps a recoverable input from becoming an error -- while
    everything that could change *which host* is contacted (a dot, slash,
    colon, credential marker, or query/fragment separator) is rejected,
    since the label is interpolated into the host position below.
    """
    raw = (os.environ.get("FRESHDESK_SUBDOMAIN") or "").strip().lower()
    if not raw:
        raise ValueError("FRESHDESK_SUBDOMAIN environment variable is missing or empty")
    if not SUBDOMAIN_PATTERN.match(raw):
        raise ValueError(
            "FRESHDESK_SUBDOMAIN must be just the tenant label -- the 'acme' in "
            f"acme.{FRESHDESK_DOMAIN}, not a full hostname or URL"
        )
    return raw


def _base_url() -> str:
    return f"https://{_subdomain()}.{FRESHDESK_DOMAIN}/api/v2"


def _api_key() -> str:
    # Stripped, not a bare os.environ.get(): a stray leading/trailing newline
    # or space in the injected key would otherwise produce a malformed Basic
    # Auth header rather than the clear "missing" error below.
    api_key = (os.environ.get("FRESHDESK_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("FRESHDESK_API_KEY environment variable is missing or empty")
    return api_key


def _extract_error_detail(response: requests.Response) -> str | None:
    """Pull a human-readable message out of a Freshdesk error body.

    Freshdesk's 400 bodies put a generic "Validation failed" in ``description``
    and the actionable part in ``errors`` (a list of {field, message, code}),
    so both are joined rather than taking the first key that matches -- a
    caller told only "Validation failed" has nothing to fix. Returns None when
    the body is not JSON or not an object, so the caller falls back to the raw
    response text.
    """
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None

    parts: list[str] = []
    for key in ("description", "message"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
            break

    errors = payload.get("errors")
    if isinstance(errors, (list, dict)) and errors:
        parts.append(json.dumps(errors, ensure_ascii=False))
    elif isinstance(errors, str) and errors:
        parts.append(errors)

    return ": ".join(parts) if parts else None


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: Any = None,
) -> Any:
    try:
        response = requests.request(
            method=method,
            url=f"{_base_url()}{path}",
            # Freshdesk authenticates with HTTP Basic using the API key as the
            # username and an ignored password ("X" by the vendor's own
            # convention) -- not a bearer token. Note this differs from their
            # remote MCP endpoint, which takes a bare `Authorization: <key>`.
            auth=(_api_key(), "X"),
            headers={"Content-Type": "application/json"},
            # Drop "" as well as None: every list filter is optional, and an
            # LLM tool-call passing e.g. status="" to mean "no filter" would
            # otherwise become a real `?status=` query param.
            params={
                k: v for k, v in (params or {}).items() if v is not None and v != ""
            },
            json=json_data,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        # A connection/timeout/proxy failure's message can itself embed
        # sensitive data -- e.g. a ProxyError echoing the ambient HTTPS_PROXY
        # URL, which may carry embedded user:pass@ credentials
        # (setup_proxy_env() exports whatever the OS has configured).
        raise RuntimeError(
            f"Freshdesk request failed: {truncate_error_text(redact_sensitive_text(str(exc)))}"
        ) from exc

    if response.status_code >= 400:
        detail = _extract_error_detail(response)
        if detail is None:
            detail = response.text.strip()
        # The response body is host-controlled content -- if it echoes request
        # headers (a misconfigured proxy/WAF error page), redact the Basic Auth
        # credential before it reaches logs or the LLM's context.
        detail = truncate_error_text(redact_sensitive_text(detail))
        # 429 is the one error a caller can act on, and Freshdesk puts the wait
        # in Retry-After. Surfacing it turns "rate limited" into a decision the
        # caller can actually make.
        retry_after = response.headers.get("Retry-After")
        suffix = (
            f" (retry after {retry_after}s)"
            if response.status_code == 429 and retry_after
            else ""
        )
        raise RuntimeError(
            f"Freshdesk API error (status {response.status_code}){suffix}"
            + (f": {detail}" if detail else "")
        )

    if response.status_code == 204 or not response.content:
        return {}
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Freshdesk returned a 2xx response with a non-JSON body: {exc}"
        ) from exc


def _clamp_per_page(per_page: int) -> int:
    return clamp_limit(per_page, max_limit=MAX_PER_PAGE)


def _validated_dict(result: Any, endpoint: str) -> dict[str, Any]:
    """Return ``result`` if it's a dict, else raise.

    Every tool body is inside a ``try/except Exception -> _error(str(e))``
    block, so raising here (rather than returning a union every call site
    would have to narrow) reaches the caller's own error envelope unchanged.
    """
    if not isinstance(result, dict):
        raise RuntimeError(f"Freshdesk returned an unexpected response for {endpoint}")
    return result


def _validated_list(result: Any, endpoint: str) -> list[Any]:
    """Return ``result`` if it's a list, else raise.

    Freshdesk's list endpoints return a bare JSON array rather than an
    envelope object (the search endpoints differ and are validated as dicts
    via ``_validated_dict``).
    """
    if not isinstance(result, list):
        raise RuntimeError(f"Freshdesk returned an unexpected response for {endpoint}")
    return result


# ---------------------------------------------------------------------------
# Tools
#
# Held until the REST probe confirms the response envelopes for list, single
# and search endpoints (xorbitsai/xagent-saas#1409). Planned set, agreed with
# the requester: list_tickets, get_ticket, search_tickets,
# list_ticket_conversations, get_contact, search_contacts, list_agents,
# create_ticket, update_ticket, reply_to_ticket, add_note_to_ticket.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    mcp.run()
