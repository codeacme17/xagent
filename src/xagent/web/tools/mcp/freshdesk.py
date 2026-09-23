"""Freshdesk MCP connector, backed by the Freshdesk REST API.

Scope note: the tools here are written against the published REST contract
(developers.freshdesk.com/api) and covered by mocked tests. They have not yet
been exercised against a live tenant, so the response *shapes* are documented
rather than observed -- the end-to-end check against a real Freshdesk account
is still outstanding on xorbitsai/xagent-saas#1409, and anything this module
assumes about an envelope is a documentation claim until then.

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

from ....config import get_tool_max_output_length
from ....core.utils.security import redact_sensitive_text
from ...utils.graphql_errors import truncate_error_text
from .utils import clamp_limit, setup_proxy_env, success_with_capped_dict

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

# The filter/search endpoint is a different, narrower contract from the list
# endpoints: Freshdesk fixes it at 30 results per page and refuses pages past
# 10, so one query can reach at most 300 tickets however it is paged.
SEARCH_PAGE_SIZE = 30
MAX_SEARCH_PAGE = 10

# Freshdesk's numeric enums. Validated locally so a plausible-but-wrong value
# (status=1, which does not exist) names the legal set instead of coming back
# as the vendor's generic "Validation failed".
STATUS_OPEN = 2
TICKET_STATUSES = {2: "Open", 3: "Pending", 4: "Resolved", 5: "Closed"}
PRIORITY_LOW = 1
TICKET_PRIORITIES = {1: "Low", 2: "Medium", 3: "High", 4: "Urgent"}


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


def _send(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: Any = None,
) -> requests.Response:
    """Issue one request and translate transport/HTTP failures.

    Split out from ``_request`` so the paginated helpers can read the
    ``Link`` response header, which the parsed body does not carry.
    """
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
        if not detail:
            # freshdesk.com is wildcard-resolved: an unknown tenant answers
            # every path with a body-less 404 from the edge rather than
            # anything naming the real problem (verified against a
            # nonexistent subdomain). Without this hint a mistyped
            # FRESHDESK_SUBDOMAIN is indistinguishable from a deleted ticket,
            # and the caller retries against a tenant that does not exist.
            # Only for a detail-less 404 -- a real Freshdesk 404 carries a
            # description, and overriding that would bury it.
            if response.status_code == 404:
                detail = (
                    "no response body -- if this happens for every request, "
                    f"check that FRESHDESK_SUBDOMAIN ({_subdomain()!r}) names "
                    "an existing Freshdesk account"
                )
            elif response.status_code in (401, 403):
                detail = "check that FRESHDESK_API_KEY is current and that the key's agent has permission for this operation"
        raise RuntimeError(
            f"Freshdesk API error (status {response.status_code}){suffix}"
            + (f": {detail}" if detail else "")
        )

    return response


def _body(response: requests.Response) -> Any:
    if response.status_code == 204 or not response.content:
        return {}
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Freshdesk returned a 2xx response with a non-JSON body: {exc}"
        ) from exc


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_data: Any = None,
) -> Any:
    return _body(_send(method, path, params=params, json_data=json_data))


def _has_next_page(response: requests.Response) -> bool:
    """Whether Freshdesk says another page exists.

    Freshdesk signals this with an RFC 5988 ``Link: <...>; rel="next"`` header
    and returns no total count on list endpoints. Reading the header is the
    only exact answer: inferring from ``len(items) == per_page`` reports a
    phantom next page whenever the last page happens to be exactly full, and
    an LLM following that signal issues one pointless call per list.
    """
    return 'rel="next"' in (response.headers.get("Link") or "")


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


def _success_with_capped_list(list_field: str, payload: dict[str, Any]) -> str:
    """Build a success payload from a page of Freshdesk objects, halving
    ``payload[list_field]`` until it fits the platform's output limit.

    Mirrors chartmogul.py's helper of the same name, including its explicit
    "data lost" message when halving actually ran: the dropped objects belong
    to the page this one call already fetched, so they are not recoverable by
    asking for the next page -- only by re-running with a smaller per_page.
    Freshdesk makes this more likely than most: a ticket carries its full
    description, so a page of 100 can be very large.
    """
    max_output_length = get_tool_max_output_length()
    items = payload.get(list_field) or []

    def _build(items: list[Any], truncated: bool, halved: bool) -> str:
        extra: dict[str, Any] = {}
        if halved:
            extra["message"] = (
                f"Returned {len(items)} {list_field} out of the full page; the "
                "rest did not fit the output size limit and cannot be recovered "
                "by fetching the next page (a smaller per_page avoids this)."
            )
        return _success(
            **{**payload, list_field: items, "truncated": truncated, **extra}
        )

    halved = False
    response = _build(items, False, halved)
    while len(response) > max_output_length and items:
        items = items[: len(items) // 2]
        halved = True
        response = _build(items, True, halved)
    if halved and len(response) > max_output_length:
        # Even an empty item list didn't buy back enough room -- drop the
        # added "message" text itself as a last resort, matching chartmogul's
        # identical fallback.
        response = _build(items, True, False)
    return response


def _ticket_id(value: Any) -> int:
    """Coerce and bounds-check a ticket id.

    Interpolated into the request path, so it is validated as a positive
    integer here rather than trusted: an LLM passing "12 OR 1=1", a float, or
    a path fragment would otherwise be pasted straight into the URL.
    """
    try:
        ticket_id = int(value)
    except (TypeError, ValueError):
        raise RuntimeError(f"ticket_id must be an integer, got {value!r}") from None
    if ticket_id <= 0:
        raise RuntimeError(f"ticket_id must be a positive integer, got {ticket_id}")
    return ticket_id


def _validated_choice(
    value: Any, allowed: dict[int, str], field_name: str
) -> int | None:
    """Validate a Freshdesk numeric enum locally.

    Rejecting here rather than forwarding turns an LLM's plausible-but-wrong
    value (status=1, which does not exist) into a message naming the legal
    values, instead of Freshdesk's generic "Validation failed".
    """
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise RuntimeError(f"{field_name} must be an integer, got {value!r}") from None
    if parsed not in allowed:
        legal = ", ".join(f"{k} ({v})" for k, v in sorted(allowed.items()))
        raise RuntimeError(f"{field_name} must be one of: {legal}; got {parsed}")
    return parsed


# ---------------------------------------------------------------------------
# Tickets
# ---------------------------------------------------------------------------


@mcp.tool()
def freshdesk_list_tickets(
    filter_name: str | None = None,
    updated_since: str | None = None,
    include: str | None = None,
    order_by: str | None = None,
    order_type: str | None = None,
    page: int = 1,
    per_page: int = DEFAULT_PER_PAGE,
) -> str:
    """
    List tickets in the Freshdesk helpdesk, newest first by default.

    This endpoint offers only Freshdesk's canned views, not arbitrary
    filtering: use freshdesk_search_tickets for conditions such as
    "open tickets assigned to X".

    filter_name: one of Freshdesk's predefined views -- "new_and_my_open",
    "watching", "spam", "deleted". Omit for the default view, which excludes
    spam and deleted tickets.
    updated_since: ISO 8601 timestamp, e.g. "2026-09-01T00:00:00Z"; returns
    only tickets updated at or after it.
    include: comma-separated side-loads, e.g. "requester,stats". Each one
    enlarges every ticket in the page, which makes size truncation more
    likely -- pair it with a smaller per_page.
    order_by: "created_at", "due_by", "updated_at" or "status".
    order_type: "asc" or "desc".
    page: 1-based page number.
    per_page: tickets per page (default 30, hard cap 100).

    The response carries has_more, taken from Freshdesk's own Link header, so
    it is exact rather than inferred. If truncated is true the page was
    trimmed to fit the output limit and the dropped tickets are NOT on the
    next page -- re-run with a smaller per_page to see them.
    """
    try:
        response = _send(
            "GET",
            "/tickets",
            params={
                "filter": filter_name,
                "updated_since": updated_since,
                "include": include,
                "order_by": order_by,
                "order_type": order_type,
                "page": max(1, int(page)),
                "per_page": _clamp_per_page(per_page),
            },
        )
        tickets = _validated_list(_body(response), "/tickets")
        return _success_with_capped_list(
            "tickets", {"tickets": tickets, "has_more": _has_next_page(response)}
        )
    except Exception as exc:  # noqa: BLE001 - tool boundary
        return _error(str(exc))


@mcp.tool()
def freshdesk_get_ticket(ticket_id: int, include: str | None = None) -> str:
    """
    Fetch one ticket by its numeric id, including its description.

    ticket_id: the ticket's numeric id (the number in its Freshdesk URL).
    include: comma-separated side-loads, e.g. "conversations,requester,stats".
    Note that "conversations" returns only the most recent ten -- use
    freshdesk_list_ticket_conversations to page through all of them.
    """
    try:
        ticket = _validated_dict(
            _request(
                "GET", f"/tickets/{_ticket_id(ticket_id)}", params={"include": include}
            ),
            f"/tickets/{ticket_id}",
        )
        return success_with_capped_dict("ticket", ticket)
    except Exception as exc:  # noqa: BLE001 - tool boundary
        return _error(str(exc))


@mcp.tool()
def freshdesk_search_tickets(query: str, page: int = 1) -> str:
    """
    Search tickets with Freshdesk's filter query language.

    query: the filter expression WITHOUT the enclosing double quotes this
    tool adds, e.g. status:2 AND priority:4, or
    agent_id:123 AND status:2, or
    created_at:>'2026-09-01'. String values take single quotes, numbers and
    booleans take none. Combine with AND/OR and parentheses. Freshdesk caps
    the expression at 512 characters.

    Queryable fields include agent_id, group_id, priority, status, tag,
    type, created_at, updated_at, due_by and custom fields.

    page: 1-based page number. Freshdesk fixes this endpoint at 30 results
    per page and refuses pages beyond 10, so at most 300 tickets are
    reachable for one query -- narrow the query rather than paging further.

    The response carries total, Freshdesk's own count of matches, which can
    exceed the number reachable through paging.
    """
    try:
        expression = (query or "").strip()
        if not expression:
            raise RuntimeError("query must not be empty")
        page_number = max(1, int(page))
        if page_number > MAX_SEARCH_PAGE:
            raise RuntimeError(
                f"Freshdesk refuses search pages beyond {MAX_SEARCH_PAGE} "
                f"(asked for {page_number}); narrow the query instead"
            )
        # The expression must reach Freshdesk wrapped in double quotes; requests
        # percent-encodes them. Quotes are added here rather than asked of the
        # caller because an LLM passing a pre-quoted string would otherwise
        # produce a doubly-quoted, always-empty search.
        response = _send(
            "GET",
            "/search/tickets",
            params={"query": f'"{expression}"', "page": page_number},
        )
        payload = _validated_dict(_body(response), "/search/tickets")
        results = payload.get("results")
        if not isinstance(results, list):
            raise RuntimeError(
                "Freshdesk returned an unexpected response for /search/tickets"
            )
        return _success_with_capped_list(
            "results",
            {
                "results": results,
                "total": payload.get("total"),
                "page": page_number,
                "has_more": page_number < MAX_SEARCH_PAGE
                and len(results) == SEARCH_PAGE_SIZE,
            },
        )
    except Exception as exc:  # noqa: BLE001 - tool boundary
        return _error(str(exc))


@mcp.tool()
def freshdesk_create_ticket(
    subject: str,
    description: str,
    email: str | None = None,
    requester_id: int | None = None,
    phone: str | None = None,
    status: int = STATUS_OPEN,
    priority: int = PRIORITY_LOW,
    responder_id: int | None = None,
    group_id: int | None = None,
    tags: list[str] | None = None,
    cc_emails: list[str] | None = None,
) -> str:
    """
    Create a ticket on behalf of a requester.

    subject: the ticket's subject line.
    description: the ticket body; Freshdesk renders it as HTML.
    email / requester_id / phone: how the requester is identified. Exactly
    one is enough and at least one is required -- Freshdesk creates a new
    contact for an unknown email or phone.
    status: 2 Open (default), 3 Pending, 4 Resolved, 5 Closed.
    priority: 1 Low (default), 2 Medium, 3 High, 4 Urgent.
    responder_id: the agent to assign; omit to leave unassigned.
    group_id: the group to route to.
    tags: tags to set on the new ticket.
    cc_emails: addresses to copy on the ticket's email notifications.
    """
    try:
        if not (subject or "").strip():
            raise RuntimeError("subject must not be empty")
        if not (description or "").strip():
            raise RuntimeError("description must not be empty")
        if (
            requester_id is None
            and not (email or "").strip()
            and not (phone or "").strip()
        ):
            raise RuntimeError(
                "one of email, requester_id or phone is required to identify "
                "the requester"
            )
        payload: dict[str, Any] = {
            "subject": subject,
            "description": description,
            "status": _validated_choice(status, TICKET_STATUSES, "status"),
            "priority": _validated_choice(priority, TICKET_PRIORITIES, "priority"),
        }
        optional = {
            "email": email,
            "requester_id": requester_id,
            "phone": phone,
            "responder_id": responder_id,
            "group_id": group_id,
            "tags": tags,
            "cc_emails": cc_emails,
        }
        payload.update({k: v for k, v in optional.items() if v not in (None, "", [])})
        ticket = _validated_dict(
            _request("POST", "/tickets", json_data=payload), "/tickets"
        )
        return success_with_capped_dict("ticket", ticket)
    except Exception as exc:  # noqa: BLE001 - tool boundary
        return _error(str(exc))


@mcp.tool()
def freshdesk_update_ticket(
    ticket_id: int,
    status: int | None = None,
    priority: int | None = None,
    responder_id: int | None = None,
    group_id: int | None = None,
    tags: list[str] | None = None,
) -> str:
    """
    Update a ticket's status, priority, assignee, group or tags.

    Only the fields passed are changed; omitted fields are left alone.

    ticket_id: the ticket's numeric id.
    status: 2 Open, 3 Pending, 4 Resolved, 5 Closed.
    priority: 1 Low, 2 Medium, 3 High, 4 Urgent.
    responder_id: the agent to assign the ticket to.
    group_id: the group to route the ticket to.
    tags: REPLACES the ticket's entire tag list rather than adding to it --
    Freshdesk has no "append a tag" operation, so read the ticket first with
    freshdesk_get_ticket and pass the existing tags plus the new one, or the
    others are removed.
    """
    try:
        payload: dict[str, Any] = {}
        validated_status = _validated_choice(status, TICKET_STATUSES, "status")
        if validated_status is not None:
            payload["status"] = validated_status
        validated_priority = _validated_choice(priority, TICKET_PRIORITIES, "priority")
        if validated_priority is not None:
            payload["priority"] = validated_priority
        if responder_id is not None:
            payload["responder_id"] = responder_id
        if group_id is not None:
            payload["group_id"] = group_id
        # An explicit empty list is a real instruction ("clear the tags") and
        # must survive, unlike None which means "leave them alone".
        if tags is not None:
            payload["tags"] = tags
        if not payload:
            raise RuntimeError(
                "pass at least one of status, priority, responder_id, group_id "
                "or tags to update"
            )
        ticket = _validated_dict(
            _request("PUT", f"/tickets/{_ticket_id(ticket_id)}", json_data=payload),
            f"/tickets/{ticket_id}",
        )
        return success_with_capped_dict("ticket", ticket)
    except Exception as exc:  # noqa: BLE001 - tool boundary
        return _error(str(exc))


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


@mcp.tool()
def freshdesk_list_ticket_conversations(
    ticket_id: int,
    page: int = 1,
    per_page: int = DEFAULT_PER_PAGE,
) -> str:
    """
    List a ticket's conversation entries -- replies and notes -- oldest first.

    ticket_id: the ticket's numeric id.
    page: 1-based page number.
    per_page: entries per page (default 30, hard cap 100).

    Each entry carries `private`: true marks an internal note that the
    requester cannot see, false a reply that was sent to them.
    """
    try:
        response = _send(
            "GET",
            f"/tickets/{_ticket_id(ticket_id)}/conversations",
            params={
                "page": max(1, int(page)),
                "per_page": _clamp_per_page(per_page),
            },
        )
        conversations = _validated_list(
            _body(response), f"/tickets/{ticket_id}/conversations"
        )
        return _success_with_capped_list(
            "conversations",
            {"conversations": conversations, "has_more": _has_next_page(response)},
        )
    except Exception as exc:  # noqa: BLE001 - tool boundary
        return _error(str(exc))


@mcp.tool()
def freshdesk_reply_to_ticket(
    ticket_id: int,
    body: str,
    cc_emails: list[str] | None = None,
    bcc_emails: list[str] | None = None,
) -> str:
    """
    Post a public reply to a ticket. THE REQUESTER IS EMAILED THIS TEXT.

    Use freshdesk_add_note_to_ticket for anything the customer should not
    see -- a reply cannot be unsent.

    ticket_id: the ticket's numeric id.
    body: the reply content; Freshdesk renders it as HTML.
    cc_emails / bcc_emails: additional recipients for this reply only.
    """
    try:
        if not (body or "").strip():
            raise RuntimeError("body must not be empty")
        payload: dict[str, Any] = {"body": body}
        if cc_emails:
            payload["cc_emails"] = cc_emails
        if bcc_emails:
            payload["bcc_emails"] = bcc_emails
        reply = _validated_dict(
            _request(
                "POST", f"/tickets/{_ticket_id(ticket_id)}/reply", json_data=payload
            ),
            f"/tickets/{ticket_id}/reply",
        )
        return success_with_capped_dict("reply", reply)
    except Exception as exc:  # noqa: BLE001 - tool boundary
        return _error(str(exc))


@mcp.tool()
def freshdesk_add_note_to_ticket(
    ticket_id: int,
    body: str,
    private: bool = True,
    notify_emails: list[str] | None = None,
) -> str:
    """
    Add a note to a ticket. Private by default.

    ticket_id: the ticket's numeric id.
    body: the note content; Freshdesk renders it as HTML.
    private: true (the default) keeps the note internal to agents. Passing
    false makes it visible to the requester in the ticket's portal view --
    which is a disclosure, so pass it deliberately rather than to "share
    context".
    notify_emails: agents to email about this note.
    """
    try:
        if not (body or "").strip():
            raise RuntimeError("body must not be empty")
        payload: dict[str, Any] = {"body": body, "private": bool(private)}
        if notify_emails:
            payload["notify_emails"] = notify_emails
        note = _validated_dict(
            _request(
                "POST", f"/tickets/{_ticket_id(ticket_id)}/notes", json_data=payload
            ),
            f"/tickets/{ticket_id}/notes",
        )
        return success_with_capped_dict("note", note)
    except Exception as exc:  # noqa: BLE001 - tool boundary
        return _error(str(exc))


# ---------------------------------------------------------------------------
# Contacts and agents
# ---------------------------------------------------------------------------


@mcp.tool()
def freshdesk_get_contact(contact_id: int) -> str:
    """
    Fetch one contact by its numeric id.

    contact_id: the contact's numeric id, as carried by a ticket's
    requester_id.
    """
    try:
        contact = _validated_dict(
            _request("GET", f"/contacts/{_ticket_id(contact_id)}"),
            f"/contacts/{contact_id}",
        )
        return success_with_capped_dict("contact", contact)
    except Exception as exc:  # noqa: BLE001 - tool boundary
        return _error(str(exc))


@mcp.tool()
def freshdesk_search_contacts(
    term: str | None = None,
    email: str | None = None,
    phone: str | None = None,
    mobile: str | None = None,
    company_id: int | None = None,
    page: int = 1,
    per_page: int = DEFAULT_PER_PAGE,
) -> str:
    """
    Find contacts by name keyword or by an exact field match.

    term: a name or email fragment; matches are ranked by Freshdesk's
    autocomplete. Use this when you only know roughly who the person is.
    email / phone / mobile / company_id: exact-match filters. Use these when
    you have the precise value -- an exact filter is authoritative, whereas
    term is a fuzzy match that can return near misses.

    term cannot be combined with the exact filters; pass either one term or
    any number of filters. With neither, this lists contacts in Freshdesk's
    default order.

    page / per_page: pagination for the filtered listing (per_page default
    30, hard cap 100). They do not apply to a term search, which returns
    Freshdesk's own ranked shortlist in one response.
    """
    try:
        keyword = (term or "").strip()
        filters = {
            "email": email,
            "phone": phone,
            "mobile": mobile,
            "company_id": company_id,
        }
        active_filters = {k: v for k, v in filters.items() if v not in (None, "")}
        if keyword and active_filters:
            raise RuntimeError(
                "pass either term or the exact filters "
                f"({', '.join(sorted(active_filters))}), not both"
            )
        if keyword:
            # Freshdesk's documented keyword endpoint. The structured
            # /search/contacts filter API is marked BETA with no published
            # query grammar, so it is deliberately not built on here.
            contacts = _validated_list(
                _request("GET", "/contacts/autocomplete", params={"term": keyword}),
                "/contacts/autocomplete",
            )
            return _success_with_capped_list(
                "contacts", {"contacts": contacts, "has_more": False}
            )
        response = _send(
            "GET",
            "/contacts",
            params={
                **active_filters,
                "page": max(1, int(page)),
                "per_page": _clamp_per_page(per_page),
            },
        )
        contacts = _validated_list(_body(response), "/contacts")
        return _success_with_capped_list(
            "contacts", {"contacts": contacts, "has_more": _has_next_page(response)}
        )
    except Exception as exc:  # noqa: BLE001 - tool boundary
        return _error(str(exc))


@mcp.tool()
def freshdesk_list_agents(
    email: str | None = None,
    state: str | None = None,
    page: int = 1,
    per_page: int = DEFAULT_PER_PAGE,
) -> str:
    """
    List the helpdesk's agents, for resolving who to assign a ticket to.

    email: exact-match filter on an agent's email.
    state: "fulltime" or "occasional".
    page: 1-based page number.
    per_page: agents per page (default 30, hard cap 100).

    Each agent's `id` is what freshdesk_update_ticket takes as responder_id.
    """
    try:
        response = _send(
            "GET",
            "/agents",
            params={
                "email": email,
                "state": state,
                "page": max(1, int(page)),
                "per_page": _clamp_per_page(per_page),
            },
        )
        agents = _validated_list(_body(response), "/agents")
        return _success_with_capped_list(
            "agents", {"agents": agents, "has_more": _has_next_page(response)}
        )
    except Exception as exc:  # noqa: BLE001 - tool boundary
        return _error(str(exc))


if __name__ == "__main__":
    mcp.run()
