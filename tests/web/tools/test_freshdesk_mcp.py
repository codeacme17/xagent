"""Tests for the Freshdesk MCP connector's scaffolding.

Scope note: this file currently covers the configuration, URL-composition and
error-translation layer only. The per-tool response parsing is deliberately
absent until the REST probe against a real tenant confirms the response
envelopes -- writing assertions against a guessed shape would only prove the
code matches the guess.
"""

import json
from typing import Any

import pytest
import requests

from xagent.web.tools.mcp import freshdesk


class _FakeResponse:
    """Minimal stand-in for requests.Response.

    Only the attributes _request()/_extract_error_detail() actually read are
    implemented, so a drift in what they read shows up as an AttributeError
    here rather than as a test passing against a mock that quietly answers
    everything.
    """

    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: Any = None,
        text: str = "",
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.content = content if content is not None else text.encode()
        self.headers = headers or {}

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


# ---------------------------------------------------------------------------
# Subdomain validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "acme",
        "acme-support",
        "a",
        "a1",
        "1acme",
        "a" * 63,
    ],
)
def test_subdomain_accepts_valid_labels(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("FRESHDESK_SUBDOMAIN", value)
    assert freshdesk._subdomain() == value


def test_subdomain_lowercases_and_strips(monkeypatch: pytest.MonkeyPatch):
    """A copy-pasted value routinely carries case and surrounding whitespace.

    Both are safe to normalize (DNS labels are case-insensitive), and doing so
    here keeps them from reaching the composed URL, where a stray space would
    become %20 in the Host rather than an actionable error.
    """
    monkeypatch.setenv("FRESHDESK_SUBDOMAIN", "  ACME-Support \n")
    assert freshdesk._subdomain() == "acme-support"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        # A full host, or anything carrying the domain, is the most likely
        # user error: accepting it would compose acme.freshdesk.com.freshdesk.com.
        "acme.freshdesk.com",
        "https://acme.freshdesk.com",
        "acme.freshdesk.com/api/v2",
        # Separators that would escape the label position in the composed URL.
        "acme/../evil",
        "acme/x",
        "acme:8443",
        "acme@evil.com",
        "acme?x=1",
        "acme#frag",
        "acme evil",
        # Label-shape rules: no leading/trailing hyphen, no underscore, <= 63.
        "-acme",
        "acme-",
        "acme_support",
        "a" * 64,
        # Non-ASCII: freshdesk.com subdomains are ASCII labels, and accepting
        # Unicode here would leave the IDNA question open at the URL layer.
        "acmé",
    ],
)
def test_subdomain_rejects_invalid_values(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("FRESHDESK_SUBDOMAIN", value)
    with pytest.raises(ValueError, match="FRESHDESK_SUBDOMAIN"):
        freshdesk._subdomain()


def test_subdomain_missing_env_is_actionable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FRESHDESK_SUBDOMAIN", raising=False)
    with pytest.raises(ValueError, match="FRESHDESK_SUBDOMAIN"):
        freshdesk._subdomain()


def test_base_url_is_composed_not_user_supplied(monkeypatch: pytest.MonkeyPatch):
    """The whole SSRF story for this connector is that the host is composed
    from a validated label, never taken from user input. Pin it.
    """
    monkeypatch.setenv("FRESHDESK_SUBDOMAIN", "acme")
    assert freshdesk._base_url() == "https://acme.freshdesk.com/api/v2"


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------


def test_api_key_strips_whitespace(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FRESHDESK_API_KEY", "  abc123\n")
    assert freshdesk._api_key() == "abc123"


@pytest.mark.parametrize("value", ["", "   "])
def test_api_key_missing_or_blank_is_actionable(
    monkeypatch: pytest.MonkeyPatch, value: str
):
    monkeypatch.setenv("FRESHDESK_API_KEY", value)
    with pytest.raises(ValueError, match="FRESHDESK_API_KEY"):
        freshdesk._api_key()


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------


def test_extract_error_detail_prefers_description():
    response = _FakeResponse(payload={"description": "Validation failed"})
    assert freshdesk._extract_error_detail(response) == "Validation failed"


def test_extract_error_detail_includes_per_field_errors():
    """Freshdesk's 400 body carries the actionable part in `errors`, not in
    `description` (which is the generic "Validation failed"). Dropping the
    field list would turn a fixable mistake into an opaque failure.
    """
    response = _FakeResponse(
        payload={
            "description": "Validation failed",
            "errors": [
                {
                    "field": "priority",
                    "message": "is not a valid value",
                    "code": "invalid_value",
                }
            ],
        }
    )
    detail = freshdesk._extract_error_detail(response)
    assert detail is not None
    assert "Validation failed" in detail
    assert "priority" in detail
    assert "is not a valid value" in detail


def test_extract_error_detail_handles_message_only_body():
    response = _FakeResponse(payload={"message": "Access denied"})
    assert freshdesk._extract_error_detail(response) == "Access denied"


def test_extract_error_detail_returns_none_for_non_json():
    response = _FakeResponse(text="<html>502 Bad Gateway</html>")
    assert freshdesk._extract_error_detail(response) is None


def test_extract_error_detail_returns_none_for_non_dict_json():
    response = _FakeResponse(payload=["unexpected"])
    assert freshdesk._extract_error_detail(response) is None


# ---------------------------------------------------------------------------
# _request
# ---------------------------------------------------------------------------


@pytest.fixture
def configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRESHDESK_SUBDOMAIN", "acme")
    monkeypatch.setenv("FRESHDESK_API_KEY", "secret-key")


def test_request_uses_basic_auth_with_key_as_username(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """Freshdesk's REST API takes the API key as the Basic Auth *username*
    with an ignored password, not as a bearer token. Getting this wrong
    authenticates as nobody and returns 401 on every call.
    """
    captured: dict[str, Any] = {}

    def fake_request(**kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _FakeResponse(payload={"ok": True}, content=b"{}")

    monkeypatch.setattr(requests, "request", fake_request)
    freshdesk._request("GET", "/tickets")

    assert captured["auth"] == ("secret-key", "X")
    assert captured["url"] == "https://acme.freshdesk.com/api/v2/tickets"
    assert captured["timeout"] == freshdesk.DEFAULT_TIMEOUT_SECONDS


def test_request_drops_none_and_empty_params(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """An LLM passing status="" to mean "no filter" must not become ?status=."""
    captured: dict[str, Any] = {}

    def fake_request(**kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _FakeResponse(payload={}, content=b"{}")

    monkeypatch.setattr(requests, "request", fake_request)
    freshdesk._request(
        "GET", "/tickets", params={"status": "", "priority": None, "page": 2}
    )

    assert captured["params"] == {"page": 2}


def test_request_raises_with_detail_on_error_status(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    monkeypatch.setattr(
        requests,
        "request",
        lambda **_: _FakeResponse(
            status_code=404,
            payload={"description": "Resource not found"},
            content=b"{}",
        ),
    )
    with pytest.raises(RuntimeError) as excinfo:
        freshdesk._request("GET", "/tickets/999999999")

    message = str(excinfo.value)
    assert "404" in message
    assert "Resource not found" in message


def test_request_surfaces_rate_limit_retry_after(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """A 429 is the one error a caller can actually act on, and Freshdesk puts
    the wait in Retry-After. Without it the LLM only learns "rate limited" and
    has no basis for when to try again -- issue #1409 lists quota exhaustion as
    an error that must be actionable.
    """
    monkeypatch.setattr(
        requests,
        "request",
        lambda **_: _FakeResponse(
            status_code=429,
            payload={"description": "You have exceeded the limit"},
            content=b"{}",
            headers={"Retry-After": "42"},
        ),
    )
    with pytest.raises(RuntimeError) as excinfo:
        freshdesk._request("GET", "/tickets")

    message = str(excinfo.value)
    assert "429" in message
    assert "42" in message


def test_request_redacts_credentials_from_transport_errors(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """setup_proxy_env() exports whatever proxy the OS has configured, and a
    ProxyError echoes that URL -- which can embed user:pass@ credentials.
    """

    def raise_proxy_error(**_: Any) -> None:
        raise requests.RequestException(
            "ProxyError: https://bob:hunter2@proxy.internal:3128"
        )

    monkeypatch.setattr(requests, "request", raise_proxy_error)
    with pytest.raises(RuntimeError) as excinfo:
        freshdesk._request("GET", "/tickets")

    assert "hunter2" not in str(excinfo.value)


def test_request_returns_empty_dict_for_204(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """Freshdesk answers a successful DELETE with 204 and no body."""
    monkeypatch.setattr(
        requests,
        "request",
        lambda **_: _FakeResponse(status_code=204, content=b""),
    )
    assert freshdesk._request("DELETE", "/tickets/1") == {}


def test_request_rejects_non_json_success_body(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """A 200 carrying an HTML body means a gateway answered, not Freshdesk."""
    monkeypatch.setattr(
        requests,
        "request",
        lambda **_: _FakeResponse(status_code=200, text="<html>hi</html>"),
    )
    with pytest.raises(RuntimeError, match="non-JSON"):
        freshdesk._request("GET", "/tickets")


# ---------------------------------------------------------------------------
# Response envelopes
# ---------------------------------------------------------------------------


def test_success_and_error_envelopes_are_json():
    assert json.loads(freshdesk._success(tickets=[])) == {
        "status": "success",
        "tickets": [],
    }
    assert json.loads(freshdesk._error("nope")) == {
        "status": "error",
        "message": "nope",
    }


def test_clamp_per_page_bounds_to_freshdesk_maximum():
    assert freshdesk._clamp_per_page(0) == 1
    assert freshdesk._clamp_per_page(-5) == 1
    assert freshdesk._clamp_per_page(10) == 10
    assert freshdesk._clamp_per_page(5000) == freshdesk.MAX_PER_PAGE


def test_validated_list_rejects_non_list_payload():
    """Freshdesk list endpoints return a bare JSON array, not an envelope."""
    assert freshdesk._validated_list([], "/tickets") == []
    assert freshdesk._validated_list([{"id": 1}], "/tickets") == [{"id": 1}]
    with pytest.raises(RuntimeError, match="/tickets"):
        freshdesk._validated_list({"description": "nope"}, "/tickets")


def test_validated_dict_rejects_non_dict_payload():
    assert freshdesk._validated_dict({"id": 1}, "/tickets/1") == {"id": 1}
    with pytest.raises(RuntimeError, match="/tickets/1"):
        freshdesk._validated_dict([1], "/tickets/1")


# ---------------------------------------------------------------------------
# Catalog wiring
# ---------------------------------------------------------------------------


def test_registry_entry_declares_the_env_vars_this_module_reads():
    """The catalog row is what the connect dialog renders fields for, so a
    rename on either side silently produces a connector that collects the
    wrong values and then fails at tool-call time with "missing env".
    """
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "freshdesk"
    )
    assert row["transport"] == "stdio"
    assert row["provider_name"] is None
    assert row["category"] == "Support"
    assert row["launch_config"]["args"] == ["-m", "xagent.web.tools.mcp.freshdesk"]
    assert row["launch_config"]["required_env"] == [
        "FRESHDESK_SUBDOMAIN",
        "FRESHDESK_API_KEY",
    ]


def test_registry_entry_classifies_as_api_key():
    """classify_app_auth is the single source the backend connect gate and both
    frontend dialogs read; "unconnectable" here would mean no Connect button.
    """
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows
    from xagent.web.mcp_apps import classify_app_auth

    row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "freshdesk"
    )
    assert classify_app_auth(row["transport"], row["launch_config"]) == "api_key"


def test_request_hints_at_wrong_subdomain_on_bodyless_404(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """freshdesk.com is wildcard-resolved, so a mistyped subdomain answers
    every path with a body-less 404 from the edge rather than anything naming
    the real problem. Without the hint that is indistinguishable from a
    deleted ticket.
    """
    monkeypatch.setattr(
        requests,
        "request",
        lambda **_: _FakeResponse(status_code=404, text="", content=b""),
    )
    with pytest.raises(RuntimeError) as excinfo:
        freshdesk._request("GET", "/tickets")

    message = str(excinfo.value)
    assert "FRESHDESK_SUBDOMAIN" in message
    assert "acme" in message


def test_request_keeps_a_real_404_description_over_the_subdomain_hint(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """A genuine Freshdesk 404 names the missing resource; burying that under
    a subdomain hint would send the caller chasing the wrong problem.
    """
    monkeypatch.setattr(
        requests,
        "request",
        lambda **_: _FakeResponse(
            status_code=404,
            payload={"description": "Resource not found"},
            content=b"{}",
        ),
    )
    with pytest.raises(RuntimeError) as excinfo:
        freshdesk._request("GET", "/tickets/999999999")

    message = str(excinfo.value)
    assert "Resource not found" in message
    assert "FRESHDESK_SUBDOMAIN" not in message


def test_request_hints_at_credentials_on_bodyless_401(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    monkeypatch.setattr(
        requests,
        "request",
        lambda **_: _FakeResponse(status_code=401, text="", content=b""),
    )
    with pytest.raises(RuntimeError) as excinfo:
        freshdesk._request("GET", "/tickets")

    assert "FRESHDESK_API_KEY" in str(excinfo.value)


def test_subdomain_hint_does_not_leak_the_api_key(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """The hint interpolates the subdomain, which is adjacent to the key in
    the same env block -- pin that only the subdomain is echoed.
    """
    monkeypatch.setattr(
        requests,
        "request",
        lambda **_: _FakeResponse(status_code=404, text="", content=b""),
    )
    with pytest.raises(RuntimeError) as excinfo:
        freshdesk._request("GET", "/tickets")

    assert "secret-key" not in str(excinfo.value)
