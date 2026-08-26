from __future__ import annotations

import pytest

from xagent.core.tools.adapters.vibe.connector_runtime import REDACTED_RUNTIME_SECRET
from xagent.web.api.public_trace_events import normalize_public_trace_event


def test_normalize_public_trace_event_redacts_tool_runtime_secrets() -> None:
    event_type, data = normalize_public_trace_event(
        "tool_execution_start",
        {
            "tool_name": "shiftcare",
            "tool_args": {
                "headers": {
                    "Authorization": "Bearer public-stream-token",
                    "X-Account": "6185",
                },
                "connector_runtime": {
                    "secrets": {"authorization": "Bearer nested-token"},
                    "auth_selector": {"resource_owner_key": "xagent:user:owner"},
                },
            },
        },
    )

    assert event_type == "tool_execution_start"
    assert "public-stream-token" not in repr(data)
    assert "nested-token" not in repr(data)
    assert "xagent:user:owner" not in repr(data)
    assert data["tool_args"]["headers"]["Authorization"] == REDACTED_RUNTIME_SECRET
    assert data["tool_args"]["headers"]["X-Account"] == "6185"
    assert (
        data["tool_args"]["connector_runtime"]["auth_selector"]["resource_owner_key"]
        == REDACTED_RUNTIME_SECRET
    )


@pytest.mark.parametrize(
    "event_type",
    ["trace_error", "task_error_general", "step_error_general"],
)
def test_normalize_public_trace_event_redacts_general_failure_diagnostics(
    event_type: str,
) -> None:
    raw_error = "provider token=secret host=/srv/private"

    normalized_event_type, data = normalize_public_trace_event(
        event_type,
        {
            "status": "failed",
            "execution_id": "execution-1730",
            "pattern": "ReActPattern",
            "success": False,
            "error_type": "agent_error",
            "error": raw_error,
            "error_message": raw_error,
            "traceback": f"Traceback: {raw_error}",
            "context": {"messages": [{"content": raw_error}]},
        },
    )

    assert normalized_event_type == event_type
    assert raw_error not in repr(data)
    assert data == {
        "status": "failed",
        "execution_id": "execution-1730",
        "pattern": "ReActPattern",
        "success": False,
        "error_message": "Task execution failed.",
    }


@pytest.mark.parametrize(
    ("trace_event_type", "expected_event_type"),
    [("task", "task_error_general"), ("step", "step_error_general")],
)
def test_live_general_failure_trace_uses_redacted_public_event(
    trace_event_type: str,
    expected_event_type: str,
) -> None:
    from xagent.core.agent.trace import STEP_ERROR, TASK_ERROR, TraceEvent
    from xagent.web.api.ws_trace_handlers import WebSocketTraceHandler

    raw_error = "live provider token=secret"
    event = TraceEvent(
        TASK_ERROR if trace_event_type == "task" else STEP_ERROR,
        task_id="42",
        step_id="step-1" if trace_event_type == "step" else None,
        data={
            "status": "failed",
            "error_message": raw_error,
            "context": {"messages": [{"content": raw_error}]},
        },
    )

    stream_event = WebSocketTraceHandler(42)._convert_trace_event_to_stream_event(event)

    assert stream_event is not None
    assert stream_event["event_type"] == expected_event_type
    assert raw_error not in repr(stream_event)
    assert stream_event["data"] == {
        "status": "failed",
        "error_message": "Task execution failed.",
    }


def test_mcp_load_summary_audit_event_is_not_fanned_out() -> None:
    from xagent.core.agent.trace import SYSTEM_INFO, TraceEvent
    from xagent.web.api.ws_trace_handlers import WebSocketTraceHandler

    event = TraceEvent(
        event_type=SYSTEM_INFO,
        task_id="42",
        data={
            "__audit_only__": True,
            "event_type": "mcp_load_summary",
            "requested_servers": ["Gmail"],
            "loaded_servers": [],
            "failures": [{"server_name": "Gmail", "reason": "oauth_token_required"}],
            "successful_tool_count": 0,
        },
    )

    assert WebSocketTraceHandler(42)._convert_trace_event_to_stream_event(event) is None
