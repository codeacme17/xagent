"""Task-command audience classification contracts."""

from __future__ import annotations

import pytest

from xagent.web.services.task_command_contract import (
    TaskCommandAudience,
    TaskCommandKind,
    classify_task_command_audience,
)
from xagent.web.services.task_command_transport import (
    TaskCommandKind as TransportTaskCommandKind,
)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        pytest.param(
            {"type": "cancel_task"},
            TaskCommandAudience.INTERNAL,
            id="legacy-missing-scope",
        ),
        pytest.param(
            {"scope": "external"},
            TaskCommandAudience.EXTERNAL,
            id="external",
        ),
        pytest.param(
            {"scope": "unknown"},
            TaskCommandAudience.UNSUPPORTED,
            id="unknown-string",
        ),
        pytest.param(
            {"scope": None},
            TaskCommandAudience.UNSUPPORTED,
            id="none",
        ),
        pytest.param(
            {"scope": {}},
            TaskCommandAudience.UNSUPPORTED,
            id="dict",
        ),
        pytest.param(
            {"scope": []},
            TaskCommandAudience.UNSUPPORTED,
            id="list",
        ),
    ],
)
def test_cancel_audience_distinguishes_legacy_external_and_unsupported_scopes(
    payload: dict[str, object],
    expected: TaskCommandAudience,
) -> None:
    assert classify_task_command_audience(TaskCommandKind.CANCEL, payload) is expected


def test_transport_reexports_the_leaf_command_kind() -> None:
    assert TransportTaskCommandKind is TaskCommandKind


def test_non_cancel_commands_keep_the_internal_audience() -> None:
    assert (
        classify_task_command_audience(
            TaskCommandKind.PAUSE,
            {"scope": "external"},
        )
        is TaskCommandAudience.INTERNAL
    )
