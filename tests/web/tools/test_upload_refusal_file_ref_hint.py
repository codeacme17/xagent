"""Path-guard refusals for the upload tools name what the agent can use.

Production runs (xorbitsai/xagent#2873) show the agent passing a relative path
or a durable-file materialization path, being refused, and only reaching the
``file:<file_id>`` form after several more failed attempts. The refusal itself
must name both ways forward, while still keeping host paths out of the message.
"""

from __future__ import annotations

import json

import pytest

from xagent.web.tools.mcp import google_drive, onedrive

_CASES = [
    pytest.param(
        onedrive._resolve_upload_file_path,
        "XAGENT_ONEDRIVE_FILE_ALLOWED_DIRS",
        id="onedrive",
    ),
    pytest.param(
        google_drive._resolve_upload_file_path,
        "XAGENT_GOOGLE_DRIVE_FILE_ALLOWED_DIRS",
        id="google-drive",
    ),
]


@pytest.fixture
def allowed_dir(tmp_path):
    path = tmp_path / "workspace"
    path.mkdir()
    return path


@pytest.mark.parametrize(("resolve", "env_var"), _CASES)
def test_refusal_names_absolute_workspace_path_and_file_ref(
    resolve, env_var, allowed_dir, tmp_path, monkeypatch
):
    monkeypatch.setenv(env_var, str(allowed_dir))
    outside = tmp_path / "xagent-materialized" / "abc" / "deck.pptx"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"deck")

    with pytest.raises(PermissionError) as excinfo:
        resolve(str(outside))

    message = str(excinfo.value)
    assert "outside the allowed" in message
    assert "absolute path inside the task workspace" in message
    assert "file:<file_id>" in message
    # The host layout still stays out of the model-facing message.
    assert str(tmp_path) not in message
    assert str(allowed_dir) not in message


@pytest.mark.parametrize(
    ("upload", "env_var"),
    [
        pytest.param(
            onedrive.onedrive_upload_file,
            "XAGENT_ONEDRIVE_FILE_ALLOWED_DIRS",
            id="onedrive",
        ),
        pytest.param(
            google_drive.google_drive_upload_file,
            "XAGENT_GOOGLE_DRIVE_FILE_ALLOWED_DIRS",
            id="google-drive",
        ),
    ],
)
def test_upload_tool_returns_the_hint_to_the_agent(
    upload, env_var, allowed_dir, tmp_path, monkeypatch
):
    # The refusal reaches the model through the tool's JSON error payload,
    # before any credential lookup or request is made.
    monkeypatch.setenv(env_var, str(allowed_dir))
    outside = tmp_path / "report.xlsx"
    outside.write_bytes(b"xlsx")

    payload = json.loads(upload(str(outside)))

    assert payload["status"] == "error"
    assert "file:<file_id>" in payload["message"]
    assert str(tmp_path) not in payload["message"]


@pytest.mark.parametrize(("resolve", "env_var"), _CASES)
def test_relative_path_refusal_carries_the_same_hint(
    resolve, env_var, allowed_dir, tmp_path, monkeypatch
):
    # A relative path resolves against the connector process's working
    # directory, not the task workspace, so it is refused like any other
    # outside path; the hint is what tells the agent to send an absolute one.
    monkeypatch.setenv(env_var, str(allowed_dir))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(PermissionError) as excinfo:
        resolve("output/report.xlsx")

    assert "file:<file_id>" in str(excinfo.value)


@pytest.mark.parametrize(("resolve", "env_var"), _CASES)
def test_containment_is_still_checked_before_existence(
    resolve, env_var, allowed_dir, tmp_path, monkeypatch
):
    monkeypatch.setenv(env_var, str(allowed_dir))
    existing = tmp_path / "exists.txt"
    existing.write_text("x")
    missing = tmp_path / "missing.txt"

    with pytest.raises(PermissionError) as existing_error:
        resolve(str(existing))
    with pytest.raises(PermissionError) as missing_error:
        resolve(str(missing))

    assert str(existing_error.value) == str(missing_error.value)


@pytest.mark.parametrize(("resolve", "env_var"), _CASES)
def test_file_inside_the_allowlist_is_still_accepted(
    resolve, env_var, allowed_dir, monkeypatch
):
    monkeypatch.setenv(env_var, str(allowed_dir))
    inside = allowed_dir / "output" / "report.xlsx"
    inside.parent.mkdir()
    inside.write_bytes(b"xlsx")

    assert resolve(str(inside)) == inside.resolve()
