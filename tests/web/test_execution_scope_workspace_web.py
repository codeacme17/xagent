"""Slice 3 of #757: scope-aware workspace paths in the web layer.

Covers ``_build_allowed_external_dirs`` scoping (shared by default,
scope-local under ``isolate_external_dirs``) and the websocket
output-path task-scope check tolerating scope segments between the user
root and the task dir.
"""


import pytest

from xagent.core.execution_scope import (
    ExecutionScope,
    set_execution_scope_resolver,
)
from xagent.web.api.chat import _build_allowed_external_dirs
from xagent.web.api.websocket import _output_path_in_current_task_scope


@pytest.fixture(autouse=True)
def _clear_resolver():
    set_execution_scope_resolver(None)
    yield
    set_execution_scope_resolver(None)


class TestAllowedExternalDirs:
    def test_unscoped_is_byte_identical(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path))
        dirs = _build_allowed_external_dirs(7)
        assert str(tmp_path / "user_7") in dirs

    def test_default_sharing_ignores_scope(self, monkeypatch, tmp_path):
        """isolate_external_dirs=False: every scope of the user still gets
        the shared user-level upload dir (already-uploaded KB files must not
        silently disappear under a new scope)."""
        monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path))
        scope = ExecutionScope(workspace_segments=("tenant-a",))
        assert _build_allowed_external_dirs(7, scope=scope) == (
            _build_allowed_external_dirs(7)
        )

    def test_isolation_builds_scope_local_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path))
        scope = ExecutionScope(
            workspace_segments=("tenant-a",), isolate_external_dirs=True
        )
        dirs = _build_allowed_external_dirs(7, scope=scope)
        assert str(tmp_path / "user_7" / "tenant-a") in dirs
        assert str(tmp_path / "user_7") not in dirs

    def test_isolation_flag_with_no_segments_stays_user_level(
        self, monkeypatch, tmp_path
    ):
        """Fields are independent: the flag with no segments isolates to the
        (segment-less) scoped root, which IS the user root."""
        monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path))
        scope = ExecutionScope(isolate_external_dirs=True)
        dirs = _build_allowed_external_dirs(7, scope=scope)
        assert str(tmp_path / "user_7") in dirs


class TestOutputPathTaskScopeCheck:
    def test_unscoped_layouts_unchanged(self):
        assert _output_path_in_current_task_scope(
            "user_1/web_task_5/output/a.txt", 5, 1
        )
        assert _output_path_in_current_task_scope("web_task_5/output/a.txt", 5, 1)
        assert not _output_path_in_current_task_scope(
            "user_1/web_task_6/output/a.txt", 5, 1
        )
        assert not _output_path_in_current_task_scope(
            "user_2/web_task_5/output/a.txt", 5, 1
        )

    def test_scoped_layout_accepted(self):
        assert _output_path_in_current_task_scope(
            "user_1/tenant-a/web_task_5/output/a.txt", 5, 1
        )
        assert _output_path_in_current_task_scope(
            "user_1/tenant-a/proj/web_task_5/output/a.txt", 5, 1
        )

    def test_scoped_layout_of_other_task_rejected(self):
        assert not _output_path_in_current_task_scope(
            "user_1/tenant-a/web_task_6/output/a.txt", 5, 1
        )
        assert not _output_path_in_current_task_scope(
            "user_1/tenant-a/web_task_5/input/a.txt", 5, 1
        )
