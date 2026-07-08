"""Relative-path containment tests for the workspace path resolvers.

Absolute-path resolution already re-checks that the target stays inside the
workspace. Relative paths were resolved against a base directory and returned
without a second check, so a ``../``-laden relative path could escape the
workspace once ``Path.resolve()`` collapsed the ``..`` segments. These tests
pin the containment re-check on every relative branch, across both
independent implementations (``TaskWorkspace`` and ``WorkspaceFileOperations``,
which do not delegate to one another).
"""

import pytest

from xagent.core.tools.core.workspace_file_tool import WorkspaceFileOperations
from xagent.core.workspace import TaskWorkspace

# --------------------------------------------------------------------------
# SITE 1 — TaskWorkspace.resolve_path / resolve_path_with_search
# --------------------------------------------------------------------------


def test_resolve_path_rejects_relative_traversal_out_of_workspace(tmp_path):
    workspace = TaskWorkspace("task7", str(tmp_path))

    with pytest.raises(ValueError):
        workspace.resolve_path("../../other/secret.txt")


def test_resolve_path_rejects_prefixed_relative_traversal(tmp_path):
    workspace = TaskWorkspace("task7", str(tmp_path))

    # An "output/"-prefixed path that then climbs out must still be rejected.
    with pytest.raises(ValueError):
        workspace.resolve_path("output/../../../escape.txt")


def test_resolve_path_allows_legitimate_relative_path(tmp_path):
    workspace = TaskWorkspace("task7", str(tmp_path))

    resolved = workspace.resolve_path("report.txt")

    assert resolved.is_relative_to(workspace.workspace_dir.resolve())
    assert resolved == (workspace.output_dir / "report.txt").resolve()


def test_resolve_path_with_search_rejects_existing_sibling_via_traversal(tmp_path):
    workspace = TaskWorkspace("task7", str(tmp_path))
    workspace.input_dir.mkdir(parents=True, exist_ok=True)

    # A real file outside the workspace that a traversal would otherwise reach:
    # input_dir/../../other/secret.txt == tmp_path/other/secret.txt.
    sibling = tmp_path / "other" / "secret.txt"
    sibling.parent.mkdir(parents=True, exist_ok=True)
    sibling.write_text("sibling end user's secret", encoding="utf-8")

    with pytest.raises(ValueError):
        workspace.resolve_path_with_search("../../other/secret.txt")


def test_resolve_path_with_search_rejects_nonexistent_traversal_without_oracle(
    tmp_path,
):
    # Containment is checked before existence, so a traversal to a path that
    # does NOT exist raises ValueError (the same as an existing target) rather
    # than FileNotFoundError. Otherwise the exception type would leak whether a
    # file exists outside the workspace.
    workspace = TaskWorkspace("task7", str(tmp_path))
    workspace.input_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError):
        workspace.resolve_path_with_search("../../other/does_not_exist.txt")


def test_resolve_path_with_search_finds_legitimate_file(tmp_path):
    workspace = TaskWorkspace("task7", str(tmp_path))
    target = workspace.output_dir / "data.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("a,b\n1,2\n", encoding="utf-8")

    resolved = workspace.resolve_path_with_search("data.csv")

    assert resolved == target.resolve()


def test_resolve_path_still_rejects_absolute_escape(tmp_path):
    # Regression: the absolute-path branch keeps rejecting out-of-workspace paths.
    workspace = TaskWorkspace("task7", str(tmp_path))
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError):
        workspace.resolve_path(str(outside))


# --------------------------------------------------------------------------
# SITE 2 — WorkspaceFileOperations._resolve_path (separate implementation;
# it ignores allowed_external_dirs and confines strictly to workspace_dir)
# --------------------------------------------------------------------------


def test_ops_resolve_path_rejects_relative_traversal(tmp_path):
    ops = WorkspaceFileOperations(TaskWorkspace("task7", str(tmp_path)))

    with pytest.raises(ValueError):
        ops._resolve_path("../../other/secret.txt", "output")


def test_ops_resolve_path_rejects_prefixed_relative_traversal(tmp_path):
    ops = WorkspaceFileOperations(TaskWorkspace("task7", str(tmp_path)))

    with pytest.raises(ValueError):
        ops._resolve_path("output/../../../escape.txt", "output")


def test_ops_resolve_path_allows_legitimate_relative(tmp_path):
    workspace = TaskWorkspace("task7", str(tmp_path))
    ops = WorkspaceFileOperations(workspace)

    resolved = ops._resolve_path("report.txt", "output")

    assert resolved.is_relative_to(workspace.workspace_dir.resolve())
    assert resolved == (workspace.output_dir / "report.txt").resolve()


def test_write_file_refuses_relative_traversal_end_to_end(tmp_path):
    workspace = TaskWorkspace("task7", str(tmp_path))
    ops = WorkspaceFileOperations(workspace)

    with pytest.raises(ValueError):
        ops.write_file("../../other/pwned.txt", "malicious content")

    # Nothing was written outside the workspace.
    assert not (tmp_path / "other" / "pwned.txt").exists()
