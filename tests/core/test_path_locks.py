from xagent.core.path_locks import PathMutationLockRegistry


def test_guard_path_removes_unused_lock_after_context_exit(tmp_path):
    registry = PathMutationLockRegistry()
    path = tmp_path / "artifact.png"
    key = str(path.resolve())

    with registry.guard_path(path) as normalized_path:
        assert normalized_path == path.resolve()
        assert key in registry._locks

    assert registry._locks == {}


def test_guard_paths_removes_each_unique_lock_after_context_exit(tmp_path):
    registry = PathMutationLockRegistry()
    first_path = tmp_path / "first.txt"
    second_path = tmp_path / "second.txt"

    with registry.guard_paths([second_path, first_path, second_path]) as normalized:
        assert normalized == (
            second_path.resolve(),
            first_path.resolve(),
            second_path.resolve(),
        )
        assert set(registry._locks) == {
            str(first_path.resolve()),
            str(second_path.resolve()),
        }

    assert registry._locks == {}
