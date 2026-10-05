"""Temporary module databases must not affect later KB reference writers."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.web.services import test_read_surface_shape_contract as surface
from xagent.core.tools.core.RAG_tools.file.register_document import register_document
from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
    LanceDBVectorIndexStore,
)
from xagent.web.models import database


@pytest.mark.parametrize("setup_failure", [False, True])
def test_module_database_teardown_preserves_standalone_registration(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    setup_failure: bool,
) -> None:
    monkeypatch.setattr(database, "_SessionLocal", None)
    monkeypatch.setattr(database, "_engine", None)
    initialize = surface.init_db
    if setup_failure:

        def fail_after_binding(**kwargs):
            initialize(**kwargs)
            raise RuntimeError("module setup failed")

        monkeypatch.setattr(surface, "init_db", fail_after_binding)
    fixture_request = Mock(spec=pytest.FixtureRequest)
    lifecycle = surface._environment.__wrapped__(tmp_path_factory, fixture_request)
    try:
        if setup_failure:
            with pytest.raises(RuntimeError, match="module setup failed"):
                next(lifecycle)
        else:
            next(lifecycle)
            with pytest.raises(StopIteration):
                next(lifecycle)
    finally:
        lifecycle.close()
        for call in reversed(fixture_request.addfinalizer.call_args_list):
            call.args[0]()

    assert database.get_optional_session_local() is None
    source = tmp_path / "standalone.txt"
    source.write_text("A document registered after the module database closes.")
    result = register_document(
        collection="standalone", source_path=str(source), file_id="standalone-file"
    )
    assert result["created"] is True
    records = LanceDBVectorIndexStore().list_document_records_by_file_ids(
        ["standalone-file"]
    )
    assert [row.doc_id for row in records] == [result["doc_id"]]
