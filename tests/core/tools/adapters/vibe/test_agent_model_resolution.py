import os
import tempfile
from unittest.mock import Mock

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from xagent.core.tools.adapters.vibe.agent_model_resolution import (
    resolve_agent_model_llms,
)
from xagent.web.models.database import Base
from xagent.web.models.model import Model


def _create_session() -> tuple[Session, str]:
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db.close()
    db_url = f"sqlite:///{temp_db.name}"
    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return session_local(), temp_db.name


def test_resolve_agent_model_llms_batches_configured_model_lookup() -> None:
    db, db_path = _create_session()
    try:
        models: dict[str, Model] = {}
        for role in ("general", "small_fast", "visual", "compact"):
            model = Model(
                model_id=f"{role}-model-id",
                category="llm",
                model_provider="openai",
                model_name=f"{role}-model",
                api_key="test-api-key",
                base_url="https://api.openai.com/v1",
                temperature=0.7,
                abilities=["chat"],
            )
            db.add(model)
            models[role] = model
        db.commit()
        for model in models.values():
            db.refresh(model)

        model_selects: list[str] = []
        engine = db.get_bind()

        def capture_model_selects(
            conn,
            cursor,
            statement,
            parameters,
            context,
            executemany,
        ) -> None:
            del conn, cursor, parameters, context, executemany
            normalized = " ".join(statement.lower().split())
            if normalized.startswith("select") and "from models" in normalized:
                model_selects.append(statement)

        llms_by_model_id = {
            model.model_id: Mock(name=f"{role}_llm") for role, model in models.items()
        }
        storage = Mock()
        storage.get_llm_by_name_with_access.side_effect = lambda model_id, user_id: (
            llms_by_model_id[model_id]
        )

        event.listen(engine, "before_cursor_execute", capture_model_selects)
        try:
            resolved = resolve_agent_model_llms(
                db,
                storage,
                {role: model.id for role, model in models.items()},
                user_id=42,
            )
        finally:
            event.remove(engine, "before_cursor_execute", capture_model_selects)

        assert resolved == (
            llms_by_model_id["general-model-id"],
            llms_by_model_id["small_fast-model-id"],
            llms_by_model_id["visual-model-id"],
            llms_by_model_id["compact-model-id"],
        )
        assert [
            call.args for call in storage.get_llm_by_name_with_access.call_args_list
        ] == [
            (models["general"].model_id, 42),
            (models["small_fast"].model_id, 42),
            (models["visual"].model_id, 42),
            (models["compact"].model_id, 42),
        ]
        assert len(model_selects) == 1
        assert " IN " in model_selects[0].upper()
    finally:
        db.close()
        try:
            os.remove(db_path)
        except OSError:
            pass
