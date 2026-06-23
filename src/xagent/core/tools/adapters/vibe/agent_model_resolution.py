"""Model resolution helpers for delegated agent tools."""

from typing import Any, Mapping

AGENT_MODEL_KEYS = ("general", "small_fast", "visual", "compact")


def resolve_agent_model_llms(
    db: Any,
    storage: Any,
    agent_models: Mapping[str, Any] | None,
    user_id: int,
) -> tuple[Any | None, Any | None, Any | None, Any | None]:
    """Resolve delegated-agent model slots with one database lookup."""
    if not agent_models:
        return None, None, None, None

    from .....web.models.model import Model as DBModel

    model_ids = [
        agent_models[model_key]
        for model_key in AGENT_MODEL_KEYS
        if agent_models.get(model_key)
    ]
    if not model_ids:
        return None, None, None, None

    models_by_id: dict[Any, Any] = {}
    for model in db.query(DBModel).filter(DBModel.id.in_(model_ids)).all():
        models_by_id[model.id] = model
        models_by_id[str(model.id)] = model

    resolved_llms = {}
    for model_key in AGENT_MODEL_KEYS:
        model_id = agent_models.get(model_key)
        if not model_id:
            continue
        model = models_by_id.get(model_id)
        if model:
            resolved_llms[model_key] = storage.get_llm_by_name_with_access(
                str(model.model_id), user_id
            )

    return (
        resolved_llms.get("general"),
        resolved_llms.get("small_fast"),
        resolved_llms.get("visual"),
        resolved_llms.get("compact"),
    )
