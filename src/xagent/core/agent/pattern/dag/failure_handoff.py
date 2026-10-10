"""What a DAG hands over when a step fails (xorbitsai/xagent#2850).

The model-written handoff after a step failure is one bounded call and can
come back empty, for instance cut off at its output cap. Completed step
results must still reach the user, so this module builds a handoff from them
without a model call. It also owns the one ERROR log line per failed step,
which is how a step failure becomes visible to error-level log searches.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from typing import Any

from .plan_generator import PlanStep

logger = logging.getLogger(__name__)


def log_step_failure(
    *,
    task_id: Any,
    step_id: str,
    code: Any,
    finish_reason: Any = None,
) -> None:
    logger.error(
        "DAG step failed: task_id=%s step_id=%s code=%s finish_reason=%s",
        task_id,
        step_id,
        code or "unknown",
        finish_reason or "none",
    )


def step_results_handoff(
    steps: Iterable[PlanStep],
    step_results: Mapping[str, Any],
    failed_step_id: str | None = None,
) -> dict[str, Any] | None:
    """``final_answer`` arguments built from completed step results.

    Returns ``None`` when no step completed: there is nothing to hand over.
    Every result is kept, including those of steps a replan has since dropped
    from the plan. Only step names and their own results appear; error text
    does not.
    """
    if not step_results:
        return None
    steps_by_id = {step.id: step for step in steps}
    sections = ["Results of the completed steps:"]
    for step_id, result in step_results.items():
        step = steps_by_id.get(step_id)
        name = _step_name(step) if step is not None else step_id
        sections.append(f"### {name}\n\n{_render_result(result)}")
    unfinished = [step for step in steps_by_id.values() if step.id not in step_results]
    if unfinished:
        sections.append(
            "Not completed:\n"
            + "\n".join(
                f"- {_step_name(step)}"
                + (" (failed)" if step.id == failed_step_id else "")
                for step in unfinished
            )
        )
    return {"answer": "\n\n".join(sections), "outcome": "partial"}


def _step_name(step: PlanStep) -> str:
    return (step.task or "").strip() or step.id


def _render_result(result: Any) -> str:
    if isinstance(result, str):
        text = result.strip()
    else:
        text = json.dumps(result, ensure_ascii=False, default=str)
    return text or "(no output)"
