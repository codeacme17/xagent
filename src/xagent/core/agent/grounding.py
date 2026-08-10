"""Shared anti-fabrication rules for prompts that emit user-facing answers.

Every prompt that produces a final user-facing answer must carry this rule;
paths without it have produced fabricated report data (issue #1235).
"""

from __future__ import annotations


def grounding_rule(*, can_call_tools: bool = True) -> str:
    """Return the grounding rule for user-facing answer generation.

    Args:
        can_call_tools: Whether the receiving LLM call may invoke work tools.
            When ``False`` (forced final answers, DAG completion assessment),
            the rule tells the model to state the gap instead of suggesting a
            tool call it cannot make.

    Returns:
        A prompt fragment forbidding unsupported claims and unsourced
        quantitative data, and requiring up-front disclosure of any
        illustrative figures.
    """
    insufficient_context_rule = (
        "If available context is insufficient, say so or use an appropriate "
        "tool to verify. "
        if can_call_tools
        else (
            "If available context is insufficient, say so instead of filling "
            "the gap with invented values. "
        )
    )
    return (
        "Do not introduce specific entities, incidents, dates, sources, "
        "causal explanations, or quantitative data (metrics, figures, "
        "statistics, percentages, table rows, or time series) that are not "
        "supported by the conversation, retrieved context, or tool results. "
        f"{insufficient_context_rule}"
        "If the answer includes figures that no tool result or provided "
        "context supports, disclose up front, before presenting them, that "
        "the figures are illustrative placeholders not drawn from any data "
        "source; never present invented numbers as real data."
    )
