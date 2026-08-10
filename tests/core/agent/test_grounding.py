from __future__ import annotations

from xagent.core.agent.grounding import grounding_rule


def test_grounding_rule_covers_quantitative_data() -> None:
    rule = grounding_rule()

    for term in (
        "entities",
        "incidents",
        "dates",
        "sources",
        "causal explanations",
        "quantitative data",
        "metrics",
        "figures",
        "statistics",
        "percentages",
        "table rows",
        "time series",
    ):
        assert term in rule
    assert "use an appropriate tool to verify" in rule
    assert "illustrative placeholders" in rule


def test_grounding_rule_without_tools_omits_tool_verification() -> None:
    rule = grounding_rule(can_call_tools=False)

    assert "use an appropriate tool" not in rule
    assert "invented values" in rule
    assert "quantitative data" in rule
    assert "illustrative placeholders" in rule
