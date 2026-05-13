"""
Test for language matching functionality in result analyzer.
"""

import pytest

from xagent.core.agent.pattern.dag_plan_execute.result_analyzer import ResultAnalyzer
from xagent.core.agent.trace import Tracer
from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.types import ChunkType, StreamChunk


class MockLLM(BaseLLM):
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
        self._model_name = "mock_llm"

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    @property
    def abilities(self) -> list[str]:
        return ["chat"]

    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        return self._model_name

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return '{"achieved": true, "reason": "Default response"}'


class CaptureStreamLLM(BaseLLM):
    def __init__(self, *, supports_json_schema: bool):
        self.calls = []
        self._supports_json_schema = supports_json_schema

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    @property
    def abilities(self) -> list[str]:
        return ["chat"]

    @property
    def model_name(self) -> str:
        return "capture_stream_llm"

    @property
    def supports_json_schema_response_format(self) -> bool:
        return self._supports_json_schema

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        return "{}"

    async def stream_chat(self, messages: list[dict[str, str]], **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        yield StreamChunk(type=ChunkType.TOKEN, content="{}", delta="{}")
        yield StreamChunk(type=ChunkType.END, finish_reason="stop")


@pytest.mark.asyncio
async def test_result_analyzer_keeps_output_config_for_schema_provider():
    llm = CaptureStreamLLM(supports_json_schema=True)
    analyzer = ResultAnalyzer(llm, Tracer())
    output_config = {"format": {"type": "json_schema", "schema": {}}}

    await analyzer._call_llm_with_retry(
        messages=[{"role": "user", "content": "Analyze"}],
        output_config=output_config,
    )

    assert llm.calls[0]["kwargs"]["output_config"] == output_config
    assert "response_format" not in llm.calls[0]["kwargs"]
    assert "SYSTEM REMINDER" not in llm.calls[0]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_result_analyzer_uses_json_object_without_schema_support():
    llm = CaptureStreamLLM(supports_json_schema=False)
    analyzer = ResultAnalyzer(llm, Tracer())

    await analyzer._call_llm_with_retry(
        messages=[{"role": "user", "content": "Analyze"}],
        output_config={"format": {"type": "json_schema", "schema": {}}},
    )

    assert llm.calls[0]["kwargs"]["response_format"] == {"type": "json_object"}
    assert "output_config" not in llm.calls[0]["kwargs"]
    assert (
        "Return exactly one valid JSON object"
        in llm.calls[0]["messages"][-1]["content"]
    )
    assert "requested schema/example" in llm.calls[0]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_chinese_goal_response():
    """Test that Chinese goals get Chinese responses"""
    responses = [
        "任务已成功完成。我已经分析了所有执行步骤，目标已经达成。最终结果显示所有步骤都成功执行，没有发现任何错误。"
    ]

    llm = MockLLM(responses)
    tracer = Tracer()
    analyzer = ResultAnalyzer(llm, tracer)

    goal = "分析这个数据并生成中文报告"
    history = [
        {
            "results": [
                {
                    "status": "completed",
                    "step_name": "data_analysis",
                    "result": {"output": "数据分析完成"},
                },
                {
                    "status": "completed",
                    "step_name": "report_generation",
                    "result": {"output": "报告生成完成"},
                },
            ]
        }
    ]

    result = await analyzer.generate_final_answer(goal, history)

    # Verify the response is in Chinese
    assert "任务" in result or "分析" in result or "报告" in result
    assert "analysis" not in result.lower() or "task" not in result.lower()


@pytest.mark.asyncio
async def test_english_goal_response():
    """Test that English goals get English responses"""
    responses = [
        "Task completed successfully. I have analyzed all execution steps and the goal has been achieved. Final results show that all steps were executed successfully without any errors."
    ]

    llm = MockLLM(responses)
    tracer = Tracer()
    analyzer = ResultAnalyzer(llm, tracer)

    goal = "Analyze this data and generate an English report"
    history = [
        {
            "results": [
                {
                    "status": "completed",
                    "step_name": "data_analysis",
                    "result": {"output": "Data analysis completed"},
                },
                {
                    "status": "completed",
                    "step_name": "report_generation",
                    "result": {"output": "Report generation completed"},
                },
            ]
        }
    ]

    result = await analyzer.generate_final_answer(goal, history)

    # Verify the response is in English
    assert (
        "task" in result.lower()
        or "analysis" in result.lower()
        or "report" in result.lower()
    )
    assert "任务" not in result or "分析" not in result


@pytest.mark.asyncio
async def test_mixed_language_goal():
    """Test that mixed language goals are handled appropriately"""
    responses = [
        "我已经成功完成了数据分析任务。The data analysis has been completed successfully with all steps executed properly."
    ]

    llm = MockLLM(responses)
    tracer = Tracer()
    analyzer = ResultAnalyzer(llm, tracer)

    goal = "分析数据并完成 analysis task"
    history = [
        {
            "results": [
                {
                    "status": "completed",
                    "step_name": "data_analysis",
                    "result": {"output": "Data analysis completed"},
                },
                {
                    "status": "completed",
                    "step_name": "report_generation",
                    "result": {"output": "Report generation completed"},
                },
            ]
        }
    ]

    result = await analyzer.generate_final_answer(goal, history)

    # Verify the response handles mixed language appropriately
    assert len(result) > 0  # Should have some response


@pytest.mark.asyncio
async def test_failure_message_language():
    """Test that failure messages are in the same language as the goal"""
    responses = [
        "TASK FAILED: 无法完成数据分析，因为缺少必要的数据输入。请提供完整的数据集后重试。"
    ]

    llm = MockLLM(responses)
    tracer = Tracer()
    analyzer = ResultAnalyzer(llm, tracer)

    goal = "分析这个中文数据集"
    history = [
        {
            "results": [
                {
                    "status": "failed",
                    "step_name": "data_analysis",
                    "error": "Missing data",
                }
            ]
        }
    ]

    result = await analyzer.generate_final_answer(goal, history)

    # Verify the failure message is in Chinese
    assert result.startswith("TASK FAILED:")
    assert "无法" in result or "数据" in result or "缺少" in result


if __name__ == "__main__":
    pytest.main([__file__])
