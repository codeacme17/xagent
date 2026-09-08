"""Same-turn duplicate-write guard on the ReAct tool-execution path.

Covers xorbitsai/xagent#2217: a write-category tool call whose (tool,
arguments) pair already succeeded earlier in the same turn must not execute
again; the model receives a structured suppression envelope carrying the
prior result instead. Only tools that explicitly declare themselves as
writes are guarded — an MCP tool whose wire annotations classify as
DESTRUCTIVE, or an internal tool marked ``non_idempotent = True``. Tools
with an undeclared or read-only hint are exempt, so legitimate repeated
reads (status polling with identical args) keep executing.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from xagent.core.agent import ExecutionContext, ReActPattern
from xagent.core.agent.pattern.react.duplicate_write_guard import (
    DESTRUCTIVE_WRITE_HINT_VALUE,
    DUPLICATE_WRITE_SUPPRESSED_KEY,
    build_suppression_envelope,
    tool_requires_duplicate_write_guard,
)
from xagent.core.tools.adapters.vibe.mcp_adapter import MCPWriteHint


class CreateRecordArgs(BaseModel):
    title: str
    amount: int = 0


class FakeLLM:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeWriteTool:
    """A create-style tool whose declared write hint is configurable."""

    def __init__(
        self,
        *,
        write_hint: Any = MCPWriteHint.DESTRUCTIVE,
        fail_first: bool = False,
        name: str = "create_record",
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._fail_first = fail_first
        if write_hint is not None:
            self.write_hint = write_hint

        class Metadata:
            description = "Create a record in the external system."

        Metadata.name = name
        self.metadata = Metadata()

    def args_type(self) -> type[BaseModel]:
        return CreateRecordArgs

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        self.calls.append(dict(args))
        if self._fail_first and len(self.calls) == 1:
            return {"success": False, "error": "transient provider error"}
        return {"success": True, "record_id": f"rec-{len(self.calls)}"}


class FakeNonIdempotentInternalTool(FakeWriteTool):
    """Internal (non-MCP) tool explicitly marked non-idempotent."""

    non_idempotent = True

    def __init__(self) -> None:
        super().__init__(write_hint=None, name="submit_form")


def _tool_call_response(name: str, args: dict[str, Any], call_id: str) -> dict:
    return {
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
        "done": False,
    }


def _pattern() -> ReActPattern:
    return ReActPattern(
        max_iterations=8,
        repeated_tool_decision_after_consecutive_tool_calls=None,
        repeated_tool_decision_after_consecutive_work_tool_calls=None,
    )


def _run_twice_llm(
    name: str,
    first_args: dict[str, Any],
    second_args: dict[str, Any],
) -> FakeLLM:
    return FakeLLM(
        responses=[
            _tool_call_response(name, first_args, "call_1"),
            _tool_call_response(name, second_args, "call_2"),
            {"content": "Done.", "done": True},
        ]
    )


def _tool_results(context: ExecutionContext) -> list[Any]:
    return [
        message.metadata["raw_result"]
        for message in context.messages
        if message.role == "tool"
    ]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_destructive_hint_value_pins_the_mcp_enum() -> None:
    # The guard module duck-types the hint instead of importing the MCP
    # adapter; this pin keeps the string aligned with the enum it mirrors.
    assert DESTRUCTIVE_WRITE_HINT_VALUE == MCPWriteHint.DESTRUCTIVE.value


def test_only_explicit_writes_require_the_guard() -> None:
    assert tool_requires_duplicate_write_guard(
        FakeWriteTool(write_hint=MCPWriteHint.DESTRUCTIVE)
    )
    assert tool_requires_duplicate_write_guard(FakeNonIdempotentInternalTool())
    # UNDECLARED and READ_ONLY are exempt by user decision on #2217: an
    # unannotated MCP tool may be a legitimate identical-args poll loop.
    assert not tool_requires_duplicate_write_guard(
        FakeWriteTool(write_hint=MCPWriteHint.UNDECLARED)
    )
    assert not tool_requires_duplicate_write_guard(
        FakeWriteTool(write_hint=MCPWriteHint.READ_ONLY)
    )
    assert not tool_requires_duplicate_write_guard(FakeWriteTool(write_hint=None))


def test_non_boolean_non_idempotent_marker_does_not_guard() -> None:
    tool = FakeWriteTool(write_hint=None)
    tool.non_idempotent = "yes"  # type: ignore[attr-defined]
    assert not tool_requires_duplicate_write_guard(tool)


# ---------------------------------------------------------------------------
# Suppression through the ReAct loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_destructive_call_is_suppressed() -> None:
    args = {"title": "invoice", "amount": 7}
    llm = _run_twice_llm("create_record", args, dict(args))
    tool = FakeWriteTool()
    context = ExecutionContext()
    context.add_user_message("Create the invoice record")

    result = await _pattern().run(context=context, tools=[tool], llm=llm)

    assert result["success"] is True
    assert tool.calls == [args]

    tool_results = _tool_results(context)
    assert len(tool_results) == 2
    envelope = tool_results[1]
    assert envelope[DUPLICATE_WRITE_SUPPRESSED_KEY] is True
    assert envelope["success"] is True
    assert envelope["suppressed_duplicate_of"] == "call_1"
    assert envelope["result"] == {"success": True, "record_id": "rec-1"}


@pytest.mark.asyncio
async def test_key_order_does_not_defeat_the_guard() -> None:
    # The args hash canonicalizes via json.dumps(sort_keys=True): the same
    # arguments serialized in a different key order are still one write.
    llm = FakeLLM(
        responses=[
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "create_record",
                            "arguments": '{"title": "invoice", "amount": 7}',
                        },
                    }
                ],
                "done": False,
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "function": {
                            "name": "create_record",
                            "arguments": '{"amount": 7, "title": "invoice"}',
                        },
                    }
                ],
                "done": False,
            },
            {"content": "Done.", "done": True},
        ]
    )
    tool = FakeWriteTool()
    context = ExecutionContext()
    context.add_user_message("Create the record")

    await _pattern().run(context=context, tools=[tool], llm=llm)

    assert len(tool.calls) == 1


@pytest.mark.asyncio
async def test_different_args_both_execute() -> None:
    llm = _run_twice_llm(
        "create_record",
        {"title": "invoice", "amount": 7},
        {"title": "invoice", "amount": 8},
    )
    tool = FakeWriteTool()
    context = ExecutionContext()
    context.add_user_message("Create both records")

    await _pattern().run(context=context, tools=[tool], llm=llm)

    assert len(tool.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "write_hint",
    [MCPWriteHint.UNDECLARED, MCPWriteHint.READ_ONLY, None],
    ids=["undeclared", "read_only", "no_hint"],
)
async def test_unguarded_tools_repeat_identical_calls(write_hint: Any) -> None:
    args = {"title": "status-poll"}
    llm = _run_twice_llm("create_record", args, dict(args))
    tool = FakeWriteTool(write_hint=write_hint)
    context = ExecutionContext()
    context.add_user_message("Poll twice")

    await _pattern().run(context=context, tools=[tool], llm=llm)

    assert len(tool.calls) == 2


@pytest.mark.asyncio
async def test_failed_first_write_is_retried() -> None:
    args = {"title": "invoice"}
    llm = _run_twice_llm("create_record", args, dict(args))
    tool = FakeWriteTool(fail_first=True)
    context = ExecutionContext()
    context.add_user_message("Create the record")

    await _pattern().run(context=context, tools=[tool], llm=llm)

    # Only a call that *succeeded* earlier in the turn suppresses a repeat.
    assert len(tool.calls) == 2


@pytest.mark.asyncio
async def test_non_idempotent_internal_tool_is_guarded() -> None:
    args = {"title": "form"}
    llm = _run_twice_llm("submit_form", args, dict(args))
    tool = FakeNonIdempotentInternalTool()
    context = ExecutionContext()
    context.add_user_message("Submit the form")

    await _pattern().run(context=context, tools=[tool], llm=llm)

    assert tool.calls == [args]


@pytest.mark.asyncio
async def test_provider_id_reuse_does_not_defeat_the_guard() -> None:
    # Provider-supplied tool_call ids are not guaranteed unique. When the
    # model reuses the completed call's id for the identical repeat, the
    # guard must still find the completed record (the scan runs before the
    # repeat writes any ledger entry) and must not overwrite the genuine
    # result with the suppression envelope, so a third reuse still attaches
    # the original result.
    args = {"title": "invoice"}
    llm = FakeLLM(
        responses=[
            _tool_call_response("create_record", args, "call_1"),
            _tool_call_response("create_record", dict(args), "call_1"),
            _tool_call_response("create_record", dict(args), "call_1"),
            {"content": "Done.", "done": True},
        ]
    )
    tool = FakeWriteTool()
    context = ExecutionContext()
    context.add_user_message("Create the record")

    await _pattern().run(context=context, tools=[tool], llm=llm)

    assert len(tool.calls) == 1
    envelopes = _tool_results(context)[1:]
    assert len(envelopes) == 2
    for envelope in envelopes:
        assert envelope[DUPLICATE_WRITE_SUPPRESSED_KEY] is True
        assert envelope["suppressed_duplicate_of"] == "call_1"
        assert envelope["result"] == {"success": True, "record_id": "rec-1"}


@pytest.mark.asyncio
async def test_third_call_attaches_the_original_result() -> None:
    args = {"title": "invoice"}
    llm = FakeLLM(
        responses=[
            _tool_call_response("create_record", args, "call_1"),
            _tool_call_response("create_record", dict(args), "call_2"),
            _tool_call_response("create_record", dict(args), "call_3"),
            {"content": "Done.", "done": True},
        ]
    )
    tool = FakeWriteTool()
    context = ExecutionContext()
    context.add_user_message("Create the record")

    await _pattern().run(context=context, tools=[tool], llm=llm)

    assert len(tool.calls) == 1
    envelopes = _tool_results(context)[1:]
    # Both suppressions must point at the genuine execution, never at a
    # previous suppression envelope.
    for envelope in envelopes:
        assert envelope["suppressed_duplicate_of"] == "call_1"
        assert envelope["result"] == {"success": True, "record_id": "rec-1"}


@pytest.mark.asyncio
async def test_guard_survives_checkpoint_state_round_trip() -> None:
    args = {"title": "invoice"}
    first_llm = FakeLLM(
        responses=[
            _tool_call_response("create_record", args, "call_1"),
            {"content": "Created.", "done": True},
        ]
    )
    first_tool = FakeWriteTool()
    first_context = ExecutionContext()
    first_context.add_user_message("Create the record")
    first_pattern = _pattern()
    await first_pattern.run(context=first_context, tools=[first_tool], llm=first_llm)
    assert len(first_tool.calls) == 1

    # Ledger state survives a checkpoint round-trip. Without turn tracking
    # (no turn_id metadata anywhere, as in embeddings that never stamp one)
    # the guard scope degrades to the pattern execution, so the restored
    # ledger keeps suppressing the identical write. The stamped-turn variants
    # of this scenario are covered by the two tests below.
    resumed_pattern = _pattern()
    resumed_pattern.load_state(json.loads(json.dumps(first_pattern.get_state())))
    resumed_llm = FakeLLM(
        responses=[
            _tool_call_response("create_record", dict(args), "call_2"),
            {"content": "Done.", "done": True},
        ]
    )
    resumed_tool = FakeWriteTool()
    resumed_context = ExecutionContext()
    resumed_context.add_user_message("Create the record")

    await resumed_pattern.run(
        context=resumed_context, tools=[resumed_tool], llm=resumed_llm
    )

    assert resumed_tool.calls == []
    envelope = _tool_results(resumed_context)[0]
    assert envelope[DUPLICATE_WRITE_SUPPRESSED_KEY] is True
    assert envelope["suppressed_duplicate_of"] == "call_1"


@pytest.mark.asyncio
async def test_intra_turn_resume_suppresses_under_the_same_turn_id() -> None:
    args = {"title": "invoice"}
    first_llm = FakeLLM(
        responses=[
            _tool_call_response("create_record", args, "call_1"),
            {"content": "Created.", "done": True},
        ]
    )
    first_tool = FakeWriteTool()
    first_context = ExecutionContext()
    first_context.add_user_message("Create the record", metadata={"turn_id": "turn-1"})
    first_pattern = _pattern()
    await first_pattern.run(context=first_context, tools=[first_tool], llm=first_llm)
    assert len(first_tool.calls) == 1

    resumed_pattern = _pattern()
    resumed_pattern.load_state(json.loads(json.dumps(first_pattern.get_state())))
    resumed_llm = FakeLLM(
        responses=[
            _tool_call_response("create_record", dict(args), "call_2"),
            {"content": "Done.", "done": True},
        ]
    )
    resumed_tool = FakeWriteTool()
    resumed_context = ExecutionContext()
    resumed_context.add_user_message(
        "Create the record", metadata={"turn_id": "turn-1"}
    )

    await resumed_pattern.run(
        context=resumed_context, tools=[resumed_tool], llm=resumed_llm
    )

    assert resumed_tool.calls == []


@pytest.mark.asyncio
async def test_identical_write_in_a_later_turn_executes() -> None:
    # The guard must be strictly per-turn (#2217): an explicit user request
    # in a later turn of the same execution repeats the identical write.
    # resume/inject_user_message continue the execution with the ledger
    # restored but a fresh turn_id on the new user message.
    args = {"title": "invoice"}
    pattern = _pattern()
    tool = FakeWriteTool()
    context = ExecutionContext()

    context.add_user_message("Create the record", metadata={"turn_id": "turn-1"})
    first_llm = FakeLLM(
        responses=[
            _tool_call_response("create_record", args, "call_1"),
            {"content": "Created.", "done": True},
        ]
    )
    await pattern.run(context=context, tools=[tool], llm=first_llm)
    assert len(tool.calls) == 1

    context.add_user_message(
        "Create the exact same record again", metadata={"turn_id": "turn-2"}
    )
    second_llm = FakeLLM(
        responses=[
            _tool_call_response("create_record", dict(args), "call_2"),
            {"content": "Created again.", "done": True},
        ]
    )
    await pattern.run(context=context, tools=[tool], llm=second_llm)

    assert len(tool.calls) == 2


# ---------------------------------------------------------------------------
# Envelope shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_envelope_strips_reserved_transport_keys() -> None:
    # The ledger stores the raw execution return; reserved transport keys
    # must not reach the model nested inside the suppression envelope.
    args = {"title": "invoice"}

    class ReservedKeysTool(FakeWriteTool):
        async def run_json_async(self, tool_args: dict[str, Any]) -> Any:
            self.calls.append(dict(tool_args))
            return {
                "success": True,
                "record_id": "rec-1",
                "_xagent_context_refs": [],
                "_xagent_supersedes_scope": "records",
            }

    llm = _run_twice_llm("create_record", args, dict(args))
    tool = ReservedKeysTool()
    context = ExecutionContext()
    context.add_user_message("Create the record")

    await _pattern().run(context=context, tools=[tool], llm=llm)

    assert len(tool.calls) == 1
    envelope = _tool_results(context)[1]
    assert envelope[DUPLICATE_WRITE_SUPPRESSED_KEY] is True
    assert "_xagent_context_refs" not in envelope["result"]
    assert "_xagent_supersedes_scope" not in envelope["result"]
    assert envelope["result"]["record_id"] == "rec-1"


def test_suppression_envelope_shape() -> None:
    envelope = build_suppression_envelope(
        tool_name="create_record",
        prior_tool_call_id="call_1",
        prior_result={"success": True, "record_id": "rec-1"},
    )
    assert envelope["success"] is True
    assert envelope[DUPLICATE_WRITE_SUPPRESSED_KEY] is True
    assert envelope["tool_name"] == "create_record"
    assert envelope["suppressed_duplicate_of"] == "call_1"
    assert envelope["result"] == {"success": True, "record_id": "rec-1"}
    assert "already succeeded earlier in this turn" in envelope["message"]
