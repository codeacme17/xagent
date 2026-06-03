"""Tests for KB operation rollback compatibility outcomes."""

from __future__ import annotations

from contextvars import Context

import pytest

from xagent.core.tools.core.RAG_tools.kb import (
    KBOperationCompatibilityFacade,
    RollbackStatus,
    SideEffectPlane,
)


def test_operation_compensation_steps_are_idempotent_and_lifo() -> None:
    facade = KBOperationCompatibilityFacade()

    with facade.start_operation(
        operation_type="document_ingestion",
        collection="demo",
    ) as operation:
        operation.record_side_effect(
            name="remove_document",
            plane=SideEffectPlane.DOCUMENT,
            payload={"doc_id": "doc-1"},
            idempotency_key="document:doc-1",
        )
        operation.record_side_effect(
            name="remove_parse",
            plane=SideEffectPlane.PARSE,
            payload={"parse_hash": "parse-1"},
            idempotency_key="parse:parse-1",
        )
        operation.record_side_effect(
            name="remove_document",
            plane=SideEffectPlane.DOCUMENT,
            payload={"doc_id": "doc-1"},
            idempotency_key="document:doc-1",
        )
        operation.finish(
            status="partial",
            rollback_status=RollbackStatus.INCOMPLETE,
            side_effects_may_remain=True,
        )

    outcome = facade.last_outcome

    assert outcome is not None
    assert [step.name for step in outcome.compensation_steps] == [
        "remove_document",
        "remove_parse",
    ]
    assert [step.name for step in outcome.compensation_plan] == [
        "remove_parse",
        "remove_document",
    ]
    assert outcome.rollback_status is RollbackStatus.INCOMPLETE
    assert outcome.side_effects_may_remain is True


def test_last_outcome_is_isolated_by_execution_context() -> None:
    facade = KBOperationCompatibilityFacade()
    initial_current_context_outcome = facade.last_outcome

    def run_operation(collection: str):
        with facade.start_operation(
            operation_type="document_ingestion",
            collection=collection,
        ):
            pass

        outcome = facade.last_outcome
        assert outcome is not None
        return outcome

    context_a = Context()
    context_b = Context()

    outcome_a = context_a.run(run_operation, "collection-a")
    outcome_b = context_b.run(run_operation, "collection-b")

    assert outcome_a.collection == "collection-a"
    assert outcome_b.collection == "collection-b"
    assert context_a.run(lambda: facade.last_outcome) is outcome_a
    assert context_b.run(lambda: facade.last_outcome) is outcome_b
    assert facade.last_outcome is initial_current_context_outcome


class _OperationCancelled(BaseException):
    pass


def test_operation_base_exception_records_error_outcome() -> None:
    facade = KBOperationCompatibilityFacade()

    with pytest.raises(_OperationCancelled):
        with facade.start_operation(
            operation_type="document_ingestion",
            collection="demo",
        ) as operation:
            operation.record_side_effect(
                name="remove_document",
                plane=SideEffectPlane.DOCUMENT,
                payload={"doc_id": "doc-1"},
                idempotency_key="document:doc-1",
            )
            raise _OperationCancelled("cancelled")

    outcome = facade.last_outcome
    assert outcome is not None
    assert outcome.status == "error"
    assert outcome.rollback_status is RollbackStatus.INCOMPLETE
    assert outcome.side_effects_may_remain is True
    assert outcome.warnings == ("_OperationCancelled: cancelled",)


def test_operation_exception_warning_includes_exception_type() -> None:
    facade = KBOperationCompatibilityFacade()

    with pytest.raises(KeyError):
        with facade.start_operation(
            operation_type="document_ingestion",
            collection="demo",
        ):
            raise KeyError("doc_id")

    outcome = facade.last_outcome
    assert outcome is not None
    assert outcome.status == "error"
    assert outcome.rollback_status is RollbackStatus.NOT_NEEDED
    assert outcome.warnings == ("KeyError: 'doc_id'",)
