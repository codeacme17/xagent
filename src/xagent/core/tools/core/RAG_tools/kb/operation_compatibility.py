"""Coordinator-owned KB operation outcome compatibility facade."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Optional
from uuid import uuid4


class RollbackStatus(StrEnum):
    """Explicit rollback state for a KB compatibility operation."""

    NOT_NEEDED = "not_needed"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    SKIPPED_BY_POLICY = "skipped_by_policy"


class PersistencePolicy(StrEnum):
    """Batch operation policy for successful child side effects."""

    PRESERVE_SUCCESSFUL_CHILDREN = "preserve_successful_children"
    ROLLBACK_ALL_CHILDREN = "rollback_all_children"


class SideEffectPlane(StrEnum):
    """Storage plane touched by a compensatable KB operation step."""

    COLLECTION = "collection"
    DOCUMENT = "document"
    PARSE = "parse"
    CHUNK = "chunk"
    EMBEDDING = "embedding"
    STATUS = "status"
    FILE = "file"
    WEB_PAGE = "web_page"


@dataclass(frozen=True)
class CompensationStep:
    """Structured description of a side effect and its compensation boundary."""

    name: str
    plane: SideEffectPlane
    payload: Mapping[str, Any] = field(default_factory=dict)
    idempotency_key: Optional[str] = None


@dataclass(frozen=True)
class KBOperationOutcome:
    """Internal outcome model for rollback-aware KB compatibility operations."""

    operation_id: str
    operation_type: str
    collection: str
    status: str
    rollback_status: RollbackStatus
    persistence_policy: PersistencePolicy
    compensation_steps: tuple[CompensationStep, ...] = ()
    child_outcomes: tuple["KBOperationOutcome", ...] = ()
    warnings: tuple[str, ...] = ()
    side_effects_may_remain: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def compensation_plan(self) -> tuple[CompensationStep, ...]:
        """Return compensation steps in LIFO execution order."""
        return tuple(reversed(self.compensation_steps))


class KBOperation:
    """Mutable operation builder stored only in the current execution context."""

    def __init__(
        self,
        *,
        operation_type: str,
        collection: str,
        persistence_policy: PersistencePolicy,
        operation_id: Optional[str] = None,
        parent_operation_id: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.operation_id = operation_id or str(uuid4())
        self.operation_type = operation_type
        self.collection = collection
        self.persistence_policy = persistence_policy
        self.parent_operation_id = parent_operation_id
        self.details: dict[str, Any] = dict(details or {})
        self.compensation_steps: list[CompensationStep] = []
        self.child_outcomes: list[KBOperationOutcome] = []
        self.warnings: list[str] = []
        self.side_effects_may_remain = False
        self._idempotency_keys: set[str] = set()
        self._outcome: KBOperationOutcome | None = None

    @property
    def outcome(self) -> KBOperationOutcome | None:
        """Return the finalized outcome when available."""
        return self._outcome

    def has_side_effects(self) -> bool:
        """Return whether this operation or any child recorded side effects."""
        return bool(self.compensation_steps) or any(
            child.side_effects_may_remain or child.compensation_steps
            for child in self.child_outcomes
        )

    def update_details(self, **details: Any) -> None:
        """Merge operation metadata captured during execution."""
        self.details.update(details)

    def record_side_effect(
        self,
        *,
        name: str,
        plane: SideEffectPlane,
        payload: Optional[Mapping[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> None:
        """Register an idempotent compensation boundary for one side effect."""
        step_payload = dict(payload or {})
        dedupe_key = idempotency_key or f"{plane.value}:{name}:{step_payload!r}"
        if dedupe_key in self._idempotency_keys:
            return

        self._idempotency_keys.add(dedupe_key)
        self.compensation_steps.append(
            CompensationStep(
                name=name,
                plane=plane,
                payload=step_payload,
                idempotency_key=dedupe_key,
            )
        )

    def add_child_outcome(self, outcome: KBOperationOutcome) -> None:
        """Attach a finalized child operation outcome."""
        self.child_outcomes.append(outcome)

    def finish(
        self,
        *,
        status: str,
        rollback_status: RollbackStatus | None = None,
        side_effects_may_remain: Optional[bool] = None,
        warnings: Optional[tuple[str, ...]] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> KBOperationOutcome:
        """Finalize and return the immutable operation outcome."""
        if details:
            self.details.update(details)
        if warnings:
            self.warnings.extend(warnings)

        if side_effects_may_remain is None:
            side_effects_may_remain = self._infer_side_effects_may_remain(status)

        if rollback_status is None:
            rollback_status = self._infer_rollback_status(
                status,
                side_effects_may_remain=side_effects_may_remain,
            )

        self.side_effects_may_remain = side_effects_may_remain
        self._outcome = KBOperationOutcome(
            operation_id=self.operation_id,
            operation_type=self.operation_type,
            collection=self.collection,
            status=status,
            rollback_status=rollback_status,
            persistence_policy=self.persistence_policy,
            compensation_steps=tuple(self.compensation_steps),
            child_outcomes=tuple(self.child_outcomes),
            warnings=tuple(self.warnings),
            side_effects_may_remain=side_effects_may_remain,
            details=dict(self.details),
        )
        return self._outcome

    def _infer_side_effects_may_remain(self, status: str) -> bool:
        if status == "success":
            return False
        return self.has_side_effects()

    def _infer_rollback_status(
        self,
        status: str,
        *,
        side_effects_may_remain: bool,
    ) -> RollbackStatus:
        if status == "success":
            return RollbackStatus.NOT_NEEDED

        child_statuses = {child.status for child in self.child_outcomes}
        has_successful_child = "success" in child_statuses
        has_failed_child = any(
            child_status != "success" for child_status in child_statuses
        )
        if (
            self.persistence_policy is PersistencePolicy.PRESERVE_SUCCESSFUL_CHILDREN
            and has_successful_child
            and has_failed_child
        ):
            return RollbackStatus.SKIPPED_BY_POLICY

        if side_effects_may_remain:
            return RollbackStatus.INCOMPLETE
        return RollbackStatus.NOT_NEEDED


_CURRENT_OPERATION: ContextVar[KBOperation | None] = ContextVar(
    "xagent_kb_current_operation",
    default=None,
)

_LAST_OUTCOME: ContextVar[KBOperationOutcome | None] = ContextVar(
    "xagent_kb_last_outcome",
    default=None,
)


def _format_exception_warning(exc: BaseException) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


class KBOperationCompatibilityFacade:
    """Compatibility facade for rollback-aware coordinator operations.

    The model is intentionally internal: public pipeline/API schemas stay stable
    while compatibility facades can record operation outcomes and child side
    effects for future handle-level compensation.
    """

    @property
    def last_outcome(self) -> KBOperationOutcome | None:
        """Return the most recently finalized operation outcome."""
        return _LAST_OUTCOME.get()

    def current_operation(self) -> KBOperation | None:
        """Return the operation active in the current context, if any."""
        return _CURRENT_OPERATION.get()

    @contextmanager
    def start_operation(
        self,
        *,
        operation_type: str,
        collection: str,
        persistence_policy: PersistencePolicy = (
            PersistencePolicy.PRESERVE_SUCCESSFUL_CHILDREN
        ),
        details: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[KBOperation]:
        """Start a root or nested operation and attach nested outcomes to parents."""
        parent_operation = _CURRENT_OPERATION.get()
        operation = KBOperation(
            operation_type=operation_type,
            collection=collection,
            persistence_policy=persistence_policy,
            parent_operation_id=(
                parent_operation.operation_id if parent_operation is not None else None
            ),
            details=details,
        )
        token = _CURRENT_OPERATION.set(operation)

        try:
            yield operation
        except BaseException as exc:  # noqa: BLE001 - record outcome before propagating
            if operation.outcome is None:
                operation.finish(
                    status="error",
                    rollback_status=(
                        RollbackStatus.INCOMPLETE
                        if operation.has_side_effects()
                        else RollbackStatus.NOT_NEEDED
                    ),
                    side_effects_may_remain=operation.has_side_effects(),
                    warnings=(_format_exception_warning(exc),),
                )
            raise
        finally:
            if operation.outcome is None:
                operation.finish(status="success")

            _CURRENT_OPERATION.reset(token)
            outcome = operation.outcome
            if outcome is not None:
                if parent_operation is not None:
                    parent_operation.add_child_outcome(outcome)
                _LAST_OUTCOME.set(outcome)

    @contextmanager
    def start_child_operation(
        self,
        *,
        operation_type: str,
        collection: str,
        persistence_policy: PersistencePolicy = (
            PersistencePolicy.PRESERVE_SUCCESSFUL_CHILDREN
        ),
        details: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[KBOperation]:
        """Start an operation intended to be attached to the current parent."""
        with self.start_operation(
            operation_type=operation_type,
            collection=collection,
            persistence_policy=persistence_policy,
            details=details,
        ) as operation:
            yield operation
