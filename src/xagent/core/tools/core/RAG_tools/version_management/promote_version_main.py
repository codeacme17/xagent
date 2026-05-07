"""Promote version main functionality for version management.

This module provides functionality for promoting candidate versions
to main versions with cascade cleanup.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Union

from ..core.exceptions import VersionManagementError
from ..core.schemas import StepType
from .cascade_cleaner import cleanup_cascade
from .list_candidates import list_candidates
from .main_pointer_manager import get_main_pointer, set_main_pointer

logger = logging.getLogger(__name__)


def _resolve_step_type(step_type_input: Union[StepType, str]) -> StepType:
    """
    Resolves the step type, converting string inputs to StepType enum members.

    Args:
        step_type_input: The input step type, which can be a StepType enum or a string.

    Returns:
        The resolved StepType enum member.

    Raises:
        VersionManagementError: If the input string does not correspond to a valid
                                  StepType member, or if the input type is unsupported.
    """
    if isinstance(step_type_input, StepType):
        return step_type_input
    elif isinstance(step_type_input, str):
        try:
            return StepType(step_type_input)
        except ValueError:
            raise VersionManagementError(
                f"Invalid step_type string: '{step_type_input}'. Expected one of: "
                + ", ".join(["'" + s.value + "'" for s in StepType])
            )
    else:
        raise VersionManagementError(
            f"Unsupported step_type type: {type(step_type_input)}. Expected StepType or str."
        )


def _call_cleanup_cascade(
    collection: str,
    doc_id: str,
    step_type: StepType,
    technical_id: str,
    old_technical_id: Optional[str] = None,
    model_tag: Optional[str] = None,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, Any]:
    """
    Helper to call cleanup_cascade to compute deleted counts or execute cleanup.

    Args:
        collection: Collection name
        doc_id: Document ID
        step_type: Processing stage type
        technical_id: Technical ID of the new main version
        old_technical_id: Technical ID of the old main version (if exists)
        model_tag: Model tag for embed step type
        preview_only: If True, only return preview without executing
        confirm: If True, execute the promotion

    Returns:
        Dictionary of deleted counts
    """
    if step_type == StepType.PARSE:
        return cleanup_cascade(
            collection=collection,
            doc_id=doc_id,
            scope="parse",
            new_parse_hash=technical_id,
            old_parse_hash=old_technical_id,
            preview_only=preview_only,
            confirm=confirm,
        )
    elif step_type == StepType.CHUNK:
        return cleanup_cascade(
            collection=collection,
            doc_id=doc_id,
            scope="chunk",
            new_parse_hash=technical_id,
            old_parse_hash=old_technical_id,
            preview_only=preview_only,
            confirm=confirm,
        )
    elif step_type == StepType.EMBED:
        if not model_tag:
            raise VersionManagementError("model_tag is required for embed step")
        return cleanup_cascade(
            collection=collection,
            doc_id=doc_id,
            scope="embed",
            model_tag=model_tag,
            preview_only=preview_only,
            confirm=confirm,
        )
    else:
        step_type_str = (
            step_type.value if isinstance(step_type, StepType) else str(step_type)
        )
        raise VersionManagementError(f"Invalid step_type: {step_type_str}")


def _resolve_selected_id(
    collection: str,
    doc_id: str,
    step_type: StepType,
    selected_id: str,
    model_tag: Optional[str] = None,
) -> tuple[str, str]:
    """Resolve selected_id to technical_id and semantic_id.

    Args:
        collection: Collection name
        doc_id: Document ID
        step_type: Processing stage type
        selected_id: Selected ID (semantic or technical)
        model_tag: Model tag for embed stage (optional)

    Returns:
        Tuple of (technical_id, semantic_id)

    Raises:
        VersionManagementError: If selected_id cannot be resolved
    """
    try:
        # Get all candidates
        candidates_result = list_candidates(
            collection=collection,
            doc_id=doc_id,
            step_type=step_type,
            model_tag=model_tag,
        )

        candidates = candidates_result.get("candidates", [])

        if not candidates:
            raise VersionManagementError(f"No candidates found for {step_type}")

        # Try to find by technical_id first
        for candidate in candidates:
            if candidate["technical_id"] == selected_id:
                return candidate["technical_id"], candidate["semantic_id"]

        # Try to find by semantic_id
        for candidate in candidates:
            if candidate["semantic_id"] == selected_id:
                return candidate["technical_id"], candidate["semantic_id"]

        # If not found, raise error
        available_ids = [c["semantic_id"] for c in candidates]
        raise VersionManagementError(
            f"Selected ID '{selected_id}' not found. Available IDs: {available_ids}"
        )

    except Exception as e:
        if isinstance(e, VersionManagementError):
            raise
        raise VersionManagementError(f"Failed to resolve selected_id: {e}")


def _calculate_cleanup_plan(
    lancedb_dir: str,
    collection: str,
    doc_id: str,
    step_type: StepType,
    technical_id: str,
    model_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculate cleanup plan for promotion.

    Args:
        lancedb_dir: LanceDB directory path
        collection: Collection name
        doc_id: Document ID
        step_type: Processing stage type
        technical_id: Technical ID of the version to promote
        model_tag: Model tag for embed stage (optional)

    Returns:
        Cleanup plan with counts and details
    """
    try:
        # Get current main pointer
        current_pointer = get_main_pointer(
            collection, doc_id, step_type.value, model_tag
        )

        old_technical_id = None
        if current_pointer:
            old_technical_id = current_pointer["technical_id"]

        # Use unified planner in preview mode to compute counts
        deleted_counts = _call_cleanup_cascade(
            collection=collection,
            doc_id=doc_id,
            step_type=step_type,
            technical_id=technical_id,
            old_technical_id=old_technical_id,
            model_tag=model_tag,
            preview_only=True,
            confirm=False,
        )

        # Generate notes
        notes = []
        if step_type == StepType.PARSE and deleted_counts.get("chunks", 0) > 0:
            notes.append("Requires re-chunk/embed")
        elif step_type == StepType.CHUNK and deleted_counts.get("embeddings", 0) > 0:
            notes.append("Requires re-embed")

        return {
            "deleted_counts": deleted_counts,
            "notes": notes,
            "current_pointer": current_pointer,
            "new_technical_id": technical_id,
        }

    except Exception as e:
        raise VersionManagementError(f"Failed to calculate cleanup plan: {e}")


def promote_version_main(
    collection: str,
    doc_id: str,
    step_type: Union[StepType, str],
    selected_id: str,
    operator: Optional[str] = None,
    preview_only: bool = False,
    confirm: bool = False,
    model_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Promotes a candidate version to be the main version for a document and step type.

    This function handles resolving the selected candidate, calculating a cleanup plan
    (if an old main version exists), updating the main pointer, and performing
    cascade cleanup of outdated data.

    Args:
        collection: The name of the collection.
        doc_id: The ID of the document.
        step_type: The processing step type (e.g., StepType.PARSE, StepType.CHUNK, StepType.EMBED)
                   or its string representation.
        selected_id: The technical ID or semantic ID of the candidate to promote.
        operator: The user or system performing the promotion.
        preview_only: If True, only calculate the cleanup plan and do not make any changes.
        confirm: If True, execute the promotion including pointer update and cleanup.

    Returns:
        A dictionary containing the result of the promotion, including the new main pointer,
        deleted counts, and any relevant messages or notes.

    Raises:
        VersionManagementError: If any error occurs during the promotion process,
                                e.g., candidate not found, invalid step type, etc.
    """
    resolved_step_type = _resolve_step_type(step_type)

    # Validate and set operator
    if not operator:
        operator = os.environ.get("USER", "unknown")
    if len(operator) > 32:
        raise VersionManagementError("Operator name too long (max 32 characters)")

    try:
        # Get LanceDB directory (use default if not set)
        lancedb_dir = os.getenv("LANCEDB_DIR")
        if not lancedb_dir:
            # Use default LanceDB directory
            from ......providers.vector_store.lancedb import LanceDBConnectionManager

            lancedb_dir = LanceDBConnectionManager.get_default_lancedb_dir()

        # Resolve selected_id to technical_id and semantic_id
        technical_id, semantic_id = _resolve_selected_id(
            collection, doc_id, resolved_step_type, selected_id, model_tag
        )

        # Calculate cleanup plan
        cleanup_plan = _calculate_cleanup_plan(
            lancedb_dir, collection, doc_id, resolved_step_type, technical_id, model_tag
        )

        if preview_only or not confirm:
            message = (
                "Preview of promotion to be applied."
                if preview_only
                else "Set confirm=True to execute the promotion."
            )
            return {
                "promoted": False,
                "preview": True,
                "message": message,
                "main_pointer": {
                    "step_type": resolved_step_type.value,
                    "semantic_id": semantic_id,
                    "technical_id": technical_id,
                    "model_tag": model_tag,
                },
                "deleted_counts": cleanup_plan["deleted_counts"],
                "notes": cleanup_plan["notes"],
            }

        # Perform cascade cleanup
        old_technical_id = None
        if cleanup_plan["current_pointer"]:
            old_technical_id = cleanup_plan["current_pointer"]["technical_id"]

        deleted_counts = _call_cleanup_cascade(
            collection=collection,
            doc_id=doc_id,
            step_type=resolved_step_type,
            technical_id=technical_id,
            old_technical_id=old_technical_id,
            model_tag=model_tag,
            preview_only=False,
            confirm=True,
        )

        if not deleted_counts:
            raise VersionManagementError(
                f"[Promotion] No records deleted for {collection}/{doc_id}/{resolved_step_type.value}"
            )

        # Update main pointer
        set_main_pointer(
            lancedb_dir=lancedb_dir,
            collection=collection,
            doc_id=doc_id,
            step_type=resolved_step_type.value,
            semantic_id=semantic_id,
            technical_id=technical_id,
            model_tag=model_tag,
            operator=operator,
        )

        # Generate notes
        notes = []
        if resolved_step_type == StepType.PARSE and deleted_counts.get("chunks", 0) > 0:
            notes.append("Requires re-chunk/embed")
        elif (
            resolved_step_type == StepType.CHUNK
            and deleted_counts.get("embeddings", 0) > 0
        ):
            notes.append("Requires re-embed")

        logger.info(
            "Promoted version for %s/%s/%s to %s (operator: %s)",
            collection,
            doc_id,
            resolved_step_type.value,
            technical_id,
            operator,
        )

        return {
            "promoted": True,
            "preview": False,
            "main_pointer": {
                "step_type": resolved_step_type.value,
                "semantic_id": semantic_id,
                "technical_id": technical_id,
                "model_tag": model_tag,
            },
            "deleted_counts": deleted_counts,
            "notes": notes,
            "operator": operator,
        }

    except Exception as e:
        if isinstance(e, VersionManagementError):
            raise
        raise VersionManagementError(f"Failed to promote version main: {e}")
