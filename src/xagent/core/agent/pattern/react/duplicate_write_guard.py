"""Same-turn duplicate-write guard for the ReAct tool-execution path.

Defends against a model re-issuing a write-category tool call with
byte-identical arguments after that call already succeeded in the same turn
(xorbitsai/xagent#2217): the repeat is not executed; the model receives a
structured envelope carrying the prior result instead.

Scope is deliberately narrow:

* Same turn only, enforced by turn_id equality on the ledger records: a
  resume or injected user message continues the execution — and its
  checkpointed ledger — under a fresh turn_id, so the guard holds across an
  intra-turn resume but never across turns; creating two identical records
  in different turns stays possible.
* Identical execution arguments only, compared via the ledger's canonical
  args hash.
* Only tools that *explicitly* declare themselves as writes. An MCP tool
  whose wire annotations classify as destructive, or an internal tool marked
  ``non_idempotent = True``. Undeclared and read-only hints are exempt: an
  unannotated MCP tool may be a legitimate identical-args poll loop, and
  suppressing it would hand the model stale data.
"""

from __future__ import annotations

from typing import Any

# Marker key stamped on every suppression envelope. Doubles as the signal
# that a ledger record is an envelope rather than a genuine execution, so a
# later duplicate always attaches the original result, never a suppression
# of a suppression.
DUPLICATE_WRITE_SUPPRESSED_KEY = "duplicate_write_suppressed"

# Mirrors ``MCPWriteHint.DESTRUCTIVE.value`` without importing the MCP
# adapter: this module sits under ``core.agent`` and duck-types the hint the
# same way the common tool contract does (``_write_hint_value`` in
# ``tools.adapters.vibe.base``). A pin test keeps the string aligned with
# the enum.
DESTRUCTIVE_WRITE_HINT_VALUE = "destructive"


def tool_requires_duplicate_write_guard(tool: Any) -> bool:
    """Whether ``tool`` explicitly declares itself a non-idempotent write.

    True for exactly two declarations, both opt-in:

    * ``tool.non_idempotent is True`` — an internal tool's own marker.
      ``is True`` so a truthy non-boolean never enrolls a tool by accident.
    * an MCP write hint whose value is ``destructive`` — the only
      classification produced from an explicit ``destructiveHint: true`` on
      the wire (see ``classify_write_hint``). UNDECLARED must stay exempt
      here even though confirmation gating treats it as a write: the safe
      direction inverts for deduplication, where suppressing an unannotated
      poll loop would silently serve stale results.
    """
    if getattr(tool, "non_idempotent", None) is True:
        return True

    hint = getattr(tool, "write_hint", None)
    if hint is None:
        return False
    value = getattr(hint, "value", None)
    if not isinstance(value, str):
        return False
    return value == DESTRUCTIVE_WRITE_HINT_VALUE


def build_suppression_envelope(
    *,
    tool_name: str,
    prior_tool_call_id: str,
    prior_result: Any,
) -> dict[str, Any]:
    """Build the model-facing envelope for a suppressed duplicate write.

    ``success: True`` deliberately: the requested effect exists — it was
    produced by the earlier call — so the loop's success accounting should
    treat this observation like the write it stands in for.
    """
    return {
        "success": True,
        DUPLICATE_WRITE_SUPPRESSED_KEY: True,
        "tool_name": tool_name,
        "suppressed_duplicate_of": prior_tool_call_id,
        "result": prior_result,
        "message": (
            f"Duplicate write suppressed: this exact {tool_name} call "
            "already succeeded earlier in this turn "
            f"(tool call {prior_tool_call_id}); it was not executed again. "
            "The previous call's result is attached under 'result' — use it "
            "instead of retrying. Repeating this write with identical "
            "arguments is only possible in a later turn."
        ),
    }
