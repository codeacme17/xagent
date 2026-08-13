"""Make a payload survive a PostgreSQL ``jsonb`` column unchanged.

Two separate hazards, both introduced by the column type itself.

**Code points jsonb refuses to store.**

The trace payload columns are ``jsonb`` on PostgreSQL (#1248). ``jsonb``
decodes JSON escapes to native text on the way in, so it rejects two shapes
the previous ``json`` columns stored happily and then failed to read back
(#1149):

- the NUL escape ``\\u0000`` -- PostgreSQL text cannot carry NUL; and
- either half of an unpaired UTF-16 surrogate, which is not a Unicode
  scalar value and has no UTF-8 form.

Both reach this process through ordinary LLM or tool output: ``json.loads``
accepts the escapes without complaint and hands back Python strings carrying
the raw code points, and ``json.dumps`` re-emits them. Sanitizing at the
write boundary keeps the rule in one place instead of at every producer.

Offending code points become U+FFFD (REPLACEMENT CHARACTER), the designated
marker for undecodable input -- replacement rather than deletion, so the
mangling stays visible and adjacent text is not silently joined.

A surrogate code point in a Python ``str`` is always garbage here: a *valid*
pair never survives ``json.loads`` (it is combined into one astral
character), and even a high+low adjacency built by hand cannot be encoded
as UTF-8. So every surrogate is replaced, without pair-matching -- unlike
the read-side guard in ``web/api/monitor.py``, which inspects the *text*
form of stored JSON and must strip valid pairs before matching.

**Numbers jsonb stores but hands back as a different type.**

``jsonb`` parses numbers into ``numeric`` and re-renders them in plain
notation, while ``json`` keeps the literal text. A float that ``repr``
writes with an exponent -- ``1e+16`` -- therefore comes back from the
database as the *int* ``10000000000000000``. Value-wise that is the same
number, but it is not the same JSON, and the checkpoint blob path
(``web/services/trace_message_storage.py``) re-hashes payloads it reads
back and compares them against the hash stored at write time: a payload
carrying such a float would be rejected as corrupt on restore, costing
the task its checkpoint.

Normalizing here, at the same boundary and before the hash is computed,
makes the stored form and the read-back form identical. Every float at or
above 1e16 is integral (the mantissa runs out of fractional bits above
2**53), so the conversion loses nothing that ``jsonb`` would not have
taken anyway.

Rows written *before* the jsonb migration are not covered by this and may
still carry such a float; see that migration's docstring for why they are
not rewritten wholesale.
"""

from __future__ import annotations

import re
from typing import Any

REPLACEMENT_CHARACTER = "�"

# NUL plus the whole surrogate range. Raw ranges are safe in a character
# class; the pattern never leaves this module in source-escape form.
_UNSTORABLE_CODE_POINTS = re.compile("[\x00\ud800-\udfff]")

# Where repr switches a float to exponent notation. Below it, json and
# jsonb agree on the text; at or above it, jsonb re-renders in plain
# notation and json.loads reads an int back.
_EXPONENT_NOTATION_THRESHOLD = 1e16


def sanitize_json_payload(value: Any) -> Any:
    """Return ``value`` in the form the jsonb column will hand back.

    Unstorable code points become U+FFFD, and floats jsonb would return as
    ints are converted up front. Walks strings, dicts (keys included),
    lists, and tuples; every other type passes through untouched. A payload
    that needs no change is returned as the *same object* -- this runs on
    every trace write and almost every payload is clean, so the clean path
    must not copy.

    Two distinct dict keys can collide after replacement (``"a\\x00"`` and
    ``"a\\ud800"`` both become ``"a\\ufffd"``); the later key wins, matching
    ``json.loads`` duplicate-key behaviour.
    """
    if isinstance(value, str):
        if _UNSTORABLE_CODE_POINTS.search(value):
            return _UNSTORABLE_CODE_POINTS.sub(REPLACEMENT_CHARACTER, value)
        return value
    # bool is an int subclass, and True/False are not numbers to normalize.
    if isinstance(value, float) and not isinstance(value, bool):
        if abs(value) >= _EXPONENT_NOTATION_THRESHOLD and value.is_integer():
            return int(value)
        return value
    if isinstance(value, dict):
        changed = False
        cleaned_dict: dict[Any, Any] = {}
        for key, item in value.items():
            cleaned_key = sanitize_json_payload(key)
            cleaned_item = sanitize_json_payload(item)
            if cleaned_key is not key or cleaned_item is not item:
                changed = True
            cleaned_dict[cleaned_key] = cleaned_item
        return cleaned_dict if changed else value
    if isinstance(value, (list, tuple)):
        cleaned_items = [sanitize_json_payload(item) for item in value]
        if all(new is old for new, old in zip(cleaned_items, value)):
            return value
        if isinstance(value, tuple):
            return tuple(cleaned_items)
        return cleaned_items
    return value
