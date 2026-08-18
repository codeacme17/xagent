"""Classify the provider fault behind a failed durable-storage operation.

``ManagedFileRef`` wraps every provider exception into
``DurableStorageOperationError`` with only the storage key in its message, and
#1467 restored the ``__cause__`` chain to the logs. A traceback is readable but
not *queryable*: an operator triaging a burst of 503s wants to know whether
they are looking at one throttle, a thousand throttles, or a rejected
credential, and that question is a field aggregation, not a text search.

This module turns the chain into structured fields. It answers only "what was
it, and is retrying plausible" -- it does not decide HTTP status. Every fault
in this family still answers 503; see ``ProviderFault.retryable``.

**Why the chain must be walked.** s3fs does not surface botocore's
``ClientError``. Its ``translate_boto_error`` maps recognized codes onto
``OSError`` subclasses (``PermissionError`` for ``AccessDenied``, and so on),
falls back to ``OSError(EIO, ...)`` for the rest, and sets ``__cause__`` to the
original ``ClientError``. So the provider code lives two levels below the wrap:

    DurableStorageOperationError("Failed to write durable object: <key>")
      __cause__ -> PermissionError("Access Denied")          <- s3fs
        __cause__ -> ClientError(response={"Error": {"Code": "AccessDenied"}})

Only the innermost link carries the code, which is why classification starts at
the outermost exception and keeps descending.

**Why duck typing rather than ``except ClientError``.** botocore is declared
but the storage backend is pluggable (local filesystem, and the alternative
vector/object backends behind extras), so this code must not require botocore
to be importable, and must not silently stop classifying when a deployment
swaps s3fs for another fsspec implementation. Reading ``.response`` off
whatever is in the chain works for anything that follows botocore's shape and
degrades to ``retryable=None`` for anything that does not.
"""

from __future__ import annotations

import errno
from dataclasses import dataclass
from typing import Iterator

# Provider error codes worth another attempt: throttles, server-side
# transients, and aborted-in-flight requests. Sourced from the S3 error
# reference and botocore's own retry configuration.
_RETRYABLE_CODES = frozenset(
    {
        "BandwidthLimitExceeded",
        "IncompleteBody",
        "InternalError",
        "OperationAborted",
        "PriorRequestNotComplete",
        "RequestThrottled",
        "RequestThrottledException",
        "RequestTimeout",
        "RequestTimeTooSkewed",
        "ServiceUnavailable",
        "SlowDown",
        "Throttling",
        "ThrottlingException",
        "TooManyRequestsException",
        "TransactionInProgressException",
    }
)

# Codes that stay broken until a human changes configuration, permissions, or
# the request itself. Retrying these is the failure mode #1467 set out to make
# diagnosable: the client sees an indefinitely retryable 503 for a condition
# that will never clear on its own.
_PERMANENT_CODES = frozenset(
    {
        "AccessDenied",
        "AccountProblem",
        "AllAccessDisabled",
        "AuthorizationHeaderMalformed",
        "CredentialsNotSupported",
        "EntityTooLarge",
        "ExpiredToken",
        "InvalidAccessKeyId",
        "InvalidBucketName",
        "InvalidObjectState",
        "InvalidRequest",
        "InvalidSecurity",
        "InvalidToken",
        "KMS.DisabledException",
        "MethodNotAllowed",
        "NoSuchBucket",
        "NotImplemented",
        "SignatureDoesNotMatch",
        "TokenRefreshRequired",
    }
)

# Transport-level botocore/urllib3 faults, which carry no ``.response``.
# Matched on class name for the same reason the codes are duck-typed.
_RETRYABLE_EXC_NAMES = frozenset(
    {
        "ConnectionClosedError",
        "ConnectionError",
        "ConnectTimeoutError",
        "EndpointConnectionError",
        "IncompleteReadError",
        "ProtocolError",
        "ReadTimeoutError",
        "ResponseStreamingError",
        "TimeoutError",
    }
)

# Credential/config faults raised before a request is ever signed.
_PERMANENT_EXC_NAMES = frozenset(
    {
        "EndpointResolutionError",
        "NoCredentialsError",
        "NoRegionError",
        "ParamValidationError",
        "PartialCredentialsError",
        "ProfileNotFound",
        "UnknownServiceError",
    }
)


def _errnos(*names: str) -> frozenset[int]:
    """Resolve ``errno`` names, skipping any the platform does not define.

    Named lookup rather than attribute access because this module is imported
    by ``core.file_storage.__init__``, so an ``AttributeError`` here would fail
    every code path that touches storage -- a far worse outcome than losing one
    errno from a classification table on an exotic platform.
    """
    resolved = (getattr(errno, name, None) for name in names)
    return frozenset(value for value in resolved if isinstance(value, int))


# ``errno`` values for backends that fail as plain OS errors -- the local
# filesystem backend, and s3fs's unmapped-code fallback.
_RETRYABLE_ERRNOS = _errnos(
    "EAGAIN",
    "EBUSY",
    "ECONNABORTED",
    "ECONNREFUSED",
    "ECONNRESET",
    "EHOSTUNREACH",
    "ENETDOWN",
    "ENETUNREACH",
    "EPIPE",
    "ETIMEDOUT",
)

# ENOSPC and EDQUOT are permanent on purpose: the write cannot succeed until an
# operator frees space or raises a quota, so retrying only amplifies load.
_PERMANENT_ERRNOS = _errnos(
    "EACCES",
    "EDQUOT",
    "EFBIG",
    "EISDIR",
    "ENAMETOOLONG",
    "ENOSPC",
    "ENOTDIR",
    "EPERM",
    "EROFS",
)

# s3fs uses EIO for every provider code it does not recognize, so an EIO frame
# says nothing; the real code is on the ``ClientError`` below it. Treating EIO
# as classifiable would stop the walk one link short of the answer.
_UNINFORMATIVE_ERRNOS = _errnos("EIO")

_MAX_CHAIN_DEPTH = 12


@dataclass(frozen=True)
class ProviderFault:
    """What the object store actually said, as queryable fields.

    ``retryable`` is ``None`` when the chain could not be classified, which is
    deliberately distinct from ``False``: "we know this is permanent" and "we
    do not recognize this" call for different operator responses, and collapsing
    them would let an unrecognized fault masquerade as a diagnosed one.
    """

    provider_class: str
    code: str | None = None
    http_status: int | None = None
    retryable: bool | None = None

    def as_log_fields(self) -> str:
        """Render as ``key=value`` pairs for a structured log line."""
        return " ".join(
            f"{name}={value}"
            for name, value in (
                ("provider_class", self.provider_class),
                ("provider_code", self.code),
                ("provider_http_status", self.http_status),
                ("retryable", self.retryable),
            )
            if value is not None
        )


def _chain(exc: BaseException) -> Iterator[BaseException]:
    """Walk ``__cause__`` then ``__context__``, outermost first.

    ``__context__`` is followed as well because an exception raised inside an
    ``except`` block without ``from`` still records what it displaced, and
    fsspec implementations are not uniformly careful about explicit chaining.
    ``raise ... from None`` is respected, though: ``__suppress_context__``
    means the author deliberately hid what was displaced, and classifying on it
    anyway would attribute the fault to an exception the raiser judged
    unrelated. This mirrors what a printed traceback shows.

    Depth is bounded and identity is tracked: a chain can be cyclic.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    depth = 0
    while current is not None and depth < _MAX_CHAIN_DEPTH:
        if id(current) in seen:
            return
        seen.add(id(current))
        yield current
        depth += 1
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__suppress_context__:
            return
        else:
            current = current.__context__


def _retryable_from_status(status: int) -> bool | None:
    if status == 429:
        return True
    # 501 is a permanent "this backend cannot do that", unlike its 5xx peers.
    if status == 501:
        return False
    if 500 <= status <= 599:
        return True
    if 400 <= status <= 499:
        return False
    return None


def _classify_one(exc: BaseException) -> ProviderFault | None:
    """Classify a single link, or return ``None`` to keep descending."""
    provider_class = type(exc).__name__

    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error")
        # Every value read out of ``response`` is untrusted: this module
        # deliberately duck-types the botocore shape so foreign backends still
        # classify, which means a library that follows the shape only loosely
        # can put anything here. Narrowing each one to the type this function
        # actually uses -- rather than guarding at the point of use -- keeps a
        # nested dict or list out of both the set lookups below (unhashable,
        # and a raised TypeError here would replace the caller's 503 with an
        # unhandled 500 precisely while an incident is being diagnosed) and out
        # of the log field, where ``str()`` of a dict is noise, not a code.
        code = error.get("Code") if isinstance(error, dict) else None
        code = code if isinstance(code, str) and code else None
        metadata = response.get("ResponseMetadata")
        status = metadata.get("HTTPStatusCode") if isinstance(metadata, dict) else None
        status = status if isinstance(status, int) else None
        if code is not None or status is not None:
            return ProviderFault(
                provider_class=provider_class,
                code=code,
                http_status=status,
                retryable=_retryable_from_code(code, status),
            )

    if provider_class in _RETRYABLE_EXC_NAMES:
        return ProviderFault(provider_class=provider_class, retryable=True)
    if provider_class in _PERMANENT_EXC_NAMES:
        return ProviderFault(provider_class=provider_class, retryable=False)

    code_attr = getattr(exc, "errno", None)
    if isinstance(code_attr, int) and code_attr not in _UNINFORMATIVE_ERRNOS:
        if code_attr in _RETRYABLE_ERRNOS:
            retryable: bool | None = True
        elif code_attr in _PERMANENT_ERRNOS:
            retryable = False
        else:
            return None
        return ProviderFault(
            provider_class=provider_class,
            code=errno.errorcode.get(code_attr, str(code_attr)),
            retryable=retryable,
        )

    return None


def _retryable_from_code(code: str | None, status: int | None) -> bool | None:
    """Decide from the provider code, falling back to the HTTP status.

    ``code`` is already narrowed to ``str | None`` by the caller, which is what
    makes the set lookups below safe rather than a hazard.
    """
    if code in _RETRYABLE_CODES:
        return True
    if code in _PERMANENT_CODES:
        return False
    return None if status is None else _retryable_from_status(status)


def classify_provider_fault(exc: BaseException) -> ProviderFault:
    """Describe the provider fault behind ``exc``.

    Always returns a value: when nothing in the chain is recognized, the result
    still names the innermost exception class, which is the one worth grouping
    on. ``retryable`` stays ``None`` in that case.
    """
    links = list(_chain(exc))
    for link in links:
        classified = _classify_one(link)
        if classified is not None:
            return classified
    return ProviderFault(provider_class=type(links[-1]).__name__)
