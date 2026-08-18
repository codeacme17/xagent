"""Classification of the provider fault behind a durable-storage failure.

#1467 put the ``__cause__`` chain into the logs; this pins the structured
fields that make a burst of those logs aggregatable. The cases below are built
to the real shapes the S3 path produces: s3fs never surfaces botocore's
``ClientError`` directly, it maps recognized codes onto ``OSError`` subclasses
and hangs the original off ``__cause__`` (``s3fs.errors.translate_boto_error``),
so every realistic fault has the code two links below the wrap.
"""

from __future__ import annotations

import errno

from xagent.core.file_storage import ProviderFault, classify_provider_fault
from xagent.web.services.managed_file_ref import DurableStorageOperationError


def _client_error(code: str | None, status: int | None) -> Exception:
    """A botocore-shaped ``ClientError`` without requiring botocore."""
    error: dict[str, object] = {}
    if code is not None:
        error["Code"] = code
        error["Message"] = f"{code} occurred"
    response: dict[str, object] = {"Error": error}
    if status is not None:
        response["ResponseMetadata"] = {"HTTPStatusCode": status}

    class ClientError(Exception):
        def __init__(self) -> None:
            super().__init__(f"An error occurred ({code})")
            self.response = response

    return ClientError()


def _s3_chain(
    translated: BaseException, code: str | None, status: int | None
) -> DurableStorageOperationError:
    """Rebuild the real chain: wrap -> s3fs translation -> ClientError."""
    translated.__cause__ = _client_error(code, status)
    fault = DurableStorageOperationError(
        "Failed to write durable object: users/7/uploads/abc/report.txt"
    )
    fault.__cause__ = translated
    return fault


def test_throttle_two_links_down_is_found_and_marked_retryable() -> None:
    """The incident shape: SlowDown behind an s3fs OSError behind the wrap."""
    exc = _s3_chain(OSError(errno.EIO, "SlowDown occurred"), "SlowDown", 503)

    fault = classify_provider_fault(exc)

    assert fault.code == "SlowDown"
    assert fault.http_status == 503
    assert fault.retryable is True


def test_rejected_credential_is_marked_permanent() -> None:
    """AccessDenied arrives as a PermissionError; retrying can never help."""
    exc = _s3_chain(PermissionError("Access Denied"), "AccessDenied", 403)

    fault = classify_provider_fault(exc)

    assert fault.code == "AccessDenied"
    assert fault.retryable is False


def test_eio_alone_does_not_end_the_walk() -> None:
    """s3fs uses EIO for every code it cannot map, so EIO must not classify.

    Treating it as an answer would stop one link short of the ``ClientError``
    that actually names the fault -- here an unlisted code whose 503 still
    decides retryability.
    """
    exc = _s3_chain(OSError(errno.EIO, "unmapped"), "SomeNewS3Code", 503)

    fault = classify_provider_fault(exc)

    assert fault.code == "SomeNewS3Code"
    assert fault.retryable is True


def test_unlisted_code_without_a_status_stays_unclassified() -> None:
    """``None`` is not ``False``: unrecognized must not read as diagnosed."""
    exc = _s3_chain(OSError(errno.EIO, "mystery"), "SomeNewS3Code", None)

    fault = classify_provider_fault(exc)

    assert fault.code == "SomeNewS3Code"
    assert fault.retryable is None


def test_transport_failure_is_classified_by_class_name() -> None:
    """Pre-response botocore faults carry no ``.response`` to read."""

    class EndpointConnectionError(Exception):
        pass

    wrap = DurableStorageOperationError("Failed to sign durable object URL: k")
    wrap.__cause__ = EndpointConnectionError("Could not connect to the endpoint")

    fault = classify_provider_fault(wrap)

    assert fault.provider_class == "EndpointConnectionError"
    assert fault.retryable is True


def test_missing_credentials_is_permanent() -> None:
    class NoCredentialsError(Exception):
        pass

    wrap = DurableStorageOperationError("Failed to write durable object: k")
    wrap.__cause__ = NoCredentialsError("Unable to locate credentials")

    fault = classify_provider_fault(wrap)

    assert fault.retryable is False


def test_local_backend_errnos_split_by_whether_waiting_helps() -> None:
    """The filesystem backend fails as a plain OSError, with no provider code."""
    reset = DurableStorageOperationError("Failed to write durable object: k")
    reset.__cause__ = OSError(errno.ECONNRESET, "Connection reset by peer")
    assert classify_provider_fault(reset).retryable is True

    full = DurableStorageOperationError("Failed to write durable object: k")
    full.__cause__ = OSError(errno.ENOSPC, "No space left on device")
    # Permanent on purpose: retrying a full disk only amplifies load.
    assert classify_provider_fault(full).retryable is False
    assert classify_provider_fault(full).code == "ENOSPC"


def test_http_status_decides_when_no_code_is_recognized() -> None:
    for status, expected in ((429, True), (500, True), (503, True), (501, False)):
        exc = _s3_chain(OSError(errno.EIO, "x"), None, status)
        fault = classify_provider_fault(exc)
        assert fault.retryable is expected, f"status {status}"
        assert fault.http_status == status


def test_an_unrecognizable_chain_still_names_the_innermost_class() -> None:
    """Always return something groupable, with retryability left unknown."""

    class WeirdBackendError(Exception):
        pass

    wrap = DurableStorageOperationError("Failed to write durable object: k")
    wrap.__cause__ = WeirdBackendError("no idea")

    fault = classify_provider_fault(wrap)

    assert fault == ProviderFault(provider_class="WeirdBackendError")
    assert fault.retryable is None


def test_a_cyclic_chain_terminates() -> None:
    """A chain can loop; the walk is bounded so classification cannot hang."""
    first = DurableStorageOperationError("first")
    second = DurableStorageOperationError("second")
    first.__cause__ = second
    second.__cause__ = first

    fault = classify_provider_fault(first)

    assert fault.retryable is None


def test_context_is_followed_when_from_was_omitted() -> None:
    """Not every fsspec implementation chains explicitly with ``from``."""
    wrap = DurableStorageOperationError("Failed to write durable object: k")
    try:
        try:
            raise OSError(errno.ECONNREFUSED, "Connection refused")
        except OSError:
            raise wrap
    except DurableStorageOperationError as raised:
        fault = classify_provider_fault(raised)

    assert fault.retryable is True
    assert fault.code == "ECONNREFUSED"


def test_log_fields_render_only_what_is_known() -> None:
    """A ``None`` field must not appear as the string ``None`` in a log line."""
    rendered = ProviderFault(provider_class="WeirdBackendError").as_log_fields()
    assert rendered == "provider_class=WeirdBackendError"

    rendered = ProviderFault(
        provider_class="ClientError",
        code="SlowDown",
        http_status=503,
        retryable=True,
    ).as_log_fields()
    assert rendered == (
        "provider_class=ClientError provider_code=SlowDown "
        "provider_http_status=503 retryable=True"
    )


def test_a_suppressed_context_is_not_classified() -> None:
    """``raise ... from None`` hides the displaced exception; honour that.

    Classifying on a context the raiser deliberately suppressed would attribute
    the fault to an exception they judged unrelated -- and could report a
    confident ``retryable`` for something the storage layer never saw.
    """
    try:
        try:
            raise OSError(errno.ECONNREFUSED, "unrelated bookkeeping failure")
        except OSError:
            raise DurableStorageOperationError(
                "Failed to write durable object: k"
            ) from None
    except DurableStorageOperationError as raised:
        fault = classify_provider_fault(raised)

    assert fault.retryable is None
    assert fault.code is None
    assert fault.provider_class == "DurableStorageOperationError"


def test_a_non_dict_response_attribute_is_ignored() -> None:
    """``.response`` is not unique to botocore; requests-style objects have one.

    Reading it blindly would either crash on a non-subscriptable object or
    invent a code from an unrelated library's response.
    """

    class HttpLibError(Exception):
        def __init__(self) -> None:
            super().__init__("boom")
            self.response = object()

    wrap = DurableStorageOperationError("Failed to write durable object: k")
    inner = HttpLibError()
    inner.__cause__ = OSError(errno.ECONNRESET, "Connection reset by peer")
    wrap.__cause__ = inner

    fault = classify_provider_fault(wrap)

    # Fell through the unusable ``.response`` and kept descending.
    assert fault.code == "ECONNRESET"
    assert fault.retryable is True


def test_a_non_string_code_cannot_crash_the_diagnostic_path() -> None:
    """An unhashable ``Code`` must not turn a handled 503 into an unhandled 500.

    This module duck-types botocore's shape so foreign backends still classify,
    which means a library following that shape loosely can put a dict or list
    under ``Code``. Feeding one to a ``frozenset`` lookup raises
    ``TypeError: unhashable type``, and because classification runs *inside* the
    caller's ``except`` block while it builds the 503, that TypeError would
    escape and replace the response -- losing the diagnosis exactly when an
    incident is being investigated.
    """
    for bad_code in ({"nested": "value"}, ["SlowDown"], object(), 503):
        exc = _s3_chain(OSError(errno.EIO, "x"), None, None)
        # Reach past the helper to plant a value botocore would never send.
        client_error = exc.__cause__.__cause__  # type: ignore[union-attr]
        client_error.response["Error"]["Code"] = bad_code  # type: ignore[attr-defined]

        fault = classify_provider_fault(exc)

        # Unusable code is dropped rather than stringified into the log field.
        assert fault.code is None, bad_code
        assert fault.retryable is None, bad_code


def test_a_non_string_code_still_defers_to_the_http_status() -> None:
    """Dropping an unusable code must not discard the status beside it."""
    exc = _s3_chain(OSError(errno.EIO, "x"), None, 503)
    client_error = exc.__cause__.__cause__  # type: ignore[union-attr]
    client_error.response["Error"]["Code"] = {"nested": "value"}  # type: ignore[attr-defined]

    fault = classify_provider_fault(exc)

    assert fault.code is None
    assert fault.http_status == 503
    assert fault.retryable is True


def test_an_empty_code_is_treated_as_absent() -> None:
    exc = _s3_chain(OSError(errno.EIO, "x"), "", 500)

    fault = classify_provider_fault(exc)

    assert fault.code is None
    assert fault.retryable is True
