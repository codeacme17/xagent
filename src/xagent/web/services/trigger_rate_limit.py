"""Rate limiting for public trigger callbacks and trigger CRUD APIs.

Backed by the ``limits`` package: Redis storage when ``XAGENT_REDIS_URL`` is
configured (shared limits across workers), in-memory storage otherwise
(per-process limits, fine for development and single-process deployments).

Rate limiting runs before audit writes so hostile callback traffic cannot
amplify database writes.
"""

from __future__ import annotations

import logging
import os
import threading

from limits import RateLimitItem, parse
from limits.storage import MemoryStorage, storage_from_string
from limits.strategies import MovingWindowRateLimiter

from ...config import (
    get_redis_url,
    get_trigger_callback_ip_rate_limit,
    get_trigger_callback_rate_limit,
    get_trigger_crud_rate_limit,
    get_trusted_proxy_hops,
)

logger = logging.getLogger(__name__)

_CALLBACK_NAMESPACE = "trigger-callback"
_CALLBACK_IP_NAMESPACE = "trigger-callback-ip"
_CRUD_NAMESPACE = "trigger-crud"

_WORKER_COUNT_ENV_VARS = ("WEB_CONCURRENCY", "UVICORN_WORKERS", "GUNICORN_WORKERS")


class TriggerRateLimiter:
    """Moving-window limiter over Redis or in-process memory."""

    def __init__(self) -> None:
        redis_url = get_redis_url()
        if redis_url:
            self.storage = storage_from_string(redis_url)
            self.backend = "redis"
        else:
            self.storage = MemoryStorage()
            self.backend = "memory"
        self._limiter = MovingWindowRateLimiter(self.storage)
        self._callback_limit = _parse_rate(
            get_trigger_callback_rate_limit(), fallback="120/minute"
        )
        self._callback_ip_limit = _parse_rate(
            get_trigger_callback_ip_rate_limit(), fallback="600/minute"
        )
        self._crud_limit = _parse_rate(
            get_trigger_crud_rate_limit(), fallback="60/minute"
        )

    def hit_callback(self, callback_id: str, remote_ip: str | None) -> bool:
        """Count one callback request; False when the limit is exceeded.

        Two buckets must both admit: per callback id + IP (protects one
        trigger from one noisy caller) and per IP alone. Without the IP-only
        bucket, an attacker rotating random callback ids would get a fresh
        bucket per request and amplify audit writes without bound.
        """
        ip_key = remote_ip or "unknown"
        if not self._limiter.hit(
            self._callback_ip_limit, _CALLBACK_IP_NAMESPACE, ip_key
        ):
            return False
        key = f"{callback_id}:{ip_key}"
        return self._limiter.hit(self._callback_limit, _CALLBACK_NAMESPACE, key)

    def hit_crud(self, user_id: int) -> bool:
        """Count one trigger CRUD request; False when the limit is exceeded."""
        return self._limiter.hit(self._crud_limit, _CRUD_NAMESPACE, str(user_id))


def _parse_rate(value: str, *, fallback: str) -> RateLimitItem:
    try:
        return parse(value)
    except ValueError:
        logger.warning(
            "Invalid trigger rate limit %r; falling back to %s", value, fallback
        )
        return parse(fallback)


_lock = threading.Lock()
_limiter: TriggerRateLimiter | None = None


def get_trigger_rate_limiter() -> TriggerRateLimiter:
    global _limiter
    if _limiter is None:
        with _lock:
            if _limiter is None:
                _limiter = TriggerRateLimiter()
    return _limiter


def reset_trigger_rate_limiter() -> None:
    """Drop the cached limiter so new env configuration takes effect (tests)."""
    global _limiter
    with _lock:
        _limiter = None


def check_callback_rate_limit(callback_id: str, remote_ip: str | None) -> bool:
    return get_trigger_rate_limiter().hit_callback(callback_id, remote_ip)


def check_trigger_crud_rate_limit(user_id: int) -> bool:
    return get_trigger_rate_limiter().hit_crud(user_id)


def _configured_worker_count() -> int:
    for env_var in _WORKER_COUNT_ENV_VARS:
        value = (os.getenv(env_var) or "").strip()
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                continue
    return 1


def warn_if_rate_limits_are_per_process() -> None:
    """Startup check: multi-process deployments need Redis for shared limits.

    Never fails startup; in-memory limiting degrades to per-process limits.
    """
    if get_redis_url():
        return
    workers = _configured_worker_count()
    if workers > 1:
        logger.warning(
            "Trigger rate limiting is using in-memory storage with %s worker "
            "processes: limits apply per process, not globally. Configure "
            "XAGENT_REDIS_URL for shared rate limits.",
            workers,
        )


def remote_ip_from_request(request: object) -> str | None:
    """Derive the caller IP, honoring trusted reverse-proxy hops.

    With ``XAGENT_TRUSTED_PROXY_HOPS=N`` (N>0), the client address is taken
    from ``X-Forwarded-For`` counting N trusted entries from the right;
    otherwise the raw socket peer address is used.
    """
    client = getattr(request, "client", None)
    peer_host = getattr(client, "host", None)
    peer_ip = str(peer_host) if peer_host is not None else None

    hops = get_trusted_proxy_hops()
    if hops <= 0:
        return peer_ip

    headers = getattr(request, "headers", {})
    forwarded_for = headers.get("x-forwarded-for") if headers else None
    if not forwarded_for:
        return peer_ip

    entries = [
        entry.strip() for entry in str(forwarded_for).split(",") if entry.strip()
    ]
    if not entries:
        return peer_ip
    # The rightmost entry was appended by the nearest trusted proxy; walking
    # `hops` entries from the right lands on the last untrusted client.
    index = max(0, len(entries) - hops)
    return entries[index]
