"""Vector index backend selection (switchable vector store).

Resolve which :class:`~.contracts.VectorIndexStore` implementation to use from
environment. Only LanceDB is implemented today; additional backends register
here and in :meth:`StorageFactory.get_vector_index_store`.

Test isolation: pytest should call ``reset_rag_storage_for_tests`` in
``storage.factory`` instead of importing a specific provider; extend that
function when adding a backend so connection caches are cleared alongside
``StorageFactory.reset_all``.
"""

from __future__ import annotations

import os
from enum import StrEnum
from typing import Final

from ..core.exceptions import ConfigurationError

# Primary env var (namespaced to avoid collisions with other libs).
VECTOR_BACKEND_ENV: Final[str] = "XAGENT_VECTOR_BACKEND"

# Backward-compatible alias used in some deployments / docs.
VECTOR_BACKEND_ENV_LEGACY: Final[str] = "VECTOR_STORE_BACKEND"


class VectorBackend(StrEnum):
    """Supported or reserved vector index backends."""

    LANCEDB = "lancedb"
    MILVUS = "milvus"
    QDRANT = "qdrant"


def _parse_backend(raw: str) -> VectorBackend:
    """Parse and validate backend string."""
    key = raw.strip().lower()
    if not key:
        return VectorBackend.LANCEDB
    try:
        return VectorBackend(key)
    except ValueError as exc:
        allowed = ", ".join(sorted(b.value for b in VectorBackend))
        raise ConfigurationError(
            f"Invalid {VECTOR_BACKEND_ENV}={raw!r}. Choose one of: {allowed}."
        ) from exc


def get_configured_vector_backend() -> VectorBackend:
    """Read configured vector backend from the environment.

    Precedence: ``XAGENT_VECTOR_BACKEND``, then ``VECTOR_STORE_BACKEND``,
    then default ``lancedb``.

    Returns:
        Selected :class:`VectorBackend`.

    Raises:
        ConfigurationError: If the value is not a known backend name.
    """
    raw = os.environ.get(VECTOR_BACKEND_ENV)
    if raw is None or raw.strip() == "":
        raw = os.environ.get(VECTOR_BACKEND_ENV_LEGACY, "")
    return _parse_backend(raw)


def require_implemented_vector_backend(backend: VectorBackend) -> None:
    """Ensure the backend has a concrete :class:`~.contracts.VectorIndexStore`.

    Call from the factory before instantiating stores. Extend this function
    when adding Milvus, Qdrant, etc.

    Args:
        backend: Resolved backend.

    Raises:
        ConfigurationError: If the backend is known but not implemented yet.
    """
    if backend is VectorBackend.LANCEDB:
        return
    raise ConfigurationError(
        f"Vector backend {backend.value!r} is not implemented yet. "
        f"Set {VECTOR_BACKEND_ENV}=lancedb (default), or contribute a "
        f"{backend.value} implementation of VectorIndexStore."
    )
