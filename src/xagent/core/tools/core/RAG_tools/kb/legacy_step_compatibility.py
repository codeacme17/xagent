"""Legacy step compatibility facade."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..core.config import (
    DEFAULT_IMAGE_CONTEXT_SIZE,
    DEFAULT_TABLE_CONTEXT_SIZE,
    DEFAULT_TIKTOKEN_ENCODING,
)
from ..core.schemas import (
    ChunkStrategy,
    DenseSearchResponse,
    FusionConfig,
    HybridSearchResponse,
    ParseMethod,
    SparseSearchResponse,
)
from .operation_compatibility import (
    KBOperation,
    KBOperationCompatibilityFacade,
    PersistencePolicy,
    RollbackStatus,
    SideEffectPlane,
)

if TYPE_CHECKING:
    from .coordinator import KBCoordinator
    from .storage_shim import KBStorageShimCompatibilityFacade


class KBLegacyStepCompatibilityFacade:
    """Compatibility boundary for legacy KB step helper functions.

    Document registration, parse, chunk, and retrieval helper modules keep their
    historical import paths and sync/async behavior. The facade provides one
    coordinator-owned storage boundary while delegating to the current helper
    implementations.
    """

    def __init__(
        self,
        coordinator: KBCoordinator | None = None,
        storage_shim: KBStorageShimCompatibilityFacade | None = None,
        operation_compatibility: KBOperationCompatibilityFacade | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._storage_shim = storage_shim
        self._operation_compatibility = operation_compatibility

    def _active_storage_shim(self) -> KBStorageShimCompatibilityFacade | None:
        if self._storage_shim is not None:
            return self._storage_shim
        if self._coordinator is not None:
            return self._coordinator.storage_shim
        return None

    def _active_operation_facade(self) -> KBOperationCompatibilityFacade | None:
        if self._operation_compatibility is not None:
            return self._operation_compatibility
        if self._coordinator is not None:
            return self._coordinator.operation_compatibility
        return None

    @contextmanager
    def _operation_context(
        self, *, operation_type: str, collection: str
    ) -> Iterator[tuple[KBOperation | None, bool]]:
        operation_facade = self._active_operation_facade()
        if operation_facade is None:
            yield None, False
            return

        current_operation = operation_facade.current_operation()
        if current_operation is not None:
            yield current_operation, False
            return

        with operation_facade.start_operation(
            operation_type=operation_type,
            collection=collection,
            persistence_policy=PersistencePolicy.PRESERVE_SUCCESSFUL_CHILDREN,
        ) as operation:
            yield operation, True

    @contextmanager
    def _storage_context(self) -> Iterator[None]:
        storage_shim = self._active_storage_shim()
        if storage_shim is None:
            yield
            return

        from ..storage.factory import bind_storage_shim_for_current_context

        with bind_storage_shim_for_current_context(storage_shim):
            yield

    def register_document(
        self,
        collection: str,
        source_path: str,
        file_type: Optional[str] = None,
        doc_id: Optional[str] = None,
        uploaded_at: Optional[str] = None,
        user_id: Optional[int] = None,
        file_id: Optional[str] = None,
        metadata_source_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        from ..file.register_document import _register_document_public_impl

        with self._operation_context(
            operation_type="legacy_register_document", collection=collection
        ) as (operation, owns_operation):
            with self._storage_context():
                result = _register_document_public_impl(
                    collection=collection,
                    source_path=source_path,
                    file_type=file_type,
                    doc_id=doc_id,
                    uploaded_at=uploaded_at,
                    user_id=user_id,
                    file_id=file_id,
                    metadata_source_path=metadata_source_path,
                )
            self._record_document_side_effect(
                operation,
                collection=collection,
                source_path=source_path,
                file_id=file_id,
                user_id=user_id,
                result=result,
            )
            self._finish_owned_operation(operation, owns_operation, status="success")
            return result

    def get_document(self, db_dir: str, collection: str, doc_id: str) -> Optional[Any]:
        from ..file.register_document import _get_document_impl

        with self._storage_context():
            return _get_document_impl(db_dir, collection, doc_id)

    def list_documents(
        self, db_dir: str, collection: str, limit: int = 100
    ) -> list[Dict[str, Any]]:
        from ..file.register_document import _list_documents_impl

        with self._storage_context():
            return _list_documents_impl(db_dir, collection, limit)

    def parse_document(
        self,
        collection: str,
        doc_id: str,
        parse_method: ParseMethod,
        params: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        from ..parse.parse_document import _parse_document_impl

        with self._operation_context(
            operation_type="legacy_parse_document", collection=collection
        ) as (operation, owns_operation):
            with self._storage_context():
                result = _parse_document_impl(
                    collection=collection,
                    doc_id=doc_id,
                    parse_method=parse_method,
                    params=params,
                    user_id=user_id,
                    is_admin=is_admin,
                    progress_callback=progress_callback,
                )
            self._record_parse_side_effect(
                operation, collection=collection, doc_id=doc_id, result=result
            )
            self._finish_owned_operation(operation, owns_operation, status="success")
            return result

    def chunk_document(
        self,
        collection: str,
        doc_id: str,
        parse_hash: str,
        chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
        chunk_size: Optional[int] = 1000,
        chunk_overlap: int = 200,
        headers_to_split_on: Optional[List[Tuple[str, str]]] = None,
        separators: Optional[List[str]] = None,
        use_token_count: bool = False,
        tiktoken_encoding: str = DEFAULT_TIKTOKEN_ENCODING,
        enable_protected_content: bool = True,
        protected_patterns: Optional[List[str]] = None,
        table_context_size: int = DEFAULT_TABLE_CONTEXT_SIZE,
        image_context_size: int = DEFAULT_IMAGE_CONTEXT_SIZE,
        user_id: Optional[int] = None,
        is_admin: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from ..chunk.chunk_document import _chunk_document_impl

        with self._operation_context(
            operation_type="legacy_chunk_document", collection=collection
        ) as (operation, owns_operation):
            with self._storage_context():
                result = _chunk_document_impl(
                    collection=collection,
                    doc_id=doc_id,
                    parse_hash=parse_hash,
                    chunk_strategy=chunk_strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    headers_to_split_on=headers_to_split_on,
                    separators=separators,
                    use_token_count=use_token_count,
                    tiktoken_encoding=tiktoken_encoding,
                    enable_protected_content=enable_protected_content,
                    protected_patterns=protected_patterns,
                    table_context_size=table_context_size,
                    image_context_size=image_context_size,
                    user_id=user_id,
                    is_admin=is_admin,
                    **kwargs,
                )
            self._record_chunk_side_effect(
                operation,
                collection=collection,
                doc_id=doc_id,
                parse_hash=parse_hash,
                result=result,
            )
            self._finish_owned_operation(operation, owns_operation, status="success")
            return result

    def chunk_recursive(
        self,
        collection: str,
        doc_id: str,
        parse_hash: str,
        chunk_size: Optional[int] = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from ..chunk.chunk_document import _chunk_recursive_impl

        with self._operation_context(
            operation_type="legacy_chunk_recursive", collection=collection
        ) as (operation, owns_operation):
            with self._storage_context():
                result = _chunk_recursive_impl(
                    collection=collection,
                    doc_id=doc_id,
                    parse_hash=parse_hash,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=separators,
                    **kwargs,
                )
            self._record_chunk_side_effect(
                operation,
                collection=collection,
                doc_id=doc_id,
                parse_hash=parse_hash,
                result=result,
            )
            self._finish_owned_operation(operation, owns_operation, status="success")
            return result

    def chunk_markdown(
        self,
        collection: str,
        doc_id: str,
        parse_hash: str,
        chunk_size: Optional[int] = 1200,
        chunk_overlap: int = 200,
        headers_to_split_on: Optional[List[Tuple[str, str]]] = None,
        separators: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from ..chunk.chunk_document import _chunk_markdown_impl

        with self._operation_context(
            operation_type="legacy_chunk_markdown", collection=collection
        ) as (operation, owns_operation):
            with self._storage_context():
                result = _chunk_markdown_impl(
                    collection=collection,
                    doc_id=doc_id,
                    parse_hash=parse_hash,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    headers_to_split_on=headers_to_split_on,
                    separators=separators,
                    **kwargs,
                )
            self._record_chunk_side_effect(
                operation,
                collection=collection,
                doc_id=doc_id,
                parse_hash=parse_hash,
                result=result,
            )
            self._finish_owned_operation(operation, owns_operation, status="success")
            return result

    def chunk_fixed_size(
        self,
        collection: str,
        doc_id: str,
        parse_hash: str,
        chunk_size: Optional[int] = 1000,
        chunk_overlap: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from ..chunk.chunk_document import _chunk_fixed_size_impl

        with self._operation_context(
            operation_type="legacy_chunk_fixed_size", collection=collection
        ) as (operation, owns_operation):
            with self._storage_context():
                result = _chunk_fixed_size_impl(
                    collection=collection,
                    doc_id=doc_id,
                    parse_hash=parse_hash,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    **kwargs,
                )
            self._record_chunk_side_effect(
                operation,
                collection=collection,
                doc_id=doc_id,
                parse_hash=parse_hash,
                result=result,
            )
            self._finish_owned_operation(operation, owns_operation, status="success")
            return result

    def _record_document_side_effect(
        self,
        operation: KBOperation | None,
        *,
        collection: str,
        source_path: str,
        file_id: Optional[str],
        user_id: Optional[int],
        result: Dict[str, Any],
    ) -> None:
        if operation is None:
            return
        doc_id = result.get("doc_id")
        if not doc_id:
            return
        operation.record_side_effect(
            name="remove_registered_document",
            plane=SideEffectPlane.DOCUMENT,
            payload={
                "collection": collection,
                "doc_id": doc_id,
                "created": result.get("created"),
                "source_path": source_path,
                "file_id": file_id,
            },
            idempotency_key=f"document:{collection}:{doc_id}",
        )
        operation.record_side_effect(
            name="clear_ingestion_status",
            plane=SideEffectPlane.STATUS,
            payload={"collection": collection, "doc_id": doc_id, "user_id": user_id},
            idempotency_key=f"status:{collection}:{doc_id}",
        )

    def _record_parse_side_effect(
        self,
        operation: KBOperation | None,
        *,
        collection: str,
        doc_id: str,
        result: Dict[str, Any],
    ) -> None:
        if operation is None or result.get("written") is False:
            return
        parse_hash = result.get("parse_hash")
        if not parse_hash:
            return
        operation.record_side_effect(
            name="remove_parse_record",
            plane=SideEffectPlane.PARSE,
            payload={
                "collection": collection,
                "doc_id": doc_id,
                "parse_hash": parse_hash,
            },
            idempotency_key=f"parse:{collection}:{doc_id}:{parse_hash}",
        )

    def _record_chunk_side_effect(
        self,
        operation: KBOperation | None,
        *,
        collection: str,
        doc_id: str,
        parse_hash: str,
        result: Dict[str, Any],
    ) -> None:
        if operation is None or result.get("created") is False:
            return
        chunk_count = int(result.get("chunk_count", 0) or 0)
        if chunk_count <= 0:
            return
        operation.record_side_effect(
            name="remove_chunk_records",
            plane=SideEffectPlane.CHUNK,
            payload={
                "collection": collection,
                "doc_id": doc_id,
                "parse_hash": parse_hash,
                "chunk_count": chunk_count,
            },
            idempotency_key=f"chunk:{collection}:{doc_id}:{parse_hash}",
        )

    @staticmethod
    def _finish_owned_operation(
        operation: KBOperation | None,
        owns_operation: bool,
        *,
        status: str,
    ) -> None:
        if operation is None or not owns_operation or operation.outcome is not None:
            return
        operation.finish(
            status=status,
            rollback_status=RollbackStatus.NOT_NEEDED,
            side_effects_may_remain=False,
        )

    def search_dense(
        self,
        collection: str,
        model_tag: str,
        query_vector: List[float],
        *,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        readonly: bool = False,
        nprobes: Optional[int] = None,
        refine_factor: Optional[int] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> DenseSearchResponse:
        from ..retrieval.search_dense import _search_dense_impl

        with self._storage_context():
            return _search_dense_impl(
                collection=collection,
                model_tag=model_tag,
                query_vector=query_vector,
                top_k=top_k,
                filters=filters,
                readonly=readonly,
                nprobes=nprobes,
                refine_factor=refine_factor,
                user_id=user_id,
                is_admin=is_admin,
            )

    async def search_dense_async(
        self,
        collection: str,
        model_tag: str,
        query_vector: List[float],
        *,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        readonly: bool = False,
        nprobes: Optional[int] = None,
        refine_factor: Optional[int] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> DenseSearchResponse:
        from ..retrieval.search_dense import _search_dense_async_impl

        with self._storage_context():
            return await _search_dense_async_impl(
                collection=collection,
                model_tag=model_tag,
                query_vector=query_vector,
                top_k=top_k,
                filters=filters,
                readonly=readonly,
                nprobes=nprobes,
                refine_factor=refine_factor,
                user_id=user_id,
                is_admin=is_admin,
            )

    def search_sparse(
        self,
        collection: str,
        model_tag: str,
        query_text: str,
        *,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        readonly: bool = False,
        nprobes: Optional[int] = None,
        refine_factor: Optional[int] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> SparseSearchResponse:
        from ..retrieval.search_sparse import _search_sparse_impl

        with self._storage_context():
            return _search_sparse_impl(
                collection=collection,
                model_tag=model_tag,
                query_text=query_text,
                top_k=top_k,
                filters=filters,
                readonly=readonly,
                nprobes=nprobes,
                refine_factor=refine_factor,
                user_id=user_id,
                is_admin=is_admin,
            )

    async def search_sparse_async(
        self,
        collection: str,
        model_tag: str,
        query_text: str,
        *,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        readonly: bool = False,
        nprobes: Optional[int] = None,
        refine_factor: Optional[int] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> SparseSearchResponse:
        from ..retrieval.search_sparse import _search_sparse_async_impl

        with self._storage_context():
            return await _search_sparse_async_impl(
                collection=collection,
                model_tag=model_tag,
                query_text=query_text,
                top_k=top_k,
                filters=filters,
                readonly=readonly,
                nprobes=nprobes,
                refine_factor=refine_factor,
                user_id=user_id,
                is_admin=is_admin,
            )

    def search_hybrid(
        self,
        collection: str,
        model_tag: str,
        query_text: str,
        query_vector: List[float],
        *,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        fusion_config: Optional[FusionConfig] = None,
        readonly: bool = False,
        nprobes: Optional[int] = None,
        refine_factor: Optional[int] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> HybridSearchResponse:
        from ..retrieval.search_hybrid import _search_hybrid_impl

        with self._storage_context():
            return _search_hybrid_impl(
                collection=collection,
                model_tag=model_tag,
                query_text=query_text,
                query_vector=query_vector,
                top_k=top_k,
                filters=filters,
                fusion_config=fusion_config,
                readonly=readonly,
                nprobes=nprobes,
                refine_factor=refine_factor,
                user_id=user_id,
                is_admin=is_admin,
            )
