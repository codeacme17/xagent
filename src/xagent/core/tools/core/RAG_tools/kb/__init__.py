"""KB semantic coordinator public surface."""

from .collection_handle import KBHandleProvider, LanceDBCollectionHandle
from .coordinator import (
    KBCoordinator,
    get_kb_coordinator,
    reset_kb_coordinator_for_tests,
)
from .file_compatibility import KBFileCompatibilityFacade
from .legacy_step_compatibility import KBLegacyStepCompatibilityFacade
from .maintenance_compatibility import (
    CollectionConfigSnapshot,
    CollectionRollbackMaintenanceResult,
    KBMaintenanceCompatibilityFacade,
)
from .management_facade import KBCoreManagementCompatibilityFacade
from .models import (
    KBAccessMode,
    KBBackendCapabilities,
    KBCollectionContext,
    KBContextRequest,
    KBStorageBackend,
    KBUserScope,
)
from .operation_compatibility import (
    CompensationStep,
    KBOperationCompatibilityFacade,
    KBOperationOutcome,
    PersistencePolicy,
    RollbackStatus,
    SideEffectPlane,
)
from .parse_display_compatibility import KBParseDisplayCompatibilityFacade
from .pipeline_compatibility import KBPipelineCompatibilityFacade
from .retrieval_compatibility import KBRetrievalHelperCompatibilityFacade
from .storage_shim import KBStorageShimCompatibilityFacade
from .vector_storage_compatibility import (
    KBVectorStorageCleanupResult,
    KBVectorStorageCompatibilityFacade,
)
from .version_compatibility import (
    KBMainPointerSnapshot,
    KBVersionCandidateCleanupSnapshot,
    KBVersionCandidateRollbackResult,
    KBVersionCompatibilityFacade,
)

__all__ = [
    "KBAccessMode",
    "KBBackendCapabilities",
    "KBCollectionContext",
    "KBContextRequest",
    "KBHandleProvider",
    "CollectionConfigSnapshot",
    "CollectionRollbackMaintenanceResult",
    "KBCoreManagementCompatibilityFacade",
    "KBCoordinator",
    "KBFileCompatibilityFacade",
    "KBLegacyStepCompatibilityFacade",
    "CompensationStep",
    "KBMainPointerSnapshot",
    "KBMaintenanceCompatibilityFacade",
    "KBOperationCompatibilityFacade",
    "KBOperationOutcome",
    "KBVersionCandidateCleanupSnapshot",
    "KBVersionCandidateRollbackResult",
    "KBParseDisplayCompatibilityFacade",
    "KBPipelineCompatibilityFacade",
    "KBRetrievalHelperCompatibilityFacade",
    "KBStorageShimCompatibilityFacade",
    "KBStorageBackend",
    "KBVectorStorageCleanupResult",
    "KBVectorStorageCompatibilityFacade",
    "KBUserScope",
    "KBVersionCompatibilityFacade",
    "LanceDBCollectionHandle",
    "PersistencePolicy",
    "RollbackStatus",
    "SideEffectPlane",
    "get_kb_coordinator",
    "reset_kb_coordinator_for_tests",
]
