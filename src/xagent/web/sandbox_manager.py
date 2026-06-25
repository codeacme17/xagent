"""
Sandbox management in application layer.
"""

import asyncio
import logging
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from ..config import (
    get_boxlite_home_dir,
    get_sandbox_max_concurrency,
    get_sandbox_cpus,
    get_sandbox_env,
    get_sandbox_host_storage_root,
    get_sandbox_image,
    get_sandbox_memory,
    get_sandbox_volumes,
    get_storage_root,
    get_uploads_dir,
)
from ..core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
    build_code_mount_volumes,
)
from ..sandbox import SandboxService
from ..sandbox.base import Sandbox, SandboxConfig, SandboxTemplate

logger = logging.getLogger(__name__)

_WORKER_LIFECYCLE_MARKER = "::worker::"


class SandboxLease:
    """Async context manager for one leased sandbox execution slot."""

    def __init__(
        self,
        provider: "SandboxLeaseProvider",
        *,
        concurrency_safe: bool,
    ) -> None:
        self._provider = provider
        self._concurrency_safe = concurrency_safe
        self._slot: int | None = None
        self._sandbox: Sandbox | None = None

    async def __aenter__(self) -> Sandbox:
        if not self._concurrency_safe:
            self._sandbox = self._provider.primary_sandbox
            return self._sandbox

        self._slot = await self._provider.acquire_worker_slot()
        try:
            self._sandbox = await self._provider.get_worker_sandbox(self._slot)
            return self._sandbox
        except Exception:
            await self._provider.release_worker_slot(self._slot)
            self._slot = None
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if self._slot is not None:
            await self._provider.release_worker_slot(self._slot)
            self._slot = None
        self._sandbox = None


class SandboxLeaseProvider:
    """Lease primary or worker sandboxes for sandboxed tool execution."""

    def __init__(
        self,
        *,
        manager: "SandboxManager",
        lifecycle_type: str,
        lifecycle_id: str,
        primary_sandbox: Sandbox,
        workspace_config: Mapping[str, Any] | None,
        max_concurrency: int,
    ) -> None:
        self._manager = manager
        self._lifecycle_type = lifecycle_type
        self._lifecycle_id = lifecycle_id
        self._workspace_config = workspace_config
        self._available_slots: asyncio.Queue[int] = asyncio.Queue()
        self._worker_locks: dict[int, asyncio.Lock] = {}
        self._workers: dict[int, Sandbox] = {}
        self.primary_sandbox = primary_sandbox
        for slot in range(max(1, max_concurrency)):
            self._available_slots.put_nowait(slot)

    def lease(self, *, concurrency_safe: bool) -> SandboxLease:
        """Return an async context manager for the requested execution mode."""
        return SandboxLease(self, concurrency_safe=concurrency_safe)

    async def acquire_worker_slot(self) -> int:
        """Reserve one worker slot, waiting when all workers are busy."""
        return await self._available_slots.get()

    async def release_worker_slot(self, slot: int) -> None:
        """Return a worker slot to the provider."""
        self._available_slots.put_nowait(slot)

    async def get_worker_sandbox(self, slot: int) -> Sandbox:
        """Get or lazily create a worker sandbox for a slot."""
        if slot in self._workers:
            return self._workers[slot]

        if slot not in self._worker_locks:
            self._worker_locks[slot] = asyncio.Lock()

        async with self._worker_locks[slot]:
            if slot in self._workers:
                return self._workers[slot]
            worker = await self._manager.get_or_create_sandbox(
                self._lifecycle_type,
                f"{self._lifecycle_id}::worker::{slot}",
                workspace_config=self._workspace_config,
            )
            self._workers[slot] = worker
            return worker

    async def cleanup_worker_sandboxes(self) -> None:
        """Delete worker sandboxes while keeping the primary sandbox cached."""
        await self._manager.delete_worker_sandboxes(
            self._lifecycle_type,
            self._lifecycle_id,
        )
        self._workers.clear()


class SandboxPathMapper:
    """Translate backend-visible workspace paths into sandbox volume tuples."""

    def __init__(
        self,
        *,
        backend_storage_root: Path,
        host_storage_root: Path | None,
        sandbox_storage_root: Path | None = None,
    ) -> None:
        self.backend_storage_root = self._as_backend_path(backend_storage_root)
        self.host_storage_root = host_storage_root
        self.sandbox_storage_root = self._as_backend_path(
            sandbox_storage_root or self.backend_storage_root
        )

    @classmethod
    def from_env(cls) -> "SandboxPathMapper":
        return cls(
            backend_storage_root=get_storage_root(),
            host_storage_root=get_sandbox_host_storage_root(),
        )

    @property
    def uses_host_storage_root(self) -> bool:
        return self.host_storage_root is not None

    @staticmethod
    def _as_backend_path(path: str | Path) -> Path:
        backend_path = Path(os.path.expandvars(str(path))).expanduser()
        if not backend_path.is_absolute():
            backend_path = Path.cwd() / backend_path
        return backend_path

    def _relative_to_backend_storage(self, backend_path: Path) -> Path | None:
        try:
            return backend_path.relative_to(self.backend_storage_root)
        except ValueError:
            return None

    def to_host_bind_source(self, backend_path: str | Path) -> Path:
        path = self._as_backend_path(backend_path)
        if self.host_storage_root is None:
            return path

        relative_path = self._relative_to_backend_storage(path)
        if relative_path is None:
            return path
        return self.host_storage_root / relative_path

    def to_sandbox_target(self, backend_path: str | Path) -> Path:
        path = self._as_backend_path(backend_path)
        if self.host_storage_root is None:
            return path

        relative_path = self._relative_to_backend_storage(path)
        if relative_path is None:
            return path
        return self.sandbox_storage_root / relative_path

    def volume_for_backend_path(
        self, backend_path: str | Path, mode: str = "rw"
    ) -> tuple[str, str, str]:
        return (
            str(self.to_host_bind_source(backend_path)),
            str(self.to_sandbox_target(backend_path)),
            mode,
        )


class SandboxManager:
    """
    Manages sandbox instances.
    """

    def __init__(self, service: SandboxService):
        """
        Initialize sandbox manager.

        Args:
            service: SandboxService instance for creating sandboxes
        """
        self._service: SandboxService = service
        self._cache: dict[str, Sandbox] = {}
        self._config_cache: dict[str, SandboxConfig] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    @staticmethod
    def make_sandbox_name(lifecycle_type: str, lifecycle_id: str) -> str:
        """Build a sandbox name from lifecycle type and id."""
        return f"{lifecycle_type}::{lifecycle_id}"

    @staticmethod
    def parse_sandbox_name(name: str) -> tuple[str, str]:
        """Parse a sandbox name into (lifecycle_type, lifecycle_id).

        Raises:
            ValueError: Invalid sandbox name format.
        """
        parts = name.split("::", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid sandbox name format: {name!r}")
        return parts[0], parts[1]

    @staticmethod
    def _base_lifecycle_id(lifecycle_id: str) -> str:
        """Return the owner lifecycle id for primary and worker sandboxes."""
        return lifecycle_id.split(_WORKER_LIFECYCLE_MARKER, 1)[0]

    @classmethod
    def _worker_sandbox_prefix(cls, lifecycle_type: str, lifecycle_id: str) -> str:
        return (
            cls.make_sandbox_name(lifecycle_type, lifecycle_id)
            + _WORKER_LIFECYCLE_MARKER
        )

    def _get_sandbox_image_and_config(self) -> tuple[str, SandboxConfig]:
        """Get sandbox image and configuration from centralized config module."""
        image = get_sandbox_image()
        config = SandboxConfig()
        path_mapper = SandboxPathMapper.from_env()

        # CPU
        cpus = get_sandbox_cpus()
        if cpus is not None:
            config.cpus = cpus

        # MEM
        memory = get_sandbox_memory()
        if memory is not None:
            config.memory = memory

        # ENV
        env = get_sandbox_env()
        if env:
            config.env = env

        # VOL
        volumes = get_sandbox_volumes(
            host_side_sources=path_mapper.uses_host_storage_root
        )
        if volumes:
            config.volumes = volumes

        return image, config

    @staticmethod
    def _append_unique_volume(
        volumes: list[tuple[str, str, str]], volume: tuple[str, str, str]
    ) -> None:
        if volume not in volumes:
            volumes.append(volume)

    @staticmethod
    def _workspace_mount_paths(
        lifecycle_type: str,
        lifecycle_id: str,
        workspace_config: Mapping[str, Any] | None,
    ) -> list[tuple[Path, bool]]:
        paths: list[tuple[Path, bool]] = []

        if workspace_config:
            base_dir = workspace_config.get("base_dir")
            if base_dir:
                paths.append((Path(str(base_dir)), True))

            for raw_dir in workspace_config.get("allowed_external_dirs") or []:
                paths.append((Path(str(raw_dir)), False))
        elif lifecycle_type == "user":
            owner_lifecycle_id = SandboxManager._base_lifecycle_id(lifecycle_id)
            paths.append((get_uploads_dir() / f"user_{owner_lifecycle_id}", True))

        return paths

    @staticmethod
    def _config_equivalent(left: SandboxConfig, right: SandboxConfig) -> bool:
        return (
            left.cpus == right.cpus
            and left.memory == right.memory
            and (left.env or {}) == (right.env or {})
            and set(left.volumes or []) == set(right.volumes or [])
        )

    @staticmethod
    def _ensure_config_equivalent(
        sandbox_name: str,
        cached_config: SandboxConfig | None,
        desired_config: SandboxConfig,
    ) -> None:
        if cached_config is None:
            return
        if SandboxManager._config_equivalent(cached_config, desired_config):
            return
        raise RuntimeError(
            f"Sandbox {sandbox_name!r} already exists with different runtime "
            "configuration. Use a distinct lifecycle id for different workspace "
            "mounts."
        )

    def _build_sandbox_config(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        *,
        ensure_dir: bool,
        workspace_config: Mapping[str, Any] | None = None,
    ) -> tuple[str, SandboxConfig]:
        image, config = self._get_sandbox_image_and_config()
        config_volumes = list(config.volumes) if config.volumes else []
        default_volumes = self._make_default_volumes(
            lifecycle_type,
            lifecycle_id,
            ensure_dir=ensure_dir,
            workspace_config=workspace_config,
        )
        config.volumes = config_volumes + default_volumes
        return image, config

    def _make_default_volumes(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        *,
        ensure_dir: bool,
        workspace_config: Mapping[str, Any] | None = None,
    ) -> list[tuple[str, str, str]]:
        """
        Build default volume mounts.

        Code directories are always mounted read-only.
        User workspace is additionally mounted read-write for user lifecycle type.

        Args:
            lifecycle_type: e.g. task|user
            lifecycle_id: e.g. task_id|user_id
            ensure_dir: When True, create the host directory
            workspace_config: Actual tool workspace configuration, when known
        """
        # Code mounts are always present (at least src/)
        volumes: list[tuple[str, str, str]] = list(build_code_mount_volumes())
        path_mapper = SandboxPathMapper.from_env()

        # Mount actual workspace roots as read-write.
        for backend_path, should_create in self._workspace_mount_paths(
            lifecycle_type,
            lifecycle_id,
            workspace_config,
        ):
            if ensure_dir:
                try:
                    if should_create or backend_path.exists():
                        os.makedirs(backend_path, exist_ok=True)
                except OSError as exc:
                    logger.warning(
                        "Failed to prepare sandbox workspace mount %s: %s",
                        backend_path,
                        exc,
                    )

            self._append_unique_volume(
                volumes, path_mapper.volume_for_backend_path(backend_path, "rw")
            )

        return volumes

    async def get_or_create_sandbox(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        *,
        workspace_config: Mapping[str, Any] | None = None,
    ) -> Sandbox:
        """
        Get or create a sandbox.

        Args:
            lifecycle_type: e.g. task|user
            lifecycle_id: e.g. task_id|user_id
            workspace_config: Actual tool workspace configuration to mount

        Returns:
            Sandbox instance
        """
        sandbox_name = self.make_sandbox_name(lifecycle_type, lifecycle_id)
        image, desired_config = self._build_sandbox_config(
            lifecycle_type,
            lifecycle_id,
            ensure_dir=False,
            workspace_config=workspace_config,
        )

        cached_config = self._config_cache.get(sandbox_name)
        if sandbox_name in self._cache:
            self._ensure_config_equivalent(sandbox_name, cached_config, desired_config)
            return self._cache[sandbox_name]

        # Acquire per-name lock to prevent concurrent creation
        async with self._locks_guard:
            if sandbox_name not in self._locks:
                self._locks[sandbox_name] = asyncio.Lock()
            lock = self._locks[sandbox_name]

        async with lock:
            # Double-check after acquiring lock
            cached_config = self._config_cache.get(sandbox_name)
            if sandbox_name in self._cache:
                self._ensure_config_equivalent(
                    sandbox_name, cached_config, desired_config
                )
                return self._cache[sandbox_name]

            # Get base image and config from environment variables
            image, config = self._build_sandbox_config(
                lifecycle_type,
                lifecycle_id,
                ensure_dir=True,
                workspace_config=workspace_config,
            )
            logger.info(
                "Getting/creating sandbox: image=%r, cpus=%r, memory=%r, volumes=%r, env_count=%r",
                image,
                config.cpus,
                config.memory,
                config.volumes,
                len(config.env or {}),
            )

            template = SandboxTemplate(type="image", image=image)

            logger.debug(f"Getting or creating sandbox for: {sandbox_name}")
            sandbox = await self._service.get_or_create(
                sandbox_name,
                template=template,
                config=config,
            )

            self._cache[sandbox_name] = sandbox
            self._config_cache[sandbox_name] = config
            return sandbox

    async def create_lease_provider(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        *,
        workspace_config: Mapping[str, Any] | None = None,
    ) -> SandboxLeaseProvider:
        """Create a lease provider for primary and worker sandboxes."""
        primary = await self.get_or_create_sandbox(
            lifecycle_type,
            lifecycle_id,
            workspace_config=workspace_config,
        )
        return SandboxLeaseProvider(
            manager=self,
            lifecycle_type=lifecycle_type,
            lifecycle_id=lifecycle_id,
            primary_sandbox=primary,
            workspace_config=workspace_config,
            max_concurrency=get_sandbox_max_concurrency(),
        )

    async def delete_sandbox(self, lifecycle_type: str, lifecycle_id: str) -> None:
        """
        Delete sandbox.

        Args:
            lifecycle_type: e.g. task|user
            lifecycle_id: e.g. task_id|user_id
        """
        sandbox_names = await self._find_lifecycle_sandbox_names(
            lifecycle_type,
            lifecycle_id,
            include_primary=True,
            include_workers=True,
        )
        await self._delete_sandbox_names(sandbox_names)

    async def delete_worker_sandboxes(
        self, lifecycle_type: str, lifecycle_id: str
    ) -> None:
        """Delete worker sandboxes for a lifecycle while preserving the primary."""
        sandbox_names = await self._find_lifecycle_sandbox_names(
            lifecycle_type,
            lifecycle_id,
            include_primary=False,
            include_workers=True,
        )
        await self._delete_sandbox_names(sandbox_names)

    async def _find_lifecycle_sandbox_names(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        *,
        include_primary: bool,
        include_workers: bool,
    ) -> set[str]:
        sandbox_name = self.make_sandbox_name(lifecycle_type, lifecycle_id)
        worker_prefix = self._worker_sandbox_prefix(lifecycle_type, lifecycle_id)
        sandbox_names = {
            name
            for name in self._cache
            if (include_primary and name == sandbox_name)
            or (include_workers and name.startswith(worker_prefix))
        }
        if include_primary:
            sandbox_names.add(sandbox_name)

        try:
            listed_sandboxes = await self._service.list_sandboxes()
        except Exception as exc:
            logger.warning("Failed to list sandboxes for cleanup: %s", exc)
            return sandbox_names

        for sb in listed_sandboxes or []:
            name = sb.name
            if include_primary and name == sandbox_name:
                sandbox_names.add(name)
            elif include_workers and name.startswith(worker_prefix):
                sandbox_names.add(name)

        return sandbox_names

    async def _delete_sandbox_names(self, sandbox_names: set[str]) -> None:
        for name in sorted(sandbox_names):
            try:
                await self._service.delete(name)
                logger.debug(f"Sandbox deleted: {name}")
            except Exception as e:
                logger.error(f"Failed to delete sandbox {name}: {e}")
            finally:
                # Always evict from cache — even on failure the instance
                # may be in an unknown state and should be recreated.
                self._cache.pop(name, None)
                self._config_cache.pop(name, None)
                self._locks.pop(name, None)

    async def warmup(self) -> None:
        """
        Warmup default image.
        Uses empty config for warmup to avoid unnecessary volume mounts.
        """
        image = get_sandbox_image()
        warmup_name = "__warmup__"
        try:
            template = SandboxTemplate(type="image", image=image)
            # Use empty config for warmup - no need for volumes/env
            warmup_config = SandboxConfig()
            async with await self._service.get_or_create(
                warmup_name, template=template, config=warmup_config
            ):
                pass
            await self._service.delete(warmup_name)
            logger.info(f"Sandbox image warmup completed: {image}")
        except Exception as e:
            logger.error(f"Failed to warmup sandbox image: {e}")

    async def cleanup(self) -> None:
        """Stop all running sandboxes.

        Delete sandboxes whose config (image, cpus, memory, volumes)
        differs from the current environment so they get recreated
        with the correct settings next time.

        Note:
            If ``get_uploads_dir()`` (via ``XAGENT_UPLOADS_DIR`` env var) changes
            between deployments, all user sandboxes will be detected as
            having stale volume mounts and will be deleted for recreation.
        """
        try:
            sandboxes = await self._service.list_sandboxes()
            if not sandboxes:
                logger.info("No sandboxes to clean up")
                return

            image, config = self._get_sandbox_image_and_config()

            for sb in sandboxes:
                try:
                    lifecycle_type, lifecycle_id = None, None
                    try:
                        lifecycle_type, lifecycle_id = self.parse_sandbox_name(sb.name)
                    except ValueError:
                        # Not a normal managed sandbox name, stop
                        if sb.state == "running":
                            box = await self._service.get_or_create(
                                sb.name, template=sb.template, config=sb.config
                            )
                            await box.stop()
                            logger.debug(f"Stopped sandbox: {sb.name}")
                        continue

                    # Delete sandbox if config changed (force recreate on next start)
                    image_changed = sb.template.image != image
                    cpus_changed = sb.config.cpus != config.cpus
                    memory_changed = sb.config.memory != config.memory

                    # volumes comparison: None and empty list are treated as equal, ignore order
                    old_volumes = sb.config.volumes or []

                    default_volumes = self._make_default_volumes(
                        lifecycle_type, lifecycle_id, ensure_dir=False
                    )
                    config_volumes = list(config.volumes) if config.volumes else []
                    # Merge volumes
                    new_volumes = config_volumes + default_volumes

                    volumes_changed = set(old_volumes) != set(new_volumes)

                    # env comparison: None and empty dict are treated as equal
                    old_env = sb.config.env or {}
                    new_env = config.env or {}
                    env_changed = old_env != new_env

                    if (
                        image_changed
                        or cpus_changed
                        or memory_changed
                        or volumes_changed
                        or env_changed
                    ):
                        changes = []
                        if image_changed:
                            changes.append(f"image: {sb.template.image} -> {image}")
                        if cpus_changed:
                            changes.append(f"cpus: {sb.config.cpus} -> {config.cpus}")
                        if memory_changed:
                            changes.append(
                                f"memory: {sb.config.memory} -> {config.memory}"
                            )
                        if env_changed:
                            old_env_str = (
                                ";".join([f"{k}={v}" for k, v in old_env.items()])
                                if old_env
                                else "none"
                            )
                            new_env_str = (
                                ";".join([f"{k}={v}" for k, v in new_env.items()])
                                if new_env
                                else "none"
                            )
                            changes.append(f"env: {old_env_str} -> {new_env_str}")
                        if volumes_changed:
                            old_vol_str = (
                                ";".join([f"{h}:{g}:{m}" for h, g, m in old_volumes])
                                if old_volumes
                                else "none"
                            )
                            new_vol_str = (
                                ";".join([f"{h}:{g}:{m}" for h, g, m in new_volumes])
                                if new_volumes
                                else "none"
                            )
                            changes.append(f"volumes: {old_vol_str} -> {new_vol_str}")
                        logger.info(
                            f"Config changed for sandbox [{sb.name}]: "
                            f"{', '.join(changes)}, deleting"
                        )
                        await self._service.delete(sb.name)
                        continue

                    # Stop running sandboxes with matching image
                    if sb.state == "running":
                        box = await self._service.get_or_create(
                            sb.name, template=sb.template, config=sb.config
                        )
                        await box.stop()
                        logger.debug(f"Stopped sandbox: {sb.name}")
                except Exception as e:
                    logger.error(f"Failed to handle sandbox {sb.name}: {e}")

            self._cache.clear()
            self._config_cache.clear()
            self._locks.clear()
            logger.info("Sandbox cleanup completed")
        except Exception as e:
            logger.error(f"Failed to cleanup sandboxes: {e}")


# Global sandbox manager instance
_sandbox_manager: Optional[SandboxManager] = None
_sandbox_manager_lock = threading.Lock()
_sandbox_manager_initialized = False


def _create_sandbox_service() -> Optional[SandboxService]:
    """
    Create sandbox service based on environment configuration.

    Environment variables:
    - SANDBOX_ENABLED: Enable/disable sandbox (default: true)
    - SANDBOX_IMPLEMENTATION: Implementation type (default: docker)
      - docker: Use Docker sandbox
      - boxlite: Use Boxlite sandbox
    - BOXLITE_HOME_DIR: Boxlite home directory (optional)

    Returns:
        SandboxService instance or None if disabled
    """
    # Check if sandbox is enabled
    sandbox_enabled = os.getenv("SANDBOX_ENABLED", "false").lower() == "true"
    if not sandbox_enabled:
        logger.info("Sandbox is disabled via SANDBOX_ENABLED environment variable")
        return None

    # Get implementation type
    implementation = os.getenv("SANDBOX_IMPLEMENTATION", "docker")

    if implementation == "boxlite":
        return _create_boxlite_service()
    elif implementation == "docker":
        return _create_docker_service()
    else:
        logger.warning(
            f"Unknown sandbox implementation: {implementation}, falling back to docker"
        )
        return _create_docker_service()


def _create_boxlite_service() -> Optional[SandboxService]:
    """Create Boxlite sandbox service."""
    try:
        from ..sandbox import BoxliteSandboxService
    except ImportError:
        logger.error("boxlite is not installed.")
        return None

    from .sandbox_store import DBBoxliteStore

    store = DBBoxliteStore()
    # Get home directory
    home_dir = get_boxlite_home_dir()

    service = None
    try:
        service = BoxliteSandboxService(
            store=store, home_dir=None if home_dir is None else str(home_dir)
        )
        logger.info(
            f"Created Boxlite sandbox service (home_dir={home_dir or 'default'})"
        )
    except Exception as e:
        logger.error(f"Failed to create Boxlite sandbox service: {e}")

    return service


def _create_docker_service() -> Optional[SandboxService]:
    """Create Docker sandbox service."""
    try:
        from ..sandbox import DockerSandboxService
    except ImportError:
        logger.error("docker sandbox dependencies are not installed.")
        return None

    from .sandbox_store import DBDockerStore

    store = DBDockerStore()

    service = None
    try:
        service = DockerSandboxService(store=store)
        logger.info("Created Docker sandbox service")
    except Exception as e:
        logger.error(f"Failed to create Docker sandbox service: {e}")

    return service


def get_sandbox_manager() -> Optional[SandboxManager]:
    """
    Get or create global sandbox manager instance.

    Thread-safe singleton pattern with double-checked locking.

    Returns:
        SandboxManager instance or None if sandbox is disabled
    """
    global _sandbox_manager, _sandbox_manager_initialized

    # Fast path: already initialized (either successfully or service was None)
    if _sandbox_manager_initialized:
        return _sandbox_manager

    # Slow path: need to initialize
    with _sandbox_manager_lock:
        # Double-check after acquiring lock
        if _sandbox_manager_initialized:
            return _sandbox_manager

        # Get sandbox service
        service = _create_sandbox_service()
        if service is None:
            _sandbox_manager_initialized = True
            return None

        # Create sandbox manager
        _sandbox_manager = SandboxManager(service)
        _sandbox_manager_initialized = True
        logger.info("Created global sandbox manager")

        return _sandbox_manager
