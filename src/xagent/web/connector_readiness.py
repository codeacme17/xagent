"""Post-deploy startability check for every catalog connector.

Usage: ``python -m xagent.web.connector_readiness [--exclude APP_ID]...
[--total-timeout SECONDS]``, run in the backend container with the running
service's environment, the same way as ``oauth_readiness``::

    docker compose exec -T backend /opt/xagent/deploy/entrypoint.sh \
        python -m xagent.web.connector_readiness

It reads every ``public_mcp_apps`` row of the deployment's database, hidden
rows included, and checks by ``launch_config`` shape whether the connector
can start here, without any credential:

* A stdio row (``command``) is spawned with a placeholder value for every
  ``env_mapping`` and ``required_env`` key, then asked for MCP ``initialize``
  and ``tools/list``; at least one tool must come back. The in-repo servers'
  tool lists are static, and the placeholder never leaves the process: the
  check stops after ``tools/list``.
* An ``npx``/``uvx`` row is reported ``SKIP sandboxed`` when the deployment
  runs those in a sandbox (``SANDBOX_ENABLED``), because the backend image is
  not where they run. With the sandbox off it is spawned like any stdio row,
  with npm and uv forced offline so a cache miss fails before the handshake
  instead of reaching a registry.
* A remote row (``url``) gets one unauthenticated ``GET`` through the
  runtime's MCP HTTP client, so the probe sees the same proxy and egress
  policy a task does. Any HTTP response passes; a DNS, TLS, egress-policy or
  connect failure fails. No OAuth flow is started.

One line per connector, ``PASS``, ``SKIP sandboxed`` or ``FAIL <class>``
with the class one of ``spawn`` (command not found), ``missing`` (the child
exited before the handshake: a Python import error, or an npm package absent
from the offline cache), ``handshake`` (no usable ``initialize``/``tools/list`` reply within the
timeout, or an error reply), ``no-tools`` and ``unreachable``. A failure line carries the last line of the
child's stderr or the exception, never the environment. The exit code is 1
when any connector fails.

Each connector is bounded by ``XAGENT_MCP_TOOL_INIT_TIMEOUT_SECONDS`` (the
same bound task startup uses) and the whole run by ``--total-timeout``, which
defaults to the connector count times that value.

The child's environment is the placeholders, the offline switches and
``PATH``, plus the SDK's inherited allowlist (``HOME``, ``USER``, ...). A task
also forwards ``static_env`` entries, the proxy and CA-bundle variables and the
file allowlists (``web/tools/config.py``); building a connector's environment
outside a task is #3261's work. A connector that needs one of those before it
answers ``tools/list`` would fail here and not in a task; no registry row does.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TextIO, cast

import anyio
import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED

from ..core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper import (
    should_sandbox_mcp_connection,
)
from ..core.tools.core.mcp.http import create_mcp_http_client

if TYPE_CHECKING:
    from ..core.tools.core.mcp.sessions import Connection

Status = Literal["PASS", "SKIP", "FAIL"]
Reason = Literal[
    "sandboxed", "spawn", "missing", "handshake", "no-tools", "unreachable"
]
Origin = Literal["builtin", "custom"]

# Obviously fake, and the only value the spawned connector ever sees for a
# credential key. The stdio check stops after tools/list, so it never reaches
# a vendor.
PLACEHOLDER = "readiness-placeholder"
# Forces a cache miss to fail instead of reaching the registry; NPM_CONFIG_*
# is how Dockerfile.sandbox verifies its own warm-up.
OFFLINE_ENV = {"NPM_CONFIG_OFFLINE": "true", "UV_OFFLINE": "1"}
_STDERR_TAIL_LINES = 5
_DETAIL_MAX_CHARS = 300
# npm ends every failure with this pointer; the error itself is the line above.
_STDERR_NOISE = ("A complete log of this run can be found in",)


@dataclass(frozen=True)
class Outcome:
    """What one check found, before it is tied to a row."""

    status: Status
    reason: Reason | None
    detail: str
    stderr_tail: str = ""


SKIPPED_SANDBOXED = Outcome(
    "SKIP", "sandboxed", "npx/uvx rows run in the sandbox image, not this one"
)
NO_LAUNCH = Outcome("FAIL", "spawn", "launch_config has neither a command nor a url")


@dataclass(frozen=True)
class ConnectorReadiness:
    app_id: str
    origin: Origin
    hidden: bool
    status: Status
    reason: Reason | None
    detail: str
    stderr_tail: str

    @classmethod
    def of(cls, row: Any, *, origin: Origin, outcome: Outcome) -> ConnectorReadiness:
        return cls(
            app_id=str(row.app_id),
            origin=origin,
            hidden=_hidden(row),
            status=outcome.status,
            reason=outcome.reason,
            detail=outcome.detail,
            stderr_tail=outcome.stderr_tail,
        )

    @property
    def failed(self) -> bool:
        return self.status == "FAIL"


def _hidden(row: Any) -> bool:
    return not bool(getattr(row, "is_visible_in_connector", True))


def sandbox_enabled_from_env() -> bool:
    """Whether a task attaches a sandbox, read the way
    ``sandbox_manager._create_sandbox_service`` reads it."""
    return os.getenv("SANDBOX_ENABLED", "false").lower() == "true"


def is_sandbox_command(command: str) -> bool:
    """Whether a task routes this stdio command into the sandbox (npx, uvx)."""
    connection = cast("Connection", {"transport": "stdio", "command": command})
    return should_sandbox_mcp_connection(connection)


def placeholder_env(launch_config: Mapping[str, Any]) -> dict[str, str]:
    """A placeholder for every key the connector expects a credential in."""
    keys: list[str] = []
    env_mapping = launch_config.get("env_mapping")
    if isinstance(env_mapping, Mapping):
        keys.extend(str(key) for key in env_mapping)
    required_env = launch_config.get("required_env")
    if isinstance(required_env, (list, tuple)):
        keys.extend(str(key) for key in required_env)
    return {key: PLACEHOLDER for key in keys}


def _first_line(text: str) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line[:_DETAIL_MAX_CHARS]


def _describe(exc: BaseException) -> str:
    message = _first_line(str(exc))
    name = type(exc).__name__
    return f"{name}: {message}" if message else name


def _leaves(exc: BaseException) -> list[BaseException]:
    """Flatten anyio task-group exception groups to their leaf exceptions."""
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for inner in exc.exceptions for leaf in _leaves(inner)]
    return [exc]


def _classify_stdio_failure(exc: BaseException) -> tuple[Reason, BaseException]:
    leaves = _leaves(exc)
    for leaf in leaves:
        if isinstance(leaf, (FileNotFoundError, PermissionError, NotADirectoryError)):
            return "spawn", leaf
    for leaf in leaves:
        if isinstance(leaf, TimeoutError):
            return "handshake", leaf
    # The SDK answers every pending request with CONNECTION_CLOSED once the
    # child's stdout closes, so an exit before the reply surfaces here.
    for leaf in leaves:
        if _is_connection_closed(leaf):
            return "missing", leaf
    return "handshake", leaves[0]


def _is_connection_closed(exc: BaseException) -> bool:
    if isinstance(exc, McpError):
        return exc.error.code == CONNECTION_CLOSED
    return isinstance(
        exc,
        (
            anyio.ClosedResourceError,
            anyio.BrokenResourceError,
            anyio.EndOfStream,
            BrokenPipeError,
        ),
    )


def _stderr_tail(errlog: TextIO) -> str:
    errlog.seek(0)
    lines = [line.rstrip() for line in errlog if line.strip()]
    return "\n".join(lines[-_STDERR_TAIL_LINES:])


def _last_stderr_line(tail: str) -> str:
    """The child's last stderr line that says something, as the failure detail."""
    for line in reversed(tail.splitlines()):
        if not any(marker in line for marker in _STDERR_NOISE):
            return _first_line(line)
    return _first_line(tail.splitlines()[-1]) if tail else ""


async def _list_tool_count(
    command: str, args: list[str], env: dict[str, str], errlog: TextIO
) -> int:
    params = StdioServerParameters(command=command, args=args, env=env)
    async with (
        stdio_client(params, errlog=errlog) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        result = await session.list_tools()
        return len(result.tools)


async def check_stdio(
    launch_config: Mapping[str, Any], *, timeout: float, offline: bool
) -> Outcome:
    """Spawn the row's command and require a tools/list reply with a tool."""
    command = str(launch_config.get("command", "")).strip()
    raw_args = launch_config.get("args")
    args = [str(arg) for arg in raw_args] if isinstance(raw_args, list) else []
    env = placeholder_env(launch_config)
    if offline:
        env.update(OFFLINE_ENV)
    # Same PATH injection as sessions._create_stdio_session; the SDK adds the
    # rest of its inherited allowlist (HOME, USER, ...) underneath.
    env.setdefault("PATH", os.environ.get("PATH", ""))

    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as errlog:
        try:
            with anyio.fail_after(timeout if timeout > 0 else None):
                count = await _list_tool_count(command, args, env, errlog)
        except BaseException as exc:  # noqa: BLE001 - classified below
            if isinstance(
                exc, (KeyboardInterrupt, SystemExit, anyio.get_cancelled_exc_class())
            ):
                raise
            reason, leaf = _classify_stdio_failure(exc)
            tail = _stderr_tail(errlog)
            if reason == "handshake" and isinstance(leaf, TimeoutError):
                detail = f"no initialize/tools/list reply within {timeout:g}s"
                if tail:
                    detail += f"; stderr: {_last_stderr_line(tail)}"
            elif tail:
                detail = _last_stderr_line(tail)
            else:
                detail = _describe(leaf)
            return Outcome("FAIL", reason, detail, tail)
        tail = _stderr_tail(errlog)
    if count == 0:
        return Outcome("FAIL", "no-tools", "tools/list returned no tools", tail)
    return Outcome("PASS", None, f"{count} tool{'s' if count != 1 else ''}", tail)


async def check_remote(url: str, *, timeout: float) -> Outcome:
    """One unauthenticated GET through the runtime's MCP HTTP client."""
    http_timeout = httpx.Timeout(timeout) if timeout > 0 else None
    try:
        async with create_mcp_http_client(
            timeout=http_timeout, endpoint_url=url
        ) as client:
            response = await client.get(url)
    except Exception as exc:  # noqa: BLE001 - any failure is "unreachable"
        return Outcome("FAIL", "unreachable", _describe(exc))
    return Outcome("PASS", None, f"HTTP {response.status_code}")


async def check_launch(
    launch_config: Any, *, sandbox_enabled: bool, timeout: float
) -> Outcome:
    """Pick the check by the row's launch_config shape and run it."""
    if not isinstance(launch_config, Mapping):
        return NO_LAUNCH
    command = launch_config.get("command")
    url = launch_config.get("url")
    if isinstance(command, str) and command.strip():
        sandbox_command = is_sandbox_command(command.strip())
        if sandbox_command and sandbox_enabled:
            return SKIPPED_SANDBOXED
        return await check_stdio(
            launch_config, timeout=timeout, offline=sandbox_command
        )
    if isinstance(url, str) and url.strip():
        return await check_remote(url.strip(), timeout=timeout)
    return NO_LAUNCH


async def check_rows(
    rows: Iterable[Any],
    *,
    builtin_app_ids: set[str],
    sandbox_enabled: bool,
    timeout: float,
    total_timeout: float | None = None,
    exclude: set[str] | None = None,
) -> list[ConnectorReadiness]:
    """Check every row sequentially, in app_id order, within the total bound.

    Raises ``ValueError`` when ``exclude`` names an app_id that is not a row,
    so a misspelled exclusion fails instead of excluding nothing.
    """
    rows_by_id = {str(row.app_id): row for row in rows}
    exclude = exclude or set()
    unknown = sorted(exclude - set(rows_by_id))
    if unknown:
        raise ValueError(f"unknown connector to exclude: {', '.join(unknown)}")

    deadline = (
        time.monotonic() + total_timeout
        if total_timeout is not None and total_timeout > 0
        else None
    )
    results: list[ConnectorReadiness] = []
    for app_id in sorted(rows_by_id):
        if app_id in exclude:
            continue
        row = rows_by_id[app_id]
        origin: Origin = "builtin" if app_id in builtin_app_ids else "custom"
        row_timeout = timeout
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                outcome = Outcome(
                    "FAIL",
                    "handshake",
                    f"not checked: total bound of {total_timeout:g}s exhausted",
                )
                results.append(
                    ConnectorReadiness.of(row, origin=origin, outcome=outcome)
                )
                continue
            row_timeout = min(timeout, remaining) if timeout > 0 else remaining
        outcome = await check_launch(
            row.launch_config, sandbox_enabled=sandbox_enabled, timeout=row_timeout
        )
        results.append(ConnectorReadiness.of(row, origin=origin, outcome=outcome))
    return results


def format_result(result: ConnectorReadiness) -> str:
    head = (
        result.status if result.reason is None else f"{result.status} {result.reason}"
    )
    label = f"connector:{result.app_id} {result.origin}"
    if result.hidden:
        label += " hidden"
    return f"{head} {label}: {result.detail}"


def _load_app_rows() -> list[Any]:
    from sqlalchemy.orm import Session

    from .models.database import create_database_engine
    from .models.public_mcp import PublicMCPApp

    engine = create_database_engine(read_only=True)
    try:
        with Session(engine) as db:
            rows = db.query(PublicMCPApp).all()
            db.expunge_all()
            return rows
    finally:
        engine.dispose()


def _parse_exclude(values: Iterable[str] | None) -> set[str]:
    return {
        item.strip()
        for value in values or ()
        for item in value.split(",")
        if item.strip()
    }


def main(argv: Sequence[str] | None = None, *, out: TextIO | None = None) -> int:
    out = sys.stdout if out is None else out
    parser = argparse.ArgumentParser(
        prog="python -m xagent.web.connector_readiness",
        description="Check that every catalog connector starts in this "
        "container, with placeholder credentials. Prints no values.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        metavar="APP_ID",
        help="catalog app ID (e.g. slack) this deployment leaves unstartable on "
        "purpose; repeatable or comma-separated",
    )
    parser.add_argument(
        "--total-timeout",
        type=float,
        metavar="SECONDS",
        help="bound for the whole run; connectors not reached in time fail as "
        "handshake. Default: connector count times the per-connector timeout "
        "(XAGENT_MCP_TOOL_INIT_TIMEOUT_SECONDS)",
    )
    args = parser.parse_args(argv)

    from ..config import get_mcp_tool_init_timeout_seconds
    from .builtin_mcp_registry import get_builtin_public_mcp_app_rows

    try:
        rows = _load_app_rows()
    except Exception as exc:
        # The class name only: a driver error can carry the database host.
        print(f"FAIL database: {type(exc).__name__}", file=out)
        return 1

    exclude = _parse_exclude(args.exclude)
    timeout = float(get_mcp_tool_init_timeout_seconds())
    total_timeout = args.total_timeout
    if total_timeout is None and timeout > 0:
        checked = [row for row in rows if str(row.app_id) not in exclude]
        total_timeout = timeout * len(checked)
    builtin_app_ids = {str(row["app_id"]) for row in get_builtin_public_mcp_app_rows()}
    try:
        results = anyio.run(
            lambda: check_rows(
                rows,
                builtin_app_ids=builtin_app_ids,
                sandbox_enabled=sandbox_enabled_from_env(),
                timeout=timeout,
                total_timeout=total_timeout,
                exclude=exclude,
            )
        )
    except ValueError as exc:
        parser.error(str(exc))

    failed = sum(result.failed for result in results)
    skipped = sum(result.status == "SKIP" for result in results)
    for result in results:
        print(format_result(result), file=out)
    print(f"{len(results)} checks, {failed} failed, {skipped} skipped", file=out)
    return 1 if failed else 0


if __name__ == "__main__":
    from dotenv import load_dotenv

    # Same as the server's own entry point (xagent.web.__main__).
    load_dotenv()
    sys.exit(main())
