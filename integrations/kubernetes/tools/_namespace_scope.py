"""Per-turn namespace scope of the namespace-scoped Kubernetes tools.

With a namespace allowlist configured, a turn's copy of such a tool offers
``namespace`` to the model, states the namespaces it may use, and refuses any
other namespace before a Kubernetes client is built. Without one, the tool is
returned unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from core.tool import RegisteredTool
from core.tool_framework.utils import tool_access_denied

_NAMESPACE = "namespace"


def _reachable_namespaces(kubernetes: Mapping[str, Any]) -> tuple[str, ...]:
    """The configured default, then the allowlist, without repeats; ``()`` with no allowlist."""
    allowlist = kubernetes.get("namespaces") or ()
    if not allowlist:
        return ()
    return tuple(dict.fromkeys((kubernetes.get(_NAMESPACE, "default"), *allowlist)))


def namespace_scoped_view(tool: RegisteredTool, sources: dict[str, Any]) -> RegisteredTool:
    """Return ``tool`` bounded to the turn's reachable namespaces, or unchanged with no allowlist.

    The reachable set reaches the guard only from configuration, never from a
    call argument, so the model cannot widen it.
    """
    reachable = _reachable_namespaces(sources.get("kubernetes", {}))
    if not reachable:
        return tool
    default = reachable[0]
    choices = ", ".join((f"{default} (default)", *reachable[1:]))
    properties = {
        **tool.input_schema.get("properties", {}),
        _NAMESPACE: {
            "type": "string",
            "default": default,
            "description": (
                f"Kubernetes namespace to query, one of: {choices}. "
                "Omit it to use the default namespace."
            ),
        },
    }
    refusal = (
        f"This namespace is not one this tool may query. Namespaces it may use: {choices}. "
        "Omit namespace to use the default namespace. Any other namespace must first be "
        "added to the Kubernetes namespace allowlist by an operator."
    )
    allowed = frozenset(reachable)
    run = tool.run

    def scoped_run(**kwargs: Any) -> Any:
        if kwargs.get(_NAMESPACE) not in allowed:
            return tool_access_denied("kubernetes", refusal)
        return run(**kwargs)

    return replace(
        tool,
        input_schema={**tool.input_schema, "properties": properties},
        injected_params=tuple(name for name in tool.injected_params if name != _NAMESPACE),
        run=scoped_run,
    )
