"""A kubernetes namespace allowlist bounds what the model is offered and what the tools query.

The allowlist and the default namespace come from one kubernetes configuration
record; each turn's tool list states the namespaces a call may use, and a call
outside them is refused before any Kubernetes client exists.
"""

from __future__ import annotations

import io
import os
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

import integrations.kubernetes.tools as kubernetes_tools
from config.constants import INTEGRATIONS_STORE_PATH_ENV, ORGANIZATION_ID_ENV, WEBAPP_URL_ENV
from config.constants.paths import integrations_store_path
from config.constants.tool_discovery import TOOL_SEARCH_NAME
from core.agent_harness.spi.integrations import resolve_and_cache_integrations
from core.agent_harness.tools import ActionToolScope
from core.agent_harness.tools.action_tools import get_action_tools_from_integrations_view
from core.agent_harness.tools.tool_provider import DefaultToolProvider
from core.llm.shared.tool_schema_normalize import build_openai_tool_specs
from core.llm.types import ToolCall
from core.tool import LiveToolCatalog, ToolExecutionResult
from core.tool.execution import execute_tool_calls
from integrations.kubernetes import classify
from integrations.store import upsert_integration
from surfaces.interactive_shell.session import Session

# Each namespace-scoped tool with the smallest input its public schema accepts.
_NAMESPACED_TOOLS: dict[str, dict[str, Any]] = {
    "kubernetes_list_pods": {},
    "kubernetes_get_pod_logs": {"pod_name": "api-0"},
    "kubernetes_get_events": {},
    "kubernetes_describe_pod": {"pod_name": "api-0"},
    "kubernetes_list_deployments": {},
    "kubernetes_list_statefulsets": {},
    "kubernetes_list_daemonsets": {},
    "kubernetes_list_services": {},
    "kubernetes_list_ingresses": {},
    "kubernetes_list_configmaps": {},
    "kubernetes_get_resource": {"resource_type": "deployment", "name": "api"},
}
_LIST_NODES = "kubernetes_list_nodes"
_CONNECTION_FIELDS = frozenset({"kubeconfig", "kubeconfig_path", "context"})
_KUBECONFIG_PATH = "/tmp/opensre-test-kubeconfig"
# The developer's real ``.env`` is loaded for the whole session; any of these
# would add a kubernetes record or switch resolution to a remote source.
_AMBIENT_ENV = (
    "KUBECONFIG",
    "KUBECONFIG_CONTENT",
    "KUBECONFIG_CONTEXT",
    "KUBECONFIG_NAMESPACE",
    "KUBECONFIG_NAMESPACES",
    "JWT_TOKEN",
    WEBAPP_URL_ENV,
    ORGANIZATION_ID_ENV,
)

ClusterCalls = list[tuple[str, Any]]


@pytest.fixture
def cluster(monkeypatch: pytest.MonkeyPatch) -> ClusterCalls:
    """Replace the Kubernetes API client; record each construction and each call's namespace."""
    calls: ClusterCalls = []

    class _RecordingClient:
        def __init__(self, config: Any) -> None:
            calls.append(("client", config.namespace))

        def __enter__(self) -> _RecordingClient:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def __getattr__(self, method: str) -> Callable[..., dict[str, Any]]:
            def call(**kwargs: Any) -> dict[str, Any]:
                calls.append((method, kwargs.get("namespace")))
                return defaultdict(list, success=True)

            return call

    monkeypatch.setattr(kubernetes_tools, "KubernetesClient", _RecordingClient)
    return calls


@pytest.fixture
def local_sources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> pytest.MonkeyPatch:
    """No ambient kubernetes or remote configuration; an empty store at a temp path."""
    for name in _AMBIENT_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(INTEGRATIONS_STORE_PATH_ENV, str(tmp_path / "integrations.json"))
    return monkeypatch


def _store_kubernetes(credentials: dict[str, Any]) -> None:
    """Write the stored kubernetes record and move the store stamp forward."""
    path = integrations_store_path()
    previous = path.stat().st_mtime_ns if path.exists() else 0
    upsert_integration(
        "kubernetes", {"credentials": {"kubeconfig_path": _KUBECONFIG_PATH, **credentials}}
    )
    # Two rewrites within the filesystem's timestamp resolution would share a stamp.
    stamp = max(path.stat().st_mtime_ns, previous + 1_000_000_000)
    os.utime(path, ns=(stamp, stamp))


def _turn(session: Session) -> tuple[list[Any], dict[str, Any]]:
    """Build one turn's tool list as the action driver does, with the kubernetes tools revealed."""
    resolved = resolve_and_cache_integrations(session)
    provider = DefaultToolProvider(session, Console(file=io.StringIO()))
    listed = provider.action_tools(confirm_fn=None, is_tty=False, resolved_integrations=resolved)
    search = next(tool for tool in listed if tool.name == TOOL_SEARCH_NAME)
    search.run(names=[*_NAMESPACED_TOOLS, _LIST_NODES])
    catalog = LiveToolCatalog.from_resources(provider.tool_resources())
    assert catalog is not None
    return list(catalog.snapshot()), resolved


def _parameters(tools: list[Any], name: str) -> dict[str, Any]:
    """The JSON-schema properties a provider is sent for ``name``."""
    specs = [spec for spec in build_openai_tool_specs(tools) if spec["function"]["name"] == name]
    assert len(specs) == 1, f"{name} is not in the turn's tool list"
    properties: dict[str, Any] = specs[0]["function"]["parameters"]["properties"]
    return properties


def _call(
    tools: list[Any], resolved: dict[str, Any], name: str, **arguments: Any
) -> ToolExecutionResult:
    call = ToolCall(id="call-1", name=name, input={**_NAMESPACED_TOOLS[name], **arguments})
    [result] = execute_tool_calls([call], tools, resolved)
    return result


def _assert_offers(tools: list[Any], *, default: str, reachable: tuple[str, ...]) -> None:
    for name in _NAMESPACED_TOOLS:
        properties = _parameters(tools, name)
        offered = properties.get("namespace")
        assert offered is not None, f"{name} does not offer namespace"
        assert offered.get("default") == default, f"{name} names another default"
        missing = [ns for ns in reachable if ns not in offered.get("description", "")]
        assert not missing, f"{name} does not state {missing}"
        assert not _CONNECTION_FIELDS & set(properties), f"{name} offers a connection field"
    assert "namespace" not in _parameters(tools, _LIST_NODES)


def _assert_refused(result: ToolExecutionResult, reachable: tuple[str, ...]) -> None:
    assert result.is_error, "a namespace outside the allowlist was not refused"
    content = str(result.content)
    missing = [ns for ns in reachable if ns not in content]
    assert not missing, f"the refusal does not name {missing}"


def test_a_stored_allowlist_follows_its_record_on_the_same_session(
    local_sources: pytest.MonkeyPatch, cluster: ClusterCalls
) -> None:
    session = Session()

    # The wizard's shape: a comma string. The default is not on the list.
    _store_kubernetes({"namespace": "opensre", "namespaces": "staging, payments"})
    tools, resolved = _turn(session)
    _assert_offers(tools, default="opensre", reachable=("opensre", "staging", "payments"))
    _call(tools, resolved, "kubernetes_list_pods", namespace="payments")
    _call(tools, resolved, "kubernetes_list_pods")
    assert cluster == [
        ("client", "payments"),
        ("list_pods", "payments"),
        ("client", "opensre"),
        ("list_pods", "opensre"),
    ]
    cluster.clear()
    refused = _call(tools, resolved, "kubernetes_list_pods", namespace="kube-system")
    _assert_refused(refused, ("opensre", "staging", "payments"))
    assert cluster == []

    # A hand-written record: a list, with the default on it.
    _store_kubernetes({"namespace": "staging", "namespaces": ["staging", "billing"]})
    tools, resolved = _turn(session)
    _assert_offers(tools, default="staging", reachable=("staging", "billing"))
    stated = _parameters(tools, "kubernetes_list_pods")["namespace"]["description"]
    assert "payments" not in stated
    assert "opensre" not in stated
    refused = _call(tools, resolved, "kubernetes_list_pods", namespace="payments")
    _assert_refused(refused, ("staging", "billing"))
    _call(tools, resolved, "kubernetes_list_pods", namespace="billing")
    assert cluster == [("client", "billing"), ("list_pods", "billing")]
    cluster.clear()

    # The allowlist removed while the process runs.
    _store_kubernetes({"namespace": "billing"})
    tools, resolved = _turn(session)
    for name in _NAMESPACED_TOOLS:
        assert "namespace" not in _parameters(tools, name), f"{name} still offers namespace"
    _call(tools, resolved, "kubernetes_list_pods", namespace="staging")
    assert cluster == [("client", "billing"), ("list_pods", "billing")]


def test_an_env_allowlist_ignores_surrounding_whitespace_and_empty_entries(
    local_sources: pytest.MonkeyPatch, cluster: ClusterCalls
) -> None:
    local_sources.setenv("KUBECONFIG", _KUBECONFIG_PATH)
    local_sources.setenv("KUBECONFIG_NAMESPACE", "opensre")
    local_sources.setenv("KUBECONFIG_NAMESPACES", " staging , ,payments ,")

    tools, resolved = _turn(Session())

    _assert_offers(tools, default="opensre", reachable=("opensre", "staging", "payments"))
    _call(tools, resolved, "kubernetes_list_pods", namespace="payments")
    assert cluster == [("client", "payments"), ("list_pods", "payments")]


def test_a_stored_record_without_an_allowlist_is_not_widened_by_the_env_allowlist(
    local_sources: pytest.MonkeyPatch, cluster: ClusterCalls
) -> None:
    local_sources.setenv("KUBECONFIG", _KUBECONFIG_PATH)
    local_sources.setenv("KUBECONFIG_NAMESPACES", "staging")
    _store_kubernetes({"namespace": "opensre"})

    tools, resolved = _turn(Session())

    for name in _NAMESPACED_TOOLS:
        assert "namespace" not in _parameters(tools, name), f"{name} offers namespace"
    _call(tools, resolved, "kubernetes_list_pods", namespace="staging")
    assert cluster == [("client", "opensre"), ("list_pods", "opensre")]


# More reachable names than a schema-validation hint would list.
_LISTED = tuple(f"team-{index:02d}" for index in range(13))
_REACHABLE = ("default", *_LISTED)


def _allowlisted_turn() -> tuple[list[Any], dict[str, Any]]:
    """A turn whose kubernetes record has the default ``default`` and thirteen listed names."""
    config, _ = classify(
        {"kubeconfig_path": _KUBECONFIG_PATH, "namespaces": ",".join(_LISTED)}, "kubernetes-1"
    )
    assert config is not None
    resolved: dict[str, Any] = {"kubernetes": config}
    scope = ActionToolScope(session=Session(), console=Console(file=io.StringIO()), is_tty=False)
    return get_action_tools_from_integrations_view(scope, resolved_integrations=resolved), resolved


@pytest.mark.parametrize("name", list(_NAMESPACED_TOOLS))
def test_each_namespaced_tool_refuses_an_unlisted_namespace_before_a_client_exists(
    name: str, cluster: ClusterCalls
) -> None:
    tools, resolved = _allowlisted_turn()

    result = _call(tools, resolved, name, namespace="kube-system")

    _assert_refused(result, _REACHABLE)
    assert cluster == []


def test_a_namespace_must_match_a_reachable_name_exactly(cluster: ClusterCalls) -> None:
    tools, resolved = _allowlisted_turn()

    near_misses = [
        _call(tools, resolved, "kubernetes_list_pods", namespace=near)
        for near in ("Team-03", " team-03", "team-03 ")
    ]
    _call(tools, resolved, "kubernetes_list_pods", namespace="team-03")
    _call(tools, resolved, "kubernetes_list_pods")

    for refused in near_misses:
        _assert_refused(refused, _REACHABLE)
        assert refused.details["failure_state"] == "denied"
        assert "setup_command" not in refused.details
    assert "Team-03" not in str(near_misses[0].content)
    assert cluster == [
        ("client", "team-03"),
        ("list_pods", "team-03"),
        ("client", "default"),
        ("list_pods", "default"),
    ]


def test_a_cluster_scoped_lookup_outside_the_set_is_refused_and_succeeds_without_one(
    cluster: ClusterCalls,
) -> None:
    tools, resolved = _allowlisted_turn()
    node = {"resource_type": "node", "name": "worker-1"}

    refused = _call(tools, resolved, "kubernetes_get_resource", namespace="kube-system", **node)
    assert cluster == []
    found = _call(tools, resolved, "kubernetes_get_resource", **node)

    _assert_refused(refused, _REACHABLE)
    assert not found.is_error
    assert cluster == [("client", "default"), ("get_resource", "default")]
