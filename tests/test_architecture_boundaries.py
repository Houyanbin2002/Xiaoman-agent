from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
INTERNAL_ROOTS = {
    "agent",
    "bootstrap",
    "bus",
    "core",
    "infra",
    "memory2",
    "plugins",
    "proactive_v2",
    "prompts",
    "session",
}


def _internal_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        imports.update(name for name in names if name.split(".")[0] in INTERNAL_ROOTS)
    return imports


def _assert_package_boundary(relative: str, allowed_roots: set[str]) -> None:
    violations: list[str] = []
    for path in sorted((PROJECT_ROOT / relative).rglob("*.py")):
        for imported in sorted(_internal_imports(path)):
            root = imported.split(".")[0]
            if root not in allowed_roots:
                violations.append(
                    f"{path.relative_to(PROJECT_ROOT)} -> {imported}"
                )
    assert violations == [], "跨层依赖:\n" + "\n".join(violations)


def test_domain_contract_packages_only_depend_on_core() -> None:
    for package in ("core/llm", "core/personal", "core/workflow"):
        _assert_package_boundary(package, {"core"})


def test_workflow_application_does_not_depend_on_infrastructure() -> None:
    _assert_package_boundary("agent/workflows", {"agent", "core"})


def test_personal_tools_only_depend_on_application_and_core() -> None:
    _assert_package_boundary("agent/tools/personal", {"agent", "core"})


def test_infrastructure_adapters_do_not_depend_on_application_runtime() -> None:
    for package in ("infra/providers", "infra/persistence"):
        _assert_package_boundary(package, {"core", "infra"})


def test_runtime_packages_never_import_the_composition_root() -> None:
    violations: list[str] = []
    for package in INTERNAL_ROOTS - {"bootstrap"}:
        for path in sorted((PROJECT_ROOT / package).rglob("*.py")):
            for imported in sorted(_internal_imports(path)):
                if imported == "bootstrap" or imported.startswith("bootstrap."):
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)} -> {imported}"
                    )
    assert violations == [], "运行时反向依赖 bootstrap:\n" + "\n".join(violations)


def test_internal_code_uses_core_llm_contracts() -> None:
    offenders: list[str] = []
    for package in INTERNAL_ROOTS:
        for path in sorted((PROJECT_ROOT / package).rglob("*.py")):
            if path == PROJECT_ROOT / "agent/provider.py":
                continue
            if "agent.provider" in _internal_imports(path):
                offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == []


def test_removed_legacy_modules_do_not_return() -> None:
    removed = (
        "agent/background/runtime.py",
        "agent/background/subagent_manager.py",
        "agent/looping/constants.py",
        "agent/looping/handlers.py",
        "agent/policies/__init__.py",
        "agent/policies/delegation.py",
        "agent/policies/history_route.py",
        "agent/tools/spawn.py",
        "agent/tools/personal.py",
        "bootstrap/dashboard_api.py",
        "bootstrap/dashboard_management.py",
        "core/personal/assistance.py",
        "core/personal/rhythm.py",
        "bus/events_proactive.py",
        "bus/internal_events.py",
        "memory2/dedup_decider.py",
        "memory2/hyde_enhancer.py",
        "memory2/injection_planner.py",
        "memory2/models.py",
        "memory2/query_rewriter.py",
        "memory2/sufficiency_checker.py",
        "prompts/proactive.py",
    )
    assert [item for item in removed if (PROJECT_ROOT / item).exists()] == []
