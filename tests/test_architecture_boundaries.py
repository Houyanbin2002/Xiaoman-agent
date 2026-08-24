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
                violations.append(f"{path.relative_to(PROJECT_ROOT)} -> {imported}")
    assert violations == [], "跨层依赖:\n" + "\n".join(violations)


def _production_module_graph() -> dict[str, set[str]]:
    modules: dict[str, Path] = {}
    for package in INTERNAL_ROOTS:
        for path in (PROJECT_ROOT / package).rglob("*.py"):
            relative = path.relative_to(PROJECT_ROOT).with_suffix("")
            parts = (
                relative.parts[:-1] if relative.name == "__init__" else relative.parts
            )
            modules[".".join(parts)] = path

    graph = {name: set[str]() for name in modules}
    for name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, ast.Import):
                candidates.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    imported_module = f"{node.module}.{alias.name}"
                    candidates.append(
                        imported_module if imported_module in modules else node.module
                    )
            graph[name].update(
                candidate for candidate in candidates if candidate in modules
            )
    return graph


def _dependency_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    index = 0
    indexes: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    active: set[str] = set()
    cycles: list[list[str]] = []

    def visit(module: str) -> None:
        nonlocal index
        indexes[module] = index
        lowlinks[module] = index
        index += 1
        stack.append(module)
        active.add(module)
        for dependency in graph[module]:
            if dependency not in indexes:
                visit(dependency)
                lowlinks[module] = min(lowlinks[module], lowlinks[dependency])
            elif dependency in active:
                lowlinks[module] = min(lowlinks[module], indexes[dependency])
        if lowlinks[module] != indexes[module]:
            return
        component: list[str] = []
        while stack:
            dependency = stack.pop()
            active.remove(dependency)
            component.append(dependency)
            if dependency == module:
                break
        if len(component) > 1:
            cycles.append(sorted(component))

    for module in graph:
        if module not in indexes:
            visit(module)
    return sorted(cycles)


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


def test_core_has_no_runtime_layer_dependencies() -> None:
    _assert_package_boundary("core", {"core"})


def test_production_modules_have_no_import_cycles() -> None:
    cycles = _dependency_cycles(_production_module_graph())
    assert cycles == [], "生产模块循环依赖:\n" + "\n".join(
        " -> ".join(cycle) for cycle in cycles
    )


def test_runtime_packages_never_import_the_composition_root() -> None:
    violations: list[str] = []
    for package in INTERNAL_ROOTS - {"bootstrap"}:
        for path in sorted((PROJECT_ROOT / package).rglob("*.py")):
            for imported in sorted(_internal_imports(path)):
                if imported == "bootstrap" or imported.startswith("bootstrap."):
                    violations.append(f"{path.relative_to(PROJECT_ROOT)} -> {imported}")
    assert violations == [], "运行时反向依赖 bootstrap:\n" + "\n".join(violations)
