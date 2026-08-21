"""
入口

两种模式：
  python main.py          启动 agent 服务（AgentLoop + 所有 channel + IPC server）
  python main.py cli      连接到运行中的 agent（CLI 客户端）
"""

from __future__ import annotations

import asyncio
import json
import signal
import sys
from contextlib import suppress
from pathlib import Path

from agent.config import Config
from agent.plugins.doctor import format_plugin_doctor_report, run_plugin_doctor
from agent.plugins.install import install_git_plugin, uninstall_plugin
from agent.plugins.marketplaces import (
    add_marketplace,
    install_marketplace_plugin,
    list_marketplaces,
    refresh_marketplace,
)
from bootstrap.app import build_app_runtime
from bootstrap.dashboard_api import run_dashboard_api
from bootstrap.init_workspace import InitSummary, init_workspace
from bootstrap.memory import build_memory_admin_runtime
from bootstrap.providers import build_providers
from core.net.http import SharedHttpResources


def _default_workspace() -> Path:
    return Path.home() / ".xiaoman" / "workspace"


def _get_flag_value(args: list[str], flag: str) -> str | None:
    if flag not in args:
        return None
    idx = args.index(flag)
    if idx + 1 >= len(args):
        raise ValueError(f"参数 {flag} 缺少值")
    return args[idx + 1]


def _print_init_summary(summary: InitSummary) -> None:
    def _print_group(title: str, paths: list[Path]) -> None:
        if not paths:
            return
        print(title)
        for path in paths:
            print(f"  {path}")

    _print_group("已创建：", summary.created)
    _print_group("已覆盖：", summary.overwritten)
    _print_group("已跳过：", summary.skipped)
    if summary.notes:
        print("说明：")
        for note in summary.notes:
            print(f"  {note}")
    if summary.next_steps:
        print("\n下一步：")
        for step in summary.next_steps:
            print(f"  {step}")


def _parse_csv_flag(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _print_help() -> None:
    print("""Xiaoman Agent / 小满

用法:
  python main.py [gateway] [--config PATH] [--workspace PATH]
  python main.py cli [--config PATH]
  python main.py dashboard [--host HOST] [--port PORT]
  python main.py setup|init [--config PATH] [--workspace PATH] [--force]
  python main.py plugin-install --source PATH_OR_URL [--marketplace NAME]
  python main.py plugin-uninstall PLUGIN_ID [--purge-data]
  python main.py plugin-marketplace-add --source OWNER/REPOSITORY_OR_URL [--ref REF]
  python main.py plugin-marketplace-list
  python main.py plugin-marketplace-update MARKETPLACE
  python main.py plugin-marketplace-install --marketplace NAME --plugin PLUGIN
  python main.py plugin-doctor [PLUGIN_ID] [--json]

选项:
  --config PATH       配置文件，默认 config.toml
  --workspace PATH    工作区目录
  --host HOST         Dashboard 监听地址，默认 127.0.0.1
  --port PORT         Dashboard 端口，默认 2236
  --unsafe-dashboard-bind
                      允许 Dashboard 监听非回环地址（接口未内置鉴权）
  --inspect-modules   输出生命周期模块拓扑后退出
  -h, --help          显示帮助
""")


def connect_cli(config_path: str = "config.toml") -> None:
    socket_path = Config.load(config_path).channels.socket
    try:
        from infra.channels.cli_tui import run_tui
    except RuntimeError as exc:
        print(exc)
        print("回退到纯文本 CLI。")
        from infra.channels.cli import CLIClient

        asyncio.run(CLIClient(socket_path).run())
        return

    run_tui(socket_path)


def run_standalone_dashboard(
    config_path: str = "config.toml",
    workspace: Path | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 2236,
    allow_remote: bool = False,
) -> None:
    """Run only the Dashboard and memory admin dependencies.

    Channel consumers, schedulers, proactive loops, plugin jobs, and IPC remain owned
    by the gateway process. This keeps the historical standalone command safe to run
    next to an existing gateway.
    """
    config = Config.load(config_path)
    resolved_workspace = workspace or _default_workspace()
    http_resources = SharedHttpResources()
    provider, light_provider, _ = build_providers(config)
    memory_runtime = build_memory_admin_runtime(
        config=config,
        workspace=resolved_workspace,
        provider=provider,
        light_provider=light_provider,
        http_resources=http_resources,
    )
    try:
        run_dashboard_api(
            workspace=resolved_workspace,
            host=host,
            port=port,
            allow_remote=allow_remote,
            memory_admin=memory_runtime.engine,
        )
    finally:
        asyncio.run(memory_runtime.aclose())


async def inspect_modules(
    config_path: str = "config.toml",
    workspace: Path | None = None,
) -> None:
    import logging
    from bootstrap.tools import build_core_runtime

    logging.getLogger().setLevel(logging.WARNING)
    config = Config.load(config_path)
    http_resources = SharedHttpResources()
    runtime = build_core_runtime(
        config,
        workspace or _default_workspace(),
        http_resources,
    )
    try:
        print(await runtime.inspect_modules())
    finally:
        await runtime.stop()
        await http_resources.aclose()


async def serve(
    config_path: str = "config.toml",
    workspace: Path | None = None,
    *,
    dashboard_host: str = "127.0.0.1",
    dashboard_port: int = 2236,
    dashboard_allow_remote: bool = False,
) -> bool:
    restart_event = asyncio.Event()
    config = Config.load(config_path)
    runtime = build_app_runtime(
        config,
        workspace=workspace or _default_workspace(),
        config_path=config_path,
        dashboard_host=dashboard_host,
        dashboard_port=dashboard_port,
        dashboard_allow_remote=dashboard_allow_remote,
        gateway_restart=restart_event.set,
    )
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    watched_signals = (signal.SIGINT, signal.SIGTERM)
    signal_handlers_registered = False
    for sig in watched_signals:
        try:
            loop.add_signal_handler(sig, stop_event.set)
            signal_handlers_registered = True
        except NotImplementedError:
            # Windows' default event loop does not support add_signal_handler.
            signal.signal(
                sig,
                lambda _sig, _frame: loop.call_soon_threadsafe(stop_event.set),
            )

    runtime_task = asyncio.create_task(runtime.run(), name="app_runtime")
    stop_task = asyncio.create_task(stop_event.wait(), name="shutdown_signal")
    restart_task = asyncio.create_task(restart_event.wait(), name="gateway_restart")
    try:
        done, _ = await asyncio.wait(
            {runtime_task, stop_task, restart_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if runtime_task in done:
            _ = stop_task.cancel()
            _ = restart_task.cancel()
            await runtime_task
            return False
        restart_requested = restart_task in done
        _ = runtime_task.cancel()
        with suppress(asyncio.CancelledError):
            await runtime_task
        return restart_requested
    finally:
        if signal_handlers_registered:
            for sig in watched_signals:
                _ = loop.remove_signal_handler(sig)
        _ = stop_task.cancel()
        with suppress(asyncio.CancelledError):
            await stop_task
        _ = restart_task.cancel()
        with suppress(asyncio.CancelledError):
            await restart_task


async def serve_gateway(
    config_path: str = "config.toml",
    workspace: Path | None = None,
    *,
    dashboard_host: str = "127.0.0.1",
    dashboard_port: int = 2236,
    dashboard_allow_remote: bool = False,
) -> None:
    restart_requested = True
    while restart_requested:
        restart_requested = await serve(
            config_path,
            workspace,
            dashboard_host=dashboard_host,
            dashboard_port=dashboard_port,
            dashboard_allow_remote=dashboard_allow_remote,
        )


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--help" in args or "-h" in args:
        _print_help()
        sys.exit(0)
    config_path = "config.toml"
    workspace: Path | None = None
    force = "--force" in args
    dashboard_host = "127.0.0.1"
    dashboard_port = 2236
    dashboard_allow_remote = "--unsafe-dashboard-bind" in args

    try:
        config_value = _get_flag_value(args, "--config")
        workspace_value = _get_flag_value(args, "--workspace")
        host_value = _get_flag_value(args, "--host")
        port_value = _get_flag_value(args, "--port")
        source_value = _get_flag_value(args, "--source")
        marketplace_value = _get_flag_value(args, "--marketplace")
        ref_value = _get_flag_value(args, "--ref")
        sparse_value = _get_flag_value(args, "--sparse")
        subdir_value = _get_flag_value(args, "--subdir")
        plugin_value = _get_flag_value(args, "--plugin")
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)

    if config_value is not None:
        config_path = config_value
    if workspace_value is not None:
        workspace = Path(workspace_value)
    if host_value is not None:
        dashboard_host = host_value
    if port_value is not None:
        dashboard_port = int(port_value)

    if args and args[0] == "setup":
        from bootstrap.setup_wizard import run_setup_wizard

        run_setup_wizard(
            config_path=Path(config_path),
            workspace=workspace or _default_workspace(),
        )
        sys.exit(0)

    if args and args[0] == "init":
        summary = init_workspace(
            config_path=config_path,
            workspace=workspace or _default_workspace(),
            force=force,
        )
        _print_init_summary(summary)
        sys.exit(0)

    if args and args[0] == "plugin-install":
        if not source_value:
            print("plugin-install 缺少 --source")
            sys.exit(1)
        marketplace = marketplace_value or "local"
        result = install_git_plugin(
            source=source_value,
            marketplace=marketplace,
            ref_name=ref_value or "",
            sparse_paths=list(dict.fromkeys([
                *_parse_csv_flag(sparse_value),
                *([subdir_value] if subdir_value else []),
            ])),
            source_subdir=subdir_value or "",
        )
        print(f"已安装插件: {result.plugin_name}@{result.marketplace}")
        print(f"版本: {result.plugin_version}")
        print(f"代码: {result.installed_path}")
        print(f"数据: {result.data_path}")
        sys.exit(0)

    if args and args[0] == "plugin-uninstall":
        if len(args) < 2 or args[1].startswith("--"):
            print("plugin-uninstall 缺少插件 ID")
            sys.exit(1)
        result = uninstall_plugin(
            plugin_id=args[1],
            purge_data="--purge-data" in args,
        )
        print(f"已卸载插件: {result.plugin_id}")
        if result.data_removed:
            print("插件数据已清理。")
        elif result.data_path is not None:
            print(f"插件数据已保留: {result.data_path}")
        print("请重启 Gateway 以移除运行中的插件能力。")
        sys.exit(0)

    if args and args[0] == "plugin-marketplace-add":
        if not source_value:
            print("plugin-marketplace-add 缺少 --source")
            sys.exit(1)
        result = add_marketplace(
            source=source_value,
            ref_name=ref_value or "",
        )
        print(f"已添加插件市场: {result.name}")
        print(f"可安装插件: {len(result.plugins)} 个")
        sys.exit(0)

    if args and args[0] == "plugin-marketplace-list":
        marketplaces = list_marketplaces()
        if not marketplaces:
            print("尚未添加插件市场。")
        for item in marketplaces:
            status = "可用" if item["available"] else "不可用"
            print(
                f"{item['name']} ({status}) - {item['plugin_count']} 个插件 - {item['source']}"
            )
        sys.exit(0)

    if args and args[0] == "plugin-marketplace-update":
        if len(args) < 2 or args[1].startswith("--"):
            print("plugin-marketplace-update 缺少市场名称")
            sys.exit(1)
        result = refresh_marketplace(args[1])
        print(f"已刷新插件市场: {result.name}（{len(result.plugins)} 个插件）")
        sys.exit(0)

    if args and args[0] == "plugin-marketplace-install":
        if not marketplace_value or not plugin_value:
            print("plugin-marketplace-install 需要 --marketplace 和 --plugin")
            sys.exit(1)
        result = install_marketplace_plugin(
            marketplace=marketplace_value,
            plugin_name=plugin_value,
        )
        print(f"已安装插件: {result.plugin_name}@{result.marketplace}")
        print(f"版本: {result.plugin_version}")
        print("请重启 Gateway 以加载新插件。")
        sys.exit(0)

    if args and args[0] == "plugin-doctor":
        target_plugin_id = ""
        if len(args) >= 2 and not args[1].startswith("--"):
            target_plugin_id = args[1]
        report = run_plugin_doctor(
            plugin_id=target_plugin_id,
            config_path=config_path,
            workspace=workspace or _default_workspace(),
        )
        if "--json" in args:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(format_plugin_doctor_report(report))
        sys.exit(1 if report.get("status") == "broken" else 0)

    if args and args[0] == "gateway":
        asyncio.run(
            serve_gateway(
                config_path,
                workspace,
                dashboard_host=dashboard_host,
                dashboard_port=dashboard_port,
                dashboard_allow_remote=dashboard_allow_remote,
            )
        )
        sys.exit(0)

    if args and args[0] == "dashboard":
        run_standalone_dashboard(
            config_path,
            workspace,
            host=dashboard_host,
            port=dashboard_port,
            allow_remote=dashboard_allow_remote,
        )
        sys.exit(0)

    if not Path(config_path).exists():
        print(
            f"找不到配置文件 {config_path!r}，请先复制 config.example.toml 为 config.toml。"
        )
        sys.exit(1)

    if "--inspect-modules" in args:
        asyncio.run(inspect_modules(config_path, workspace))
    elif "cli" in args:
        connect_cli(config_path)
    else:
        asyncio.run(
            serve(
                config_path,
                workspace,
                dashboard_host=dashboard_host,
                dashboard_port=dashboard_port,
                dashboard_allow_remote=dashboard_allow_remote,
            )
        )
