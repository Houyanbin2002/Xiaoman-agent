from __future__ import annotations

from pathlib import Path

from agent.config_models import Config
from core.memory.coordinator import CompositeMemoryEngine
from core.memory.personal_semantic import PersonalSemanticRecallService
from core.memory.plugin import MemoryPluginBuildDeps, MemoryPluginRuntime
from infra.persistence.personal_memory_vector_store import PersonalMemoryVectorStore
from plugins.akasha.config import (
    ensure_akasha_config_file,
    load_akasha_config,
    resolve_akasha_db_path,
)
from plugins.akasha.engine import AkashaMemoryEngine
from plugins.default_memory.config import (
    ensure_default_memory_config_file,
    load_default_memory_config,
    resolve_memory_db_path,
)
from plugins.default_memory.engine import DefaultMemoryEngine


class MemoryPlugin:
    plugin_id = "akasha"

    # 准备 Akasha sidecar 存储。
    def ensure_workspace_storage(
        self,
        *,
        config: Config,
        workspace: Path,
    ) -> list[tuple[Path, bool]]:
        # 1. 确保插件配置存在，并按配置解析数据库路径。
        _ = config
        _ = ensure_akasha_config_file()
        akasha_config = load_akasha_config()
        db_path = resolve_akasha_db_path(
            workspace=workspace,
            akasha_config=akasha_config,
        )
        existed = db_path.exists()

        # 2. Akasha 只是经历索引；执行经验使用独立的结构化存储。
        AkashaMemoryEngine.ensure_workspace_storage(
            akasha_config=akasha_config,
            workspace=workspace,
        )
        _ = ensure_default_memory_config_file()
        default_config = load_default_memory_config()
        execution_path = resolve_memory_db_path(
            workspace=workspace,
            default_config=default_config,
        )
        execution_existed = execution_path.exists()
        DefaultMemoryEngine.ensure_workspace_storage(
            default_config=default_config,
            workspace=workspace,
        )
        return [(db_path, existed), (execution_path, execution_existed)]

    # 构造 Akasha memory runtime。
    def build(
        self,
        deps: MemoryPluginBuildDeps,
    ) -> MemoryPluginRuntime:
        # 1. 三个记忆域由协调器组合，彼此不复制对方的数据。
        akasha_config = load_akasha_config()
        episodic = AkashaMemoryEngine(
            config=deps.config,
            akasha_config=akasha_config,
            workspace=deps.workspace,
            http_resources=deps.http_resources,
            event_publisher=deps.event_publisher,
        )
        default_config = load_default_memory_config()
        structured = DefaultMemoryEngine(
            config=deps.config,
            default_config=default_config,
            workspace=deps.workspace,
            provider=deps.provider,
            light_provider=deps.light_provider,
            http_resources=deps.http_resources,
            event_publisher=deps.event_publisher,
            enable_conversation_ingest=False,
        )
        personal_semantic = PersonalSemanticRecallService(
            store=PersonalMemoryVectorStore(deps.workspace / "personal.db"),
            embedder=structured.embedding_provider,
            model=_embedding_index_key(
                deps.config.memory.embedding.model,
                deps.config.memory.embedding.output_dimensionality,
            ),
        )
        engine = CompositeMemoryEngine(
            structured=structured,
            episodic=episodic,
            personal_semantic=personal_semantic,
        )
        return MemoryPluginRuntime(
            engine=engine,
            closeables=[engine],
            admin=engine,
        )


def _embedding_index_key(model: str, dimensions: int | None) -> str:
    return f"{model.strip() or 'default'}:{dimensions or 'native'}"
