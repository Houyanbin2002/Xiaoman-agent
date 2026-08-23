# LangGraph 执行架构迁移说明

## 当前执行架构

```text
频道 / Dashboard
        |
        v
外部消息网关（只负责收发、鉴权和格式转换）
        |
        v
主 Agent StateGraph
  model -> tool(逐工具检查点) -> governance -> finalize
        |                  |
        |                  +-- SQLite Checkpointer / thread_id
        +-- task_create --> Workflow StateGraph
                              dependency fan-out
                              interrupt / Command(resume)
                              retry delay / terminal notify

隔离 SubAgent = 同一 Agent StateGraph + 独立 thread、工具集和执行策略
Proactive = phase modules 编译为顺序 StateGraph
```

## 数据边界

- `langgraph-checkpoints.db`：主 Agent 的 thread 状态、模型/工具节点位置和恢复点。
- `langgraph-workflows.db`：Workflow 的依赖路由、等待节点和恢复点。
- `langgraph-workflow-index.db`：为 Dashboard 和 `task_manage` 提供同步列表查询的投影，不负责调度。
- 长期记忆仍由项目的记忆治理层负责抽取、去重、置信度和保留策略；LangGraph Store 是运行时接口，不替代这些业务规则。
- EventBus 只保留跨边界通知、插件观察和频道流式输出；Agent 内部控制流由图边决定。

## 恢复语义

- 每次模型节点、每个工具调用、Workflow 推进节点都会提交检查点。
- 用户终止或进程退出后，同一 `thread_id` 检测到未完成节点会自动继续。
- 外部副作用仍必须通过权限 Hook；节点恢复提供的是至少一次执行语义，工具本身应使用幂等键避免极窄窗口内的重复提交。
- 新建会话会获得新的 `thread_id`，不会继承旧会话的未完成任务。

## 已删除的旧路径

- 主 Agent 里的手写 `while` ReAct 循环。
- Workflow Runtime 的全局轮询、抢占并派发步骤循环。
- 生产环境对旧 `workflows.db` 的兼容读取。
