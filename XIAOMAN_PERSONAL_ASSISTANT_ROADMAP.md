# Xiaoman 个人助手路线图

Xiaoman 的目标不是做一个“会聊天的工具箱”，而是做一个长期陪伴型个人助手：能理解你的身体状态、训练计划、任务压力、个人笔记和情绪节奏，并在合适的时候主动帮你回顾、提醒和调整。

## 当前实现进度

已完成：

- `personal.db` 统一个人数据模型与版本化迁移
- 来源、可信度、敏感等级、有效期、锁定、确认、替代、过期和遗忘生命周期
- 可靠事件、去重、租约重试、死信、幂等操作、审批与审计底座
- 晨间简报、晚间回顾、承诺捕获的 Task Runtime 模板
- Dashboard“我的一天”工作台与个人数据 API
- 事实、偏好、临时状态和历史事件的记忆治理模型
- 记忆来源/可信度/有效期/锁定、敏感分级、冲突确认、完整导出和彻底删除
- 会议准备、截止风险、睡眠训练冲突、持续延期、关系联络、情绪下降和来源变化七类主动规则
- 主动洞察的去重冷却、持久化、稍后提醒、忽略、关闭同类规则和 Dashboard 管理
- 主动协助轮询与聊天投递；每次消息说明“为什么现在提醒”
- 统一个人节奏上下文：场景、专注、免打扰和精力状态
- 可用时间任务推荐，以及可由插件注册的新推荐来源
- 关系、重要日期、账单续费、旅行清单和目标领域记录
- 周报、月报、目标偏差分析和可注册报告指标
- AI 可沉淀的通用主动关注意图；推断项待确认、明确授权项可启用
- 根据接受、忽略和暂缓反馈学习提醒冷却与表达风格

仍需外部授权或数据源：

- Notion 或其他待办/笔记服务
- 日历服务
- 小米手表或健身记录数据桥接
- 手机端主动通知渠道

## 产品目标

Xiaoman 应该具备四类核心能力：

- 健康与训练：读取手表、健身 App、训练记录，理解当天状态并给出复盘和计划调整。
- 日程与事项：接入 Notion 或提醒系统，主动提醒当天要完成的事。
- 生活习惯：定时提醒喝水、休息、睡觉、运动恢复。
- 个人记忆与情感陪伴：读取个人笔记，沉淀长期记忆，在交流中保持持续理解和温度。

## 当前项目可复用能力

Xiaoman 已经具备适合 Xiaoman 的底座：

- `proactive_v2/`：主动推送框架，可用于训练复盘、喝水提醒、事项提醒。
- `agent/tools/schedule.py`：定时任务工具，可用于喝水、每日回顾、睡前总结。
- `core/memory/` + `plugins/default_memory/`：长期记忆与语义检索，可承载个人画像、偏好和笔记摘要。
- `agent/plugins/`：插件系统，适合接入手表、Notion、健身 App、笔记系统。
- `proactive_v2` MCP source：适合把外部数据源统一抽象成 alert/content/context。
- `Dashboard`：可扩展成 Xiaoman 的个人状态面板。
- `FitbitIntegrationConfig`：目前只是配置占位，可以作为健康集成的起点。
- `core/workflow/` + `agent/workflows/`：已实现统一任务中心；Workflow 负责持久化编排，Subagent 作为受跟踪的步骤执行器，可承载训练复盘、等待用户反馈、审批、重试和跨重启继续执行。

## 推荐架构

建议把 Xiaoman 拆成几个领域插件，而不是把逻辑直接塞进 Agent 核心。

```text
Xiaoman Core
  ├─ Durable Workflow Runtime
  │   ├─ Agent steps
  │   ├─ User-input waits
  │   ├─ Approval gates
  │   └─ Retry / recovery / task ledger
  ├─ Health Plugin
  │   ├─ Watch connector
  │   ├─ Fitness app connector
  │   ├─ Daily health summary
  │   └─ Training review signals
  ├─ Planner Plugin
  │   ├─ Notion tasks
  │   ├─ Daily priorities
  │   └─ Reminder scheduling
  ├─ Notes Memory Plugin
  │   ├─ Personal note ingestion
  │   ├─ Semantic indexing
  │   └─ Memory synchronization
  ├─ Habit Plugin
  │   ├─ Water reminders
  │   ├─ Sleep reminders
  │   └─ Recovery prompts
  └─ Companion Profile
      ├─ Tone and personality
      ├─ Emotional check-ins
      └─ Long-term relationship memory
```

## 能力设计

### 1. 手表与健康数据

目标：

- 获取步数、心率、睡眠、静息心率、HRV、卡路里、训练负荷等数据。
- 结合当天训练和近期趋势判断身体状态。
- 主动给出恢复建议、训练强度调整和睡眠提醒。

优先接入路径：

- Fitbit：项目已有配置占位，适合作为第一版健康集成。
- Apple Health：Windows 本地接入难度较高，通常需要 iPhone 导出、快捷指令、第三方同步或中转服务。
- Garmin / Strava / Keep / Zepp / 小米运动：根据你实际使用的 App 选择。

Proactive 输出示例：

```text
你昨晚睡眠 5h42m，静息心率比平时高，今天腿部训练建议降一档；
可以保留动作，但把最后两组改成 RPE 7 左右。
```

### 2. 健身记录 App 与训练回顾

目标：

- 读取当天训练项目、组数、重量、时长、主观疲劳。
- 主动生成训练复盘。
- 对后续计划做小幅调整。

建议数据结构：

```json
{
  "date": "2026-07-09",
  "type": "strength",
  "duration_minutes": 72,
  "exercises": [
    {
      "name": "bench_press",
      "sets": 5,
      "reps": "5,5,5,4,4",
      "weight": "80kg",
      "rpe": 8
    }
  ],
  "notes": "最后两组速度下降明显"
}
```

Xiaoman 可以做：

- 当日训练总结
- 动作表现趋势判断
- 疲劳风险提醒
- 下一次训练建议
- 与睡眠、心率等健康数据联合分析

### 3. 喝水和生活习惯提醒

目标：

- 按时间段提醒喝水。
- 根据运动、天气、咖啡摄入、睡眠等因素调整频率。
- 避免机械打扰，支持“今天别提醒我太多”。

第一版可以直接基于 `schedule` 工具实现：

- 早上启动一组当天提醒
- 中午检查是否需要补水
- 训练前后增加提醒
- 晚上降低频率

后续可以升级成 Habit Plugin，由 proactive 根据状态决定是否提醒。

### 4. Notion 与每日事项

目标：

- 读取 Notion 中今日任务、项目、截止日期。
- 主动提醒当天重点。
- 晚上回顾完成情况。
- 帮你把聊天中出现的新任务写回 Notion。

建议能力：

- `notion_list_today_tasks`
- `notion_create_task`
- `notion_update_task_status`
- `notion_summarize_day`

Proactive 输出示例：

```text
今天还有 3 件事没收口：项目日报、健身记录、给导师回邮件。
我建议先处理邮件，时间最短，而且会减少后面心理占用。
```

### 5. 个人笔记作为记忆

目标：

- 把你的个人笔记变成 Xiaoman 可检索的长期上下文。
- 不只是全文塞进 prompt，而是摘要、索引、检索、引用。
- 对关键偏好、价值观、长期项目形成稳定记忆。

建议流程：

```text
个人笔记源
  -> 增量扫描
  -> 文档切块
  -> 摘要与标签
  -> 写入语义记忆
  -> 重要事实进入 personal.db 统一治理
```

适合接入：

- Notion 页面
- Obsidian / Markdown 笔记
- 本地文件夹
- 飞书文档
- Logseq

原则：

- 原文保留在原系统
- Xiaoman 存摘要、索引和引用
- 敏感内容需要可删除、可审计

### 6. 情感陪伴

目标：

- Xiaoman 要能记住你的压力源、偏好、近期状态。
- 主动关心，但不过度打扰。
- 给出支持时不要空泛，要结合具体事实。

可以通过三层实现：

- `SELF.md`：定义 Xiaoman 的人格、语气、边界。
- `personal.db`：治理你的长期偏好、压力模式、重要关系和目标。
- `RECENT_CONTEXT.md`：保留近期状态，例如最近熬夜、训练压力、项目 deadline。

Proactive 输出示例：

```text
你这两天一直在处理环境和项目迁移，晚上如果还要训练，建议别冲重量。
可以做一个 40 分钟维持训练，然后早点睡。你现在更需要恢复，不是硬扛。
```

## 实施阶段

### Phase 1：本地可用的 Xiaoman

目标：先让 Xiaoman 能稳定聊天、记事、提醒。

- 配好主模型。
- 打开长期记忆。
- 调整 `SELF.md` 和 system prompt。
- 接入一个主要通信渠道。
- 实现喝水提醒和每日总结。

验收标准：

- 能正常启动。
- 能主动发提醒。
- 能记住用户明确要求记住的事实。
- 能查询和取消提醒。

### Phase 2：Notion 和任务助手

目标：让 Xiaoman 能知道你今天要做什么。

- 接入 Notion API。
- 读取今日任务。
- 主动提醒截止事项。
- 支持从聊天创建任务。
- 支持晚上复盘任务完成情况。

验收标准：

- 每天早上能生成今日重点。
- 到点能提醒未完成事项。
- 可以通过对话新增/更新任务。

### Phase 3：健康与训练回顾

目标：让 Xiaoman 能理解你的身体状态和训练。

- 选择一个健康数据源优先接入。
- 选择一个健身记录源优先接入。
- 建立 daily health summary。
- 建立 training review proactive source。

验收标准：

- 训练后能主动总结。
- 能结合睡眠/心率/训练负荷给建议。
- 能生成下一次训练微调建议。

### Phase 4：个人笔记记忆

目标：让 Xiaoman 能读懂你的长期笔记。

- 接入 Notion 或本地 Markdown 笔记。
- 做增量索引。
- 把稳定偏好和长期项目沉淀到记忆。
- 支持回答“我之前怎么想这个问题的？”

验收标准：

- 能检索笔记中的历史观点。
- 能引用来源。
- 能把长期事实写入记忆。

### Phase 5：个人状态面板

目标：让 Dashboard 成为 Xiaoman 的控制台。

- 今日健康状态
- 今日任务
- 喝水/习惯提醒
- 近期记忆
- 主动推送记录
- 插件连接状态

## 技术优先级

建议先做低风险、高收益的能力：

1. 喝水提醒：现有 scheduler 就能支撑。
2. Notion 今日任务：API 清晰，数据结构简单。
3. 长期记忆开启与人格设定：直接提升陪伴感。
4. 健身记录导入：先用手动 CSV/JSON，再接正式 App。
5. 手表数据：最后做，因为 OAuth、厂商 API、同步链路最容易卡。

## 数据与隐私原则

Xiaoman 会接触非常私人的数据，必须从一开始就设计边界：

- API key 只放本地配置，不提交仓库。
- 健康数据、笔记、任务都保存在本地 workspace 或用户指定系统。
- 所有自动写入记忆的内容都应可查看、可删除。
- 情绪与健康建议只能作为生活辅助，不替代医疗建议。
- 主动推送要有频率限制和安静时段。

## 下一步建议

最小可用版本可以先做三件事：

1. 打开并验证长期记忆。
2. 做一个 Habit Plugin，用 scheduler 实现喝水提醒。
3. 做一个 Notion Plugin，读取今日任务并支持每日早晚主动总结。

这三件事完成后，Xiaoman 就会从“能聊天的 Agent”变成“开始参与日常生活的个人助手”。
