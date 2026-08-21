# Architecture Boundaries

This project is evolving from a general Xiaoman Agent runtime into Xiaoman, a personal assistant. The codebase should grow by clear responsibility boundaries rather than by adding more logic to the current bootstrap and agent modules.

## Dependency Direction

Keep dependencies flowing toward contracts; only the composition root may see every
layer:

```text
main.py
  -> bootstrap (composition root)
      -> agent / proactive / plugins
      -> infra implementations

agent / proactive / plugins -> core contracts
infra implementations       -> core contracts
```

Domain and application code depend on ports in `core/`; concrete providers and
persistence adapters implement those ports in `infra/`. `bootstrap/` is the only
place that joins the two sides. No runtime or lower-level package may import
application bootstrap code.

## Package Responsibilities

### `bootstrap/`

Application assembly only.

Allowed:

- Build providers, runtime objects, channels, dashboard, proactive loop.
- Wire dependencies together.
- Start and stop long-running services.

Avoid:

- Domain logic.
- Concrete integration logic such as Notion, watches, fitness apps, notes, or habits.
- Tool behavior beyond registering toolsets.

### `agent/`

Passive conversation runtime.

Allowed:

- Agent loop and turn pipeline.
- Prompt rendering.
- Tool orchestration.
- LLM interaction through provider abstractions.
- Retrieval hooks used during a conversation.

Avoid:

- External product integrations.
- Dashboard API logic.
- Proactive polling policy.

### `agent/workflows/`

Durable task orchestration runtime. Workflow is the internal state-machine
implementation; users and the main agent interact with it as the Task Center.

Allowed:

- Claim runnable workflow steps.
- Execute conversational steps through `AgentLoop.process_direct`.
- Execute isolated research/scripting steps through the internal Subagent executor.
- Deliver waiting, approval, completion, and blocked notifications.
- Recover interrupted work after process restart.

Avoid:

- Vendor-specific health, notes, or task rules.
- Defining Xiaoman domain workflows directly in the runtime.

### `proactive_v2/`

主动推送和 Drift runtime.

Allowed:

- Tick scheduling.
- Alert/content/context source evaluation.
- Proactive judge and resolver.
- Drift task selection and execution.

Avoid:

- Vendor-specific integrations directly embedded in proactive logic.
- Long-term memory implementation details beyond memory contracts.

Future rename target: `proactive/`.

### `core/`

Stable domain contracts and shared primitives.

Allowed:

- Memory protocols.
- Shared event and request models.
- Cross-cutting domain abstractions.
- Workflow and step state contracts under `core/workflow/`.

Avoid:

- Concrete HTTP clients.
- Runtime startup code.
- UI/dashboard code.

### `infra/`

Concrete infrastructure.

Allowed:

- HTTP resources.
- Persistence adapters.
- LLM provider implementations.
- Channel transport implementation details.
- SQLite workflow state and event ledger under `infra/persistence/`.

Avoid:

- Personal-assistant business rules.
- Agent turn policy.

### `plugins/`

Internal system extension implementations.

Allowed:

- Memory-engine and channel entrypoints.
- Lifecycle hooks and internal dashboard diagnostics.
- Component-owned data/config.

Avoid:

- Core runtime startup.
- Global assumptions about user configuration.
- User-installable Skills or MCP packages.

### `integrations/`

Personal assistant integrations.

Recommended domains:

- Notion tasks.
- Health/watch data.
- Fitness records.
- Personal notes.
- Habit reminders.

Integrations should expose model-callable capabilities through MCP, and data-only sources through proactive modules or memory ingestion APIs, rather than modifying the core agent loop.

### `session/`

Conversation state and message history.

Allowed:

- Session persistence.
- Message records.
- Conversation lookup support.

Avoid:

- LLM prompting decisions.
- Integration-specific data.

### `bus/`

Message/event delivery infrastructure.

Allowed:

- Inbound/outbound queues.
- Processing state.
- Event dispatch.

Avoid:

- Channel-specific authentication.
- Agent reasoning logic.

## Xiaoman Feature Placement

Use this placement for upcoming personal assistant features:

| Feature | Preferred Location |
| --- | --- |
| Water reminders | `integrations/habits` + shared scheduler |
| Notion daily tasks | Notion MCP + optional Skill |
| Watch/health sync | `integrations/health` ingestion adapter |
| Fitness training review | `integrations/fitness` + personal records |
| Personal note ingestion | standard MCP or `integrations/notes` ingestion adapter |
| Emotional check-ins | proactive module + memory profile |
| Dashboard widgets | plugin dashboard panels |
| Stateful or background execution | task definitions using the shared Task Runtime |

## Personal Data Boundary

Structured personal state is owned by `core/personal/` and persisted in the
workspace `personal.db`. It is separate from conversation history and from the
domain-neutral Task Runtime.

The shared record envelope covers profile data, commitments, calendar events,
health observations, daily plans, check-ins, notification policies, and governed
memories. Every record carries its source, confidence, sensitivity, validity
window, update policy, lifecycle status, revision, and optional lineage.

The same envelope now covers context state, relationships, important dates,
financial obligations, trips, goals, periodic reports, and AI-authored proactive
intents. These are domain facts, not separate execution pipelines.

Lifecycle transitions are explicit:

- `active` records can be confirmed, updated, expired, superseded, or forgotten.
- User-locked records reject automatic updates.
- Forgetting purges record content and redacts historical snapshots while keeping
  a non-sensitive tombstone for audit integrity.
- All timestamps used for validity comparisons are normalized to UTC.

`personal.db` also contains a durable event inbox and idempotent operation ledger.
Event processing uses leases, retry budgets, deduplication keys, and dead-letter
state. External writes use idempotency keys and an approval/audit state machine.
Redis and a distributed message broker are intentionally not required for the
single-user, single-process deployment.

Memory responsibilities are split deliberately:

- `personal.db` is the source of truth for structured personal facts and their
  lifecycle.
- The default Memory v2 store and `sqlite-vec` remain the semantic retrieval and
  conversational-memory layer.
- Integrations should project approved personal records into semantic retrieval
  when useful; they must not create a second structured fact store.

Governed memories pass through `MemoryGovernanceService` before they become
available to the Agent. It distinguishes facts, preferences, temporary state,
and historical events; derives sensitivity and access policy from the data
category; and persists conflicting candidates for explicit user resolution.
Health, emotional, account, relationship, location, and financial memories can
require confirmation or remain owner-only. Users can lock, expire, soft-forget,
hard-delete, and export memories together with their revision and conflict
history.

Event-aware assistance is owned by `core/attention/`. Personal records use the
same signal contract as MCP sources: records may declare an explicit
`attention_signal`, while standard temporal fields receive one conservative,
domain-independent due signal. There is no second insight store, fixed scenario
rule table, or Attention Draft. `ActionPlan` is the single durable decision and
execution record. The shared turn orchestrator remains the only delivery path
for `message.notify`; other capabilities execute through registered handlers.

Conversation consolidation and signal providers may emit traceable
`AttentionObservation` records. `AttentionLearningService` is the only path that
turns those records into behavior patterns or declarative policies. Direct user
instructions can activate immediately; inferred knowledge remains proposed
until repeated evidence crosses the configured threshold. Learned knowledge
decays in bounded periods and feedback updates only the affected dimension.

The opportunity-and-action design is recorded in
[`_handbook/attention-engine-v2.md`](_handbook/attention-engine-v2.md). The
Proactive Loop only composes and delivers a planned outbound message; it no
longer owns a separate attention decision model.

## Personal Rhythm And Opportunities

`core/personal/rhythm/` is the shared context and opportunity layer. It creates
one snapshot from the active scene, focus state, notification boundary, recent
energy check-in, and current time. On-demand guidance, proactive delivery, and
periodic reports all consume this snapshot.

The main extension contracts are:

- `register_recommendation_provider`: plugins contribute candidates without
  replacing ranking, context, or time-window handling.
- `register_report_contributor`: integrations add metrics, deviations, and
  recommendations to the shared weekly/monthly report.
- `SignalProviderRegistry`: integrations contribute evidence-backed changes
  without adding domain branches to the engine.
- `ActionCapabilityRegistry`: integrations declare reusable actions, timing and
  risk so every opportunity can match them through the same planner.
- `proactive_intent`: the Agent can persist interval, time, inactivity, or field
  conditions as data. Inferred intents remain proposed until the user enables
  them; explicitly requested intents can become active immediately.

The Task Runtime is used only when an action needs dependencies, approval,
waiting, retry, or restart recovery. A recommendation, context transition, or
rule evaluation does not create its own workflow.

Feedback is stored per proactive rule. Accepted, dismissed, and snoozed insights
adjust an effective cooldown multiplier and an auto-selected neutral, gentle,
concise, or direct delivery style. Explicit user tone settings override learned
style without deleting feedback history.

The standard personal routines live in `agent/workflows/personal.py` and create
ordinary Task Runtime instances. Morning briefs, evening reviews, and commitment
capture therefore use the same waiting, approval, retry, and recovery behavior as
all other durable tasks.

## Unified Task Runtime

The default `workflow` toolset exposes one public task control surface:

- `task_create`: creates a durable dependency graph containing Agent steps, isolated Subagent steps, user-input waits, and approval gates.
- `task_manage`: inspects progress, supplies user responses, approves or rejects gates, retries failed steps, and cancels tasks.

The former detached-task path has been removed. `SubagentExecutor` is a narrow,
internal step executor owned by the Task Runtime. A Subagent step is awaited, its
result is committed to the durable step ledger, and retries reuse a stable
task-step directory. Task steps cannot recursively create tasks or detached
background jobs.

Each task is now one LangGraph thread. `langgraph-workflows.db` stores graph
checkpoints and interrupt positions; `langgraph-workflow-index.db` is only the
query projection used by tools and the dashboard. Independent dependency nodes
run concurrently, approval and user-input nodes pause with LangGraph
`interrupt`, and management actions resume them with `Command(resume=...)`.
There is no compatibility read from the former `workflows.db` scheduler.

Automatic Agent steps are read-only by default. A write or external-side-effect
tool must be named explicitly in the step's persisted `allowed_tools` and the step
must directly depend on a successfully approved `approval` step. Approval messages
show the exact tools that will be opened; the runtime checks the grant again before
execution. `task_create`, `task_manage`, and `message_push` can never be opened from
inside a task. Subagent `scripting` and `general` profiles also require approval and
their shell is confined to the stable task-step directory.

Cancellation is cooperative and durable: the runtime cancels every active asyncio
step for the task, while persistence uses a RUNNING-state compare-and-set so a late
result cannot overwrite `cancelled` state.

Domain integrations should create workflow definitions and use MCP/tools for external effects. The Workflow Runtime must remain domain-neutral.

## Refactoring Rules

1. Preserve stable package-level public APIs while moving code; delete obsolete
   concrete modules after all internal callers and tests migrate.
2. Move lifecycle orchestration before moving domain logic.
3. Add tests around each extracted boundary.
4. Do not put Xiaoman-specific integrations in `agent/`.
5. Prefer Skill for procedures, MCP for model-callable tools, and internal system extensions only for runtime backends or channels.
6. Keep API keys and personal data out of tracked files.

## Current Package Map

Application assembly is split by lifecycle rather than accumulated in one bootstrap
module:

- Core runtime aggregate: `bootstrap/core_runtime.py`
- Personal runtime aggregate: `bootstrap/personal.py`
- Plugin activation and skill synchronization: `bootstrap/plugin_runtime.py`
- Independent Skill installation: `agent/skill_packages.py`
- MCP connection, catalog, credentials, and transports: `agent/mcp/`
- Internal system extension compatibility and activation: `agent/plugins/`
- Toolset composition and dependency ordering: `bootstrap/toolsets/`
- Dashboard base API, diagnostics, and server lifecycle: `bootstrap/dashboard_api/`
- Isolated Task Runtime step execution: `agent/background/subagent_executor.py`
- Dashboard management composition: `bootstrap/dashboard_management/`
- Dashboard route groups: `bootstrap/dashboard_management/routes/`

LLM dependencies follow a ports-and-adapters boundary:

- Provider-neutral models and ports: `core/llm/`
- Concrete OpenAI-compatible provider: `infra/providers/llm_provider.py`
- `agent/provider.py` is only a deprecated third-party compatibility import surface;
  project code must not import it.

The unified Task Runtime is implemented across these boundaries:

- Contracts: `core/workflow/`
- Persistence: `infra/persistence/workflow_store.py`
- Execution: `agent/workflows/runtime.py`
- Public task tools: `agent/tools/workflow.py`
- Isolated step executor: `agent/background/subagent_executor.py`
- Bootstrap wiring: `bootstrap/toolsets/workflow.py`

The Xiaoman personal foundation is implemented across these boundaries:

- Models and lifecycle service: `core/personal/`
- Personal fact persistence: `infra/persistence/personal_store.py`
- Memory governance and conflicts: `core/personal/governance.py` and
  `infra/persistence/memory_governance_store.py`
- Attention signals, policies, opportunities and action plans: `core/attention/`
  and `infra/persistence/attention_engine_store.py`
- Proactive message composition and delivery runtime: `proactive_v2/`
- Context, recommendation, reporting, and AI follow-up layer:
  `core/personal/rhythm/`
- Events and idempotent operations: `infra/persistence/personal_automation_store.py`
- Agent tools: `agent/tools/personal/`
- Standard routines: `agent/workflows/personal.py`
- Dashboard API: `bootstrap/dashboard_management/`
- Dashboard workbench features: `frontend/dashboard/src/features/`
- Dashboard shared UI, types, and hooks: `frontend/dashboard/src/shared/`

The boundaries above are guarded by `tests/test_architecture_boundaries.py`. New
features should extend these packages rather than rebuild service locators or add a
second background-task system.

The live Dashboard management surface is part of the gateway runtime and listens on
`127.0.0.1` by default. Non-loopback binding requires an explicit unsafe opt-in, and
browser WebSocket chat accepts only loopback same-origin requests. The standalone
`dashboard` command intentionally starts only diagnostic/memory dependencies; it
must not start channels, schedulers, proactive loops, IPC, or plugin jobs.

The remaining direction is:

1. Introduce Xiaoman integrations as plugins instead of expanding the agent core.
2. Split large modules only when they contain multiple change reasons; file size by
   itself is not a package boundary.
3. Add an architecture test whenever a new cross-layer port is introduced.
