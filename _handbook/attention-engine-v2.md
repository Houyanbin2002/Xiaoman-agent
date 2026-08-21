# Attention And Action Engine v2

The attention engine is a generic opportunity-to-action planner. It does not
encode commute, fitness, lunch, or other life scenarios in the core.

## Two lanes

The commitment lane remains owned by `SchedulerService`. Exact reminders bypass
attention scoring and are delivered at their trigger time unless cancelled or
the transport fails.

The opportunity lane is owned by `core/attention/`:

1. Signal providers emit namespaced, evidence-backed `AttentionSignal` values.
2. Conversation consolidation and providers persist traceable
   `AttentionObservation` evidence.
3. One learning service aggregates evidence into confidence-scored patterns
   and declarative policies. Explicit user instructions activate immediately;
   inference requires repeated evidence and later decays.
4. Patterns and live signals materialize bounded `OpportunityWindow` instances.
5. Registered action manifests are matched without importing provider code.
6. Kernel invariants and versioned declarative policies filter the candidates.
7. `UtilityScorer` ranks relevance against interruption, repetition,
   uncertainty, and risk.
8. The result is a durable, idempotent `ActionPlan`, never an implicit side
   effect.
9. Approved plans execute through registered handlers and record feedback.

## Package ownership

- `core/attention/signals/`: open signal schema and provider registry.
- `core/attention/patterns/`: recurring behavior patterns, evidence and decay.
- `core/attention/learning/`: observation validation, deduplication, promotion
  and decay for patterns and policies.
- `core/attention/opportunities/`: current recurring and event windows.
- `core/attention/policies/`: safe declarative policies and kernel boundaries.
- `core/attention/actions/`: capability manifests, plans and execution states.
- `core/attention/planning/`: generic signal/window/capability matching.
- `core/attention/scoring/`: explainable utility score.
- `core/attention/feedback/`: dimension-specific learning signals.
- `infra/persistence/attention_engine_store.py`: SQLite repository in
  `personal.db`.
- `bootstrap/attention.py`: the only composition root for this subsystem.

## Extensibility contract

A new MCP or Skill can contribute:

- a signal provider;
- a read-only or side-effecting action manifest;
- an optional action handler;
- an optional dynamic policy.

Signal kinds are provider-owned namespaced strings. The core must never branch
on a domain signal name. A provider can also attach a generic
`pattern_observation`, `policy_observation`, or event `opportunity` payload.
Conversation consolidation uses the same observation protocol, so all learned
rules share validation, provenance, promotion, feedback and expiry semantics.

Existing proactive MCP alert payloads are normalized by
`McpAlertSignalAdapter`. Providers may add `signal_kind`, `domain`, feature
scores, capability hints, and opportunity metadata; missing optional fields
receive conservative protocol defaults. Successful delivery ACKs the original
MCP event and completes the corresponding action plan, so external data cannot
bypass v2 planning.

## Runtime status

The replacement is complete. The fixed personal-assistance rule engine,
assistance insight store, and Attention Draft store have been removed. Personal
records contribute signals through `PersonalRecordSignalProvider`; MCP alerts
use `McpAlertSignalAdapter`. Both paths enter the same engine and produce one
durable `ActionPlan`. The Proactive Loop is retained only as the outbound message
composition and delivery adapter for `message.notify`.
