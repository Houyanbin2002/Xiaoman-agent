import type { LucideIcon } from "lucide-react";

export type ViewId =
  | "chat"
  | "today"
  | "overview"
  | "memory"
  | "workflows"
  | "channels"
  | "models"
  | "skills"
  | "mcp"
  | "tools"
  | "schedules"
  | "proactive"
  | "sessions"
  | "settings";

export interface Overview {
  status: string;
  assistant: string;
  model: string;
  provider: string;
  memory_enabled: boolean;
  memory_engine: string;
  memory_status: MemoryHealthStatus;
  counts: Record<string, number>;
  channels: ChannelRow[];
}

export interface ChannelRow {
  id: string;
  label: string;
  configured: boolean;
  connected: boolean;
  detail: string;
  allow_from: string[];
  kind: "local" | "gateway";
  fields?: Record<string, { value: string; configured: boolean }>;
  docs_url?: string;
  docs_label?: string;
}

export interface ModelRow {
  slot: string;
  kind?: "chat" | "embedding";
  label: string;
  model: string;
  provider: string;
  base_url: string;
  api_key_configured: boolean;
  engine?: string;
  output_dimensionality?: number;
  usage: string;
  hot_reload: boolean;
}

export interface ModelCatalogItem {
  id: string;
  owned_by: string;
}

export interface ModelCatalogResponse {
  items: ModelCatalogItem[];
  total: number;
  base_url: string;
}

export interface ModelUpdateResponse {
  saved: boolean;
  hot_reloaded: boolean;
  restart_required: boolean;
  slot: string;
  model: string;
}

export interface McpRow {
  name: string;
  connected: boolean;
  status: "disconnected" | "connecting" | "authorization_required" | "authorizing" | "connected" | "error";
  error: string;
  authorization_required: boolean;
  system_managed: boolean;
  tool_names: string[];
  transport: "stdio" | "streamable_http" | "sse";
  url: string;
  auth_type: "none" | "oauth" | "bearer" | "headers";
  command: string[];
  cwd: string;
  env_keys: string[];
  header_names: string[];
}

export interface McpCatalogRow {
  id: string;
  name: string;
  description: string;
  provider: string;
  transport: "stdio" | "streamable_http" | "sse";
  requires_oauth: boolean;
  installed: boolean;
  configuration: {
    name: string;
    transport: "stdio" | "streamable_http" | "sse";
    command?: string[];
    url?: string;
    auth_type?: "none" | "oauth" | "bearer" | "headers";
    scopes?: string;
    requires_oauth_client?: boolean;
    requires_vault_path?: boolean;
    docs_url?: string;
  } | null;
}

export interface MarketplaceField {
  name: string;
  label: string;
  required: boolean;
  secret: boolean;
  placeholder: string;
}

export interface MarketplaceItem {
  id: string;
  kind: "skill" | "mcp";
  name: string;
  description: string;
  provider: string;
  source_url: string;
  version: string;
  icon_url: string;
  install_count: number | null;
  verified: boolean;
  deprecated: boolean;
  installed: boolean;
  install_mode: "direct" | "configure" | "oauth" | "unsupported";
  configuration_fields: MarketplaceField[];
  unsupported_reason: string;
}

export interface MarketplaceResponse {
  items: MarketplaceItem[];
  kind: "skill" | "mcp";
  query: string;
}

export interface MarketplaceInstallResponse {
  status: "installed" | "already_installed" | "authorization_required" | "unsupported";
  item_id: string;
  kind: "skill" | "mcp";
  resource_name: string;
  message: string;
}

export interface SkillRow {
  name: string;
  display_name: string;
  source: string;
  source_id: string;
  origin: "builtin" | "workspace" | "standalone" | "system";
  source_label: string;
  provider_id: string;
  provider_name: string;
  can_uninstall: boolean;
  description: string;
  when_to_use: string;
  always: boolean;
  available: boolean;
  missing: string;
}

export interface ToolRow {
  name: string;
  description: string;
  risk: string;
  always_on: boolean;
  source_type: string;
  source_name: string;
}

export interface WorkflowRow {
  id: string;
  short_id: string;
  name: string;
  goal: string;
  status: string;
  step_count: number;
  completed_steps: number;
  waiting_steps: string[];
  failed_steps: string[];
  waiting_actions: Array<{ id: string; title: string; description: string; kind: "approval" | "wait_user" | string }>;
  failed_actions: Array<{ id: string; title: string; description: string; error: string }>;
  updated_at: string;
  error: string;
}

export interface ScheduleRow {
  id: string;
  name?: string | null;
  trigger: string;
  tier: string;
  fire_at: string;
  channel: string;
  chat_id: string;
  message?: string | null;
  prompt?: string | null;
  run_count: number;
  enabled?: boolean;
  interval_seconds?: number | null;
  cron_expr?: string | null;
  last_attempt_at?: string | null;
  last_status?: "pending" | "sent" | "failed" | string;
  last_error?: string | null;
}

export interface PersonalOverview {
  counts: Record<string, number>;
  total_active: number;
  profile_configured: boolean;
  routines_available: boolean;
}

export interface PersonalRecordRow {
  id: string;
  entity_type: string;
  record_key: string;
  title: string;
  summary: string;
  data: Record<string, unknown>;
  source: string;
  source_ref: string;
  confidence: number;
  sensitivity: string;
  data_category: string;
  access_policy: string;
  status: string;
  expires_at: string | null;
  last_confirmed_at: string | null;
  user_locked: boolean;
  allow_auto_update: boolean;
  revision: number;
  created_at: string;
  updated_at: string;
}

export interface PersonalTodayResponse {
  date: string;
  timezone: string;
  records: PersonalRecordRow[];
  counts: Record<string, number>;
  overdue_count: number;
  sources: Record<string, number>;
}

export interface ExternalSourceRow {
  id: string;
  provider: string;
  server_name: string;
  name: string;
  resource_url: string;
  entity_type: string;
  mapping: Record<string, unknown>;
  poll_interval_minutes: number;
  enabled: boolean;
  last_synced_at: string | null;
  last_error: string;
  last_item_count: number;
  created_at: string;
  updated_at: string;
}

export type MemoryHealthStatus = "healthy" | "degraded" | "unhealthy" | "unchecked" | "pending_restart" | "disabled";

export interface StoredMemoryRow {
  id: string;
  memory_type: string;
  summary: string;
  content?: string;
  source_ref: string;
  status: string;
  created_at: string;
  updated_at: string;
  happened_at: string;
  extra_json: Record<string, unknown>;
  has_embedding: boolean;
  embedding_dim: number;
}

export interface ExecutionMemoryRow {
  id: string;
  summary: string;
  source_ref: string;
  status: string;
  created_at: string;
  updated_at: string;
  extra_json: Record<string, unknown>;
  execution: {
    item_id: string;
    kind: string;
    scope: {
      kind: string;
      workspace_id: string;
      project_id: string;
      tool_name: string;
      plugin_name: string;
      platform: string;
      version_key: string;
      version_value: string;
    };
    verification_status: string;
    success_count: number;
    failure_count: number;
    last_verified_at: string | null;
    expires_at: string | null;
    evidence_refs: string[];
  };
}

export interface MemoryConflictRow {
  id: string;
  record_key: string;
  existing_record_id: string | null;
  candidate: Record<string, unknown>;
  reason: string;
  status: string;
  created_at: string;
  existing: PersonalRecordRow | null;
}

export interface MemoryKnowledgeNode {
  id: string;
  label: string;
  kind: string;
  memory_ids: string[];
  confidence: number;
}

export interface MemoryKnowledgeEdge {
  id: string;
  source: string;
  target: string;
  label: string;
  kind: string;
  memory_ids: string[];
}

export interface MemoryKnowledgeGraph {
  center_id: string;
  nodes: MemoryKnowledgeNode[];
  edges: MemoryKnowledgeEdge[];
}

export interface AttentionEngineOverview {
  active_signals: number;
  active_events: number;
  pending_wakes: number;
  next_wake_at: string | null;
  source_sync_pending: number;
  patterns: number;
  active_patterns: number;
  policies: number;
  enabled_policies: number;
  plans: number;
  pending_approval: number;
  capabilities: number;
  runtime_enabled: boolean;
  target_configured: boolean;
  target_channel: string;
  target_chat_id: string;
  available_targets: { channel: string; chat_id: string }[];
  provider_failures: { provider_id: string; error_type: string; message: string }[];
}

export interface AttentionEntityRow {
  id: string;
  source_id: string;
  external_id: string;
  kind: string;
  title: string;
  state: "open" | "completed" | "cancelled";
  source_version: string;
  payload_ref: string;
  updated_at: string;
  start_at: string;
  due_at: string;
  local_override: Record<string, unknown>;
}

export interface AttentionEventRow {
  id: string;
  entity_id: string;
  source_id: string;
  kind: string;
  occurred_at: string;
  due_at: string;
  active_from: string;
  expires_at: string;
  urgency: number;
  confidence: number;
  delivery_semantics: "exact" | "before_deadline" | "opportunistic" | "silent";
  dedupe_key: string;
  source_version: string;
  payload_ref: string;
  status: "active" | "completed" | "cancelled" | "expired";
  entity: AttentionEntityRow | null;
}

export interface AttentionWakeRow {
  id: string;
  event_id: string;
  wake_at: string;
  reason: string;
  attempt: number;
  max_attempts: number;
  status: "pending" | "processing" | "completed" | "cancelled" | "dead";
  last_decision: string;
  created_at: string;
  updated_at: string;
  event: Omit<AttentionEventRow, "entity"> | null;
  entity: AttentionEntityRow | null;
}

export interface AttentionSignalRow {
  id: string;
  kind: string;
  domain: string;
  occurred_at: string;
  expires_at: string | null;
  valence: string;
  severity: number;
  urgency: number;
  actionability: number;
  confidence: number;
  summary: string;
  source: { type: string; name: string; reference: string };
  evidence: Record<string, unknown>[];
  suggested_capabilities: string[];
  metadata: Record<string, unknown>;
}

export interface AttentionPatternRow {
  id: string;
  kind: string;
  scene: string;
  recurrence: {
    timezone: string;
    days: string[];
    start: string;
    end: string;
  };
  available_minutes: number;
  confidence: number;
  observation_count: number;
  source: "user" | "learned" | "imported";
  status: "proposed" | "active" | "suspended" | "rejected" | "expired";
  last_observed_at: string | null;
  expires_at: string | null;
  user_locked: boolean;
  metadata: Record<string, unknown>;
}

export interface AttentionPolicyRow {
  id: string;
  scope: Record<string, unknown>;
  conditions: Record<string, unknown>;
  effect: string;
  priority: number;
  score_adjustment: number;
  version: number;
  enabled: boolean;
  status: "proposed" | "active" | "suspended" | "rejected" | "expired";
  confidence: number;
  observation_count: number;
  last_observed_at: string | null;
  effective_from: string | null;
  expires_at: string | null;
  source: string;
  user_locked: boolean;
  metadata: Record<string, unknown>;
}

export interface AttentionObservationRow {
  id: string;
  kind: "opportunity" | "policy";
  rule_key: string;
  statement: string;
  confidence: number;
  explicit: boolean;
  source_type: string;
  source_ref: string;
  observed_at: string;
  payload: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

export interface AttentionActionPlanRow {
  id: string;
  signal_ids: string[];
  opportunity_id: string;
  capability_id: string;
  action_type: string;
  decision_reason: string;
  score: number;
  score_components: Record<string, number>;
  risk: string;
  approval: string;
  status: string;
  policy_ids: string[];
  created_at: string;
  expires_at: string | null;
  updated_at: string;
  result: Record<string, unknown> | null;
  error: string;
}

export type RhythmDomain = "commitment" | "relationship" | "important_date" | "financial_obligation" | "trip" | "goal" | "proactive_intent";

export interface RhythmContext {
  observed_at: string;
  timezone: string;
  scene: string;
  scene_ends_at: string | null;
  focus_active: boolean;
  focus_label: string;
  focus_ends_at: string | null;
  do_not_disturb: boolean;
  allow_high_priority: boolean;
  energy: string;
}

export interface RhythmOverview {
  context: RhythmContext;
  counts: Record<string, number>;
}

export interface RhythmRecommendation {
  candidate_id: string;
  source_type: string;
  title: string;
  next_action: string;
  estimated_minutes: number;
  score: number;
  reason: string;
  due_at: string | null;
  due_text: string;
  context: string;
  energy: string;
}

export interface RhythmRecommendationResult {
  available_minutes: number;
  context: RhythmContext;
  recommendations: RhythmRecommendation[];
}

export interface RhythmReport {
  period: string;
  period_start: string;
  period_end: string;
  metrics: Record<string, number | null>;
  deviations: Array<{ record_id: string; title: string; expected_progress: number; actual_progress: number; gap: number; due_at: string }>;
  recommendations: string[];
  record_id: string | null;
}

export interface RhythmFormState {
  type: RhythmDomain;
  title: string;
  summary: string;
  due_at: string;
  estimated_minutes: string;
  priority: string;
  energy: string;
  context: string;
  next_action: string;
  person_name: string;
  relationship: string;
  last_contact_at: string;
  contact_interval_days: string;
  date: string;
  repeat_yearly: boolean;
  preparation_days: string;
  obligation_type: string;
  amount: string;
  currency: string;
  recurrence: string;
  auto_renew: boolean;
  reminder_days: string;
  destination: string;
  depart_at: string;
  return_at: string;
  checklist: string;
  target: string;
  current: string;
  unit: string;
  start_at: string;
  direction: string;
  message: string;
  reason: string;
  trigger_type: string;
  next_trigger_at: string;
  interval_minutes: string;
  inactivity_days: string;
  target_entity_type: string;
  target_record_key: string;
  enabled: boolean;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
  thinking?: string;
  pending?: boolean;
  state?: "stopped" | "error";
  attachments?: Array<{ name: string; size?: number; mime_type?: string }>;
}

export interface NavItem {
  id: ViewId;
  label: string;
  icon: LucideIcon;
}
