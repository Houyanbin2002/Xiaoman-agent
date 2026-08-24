export interface PageResult<T> {
  items: T[];
  total: number;
  page?: number;
  page_size?: number;
}

export interface SessionRow {
  key: string;
  title: string;
  created_at: string;
  updated_at: string;
  last_consolidated: number;
  metadata: Record<string, unknown>;
  last_user_at: string | null;
  last_proactive_at: string | null;
  message_count: number;
}

export interface MessageRow {
  id: string;
  session_key: string;
  seq: number;
  role: string;
  content: string;
  tool_chain: unknown;
  extra: Record<string, unknown>;
  timestamp: string;
}

export interface TraceRow {
  id: string;
  flow: "passive" | "workflow" | "proactive" | string;
  session_key: string;
  title: string;
  status: string;
  parent_trace_id: string;
  started_at: string;
  updated_at: string;
  finished_at: string | null;
  metadata: Record<string, unknown>;
  event_count: number;
}

export interface TraceEventRow {
  id: number;
  trace_id: string;
  span_id: string;
  parent_span_id: string;
  category: string;
  name: string;
  status: string;
  started_at: string;
  finished_at: string | null;
  duration_ms: number | null;
  summary: string;
  payload: Record<string, unknown>;
}

export interface TraceDetail {
  trace: TraceRow;
  events: TraceEventRow[];
}

export interface ProactiveOverview {
  counts: Record<string, number>;
  result_counts: Record<string, number>;
  flow_counts: Record<string, number>;
  last_tick_at: string | null;
  last_send_at: string | null;
  last_skip_reason: string | null;
  recent_tick: ProactiveTick | null;
}

export interface ProactiveTick {
  tick_id: string;
  session_key: string;
  started_at: string;
  finished_at?: string | null;
  gate_exit?: string | null;
  terminal_action?: string | null;
  skip_reason?: string | null;
  steps_taken?: number;
  drift_entered?: boolean | number;
  final_message?: string | null;
  alert_count?: number;
  content_count?: number;
  context_count?: number;
  interesting_ids?: string[];
  discarded_ids?: string[];
  cited_ids?: string[];
}

export interface ProactiveStep {
  step_index: number;
  phase: string;
  tool_name: string;
  tool_call_id: string;
  tool_args: unknown;
  tool_result_text: string;
  terminal_action_after: string;
  skip_reason_after: string;
  final_message_after: string;
  interesting_ids_after: string[];
  discarded_ids_after: string[];
  cited_ids_after: string[];
}
