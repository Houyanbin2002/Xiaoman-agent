import React, { useCallback, useEffect, useState } from "react";
import {
  Bot,
  Brain,
  CheckCircle2,
  ChevronRight,
  Circle,
  GitBranch,
  History,
  LoaderCircle,
  MessageCircle,
  Radio,
  Search,
  Sparkles,
  Trash2,
  Wrench,
  XCircle,
} from "lucide-react";
import { api, asPageResult } from "../../api";
import { relativeTime, shortTs } from "../../format";
import { MarkdownView } from "../../MarkdownView";
import { EmptyState, IconButton, PageIntro } from "../../shared/components/ui";
import { useAsyncData } from "../../shared/hooks/useAsyncData";
import type {
  MessageRow,
  PageResult,
  SessionRow,
  TraceDetail,
  TraceEventRow,
  TraceRow,
} from "../../types";

const MESSAGE_PAGE_SIZE = 100;

function chronological(items: MessageRow[]): MessageRow[] {
  return [...items].reverse();
}

function traceFlowLabel(flow: string): string {
  return { passive: "对话", workflow: "后台任务", proactive: "主动协助" }[flow] || "任务";
}

function traceStatusLabel(status: string): string {
  return {
    running: "进行中",
    completed: "已完成",
    failed: "未完成",
    interrupted: "已停止",
    cancelled: "已取消",
    blocked: "等待继续",
  }[status] || status;
}

function eventLabel(category: string): string {
  return {
    turn: "对话",
    memory: "相关记忆",
    model: "模型判断",
    tool: "使用工具",
    workflow: "后台步骤",
    attention: "主动计划",
    proactive: "主动感知",
  }[category] || "执行步骤";
}

function EventIcon({ event }: { event: TraceEventRow }): React.ReactElement {
  if (event.status === "failed" || event.status === "error") return <XCircle size={16} />;
  if (event.status === "running") return <LoaderCircle className="spin" size={16} />;
  const icons: Record<string, React.ReactElement> = {
    turn: <MessageCircle size={16} />,
    memory: <Brain size={16} />,
    model: <Bot size={16} />,
    tool: <Wrench size={16} />,
    workflow: <GitBranch size={16} />,
    attention: <Sparkles size={16} />,
    proactive: <Radio size={16} />,
  };
  return icons[event.category] || <CheckCircle2 size={16} />;
}

function durationLabel(milliseconds: number | null): string {
  if (milliseconds === null) return "";
  if (milliseconds < 1000) return `${milliseconds} ms`;
  if (milliseconds < 60_000) return `${(milliseconds / 1000).toFixed(1)} 秒`;
  return `${Math.floor(milliseconds / 60_000)} 分 ${Math.round((milliseconds % 60_000) / 1000)} 秒`;
}

function TraceTimeline({ detail }: { detail: TraceDetail }): React.ReactElement {
  return <div className="trace-replay">
    <div className="trace-replay-head">
      <div><span>{traceFlowLabel(detail.trace.flow)}</span><h4>{detail.trace.title}</h4></div>
      <span className={`trace-status ${detail.trace.status}`}>{traceStatusLabel(detail.trace.status)}</span>
    </div>
    <p className="trace-replay-meta">{shortTs(detail.trace.started_at)} · {detail.events.length} 个执行节点</p>
    <div className="trace-timeline">
      {detail.events.map((event) => <article className={`trace-event ${event.status}`} key={event.id}>
        <div className="trace-event-marker"><EventIcon event={event} /></div>
        <div className="trace-event-card">
          <div className="trace-event-top"><span>{eventLabel(event.category)}</span><small>{durationLabel(event.duration_ms)}</small></div>
          <strong>{event.summary}</strong>
          {Object.keys(event.payload).length ? <details><summary>查看本步记录</summary><pre>{JSON.stringify(event.payload, null, 2)}</pre></details> : null}
        </div>
      </article>)}
      {!detail.events.length ? <EmptyState icon={Circle} title="这条航迹还没有节点" text="任务开始执行后，关键步骤会依次出现在这里。" /> : null}
    </div>
  </div>;
}

export function SessionsView({ initialSessionKey }: { initialSessionKey?: string }): React.ReactElement {
  const [query, setQuery] = useState("");
  const resource = useAsyncData(
    async () => asPageResult(await api<PageResult<SessionRow>>(`/api/dashboard/sessions?q=${encodeURIComponent(query)}&page_size=200`)),
    [query],
  );
  const [active, setActive] = useState<SessionRow | null>(null);
  const [messages, setMessages] = useState<MessageRow[]>([]);
  const [messagePage, setMessagePage] = useState(1);
  const [messageTotal, setMessageTotal] = useState(0);
  const [loadingOlder, setLoadingOlder] = useState(false);
  const [detailMode, setDetailMode] = useState<"messages" | "trace">("messages");
  const [traces, setTraces] = useState<TraceRow[]>([]);
  const [traceDetail, setTraceDetail] = useState<TraceDetail | null>(null);
  const [traceLoading, setTraceLoading] = useState(false);

  const loadTrace = useCallback(async (trace: TraceRow): Promise<void> => {
    setTraceLoading(true);
    try {
      setTraceDetail(await api<TraceDetail>(`/api/dashboard/traces/${encodeURIComponent(trace.id)}`));
    } finally {
      setTraceLoading(false);
    }
  }, []);

  const open = useCallback(async (session: SessionRow): Promise<void> => {
    setActive(session);
    setMessagePage(1);
    setTraceDetail(null);
    const [messagePayload, tracePayload] = await Promise.all([
      api<PageResult<MessageRow>>(`/api/dashboard/sessions/${encodeURIComponent(session.key)}/messages?page=1&page_size=${MESSAGE_PAGE_SIZE}&sort_by=seq&sort_order=desc`),
      api<PageResult<TraceRow>>(`/api/dashboard/traces?session_key=${encodeURIComponent(session.key)}&limit=100`)
        .catch(() => ({ items: [], total: 0 })),
    ]);
    setMessages(chronological(messagePayload.items));
    setMessageTotal(messagePayload.total);
    setTraces(tracePayload.items);
    if (tracePayload.items[0]) await loadTrace(tracePayload.items[0]);
  }, [loadTrace]);

  const loadOlder = async (): Promise<void> => {
    if (!active || loadingOlder || messages.length >= messageTotal) return;
    setLoadingOlder(true);
    try {
      const nextPage = messagePage + 1;
      const payload = await api<PageResult<MessageRow>>(`/api/dashboard/sessions/${encodeURIComponent(active.key)}/messages?page=${nextPage}&page_size=${MESSAGE_PAGE_SIZE}&sort_by=seq&sort_order=desc`);
      setMessages((current) => {
        const existing = new Set(current.map((message) => message.id));
        return [...chronological(payload.items).filter((message) => !existing.has(message.id)), ...current];
      });
      setMessagePage(nextPage);
      setMessageTotal(payload.total);
    } finally {
      setLoadingOlder(false);
    }
  };

  useEffect(() => {
    if (!initialSessionKey || active?.key === initialSessionKey || !resource.data) return;
    const session = resource.data.items.find((item) => item.key === initialSessionKey);
    if (session) void open(session);
  }, [active?.key, initialSessionKey, open, resource.data]);

  const remove = async (session: SessionRow): Promise<void> => {
    await api(`/api/dashboard/sessions/${encodeURIComponent(session.key)}`, { method: "DELETE" });
    if (active?.key === session.key) {
      setActive(null);
      setMessages([]);
      setTraces([]);
      setTraceDetail(null);
      setMessageTotal(0);
    }
    resource.reload();
  };

  return <>
    <PageIntro
      title="会话记录"
      description="回看连续对话，也能沿着任务航迹了解每一步是怎样完成的。"
      actions={<div className="search-box"><Search size={16} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索会话" /></div>}
    />
    <div className="split-view sessions-workbench">
      <div className="session-list">
        {resource.data?.items.map((session) => <button className={`session-item ${active?.key === session.key ? "active" : ""}`} key={session.key} onClick={() => void open(session)}><span className="session-icon"><MessageCircle size={17} /></span><span><strong>{session.title}</strong><small>{session.message_count} 条消息 · {relativeTime(session.updated_at)}</small></span><ChevronRight size={16} /></button>)}
      </div>
      <div className="session-detail">
        {active ? <>
          <div className="session-detail-head">
            <div><h3>{active.title}</h3><p>{active.key} · {active.message_count} 条消息</p></div>
            <div className="session-detail-actions">
              <div className="detail-mode-switch" aria-label="详情类型">
                <button className={detailMode === "messages" ? "active" : ""} onClick={() => setDetailMode("messages")}><MessageCircle size={14} />对话</button>
                <button className={detailMode === "trace" ? "active" : ""} onClick={() => setDetailMode("trace")}><GitBranch size={14} />任务航迹 <span>{traces.length}</span></button>
              </div>
              <IconButton icon={Trash2} label="删除会话" danger onClick={() => void remove(active)} />
            </div>
          </div>
          {detailMode === "messages" ? <div className="history-list">
            {messages.length < messageTotal ? <button type="button" className="load-older-messages" disabled={loadingOlder} onClick={() => void loadOlder()}>{loadingOlder ? "正在加载…" : `加载更早消息（还有 ${messageTotal - messages.length} 条）`}</button> : null}
            {messages.map((message) => <div className={`history-message ${message.role}`} key={message.id}><span>{message.role}</span><MarkdownView content={message.content} /></div>)}
          </div> : <div className="trace-workbench">
            <aside className="trace-list" aria-label="任务航迹列表">
              {traces.map((trace) => <button className={traceDetail?.trace.id === trace.id ? "active" : ""} key={trace.id} onClick={() => void loadTrace(trace)}>
                <span><strong>{trace.title}</strong><small>{traceFlowLabel(trace.flow)} · {relativeTime(trace.started_at)}</small></span>
                {trace.status === "running" ? <LoaderCircle className="spin" size={14} /> : <ChevronRight size={14} />}
              </button>)}
            </aside>
            <section className="trace-stage">
              {traceLoading && !traceDetail ? <EmptyState icon={LoaderCircle} title="正在整理任务航迹" text="稍等一下，很快就好。" /> : traceDetail ? <TraceTimeline detail={traceDetail} /> : <EmptyState icon={GitBranch} title="还没有任务航迹" text="新的对话、后台任务和主动协助会自动记录在这里。" />}
            </section>
          </div>}
        </> : <EmptyState icon={History} title="选择一个会话" text="会话详情、消息历史和任务航迹会显示在这里。" />}
      </div>
    </div>
  </>;
}
