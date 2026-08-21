import React, { useMemo, useState } from "react";
import {
  BellRing,
  Bot,
  Brain,
  Check,
  ChevronRight,
  Clock3,
  MessageSquareText,
  Plus,
  RefreshCw,
  Search,
  ShieldCheck,
  Sunrise,
  Trash2,
  Wrench,
  X,
} from "lucide-react";
import { api } from "../../api";
import { relativeTime } from "../../format";
import { Badge, EmptyState, ErrorBanner, IconButton, LoadingState, Modal, PageIntro } from "../../shared/components/ui";
import { useAsyncData } from "../../shared/hooks/useAsyncData";
import type { ExecutionMemoryRow, MemoryConflictRow, MemoryKnowledgeGraph, PersonalRecordRow, StoredMemoryRow } from "../../shared/types";
import type { PageResult } from "../../types";
import { MemoryKnowledgeGraph as MemoryKnowledgeGraphView } from "./MemoryKnowledgeGraph";

interface MemoryEngineInfo {
  name: string;
  profile: string;
  capabilities: string[];
}

interface MemoryOptimizerStatus {
  enabled: boolean;
  running: boolean;
  last_status: string;
  last_error: string | null;
}

type MemoryTab = "about_me" | "graph" | "agent" | "recall" | "pending";

const TYPE_LABELS: Record<string, string> = {
  requested: "明确记住",
  fact: "事实",
  preference: "偏好",
  temporary_state: "临时状态",
  historical_event: "重要经历",
  episode: "重要经历",
  relationship: "关系",
  procedure: "助手操作上下文",
};

const POLICY_LABELS: Record<string, string> = {
  standard: "可直接使用",
  confirm_write: "写入需确认",
  confirm_read: "读取需确认",
  owner_only: "仅你可见",
};

const SENSITIVITY_LABELS: Record<string, string> = {
  personal: "个人信息",
  sensitive: "敏感信息",
  restricted: "严格限制",
};

const INITIAL_FORM = { kind: "requested", summary: "", content: "" };

export function MemoryView(): React.ReactElement {
  const [tab, setTab] = useState<MemoryTab>("about_me");
  const [query, setQuery] = useState("");
  const [actionError, setActionError] = useState("");
  const [busy, setBusy] = useState("");
  const [adding, setAdding] = useState(false);
  const [form, setForm] = useState(INITIAL_FORM);

  const engine = useAsyncData(() => api<MemoryEngineInfo>("/api/dashboard/memory/engine-info"), []);
  const optimizer = useAsyncData(() => api<MemoryOptimizerStatus>("/api/dashboard/memory/optimizer"), []);
  const governed = useAsyncData(
    () => api<PersonalRecordRow[]>("/api/dashboard/control/memory-governance/memories"),
    [],
  );
  const graph = useAsyncData(
    () => api<MemoryKnowledgeGraph>("/api/dashboard/control/memory-governance/graph"),
    [],
  );
  const conflicts = useAsyncData(
    () => api<MemoryConflictRow[]>("/api/dashboard/control/memory-governance/conflicts?pending_only=true"),
    [],
  );
  const memories = useAsyncData(
    () => api<PageResult<StoredMemoryRow>>(`/api/dashboard/memories?q=${encodeURIComponent(query)}&page=1&page_size=100&sort_by=updated_at&sort_order=desc`),
    [query],
  );
  const execution = useAsyncData(
    () => api<{ items: ExecutionMemoryRow[]; total: number }>("/api/dashboard/memory/execution"),
    [],
  );

  const normalizedQuery = query.trim().toLocaleLowerCase();
  const visibleGoverned = useMemo(
    () => filterGoverned(governed.data ?? [], normalizedQuery),
    [governed.data, normalizedQuery],
  );
  const visibleExecution = useMemo(
    () => filterExecution(execution.data?.items ?? [], normalizedQuery),
    [execution.data, normalizedQuery],
  );

  const reload = (): void => {
    engine.reload(); optimizer.reload(); governed.reload(); graph.reload(); conflicts.reload(); memories.reload(); execution.reload();
  };

  const runBusy = async (key: string, action: () => Promise<void>): Promise<void> => {
    setBusy(key); setActionError("");
    try { await action(); }
    catch (reason) { setActionError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(""); }
  };

  const forgetGoverned = async (record: PersonalRecordRow): Promise<void> => {
    if (!window.confirm(`确认让小满忘记“${record.title}”？`)) return;
    await runBusy(record.id, async () => {
      await api(`/api/dashboard/control/memory-governance/memories/${record.id}`, { method: "DELETE" });
      governed.reload(); graph.reload();
    });
  };

  const removeConversationIndex = async (memory: StoredMemoryRow): Promise<void> => {
    if (!window.confirm("确认删除这条回忆索引？聊天记录本身不会被删除。")) return;
    await runBusy(memory.id, async () => {
      await api("/api/dashboard/memories/batch-delete", { method: "POST", body: JSON.stringify({ ids: [memory.id] }) });
      memories.reload();
    });
  };

  const removeExecutionMemory = async (memory: ExecutionMemoryRow): Promise<void> => {
    if (!window.confirm("确认删除这条小满执行经验？之后不会再用它指导工具执行。")) return;
    await runBusy(memory.id, async () => {
      await api("/api/dashboard/memories/batch-delete", { method: "POST", body: JSON.stringify({ ids: [memory.id] }) });
      execution.reload();
    });
  };

  const resolveConflict = async (conflict: MemoryConflictRow, action: "keep_existing" | "accept_candidate"): Promise<void> => {
    await runBusy(conflict.id, async () => {
      await api(`/api/dashboard/control/memory-governance/conflicts/${conflict.id}/resolve`, {
        method: "POST",
        body: JSON.stringify({ action, note: action === "keep_existing" ? "从记忆管理保留原内容" : "从记忆管理接受新内容" }),
      });
      conflicts.reload(); governed.reload(); graph.reload();
    });
  };

  const saveMemory = async (): Promise<void> => {
    if (!form.summary.trim() || !form.content.trim()) return;
    await runBusy("create", async () => {
      await api("/api/dashboard/control/memory-governance/memories", {
        method: "POST",
        body: JSON.stringify({
          kind: form.kind,
          summary: form.summary.trim(),
          content: form.content.trim(),
          subject: "我",
          predicate: relationLabelForKind(form.kind),
          value: form.summary.trim(),
          data_category: "general",
          confidence: 1,
        }),
      });
      setForm(INITIAL_FORM); setAdding(false); governed.reload(); graph.reload(); conflicts.reload();
    });
  };

  const pendingCount = conflicts.data?.length ?? 0;
  const engineLabel = engine.data?.name === "xiaoman" ? "小满统一记忆" : engine.data?.name === "akasha" ? "小满记忆" : engine.data?.name || "记忆引擎";
  const error = engine.error || optimizer.error || governed.error || graph.error || conflicts.error || memories.error || execution.error || actionError;

  return <>
    <PageIntro
      title="关于我"
      description="查看小满长期了解的事实、偏好、关系和重要经历，也可以随时纠正或让他忘记。"
      actions={<><IconButton icon={RefreshCw} label="刷新记忆" onClick={reload} /><button className="primary-button" onClick={() => setAdding(true)}><Plus size={16} />记住一件事</button></>}
    />
    <ErrorBanner message={error} />

    <section className="memory-map" aria-label="记忆工作方式">
      <div className="memory-map-intro">
        <span className="memory-map-orbit"><Brain size={23} /><i /></span>
        <div><small>小满怎样记住你</small><strong>{engineLabel}</strong><p>个人理解、执行经验和聊天回忆各自管理，需要时再一起帮助小满理解当下。</p></div>
      </div>
      <div className="memory-map-layers unified">
        <button onClick={() => setTab("about_me")} className={tab === "about_me" ? "active" : ""}><ShieldCheck size={17} /><span><small>稳定理解</small><strong>关于我 · {governed.data?.length ?? 0} 条</strong></span></button>
        <ChevronRight size={15} />
        <button onClick={() => setTab("agent")} className={tab === "agent" ? "active" : ""}><Wrench size={17} /><span><small>实际结果验证</small><strong>小满经验 · {execution.data?.total ?? 0} 条</strong></span></button>
        <ChevronRight size={15} />
        <button onClick={() => setTab("recall")} className={tab === "recall" ? "active" : ""}><MessageSquareText size={17} /><span><small>可回到原对话</small><strong>聊天回忆 · {memories.data?.total ?? 0} 段</strong></span></button>
      </div>
      <div className="memory-consumers"><small>同一份长期记忆用于</small><span><Bot size={14} />聊天</span><span><Sunrise size={14} />我的一天</span><span><BellRing size={14} />主动协助</span></div>
    </section>

    <div className="memory-workbench-tabs" role="tablist" aria-label="记忆分类">
      <MemoryTabButton active={tab === "about_me"} label="关于我" count={governed.data?.length ?? 0} onClick={() => setTab("about_me")} />
      <MemoryTabButton active={tab === "graph"} label="记忆图谱" count={Math.max(0, (graph.data?.nodes.length ?? 1) - 1)} onClick={() => setTab("graph")} />
      <MemoryTabButton active={tab === "agent"} label="小满经验" count={execution.data?.total ?? 0} onClick={() => setTab("agent")} />
      <MemoryTabButton active={tab === "recall"} label="聊天回忆" count={memories.data?.total ?? 0} onClick={() => setTab("recall")} />
      <MemoryTabButton active={tab === "pending"} label="待确认" count={pendingCount} onClick={() => setTab("pending")} tone={pendingCount ? "amber" : "gray"} />
    </div>

    <div className="memory-library-toolbar">
      <div className="search-box"><Search size={16} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索当前记忆" aria-label="搜索当前记忆" />{query ? <button aria-label="清空搜索" onClick={() => setQuery("")}><X size={14} /></button> : null}</div>
      <span>{optimizer.data?.running ? "正在整理记忆生命周期" : optimizer.data?.enabled ? "记忆会自动检查过期并更新快照" : "自动治理当前未启用"}</span>
    </div>

    {tab === "about_me" ? <LongTermPanel loading={governed.loading} records={visibleGoverned} hasQuery={Boolean(query)} busy={busy} onForget={forgetGoverned} onAdd={() => setAdding(true)} /> : null}
    {tab === "graph" ? <MemoryKnowledgeGraphView graph={graph.data} records={governed.data ?? []} loading={graph.loading} query={query} /> : null}
    {tab === "agent" ? <ExecutionPanel loading={execution.loading} memories={visibleExecution} hasQuery={Boolean(query)} busy={busy} onRemove={removeExecutionMemory} /> : null}
    {tab === "recall" ? <RecallPanel loading={memories.loading} memories={memories.data?.items ?? []} hasQuery={Boolean(query)} busy={busy} onRemove={removeConversationIndex} /> : null}
    {tab === "pending" ? <PendingPanel loading={conflicts.loading} conflicts={conflicts.data ?? []} busy={busy} onResolve={resolveConflict} /> : null}

    {adding ? <Modal title="让小满记住一件事" description="这会进入统一长期记忆库，并在聊天、我的一天和主动协助中遵守同一套使用边界。" onClose={() => setAdding(false)}><div className="form-stack"><label>类型<select value={form.kind} onChange={(event) => setForm({ ...form, kind: event.target.value })}><option value="requested">明确要求记住</option><option value="fact">事实</option><option value="preference">偏好</option><option value="relationship">关系</option><option value="historical_event">重要经历</option></select></label><label>简短标题<input value={form.summary} onChange={(event) => setForm({ ...form, summary: event.target.value })} placeholder="例如：重要任务尽量安排在上午" /></label><label>具体内容<textarea rows={5} value={form.content} onChange={(event) => setForm({ ...form, content: event.target.value })} placeholder="把希望小满以后持续考虑的信息写清楚" /></label><div className="dialog-actions"><button className="secondary-button" onClick={() => setAdding(false)}>取消</button><button className="primary-button" disabled={!form.summary.trim() || !form.content.trim() || busy === "create"} onClick={() => void saveMemory()}><Check size={16} />{busy === "create" ? "保存中" : "保存记忆"}</button></div></div></Modal> : null}
  </>;
}

function MemoryTabButton(props: { active: boolean; label: string; count: number; tone?: "amber" | "gray"; onClick: () => void }): React.ReactElement {
  return <button type="button" role="tab" aria-selected={props.active} className={props.active ? "active" : ""} onClick={props.onClick}><span>{props.label}</span><Badge tone={props.tone ?? (props.active ? "blue" : "gray")}>{props.count}</Badge></button>;
}

function LongTermPanel(props: { loading: boolean; records: PersonalRecordRow[]; hasQuery: boolean; busy: string; onForget: (record: PersonalRecordRow) => Promise<void>; onAdd: () => void }): React.ReactElement {
  if (props.loading) return <LoadingState />;
  return <section className="memory-panel"><div className="memory-panel-heading"><div><h2>关于我</h2><p>这里只保存理解你所需的稳定事实、偏好、关系和明确要求；每条内容都有来源、权限和版本记录。</p></div><button className="secondary-button" onClick={props.onAdd}><Plus size={15} />新增</button></div>{props.records.length ? <div className="stored-memory-list">{props.records.map((record) => { const kind = String(record.data.kind ?? "fact"); const content = String(record.data.content ?? record.summary ?? ""); return <article className="stored-memory-row governed" key={record.id}><span className="stored-memory-icon remembered"><ShieldCheck size={17} /></span><div className="stored-memory-main"><div className="stored-memory-badges"><Badge tone="green">{TYPE_LABELS[kind] ?? "长期信息"}</Badge><Badge tone={record.access_policy === "standard" ? "gray" : "amber"}>{POLICY_LABELS[record.access_policy] ?? record.access_policy}</Badge>{record.user_locked ? <Badge tone="amber">已锁定</Badge> : null}</div><strong>{record.title || "未命名记忆"}</strong>{content && content !== record.title ? <p>{content}</p> : null}<div className="memory-record-meta"><span>{Math.round(record.confidence * 100)}% 可信</span><span>{SENSITIVITY_LABELS[record.sensitivity] ?? record.sensitivity}</span><span>来源：{sourceLabel(record.source)}</span>{record.expires_at ? <span>有效至 {relativeTime(record.expires_at)}</span> : <span>长期有效</span>}<span>更新于 {relativeTime(record.updated_at)}</span></div></div><IconButton icon={Trash2} label="忘记这条内容" danger disabled={props.busy === record.id} onClick={() => void props.onForget(record)} /></article>; })}</div> : <EmptyState icon={ShieldCheck} title={props.hasQuery ? "没有匹配的个人记忆" : "还没有形成关于你的稳定记忆"} text={props.hasQuery ? "换一个关键词，或清空搜索条件。" : "继续聊天即可。稳定信息会自动进入治理流程，你也可以明确告诉小满‘记住这件事’。"} />}</section>;
}

function ExecutionPanel(props: { loading: boolean; memories: ExecutionMemoryRow[]; hasQuery: boolean; busy: string; onRemove: (memory: ExecutionMemoryRow) => Promise<void> }): React.ReactElement {
  if (props.loading) return <LoadingState />;
  return <section className="memory-panel"><div className="memory-panel-heading"><div><h2>小满的执行经验</h2><p>记录工具怎么用、项目约定和环境差异。只有真实执行成功才会提高可信度，连续失败会自动停用。</p></div><Badge tone="blue">按项目与工具隔离</Badge></div>{props.memories.length ? <div className="stored-memory-list">{props.memories.map((memory) => { const state = memory.execution; const scope = executionScopeLabel(state.scope); const attempts = state.success_count + state.failure_count; return <article className="stored-memory-row" key={memory.id}><span className="stored-memory-icon"><Wrench size={17} /></span><div className="stored-memory-main"><div className="stored-memory-badges"><Badge tone={state.verification_status === "verified" ? "green" : state.verification_status === "quarantined" ? "amber" : "gray"}>{executionStatusLabel(state.verification_status)}</Badge><Badge tone="gray">{executionKindLabel(state.kind)}</Badge></div><strong>{memory.summary || "未命名执行经验"}</strong><div className="memory-record-meta"><span>{scope}</span><span>{attempts ? `成功 ${state.success_count} · 失败 ${state.failure_count}` : "等待实际执行验证"}</span>{state.last_verified_at ? <span>验证于 {relativeTime(state.last_verified_at)}</span> : null}<span>更新于 {relativeTime(memory.updated_at)}</span></div></div><IconButton icon={Trash2} label="删除这条执行经验" danger disabled={props.busy === memory.id} onClick={() => void props.onRemove(memory)} /></article>; })}</div> : <EmptyState icon={Wrench} title={props.hasQuery ? "没有匹配的小满经验" : "还没有可复用的执行经验"} text={props.hasQuery ? "换一个关键词，或清空搜索条件。" : "小满在完成带工具的任务后，会把真正有用且适用范围明确的做法整理到这里。"} />}</section>;
}

function RecallPanel(props: { loading: boolean; memories: StoredMemoryRow[]; hasQuery: boolean; busy: string; onRemove: (memory: StoredMemoryRow) => Promise<void> }): React.ReactElement {
  if (props.loading) return <LoadingState />;
  return <section className="memory-panel"><div className="memory-panel-heading"><div><h2>聊天中的过往回忆</h2><p>小满会结合语义、关键词和关联寻找原始聊天，但不会把每一句对话都直接当成长期事实。</p></div><Badge tone="blue">可回到原对话</Badge></div>{props.memories.length ? <div className="stored-memory-list">{props.memories.map((memory) => { const recallCount = Number(memory.extra_json.recall_count ?? 0); const session = String(memory.extra_json.session_key ?? ""); return <article className="stored-memory-row" key={memory.id}><span className="stored-memory-icon"><MessageSquareText size={17} /></span><div className="stored-memory-main"><div className="stored-memory-badges"><Badge tone="blue">对话回忆</Badge><Badge tone="gray">可检索</Badge></div><strong>{memory.summary || "未命名回忆"}</strong><div className="memory-record-meta"><span>{session || "本地对话"}</span><span>{recallCount ? `已被找回 ${recallCount} 次` : "尚未被主动找回"}</span><span>收录于 {relativeTime(memory.updated_at || memory.created_at)}</span></div></div><IconButton icon={Trash2} label="删除这条回忆索引" danger disabled={props.busy === memory.id} onClick={() => void props.onRemove(memory)} /></article>; })}</div> : <EmptyState icon={MessageSquareText} title={props.hasQuery ? "没有匹配的聊天回忆" : "还没有可找回的对话"} text={props.hasQuery ? "换一个关键词，或清空搜索条件。" : "继续和小满对话后，系统会自动建立可找回的聊天回忆。"} />}</section>;
}

function PendingPanel(props: { loading: boolean; conflicts: MemoryConflictRow[]; busy: string; onResolve: (conflict: MemoryConflictRow, action: "keep_existing" | "accept_candidate") => Promise<void> }): React.ReactElement {
  if (props.loading) return <LoadingState />;
  return <section className="memory-panel"><div className="memory-panel-heading"><div><h2>需要你决定的记忆</h2><p>自动发现的信息只有在发生冲突、涉及敏感内容或需要改写旧记忆时才会停在这里。</p></div></div>{props.conflicts.length ? <div className="memory-conflict-list">{props.conflicts.map((conflict) => { const candidate = conflict.candidate; const candidateData = candidate.data && typeof candidate.data === "object" ? candidate.data as Record<string, unknown> : {}; const candidateText = String(candidateData.content ?? candidate.summary ?? "新记忆"); return <article className="memory-conflict-card" key={conflict.id}><div className="memory-conflict-icon"><ShieldCheck size={17} /></div><div><Badge tone="amber">需要你确认</Badge><h3>{String(candidate.summary ?? "记忆内容发生变化")}</h3>{conflict.existing ? <p><small>原来</small>{String(conflict.existing.data.content ?? conflict.existing.summary)}</p> : null}<p><small>新的</small>{candidateText}</p><div className="memory-conflict-actions"><button className="secondary-button" disabled={props.busy === conflict.id} onClick={() => void props.onResolve(conflict, "keep_existing")}><X size={14} />保留原内容</button><button className="primary-button" disabled={props.busy === conflict.id} onClick={() => void props.onResolve(conflict, "accept_candidate")}><Check size={14} />使用新内容</button></div></div></article>; })}</div> : <EmptyState icon={Clock3} title="没有需要确认的记忆" text="自动发现和明确记住的内容已经通过统一治理，没有未处理冲突。" />}</section>;
}

function filterGoverned(records: PersonalRecordRow[], query: string): PersonalRecordRow[] {
  if (!query) return records;
  return records.filter((record) => `${record.title}\n${record.summary}\n${String(record.data.content ?? "")}`.toLocaleLowerCase().includes(query));
}

function filterExecution(records: ExecutionMemoryRow[], query: string): ExecutionMemoryRow[] {
  if (!query) return records;
  return records.filter((record) => `${record.summary}\n${record.execution.scope.tool_name}\n${record.execution.scope.project_id}`.toLocaleLowerCase().includes(query));
}

function executionStatusLabel(status: string): string {
  return ({ candidate: "待验证", verified: "已验证", stale: "可能过时", quarantined: "已停用", superseded: "已替代" } as Record<string, string>)[status] ?? status;
}

function executionKindLabel(kind: string): string {
  return ({ environment: "环境信息", project_convention: "项目约定", tool_lesson: "工具经验", procedure: "执行流程", decision: "技术决策", capability: "能力边界" } as Record<string, string>)[kind] ?? "执行经验";
}

function executionScopeLabel(scope: ExecutionMemoryRow["execution"]["scope"]): string {
  if (scope.tool_name) return `工具：${scope.tool_name}`;
  if (scope.plugin_name) return `系统组件：${scope.plugin_name}`;
  if (scope.project_id) return `项目：${scope.project_id}`;
  if (scope.workspace_id) return `工作区：${scope.workspace_id}`;
  return "全局适用";
}

function sourceLabel(source: string): string {
  if (source === "conversation_consolidation") return "对话自动整理";
  if (source === "dashboard") return "你在记忆管理中添加";
  if (source === "explicit_memory") return "你明确要求记住";
  return source || "小满";
}

function relationLabelForKind(kind: string): string {
  return ({ requested: "明确记住", fact: "事实", preference: "偏好", relationship: "关系", historical_event: "经历" } as Record<string, string>)[kind] ?? "关联";
}
