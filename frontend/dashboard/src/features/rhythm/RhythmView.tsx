import React, { useMemo, useState } from "react";
import { BedDouble, BellRing, CakeSlice, CalendarClock, Check, CheckCircle2, Clock3, Compass, CreditCard, DoorOpen, Focus, Gauge, HeartPulse, House, ListChecks, Pencil, Plane, Plus, RefreshCw, Save, Target, Trash2, type LucideIcon } from "lucide-react";
import { api } from "../../api";
import { relativeTime, shortTs } from "../../format";
import { Badge, EmptyState, ErrorBanner, IconButton, LoadingState, Modal, PageIntro } from "../../shared/components/ui";
import { personalTypeLabels } from "../../shared/constants/personal";
import { useAsyncData } from "../../shared/hooks/useAsyncData";
import type { PersonalRecordRow, RhythmDomain, RhythmFormState, RhythmOverview, RhythmRecommendationResult, RhythmReport } from "../../shared/types";

const rhythmDomains: Array<{ id: RhythmDomain; label: string; icon: LucideIcon }> = [
  { id: "commitment", label: "可执行任务", icon: ListChecks },
  { id: "relationship", label: "重要关系", icon: HeartPulse },
  { id: "important_date", label: "重要日期", icon: CakeSlice },
  { id: "financial_obligation", label: "账单续费", icon: CreditCard },
  { id: "trip", label: "旅行", icon: Plane },
  { id: "goal", label: "目标", icon: Target },
  { id: "proactive_intent", label: "主动关注", icon: BellRing },
];

function emptyRhythmForm(type: RhythmDomain): RhythmFormState {
  return {
    type, title: "", summary: "", due_at: "", estimated_minutes: "30", priority: "normal", energy: "medium", context: "any", next_action: "",
    person_name: "", relationship: "", last_contact_at: "", contact_interval_days: "30", date: "", repeat_yearly: true, preparation_days: "7",
    obligation_type: "subscription", amount: "", currency: "CNY", recurrence: "none", auto_renew: false, reminder_days: "7",
    destination: "", depart_at: "", return_at: "", checklist: "", target: "", current: "0", unit: "", start_at: "", direction: "increase",
    message: "", reason: "", trigger_type: "interval", next_trigger_at: "", interval_minutes: "10080", inactivity_days: "30",
    target_entity_type: "", target_record_key: "", enabled: true,
  };
}

function localDateTimeValue(value: unknown): string {
  if (!value) return "";
  const date = new Date(String(value));
  if (Number.isNaN(date.getTime())) return "";
  return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
}

function localInputToIso(value: string): string | null {
  if (!value.trim()) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date.toISOString();
}

export function RhythmView(props: { embedded?: boolean } = {}): React.ReactElement {
  const base = "/api/dashboard/control/rhythm";
  const [minutes, setMinutes] = useState(30);
  const [domain, setDomain] = useState<RhythmDomain>("commitment");
  const [form, setForm] = useState<RhythmFormState>(() => emptyRhythmForm("commitment"));
  const [editing, setEditing] = useState<PersonalRecordRow | null>(null);
  const [adding, setAdding] = useState(false);
  const [report, setReport] = useState<RhythmReport | null>(null);
  const [busy, setBusy] = useState("");
  const [actionError, setActionError] = useState("");
  const overview = useAsyncData(() => api<RhythmOverview>(`${base}/overview`), []);
  const records = useAsyncData(() => api<PersonalRecordRow[]>("/api/dashboard/control/personal/records?limit=1000"), []);
  const recommendations = useAsyncData(() => api<RhythmRecommendationResult>(`${base}/recommendations?minutes=${minutes}&limit=5`), [minutes]);
  const context = overview.data?.context;
  const visibleRecords = useMemo(() => (records.data ?? []).filter((record) => record.entity_type === domain), [records.data, domain]);
  const refresh = (): void => { overview.reload(); records.reload(); recommendations.reload(); };
  const sceneOptions: Array<{ value: string; label: string; icon: LucideIcon }> = [
    { value: "neutral", label: "日常", icon: Compass },
    { value: "leaving", label: "出门", icon: DoorOpen },
    { value: "home", label: "回家", icon: House },
    { value: "bedtime", label: "睡前", icon: BedDouble },
    { value: "travel", label: "旅行中", icon: Plane },
  ];

  const setScene = async (scene: string): Promise<void> => {
    setBusy(`scene:${scene}`); setActionError("");
    try { await api(`${base}/scene`, { method: "POST", body: JSON.stringify({ scene }) }); refresh(); }
    catch (reason) { setActionError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(""); }
  };
  const startFocus = async (duration: number): Promise<void> => {
    setBusy("focus"); setActionError("");
    try { await api(`${base}/focus`, { method: "POST", body: JSON.stringify({ minutes: duration, label: "专注", allow_high_priority: true }) }); refresh(); }
    catch (reason) { setActionError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(""); }
  };
  const stopFocus = async (): Promise<void> => {
    setBusy("focus"); setActionError("");
    try { await api(`${base}/focus`, { method: "DELETE" }); refresh(); }
    catch (reason) { setActionError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(""); }
  };
  const generateReport = async (period: "week" | "month"): Promise<void> => {
    setBusy(`report:${period}`); setActionError("");
    try { setReport(await api<RhythmReport>(`${base}/reports`, { method: "POST", body: JSON.stringify({ period, persist: true }) })); records.reload(); }
    catch (reason) { setActionError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(""); }
  };
  const openAdd = (): void => { setEditing(null); setForm(emptyRhythmForm(domain)); setAdding(true); };
  const openEdit = (record: PersonalRecordRow): void => {
    const data = record.data;
    const checklist = Array.isArray(data.checklist) ? data.checklist.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object").map((item) => String(item.title ?? "")).filter(Boolean).join("\n") : "";
    setEditing(record);
    setForm({
      ...emptyRhythmForm(record.entity_type as RhythmDomain),
      type: record.entity_type as RhythmDomain,
      title: record.title,
      summary: record.summary,
      due_at: localDateTimeValue(data.due_at),
      estimated_minutes: String(data.estimated_minutes ?? 30),
      priority: String(data.priority ?? "normal"),
      energy: String(data.energy ?? "medium"),
      context: Array.isArray(data.contexts) ? String(data.contexts[0] ?? "any") : "any",
      next_action: String(data.next_action ?? ""),
      person_name: String(data.person_name ?? ""), relationship: String(data.relationship ?? ""), last_contact_at: localDateTimeValue(data.last_contact_at), contact_interval_days: String(data.contact_interval_days ?? 30),
      date: String(data.date ?? ""), repeat_yearly: Boolean(data.repeat_yearly ?? true), preparation_days: String(data.preparation_days ?? 7),
      obligation_type: String(data.obligation_type ?? "subscription"), amount: data.amount == null ? "" : String(data.amount), currency: String(data.currency ?? "CNY"), recurrence: String(data.recurrence ?? (data.condition as Record<string, unknown> | undefined)?.recurrence ?? "none"), auto_renew: Boolean(data.auto_renew), reminder_days: String(data.reminder_days ?? 7),
      destination: String(data.destination ?? ""), depart_at: localDateTimeValue(data.depart_at), return_at: localDateTimeValue(data.return_at), checklist,
      target: String(data.target ?? ""), current: String(data.current ?? 0), unit: String(data.unit ?? ""), start_at: localDateTimeValue(data.start_at), direction: String(data.direction ?? "increase"),
      message: String(data.message ?? ""), reason: String(data.reason ?? record.summary), trigger_type: String(data.trigger_type ?? "interval"), next_trigger_at: localDateTimeValue(data.next_trigger_at), interval_minutes: String(data.interval_minutes ?? 10080), inactivity_days: String(data.inactivity_days ?? 30),
      target_entity_type: String(data.target_entity_type ?? ""), target_record_key: String(data.target_record_key ?? ""), enabled: Boolean(data.enabled ?? false),
    });
    setAdding(true);
  };
  const buildData = (): Record<string, unknown> => {
    const currentData = editing?.data ?? {};
    if (form.type === "commitment") return { ...currentData, state: String(currentData.state ?? "open"), due_at: localInputToIso(form.due_at), estimated_minutes: Number(form.estimated_minutes || 30), priority: form.priority, energy: form.energy, contexts: [form.context], next_action: form.next_action, progress: Number(currentData.progress ?? 0) };
    if (form.type === "relationship") return { ...currentData, person_name: form.person_name || form.title, relationship: form.relationship, last_contact_at: localInputToIso(form.last_contact_at), contact_interval_days: Number(form.contact_interval_days || 30), important: true };
    if (form.type === "important_date") return { ...currentData, date: form.date, description: form.summary, person_name: form.person_name, repeat_yearly: form.repeat_yearly, preparation_days: Number(form.preparation_days || 7) };
    if (form.type === "financial_obligation") return { ...currentData, obligation_type: form.obligation_type, due_at: localInputToIso(form.due_at), amount: form.amount ? Number(form.amount) : null, currency: form.currency, recurrence: form.recurrence, auto_renew: form.auto_renew, reminder_days: Number(form.reminder_days || 7), state: String(currentData.state ?? "active") };
    if (form.type === "trip") {
      const oldItems = Array.isArray(currentData.checklist) ? currentData.checklist.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object") : [];
      const checklist = form.checklist.split("\n").map((item) => item.trim()).filter(Boolean).map((title) => ({ title, done: Boolean(oldItems.find((item) => String(item.title) === title)?.done), estimated_minutes: Number(oldItems.find((item) => String(item.title) === title)?.estimated_minutes ?? 10) }));
      return { ...currentData, destination: form.destination || form.title, depart_at: localInputToIso(form.depart_at), return_at: localInputToIso(form.return_at), state: String(currentData.state ?? "planning"), checklist, itinerary: Array.isArray(currentData.itinerary) ? currentData.itinerary : [] };
    }
    if (form.type === "goal") return { ...currentData, target: Number(form.target), current: Number(form.current), unit: form.unit, start_at: localInputToIso(form.start_at), due_at: localInputToIso(form.due_at), direction: form.direction, state: String(currentData.state ?? "active") };
    return { ...currentData, trigger_type: form.trigger_type, message: form.message, reason: form.reason, next_trigger_at: localInputToIso(form.next_trigger_at), interval_minutes: form.interval_minutes ? Number(form.interval_minutes) : null, target_entity_type: form.target_entity_type, target_record_key: form.target_record_key, inactivity_days: form.inactivity_days ? Number(form.inactivity_days) : null, condition: { recurrence: form.recurrence }, enabled: form.enabled, status: form.enabled ? "active" : "proposed", cooldown_minutes: 60 };
  };
  const saveRecord = async (): Promise<void> => {
    setBusy("save"); setActionError("");
    try {
      const data = buildData();
      if (editing) {
        await api(`/api/dashboard/control/personal/records/${editing.id}`, { method: "PATCH", body: JSON.stringify({ title: form.title, summary: form.summary || form.reason || form.title, data, reason: "从个人节奏工作台修改" }) });
      } else if (form.type === "proactive_intent") {
        await api(`${base}/follow-ups`, { method: "POST", body: JSON.stringify({ title: form.title, message: form.message, reason: form.reason, trigger_type: form.trigger_type, next_trigger_at: localInputToIso(form.next_trigger_at), interval_minutes: form.interval_minutes ? Number(form.interval_minutes) : null, target_entity_type: form.target_entity_type, target_record_key: form.target_record_key, inactivity_days: form.inactivity_days ? Number(form.inactivity_days) : null, condition: { recurrence: form.recurrence }, cooldown_minutes: 60, enabled: form.enabled }) });
      } else {
        await api(`${base}/records/${form.type}`, { method: "POST", body: JSON.stringify({ title: form.title, summary: form.summary || form.title, data }) });
      }
      setAdding(false); setEditing(null); refresh();
    } catch (reason) { setActionError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(""); }
  };
  const updateRecordData = async (record: PersonalRecordRow, nextData: Record<string, unknown>, reason: string): Promise<void> => {
    setBusy(record.id); setActionError("");
    try { await api(`/api/dashboard/control/personal/records/${record.id}`, { method: "PATCH", body: JSON.stringify({ data: nextData, reason }) }); refresh(); }
    catch (error) { setActionError(error instanceof Error ? error.message : String(error)); }
    finally { setBusy(""); }
  };
  const removeRecord = async (record: PersonalRecordRow): Promise<void> => {
    if (!window.confirm(`确认移除“${record.title}”？`)) return;
    await api(`/api/dashboard/control/personal/records/${record.id}`, { method: "DELETE", body: JSON.stringify({ reason: "从个人节奏工作台移除" }) }); refresh();
  };
  const recordDetail = (record: PersonalRecordRow): string => {
    const data = record.data;
    if (record.entity_type === "commitment") return `${data.estimated_minutes ?? "未估时"} 分钟 · ${data.next_action ?? record.summary}${data.due_text ? ` · ${data.due_text}` : ""}`;
    if (record.entity_type === "relationship") return `上次联系 ${data.last_contact_at ? relativeTime(String(data.last_contact_at)) : "未记录"} · 周期 ${data.contact_interval_days ?? 30} 天`;
    if (record.entity_type === "important_date") return `${data.date ?? "未设置日期"} · 提前 ${data.preparation_days ?? 7} 天准备`;
    if (record.entity_type === "financial_obligation") return `${data.obligation_type ?? "账单"} · ${data.due_at ? shortTs(String(data.due_at)) : "未设置到期日"} · ${data.auto_renew ? "自动续费" : "手动处理"}`;
    if (record.entity_type === "trip") return `${data.destination ?? record.title} · ${data.depart_at ? shortTs(String(data.depart_at)) : "未设置出发时间"}`;
    if (record.entity_type === "goal") return `${data.current ?? 0} / ${data.target ?? 0} ${data.unit ?? ""} · 截止 ${data.due_at ? shortTs(String(data.due_at)) : "未设置"}`;
    return `${data.enabled ? "已启用" : "待启用"} · ${data.trigger_type ?? "interval"} · ${data.next_trigger_at ? shortTs(String(data.next_trigger_at)) : "等待条件"}`;
  };
  const DomainIcon = rhythmDomains.find((item) => item.id === domain)?.icon ?? ListChecks;
  return <>
    {props.embedded ? <div className="embedded-section-heading"><div><h2>节奏与关注</h2><p>根据时间、场景和精力安排下一步，同时管理需要持续关注的生活事项。</p></div><IconButton icon={RefreshCw} label="刷新节奏数据" onClick={refresh} /></div> : <PageIntro title="个人节奏" description="小满把时间窗口、场景、精力和长期事项放在同一张上下文里，再决定推荐、保持安静或主动联系。" actions={<IconButton icon={RefreshCw} label="刷新" onClick={refresh} />} />}
    <ErrorBanner message={overview.error || records.error || recommendations.error || actionError} />
    <section className="rhythm-context-band">
      <div className="rhythm-context-copy"><span>当前上下文</span><strong>{context?.focus_active ? context.focus_label : sceneOptions.find((item) => item.value === context?.scene)?.label ?? "日常"}</strong><p>{context?.do_not_disturb ? "免打扰已生效，非紧急提醒会延后。" : `当前精力：${context?.energy === "low" ? "偏低" : context?.energy === "high" ? "充足" : "一般"}`}</p></div>
      <div className="scene-segments" aria-label="场景模式">{sceneOptions.map((item) => { const Icon = item.icon; return <button key={item.value} className={context?.scene === item.value && !context.focus_active ? "active" : ""} title={item.label} disabled={busy.startsWith("scene:")} onClick={() => void setScene(item.value)}><Icon size={16} /><span>{item.label}</span></button>; })}</div>
      <div className="focus-control"><div><Focus size={16} /><span>{context?.focus_active ? `至 ${context.focus_ends_at ? shortTs(context.focus_ends_at) : "稍后"}` : "开始专注"}</span></div>{context?.focus_active ? <button className="secondary-button" disabled={busy === "focus"} onClick={() => void stopFocus()}>结束</button> : <div className="focus-durations">{[25, 50, 90].map((value) => <button key={value} disabled={busy === "focus"} onClick={() => void startFocus(value)}>{value}</button>)}</div>}</div>
    </section>
    <section className="rhythm-opportunity">
      <div className="rhythm-section-heading"><div><h2>我现在有</h2><p>候选任务按估时、截止风险、场景和精力动态排序。</p></div><div className="time-segments">{[15, 30, 60, 90].map((value) => <button className={minutes === value ? "active" : ""} key={value} onClick={() => setMinutes(value)}>{value} 分钟</button>)}</div></div>
      {recommendations.loading && !recommendations.data ? <LoadingState /> : recommendations.data?.recommendations.length ? <div className="rhythm-recommendations">{recommendations.data.recommendations.map((item, index) => <div className="rhythm-recommendation" key={item.candidate_id}><span className="recommendation-rank">{index + 1}</span><div><div><strong>{item.title}</strong><Badge tone={item.energy === "high" ? "amber" : item.energy === "low" ? "green" : "blue"}>{item.estimated_minutes} 分钟</Badge></div><p>{item.next_action}</p><small>{item.reason}</small></div></div>)}</div> : <EmptyState icon={Clock3} title={`还没有适合 ${minutes} 分钟完成的事项`} text="为任务补充估时、下一步、所需精力和适用场景后，小满会在合适的时候把它推荐出来。" />}
    </section>
    <section className="rhythm-report-section">
      <div className="rhythm-section-heading"><div><h2>节奏回顾</h2><p>从同一份完成记录、健康状态和目标进度生成，不另建统计口径。</p></div><div className="rhythm-report-actions"><button className="secondary-button" disabled={busy.startsWith("report:")} onClick={() => void generateReport("week")}><Gauge size={15} />周报</button><button className="secondary-button" disabled={busy.startsWith("report:")} onClick={() => void generateReport("month")}><CalendarClock size={15} />月报</button></div></div>
      {report ? <div className="rhythm-report-result"><div className="report-metrics"><div><span>已完成</span><strong>{report.metrics.commitments_completed ?? 0}</strong></div><div><span>待推进</span><strong>{report.metrics.commitments_open ?? 0}</strong></div><div><span>逾期</span><strong>{report.metrics.commitments_overdue ?? 0}</strong></div><div><span>目标偏差</span><strong>{report.deviations.length}</strong></div></div><div className="report-findings">{report.deviations.map((item) => <p key={item.record_id}><Target size={14} /><strong>{item.title}</strong> 实际 {Math.round(item.actual_progress * 100)}%，时间进度 {Math.round(item.expected_progress * 100)}%</p>)}{report.recommendations.map((item) => <p key={item}><CheckCircle2 size={14} />{item}</p>)}</div></div> : <div className="report-placeholder"><Gauge size={19} /><span>生成一次周报或月报，查看任务完成、状态趋势和目标偏差。</span></div>}
    </section>
    <section className="rhythm-record-section">
      <div className="rhythm-record-toolbar"><div className="filter-tabs">{rhythmDomains.map((item) => <button key={item.id} className={domain === item.id ? "active" : ""} onClick={() => setDomain(item.id)}>{item.label}</button>)}</div><button className="primary-button" onClick={openAdd}><Plus size={15} />新增{rhythmDomains.find((item) => item.id === domain)?.label}</button></div>
      {records.loading && !records.data ? <LoadingState /> : visibleRecords.length ? <div className="rhythm-record-list">{visibleRecords.map((record) => { const checklist = Array.isArray(record.data.checklist) ? record.data.checklist.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object") : []; return <div className="rhythm-record-row" key={record.id}><span className={`rhythm-record-icon ${record.entity_type}`}><DomainIcon size={17} /></span><div className="rhythm-record-main"><div><strong>{record.title}</strong><Badge tone={record.data.enabled === false ? "amber" : "gray"}>{rhythmDomains.find((item) => item.id === record.entity_type)?.label ?? personalTypeLabels[record.entity_type] ?? record.entity_type}</Badge></div><p>{recordDetail(record)}</p>{record.entity_type === "trip" && checklist.length ? <div className="trip-checklist">{checklist.map((item, index) => <button className={item.done ? "done" : ""} key={`${String(item.title)}-${index}`} onClick={() => { const next = checklist.map((candidate, candidateIndex) => candidateIndex === index ? { ...candidate, done: !candidate.done } : candidate); void updateRecordData(record, { ...record.data, checklist: next }, "更新旅行准备清单"); }}><Check size={12} />{String(item.title)}</button>)}</div> : null}<small>更新于 {relativeTime(record.updated_at)}</small></div><div className="rhythm-record-actions">{record.entity_type === "commitment" && record.data.state !== "completed" ? <button className="text-button" disabled={busy === record.id} onClick={() => void updateRecordData(record, { ...record.data, state: "completed", completed_at: new Date().toISOString(), progress: 1 }, "完成任务")}>完成</button> : null}{record.entity_type === "relationship" ? <button className="text-button" disabled={busy === record.id} onClick={() => void updateRecordData(record, { ...record.data, last_contact_at: new Date().toISOString() }, "记录最近联系")}>刚联系过</button> : null}{record.entity_type === "proactive_intent" ? <button className={`toggle${record.data.enabled ? " on" : ""}`} aria-label={record.data.enabled ? "关闭主动关注" : "启用主动关注"} onClick={() => void updateRecordData(record, { ...record.data, enabled: !record.data.enabled, status: record.data.enabled ? "proposed" : "active" }, "切换主动关注")}><span /></button> : null}<IconButton icon={Pencil} label="编辑" onClick={() => openEdit(record)} /><IconButton icon={Trash2} label="移除" danger onClick={() => void removeRecord(record)} /></div></div>; })}</div> : <EmptyState icon={DomainIcon} title={`还没有${rhythmDomains.find((item) => item.id === domain)?.label}`} text="可以在这里录入，也可以直接在聊天中告诉小满。" />}
    </section>
    {adding ? <Modal title={`${editing ? "编辑" : "新增"}${rhythmDomains.find((item) => item.id === form.type)?.label ?? personalTypeLabels[form.type] ?? form.type}`} description={form.type === "proactive_intent" ? "你主动创建的关注会立即启用；小满推断出的关注会先等你确认。" : form.type === "commitment" ? "补充预计用时和明确下一步，小满才能在合适的时间向你推荐。" : "这些信息会用于日常提醒、下一步建议和节奏回顾。"} onClose={() => setAdding(false)}><div className="form-stack form-two"><label className="span-two">名称<input value={form.title} onChange={(event) => setForm({ ...form, title: event.target.value })} /></label>{form.type !== "proactive_intent" ? <label className="span-two">说明<textarea rows={2} value={form.summary} onChange={(event) => setForm({ ...form, summary: event.target.value })} /></label> : null}
      {form.type === "commitment" ? <><label>预计分钟<input type="number" min="5" value={form.estimated_minutes} onChange={(event) => setForm({ ...form, estimated_minutes: event.target.value })} /></label><label>截止时间<input type="datetime-local" value={form.due_at} onChange={(event) => setForm({ ...form, due_at: event.target.value })} /></label><label>优先级<select value={form.priority} onChange={(event) => setForm({ ...form, priority: event.target.value })}><option value="normal">普通</option><option value="high">高</option><option value="urgent">紧急</option></select></label><label>所需精力<select value={form.energy} onChange={(event) => setForm({ ...form, energy: event.target.value })}><option value="low">低</option><option value="medium">中</option><option value="high">高</option></select></label><label>适用场景<select value={form.context} onChange={(event) => setForm({ ...form, context: event.target.value })}><option value="any">任意</option><option value="home">回家</option><option value="leaving">出门</option><option value="bedtime">睡前</option><option value="travel">旅行中</option></select></label><label>明确下一步<input value={form.next_action} onChange={(event) => setForm({ ...form, next_action: event.target.value })} /></label></> : null}
      {form.type === "relationship" ? <><label>姓名<input value={form.person_name} onChange={(event) => setForm({ ...form, person_name: event.target.value })} /></label><label>关系<input value={form.relationship} onChange={(event) => setForm({ ...form, relationship: event.target.value })} /></label><label>最近联系<input type="datetime-local" value={form.last_contact_at} onChange={(event) => setForm({ ...form, last_contact_at: event.target.value })} /></label><label>联络周期（天）<input type="number" min="1" value={form.contact_interval_days} onChange={(event) => setForm({ ...form, contact_interval_days: event.target.value })} /></label></> : null}
      {form.type === "important_date" ? <><label>日期<input type="date" value={form.date} onChange={(event) => setForm({ ...form, date: event.target.value })} /></label><label>相关的人<input value={form.person_name} onChange={(event) => setForm({ ...form, person_name: event.target.value })} /></label><label>提前准备（天）<input type="number" min="0" value={form.preparation_days} onChange={(event) => setForm({ ...form, preparation_days: event.target.value })} /></label><label className="check-label"><input type="checkbox" checked={form.repeat_yearly} onChange={(event) => setForm({ ...form, repeat_yearly: event.target.checked })} />每年重复</label></> : null}
      {form.type === "financial_obligation" ? <><label>类型<select value={form.obligation_type} onChange={(event) => setForm({ ...form, obligation_type: event.target.value })}><option value="subscription">订阅</option><option value="bill">账单</option><option value="renewal">续费</option></select></label><label>到期时间<input type="datetime-local" value={form.due_at} onChange={(event) => setForm({ ...form, due_at: event.target.value })} /></label><label>金额<input type="number" min="0" step="0.01" value={form.amount} onChange={(event) => setForm({ ...form, amount: event.target.value })} /></label><label>币种<input value={form.currency} onChange={(event) => setForm({ ...form, currency: event.target.value })} /></label><label>提醒提前天数<input type="number" min="0" value={form.reminder_days} onChange={(event) => setForm({ ...form, reminder_days: event.target.value })} /></label><label className="check-label"><input type="checkbox" checked={form.auto_renew} onChange={(event) => setForm({ ...form, auto_renew: event.target.checked })} />自动续费</label></> : null}
      {form.type === "trip" ? <><label>目的地<input value={form.destination} onChange={(event) => setForm({ ...form, destination: event.target.value })} /></label><label>出发时间<input type="datetime-local" value={form.depart_at} onChange={(event) => setForm({ ...form, depart_at: event.target.value })} /></label><label>返回时间<input type="datetime-local" value={form.return_at} onChange={(event) => setForm({ ...form, return_at: event.target.value })} /></label><label className="span-two">准备清单（每行一项）<textarea rows={5} value={form.checklist} onChange={(event) => setForm({ ...form, checklist: event.target.value })} /></label></> : null}
      {form.type === "goal" ? <><label>目标值<input type="number" value={form.target} onChange={(event) => setForm({ ...form, target: event.target.value })} /></label><label>当前值<input type="number" value={form.current} onChange={(event) => setForm({ ...form, current: event.target.value })} /></label><label>单位<input value={form.unit} onChange={(event) => setForm({ ...form, unit: event.target.value })} /></label><label>方向<select value={form.direction} onChange={(event) => setForm({ ...form, direction: event.target.value })}><option value="increase">逐步增加</option><option value="decrease">逐步降低</option></select></label><label>开始时间<input type="datetime-local" value={form.start_at} onChange={(event) => setForm({ ...form, start_at: event.target.value })} /></label><label>截止时间<input type="datetime-local" value={form.due_at} onChange={(event) => setForm({ ...form, due_at: event.target.value })} /></label></> : null}
      {form.type === "proactive_intent" ? <><label className="span-two">为什么持续关注<textarea rows={2} value={form.reason} onChange={(event) => setForm({ ...form, reason: event.target.value })} /></label><label className="span-two">联系你时建议什么<textarea rows={2} value={form.message} onChange={(event) => setForm({ ...form, message: event.target.value })} /></label><label>触发方式<select value={form.trigger_type} onChange={(event) => setForm({ ...form, trigger_type: event.target.value })}><option value="interval">固定频率</option><option value="at_time">指定时间</option><option value="inactivity">长期无进展</option></select></label>{form.trigger_type === "interval" ? <label>联系频率<select value={form.interval_minutes} onChange={(event) => setForm({ ...form, interval_minutes: event.target.value })}><option value="1440">每天</option><option value="10080">每周</option><option value="20160">每两周</option><option value="43200">每月</option></select></label> : null}{form.trigger_type === "at_time" ? <><label>下次触发<input type="datetime-local" value={form.next_trigger_at} onChange={(event) => setForm({ ...form, next_trigger_at: event.target.value })} /></label><label>重复<select value={form.recurrence} onChange={(event) => setForm({ ...form, recurrence: event.target.value })}><option value="none">仅一次</option><option value="daily">每天</option><option value="weekly">每周</option><option value="monthly">每月</option></select></label></> : null}{form.trigger_type === "inactivity" ? <><label>观察的数据<select value={form.target_entity_type} onChange={(event) => setForm({ ...form, target_entity_type: event.target.value })}><option value="commitment">任务</option><option value="goal">目标</option><option value="relationship">关系</option><option value="trip">旅行</option></select></label><label>无进展天数<input type="number" min="1" value={form.inactivity_days} onChange={(event) => setForm({ ...form, inactivity_days: event.target.value })} /></label></> : null}<label className="check-label span-two"><input type="checkbox" checked={form.enabled} onChange={(event) => setForm({ ...form, enabled: event.target.checked })} />立即启用主动关注</label></> : null}
      <div className="dialog-actions span-two"><button className="secondary-button" onClick={() => setAdding(false)}>取消</button><button className="primary-button" disabled={!form.title.trim() || busy === "save"} onClick={() => void saveRecord()}><Save size={15} />{busy === "save" ? "保存中" : "保存"}</button></div></div></Modal> : null}
  </>;
}
