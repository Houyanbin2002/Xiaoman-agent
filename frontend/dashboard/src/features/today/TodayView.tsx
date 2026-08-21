import React, { useMemo, useState } from "react";
import { CalendarClock, CheckCircle2, Compass, HeartPulse, MoonStar, Plus, RefreshCw, Sunrise, Trash2, Workflow } from "lucide-react";
import { api } from "../../api";
import { relativeTime } from "../../format";
import { Badge, EmptyState, ErrorBanner, IconButton, LoadingState, Modal, PageIntro } from "../../shared/components/ui";
import { personalTypeLabels } from "../../shared/constants/personal";
import { useAsyncData } from "../../shared/hooks/useAsyncData";
import type { ExternalSourceRow, PersonalOverview, PersonalRecordRow, PersonalTodayResponse, ViewId } from "../../shared/types";
import { RhythmView } from "../rhythm/RhythmView";

const DAILY_RECORD_TYPES = ["commitment", "daily_plan", "calendar_event", "health_observation", "check_in"] as const;

export function TodayView(props: { navigate: (view: ViewId) => void }): React.ReactElement {
  const [section, setSection] = useState<"today" | "rhythm">("today");
  return <>
    <PageIntro title="我的一天" description="把今天要做的事、当前状态和持续关注放在同一个工作台里。" />
    <div className="day-workspace-tabs" role="tablist" aria-label="我的一天视图">
      <button type="button" role="tab" aria-selected={section === "today"} className={section === "today" ? "active" : ""} onClick={() => setSection("today")}><Sunrise size={16} /><span><strong>今日安排</strong><small>计划、待办与每日例程</small></span></button>
      <button type="button" role="tab" aria-selected={section === "rhythm"} className={section === "rhythm" ? "active" : ""} onClick={() => setSection("rhythm")}><Compass size={16} /><span><strong>节奏与关注</strong><small>场景、推荐与主动联系</small></span></button>
    </div>
    {section === "today" ? <TodayDailyPanel navigate={props.navigate} /> : <RhythmView embedded />}
  </>;
}

function TodayDailyPanel(props: { navigate: (view: ViewId) => void }): React.ReactElement {
  const [type, setType] = useState("");
  const [candidate, setCandidate] = useState("");
  const [capturing, setCapturing] = useState(false);
  const [busy, setBusy] = useState("");
  const [actionError, setActionError] = useState("");
  const today = useMemo(() => {
    const now = new Date();
    const month = String(now.getMonth() + 1).padStart(2, "0");
    const day = String(now.getDate()).padStart(2, "0");
    return `${now.getFullYear()}-${month}-${day}`;
  }, []);
  const overview = useAsyncData(() => api<PersonalOverview>("/api/dashboard/control/personal/overview"), []);
  const daily = useAsyncData(() => api<PersonalTodayResponse>(`/api/dashboard/control/personal/today?local_date=${today}&timezone=Asia%2FShanghai`), [today]);
  const sources = useAsyncData(() => api<ExternalSourceRow[]>("/api/dashboard/control/sources"), []);

  const runRoutine = async (routine: "morning_brief" | "evening_review"): Promise<void> => {
    setBusy(routine); setActionError("");
    try {
      await api("/api/dashboard/control/personal/routines", { method: "POST", body: JSON.stringify({ routine, local_date: today, timezone: "Asia/Shanghai", chat_id: "xiaoman-console" }) });
      props.navigate("workflows");
    } catch (reason) { setActionError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(""); }
  };

  const capture = async (): Promise<void> => {
    if (!candidate.trim()) return;
    setBusy("capture"); setActionError("");
    try {
      await api("/api/dashboard/control/personal/routines", { method: "POST", body: JSON.stringify({ routine: "capture_commitment", candidate: candidate.trim(), timezone: "Asia/Shanghai", chat_id: "xiaoman-console" }) });
      setCandidate(""); setCapturing(false); props.navigate("workflows");
    } catch (reason) { setActionError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(""); }
  };

  const forget = async (record: PersonalRecordRow): Promise<void> => {
    if (!window.confirm(`确认遗忘“${record.title}”？内容和历史快照将被脱敏。`)) return;
    await api(`/api/dashboard/control/personal/records/${record.id}`, { method: "DELETE", body: JSON.stringify({ reason: "从我的一天工作台遗忘" }) });
    daily.reload(); overview.reload();
  };

  const syncSources = async (): Promise<void> => {
    const enabled = (sources.data ?? []).filter((source) => source.enabled);
    if (!enabled.length) return;
    setBusy("sync"); setActionError("");
    try {
      const results = await Promise.all(enabled.map((source) => api<{ error: string }>(`/api/dashboard/control/sources/${source.id}/sync`, { method: "POST" })));
      const failed = results.find((result) => result.error);
      if (failed?.error) throw new Error(failed.error);
      daily.reload(); sources.reload();
    } catch (reason) { setActionError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(""); }
  };

  const filters = ["", ...DAILY_RECORD_TYPES];
  const visibleRecords = (daily.data?.records ?? []).filter((record) => !type || record.entity_type === type);
  const activeSources = (sources.data ?? []).filter((source) => source.enabled);
  return <>
    <ErrorBanner message={overview.error || daily.error || sources.error || actionError} />
    <section className="routine-bar">
      <div className="routine-copy"><span className="routine-date">{today}</span><strong>今天从哪里开始？</strong><p>例程会进入任务中心，在需要判断或写入数据时停下来等你。</p></div>
      <div className="routine-actions">
        <button className="secondary-button" disabled={Boolean(busy)} onClick={() => void runRoutine("morning_brief")}><Sunrise size={16} />{busy === "morning_brief" ? "创建中" : "晨间简报"}</button>
        <button className="secondary-button" disabled={Boolean(busy)} onClick={() => void runRoutine("evening_review")}><MoonStar size={16} />{busy === "evening_review" ? "创建中" : "晚间回顾"}</button>
        <IconButton icon={RefreshCw} label="刷新今日数据" onClick={() => { overview.reload(); daily.reload(); sources.reload(); }} />
        <button className="primary-button" disabled={Boolean(busy)} onClick={() => setCapturing(true)}><Plus size={16} />记录待办</button>
      </div>
    </section>
    <section className="today-source-strip">
      <div><strong>数据来源</strong><span>{activeSources.length ? `${activeSources.length} 个订阅会自动汇入今天` : "当前只有小满本地记录"}</span></div>
      <div className="today-source-list">{activeSources.map((source) => <span key={source.id} className={source.last_error ? "error" : ""}><b>{source.name}</b>{source.last_error ? "同步异常" : source.last_synced_at ? `更新于 ${relativeTime(source.last_synced_at)}` : "等待首次同步"}</span>)}</div>
      {activeSources.length ? <button className="secondary-button" disabled={Boolean(busy)} onClick={() => void syncSources()}><RefreshCw size={15} className={busy === "sync" ? "spin" : ""} />{busy === "sync" ? "同步中" : "立即同步"}</button> : null}
    </section>
    <div className="personal-metrics">
      <div><span>今日关注</span><strong>{daily.data?.records.length ?? 0}</strong><small>只统计今天与逾期事项</small></div>
      <div><span>逾期待办</span><strong>{daily.data?.overdue_count ?? 0}</strong><small>需要优先确认</small></div>
      <div><span>外部来源</span><strong>{Object.keys(daily.data?.sources ?? {}).length}</strong><small>统一汇入个人事实库</small></div>
      <div><span>个人资料</span><strong>{overview.data?.profile_configured ? "已建立" : "待建立"}</strong><small>偏好与边界</small></div>
    </div>
    <div className="personal-toolbar">
      <div className="filter-tabs">{filters.map((value) => <button key={value} className={type === value ? "active" : ""} onClick={() => setType(value)}>{value ? personalTypeLabels[value] : "全部"}</button>)}</div>
    </div>
    {daily.loading && !daily.data ? <LoadingState /> : visibleRecords.length ? <div className="personal-records">{visibleRecords.map((record) => <div className="personal-record-row" key={record.id}>
      <span className={`personal-record-icon ${record.entity_type === "health_observation" ? "health" : ""}`}>{record.entity_type === "health_observation" ? <HeartPulse size={17} /> : record.entity_type === "daily_plan" ? <CalendarClock size={17} /> : <CheckCircle2 size={17} />}</span>
      <div className="personal-record-main"><div><strong>{record.title}</strong><Badge tone="blue">{personalTypeLabels[record.entity_type] ?? record.entity_type}</Badge><Badge>{sourceLabel(record.source)}</Badge>{record.user_locked ? <Badge tone="amber">已锁定</Badge> : null}</div><p>{record.summary || "暂无摘要"}</p><small>更新于 {relativeTime(record.updated_at)}</small></div>
      <div className="record-lifecycle"><IconButton icon={Trash2} label="遗忘记录" danger onClick={() => void forget(record)} /></div>
    </div>)}</div> : <EmptyState icon={Sunrise} title="今天还没有安排" text="运行晨间简报、记录待办，或连接 Notion 等外部数据源。" />}
    {capturing ? <Modal title="记录待办" description="小满会先帮你整理成可执行事项，再在任务中心等你确认。" onClose={() => setCapturing(false)}><div className="form-stack"><label>你准备完成什么？<textarea rows={5} value={candidate} onChange={(event) => setCandidate(event.target.value)} placeholder="例如：下周五前完成个人助手架构报告" /></label><div className="dialog-actions"><button className="secondary-button" onClick={() => setCapturing(false)}>取消</button><button className="primary-button" disabled={!candidate.trim() || busy === "capture"} onClick={() => void capture()}><Workflow size={16} />{busy === "capture" ? "创建中" : "进入确认流程"}</button></div></div></Modal> : null}
  </>;
}

function sourceLabel(source: string): string {
  if (source === "notion") return "Notion";
  if (source === "dashboard") return "小满";
  if (source === "conversation") return "对话";
  return source || "本地";
}
