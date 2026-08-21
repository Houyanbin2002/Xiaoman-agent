import React, { useMemo, useState } from "react";
import {
  Activity,
  BellRing,
  BrainCircuit,
  CalendarClock,
  Check,
  ChevronRight,
  History,
  Pause,
  Play,
  RefreshCw,
  Route,
  Settings2,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { api, asPageResult } from "../../api";
import { relativeTime, shortTs, stripMarkdown } from "../../format";
import { Badge, EmptyState, ErrorBanner, IconButton, LoadingState, Modal, PageIntro } from "../../shared/components/ui";
import { useAsyncData } from "../../shared/hooks/useAsyncData";
import { restartGatewayAndWait } from "../../shared/gateway";
import type {
  AttentionActionPlanRow,
  AttentionEngineOverview,
  AttentionEventRow,
  AttentionObservationRow,
  AttentionPatternRow,
  AttentionPolicyRow,
  AttentionSignalRow,
  AttentionWakeRow,
} from "../../shared/types";
import type { PageResult, ProactiveOverview, ProactiveTick } from "../../types";

const dayLabels: Record<string, string> = { mon: "一", tue: "二", wed: "三", thu: "四", fri: "五", sat: "六", sun: "日" };

function patternSchedule(pattern: AttentionPatternRow): string {
  const days = pattern.recurrence.days.map((day) => dayLabels[day] ?? day).join("、");
  return `周${days} ${pattern.recurrence.start}–${pattern.recurrence.end}`;
}

function planStatusLabel(status: string): string {
  return ({ proposed: "待执行", pending_approval: "等你确认", approved: "已确认", executing: "进行中", succeeded: "已联系", skipped: "保持安静", deferred: "稍后再看", expired: "已过期", failed: "未完成" } as Record<string, string>)[status] ?? status;
}

function planActionLabel(capabilityId: string): string {
  if (capabilityId === "message.notify") return "主动联系你";
  if (capabilityId.includes("recommend")) return "为你推荐内容";
  if (capabilityId.includes("task")) return "协助处理任务";
  return capabilityId.replace(/[._-]+/g, " ");
}

function planReasonLabel(reason: string, capabilityId: string): string {
  const technical = reason.match(/^[a-z_]+机会窗口有效；(\d+)\s*个可信信号可由\s*(.+?)\s*处理$/i);
  if (technical) {
    return `当前时机合适，有 ${technical[1]} 项可信变化值得${planActionLabel(capabilityId)}。`;
  }
  return reason
    .replace(/\bneutral\b/gi, "普通")
    .replace(/机会窗口有效/g, "当前时机合适")
    .replace(/主动消息/g, "主动联系");
}

function evaluationLabel(reason: string): string {
  return ({
    action_planned: "小满发现了一件值得主动协助的事。",
    no_active_opportunity: "现在不是合适的联系时机，小满会保持安静。",
    no_compatible_action: "目前没有适合处理这件事的能力。",
    no_new_action: "值得关注的内容都已经处理过了。",
    all_actions_filtered: "这次没有足够价值，或可能造成打扰。",
  } as Record<string, string>)[reason] ?? reason;
}

const policyFieldLabels: Record<string, string> = {
  domain: "领域", action_type: "动作", capability_id: "能力", risk: "风险", scene: "场景", channel: "渠道",
  focus_active: "专注中", do_not_disturb: "免打扰", severity_min: "最低重要度", severity_max: "最高重要度", confidence_min: "最低可信度",
};

function readableValue(value: unknown): string {
  if (Array.isArray(value)) return value.map(readableValue).join("、");
  if (typeof value === "boolean") return value ? "是" : "否";
  return String(value);
}

function policyConditions(policy: AttentionPolicyRow): string {
  const fields = { ...policy.scope, ...policy.conditions };
  const summary = Object.entries(fields).map(([key, value]) => {
    const label = key.startsWith("attribute.") ? key.slice(10) : (policyFieldLabels[key] ?? key);
    return `${label}：${readableValue(value)}`;
  });
  return summary.length ? summary.join(" · ") : "适用于所有主动协助";
}

function policyTitle(policy: AttentionPolicyRow): string {
  if (policy.effect === "deny") return "符合条件时保持安静";
  if (policy.effect === "require_approval") return "行动前先询问你";
  if (policy.effect === "defer") return "符合条件时稍后处理";
  if (policy.effect === "limit_frequency") return "减少主动出现频率";
  if (policy.effect === "adjust_score") return `${policy.score_adjustment < 0 ? "降低" : "提高"}主动优先级 ${Math.round(Math.abs(policy.score_adjustment) * 100)}%`;
  return policy.effect;
}

function evidenceSource(source: string): string {
  return ({ conversation: "对话", personal_record: "个人记录", mcp: "外部数据" } as Record<string, string>)[source] ?? source;
}

function deliveryLabel(value: AttentionEventRow["delivery_semantics"]): string {
  return ({ exact: "按时提醒", before_deadline: "截止前留意", opportunistic: "合适时联系", silent: "只记住" } as Record<string, string>)[value] ?? value;
}

function channelLabel(value: string): string {
  return ({ qqbot: "QQ", weixin: "微信", wecom: "企业微信", telegram: "Telegram" } as Record<string, string>)[value.toLowerCase()] ?? value;
}

function toneForPlan(status: string): "green" | "red" | "amber" | "blue" | "gray" {
  if (status === "succeeded") return "green";
  if (status === "failed") return "red";
  if (status === "pending_approval") return "amber";
  if (["skipped", "expired", "deferred"].includes(status)) return "gray";
  return "blue";
}

export function ProactiveView(): React.ReactElement {
  const base = "/api/dashboard/control/attention";
  const [busy, setBusy] = useState("");
  const [actionError, setActionError] = useState("");
  const [evaluationNote, setEvaluationNote] = useState("");
  const [runtimeEditor, setRuntimeEditor] = useState<{ enabled: boolean; channel: string; chatId: string } | null>(null);
  const [restartConfirm, setRestartConfirm] = useState(false);
  const overview = useAsyncData(() => api<AttentionEngineOverview>(`${base}/overview`), []);
  const events = useAsyncData(() => api<AttentionEventRow[]>(`${base}/events?limit=50`), []);
  const wakes = useAsyncData(() => api<AttentionWakeRow[]>(`${base}/wakes?limit=50`), []);
  const signals = useAsyncData(() => api<AttentionSignalRow[]>(`${base}/signals?limit=50`), []);
  const patterns = useAsyncData(() => api<AttentionPatternRow[]>(`${base}/patterns`), []);
  const plans = useAsyncData(() => api<AttentionActionPlanRow[]>(`${base}/plans?limit=20`), []);
  const policies = useAsyncData(() => api<AttentionPolicyRow[]>(`${base}/policies`), []);
  const observations = useAsyncData(() => api<AttentionObservationRow[]>(`${base}/observations?limit=30`), []);
  const runtimeOverview = useAsyncData(() => api<ProactiveOverview>("/api/dashboard/proactive/overview"), []);
  const ticks = useAsyncData(async () => asPageResult(await api<PageResult<ProactiveTick>>("/api/dashboard/proactive/tick_logs?page_size=50")), []);

  const pendingPlans = useMemo(() => (plans.data ?? []).filter((plan) => plan.status === "pending_approval"), [plans.data]);
  const recentPlans = useMemo(() => (plans.data ?? []).filter((plan) => plan.status !== "pending_approval").slice(0, 8), [plans.data]);
  const activePatterns = useMemo(() => (patterns.data ?? []).filter((pattern) => pattern.status === "active"), [patterns.data]);
  const suspendedPatterns = useMemo(() => (patterns.data ?? []).filter((pattern) => pattern.status === "suspended"), [patterns.data]);
  const reviewPatterns = useMemo(() => (patterns.data ?? []).filter((pattern) => pattern.status === "proposed"), [patterns.data]);
  const reviewPolicies = useMemo(() => (policies.data ?? []).filter((policy) => policy.status === "proposed"), [policies.data]);
  const confirmationCount = pendingPlans.length + reviewPatterns.length + reviewPolicies.length;

  const refresh = (): void => {
    overview.reload(); events.reload(); wakes.reload(); signals.reload(); patterns.reload(); plans.reload(); policies.reload(); observations.reload(); runtimeOverview.reload(); ticks.reload();
    setEvaluationNote("已刷新，小满会用最新信息继续留意。");
  };

  const restartGateway = async (): Promise<void> => {
    setRestartConfirm(false); setBusy("restart"); setActionError(""); setEvaluationNote("正在重新连接，小满会短暂离线……");
    try {
      await restartGatewayAndWait(); refresh();
      setEvaluationNote("连接已恢复，新的接收设置已经生效。");
    } catch (reason) {
      setEvaluationNote(""); setActionError(reason instanceof Error ? reason.message : "重启失败，请稍后重试");
    } finally { setBusy(""); }
  };

  const evaluate = async (): Promise<void> => {
    setBusy("evaluate"); setActionError(""); setEvaluationNote("");
    try {
      const result = await api<{ reason: string; candidate_count: number; denied_count: number; below_threshold_count: number }>(`${base}/evaluate`, { method: "POST" });
      setEvaluationNote(evaluationLabel(result.reason));
      overview.reload(); events.reload(); wakes.reload(); signals.reload(); patterns.reload(); plans.reload();
    } catch (reason) { setActionError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(""); }
  };

  const openRuntimeEditor = (): void => {
    const data = overview.data;
    setRuntimeEditor({ enabled: data?.runtime_enabled ?? false, channel: data?.target_channel || "qqbot", chatId: data?.target_chat_id || "" });
  };

  const saveRuntime = async (): Promise<void> => {
    if (!runtimeEditor) return;
    setBusy("runtime"); setActionError("");
    try {
      await api(`${base}/runtime`, { method: "PATCH", body: JSON.stringify({ enabled: runtimeEditor.enabled, channel: runtimeEditor.channel, chat_id: runtimeEditor.chatId }) });
      setRuntimeEditor(null); setEvaluationNote("主动联系设置已保存，重启连接后生效。"); overview.reload();
    } catch (reason) { setActionError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(""); }
  };

  const setPatternStatus = async (pattern: AttentionPatternRow, status: "active" | "suspended" | "rejected"): Promise<void> => {
    setBusy(pattern.id); setActionError("");
    try { await api(`${base}/patterns/${pattern.id}/status`, { method: "PATCH", body: JSON.stringify({ status }) }); patterns.reload(); overview.reload(); }
    catch (reason) { setActionError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(""); }
  };

  const recordPlanFeedback = async (plan: AttentionActionPlanRow, kind: "accepted" | "wrong_time" | "disliked" | "too_frequent" | "inaccurate"): Promise<void> => {
    setBusy(plan.id); setActionError("");
    try { await api(`${base}/plans/${plan.id}/feedback`, { method: "POST", body: JSON.stringify({ kind }) }); plans.reload(); patterns.reload(); }
    catch (reason) { setActionError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(""); }
  };

  const decidePlan = async (plan: AttentionActionPlanRow, decision: "approve" | "skip"): Promise<void> => {
    setBusy(plan.id); setActionError("");
    try { await api(`${base}/plans/${plan.id}/${decision}`, { method: "POST" }); plans.reload(); overview.reload(); }
    catch (reason) { setActionError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(""); }
  };

  const setPolicyStatus = async (policy: AttentionPolicyRow, status: "active" | "suspended" | "rejected"): Promise<void> => {
    setBusy(policy.id); setActionError("");
    try { await api(`${base}/policies/${policy.id}/status`, { method: "PATCH", body: JSON.stringify({ status }) }); policies.reload(); overview.reload(); }
    catch (reason) { setActionError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(""); }
  };

  const allError = overview.error || events.error || wakes.error || signals.error || patterns.error || plans.error || policies.error || observations.error || runtimeOverview.error || ticks.error || actionError;
  const isReady = Boolean(overview.data?.runtime_enabled && overview.data?.target_configured);
  const targetLabel = overview.data?.target_channel ? `${channelLabel(overview.data.target_channel)} · ${overview.data.target_chat_id || "未选账号"}` : "还未选择联系渠道";

  return <div className="proactive-product-page">
    <PageIntro
      title="主动协助"
      description="小满会结合你的安排、状态和长期习惯，在真正值得、时机合适时主动找你。"
      actions={<><button className="secondary-button" onClick={openRuntimeEditor}><Settings2 size={16} />联系设置</button><IconButton icon={RefreshCw} label="刷新主动协助" onClick={refresh} /></>}
    />
    <ErrorBanner message={allError} />
    {evaluationNote ? <div className="proactive-feedback"><Sparkles size={16} />{evaluationNote}</div> : null}

    <section className={`proactive-hero ${isReady ? "ready" : "paused"}`}>
      <div className="proactive-hero-mark"><BellRing size={25} /></div>
      <div className="proactive-hero-copy">
        <span>{isReady ? "主动协助已开启" : "主动协助还未就绪"}</span>
        <h2>{isReady ? "小满正在安静地替你留意" : "完成接收设置后，小满才能主动找到你"}</h2>
        <p>{targetLabel}{overview.data?.next_wake_at ? ` · 下次关注 ${shortTs(overview.data.next_wake_at)}` : " · 没有重要变化时不会打扰你"}</p>
      </div>
      <div className="proactive-hero-stats"><div><strong>{overview.data?.active_events ?? 0}</strong><span>正在留意</span></div><div><strong>{confirmationCount}</strong><span>等你确认</span></div></div>
    </section>

    {overview.data?.provider_failures.length ? <div className="proactive-provider-note"><Activity size={15} />有 {overview.data.provider_failures.length} 个信息来源暂时读取失败，其他来源仍在正常工作。</div> : null}

    <section className="proactive-section">
      <div className="proactive-section-head"><div><span className="section-kicker">此刻</span><h2>正在留意</h2><p>这些事情可能在合适的时候需要提醒、建议或行动。</p></div><Badge tone="blue">{(events.data?.length ?? 0) + activePatterns.length} 项</Badge></div>
      <div className="proactive-watch-grid">
        <div className="proactive-watch-list">
          {events.loading && !events.data ? <LoadingState /> : events.data?.length ? events.data.slice(0, 8).map((event) => <article className="proactive-item" key={event.id}>
            <span className="proactive-item-icon"><Activity size={17} /></span>
            <div><div className="proactive-item-title"><strong>{event.entity?.title || event.kind}</strong><Badge tone={event.delivery_semantics === "exact" ? "blue" : event.urgency >= 0.8 ? "red" : "amber"}>{deliveryLabel(event.delivery_semantics)}</Badge></div><p>{event.due_at ? `${shortTs(event.due_at)} 前后需要关注` : "等待合适的时机"}</p><small>可信度 {Math.round(event.confidence * 100)}%{event.entity?.local_override.source_sync === "pending" ? " · 等待同步回来源" : ""}</small></div>
          </article>) : <EmptyState icon={Activity} title="现在没有需要特别留意的事" text="新的待办、日程、情绪变化或外部数据出现后，小满会把真正重要的内容放到这里。" />}
        </div>
        <aside className="proactive-window-card">
          <div><CalendarClock size={18} /><span>合适的联系时机</span></div>
          {activePatterns.length ? activePatterns.slice(0, 4).map((pattern) => <article key={pattern.id}><strong>{pattern.scene}</strong><p>{patternSchedule(pattern)} · 约 {pattern.available_minutes} 分钟</p><button type="button" disabled={busy === pattern.id} onClick={() => void setPatternStatus(pattern, "suspended")}><Pause size={13} />暂停</button></article>) : <div className="proactive-window-empty"><p>还没有稳定的机会窗口。</p><small>你可以直接告诉小满“工作日八点通勤约二十分钟”，也可以让他从多次记录中逐步学习。</small></div>}
        </aside>
      </div>
    </section>

    <section className="proactive-section">
      <div className="proactive-section-head"><div><span className="section-kicker">确认</span><h2>需要你确认</h2><p>小满推断出的习惯和有外部影响的行动，不会悄悄替你决定。</p></div>{confirmationCount ? <Badge tone="amber">{confirmationCount} 项</Badge> : null}</div>
      {confirmationCount ? <div className="proactive-confirm-list">
        {pendingPlans.map((plan) => <article key={plan.id}><span><Route size={18} /></span><div><strong>{planActionLabel(plan.capability_id)}</strong><p>{planReasonLabel(plan.decision_reason, plan.capability_id)}</p><small>{relativeTime(plan.created_at)} · 判断价值 {Math.round(plan.score * 100)}%</small></div><div className="proactive-confirm-actions"><button className="primary-button" disabled={busy === plan.id} onClick={() => void decidePlan(plan, "approve")}><Check size={14} />同意这次</button><button className="secondary-button" disabled={busy === plan.id} onClick={() => void decidePlan(plan, "skip")}>这次不做</button></div></article>)}
        {reviewPatterns.map((pattern) => <article key={pattern.id}><span><CalendarClock size={18} /></span><div><strong>{pattern.scene}</strong><p>{patternSchedule(pattern)} · 可用约 {pattern.available_minutes} 分钟</p><small>小满根据 {pattern.observation_count} 次观察提出 · 可信度 {Math.round(pattern.confidence * 100)}%</small></div><div className="proactive-confirm-actions"><button className="primary-button" disabled={busy === pattern.id} onClick={() => void setPatternStatus(pattern, "active")}><Check size={14} />确认规律</button><button className="secondary-button" disabled={busy === pattern.id} onClick={() => void setPatternStatus(pattern, "rejected")}>不符合</button></div></article>)}
        {reviewPolicies.map((policy) => <article key={policy.id}><span><ShieldCheck size={18} /></span><div><strong>{policyTitle(policy)}</strong><p>{policyConditions(policy)}</p><small>小满推断 · 可信度 {Math.round(policy.confidence * 100)}%</small></div><div className="proactive-confirm-actions"><button className="primary-button" disabled={busy === policy.id} onClick={() => void setPolicyStatus(policy, "active")}><Check size={14} />采用</button><button className="secondary-button" disabled={busy === policy.id} onClick={() => void setPolicyStatus(policy, "rejected")}>忽略</button></div></article>)}
      </div> : <div className="proactive-calm"><Check size={19} /><div><strong>目前没有需要你处理的确认</strong><p>小满会自行保持安静；只有新的规律或需要授权的行动出现时才会放到这里。</p></div></div>}
    </section>

    <section className="proactive-section">
      <div className="proactive-section-head"><div><span className="section-kicker">记录</span><h2>联系记录</h2><p>查看小满为什么联系你或为什么选择保持安静，并用反馈帮助他逐步适应你。</p></div></div>
      {plans.loading && !plans.data ? <LoadingState /> : recentPlans.length ? <div className="proactive-history-list">{recentPlans.map((plan) => <article key={plan.id}>
        <span className={`proactive-history-dot ${plan.status}`} />
        <div><div><strong>{planActionLabel(plan.capability_id)}</strong><Badge tone={toneForPlan(plan.status)}>{planStatusLabel(plan.status)}</Badge></div><p>{planReasonLabel(plan.decision_reason, plan.capability_id)}</p><small>{relativeTime(plan.created_at)}</small>{plan.status === "succeeded" ? <div className="proactive-feedback-actions"><button disabled={busy === plan.id} onClick={() => void recordPlanFeedback(plan, "accepted")}>有帮助</button><button disabled={busy === plan.id} onClick={() => void recordPlanFeedback(plan, "wrong_time")}>时间不对</button><button disabled={busy === plan.id} onClick={() => void recordPlanFeedback(plan, "too_frequent")}>太频繁</button><button disabled={busy === plan.id} onClick={() => void recordPlanFeedback(plan, "disliked")}>不感兴趣</button><button disabled={busy === plan.id} onClick={() => void recordPlanFeedback(plan, "inaccurate")}>判断不准</button></div> : null}</div>
      </article>)}</div> : <EmptyState icon={History} title="还没有主动联系记录" text="当小满第一次主动协助或有意保持安静后，原因会记录在这里。" />}
    </section>

    <details className="proactive-advanced">
      <summary><span><Settings2 size={17} />高级信息与维护</span><small>信号、学习证据、运行记录和连接维护</small><ChevronRight size={16} /></summary>
      <div className="proactive-advanced-actions"><button className="secondary-button" disabled={busy === "evaluate"} onClick={() => void evaluate()}><Activity size={15} />{busy === "evaluate" ? "判断中" : "立即运行一次判断"}</button><button className="secondary-button" disabled={busy === "restart"} onClick={() => setRestartConfirm(true)}><RefreshCw size={15} />{busy === "restart" ? "正在重启" : "重启连接服务"}</button></div>
      <div className="proactive-advanced-grid">
        <div><h3>当前信号</h3>{signals.data?.length ? signals.data.slice(0, 10).map((signal) => <p key={signal.id}><Activity size={13} /><span>{signal.summary}</span><small>{signal.source.name} · {relativeTime(signal.occurred_at)}</small></p>) : <small>暂无活跃信号</small>}</div>
        <div><h3>学习证据</h3>{observations.data?.length ? observations.data.slice(0, 10).map((item) => <p key={item.id}><BrainCircuit size={13} /><span>{item.statement}</span><small>{evidenceSource(item.source_type)} · {Math.round(item.confidence * 100)}%</small></p>) : <small>暂无学习证据</small>}</div>
        <div><h3>等待唤醒</h3>{wakes.data?.length ? wakes.data.slice(0, 10).map((wake) => <p key={wake.id}><CalendarClock size={13} /><span>{wake.entity?.title || wake.event?.kind || "待判断事项"}</span><small>{shortTs(wake.wake_at)} · {wake.reason}</small></p>) : <small>当前无需定时唤醒</small>}</div>
        <div><h3>暂停的规律</h3>{suspendedPatterns.length ? suspendedPatterns.map((pattern) => <p key={pattern.id}><Pause size={13} /><span>{pattern.scene}</span><button disabled={busy === pattern.id} onClick={() => void setPatternStatus(pattern, "active")}><Play size={12} />恢复</button></p>) : <small>没有暂停的规律</small>}</div>
      </div>
      <div className="proactive-runtime-log"><h3>最近运行记录</h3>{ticks.data?.items.slice(0, 12).map((tick) => <p key={tick.tick_id}><span>{shortTs(tick.started_at)}</span><Badge tone={tick.terminal_action === "reply" ? "green" : "gray"}>{tick.terminal_action === "reply" ? "已联系" : "保持安静"}</Badge><small>{stripMarkdown(tick.final_message || tick.skip_reason || "无摘要")}</small></p>)}</div>
    </details>

    {runtimeEditor ? <Modal title="小满怎样主动联系你" description="选择小满主动找你时使用的渠道和账号。" onClose={() => setRuntimeEditor(null)}><div className="form-stack">
      <label className="check-label"><input type="checkbox" checked={runtimeEditor.enabled} onChange={(event) => setRuntimeEditor({ ...runtimeEditor, enabled: event.target.checked })} />允许小满主动联系我</label>
      <label>联系渠道<select value={runtimeEditor.channel} onChange={(event) => setRuntimeEditor({ ...runtimeEditor, channel: event.target.value })}><option value="qqbot">QQ</option><option value="weixin">微信</option><option value="wecom">企业微信</option><option value="telegram">Telegram</option></select></label>
      <label>接收账号<input value={runtimeEditor.chatId} onChange={(event) => setRuntimeEditor({ ...runtimeEditor, chatId: event.target.value })} list="attention-target-options" placeholder="选择或填写自己的账号" /></label>
      <datalist id="attention-target-options">{overview.data?.available_targets.filter((item) => item.channel === runtimeEditor.channel).map((item) => <option key={`${item.channel}:${item.chat_id}`} value={item.chat_id} />)}</datalist>
      <small>如果没有可选账号，请先在“设置与扩展 → 联系小满”中完成渠道连接。</small>
      <div className="dialog-actions"><button className="secondary-button" onClick={() => setRuntimeEditor(null)}>取消</button><button className="primary-button" disabled={busy === "runtime"} onClick={() => void saveRuntime()}>{busy === "runtime" ? "保存中" : "保存设置"}</button></div>
    </div></Modal> : null}
    {restartConfirm ? <Modal title="重新连接小满" description="重新加载主动联系设置和 QQ、微信等渠道。" onClose={() => setRestartConfirm(false)}><div className="form-stack"><div className="gateway-restart-warning"><RefreshCw size={18} /><div><strong>连接会短暂中断</strong><span>正在生成的回复会停止；聊天记录、任务和配置不会丢失。</span></div></div><div className="dialog-actions"><button type="button" className="secondary-button" onClick={() => setRestartConfirm(false)}>取消</button><button type="button" className="primary-button" onClick={() => void restartGateway()}><RefreshCw size={15} />立即重启</button></div></div></Modal> : null}
  </div>;
}
