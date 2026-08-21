import React, { useState } from "react";
import { CalendarClock, Clock3, Pencil, Plus, Trash2 } from "lucide-react";
import { api } from "../../api";
import { shortTs } from "../../format";
import { Badge, EmptyState, ErrorBanner, IconButton, LoadingState, Modal, PageIntro } from "../../shared/components/ui";
import { useAsyncData } from "../../shared/hooks/useAsyncData";
import type { ScheduleRow } from "../../shared/types";

const TRIGGER_LABELS: Record<string, string> = {
  after: "一次提醒",
  at: "指定时间",
  every: "循环",
};

const DEFAULT_FORM = {
  name: "",
  when: "1h",
  message: "记得喝水",
  channel: "dashboard",
  chat_id: "xiaoman-reminders",
  trigger: "after",
  tier: "instant",
};

function localDateTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60_000);
  return local.toISOString().slice(0, 16);
}

function editForm(job: ScheduleRow): typeof DEFAULT_FORM {
  const recurring = job.trigger === "every";
  return {
    name: job.name ?? "",
    when: recurring
      ? job.cron_expr || `${job.interval_seconds || 86400}s`
      : localDateTime(job.fire_at),
    message: job.message || job.prompt || "",
    channel: job.channel || "dashboard",
    chat_id: job.chat_id || "xiaoman-reminders",
    trigger: recurring ? "every" : "at",
    tier: job.tier || "instant",
  };
}

export function SchedulesView(): React.ReactElement {
  const resource = useAsyncData(() => api<ScheduleRow[]>("/api/dashboard/control/schedules"), []);
  const [adding, setAdding] = useState(false);
  const [busy, setBusy] = useState("");
  const [actionError, setActionError] = useState("");
  const [editing, setEditing] = useState<ScheduleRow | null>(null);
  const [form, setForm] = useState(DEFAULT_FORM);

  const save = async (): Promise<void> => {
    setBusy("save");
    setActionError("");
    try {
      await api(editing ? `/api/dashboard/control/schedules/${editing.id}` : "/api/dashboard/control/schedules", {
        method: editing ? "PATCH" : "POST",
        body: JSON.stringify({
          ...form,
          timezone: "Asia/Shanghai",
          request_time: new Date().toISOString(),
          prompt: form.tier === "soft" ? form.message : null,
          message: form.tier === "instant" ? form.message : null,
        }),
      });
      setAdding(false);
      setEditing(null);
      setForm(DEFAULT_FORM);
      resource.reload();
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy("");
    }
  };
  const openAdd = (): void => { setEditing(null); setForm(DEFAULT_FORM); setAdding(true); };
  const openEdit = (job: ScheduleRow): void => { setEditing(job); setForm(editForm(job)); setAdding(true); };
  const closeEditor = (): void => { setAdding(false); setEditing(null); };
  const remove = async (job: ScheduleRow): Promise<void> => {
    if (!window.confirm(`确认取消“${job.name || "这项定时任务"}”？`)) return;
    setBusy(job.id);
    setActionError("");
    try {
      await api(`/api/dashboard/control/schedules/${job.id}`, { method: "DELETE" });
      resource.reload();
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy("");
    }
  };
  const changeTrigger = (trigger: string): void => {
    setForm((current) => ({
      ...current,
      trigger,
      when: trigger === "after" ? "1h" : trigger === "at" ? "21:30" : "1d",
    }));
  };
  const whenLabel = form.trigger === "after" ? "多久后" : form.trigger === "at" ? "执行时间" : "循环频率";
  const whenPlaceholder = form.trigger === "after"
    ? "例如 30m、2h、1d"
    : form.trigger === "at"
      ? "例如 21:30 或 2026-07-15T09:00"
      : "例如 1d 或 0 9 * * *";

  return <>
    <PageIntro
      title="定时任务"
      description="普通提醒会按原文送达；需要小满临时整理的内容会在触发时生成。"
      actions={<button className="primary-button" onClick={openAdd}><Plus size={16} />新建任务</button>}
    />
    <ErrorBanner message={resource.error || actionError} />
    {resource.data?.length ? (
      <div className="resource-list">
        {resource.data.map((job) => (
          <div className="resource-row" key={job.id}>
            <span className="resource-icon amber"><Clock3 size={18} /></span>
            <div className="resource-main">
              <div>
                <strong>{job.name || "未命名提醒"}</strong>
                <Badge tone={job.tier === "soft" ? "blue" : "green"}>{job.tier === "soft" ? "AI 生成" : "固定消息"}</Badge>
                <Badge>{TRIGGER_LABELS[job.trigger] ?? job.trigger}</Badge>
                {job.last_status === "failed" ? <Badge tone="amber">发送失败</Badge> : null}
              </div>
              <p>{job.message || job.prompt || "无内容"}</p>
              {job.last_error ? <small className="field-error">{job.last_error}</small> : null}
            </div>
            <div className="schedule-time"><strong>{job.enabled === false ? "等待处理" : shortTs(job.fire_at)}</strong><small>成功 {job.run_count} 次{job.last_attempt_at ? ` · 最近 ${shortTs(job.last_attempt_at)}` : ""}</small></div>
            <IconButton icon={Pencil} label="修改" disabled={busy === job.id} onClick={() => openEdit(job)} />
            <IconButton icon={Trash2} label={job.enabled === false ? "删除" : "取消"} danger disabled={busy === job.id} onClick={() => void remove(job)} />
          </div>
        ))}
      </div>
    ) : !resource.loading ? (
      <EmptyState icon={CalendarClock} title="没有待执行任务" text="创建喝水提醒、睡前总结或每日计划回顾。" />
    ) : <LoadingState />}
    {adding ? (
      <Modal title={editing ? "修改定时任务" : "新建定时任务"} description="选择直接发送固定内容，或让小满在触发时根据当时情况生成。" onClose={closeEditor}>
        <div className="form-stack form-two">
          <label>任务名称<input value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} placeholder="例如 每日计划回顾" /></label>
          <label>提醒内容<select value={form.tier} onChange={(event) => setForm({ ...form, tier: event.target.value })}><option value="instant">发送固定消息</option><option value="soft">由小满实时生成</option></select></label>
          <label>触发方式<select value={form.trigger} onChange={(event) => changeTrigger(event.target.value)}><option value="after">过一段时间</option><option value="at">指定时间</option><option value="every">循环提醒</option></select></label>
          <label>{whenLabel}<input value={form.when} onChange={(event) => setForm({ ...form, when: event.target.value })} placeholder={whenPlaceholder} /></label>
          <label className="span-two">{form.tier === "soft" ? "希望小满生成什么" : "提醒内容"}<textarea value={form.message} onChange={(event) => setForm({ ...form, message: event.target.value })} rows={3} placeholder={form.tier === "soft" ? "例如：根据今天的待办给我一份简短回顾" : "例如：记得喝水"} /></label>
          <div className="dialog-actions span-two">
            <button className="secondary-button" onClick={closeEditor}>取消</button>
            <button className="primary-button" disabled={busy === "save" || !form.when.trim() || !form.message.trim()} onClick={() => void save()}><CalendarClock size={16} />{busy === "save" ? "保存中" : editing ? "保存修改" : "创建"}</button>
          </div>
        </div>
      </Modal>
    ) : null}
  </>;
}
