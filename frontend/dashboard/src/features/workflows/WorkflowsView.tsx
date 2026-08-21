import React, { useState } from "react";
import { Ban, CheckCircle2, CircleAlert, RefreshCw, RotateCcw, Send, Trash2, Workflow } from "lucide-react";
import { api } from "../../api";
import { relativeTime } from "../../format";
import { Badge, EmptyState, ErrorBanner, IconButton, LoadingState, PageIntro } from "../../shared/components/ui";
import { useAsyncData } from "../../shared/hooks/useAsyncData";
import type { WorkflowRow } from "../../shared/types";

const STATUS_LABELS: Record<string, string> = {
  draft: "准备中",
  running: "运行中",
  waiting: "等待你",
  blocked: "受阻",
  succeeded: "已完成",
  failed: "失败",
  cancelled: "已取消",
};

function statusTone(status: string): "green" | "blue" | "amber" | "red" | "gray" {
  if (status === "succeeded") return "green";
  if (status === "running") return "blue";
  if (status === "waiting") return "amber";
  if (status === "blocked" || status === "failed") return "red";
  return "gray";
}

interface WorkflowActionsProps {
  workflow: WorkflowRow;
  busy: boolean;
  responses: Record<string, string>;
  setResponse: (stepId: string, value: string) => void;
  approve: (stepId: string, approved: boolean) => void;
  respond: (stepId: string) => void;
  retry: (stepId: string) => void;
}

function WorkflowActions(props: WorkflowActionsProps): React.ReactElement {
  return <>
    {(props.workflow.waiting_actions ?? []).map((step) => {
      const responseKey = `${props.workflow.id}:${step.id}`;
      return (
        <section className="workflow-action-panel" key={step.id}>
          <div><strong>{step.title}</strong><p>{step.description}</p></div>
          {step.kind === "approval" ? (
            <div className="workflow-action-buttons">
              <button className="secondary-button" disabled={props.busy} onClick={() => props.approve(step.id, false)}>拒绝</button>
              <button className="primary-button" disabled={props.busy} onClick={() => props.approve(step.id, true)}>批准并继续</button>
            </div>
          ) : (
            <div className="workflow-response">
              <textarea rows={2} value={props.responses[responseKey] ?? ""} onChange={(event) => props.setResponse(step.id, event.target.value)} placeholder="补充小满继续执行所需的信息" />
              <button className="primary-button" disabled={props.busy || !(props.responses[responseKey] ?? "").trim()} onClick={() => props.respond(step.id)}><Send size={14} />提交并继续</button>
            </div>
          )}
        </section>
      );
    })}
    {(props.workflow.failed_actions ?? []).map((step) => (
      <section className="workflow-action-panel failed" key={step.id}>
        <div><strong>{step.title}</strong><p>{step.error || step.description || "这一步执行失败，可以重新尝试。"}</p></div>
        <button className="secondary-button" disabled={props.busy} onClick={() => props.retry(step.id)}><RotateCcw size={14} />重试这一步</button>
      </section>
    ))}
  </>;
}

export function WorkflowsView(): React.ReactElement {
  const resource = useAsyncData(() => api<WorkflowRow[]>("/api/dashboard/control/tasks"), []);
  const [responses, setResponses] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState("");
  const [actionError, setActionError] = useState("");

  const runAction = async (key: string, action: () => Promise<unknown>): Promise<void> => {
    setBusy(key);
    setActionError("");
    try {
      await action();
      resource.reload();
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy("");
    }
  };
  const cancel = async (workflow: WorkflowRow): Promise<void> => {
    if (!window.confirm(`确认停止“${workflow.name}”？已经完成的步骤会保留，但任务不会继续运行。`)) return;
    await runAction(`cancel:${workflow.id}`, () => api(`/api/dashboard/control/tasks/${workflow.id}/cancel`, {
      method: "POST",
      body: JSON.stringify({ note: "从任务中心取消" }),
    }));
  };
  const approve = async (workflow: WorkflowRow, stepId: string, approved: boolean): Promise<void> => {
    await runAction(`${approved ? "approve" : "reject"}:${workflow.id}:${stepId}`, () => api(`/api/dashboard/control/tasks/${workflow.id}/approval`, {
      method: "POST",
      body: JSON.stringify({ step_id: stepId, approved, note: approved ? "从任务中心批准" : "从任务中心拒绝" }),
    }));
  };
  const respond = async (workflow: WorkflowRow, stepId: string): Promise<void> => {
    const response = (responses[`${workflow.id}:${stepId}`] ?? "").trim();
    if (!response) return;
    await runAction(`respond:${workflow.id}:${stepId}`, () => api(`/api/dashboard/control/tasks/${workflow.id}/respond`, {
      method: "POST",
      body: JSON.stringify({ step_id: stepId, response }),
    }));
  };
  const retry = async (workflow: WorkflowRow, stepId: string): Promise<void> => {
    await runAction(`retry:${workflow.id}:${stepId}`, () => api(`/api/dashboard/control/tasks/${workflow.id}/retry`, {
      method: "POST",
      body: JSON.stringify({ step_id: stepId, note: "从任务中心重试" }),
    }));
  };
  const removeHistory = async (workflow: WorkflowRow): Promise<void> => {
    if (!window.confirm(`确认删除“${workflow.name}”的任务记录？此操作不会撤销已经完成的工作。`)) return;
    await runAction(`delete:${workflow.id}`, () => api(`/api/dashboard/control/tasks/${workflow.id}`, { method: "DELETE" }));
  };

  return <>
    <PageIntro title="任务" description="查看小满正在做什么、哪里需要你确认，以及哪些事情已经完成。" actions={<IconButton icon={RefreshCw} label="刷新任务" onClick={resource.reload} />} />
    <ErrorBanner message={resource.error || actionError} />
    {resource.data?.length ? (
      <div className="workflow-list">
        {resource.data.map((workflow) => {
          const percent = workflow.step_count ? Math.round(workflow.completed_steps / workflow.step_count * 100) : 0;
          const isBusy = busy.includes(workflow.id);
          return (
            <div className="workflow-row" key={workflow.id}>
              <div className="workflow-symbol"><Workflow size={19} /></div>
              <div className="workflow-body">
                <div className="workflow-title"><strong>{workflow.name}</strong><Badge tone={statusTone(workflow.status)}>{STATUS_LABELS[workflow.status] ?? workflow.status}</Badge></div>
                <p>{workflow.goal}</p>
                <div className="progress-line"><span style={{ width: `${percent}%` }} /></div>
                <small>{workflow.completed_steps}/{workflow.step_count} 步完成 · 更新于 {relativeTime(workflow.updated_at)}</small>
                {workflow.error ? <div className="inline-error"><CircleAlert size={13} />{workflow.error}</div> : null}
                <WorkflowActions
                  workflow={workflow}
                  busy={isBusy}
                  responses={responses}
                  setResponse={(stepId, value) => setResponses((current) => ({ ...current, [`${workflow.id}:${stepId}`]: value }))}
                  approve={(stepId, approved) => void approve(workflow, stepId, approved)}
                  respond={(stepId) => void respond(workflow, stepId)}
                  retry={(stepId) => void retry(workflow, stepId)}
                />
              </div>
              {!['succeeded', 'cancelled'].includes(workflow.status) ? (
                <IconButton icon={Ban} label="取消任务" danger disabled={isBusy} onClick={() => void cancel(workflow)} />
              ) : (
                <div className="workflow-terminal-actions">
                  {workflow.status === "succeeded" ? <CheckCircle2 size={19} className="success-icon" /> : <Ban size={19} className="cancelled-icon" />}
                  <IconButton icon={Trash2} label="删除任务记录" danger disabled={isBusy} onClick={() => void removeHistory(workflow)} />
                </div>
              )}
            </div>
          );
        })}
      </div>
    ) : !resource.loading ? (
      <EmptyState icon={Workflow} title="还没有后台任务" text="当事情需要持续执行、等待确认或跨重启恢复时，小满会把它放进任务中心。" />
    ) : <LoadingState />}
  </>;
}
