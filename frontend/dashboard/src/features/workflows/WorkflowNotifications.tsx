import type { WorkflowNotification } from "../../shared/types";

const LABELS: Record<string, string> = {
  pending: "等待发送", sending: "正在发送", accepted: "发送端已确认（不代表已读）",
  confirmed: "你已确认收到", failed: "未能发送", unknown: "送达结果未知", cancelled: "已失效",
};

export function WorkflowNotifications({ notifications, busy, resolve }: {
  notifications: WorkflowNotification[];
  busy: boolean;
  resolve: (notification: WorkflowNotification, action: "retry" | "received") => void;
}): React.ReactElement {
  return <>{notifications.filter((item) => item.is_current).map((item) => (
    <section className="workflow-action-panel" key={item.id}>
      <div>
        <strong>{item.step_id ? "确认请求通知" : "任务结果通知"} · {LABELS[item.status] ?? item.status}</strong>
        <p>{item.channel || "未配置渠道"} · 已尝试 {item.attempts} 次。通知状态不影响任务执行结果。</p>
        {item.error ? <p role="status">{item.error}</p> : null}
        {item.status === "failed" && item.next_attempt_at ? <p>下次自动重试：{new Date(item.next_attempt_at * 1000).toLocaleString()}</p> : null}
        {item.status === "failed" && !item.next_attempt_at ? <p>自动重试已停止，修复渠道后可单独补发。</p> : null}
        {item.status === "unknown" ? <p>已暂停自动重发。请先核对原渠道，避免重复打扰。</p> : null}
        <details><summary>查看通知内容</summary><p style={{ whiteSpace: "pre-wrap" }}>{item.message}</p></details>
      </div>
      {["failed", "unknown"].includes(item.status) ? <div className="workflow-action-buttons">
        <button className="secondary-button" disabled={busy} onClick={() => resolve(item, "received")}>已收到，不补发</button>
        <button className="secondary-button" disabled={busy} onClick={() => resolve(item, "retry")}>只补发通知</button>
      </div> : null}
    </section>
  ))}</>;
}
