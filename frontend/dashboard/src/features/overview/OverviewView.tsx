import React from "react";
import { Brain, Cable, CalendarClock, ChevronRight, CircleDot, Radio, RefreshCw, Sparkles, WandSparkles, Workflow, Wrench, type LucideIcon } from "lucide-react";
import { api } from "../../api";
import { Badge, ErrorBanner, IconButton, LoadingState, PageIntro } from "../../shared/components/ui";
import { useAsyncData } from "../../shared/hooks/useAsyncData";
import type { Overview, ViewId } from "../../shared/types";

export function OverviewView(props: { navigate: (view: ViewId) => void }): React.ReactElement {
  const resource = useAsyncData(() => api<Overview>("/api/dashboard/control/overview"), []);
  const overview = resource.data;
  const memoryState = overview ? ({
    healthy: { label: "健康", title: "长期记忆运行正常", tone: "green" as const },
    degraded: { label: "降级", title: "长期记忆正在降级运行", tone: "amber" as const },
    unhealthy: { label: "故障", title: "长期记忆需要修复", tone: "red" as const },
    unchecked: { label: "待检查", title: "长期记忆尚未完成健康检查", tone: "gray" as const },
    pending_restart: { label: "待重启", title: "记忆配置等待重启生效", tone: "amber" as const },
    disabled: { label: "已关闭", title: "长期记忆当前关闭", tone: "gray" as const },
  })[overview.memory_status] : null;
  const capabilityRows: { key: string; title: string; text: string; icon: LucideIcon; view: ViewId; tone: string }[] = [
    { key: "tasks_active", title: "任务中心", text: "统一的持久化执行", icon: Workflow, view: "workflows", tone: "mint" },
    { key: "mcp_servers", title: "MCP 工具", text: "标准化外部能力", icon: Cable, view: "mcp", tone: "blue" },
    { key: "skills", title: "技能", text: "可复用的方法与流程", icon: WandSparkles, view: "skills", tone: "amber" },
    { key: "channels", title: "消息渠道", text: "网页与常用聊天工具", icon: Radio, view: "channels", tone: "coral" },
    { key: "schedules", title: "定时任务", text: "提醒与周期自动化", icon: CalendarClock, view: "schedules", tone: "violet" },
    { key: "tools", title: "工具", text: "小满当前可用能力", icon: Wrench, view: "tools", tone: "gray" },
  ];
  return (
    <>
      <PageIntro title="运行概览" description="小满当前的模型、连接、记忆和自动化能力。" actions={<IconButton icon={RefreshCw} label="刷新" onClick={resource.reload} />} />
      <ErrorBanner message={resource.error} />
      {!overview && resource.loading ? <LoadingState /> : overview ? (
        <div className="overview-layout">
          <section className="status-hero">
            <div className="status-copy"><Badge tone="green"><CircleDot size={10} /> 服务在线</Badge><h2>{overview.assistant} 已准备好</h2><p>当前使用 <strong>{overview.model}</strong> 理解你的需求并完成任务。</p></div>
            <div className="status-orbit"><Sparkles size={30} /><span>就绪</span></div>
          </section>
          <div className="capability-grid">
            {capabilityRows.map((item) => {
              const Icon = item.icon;
              return <button className="capability-card" key={item.key} onClick={() => props.navigate(item.view)}><span className={`cap-icon ${item.tone}`}><Icon size={19} /></span><span><strong>{overview.counts[item.key] ?? 0}</strong><small>{item.title}</small><em>{item.text}</em></span><ChevronRight size={17} /></button>;
            })}
          </div>
          <section className="section-panel">
            <div className="section-heading"><div><h3>连接状态</h3><p>消息入口和外部触达通道</p></div><button className="text-button" onClick={() => props.navigate("channels")}>管理频道<ChevronRight size={15} /></button></div>
            <div className="connection-list">{overview.channels.map((channel) => <div className="connection-row" key={channel.id}><span className={`connection-dot ${channel.connected ? "online" : "offline"}`} /><div><strong>{channel.label}</strong><small>{channel.detail}</small></div><Badge tone={channel.connected ? "green" : "gray"}>{channel.connected ? "已连接" : "未配置"}</Badge></div>)}</div>
          </section>
          <section className="section-panel memory-health">
            <div className="section-heading"><div><h3>记忆系统</h3><p>长期上下文和个人偏好</p></div><button className="text-button" onClick={() => props.navigate("memory")}>打开记忆<ChevronRight size={15} /></button></div>
            <div className="memory-health-body"><Brain size={24} /><div><strong>{memoryState?.title}</strong><p>{overview.memory_engine === "akasha" || overview.memory_engine === "xiaoman" ? "小满统一记忆" : overview.memory_engine} · {overview.memory_enabled ? "随小满自动运行" : "当前不可用"}</p></div>{memoryState ? <Badge tone={memoryState.tone}>{memoryState.label}</Badge> : null}</div>
          </section>
        </div>
      ) : null}
    </>
  );
}
