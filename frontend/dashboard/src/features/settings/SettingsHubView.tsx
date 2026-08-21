import React from "react";
import { BellRing, Cable, Cpu, Radio, Settings2 } from "lucide-react";
import type { ViewId } from "../../shared/types";

interface SettingsHubViewProps { navigate: (view: ViewId) => void; }

const settingsGroups: { title: string; description: string; target: ViewId; icon: typeof Cpu }[] = [
  { title: "模型与回复", description: "选择小满聊天、视觉和复杂任务使用的模型。", target: "models", icon: Cpu },
  { title: "联系小满", description: "接入 QQ、微信、企业微信和 Telegram。", target: "channels", icon: Radio },
  { title: "扩展能力", description: "连接技能与 MCP，让小满能够使用更多工具。", target: "mcp", icon: Cable },
  { title: "主动联系设置", description: "决定小满通过哪个渠道主动找你。", target: "proactive", icon: BellRing },
  { title: "系统状态", description: "查看运行状态、任务航迹和高级诊断。", target: "overview", icon: Settings2 },
];

export function SettingsHubView({ navigate }: SettingsHubViewProps): React.ReactElement {
  return <div className="settings-hub">
    <header className="settings-hub-head"><h1>设置与扩展</h1><p>按你想完成的事情管理小满；技术细节只在需要时展开。</p></header>
    <div className="settings-hub-grid">{settingsGroups.map((item) => {
      const Icon = item.icon;
      return <button type="button" key={item.target} onClick={() => navigate(item.target)}><span><Icon size={21} /></span><div><strong>{item.title}</strong><p>{item.description}</p></div><b aria-hidden="true">›</b></button>;
    })}</div>
    <section className="settings-secondary"><div><h2>其他管理</h2><p>不常用，但仍可以随时进入。</p></div><div><button type="button" onClick={() => navigate("tools")}>工具目录</button><button type="button" onClick={() => navigate("sessions")}>任务航迹</button></div></section>
  </div>;
}
