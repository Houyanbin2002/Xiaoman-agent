import React, { useMemo, useState } from "react";
import { Search } from "lucide-react";
import { api } from "../../api";
import { Badge, PageIntro } from "../../shared/components/ui";
import { useAsyncData } from "../../shared/hooks/useAsyncData";
import type { ToolRow } from "../../shared/types";

export function ToolsView(): React.ReactElement {
  const resource = useAsyncData(() => api<ToolRow[]>("/api/dashboard/control/tools"), []);
  const [query, setQuery] = useState("");
  const [source, setSource] = useState("all");
  const rows = useMemo(() => (resource.data ?? []).filter((item) => (source === "all" || item.source_type === source) && `${item.name} ${item.description}`.toLowerCase().includes(query.toLowerCase())), [query, resource.data, source]);
  const sourceLabels: Record<string, string> = { builtin: "内置", plugin: "系统组件", mcp: "MCP" };
  const riskLabels: Record<string, string> = { "read-only": "只读", write: "写入", "external-side-effect": "外部操作" };
  return <><PageIntro title="工具目录" description="当前模型在推理过程中可以发现和调用的执行能力。" actions={<div className="filter-actions"><select value={source} onChange={(event) => setSource(event.target.value)}><option value="all">全部来源</option><option value="builtin">内置</option><option value="plugin">系统组件</option><option value="mcp">MCP</option></select><div className="search-box"><Search size={16} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索工具" /></div></div>} /><div className="data-table"><div className="data-head tools-columns"><span>工具</span><span>来源</span><span>风险</span><span>可见性</span></div>{rows.map((tool) => <div className="data-row tools-columns" key={tool.name}><div><strong className="mono-name">{tool.name}</strong><p>{tool.description}</p></div><span><Badge tone={tool.source_type === "mcp" ? "blue" : tool.source_type === "plugin" ? "green" : "gray"}>{tool.source_name && tool.source_name !== tool.source_type ? tool.source_name : sourceLabels[tool.source_type] ?? tool.source_type}</Badge></span><span><Badge tone={tool.risk === "read-only" ? "green" : tool.risk === "write" ? "amber" : "red"}>{riskLabels[tool.risk] ?? tool.risk}</Badge></span><span>{tool.always_on ? "常驻" : "按需发现"}</span></div>)}</div></>;
}
