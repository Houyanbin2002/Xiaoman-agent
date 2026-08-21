import React, { useEffect, useRef } from "react";
import {
  ArrowUpRight,
  KeyRound,
  Pause,
  Play,
  Plus,
  RadioTower,
  RefreshCw,
  Trash2,
  Wrench,
  X,
} from "lucide-react";
import { Badge } from "../../shared/components/ui";
import type { ExternalSourceRow, McpCatalogRow, McpRow } from "../../shared/types";
import { CapabilityLogo } from "../extensions/CapabilityLogo";
import {
  brandForCapability,
  sourceStateForServer,
  statusPresentation,
} from "../extensions/capabilityPresentation";

interface McpCatalogCardProps {
  entry: McpCatalogRow;
  busy: boolean;
  onInstall: (entry: McpCatalogRow) => void;
}

export function McpCatalogCard({
  entry,
  busy,
  onInstall,
}: McpCatalogCardProps): React.ReactElement {
  const brand = brandForCapability(entry.name, entry.provider);
  return (
    <article className="capability-card extension-capability-card mcp-capability-card">
      <div className="extension-card-head">
        <CapabilityLogo brand={brand} />
        <div className="extension-card-identity">
          <h3>{entry.name}</h3>
          <small>{entry.provider}</small>
        </div>
        {entry.installed ? <Badge tone="green">已添加</Badge> : null}
      </div>
      <p className="extension-card-description">{entry.description}</p>
      <div className="extension-card-footer">
        <div className="tag-row">
          <Badge tone="gray">{entry.transport === "stdio" ? "本地" : "远程"}</Badge>
          {entry.requires_oauth ? <Badge tone="blue">账号连接</Badge> : null}
        </div>
        <button
          type="button"
          className="extension-card-action is-primary"
          disabled={entry.installed || busy}
          onClick={() => onInstall(entry)}
        >
          {busy ? <RefreshCw size={14} className="spin" /> : <Plus size={15} />}
          {entry.installed ? "已添加" : entry.configuration ? "配置" : "添加"}
        </button>
      </div>
    </article>
  );
}

interface McpConnectionCardProps {
  server: McpRow;
  sources: ExternalSourceRow[];
  busy: boolean;
  onOpen: (server: McpRow, trigger: HTMLButtonElement) => void;
  onAuthorize: (name: string) => void;
}

export function McpConnectionCard({
  server,
  sources,
  busy,
  onOpen,
  onAuthorize,
}: McpConnectionCardProps): React.ReactElement {
  const brand = brandForCapability(server.name);
  const status = statusPresentation(server.status);
  const sourceState = sourceStateForServer(server.name, sources);
  const description = connectionDescription(brand.label, server.tool_names.length);
  return (
    <article className={`capability-card extension-capability-card mcp-capability-card status-${server.status}`}>
      <button
        type="button"
        className="extension-card-open"
        onClick={(event) => onOpen(server, event.currentTarget)}
        aria-label={`查看 ${brand.label} 连接详情`}
      />
      <div className="extension-card-head">
        <CapabilityLogo brand={brand} />
        <div className="extension-card-identity">
          <h3>{brand.label}</h3>
          <small>{server.name === brand.label ? "已安装连接" : server.name}</small>
        </div>
        <Badge tone={status.tone}>{status.label}</Badge>
      </div>
      <p className="extension-card-description">{server.error || description}</p>
      <div className="extension-card-footer">
        <div className="tag-row">
          {sourceState === "active" ? <Badge tone="blue"><RadioTower size={11} />信号源</Badge> : null}
          {sourceState === "paused" ? <Badge tone="gray"><Pause size={11} />信号源已暂停</Badge> : null}
          {server.system_managed ? <Badge tone="gray">系统提供</Badge> : null}
        </div>
        {server.auth_type === "oauth" && !server.connected ? (
          <button
            type="button"
            className="extension-card-action is-primary is-above-card"
            disabled={busy}
            onClick={() => onAuthorize(server.name)}
          >
            {busy ? <RefreshCw size={14} className="spin" /> : <KeyRound size={14} />}
            授权
          </button>
        ) : (
          <button
            type="button"
            className="extension-card-action is-above-card"
            onClick={(event) => onOpen(server, event.currentTarget)}
          >
            查看详情<ArrowUpRight size={14} />
          </button>
        )}
      </div>
    </article>
  );
}

interface McpDetailDrawerProps {
  server: McpRow;
  sources: ExternalSourceRow[];
  busy: string;
  onClose: () => void;
  onAuthorize: (name: string) => void;
  onCreateSource: (server: McpRow) => void;
  onSetSourceEnabled: (source: ExternalSourceRow, enabled: boolean) => void;
  onSyncSource: (source: ExternalSourceRow) => void;
  onDeleteSource: (source: ExternalSourceRow) => void;
  onRemove: (name: string) => void;
}

export function McpDetailDrawer({
  server,
  sources,
  busy,
  onClose,
  onAuthorize,
  onCreateSource,
  onSetSourceEnabled,
  onSyncSource,
  onDeleteSource,
  onRemove,
}: McpDetailDrawerProps): React.ReactElement {
  const closeButton = useRef<HTMLButtonElement>(null);
  const brand = brandForCapability(server.name);
  const status = statusPresentation(server.status);
  const sourceState = sourceStateForServer(server.name, sources);

  useEffect(() => {
    closeButton.current?.focus();
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  return (
    <div className="capability-drawer-layer">
      <button type="button" className="capability-drawer-backdrop" onClick={onClose} aria-label="关闭连接详情" />
      <aside className="capability-drawer" role="dialog" aria-modal="true" aria-labelledby="mcp-detail-title">
        <header className="capability-drawer-head">
          <CapabilityLogo brand={brand} size="lg" />
          <div>
            <span className="capability-drawer-eyebrow">工具连接</span>
            <h2 id="mcp-detail-title">{brand.label}</h2>
            <div className="tag-row">
              <Badge tone={status.tone}>{status.label}</Badge>
              {sourceState === "active" ? <Badge tone="blue">信号源</Badge> : null}
              {sourceState === "paused" ? <Badge tone="gray">信号源已暂停</Badge> : null}
            </div>
          </div>
          <button ref={closeButton} type="button" className="capability-drawer-close" onClick={onClose} aria-label="关闭"><X size={18} /></button>
        </header>

        {server.error ? <div className="capability-drawer-error"><strong>连接没有完成</strong><p>{server.error}</p></div> : null}

        <section className="capability-drawer-section">
          <div className="capability-drawer-title"><Wrench size={17} /><div><strong>可用能力</strong><small>小满可在任务中按需调用</small></div></div>
          {server.tool_names.length ? <div className="capability-tool-list">{server.tool_names.map((name) => <span key={name}>{friendlyToolName(name)}</span>)}</div> : <p className="capability-drawer-empty">连接成功后，可用工具会出现在这里。</p>}
        </section>

        <section className="capability-drawer-section">
          <div className="capability-drawer-title"><RadioTower size={17} /><div><strong>主动信号源</strong><small>只有这里启用的数据才会进入主动协助</small></div></div>
          {sources.length ? <div className="capability-source-list">{sources.map((source) => <article key={source.id}>
            <div><strong>{source.name}</strong><small>{source.last_error || `${source.last_item_count} 条 · 每 ${source.poll_interval_minutes} 分钟同步`}</small></div>
            <div>
              <button type="button" onClick={() => onSetSourceEnabled(source, !source.enabled)}>{source.enabled ? <Pause size={14} /> : <Play size={14} />}{source.enabled ? "暂停" : "启用"}</button>
              <button type="button" disabled={busy === source.id} onClick={() => onSyncSource(source)}><RefreshCw size={14} className={busy === source.id ? "spin" : ""} />同步</button>
              <button type="button" className="danger" onClick={() => onDeleteSource(source)}><Trash2 size={14} />删除</button>
            </div>
          </article>)}</div> : <p className="capability-drawer-empty">这个连接尚未向主动协助提供数据。</p>}
          {server.connected && server.tool_names.length ? <button type="button" className="secondary-button capability-source-add" onClick={() => onCreateSource(server)}><Plus size={15} />添加信号源</button> : null}
        </section>

        <details className="capability-advanced">
          <summary>连接信息</summary>
          <dl>
            <div><dt>连接名称</dt><dd>{server.name}</dd></div>
            <div><dt>连接方式</dt><dd>{transportLabel(server.transport)}</dd></div>
            <div><dt>服务位置</dt><dd>{server.url || server.command.join(" ") || "系统管理"}</dd></div>
            <div><dt>工具数量</dt><dd>{server.tool_names.length} 项</dd></div>
          </dl>
        </details>

        <footer className="capability-drawer-actions">
          {server.auth_type === "oauth" && !server.connected ? <button type="button" className="primary-button" disabled={busy === server.name} onClick={() => onAuthorize(server.name)}><KeyRound size={15} />完成授权</button> : null}
          {!server.system_managed ? <button type="button" className="danger-text-button" disabled={busy === server.name} onClick={() => onRemove(server.name)}><Trash2 size={15} />移除连接</button> : null}
        </footer>
      </aside>
    </div>
  );
}

function connectionDescription(label: string, toolCount: number): string {
  const descriptions: Record<string, string> = {
    Notion: "搜索、读取和更新你的 Notion 工作区内容。",
    Gmail: "搜索邮件、读取会话并创建邮件草稿。",
    Obsidian: "读取和整理本地 Obsidian 知识库。",
    文档解析: "读取 PDF、Word、Excel、PPT 和常见文本文件。",
  };
  return descriptions[label] ?? (toolCount ? `为小满提供 ${toolCount} 项可调用能力。` : "等待连接后发现可用能力。");
}

function friendlyToolName(name: string): string {
  const value = name.replace(/^mcp_[^_]+__/, "").replace(/[_-]+/g, " ").trim();
  return value || name;
}

function transportLabel(transport: McpRow["transport"]): string {
  if (transport === "stdio") return "本地应用";
  if (transport === "sse") return "远程连接（SSE）";
  return "远程连接";
}
