import React, { useCallback, useEffect, useRef, useState } from "react";
import { Cable, Plus, RadioTower, RefreshCw, Rss, Trash2 } from "lucide-react";
import { api } from "../../api";
import {
  EmptyState,
  ErrorBanner,
  LoadingState,
  Modal,
  PageIntro,
} from "../../shared/components/ui";
import { useAsyncData } from "../../shared/hooks/useAsyncData";
import type { ExternalSourceRow, McpCatalogRow, McpRow } from "../../shared/types";
import {
  McpCatalogCard,
  McpConnectionCard,
  McpDetailDrawer,
} from "./McpCards";

type McpTransport = "stdio" | "streamable_http" | "sse";
type McpAuth = "none" | "oauth" | "bearer" | "headers";

interface McpForm {
  name: string;
  transport: McpTransport;
  command: string;
  url: string;
  cwd: string;
  env: string;
  authType: McpAuth;
  scopes: string;
  bearerToken: string;
  headers: string;
  oauthClientId: string;
  oauthClientSecret: string;
  setupNote: string;
  docsUrl: string;
}

interface SourceForm {
  serverName: string;
  name: string;
  resourceKey: string;
  entityType: string;
  toolName: string;
  arguments: string;
  itemsPath: string;
  fields: string;
  data: string;
  pollIntervalMinutes: string;
}

interface RssSourceForm {
  name: string;
  url: string;
  pollIntervalMinutes: string;
}

const EMPTY_RSS_FORM: RssSourceForm = {
  name: "",
  url: "",
  pollIntervalMinutes: "5",
};

const EMPTY_FORM: McpForm = {
  name: "",
  transport: "streamable_http",
  command: "[]",
  url: "",
  cwd: "",
  env: "{}",
  authType: "oauth",
  scopes: "",
  bearerToken: "",
  headers: "{}",
  oauthClientId: "",
  oauthClientSecret: "",
  setupNote: "",
  docsUrl: "",
};

const DEFAULT_ATTENTION_MAPPING = JSON.stringify({
  attention_signal: {
    const: {
      enabled: true,
      kind: "mcp.observation",
      confidence: 0.7,
      valid_for_minutes: 360,
    },
  },
}, null, 2);

interface McpViewProps {
  embedded?: boolean;
  showCatalog?: boolean;
}

export function McpView({ embedded = false, showCatalog = true }: McpViewProps): React.ReactElement {
  const [revision, setRevision] = useState(0);
  const [preset, setPreset] = useState<McpForm | null>(null);
  const consumePreset = useCallback(() => setPreset(null), []);
  return <>
    {!embedded ? <PageIntro
      title="扩展能力"
      description="连接标准 MCP 服务，为小满增加新工具；只有你明确选择的数据才会进入主动协助。"
    /> : null}
    <McpConnectionsPanel
      refreshToken={revision}
      preset={preset}
      onPresetConsumed={consumePreset}
    />
    {showCatalog ? <McpCatalogPanel
      onChanged={() => setRevision((value) => value + 1)}
      onConfigure={setPreset}
    /> : null}
  </>;
}

function McpCatalogPanel(props: {
  onChanged: () => void;
  onConfigure: (form: McpForm) => void;
}): React.ReactElement {
  const resource = useAsyncData(() => api<McpCatalogRow[]>("/api/dashboard/control/mcp/catalog"), []);
  const [busy, setBusy] = useState("");
  const [actionError, setActionError] = useState("");

  const install = async (entry: McpCatalogRow): Promise<void> => {
    if (entry.configuration) {
      props.onConfigure(formFromCatalog(entry));
      return;
    }
    setBusy(entry.id); setActionError("");
    try {
      await api(`/api/dashboard/control/mcp/catalog/${encodeURIComponent(entry.id)}`, { method: "POST" });
      resource.reload();
      props.onChanged();
    } catch (error) {
      setActionError(error instanceof Error ? error.message : String(error));
    } finally { setBusy(""); }
  };

  const entries = resource.data?.filter((entry) => !entry.installed) ?? [];

  return <section className="extension-catalog" aria-label="发现工具连接">
    <div className="extension-section-heading">
      <div><strong>发现连接</strong><p>添加常用应用，让小满获得新的读取和执行能力。</p></div>
    </div>
    <ErrorBanner message={resource.error || actionError} />
    {entries.length ? <div className="capability-grid extension-capability-grid mcp-card-grid">
      {entries.map((entry) => <McpCatalogCard key={entry.id} entry={entry} busy={busy === entry.id} onInstall={(item) => void install(item)} />)}
    </div> : !resource.loading ? <div className="extension-catalog-complete">推荐连接已经全部添加。</div> : null}
  </section>;
}

export function McpConnectionsPanel(props: {
  refreshToken?: number;
  preset?: McpForm | null;
  onPresetConsumed?: () => void;
}): React.ReactElement {
  const { onPresetConsumed, preset } = props;
  const refreshToken = props.refreshToken ?? 0;
  const resource = useAsyncData(() => api<McpRow[]>("/api/dashboard/control/mcp"), [refreshToken]);
  const sources = useAsyncData(() => api<ExternalSourceRow[]>("/api/dashboard/control/sources"), [refreshToken]);
  const [adding, setAdding] = useState(false);
  const [busy, setBusy] = useState("");
  const [actionError, setActionError] = useState("");
  const [form, setForm] = useState<McpForm>(EMPTY_FORM);
  const [sourceForm, setSourceForm] = useState<SourceForm | null>(null);
  const [rssForm, setRssForm] = useState<RssSourceForm | null>(null);
  const [detailName, setDetailName] = useState("");
  const detailTrigger = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!preset) return;
    setForm(preset);
    setAdding(true);
    onPresetConsumed?.();
  }, [preset, onPresetConsumed]);

  const add = async (): Promise<void> => {
    setActionError("");
    setBusy("add");
    try {
      await api("/api/dashboard/control/mcp", {
        method: "POST",
        body: JSON.stringify({
          name: form.name.trim(),
          transport: form.transport,
          command: form.transport === "stdio" ? JSON.parse(form.command) as string[] : [],
          url: form.transport === "stdio" ? "" : form.url.trim(),
          cwd: form.transport === "stdio" ? form.cwd || null : null,
          env: form.transport === "stdio" ? JSON.parse(form.env) as Record<string, string> : {},
          auth_type: form.transport === "stdio" ? "none" : form.authType,
          scopes: form.scopes,
          bearer_token: form.authType === "bearer" ? form.bearerToken : "",
          headers: form.authType === "headers" ? JSON.parse(form.headers) as Record<string, string> : {},
          oauth_client_id: form.authType === "oauth" ? form.oauthClientId : "",
          oauth_client_secret: form.authType === "oauth" ? form.oauthClientSecret : "",
        }),
      });
      setAdding(false);
      setForm(EMPTY_FORM);
      resource.reload();
    } catch (error) {
      setActionError(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy("");
    }
  };

  const authorize = async (name: string): Promise<void> => {
    setActionError("");
    setBusy(name);
    const popup = window.open("", "xiaoman-mcp-oauth", "popup,width=760,height=820");
    try {
      const result = await api<{ authorization_url: string }>(
        `/api/dashboard/control/mcp/${encodeURIComponent(name)}/authorize`,
        { method: "POST" },
      );
      if (popup) popup.location.href = result.authorization_url;
      else window.open(result.authorization_url, "_blank");
      resource.reload();
      let attempts = 0;
      const poll = window.setInterval(() => {
        attempts += 1;
        resource.reload();
        if (attempts >= 40) window.clearInterval(poll);
      }, 1500);
    } catch (error) {
      popup?.close();
      setActionError(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy("");
    }
  };

  const remove = async (name: string): Promise<void> => {
    if (!window.confirm(`确认移除“${name}”连接？小满将无法继续使用它提供的工具。`)) return;
    setActionError(""); setBusy(name);
    try {
      await api(`/api/dashboard/control/mcp/${encodeURIComponent(name)}`, { method: "DELETE" });
      setDetailName("");
      resource.reload();
    } catch (error) { setActionError(error instanceof Error ? error.message : String(error)); }
    finally { setBusy(""); }
  };

  const openSource = (server: McpRow): void => {
    setSourceForm({
      serverName: server.name,
      name: `${server.name} 主动数据`,
      resourceKey: `${server.name}-feed`,
      entityType: "monitor_observation",
      toolName: server.tool_names[0] ?? "",
      arguments: "{}",
      itemsPath: "",
      fields: '{"id":"id","title":"title","summary":"summary","source_ref":"url"}',
      data: DEFAULT_ATTENTION_MAPPING,
      pollIntervalMinutes: "15",
    });
  };

  const createSource = async (): Promise<void> => {
    if (!sourceForm) return;
    setBusy("source-add"); setActionError("");
    try {
      const created = await api<ExternalSourceRow>("/api/dashboard/control/sources", {
        method: "POST",
        body: JSON.stringify({
          provider: "mcp",
          server_name: sourceForm.serverName,
          name: sourceForm.name.trim(),
          resource_url: `mcp://${sourceForm.serverName}/${sourceForm.resourceKey.trim()}`,
          entity_type: sourceForm.entityType,
          mapping: {
            tool_name: sourceForm.toolName,
            arguments: JSON.parse(sourceForm.arguments) as Record<string, unknown>,
            items_path: sourceForm.itemsPath.trim(),
            fields: JSON.parse(sourceForm.fields) as Record<string, unknown>,
            data: JSON.parse(sourceForm.data) as Record<string, unknown>,
          },
          poll_interval_minutes: Number(sourceForm.pollIntervalMinutes),
          enabled: true,
        }),
      });
      const synced = await api<{ error: string }>(`/api/dashboard/control/sources/${created.id}/sync`, { method: "POST" });
      if (synced.error) throw new Error(`订阅已保存，首次同步失败：${synced.error}`);
      setSourceForm(null); sources.reload();
    } catch (error) { setActionError(error instanceof Error ? error.message : String(error)); }
    finally { setBusy(""); }
  };

  const createRssSource = async (): Promise<void> => {
    if (!rssForm) return;
    setBusy("rss-add"); setActionError("");
    try {
      const created = await api<ExternalSourceRow>("/api/dashboard/control/sources", {
        method: "POST",
        body: JSON.stringify({
          provider: "rss",
          server_name: "rss",
          name: rssForm.name.trim(),
          resource_url: rssForm.url.trim(),
          entity_type: "monitor_observation",
          mapping: {
            domain: "interest",
            notify_initial: false,
            valid_for_minutes: 1440,
            max_items: 50,
          },
          poll_interval_minutes: Number(rssForm.pollIntervalMinutes),
          enabled: true,
        }),
      });
      sources.reload();
      const synced = await api<{ error: string }>(`/api/dashboard/control/sources/${created.id}/sync`, { method: "POST" });
      if (synced.error) throw new Error(`订阅已保存，但读取失败：${synced.error}`);
      setRssForm(null);
      sources.reload();
    } catch (error) { setActionError(error instanceof Error ? error.message : String(error)); }
    finally { setBusy(""); }
  };

  const setSourceEnabled = async (source: ExternalSourceRow, enabled: boolean): Promise<void> => {
    setBusy(source.id); setActionError("");
    try {
      await api(`/api/dashboard/control/sources/${source.id}`, { method: "PATCH", body: JSON.stringify({ enabled }) });
      sources.reload();
    } catch (error) { setActionError(error instanceof Error ? error.message : String(error)); }
    finally { setBusy(""); }
  };

  const syncSource = async (source: ExternalSourceRow): Promise<void> => {
    setBusy(source.id); setActionError("");
    try {
      const result = await api<{ error: string }>(`/api/dashboard/control/sources/${source.id}/sync`, { method: "POST" });
      if (result.error) throw new Error(result.error);
      sources.reload();
    } catch (error) { setActionError(error instanceof Error ? error.message : String(error)); }
    finally { setBusy(""); }
  };

  const deleteSource = async (source: ExternalSourceRow): Promise<void> => {
    if (!window.confirm(`确认删除“${source.name}”信号源？${source.provider === "mcp" ? "MCP 连接本身会保留。" : "之后将不再读取这个地址。"}`)) return;
    setBusy(source.id); setActionError("");
    try {
      await api(`/api/dashboard/control/sources/${source.id}`, { method: "DELETE" });
      sources.reload();
    } catch (error) { setActionError(error instanceof Error ? error.message : String(error)); }
    finally { setBusy(""); }
  };

  const openDetails = (server: McpRow, trigger: HTMLButtonElement): void => {
    detailTrigger.current = trigger;
    setDetailName(server.name);
  };

  const closeDetails = useCallback((): void => {
    setDetailName("");
    window.requestAnimationFrame(() => detailTrigger.current?.focus());
  }, []);

  const selectedServer = resource.data?.find((server) => server.name === detailName) ?? null;
  const allSources = sources.data ?? [];
  const rssSources = allSources.filter((source) => source.provider === "rss");
  const selectedSources = selectedServer
    ? allSources.filter((source) => source.server_name === selectedServer.name)
    : [];

  return <>
    <div className="extension-section-heading">
      <div><strong>我的连接</strong><p>状态和信号源一目了然；详细工具与设置按需展开。</p></div>
      <div className="extension-section-actions">
        <button className="secondary-button" onClick={() => setRssForm(EMPTY_RSS_FORM)}><Rss size={16} />添加 RSS 订阅</button>
        <button className="primary-button" onClick={() => { setForm(EMPTY_FORM); setAdding(true); }}><Plus size={16} />添加连接</button>
      </div>
    </div>
    <ErrorBanner message={resource.error || sources.error || actionError} />
    {resource.loading && !resource.data ? <LoadingState /> : resource.data?.length ? <div className="capability-grid extension-capability-grid mcp-card-grid">
      {resource.data.map((server) => <McpConnectionCard key={server.name} server={server} sources={allSources} busy={busy === server.name} onOpen={openDetails} onAuthorize={(name) => void authorize(name)} />)}
    </div> : <EmptyState icon={Cable} title="还没有工具连接" text="从下方发现连接中添加，或配置一个本地、远程 MCP。" />}

    {rssSources.length ? <section className="rss-subscription-section" aria-label="内容订阅">
      <div className="rss-subscription-heading">
        <div className="rss-subscription-icon"><Rss size={18} /></div>
        <div><strong>内容订阅</strong><p>定时读取你明确添加的 RSS；新内容再交给注意力引擎判断是否联系你。</p></div>
      </div>
      <div className="capability-source-list rss-subscription-list">
        {rssSources.map((source) => <article key={source.id}>
          <div>
            <strong>{source.name}</strong>
            <small>{source.resource_url}</small>
            <small>{source.last_error ? `读取异常：${source.last_error}` : source.last_synced_at ? `最近读取 ${source.last_item_count} 条 · 每 ${source.poll_interval_minutes} 分钟` : "等待首次读取"}</small>
          </div>
          <div>
            <span className={`rss-source-status ${source.last_error ? "is-error" : source.enabled ? "is-active" : ""}`}>{source.last_error ? "异常" : source.enabled ? "订阅中" : "已暂停"}</span>
            <button disabled={busy === source.id} onClick={() => void setSourceEnabled(source, !source.enabled)}>{source.enabled ? "暂停" : "继续"}</button>
            <button disabled={busy === source.id} onClick={() => void syncSource(source)}><RefreshCw size={13} />立即读取</button>
            <button className="danger" disabled={busy === source.id} onClick={() => void deleteSource(source)}><Trash2 size={13} />删除</button>
          </div>
        </article>)}
      </div>
    </section> : null}

    {selectedServer ? <McpDetailDrawer
      server={selectedServer}
      sources={selectedSources}
      busy={busy}
      onClose={closeDetails}
      onAuthorize={(name) => void authorize(name)}
      onCreateSource={openSource}
      onSetSourceEnabled={(source, enabled) => void setSourceEnabled(source, enabled)}
      onSyncSource={(source) => void syncSource(source)}
      onDeleteSource={(source) => void deleteSource(source)}
      onRemove={(name) => void remove(name)}
    /> : null}

    {adding ? <Modal title="添加 MCP 连接" description="安装连接不会自动读取数据；敏感凭据只保存到系统凭据库。" onClose={() => setAdding(false)}>
      <div className="form-stack">
        {form.setupNote ? <p>{form.setupNote} {form.docsUrl ? <a href={form.docsUrl} target="_blank" rel="noreferrer">查看配置文档</a> : null}</p> : null}
        <label>连接名称<input value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} placeholder="notion" autoFocus /></label>
        <label>连接方式<select value={form.transport} onChange={(event) => setForm({ ...form, transport: event.target.value as McpTransport })}><option value="streamable_http">远程 Streamable HTTP（推荐）</option><option value="sse">远程 SSE（兼容旧服务）</option><option value="stdio">本地 stdio 子进程</option></select></label>
        {form.transport === "stdio" ? <>
          <label>启动命令<textarea value={form.command} onChange={(event) => setForm({ ...form, command: event.target.value })} rows={4} placeholder='["npx", "-y", "server"]' /></label>
          <label>工作目录<input value={form.cwd} onChange={(event) => setForm({ ...form, cwd: event.target.value })} placeholder="可选" /></label>
          <label>环境变量<textarea value={form.env} onChange={(event) => setForm({ ...form, env: event.target.value })} rows={3} placeholder='{"TOKEN":"..."}' /></label>
        </> : <>
          <label>服务地址<input value={form.url} onChange={(event) => setForm({ ...form, url: event.target.value })} placeholder="https://example.com/mcp" /></label>
          <label>认证方式<select value={form.authType} onChange={(event) => setForm({ ...form, authType: event.target.value as McpAuth })}><option value="oauth">OAuth 2.1 + PKCE（推荐）</option><option value="none">无需认证</option><option value="bearer">Bearer Token</option><option value="headers">自定义 Header</option></select></label>
          {form.authType === "oauth" ? <><label>请求范围<input autoComplete="off" value={form.scopes} onChange={(event) => setForm({ ...form, scopes: event.target.value })} placeholder="留空则自动发现" /></label><label>OAuth Client ID<input name="xiaoman-mcp-oauth-client-id" autoComplete="off" value={form.oauthClientId} onChange={(event) => setForm({ ...form, oauthClientId: event.target.value })} placeholder="服务要求预注册客户端时填写" /></label><label>OAuth Client Secret<input name="xiaoman-mcp-oauth-client-secret" type="password" autoComplete="new-password" value={form.oauthClientSecret} onChange={(event) => setForm({ ...form, oauthClientSecret: event.target.value })} placeholder="安全保存到系统凭据库" /></label></> : null}
          {form.authType === "bearer" ? <label>Bearer Token<input name="xiaoman-mcp-bearer-token" type="password" autoComplete="new-password" value={form.bearerToken} onChange={(event) => setForm({ ...form, bearerToken: event.target.value })} placeholder="安全保存到系统凭据库" /></label> : null}
          {form.authType === "headers" ? <label>认证 Header<textarea value={form.headers} onChange={(event) => setForm({ ...form, headers: event.target.value })} rows={3} placeholder='{"X-API-Key":"..."}' /></label> : null}
        </>}
        <div className="dialog-actions"><button className="secondary-button" onClick={() => setAdding(false)}>取消</button><button className="primary-button" disabled={!form.name.trim() || busy === "add" || (form.transport !== "stdio" && !form.url.trim())} onClick={() => void add()}><Cable size={16} />保存连接</button></div>
      </div>
    </Modal> : null}

    {rssForm ? <Modal title="添加 RSS 订阅" description="适合博客、新闻和公开账号动态。首次读取只建立基线，不会把历史内容全部提醒给你。" onClose={() => setRssForm(null)}>
      <div className="form-stack">
        <label>订阅名称<input value={rssForm.name} onChange={(event) => setRssForm({ ...rssForm, name: event.target.value })} placeholder="例如：关注某位 X 博主" autoFocus /></label>
        <label>RSS 地址<input value={rssForm.url} onChange={(event) => setRssForm({ ...rssForm, url: event.target.value })} placeholder="https://example.com/user/rss" /></label>
        <p className="form-help">支持标准 RSS 和 Atom。XCancel、Nitter 等公共服务可能要求白名单或临时不可用，保存时会实际读取验证。</p>
        <label>检查间隔（分钟）<input type="number" min="3" max="1440" value={rssForm.pollIntervalMinutes} onChange={(event) => setRssForm({ ...rssForm, pollIntervalMinutes: event.target.value })} /></label>
        <div className="dialog-actions"><button className="secondary-button" onClick={() => setRssForm(null)}>取消</button><button className="primary-button" disabled={!rssForm.name.trim() || !/^https?:\/\//i.test(rssForm.url.trim()) || busy === "rss-add"} onClick={() => void createRssSource()}><Rss size={16} />保存并验证</button></div>
      </div>
    </Modal> : null}

    {sourceForm ? <Modal title="添加为信号源" description="只会定期调用你选择的读取工具；安装的其他工具不会被自动读取。" onClose={() => setSourceForm(null)}><div className="form-stack">
      <label>订阅名称<input value={sourceForm.name} onChange={(event) => setSourceForm({ ...sourceForm, name: event.target.value })} autoFocus /></label>
      <label>资源标识<input value={sourceForm.resourceKey} onChange={(event) => setSourceForm({ ...sourceForm, resourceKey: event.target.value })} placeholder="important-unread" /></label>
      <label>读取工具<select value={sourceForm.toolName} onChange={(event) => setSourceForm({ ...sourceForm, toolName: event.target.value })}>{resource.data?.find((row) => row.name === sourceForm.serverName)?.tool_names.map((name) => <option key={name} value={name}>{name}</option>)}</select></label>
      <label>个人数据类型<select value={sourceForm.entityType} onChange={(event) => setSourceForm({ ...sourceForm, entityType: event.target.value, data: event.target.value === "monitor_observation" ? DEFAULT_ATTENTION_MAPPING : "{}" })}><option value="monitor_observation">观察信号（参与主动判断）</option><option value="commitment">待办（按时间进入注意力）</option><option value="calendar_event">日程（按时间进入注意力）</option><option value="health_observation">健康记录</option><option value="memory">长期信息</option></select></label>
      <label>调用参数<textarea rows={3} value={sourceForm.arguments} onChange={(event) => setSourceForm({ ...sourceForm, arguments: event.target.value })} /></label>
      <label>列表路径<input value={sourceForm.itemsPath} onChange={(event) => setSourceForm({ ...sourceForm, itemsPath: event.target.value })} placeholder="例如 threads；根结果为列表时留空" /></label>
      <label>基础字段映射<textarea rows={4} value={sourceForm.fields} onChange={(event) => setSourceForm({ ...sourceForm, fields: event.target.value })} /></label>
      <label>业务字段映射<textarea rows={4} value={sourceForm.data} onChange={(event) => setSourceForm({ ...sourceForm, data: event.target.value })} placeholder='{"due_at":"date","state":{"const":"open"}}' /></label>
      <label>同步间隔（分钟）<input type="number" min="1" max="1440" value={sourceForm.pollIntervalMinutes} onChange={(event) => setSourceForm({ ...sourceForm, pollIntervalMinutes: event.target.value })} /></label>
      <div className="dialog-actions"><button className="secondary-button" onClick={() => setSourceForm(null)}>取消</button><button className="primary-button" disabled={!sourceForm.name.trim() || !sourceForm.resourceKey.trim() || !sourceForm.toolName || busy === "source-add"} onClick={() => void createSource()}><RadioTower size={16} />保存并首次同步</button></div>
    </div></Modal> : null}
  </>;
}

function formFromCatalog(entry: McpCatalogRow): McpForm {
  const setup = entry.configuration;
  if (!setup) return EMPTY_FORM;
  return {
    ...EMPTY_FORM,
    name: setup.name,
    transport: setup.transport,
    command: JSON.stringify(setup.command ?? [], null, 2),
    url: setup.url ?? "",
    authType: setup.auth_type ?? "none",
    scopes: setup.scopes ?? "",
    setupNote: setup.requires_oauth_client
      ? "Gmail 需要先在 Google Cloud 启用 Gmail API 与 Gmail MCP API，并创建 Web OAuth Client；回调地址为 http://127.0.0.1:2236/api/dashboard/control/mcp/oauth/callback/gmail。"
      : setup.requires_vault_path
        ? "请把启动命令最后一项替换为你的 Obsidian Vault 绝对路径。"
        : "",
    docsUrl: setup.docs_url ?? "",
  };
}
