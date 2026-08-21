import React, { useEffect, useMemo, useState } from "react";
import {
  ArrowUpRight,
  BadgeCheck,
  Download,
  KeyRound,
  PackageOpen,
  RefreshCw,
  Search,
  Sparkles,
  Wrench,
  X,
} from "lucide-react";
import { api } from "../../api";
import { Badge, EmptyState, ErrorBanner, LoadingState, Modal } from "../../shared/components/ui";
import type {
  MarketplaceInstallResponse,
  MarketplaceItem,
  MarketplaceResponse,
} from "../../shared/types";
import { CapabilityLogo } from "./CapabilityLogo";
import { brandForCapability } from "./capabilityPresentation";

type MarketFilter = "all" | "skill" | "mcp";

interface MarketplaceViewProps {
  onOpenInstalled: (kind: "skill" | "mcp") => void;
}

const filterLabels: Record<MarketFilter, string> = {
  all: "全部",
  skill: "技能",
  mcp: "工具连接",
};

export function MarketplaceView({
  onOpenInstalled,
}: MarketplaceViewProps): React.ReactElement {
  const [query, setQuery] = useState("");
  const [deferredQuery, setDeferredQuery] = useState("");
  const [filter, setFilter] = useState<MarketFilter>("all");
  const [rows, setRows] = useState<MarketplaceItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState("");
  const [revision, setRevision] = useState(0);
  const [selected, setSelected] = useState<MarketplaceItem | null>(null);
  const [configureTarget, setConfigureTarget] = useState<MarketplaceItem | null>(null);
  const [configuration, setConfiguration] = useState<Record<string, string>>({});

  useEffect(() => {
    const timer = window.setTimeout(() => setDeferredQuery(query.trim()), 320);
    return () => window.clearTimeout(timer);
  }, [query]);

  useEffect(() => {
    let active = true;
    const load = async (): Promise<void> => {
      setLoading(true);
      setError("");
      try {
        const kinds: Array<"skill" | "mcp"> = filter === "all"
          ? ["skill", "mcp"]
          : [filter];
        const outcomes = await Promise.allSettled(kinds.map((kind) => api<MarketplaceResponse>(
          `/api/dashboard/control/marketplace?kind=${kind}&q=${encodeURIComponent(deferredQuery)}&limit=60`,
        )));
        if (!active) return;
        const responses = outcomes.flatMap((outcome) => (
          outcome.status === "fulfilled" ? [outcome.value] : []
        ));
        if (!responses.length) {
          const failure = outcomes.find((outcome) => outcome.status === "rejected");
          throw failure?.reason ?? new Error("扩展市场暂时不可用");
        }
        const merged = responses.flatMap((response) => response.items);
        setRows(merged.filter((item, index) => (
          merged.findIndex((candidate) => candidate.kind === item.kind && candidate.id === item.id) === index
        )));
        if (responses.length < outcomes.length) {
          setError("部分市场来源暂时不可用，已显示其余可用结果。");
        }
      } catch (reason) {
        if (active) {
          setRows([]);
          setError(reason instanceof Error ? reason.message : String(reason));
        }
      } finally {
        if (active) setLoading(false);
      }
    };
    void load();
    return () => { active = false; };
  }, [deferredQuery, filter, revision]);

  const visibleRows = useMemo(
    () => [...rows].sort((left, right) => Number(left.deprecated) - Number(right.deprecated)),
    [rows],
  );

  const refresh = async (): Promise<void> => {
    setBusy("refresh"); setError(""); setNotice("");
    try {
      const suffix = filter === "all" ? "" : `?kind=${filter}`;
      await api(`/api/dashboard/control/marketplace/refresh${suffix}`, { method: "POST" });
      setRevision((value) => value + 1);
      setNotice("市场目录已刷新。");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally { setBusy(""); }
  };

  const beginInstall = (item: MarketplaceItem): void => {
    if (item.install_mode === "unsupported" || item.installed) return;
    if (item.install_mode === "configure" || item.configuration_fields.length) {
      setConfiguration({});
      setConfigureTarget(item);
      return;
    }
    void install(item, {});
  };

  const install = async (
    item: MarketplaceItem,
    values: Record<string, string>,
  ): Promise<void> => {
    setBusy(`${item.kind}:${item.id}`); setError(""); setNotice("");
    try {
      const result = await api<MarketplaceInstallResponse>(
        "/api/dashboard/control/marketplace/install",
        {
          method: "POST",
          body: JSON.stringify({ kind: item.kind, item_id: item.id, configuration: values }),
        },
      );
      setRows((current) => current.map((row) => (
        row.id === item.id && row.kind === item.kind ? { ...row, installed: true } : row
      )));
      setSelected((current) => current?.id === item.id ? { ...current, installed: true } : current);
      setConfigureTarget(null);
      if (result.status === "authorization_required") {
        setNotice(`${item.name} 已添加，接下来完成账号授权。`);
        onOpenInstalled("mcp");
      } else {
        setNotice(`${item.name} 已安装，可以立即使用。`);
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally { setBusy(""); }
  };

  return <section className="marketplace-shelf" aria-label="扩展能力市场">
    <div className="marketplace-search-band">
      <div>
        <span className="marketplace-eyebrow"><Sparkles size={13} />能力市场</span>
        <h2>给小满找到新的做事方法</h2>
        <p>技能提供方法，工具连接让小满实际读取或操作外部应用。</p>
      </div>
      <div className="marketplace-search-controls">
        <label className="search-box marketplace-search">
          <Search size={17} />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="搜索技能、应用或任务"
            aria-label="搜索扩展能力市场"
          />
        </label>
        <button
          type="button"
          className="marketplace-refresh"
          disabled={busy === "refresh"}
          onClick={() => void refresh()}
          aria-label="刷新市场"
        >
          <RefreshCw size={16} className={busy === "refresh" ? "spin" : ""} />
        </button>
      </div>
    </div>

    <div className="extension-filter-row marketplace-filter-row">
      <div className="filter-tabs skill-filter-tabs" aria-label="市场类型筛选">
        {(Object.keys(filterLabels) as MarketFilter[]).map((value) => <button
          type="button"
          key={value}
          className={filter === value ? "active" : ""}
          onClick={() => setFilter(value)}
        >{filterLabels[value]}</button>)}
      </div>
      <span>{visibleRows.length} 项可发现能力</span>
    </div>

    <ErrorBanner message={error} />
    {notice ? <div className="action-notice marketplace-notice">{notice}</div> : null}
    {loading ? <LoadingState /> : visibleRows.length ? <div className="capability-grid extension-capability-grid marketplace-grid">
      {visibleRows.map((item) => <MarketplaceCard
        key={`${item.kind}:${item.id}`}
        item={item}
        busy={busy === `${item.kind}:${item.id}`}
        onOpen={setSelected}
        onInstall={beginInstall}
      />)}
    </div> : <EmptyState
      icon={PackageOpen}
      title={deferredQuery ? "没有找到匹配能力" : "市场暂时没有可展示内容"}
      text={deferredQuery ? "换一个更具体的任务或应用名称试试。" : "刷新市场，或搜索你想让小满完成的任务。"}
    />}

    {selected ? <MarketplaceDrawer
      item={selected}
      busy={busy === `${selected.kind}:${selected.id}`}
      onClose={() => setSelected(null)}
      onInstall={beginInstall}
      onOpenInstalled={onOpenInstalled}
    /> : null}

    {configureTarget ? <Modal
      title={`配置 ${configureTarget.name}`}
      description="只填写这个工具运行所需的内容；安装后不会自动成为主动协助信号源。"
      onClose={() => setConfigureTarget(null)}
    >
      <div className="form-stack">
        {configureTarget.configuration_fields.map((field) => <label key={field.name}>
          {field.label}{field.required ? " *" : ""}
          <input
            type={field.secret ? "password" : "text"}
            autoComplete={field.secret ? "new-password" : "off"}
            value={configuration[field.name] ?? ""}
            placeholder={field.placeholder}
            onChange={(event) => setConfiguration({ ...configuration, [field.name]: event.target.value })}
          />
        </label>)}
        <div className="dialog-actions">
          <button className="secondary-button" onClick={() => setConfigureTarget(null)}>取消</button>
          <button
            className="primary-button"
            disabled={busy !== "" || configureTarget.configuration_fields.some((field) => field.required && !configuration[field.name]?.trim())}
            onClick={() => void install(configureTarget, configuration)}
          ><Download size={16} />安装</button>
        </div>
      </div>
    </Modal> : null}
  </section>;
}

function MarketplaceCard(props: {
  item: MarketplaceItem;
  busy: boolean;
  onOpen: (item: MarketplaceItem) => void;
  onInstall: (item: MarketplaceItem) => void;
}): React.ReactElement {
  const { item } = props;
  const knownBrand = brandForCapability(item.name, item.provider);
  const brand = item.icon_url ? { ...knownBrand, logo: item.icon_url } : knownBrand;
  const unsupported = item.install_mode === "unsupported";
  return <article className={`extension-capability-card marketplace-card${item.deprecated ? " is-deprecated" : ""}`}>
    <button type="button" className="extension-card-open" onClick={() => props.onOpen(item)} aria-label={`查看 ${item.name} 详情`} />
    <div className="extension-card-head">
      <CapabilityLogo brand={brand} />
      <div className="extension-card-identity"><h3>{item.name}</h3><small>{item.provider || item.id}</small></div>
      {item.installed ? <Badge tone="green">已安装</Badge> : item.deprecated ? <Badge tone="amber">已弃用</Badge> : null}
    </div>
    <p className="extension-card-description">{item.description || "为小满增加一项可复用能力。"}</p>
    <div className="extension-card-footer">
      <div className="tag-row">
        <Badge tone={item.kind === "skill" ? "blue" : "gray"}>{item.kind === "skill" ? "技能" : "工具连接"}</Badge>
        {item.verified ? <Badge tone="green"><BadgeCheck size={11} />来源已验证</Badge> : null}
      </div>
      <button
        type="button"
        className="extension-card-action is-primary is-above-card"
        disabled={item.installed || unsupported || props.busy}
        onClick={() => props.onInstall(item)}
      >
        {props.busy ? <RefreshCw size={14} className="spin" /> : item.install_mode === "oauth" ? <KeyRound size={14} /> : <Download size={14} />}
        {item.installed ? "已安装" : unsupported ? "暂不支持" : item.install_mode === "oauth" ? "连接账号" : item.install_mode === "configure" ? "配置安装" : "安装"}
      </button>
    </div>
  </article>;
}

function MarketplaceDrawer(props: {
  item: MarketplaceItem;
  busy: boolean;
  onClose: () => void;
  onInstall: (item: MarketplaceItem) => void;
  onOpenInstalled: (kind: "skill" | "mcp") => void;
}): React.ReactElement {
  const { item } = props;
  const knownBrand = brandForCapability(item.name, item.provider);
  const brand = item.icon_url ? { ...knownBrand, logo: item.icon_url } : knownBrand;
  return <div className="capability-drawer-layer" role="presentation">
    <button className="capability-drawer-backdrop" type="button" aria-label="关闭详情" onClick={props.onClose} />
    <aside className="capability-drawer marketplace-drawer" role="dialog" aria-modal="true" aria-labelledby="marketplace-detail-title">
      <header className="capability-drawer-head">
        <CapabilityLogo brand={brand} size="lg" />
        <div><span className="capability-drawer-eyebrow">{item.kind === "skill" ? "技能" : "工具连接"}</span><h2 id="marketplace-detail-title">{item.name}</h2><div className="tag-row">{item.verified ? <Badge tone="green">来源已验证</Badge> : null}{item.version ? <Badge tone="gray">v{item.version}</Badge> : null}</div></div>
        <button className="capability-drawer-close" type="button" onClick={props.onClose} aria-label="关闭"><X size={18} /></button>
      </header>
      <section className="capability-drawer-section marketplace-detail-copy">
        <div className="capability-drawer-title"><Wrench size={16} /><div><strong>它能做什么</strong><small>面向日常任务的能力说明</small></div></div>
        <p>{item.description || "该来源没有提供额外说明。"}</p>
      </section>
      {item.install_mode === "unsupported" ? <div className="capability-drawer-error"><strong>当前版本暂不支持自动安装</strong><p>{item.unsupported_reason}</p></div> : null}
      <details className="capability-advanced"><summary>来源与版本</summary><dl><div><dt>提供方</dt><dd>{item.provider || "未注明"}</dd></div><div><dt>市场 ID</dt><dd>{item.id}</dd></div>{item.source_url ? <div><dt>来源页面</dt><dd><a href={item.source_url} target="_blank" rel="noreferrer">打开来源 <ArrowUpRight size={12} /></a></dd></div> : null}</dl></details>
      <div className="capability-drawer-actions marketplace-drawer-actions">
        {item.installed ? <button className="primary-button" onClick={() => props.onOpenInstalled(item.kind)}>查看已安装能力</button> : <button className="primary-button" disabled={item.install_mode === "unsupported" || props.busy} onClick={() => props.onInstall(item)}>{props.busy ? <RefreshCw size={15} className="spin" /> : <Download size={15} />}{item.install_mode === "unsupported" ? "暂不支持" : "安装"}</button>}
      </div>
    </aside>
  </div>;
}
