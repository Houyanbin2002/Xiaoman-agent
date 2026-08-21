import React, { useMemo, useState } from "react";
import { Download, Search, Trash2, WandSparkles } from "lucide-react";
import { api } from "../../api";
import { Badge, EmptyState, ErrorBanner, Modal, PageIntro } from "../../shared/components/ui";
import { useAsyncData } from "../../shared/hooks/useAsyncData";
import type { SkillRow } from "../../shared/types";

type SkillFilter = "all" | "standalone" | "workspace" | "builtin";

interface SkillsViewProps {
  embedded?: boolean;
}

const filterLabels: Record<SkillFilter, string> = {
  all: "全部",
  standalone: "独立安装",
  workspace: "工作区",
  builtin: "内置",
};

export function SkillsView({ embedded = false }: SkillsViewProps): React.ReactElement {
  const resource = useAsyncData(() => api<SkillRow[]>("/api/dashboard/control/skills"), []);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<SkillFilter>("all");
  const [installing, setInstalling] = useState(false);
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState("");
  const [notice, setNotice] = useState("");
  const [uninstallTarget, setUninstallTarget] = useState<SkillRow | null>(null);
  const [form, setForm] = useState({ source: "", ref: "", subdir: "" });
  const rows = useMemo(() => (resource.data ?? []).filter((item) => {
    const matchesFilter = filter === "all" || item.origin === filter;
    const haystack = `${item.display_name} ${item.description} ${item.source_label}`.toLowerCase();
    return matchesFilter && haystack.includes(query.toLowerCase());
  }), [filter, query, resource.data]);

  const install = async (): Promise<void> => {
    setBusy(true); setActionError(""); setNotice("");
    try {
      await api("/api/dashboard/control/skills/install", {
        method: "POST",
        body: JSON.stringify({
          source: form.source,
          ref: form.ref,
          subdir: form.subdir,
        }),
      });
      setInstalling(false);
      setForm({ source: "", ref: "", subdir: "" });
      resource.reload();
      setNotice("技能已安装，新的任务可以立即发现并使用它。");
    } catch (error) {
      setActionError(error instanceof Error ? error.message : String(error));
    } finally { setBusy(false); }
  };

  const uninstall = async (): Promise<void> => {
    if (!uninstallTarget?.provider_id) return;
    setBusy(true); setActionError(""); setNotice("");
    try {
      await api(`/api/dashboard/control/skills/${encodeURIComponent(uninstallTarget.name)}`, { method: "DELETE" });
      setUninstallTarget(null);
      resource.reload();
      setNotice("技能已卸载；正在运行的旧对话可能需要重新开始才能刷新能力列表。");
    } catch (error) {
      setActionError(error instanceof Error ? error.message : String(error));
    } finally { setBusy(false); }
  };

  return <>
    {!embedded ? <PageIntro
      title="技能"
      description="技能是小满处理某类任务时会主动采用的方法、步骤和专业知识。"
      actions={<><div className="search-box"><Search size={16} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索技能" /></div><button className="primary-button" onClick={() => setInstalling(true)}><Download size={16} />安装技能</button></>}
    /> : null}
    {embedded ? <div className="extension-toolbar">
      <div className="search-box"><Search size={16} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索技能" /></div>
      <button className="primary-button" onClick={() => setInstalling(true)}><Download size={16} />安装技能</button>
    </div> : null}
    <div className="extension-filter-row">
      <div className="filter-tabs skill-filter-tabs" aria-label="技能来源筛选">
        {(Object.keys(filterLabels) as SkillFilter[]).map((item) => <button type="button" key={item} className={filter === item ? "active" : ""} onClick={() => setFilter(item)}>{filterLabels[item]}</button>)}
      </div>
      <span>{rows.length} 项能力</span>
    </div>
    <ErrorBanner message={resource.error || actionError} />
    {notice ? <div className="action-notice">{notice}</div> : null}
    <div className="capability-grid extension-capability-grid skill-grid">
      {rows.map((skill) => <article className="capability-card extension-capability-card skill-card" key={skill.name}>
        <div className="extension-card-head">
          <span className="skill-icon extension-skill-icon"><WandSparkles size={20} /></span>
          <div className="extension-card-identity">
            <h3>{skill.display_name}</h3>
            <small>{skill.name}</small>
          </div>
          <Badge tone={skill.available ? "green" : "amber"}>{skill.available ? "可用" : "缺少依赖"}</Badge>
        </div>
        <p className="extension-card-description">{skill.description || skill.when_to_use || "为小满提供可复用的专业处理方法。"}</p>
        {skill.missing ? <small className="missing-text extension-card-warning">{skill.missing}</small> : null}
        <div className="extension-card-footer">
          <div className="tag-row">
            <Badge tone={skill.origin === "standalone" ? "green" : skill.origin === "workspace" ? "blue" : "gray"}>{skill.source_label}</Badge>
            {skill.always ? <Badge tone="amber">常驻</Badge> : null}
          </div>
          {skill.can_uninstall ? <button type="button" className="skill-uninstall extension-card-action" disabled={busy} onClick={() => setUninstallTarget(skill)}><Trash2 size={14} />卸载</button> : <span className="extension-card-installed">已启用</span>}
        </div>
      </article>)}
    </div>
    {!resource.loading && rows.length === 0 ? <EmptyState icon={WandSparkles} title="没有匹配的技能" text="调整筛选条件，或安装一个独立技能。" /> : null}

    {installing ? <Modal title="安装独立技能" description="填写一个包含技能说明文件的 Git 仓库；技能只提供方法和流程，不会额外安装外部工具。" onClose={() => setInstalling(false)}>
      <div className="form-stack">
        <label>技能仓库<input value={form.source} onChange={(event) => setForm({ ...form, source: event.target.value })} placeholder="owner/skills-repository" autoFocus /></label>
        <label>仓库内技能目录（可选）<input value={form.subdir} onChange={(event) => setForm({ ...form, subdir: event.target.value })} placeholder="skills/find-skills" /></label>
        <label>分支、标签或提交（可选）<input value={form.ref} onChange={(event) => setForm({ ...form, ref: event.target.value })} placeholder="main" /></label>
        <div className="dialog-actions"><button className="secondary-button" onClick={() => setInstalling(false)}>取消</button><button className="primary-button" disabled={busy || !form.source.trim()} onClick={() => void install()}><Download size={16} />{busy ? "安装中" : "安装技能"}</button></div>
      </div>
    </Modal> : null}

    {uninstallTarget ? <Modal title={`卸载 ${uninstallTarget.display_name}`} description="这个独立技能会从能力目录移除，不影响 MCP 工具或系统组件。" onClose={() => setUninstallTarget(null)}>
      <div className="dialog-actions"><button className="secondary-button" onClick={() => setUninstallTarget(null)}>取消</button><button className="primary-button" disabled={busy} onClick={() => void uninstall()}><Trash2 size={16} />{busy ? "卸载中" : "确认卸载"}</button></div>
    </Modal> : null}
  </>;
}
