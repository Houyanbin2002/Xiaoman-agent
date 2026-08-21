import React, { useState } from "react";
import { BrainCircuit, Cpu, Save, Zap } from "lucide-react";
import { api } from "../../api";
import { Badge, ErrorBanner, LoadingState, Modal, PageIntro } from "../../shared/components/ui";
import { useAsyncData } from "../../shared/hooks/useAsyncData";
import type { ModelRow, ModelUpdateResponse } from "../../shared/types";
import { ModelCatalogPicker } from "./ModelCatalogPicker";

function modelLabel(model: ModelRow): string {
  if (model.slot === "agent" || model.slot === "subagent") return "复杂任务模型";
  if (model.slot === "memory") return "记忆检索模型";
  return model.label;
}

function modelUsage(model: ModelRow): string {
  if (model.slot === "agent" || model.slot === "subagent") return "负责拆分复杂任务、并行处理和后台工作";
  if (model.slot === "memory") return "负责从长期记忆中找回与当前问题相关的内容";
  return model.usage;
}

export function ModelsView(): React.ReactElement {
  const resource = useAsyncData(() => api<ModelRow[]>("/api/dashboard/control/models"), []);
  const [editing, setEditing] = useState<ModelRow | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [saving, setSaving] = useState(false);
  const [actionError, setActionError] = useState("");
  const [notice, setNotice] = useState("");

  const openEditor = (model: ModelRow): void => {
    setEditing({ ...model });
    setApiKey("");
    setActionError("");
  };

  const save = async (): Promise<void> => {
    if (!editing) return;
    setSaving(true);
    setActionError("");
    try {
      const result = await api<ModelUpdateResponse>(`/api/dashboard/control/models/${editing.slot}`, {
        method: "PATCH",
        body: JSON.stringify({
          model: editing.model,
          provider: editing.provider,
          base_url: editing.base_url,
          api_key: apiKey || null,
          output_dimensionality: editing.slot === "memory" ? editing.output_dimensionality ?? 1024 : null,
        }),
      });
      const label = modelLabel(editing);
      setNotice(result.hot_reloaded ? `${label}已热更新，下一次请求直接使用 ${result.model}` : `${label}已保存`);
      setEditing(null);
      setApiKey("");
      resource.reload();
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSaving(false);
    }
  };

  return <>
    <PageIntro
      title="模型与回复"
      description="为聊天、图片理解、复杂任务和记忆检索选择合适的模型；保存后立即用于新请求。"
      actions={<Badge tone="green"><Zap size={11} />运行时热更新</Badge>}
    />
    <ErrorBanner message={resource.error || actionError} />
    {notice ? <div className="model-update-notice"><Zap size={15} />{notice}</div> : null}
    {resource.loading && !resource.data ? <LoadingState /> : (
      <div className="model-grid">
        {resource.data?.map((model) => {
          const ModelIcon = model.slot === "memory" ? BrainCircuit : Cpu;
          return (
            <button className={`model-card${model.slot === "memory" ? " memory-model-card" : ""}`} key={model.slot} onClick={() => openEditor(model)}>
              <div className="model-card-head"><span className="model-icon"><ModelIcon size={20} /></span><Badge tone={model.api_key_configured ? "green" : "amber"}>{model.api_key_configured ? "密钥已配置" : "缺少密钥"}</Badge></div>
              <h3>{modelLabel(model)}</h3>
              <strong>{model.model || "尚未配置"}</strong>
              <p className="model-card-usage">{modelUsage(model)}</p>
              <small>{model.slot === "memory" ? `${model.output_dimensionality ?? 1024} 维语义检索` : "点击切换模型或调整接口"}</small>
            </button>
          );
        })}
      </div>
    )}
    {editing ? (
      <Modal
        title={`编辑${modelLabel(editing)}`}
        description={`${modelUsage(editing)}。模型目录从当前接口实时获取，保存后立即生效。`}
        onClose={() => setEditing(null)}
      >
        <div className="form-stack model-editor-form">
          <label>模型名称
            <ModelCatalogPicker
              slot={editing.slot}
              value={editing.model}
              baseUrl={editing.base_url}
              apiKey={apiKey}
              onChange={(model) => setEditing({ ...editing, model })}
            />
          </label>
          <div className="model-connection-grid">
            <label>服务类型<input value={editing.provider} onChange={(event) => setEditing({ ...editing, provider: event.target.value })} /></label>
            {editing.slot === "memory" ? <label>向量维度<input type="number" min="64" max="4096" step="64" value={editing.output_dimensionality ?? 1024} onChange={(event) => setEditing({ ...editing, output_dimensionality: Number(event.target.value) || 1024 })} /></label> : null}
          </div>
          <label>接口地址<input value={editing.base_url} onChange={(event) => setEditing({ ...editing, base_url: event.target.value })} /></label>
          <label>新 API Key<input type="password" autoComplete="new-password" placeholder="留空则使用当前密钥" value={apiKey} onChange={(event) => setApiKey(event.target.value)} /></label>
          <div className="model-runtime-note"><Zap size={15} /><span>正在执行的任务保持原连接；保存后的新请求立即切换到新模型，不中断当前任务。</span></div>
          <div className="dialog-actions">
            <button className="secondary-button" onClick={() => setEditing(null)}>取消</button>
            <button className="primary-button" onClick={() => void save()} disabled={saving || !editing.model.trim() || !editing.base_url.trim()}><Save size={16} />{saving ? "热更新中" : "保存并热更新"}</button>
          </div>
        </div>
      </Modal>
    ) : null}
  </>;
}
