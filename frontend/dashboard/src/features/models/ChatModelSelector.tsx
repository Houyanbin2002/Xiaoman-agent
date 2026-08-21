import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Check, ChevronDown, Cpu, LoaderCircle, RefreshCw, Search } from "lucide-react";
import { api } from "../../api";
import type { ModelCatalogItem, ModelCatalogResponse, ModelRow, ModelUpdateResponse } from "../../shared/types";

interface ChatModelSelectorProps {
  disabled?: boolean;
}

export function ChatModelSelector(props: ChatModelSelectorProps): React.ReactElement {
  const [current, setCurrent] = useState<ModelRow | null>(null);
  const [items, setItems] = useState<ModelCatalogItem[]>([]);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [savingModel, setSavingModel] = useState("");
  const [error, setError] = useState("");
  const [loadedBaseUrl, setLoadedBaseUrl] = useState("");
  const rootRef = useRef<HTMLDivElement>(null);

  const loadCurrent = useCallback(async (): Promise<ModelRow | null> => {
    setLoading(true);
    setError("");
    try {
      const rows = await api<ModelRow[]>("/api/dashboard/control/models");
      const main = rows.find((row) => row.slot === "main") ?? null;
      setCurrent(main);
      if (!main) setError("主模型尚未配置");
      return main;
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "主模型读取失败");
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const loadCatalog = useCallback(async (model: ModelRow): Promise<void> => {
    setCatalogLoading(true);
    setError("");
    try {
      const result = await api<ModelCatalogResponse>("/api/dashboard/control/models/main/catalog", {
        method: "POST",
        body: JSON.stringify({ base_url: model.base_url, api_key: null }),
      });
      setItems(result.items);
      setLoadedBaseUrl(result.base_url);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "模型列表获取失败");
    } finally {
      setCatalogLoading(false);
    }
  }, []);

  useEffect(() => { void loadCurrent(); }, [loadCurrent]);

  useEffect(() => {
    if (!open) return undefined;
    const close = (event: MouseEvent): void => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const escape = (event: KeyboardEvent): void => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", escape);
    };
  }, [open]);

  const filtered = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    const matches = needle
      ? items.filter((item) => `${item.id} ${item.owned_by}`.toLocaleLowerCase().includes(needle))
      : items;
    if (needle) return matches.slice(0, 80);
    const selected = matches.find((item) => item.id === current?.model);
    const rest = matches.filter((item) => item.id !== current?.model).slice(0, 49);
    return selected ? [selected, ...rest] : rest;
  }, [current?.model, items, query]);

  const toggle = async (): Promise<void> => {
    if (props.disabled || savingModel) return;
    const next = !open;
    setOpen(next);
    setQuery("");
    setError("");
    if (!next) return;
    const main = current ?? await loadCurrent();
    if (main && loadedBaseUrl !== main.base_url && !catalogLoading) await loadCatalog(main);
  };

  const select = async (model: string): Promise<void> => {
    if (!current || model === current.model || savingModel) {
      setOpen(false);
      return;
    }
    setSavingModel(model);
    setError("");
    try {
      const result = await api<ModelUpdateResponse>("/api/dashboard/control/models/main", {
        method: "PATCH",
        body: JSON.stringify({
          model,
          provider: current.provider,
          base_url: current.base_url,
          api_key: null,
        }),
      });
      if (!result.hot_reloaded) throw new Error("模型已保存，但运行时未能热更新");
      setCurrent({ ...current, model: result.model });
      setOpen(false);
      setQuery("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "模型切换失败");
    } finally {
      setSavingModel("");
    }
  };

  const label = current?.model || (loading ? "读取中" : "主模型");
  return (
    <div className={`chat-model-selector${open ? " is-open" : ""}`} ref={rootRef}>
      <button type="button" className="compact-model" aria-haspopup="listbox" aria-expanded={open} aria-label={`当前主模型：${label}`} title={`当前主模型：${label}`} disabled={props.disabled || loading} onClick={() => void toggle()}>
        {savingModel ? <LoaderCircle size={13} className="spin" /> : null}
        <span>{label}</span><ChevronDown size={14} />
      </button>
      {open ? <div className="chat-model-menu">
        <div className="chat-model-menu-head"><span><Cpu size={16} /></span><div><strong>切换主模型</strong><small>下一次发送立即生效</small></div><button type="button" onClick={() => current && void loadCatalog(current)} disabled={catalogLoading} aria-label="刷新模型列表" title="刷新模型列表"><RefreshCw size={14} className={catalogLoading ? "spin" : ""} /></button></div>
        <label className="chat-model-search"><Search size={14} /><input autoFocus value={query} onChange={(event) => setQuery(event.target.value)} placeholder={`搜索 ${items.length || "可用"} 个模型`} /></label>
        {error ? <div className="chat-model-state error">{error}</div> : null}
        {catalogLoading && !items.length ? <div className="chat-model-state"><LoaderCircle size={15} className="spin" />正在获取当前 API 的模型</div> : null}
        {!catalogLoading && !error && !filtered.length ? <div className="chat-model-state">没有匹配的模型</div> : null}
        <div className="chat-model-list" role="listbox" aria-label="主模型列表">
          {filtered.map((item) => <button type="button" role="option" aria-selected={item.id === current?.model} key={item.id} onClick={() => void select(item.id)} disabled={Boolean(savingModel)}><span><strong>{item.id}</strong>{item.owned_by ? <small>{item.owned_by}</small> : null}</span>{savingModel === item.id ? <LoaderCircle size={14} className="spin" /> : item.id === current?.model ? <Check size={14} /> : null}</button>)}
        </div>
        <div className="chat-model-menu-foot">正在执行的回复不受影响</div>
      </div> : null}
    </div>
  );
}
