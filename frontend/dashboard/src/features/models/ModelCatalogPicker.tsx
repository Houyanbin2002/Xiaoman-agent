import React, { useEffect, useMemo, useState } from "react";
import { Check, ChevronDown, Database, LoaderCircle, RefreshCw, Search } from "lucide-react";
import { api } from "../../api";
import type { ModelCatalogItem, ModelCatalogResponse } from "../../shared/types";

interface ModelCatalogPickerProps {
  slot: string;
  value: string;
  baseUrl: string;
  apiKey: string;
  onChange: (value: string) => void;
}

export function ModelCatalogPicker(props: ModelCatalogPickerProps): React.ReactElement {
  const [expanded, setExpanded] = useState(false);
  const [query, setQuery] = useState("");
  const [items, setItems] = useState<ModelCatalogItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [loadedFor, setLoadedFor] = useState("");
  const connectionKey = `${props.slot}:${props.baseUrl}`;

  useEffect(() => {
    setItems([]);
    setLoadedFor("");
    setError("");
  }, [connectionKey]);

  const loadCatalog = async (): Promise<void> => {
    if (!props.baseUrl.trim()) {
      setError("请先填写 Base URL。");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const result = await api<ModelCatalogResponse>(`/api/dashboard/control/models/${props.slot}/catalog`, {
        method: "POST",
        body: JSON.stringify({ base_url: props.baseUrl, api_key: props.apiKey || null }),
      });
      setItems(result.items);
      setLoadedFor(connectionKey);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setLoading(false);
    }
  };

  const toggle = (): void => {
    const next = !expanded;
    setExpanded(next);
    setQuery("");
    if (next && loadedFor !== connectionKey && !loading) void loadCatalog();
  };

  const filtered = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    const candidates = needle
      ? items.filter((item) => `${item.id} ${item.owned_by}`.toLocaleLowerCase().includes(needle))
      : items;
    if (props.slot !== "memory") return candidates;
    return [...candidates].sort((left, right) => {
      const embedding = (value: string): number => /embed|embedding|bge|gte/i.test(value) ? 0 : 1;
      return embedding(left.id) - embedding(right.id) || left.id.localeCompare(right.id);
    });
  }, [items, props.slot, query]);
  const displayed = useMemo(() => {
    if (query.trim()) return filtered.slice(0, 80);
    const selected = filtered.find((item) => item.id === props.value);
    const rest = filtered.filter((item) => item.id !== props.value);
    return selected ? [selected, ...rest.slice(0, 49)] : rest.slice(0, 50);
  }, [filtered, props.value, query]);

  const selectModel = (model: string): void => {
    props.onChange(model);
    setExpanded(false);
    setQuery("");
  };
  const manualCandidate = query.trim();
  const hasExactMatch = items.some((item) => item.id === manualCandidate);

  return (
    <div className={`model-catalog-picker${expanded ? " is-open" : ""}`}>
      <button
        type="button"
        className="model-catalog-trigger"
        aria-expanded={expanded}
        aria-haspopup="listbox"
        onClick={toggle}
      >
        <span className="model-catalog-current">
          <Database size={15} />
          <span><small>当前选择</small><strong>{props.value || "选择模型"}</strong></span>
        </span>
        <ChevronDown size={17} />
      </button>
      {expanded ? (
        <div className="model-catalog-popover">
          <div className="model-catalog-toolbar">
            <label className="model-catalog-search">
              <Search size={15} />
              <input
                autoFocus
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="搜索模型，或输入模型 ID"
              />
            </label>
            <button type="button" className="model-catalog-refresh" aria-label="重新获取模型" onClick={() => void loadCatalog()} disabled={loading}>
              <RefreshCw size={15} className={loading ? "spin" : ""} />
            </button>
          </div>
          <div className="model-catalog-meta">
            <span>{loading ? "正在从当前 API 获取模型…" : `当前接口返回 ${items.length} 个模型`}</span>
            <span>{filtered.length > displayed.length ? `先显示 ${displayed.length} 个，搜索可查看更多` : `显示 ${displayed.length} 个`}</span>
          </div>
          <div className="model-catalog-list" role="listbox" aria-label="可用模型">
            {loading && !items.length ? <div className="model-catalog-state"><LoaderCircle size={16} className="spin" />正在载入模型目录</div> : null}
            {error ? <div className="model-catalog-state is-error">{error}<small>仍可在搜索框输入模型 ID 后直接使用。</small></div> : null}
            {!loading && !error && !filtered.length && !manualCandidate ? <div className="model-catalog-state">当前搜索没有结果</div> : null}
            {displayed.map((item) => (
              <button type="button" role="option" aria-selected={item.id === props.value} key={item.id} className="model-catalog-option" onClick={() => selectModel(item.id)}>
                <span><strong>{item.id}</strong>{item.owned_by ? <small>{item.owned_by}</small> : null}</span>
                {item.id === props.value ? <Check size={15} /> : null}
              </button>
            ))}
            {manualCandidate && !hasExactMatch ? (
              <button type="button" className="model-catalog-option is-manual" onClick={() => selectModel(manualCandidate)}>
                <span><small>直接使用模型 ID</small><strong>{manualCandidate}</strong></span>
                <Check size={15} />
              </button>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}
