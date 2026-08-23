import React, { useEffect, useState } from "react";
import { Check, LoaderCircle, MessageCircleMore } from "lucide-react";

import { api } from "../../api";
import type { ConversationStyleResponse } from "../../shared/types";

const fallbackStyles: ConversationStyleResponse = {
  active_style: "balanced",
  styles: [
    { id: "balanced", name: "自然", description: "根据问题自动调节详略与语气" },
    { id: "concise", name: "简洁", description: "先给结论，只保留必要信息" },
    { id: "warm", name: "温和", description: "耐心、有共情，但不说空话" },
    { id: "professional", name: "专业", description: "结构清楚，措辞严谨，依据明确" },
    { id: "candid", name: "直率", description: "明确判断，直接指出问题与取舍" },
    { id: "lively", name: "活泼", description: "轻松有趣，表达更有节奏" },
  ],
};

export function ConversationStyleSettings(): React.ReactElement {
  const [catalog, setCatalog] = useState(fallbackStyles);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    let mounted = true;
    api<ConversationStyleResponse>("/api/dashboard/control/conversation-styles")
      .then((payload) => {
        if (mounted) setCatalog(payload);
      })
      .catch((reason: unknown) => {
        if (mounted) setError(reason instanceof Error ? reason.message : "风格设置暂时不可用");
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
    return () => { mounted = false; };
  }, []);

  const selectStyle = async (styleId: string): Promise<void> => {
    if (styleId === catalog.active_style) return;
    setSaving(styleId);
    setError("");
    try {
      const payload = await api<ConversationStyleResponse>("/api/dashboard/control/conversation-styles", {
        method: "PATCH",
        body: JSON.stringify({ style_id: styleId }),
      });
      setCatalog(payload);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "切换失败，请稍后重试");
    } finally {
      setSaving("");
    }
  };

  return (
    <section className="settings-style-card" aria-labelledby="conversation-style-title">
      <div className="settings-style-head">
        <span><MessageCircleMore size={21} /></span>
        <div><h2 id="conversation-style-title">对话风格</h2><p>选择小满与你交流时的表达方式，从下一条回复开始生效。</p></div>
        {loading ? <LoaderCircle className="spin" size={17} aria-label="正在读取对话风格" /> : null}
      </div>
      <div className="settings-style-grid" role="radiogroup" aria-label="对话风格">
        {catalog.styles.map((style) => {
          const selected = style.id === catalog.active_style;
          return <button
            type="button"
            role="radio"
            aria-checked={selected}
            className="settings-style-option"
            key={style.id}
            disabled={Boolean(saving)}
            onClick={() => void selectStyle(style.id)}
          >
            <span><strong>{style.name}</strong><small>{style.description}</small></span>
            {saving === style.id ? <LoaderCircle className="spin" size={16} /> : selected ? <Check size={16} /> : null}
          </button>;
        })}
      </div>
      <p className={`settings-style-note ${error ? "error" : ""}`}>{error || "只改变表达方式，不影响能力、记忆、权限与事实标准。"}</p>
    </section>
  );
}
