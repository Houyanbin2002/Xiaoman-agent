import React, { useState } from "react";
import { McpView } from "../mcp/McpView";
import { SkillsView } from "../skills/SkillsView";
import { PageIntro } from "../../shared/components/ui";
import { MarketplaceView } from "./MarketplaceView";

interface ExtensionsViewProps {
  activeTab: "skills" | "mcp";
  onTabChange: (tab: "skills" | "mcp") => void;
}

export function ExtensionsView({
  activeTab,
  onTabChange,
}: ExtensionsViewProps): React.ReactElement {
  const [mode, setMode] = useState<"installed" | "market">("installed");
  const openInstalled = (kind: "skills" | "mcp" | "skill"): void => {
    setMode("installed");
    onTabChange(kind === "skill" ? "skills" : kind);
  };
  return (
    <div className="extensions-view">
      <PageIntro
        title="扩展能力"
        description="为小满增加处理方法，以及可以实际使用的外部应用和工具。"
      />
      <div className="extension-tabs extension-mode-tabs" role="tablist" aria-label="扩展能力区域">
        <button type="button" role="tab" aria-selected={mode === "installed"} className={mode === "installed" ? "active" : ""} onClick={() => setMode("installed")}>已安装</button>
        <button type="button" role="tab" aria-selected={mode === "market"} className={mode === "market" ? "active" : ""} onClick={() => setMode("market")}>市场</button>
      </div>
      {mode === "installed" ? <>
      <div className="extension-tabs extension-kind-tabs" role="tablist" aria-label="已安装能力类型">
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === "skills"}
          className={activeTab === "skills" ? "active" : ""}
          onClick={() => onTabChange("skills")}
        >
          技能
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === "mcp"}
          className={activeTab === "mcp" ? "active" : ""}
          onClick={() => onTabChange("mcp")}
        >
          工具连接
        </button>
      </div>
      <section
        className="extension-tab-panel"
        role="tabpanel"
        aria-label={activeTab === "skills" ? "技能" : "工具连接"}
      >
        {activeTab === "skills" ? <SkillsView embedded /> : <McpView embedded showCatalog={false} />}
      </section>
      </> : <MarketplaceView onOpenInstalled={(kind) => openInstalled(kind)} />}
    </div>
  );
}
