import React from "react";
import { Settings2, SquarePen, X } from "lucide-react";
import { IconButton } from "../shared/components/ui";
import type { ViewId } from "../shared/types";
import { activeProductView, productNavItems } from "./navigation";

interface ProductSidebarProps {
  view: ViewId;
  mobileOpen: boolean;
  conversationRail: React.ReactNode;
  onNavigate: (view: ViewId) => void;
  onNewChat: () => void;
  onCloseMobile: () => void;
}

export function ProductSidebar(props: ProductSidebarProps): React.ReactElement {
  const activeView = activeProductView(props.view);
  const showConversations = activeView === "chat";
  return (
    <aside className={`sidebar product-sidebar ${props.mobileOpen ? "mobile-open" : ""}`} aria-label="主导航">
      <div className="product-brand">
        <img src="/assets/assets/xiaoman-avatar.png" alt="小满" />
        <div><strong>XiaoMan</strong><span>懂你的个人智能</span></div>
        <div className="brand-actions"><IconButton icon={X} label="关闭导航" onClick={props.onCloseMobile} /></div>
      </div>
      {showConversations ? <button type="button" className="product-new-chat" onClick={props.onNewChat}><SquarePen size={17} /><span>新建对话</span></button> : null}
      <nav className="product-nav" aria-label="日常功能">
        {productNavItems.map((item) => {
          const Icon = item.icon;
          const active = activeView === item.id;
          return <button type="button" key={item.id} className={active ? "active" : ""} aria-current={active ? "page" : undefined} onClick={() => props.onNavigate(item.id)}><Icon size={18} /><span>{item.label}</span></button>;
        })}
      </nav>
      {showConversations ? <div className="product-conversation-slot">{props.conversationRail}</div> : <div className="product-sidebar-space" />}
      <button type="button" className={`product-settings ${activeView === "settings" ? "active" : ""}`} onClick={() => props.onNavigate("settings")}><Settings2 size={18} /><span>设置与扩展</span></button>
    </aside>
  );
}
