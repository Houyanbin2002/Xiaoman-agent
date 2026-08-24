import React, { useCallback, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Bot,
  ChevronDown,
  ChevronRight,
  LoaderCircle,
  Menu,
  MessageCircleMore,
  MessagesSquare,
  PanelLeftClose,
  PanelLeftOpen,
  Radio,
  Send,
  Square,
} from "lucide-react";
import { ProductSidebar } from "./app/ProductSidebar";
import { IconButton, LoadingState } from "./shared/components/ui";
import { api } from "./api";
import type { ViewId } from "./shared/types";
import type { PageResult, SessionRow } from "./types";
import "./styles/index.css";

const ChatView = React.lazy(() => import("./features/chat/ChatView").then(({ ChatView: view }) => ({ default: view })));
const TodayView = React.lazy(() => import("./features/today/TodayView").then(({ TodayView: view }) => ({ default: view })));
const OverviewView = React.lazy(() => import("./features/overview/OverviewView").then(({ OverviewView: view }) => ({ default: view })));
const MemoryView = React.lazy(() => import("./features/memory/MemoryView").then(({ MemoryView: view }) => ({ default: view })));
const WorkflowsView = React.lazy(() => import("./features/workflows/WorkflowsView").then(({ WorkflowsView: view }) => ({ default: view })));
const ChannelsView = React.lazy(() => import("./features/channels/ChannelsView").then(({ ChannelsView: view }) => ({ default: view })));
const ModelsView = React.lazy(() => import("./features/models/ModelsView").then(({ ModelsView: view }) => ({ default: view })));
const ExtensionsView = React.lazy(() => import("./features/extensions/ExtensionsView").then(({ ExtensionsView: view }) => ({ default: view })));
const ToolsView = React.lazy(() => import("./features/tools/ToolsView").then(({ ToolsView: view }) => ({ default: view })));
const SchedulesView = React.lazy(() => import("./features/schedules/SchedulesView").then(({ SchedulesView: view }) => ({ default: view })));
const ProactiveView = React.lazy(() => import("./features/proactive/ProactiveView").then(({ ProactiveView: view }) => ({ default: view })));
const SessionsView = React.lazy(() => import("./features/sessions/SessionsView").then(({ SessionsView: view }) => ({ default: view })));
const SettingsHubView = React.lazy(() => import("./features/settings/SettingsHubView").then(({ SettingsHubView: view }) => ({ default: view })));

interface RecentChat {
  chatId: string;
  sessionKey: string;
  title: string;
  updatedAt: string;
  messageCount: number;
  runStatus?: "running";
  channel: string;
  isWeb: boolean;
}

interface ActiveChatRun {
  chat_id: string;
  prompt: string;
  title: string;
  started_at: string;
  status: string;
}

interface ActiveGatewayRun {
  session_key: string;
  channel: string;
  chat_id: string;
  prompt: string;
  status: string;
}

function recentChatFromSession(session: SessionRow): RecentChat | null {
  const [keyChannel, ...chatParts] = session.key.split(":");
  const metadataChannel = typeof session.metadata.gateway_channel === "string" ? session.metadata.gateway_channel : "";
  const rawChannel = metadataChannel || keyChannel || "";
  const channel = rawChannel.startsWith("telegram") ? "telegram" : rawChannel;
  if (!metadataChannel && !["dashboard", "weixin", "wecom", "qqbot", "telegram", "feishu"].includes(channel)) return null;
  const isWeb = channel === "dashboard";
  return {
    chatId: chatParts.join(":") || session.key,
    sessionKey: session.key,
    title: session.title || "新对话",
    updatedAt: session.updated_at,
    messageCount: session.message_count,
    channel,
    isWeb,
  };
}

const conversationGroups = [
  { id: "dashboard", label: "会话", icon: MessagesSquare },
  { id: "weixin", label: "微信", icon: MessageCircleMore },
  { id: "wecom", label: "企业微信", icon: MessageCircleMore },
  { id: "qqbot", label: "QQ", icon: Bot },
  { id: "telegram", label: "Telegram", icon: Send },
  { id: "gateway", label: "其他渠道", icon: Radio },
];

function chatTitle(content: string): string {
  const title = content.replace(/\s+/g, " ").trim();
  return title.length > 32 ? `${title.slice(0, 32).trimEnd()}…` : title || "新对话";
}

const viewMeta: Record<ViewId, { title: string; description: string }> = {
  chat: { title: "对话", description: "和小满聊聊，或者交给他一件需要完成的事。" },
  today: { title: "今天", description: "先看现在最值得投入注意力的事情。" },
  overview: { title: "系统状态", description: "查看小满的模型、连接和运行状态。" },
  memory: { title: "关于我", description: "管理小满长期理解和使用的个人信息。" },
  workflows: { title: "任务", description: "查看正在进行、等待确认和已经完成的事情。" },
  channels: { title: "联系小满", description: "接入 QQ、微信、企业微信和 Telegram。" },
  models: { title: "模型与回复", description: "选择小满在不同任务中使用的模型。" },
  skills: { title: "扩展能力", description: "管理小满的技能和外部工具连接。" },
  mcp: { title: "扩展能力", description: "连接标准外部工具，为小满增加新能力。" },
  tools: { title: "工具目录", description: "检查当前 Agent 可调用的内置与 MCP 工具。" },
  schedules: { title: "提醒与定时", description: "创建和管理明确时间的提醒。" },
  proactive: { title: "主动协助", description: "查看小满正在留意什么，以及为什么联系你。" },
  sessions: { title: "任务航迹", description: "查看一项任务是如何完成的。" },
  settings: { title: "设置与扩展", description: "管理小满的回复、连接、能力和系统状态。" },
};

function initialView(): ViewId {
  const raw = window.location.hash.replace(/^#/, "");
  const candidate = (raw === "plugins" ? "mcp" : raw) as ViewId;
  return candidate in viewMeta ? candidate : "chat";
}

function App(): React.ReactElement {
  const [view, setView] = useState<ViewId>(initialView);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [mobileNav, setMobileNav] = useState(false);
  const [chatId, setChatId] = useState(() => `xiaoman-console-${crypto.randomUUID()}`);
  const [recentChats, setRecentChats] = useState<RecentChat[]>([]);
  const [optimisticChats, setOptimisticChats] = useState<RecentChat[]>([]);
  const [activeRuns, setActiveRuns] = useState<ActiveChatRun[]>([]);
  const [activeGatewayRuns, setActiveGatewayRuns] = useState<ActiveGatewayRun[]>([]);
  const [selectedSessionKey, setSelectedSessionKey] = useState<string>();
  const [collapsedConversations, setCollapsedConversations] = useState<Record<string, boolean>>({});
  const loadDashboardState = useCallback(async (): Promise<void> => {
    try {
      const [sessions, runs, gatewayRuns] = await Promise.all([
        api<PageResult<SessionRow>>("/api/dashboard/sessions?page=1&page_size=120&sort_by=updated_at&sort_order=desc"),
        api<{ items: ActiveChatRun[] }>("/api/dashboard/chat/runs"),
        api<{ items: ActiveGatewayRun[] }>("/api/dashboard/chat/gateway-runs"),
      ]);
      const chats = sessions.items.map(recentChatFromSession).filter((item): item is RecentChat => item !== null);
      setRecentChats(chats);
      setActiveRuns(runs.items);
      setActiveGatewayRuns(gatewayRuns.items);
      setOptimisticChats((current) => current.filter((item) => !chats.some((saved) => saved.chatId === item.chatId)));
    } catch {
      // Keep the last known sidebar state during a transient gateway reconnect.
    }
  }, []);
  useEffect(() => {
    void loadDashboardState();
    const timer = window.setInterval(() => void loadDashboardState(), 2500);
    return () => window.clearInterval(timer);
  }, [loadDashboardState]);
  const sidebarChats = useMemo(
    () => {
      const merged = new Map<string, RecentChat>();
      for (const chat of recentChats) merged.set(chat.sessionKey, chat);
      for (const chat of optimisticChats) {
        const current = merged.get(chat.sessionKey);
        merged.set(chat.sessionKey, current ? { ...current, ...chat, messageCount: Math.max(current.messageCount, chat.messageCount) } : chat);
      }
      for (const run of activeRuns) {
        const sessionKey = `dashboard:${run.chat_id}`;
        const current = merged.get(sessionKey);
        merged.set(sessionKey, {
          chatId: run.chat_id,
          sessionKey: `dashboard:${run.chat_id}`,
          title: current?.title || run.title || chatTitle(run.prompt),
          updatedAt: run.started_at,
          messageCount: Math.max(1, current?.messageCount ?? 0),
          runStatus: "running",
          channel: "dashboard",
          isWeb: true,
        });
      }
      for (const run of activeGatewayRuns) {
        const current = merged.get(run.session_key);
        if (current) merged.set(run.session_key, { ...current, runStatus: "running" });
      }
      return [...merged.values()].sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
    },
    [activeGatewayRuns, activeRuns, optimisticChats, recentChats],
  );
  const groupedChats = useMemo(() => {
    const groups = new Map<string, RecentChat[]>();
    for (const chat of sidebarChats) {
      const group = conversationGroups.some((item) => item.id === chat.channel) ? chat.channel : "gateway";
      groups.set(group, [...(groups.get(group) ?? []), chat]);
    }
    return groups;
  }, [sidebarChats]);
  const navigate = (next: ViewId): void => {
    setView(next);
    setMobileNav(false);
    window.history.replaceState(null, "", `#${next}`);
  };
  const startNewChat = (): void => {
    const nextChatId = `xiaoman-console-${crypto.randomUUID()}`;
    const nextChat = { chatId: nextChatId, sessionKey: `dashboard:${nextChatId}`, title: "新对话", updatedAt: new Date().toISOString(), messageCount: 0, channel: "dashboard", isWeb: true } satisfies RecentChat;
    setChatId(nextChatId);
    setOptimisticChats((current) => [nextChat, ...current.filter((item) => item.chatId !== nextChatId && item.messageCount > 0)]);
    navigate("chat");
  };
  const openChat = (chat: RecentChat): void => {
    if (chat.isWeb) {
      setChatId(chat.chatId);
      navigate("chat");
      return;
    }
    setSelectedSessionKey(chat.sessionKey);
    navigate("sessions");
  };
  const registerChatStart = useCallback((activeChatId: string, firstMessage: string): void => {
    setOptimisticChats((current) => {
      const existing = current.find((item) => item.chatId === activeChatId) ?? recentChats.find((item) => item.chatId === activeChatId);
      const next = {
        chatId: activeChatId,
        sessionKey: `dashboard:${activeChatId}`,
        title: existing && existing.title !== "新对话" ? existing.title : chatTitle(firstMessage),
        updatedAt: new Date().toISOString(),
        messageCount: Math.max(1, (existing?.messageCount ?? 0) + 1),
        runStatus: "running",
        channel: "dashboard",
        isWeb: true,
      } satisfies RecentChat;
      return [next, ...current.filter((item) => item.chatId !== activeChatId)];
    });
  }, [recentChats]);
  const updateChatRunState = useCallback((activeChatId: string, running: boolean, firstMessage?: string): void => {
    setOptimisticChats((current) => current.map((item) => item.chatId === activeChatId ? { ...item, title: firstMessage ? chatTitle(firstMessage) : item.title, runStatus: running ? "running" : undefined } : item));
    if (!running) setActiveRuns((current) => current.filter((run) => run.chat_id !== activeChatId));
  }, []);
  const stopChatRun = useCallback(async (chat: RecentChat): Promise<void> => {
    try {
      await api(`/api/dashboard/chat/runs/stop?session_key=${encodeURIComponent(chat.sessionKey)}`, { method: "POST" });
    } finally {
      await loadDashboardState();
    }
  }, [loadDashboardState]);
  const conversationRail = <section className="chat-history conversation-rail" aria-label="聊天记录">{conversationGroups.map((group) => {
    const chats = groupedChats.get(group.id) ?? [];
    const GroupIcon = group.icon;
    const collapsed = Boolean(collapsedConversations[group.id]);
    if (!chats.length && group.id !== "dashboard") return null;
    return <div className={`conversation-group channel-${group.id}`} key={group.id}>
      <button type="button" className="conversation-group-trigger" onClick={() => setCollapsedConversations((current) => ({ ...current, [group.id]: !current[group.id] }))} aria-expanded={!collapsed}><GroupIcon size={13} /><span>{group.label}</span><small>{chats.length}</small>{collapsed ? <ChevronRight size={12} /> : <ChevronDown size={12} />}</button>
      {!collapsed ? <div className={`conversation-group-list ${group.id === "dashboard" ? "web-list" : ""}`}>{chats.length ? chats.map((chat) => <div className="conversation-entry" key={chat.sessionKey}>
        <button type="button" className={`conversation-open ${(chat.isWeb ? view === "chat" && chat.chatId === chatId : view === "sessions" && chat.sessionKey === selectedSessionKey) ? "active" : ""}`} onClick={() => openChat(chat)} title={`${chat.title} · ${chat.messageCount} 条消息${chat.runStatus ? " · 运行中" : ""}`}><span className="conversation-dot" /><span className="chat-history-title">{chat.title}</span>{chat.runStatus ? <LoaderCircle className="chat-running" size={12} /> : null}</button>
        {chat.runStatus ? <button type="button" className="conversation-stop" onClick={() => void stopChatRun(chat)} title="停止这个任务" aria-label={`停止 ${chat.title}`}><Square size={9} fill="currentColor" /></button> : null}
      </div>) : <span className="chat-history-empty">发送第一条消息后，会话会保存在这里</span>}</div> : null}
    </div>;
  })}</section>;
  const content = (() => {
    switch (view) {
      case "chat": return <ChatView key={chatId} chatId={chatId} onStarted={registerChatStart} onRunState={updateChatRunState} onActivity={loadDashboardState} />;
      case "today": return <TodayView navigate={navigate} />;
      case "overview": return <OverviewView navigate={navigate} />;
      case "memory": return <MemoryView />;
      case "workflows": return <WorkflowsView />;
      case "channels": return <ChannelsView />;
      case "models": return <ModelsView />;
      case "skills":
      case "mcp": return <ExtensionsView activeTab={view} onTabChange={navigate} />;
      case "tools": return <ToolsView />;
      case "schedules": return <SchedulesView />;
      case "proactive": return <ProactiveView />;
      case "sessions": return <SessionsView initialSessionKey={selectedSessionKey} />;
      case "settings": return <SettingsHubView navigate={navigate} />;
    }
  })();
  return (
    <div className="desktop-app">
      <div className={`app-shell ${sidebarOpen ? "" : "sidebar-collapsed"}`} data-view={view}>
      {mobileNav ? <button className="nav-backdrop" onClick={() => setMobileNav(false)} aria-label="关闭导航" /> : null}
      <ProductSidebar view={view} mobileOpen={mobileNav} conversationRail={conversationRail} onNavigate={navigate} onNewChat={startNewChat} onCloseMobile={() => setMobileNav(false)} />
      <main className="main-shell">
        <header className="app-topbar">
          <div className="topbar-left"><button className="mobile-menu" onClick={() => setMobileNav(true)} aria-label="打开导航" title="打开导航"><Menu size={19} /></button><IconButton icon={sidebarOpen ? PanelLeftClose : PanelLeftOpen} label={sidebarOpen ? "收起侧栏" : "展开侧栏"} onClick={() => setSidebarOpen((value) => !value)} /><span className="product-topbar-date">{new Intl.DateTimeFormat("zh-CN", { year: "numeric", month: "long", day: "numeric", weekday: "long" }).format(new Date())}</span></div>
        </header>
        <div className={`content-shell ${view === "chat" ? "chat-content" : ""}`}>
          <React.Suspense fallback={<LoadingState />}>{content}</React.Suspense>
        </div>
      </main>
      </div>
    </div>
  );
}

const rootContainer = document.getElementById("root") as HTMLElement & { __xiaomanRoot?: ReturnType<typeof createRoot> };
const dashboardRoot = rootContainer.__xiaomanRoot ?? createRoot(rootContainer);
rootContainer.__xiaomanRoot = dashboardRoot;
dashboardRoot.render(<React.StrictMode><App /></React.StrictMode>);
