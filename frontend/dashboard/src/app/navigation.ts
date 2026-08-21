import { ListChecks, MessagesSquare, Sparkles, Sunrise, UserRound, type LucideIcon } from "lucide-react";
import type { ViewId } from "../shared/types";

export interface ProductNavItem {
  id: ViewId;
  label: string;
  icon: LucideIcon;
}

export const productNavItems: ProductNavItem[] = [
  { id: "today", label: "今天", icon: Sunrise },
  { id: "chat", label: "对话", icon: MessagesSquare },
  { id: "proactive", label: "主动协助", icon: Sparkles },
  { id: "workflows", label: "任务", icon: ListChecks },
  { id: "memory", label: "关于我", icon: UserRound },
];

const settingsViews = new Set<ViewId>(["settings", "overview", "channels", "models", "skills", "mcp", "tools", "schedules"]);

export function activeProductView(view: ViewId): ViewId {
  if (view === "sessions") return "chat";
  if (settingsViews.has(view)) return "settings";
  return view;
}
