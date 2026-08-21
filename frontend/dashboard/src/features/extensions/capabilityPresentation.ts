import type { ExternalSourceRow, McpRow } from "../../shared/types";

export interface CapabilityBrand {
  label: string;
  logo: string | null;
  accent: string;
}

export type CapabilitySourceState = "active" | "paused" | "none";
export type CapabilityStatusTone = "green" | "red" | "amber" | "gray";

const GENERIC_BRAND: CapabilityBrand = {
  label: "工具连接",
  logo: null,
  accent: "#5a55c9",
};

const BRANDS: Array<{ keys: string[]; brand: CapabilityBrand }> = [
  {
    keys: ["notion"],
    brand: {
      label: "Notion",
      logo: "/assets/assets/brands/notion.svg",
      accent: "#111111",
    },
  },
  {
    keys: ["gmail", "google mail"],
    brand: {
      label: "Gmail",
      logo: "/assets/assets/brands/gmail.svg",
      accent: "#ea4335",
    },
  },
  {
    keys: ["github"],
    brand: {
      label: "GitHub",
      logo: "/assets/assets/brands/github.svg",
      accent: "#181717",
    },
  },
  {
    keys: ["obsidian", "mcpvault"],
    brand: {
      label: "Obsidian",
      logo: "/assets/assets/brands/obsidian.svg",
      accent: "#7c3aed",
    },
  },
  {
    keys: ["markitdown", "microsoft", "文档解析"],
    brand: {
      label: "文档解析",
      logo: "/assets/assets/brands/microsoft.svg",
      accent: "#2563eb",
    },
  },
];

export function brandForCapability(
  name: string,
  provider = "",
): CapabilityBrand {
  const identity = `${name} ${provider}`.trim().toLowerCase();
  return BRANDS.find(({ keys }) => keys.some((key) => identity.includes(key)))?.brand
    ?? { ...GENERIC_BRAND, label: name || GENERIC_BRAND.label };
}

export function sourceStateForServer(
  serverName: string,
  sources: ExternalSourceRow[],
): CapabilitySourceState {
  const matching = sources.filter((source) => source.server_name === serverName);
  if (matching.some((source) => source.enabled)) return "active";
  return matching.length ? "paused" : "none";
}

export function statusPresentation(
  status: McpRow["status"],
): { label: string; tone: CapabilityStatusTone } {
  switch (status) {
    case "connected": return { label: "已连接", tone: "green" };
    case "connecting": return { label: "连接中", tone: "amber" };
    case "authorizing": return { label: "授权中", tone: "amber" };
    case "authorization_required": return { label: "待授权", tone: "amber" };
    case "error": return { label: "连接失败", tone: "red" };
    case "disconnected": return { label: "未连接", tone: "gray" };
  }
}
