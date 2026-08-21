import { api } from "../api";

interface GatewayStatus {
  status: string;
  instance_id: string;
}

interface GatewayRestartAccepted {
  accepted: boolean;
  instance_id: string;
}

export async function restartGatewayAndWait(): Promise<void> {
  const accepted = await api<GatewayRestartAccepted>("/api/dashboard/control/gateway/restart", { method: "POST" });
  const deadline = Date.now() + 90_000;
  while (Date.now() < deadline) {
    await new Promise((resolve) => window.setTimeout(resolve, 850));
    try {
      const status = await api<GatewayStatus>("/api/dashboard/control/gateway/status", { signal: AbortSignal.timeout(3_000) });
      if (status.status === "running" && status.instance_id !== accepted.instance_id) return;
    } catch {
      // The short connection gap is expected while the gateway releases the port.
    }
  }
  throw new Error("网关未在 90 秒内恢复，请检查启动日志。");
}
