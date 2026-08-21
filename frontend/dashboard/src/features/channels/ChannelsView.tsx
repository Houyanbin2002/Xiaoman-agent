import React, { useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  Bot,
  ExternalLink,
  Gauge,
  MessageCircleMore,
  Network,
  Radio,
  RefreshCw,
  Save,
  ShieldCheck,
  Smartphone,
} from "lucide-react";
import { api } from "../../api";
import { Badge, ErrorBanner, Modal, PageIntro } from "../../shared/components/ui";
import { useAsyncData } from "../../shared/hooks/useAsyncData";
import { restartGatewayAndWait } from "../../shared/gateway";
import type { ChannelRow } from "../../shared/types";

interface ChannelDraft {
  row: ChannelRow;
  appId: string;
  clientSecret: string;
  botId: string;
  secret: string;
  token: string;
  allowFrom: string;
}

interface WeixinQrState {
  flowId: string;
  image: string;
  status: string;
  error?: string;
}

interface ChannelSaveResult {
  saved: boolean;
  restart_required: boolean;
  channel: ChannelRow;
}

function makeDraft(row: ChannelRow): ChannelDraft {
  return {
    row,
    appId: row.fields?.app_id?.value ?? "",
    clientSecret: "",
    botId: row.fields?.bot_id?.value ?? "",
    secret: "",
    token: "",
    allowFrom: row.allow_from.join(", "),
  };
}

export function ChannelsView(): React.ReactElement {
  const resource = useAsyncData(() => api<ChannelRow[]>("/api/dashboard/control/channels"), []);
  const [draft, setDraft] = useState<ChannelDraft | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState("");
  const [notice, setNotice] = useState("");
  const [restartError, setRestartError] = useState("");
  const [restartConfirm, setRestartConfirm] = useState(false);
  const [restarting, setRestarting] = useState(false);
  const [weixinQr, setWeixinQr] = useState<WeixinQrState | null>(null);
  const rows = useMemo(() => resource.data ?? [], [resource.data]);

  const save = async (): Promise<void> => {
    if (!draft) return;
    const validationError = validateDraft(draft);
    if (validationError) {
      setSaveError(validationError);
      return;
    }
    setSaveError("");
    setSaving(true);
    try {
      const allowFrom = draft.allowFrom.split(",").map((item) => item.trim()).filter(Boolean);
      const result = await api<ChannelSaveResult>(`/api/dashboard/control/channels/${draft.row.id}`, {
        method: "PATCH",
        body: JSON.stringify({
          enabled: true,
          token: draft.token || null,
          app_id: draft.appId,
          client_secret: draft.clientSecret || null,
          bot_id: draft.botId,
          secret: draft.secret || null,
          allow_from: allowFrom,
        }),
      });
      setDraft(null);
      setNotice(result.restart_required ? "配置已保存，卡片状态已更新；重启小满后建立渠道连接。" : "配置已保存并生效。");
      resource.reload();
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : "保存失败，请稍后重试");
    } finally {
      setSaving(false);
    }
  };

  const beginWeixinQr = async (): Promise<void> => {
    setWeixinQr({ flowId: "", image: "", status: "loading" });
    try {
      const payload = await api<{ flow_id: string; image: string; status: string }>("/api/dashboard/control/channels/weixin/qr", { method: "POST" });
      setWeixinQr({ flowId: payload.flow_id, image: payload.image, status: payload.status });
    } catch (error) {
      setWeixinQr({ flowId: "", image: "", status: "error", error: error instanceof Error ? error.message : "二维码获取失败" });
    }
  };

  const restartGateway = async (): Promise<void> => {
    setRestartConfirm(false);
    setRestartError("");
    setRestarting(true);
    setNotice("正在重新连接聊天渠道，页面会短暂断开…");
    try {
      await restartGatewayAndWait();
      setNotice("聊天渠道已重新连接，最新配置已经加载。");
      resource.reload();
    } catch (error) {
      setNotice("");
      setRestartError(error instanceof Error ? error.message : "重新连接失败，请稍后重试");
    } finally {
      setRestarting(false);
    }
  };

  useEffect(() => {
    if (!weixinQr?.flowId || ["confirmed", "expired", "error"].includes(weixinQr.status)) return;
    const timer = window.setInterval(() => {
      void api<{ status: string; account_id?: string }>(`/api/dashboard/control/channels/weixin/qr/${encodeURIComponent(weixinQr.flowId)}`)
        .then((payload) => {
          setWeixinQr((current) => current ? { ...current, status: payload.status } : current);
          if (payload.status === "confirmed") {
            setNotice("微信扫码授权成功，凭据已加密保存在系统凭据库；重启小满后生效。");
            resource.reload();
          }
        })
        .catch((error) => setWeixinQr((current) => current ? { ...current, status: "error", error: error instanceof Error ? error.message : "登录状态检查失败" } : current));
    }, 1800);
    return () => window.clearInterval(timer);
  }, [resource, weixinQr?.flowId, weixinQr?.status]);

  return <>
    <PageIntro title="联系小满" description="把小满接入你常用的聊天工具；每个渠道都会保留自己的对话，不会互相覆盖。" actions={<button type="button" className="gateway-restart-button" disabled={restarting} onClick={() => setRestartConfirm(true)}><RefreshCw size={15} className={restarting ? "gateway-restart-icon" : ""} />{restarting ? "正在重连" : "重新连接"}</button>} />
    <ErrorBanner message={resource.error || restartError} />
    {notice ? <div className="channel-notice"><ShieldCheck size={16} />{notice}</div> : null}
    <div className="channel-section-heading"><span>网页与本机</span><small>随小满启动，无需额外配置</small></div>
    <div className="channel-grid channel-grid-local">{rows.filter((row) => row.kind === "local").map((channel) => <ChannelCard key={channel.id} channel={channel} onEdit={() => { setSaveError(""); setDraft(makeDraft(channel)); }} />)}</div>
    <div className="channel-section-heading"><span>聊天渠道</span><small>使用官方机器人连接，凭据只保存在本机</small></div>
    <div className="channel-grid">{rows.filter((row) => row.kind === "gateway").map((channel) => <ChannelCard key={channel.id} channel={channel} onEdit={() => { setSaveError(""); setDraft(makeDraft(channel)); }} />)}</div>
    {draft ? <ChannelDialog draft={draft} saving={saving} saveError={saveError} weixinQr={weixinQr} setDraft={(value) => { setSaveError(""); setDraft(value); }} onStartWeixinQr={() => void beginWeixinQr()} onClose={() => { setDraft(null); setWeixinQr(null); setSaveError(""); }} onSave={() => void save()} /> : null}
    {restartConfirm ? <Modal title="重新连接小满" description="重新加载 QQ、微信等聊天渠道。" onClose={() => setRestartConfirm(false)}><div className="form-stack"><div className="gateway-restart-warning"><RefreshCw size={18} /><div><strong>连接会短暂中断</strong><span>正在生成的回复会停止；已保存的聊天记录、任务和配置不会丢失。</span></div></div><div className="dialog-actions"><button type="button" className="secondary-button" onClick={() => setRestartConfirm(false)}>取消</button><button type="button" className="primary-button" onClick={() => void restartGateway()}><RefreshCw size={15} />立即重连</button></div></div></Modal> : null}
  </>;
}

function ChannelCard(props: { channel: ChannelRow; onEdit(): void }): React.ReactElement {
  const { channel } = props;
  return <article className={`channel-card channel-${channel.id}`}>
    <div className="channel-card-top"><span className="channel-icon"><ChannelIcon id={channel.id} /></span><Badge tone={channel.connected ? "green" : channel.configured ? "amber" : "gray"}>{channel.connected ? "在线" : channel.configured ? "待重启" : "未配置"}</Badge></div>
    <h3>{channel.label}</h3>
    <p>{channel.detail}</p>
    <small>{channel.allow_from.length ? `仅允许 ${channel.allow_from.length} 个账号` : channel.kind === "gateway" ? "白名单为空时接受所有账号" : "由系统管理"}</small>
    <div className="channel-card-actions">
      {channel.kind === "gateway" ? <button className="secondary-button" onClick={props.onEdit}>配置接入</button> : <span className="managed-label"><ShieldCheck size={14} />由系统管理</span>}
      {channel.docs_url ? <a href={channel.docs_url} target="_blank" rel="noreferrer">{channel.docs_label ?? "官方文档"}<ExternalLink size={13} /></a> : null}
    </div>
  </article>;
}

function ChannelDialog(props: { draft: ChannelDraft; saving: boolean; saveError: string; weixinQr: WeixinQrState | null; setDraft(value: ChannelDraft): void; onStartWeixinQr(): void; onClose(): void; onSave(): void }): React.ReactElement {
  const { draft } = props;
  const update = (patch: Partial<ChannelDraft>): void => props.setDraft({ ...draft, ...patch });
  return <Modal title={`配置 ${draft.row.label}`} description={dialogDescription(draft.row.id)} onClose={props.onClose}>
    <div className="form-stack">
      {draft.row.id === "telegram" ? <label>Bot Token<input type="password" autoComplete="new-password" value={draft.token} onChange={(event) => update({ token: event.target.value })} placeholder={draft.row.fields?.token?.configured ? "留空保留当前 Token" : "输入 BotFather Token"} /></label> : null}
      {draft.row.id === "qqbot" ? <>
        <label>AppID<input value={draft.appId} onChange={(event) => update({ appId: event.target.value })} placeholder="QQ 开放平台 AppID" /></label>
        <label>AppSecret<input type="password" autoComplete="new-password" value={draft.clientSecret} onChange={(event) => update({ clientSecret: event.target.value })} placeholder={draft.row.fields?.client_secret?.configured ? "留空保留当前 AppSecret" : "输入 AppSecret"} /></label>
      </> : null}
      {draft.row.id === "wecom" ? <>
        <label>Bot ID<input value={draft.botId} onChange={(event) => update({ botId: event.target.value })} placeholder="企业微信智能机器人 Bot ID" /></label>
        <label>Secret<input type="password" autoComplete="new-password" value={draft.secret} onChange={(event) => update({ secret: event.target.value })} placeholder={draft.row.fields?.secret?.configured ? "留空保留当前 Secret" : "输入机器人 Secret"} /></label>
      </> : null}
      {draft.row.id === "weixin" ? <div className="weixin-qr-panel">
        <div className="weixin-qr-copy"><Smartphone size={18} /><div><strong>微信扫码连接</strong><span>会生成独立的 iLink 机器人身份，不会接管你的普通微信客户端。</span></div></div>
        {props.weixinQr?.image ? <img src={props.weixinQr.image} alt="微信登录二维码" /> : null}
        <p>{weixinQrStatus(props.weixinQr)}</p>
        <button type="button" className="secondary-button" disabled={props.weixinQr?.status === "loading"} onClick={props.onStartWeixinQr}>{props.weixinQr?.image ? "刷新二维码" : "获取登录二维码"}</button>
      </div> : null}
      <label>允许的账号<input value={draft.allowFrom} onChange={(event) => update({ allowFrom: event.target.value })} placeholder={draft.row.id === "qqbot" ? "QQ user_openid，多个用逗号分隔" : draft.row.id === "wecom" ? "企业微信 userid，多个用逗号分隔" : "多个账号用逗号分隔"} /></label>
      <p className="channel-form-hint">{draft.row.id === "weixin" ? "扫码 Token 使用 Windows 凭据管理器加密保存；配置文件只记录账号标识。" : "密钥不会通过接口回显；留空密钥字段会保留当前值。"}</p>
      {props.saveError ? <div className="channel-save-error" role="alert"><AlertCircle size={15} /><span>{props.saveError}</span></div> : null}
      <div className="dialog-actions"><button type="button" className="secondary-button" onClick={props.onClose}>取消</button><button type="button" className="primary-button" disabled={props.saving} onClick={props.onSave}><Save size={16} />{props.saving ? "保存中" : draft.row.id === "weixin" ? "保存白名单" : "保存配置"}</button></div>
    </div>
  </Modal>;
}

function validateDraft(draft: ChannelDraft): string {
  if (draft.row.id === "telegram" && !draft.token.trim() && !draft.row.fields?.token?.configured) return "请填写 Bot Token。";
  if (draft.row.id === "qqbot") {
    if (!draft.appId.trim()) return "请填写 QQ 开放平台 AppID。";
    if (!draft.clientSecret.trim() && !draft.row.fields?.client_secret?.configured) return "请填写 QQ 开放平台 AppSecret。";
  }
  if (draft.row.id === "wecom") {
    if (!draft.botId.trim()) return "请填写企业微信 Bot ID。";
    if (!draft.secret.trim() && !draft.row.fields?.secret?.configured) return "请填写企业微信 Secret。";
  }
  return "";
}

function ChannelIcon({ id }: { id: string }): React.ReactElement {
  if (id === "dashboard") return <Gauge size={20} />;
  if (id === "ipc") return <Network size={20} />;
  if (id === "qqbot") return <Bot size={20} />;
  if (id === "wecom") return <MessageCircleMore size={20} />;
  if (id === "weixin") return <Smartphone size={20} />;
  return <Radio size={20} />;
}

function dialogDescription(id: string): string {
  if (id === "qqbot") return "在 QQ 开放平台创建机器人后填写 AppID 和 AppSecret，使用官方 Gateway 长连接。";
  if (id === "wecom") return "在企业微信「智能机器人」中选择 API 模式与长连接，再复制 Bot ID 和 Secret。";
  if (id === "weixin") return "使用腾讯 iLink Bot API，用微信扫码创建个人微信机器人身份。";
  return "凭据只写入本机 config.toml，不会从 API 回显。";
}

function weixinQrStatus(state: WeixinQrState | null): string {
  if (!state) return "点击后使用微信扫一扫，并在手机内确认。";
  if (state.status === "loading") return "正在向微信申请二维码…";
  if (state.status === "scaned") return "已扫码，请在微信内确认登录。";
  if (state.status === "confirmed") return "连接凭据已保存，重启小满后上线。";
  if (state.status === "expired") return "二维码已过期，请刷新。";
  if (state.status === "error") return state.error || "连接失败，请重试。";
  return "请使用微信扫描二维码。";
}
