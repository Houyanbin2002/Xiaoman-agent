import React, { useCallback, useEffect, useRef, useState } from "react";
import { Check, ChevronDown, ChevronRight, CircleAlert, Copy, FileText, Hand, LoaderCircle, Maximize2, Plus, Send, ShieldAlert, ShieldCheck, Square, SquareTerminal, Upload, X } from "lucide-react";
import { api } from "../../api";
import { MarkdownView } from "../../MarkdownView";
import type { MessageRow, PageResult } from "../../types";
import type { ChatMessage } from "../../shared/types";
import { ChatModelSelector } from "../models/ChatModelSelector";

interface ActiveRun {
  id: string;
  prompt: string;
  startedAt: string;
  status: "thinking" | "approval" | "stopping";
  thinking: string;
}

type PermissionMode = "request_approval" | "auto_approve" | "full_access";

interface PendingApproval {
  id: string;
  toolName: string;
  risk: "low" | "medium" | "high";
  title: string;
  description: string;
  preview: string;
}

interface ChatEvent {
  type: string;
  run_id?: string;
  status?: string;
  prompt?: string;
  started_at?: string;
  content?: string;
  thinking?: string;
  permission_mode?: string;
  delta?: string;
  message?: string;
  approval_id?: string;
  tool_name?: string;
  risk?: string;
  title?: string;
  description?: string;
  preview?: string;
  attachments?: UploadedAttachment[];
}

interface UploadedAttachment {
  id: string;
  name: string;
  size: number;
  mime_type: string;
  parsed?: boolean;
}

const ATTACHMENT_ACCEPT = ".pdf,.docx,.xlsx,.xls,.pptx,.csv,.tsv,.txt,.md,.json,.xml,.html,.htm,.epub,.png,.jpg,.jpeg,.webp,.gif";
const MAX_ATTACHMENTS = 8;
const MAX_ATTACHMENT_BYTES = 128 * 1024 * 1024;
const CLIPBOARD_IMAGE_EXTENSIONS: Record<string, string> = {
  "image/gif": "gif",
  "image/jpeg": "jpg",
  "image/png": "png",
  "image/webp": "webp",
};

function clipboardImageFiles(clipboardData: DataTransfer): File[] {
  const timestamp = Date.now();
  return Array.from(clipboardData.items).flatMap((item, index) => {
    const extension = CLIPBOARD_IMAGE_EXTENSIONS[item.type.toLowerCase()];
    if (item.kind !== "file" || !extension) return [];
    const file = item.getAsFile();
    if (!file) return [];
    if (/\.(?:gif|jpe?g|png|webp)$/i.test(file.name)) return [file];
    return [new File([file], `clipboard-${timestamp}-${index + 1}.${extension}`, {
      type: file.type,
      lastModified: file.lastModified,
    })];
  });
}

const PERMISSION_STORAGE_KEY = "xiaoman:chat-permission-mode";
const TOOL_ACTION_LABELS: Record<string, string> = {
  write_file: "写入文件",
  edit_file: "编辑文件",
  shell: "运行命令",
  web_search: "搜索互联网",
  web_fetch: "访问网页",
  message_push: "发送外部消息",
  schedule: "创建定时任务",
  cancel_schedule: "取消定时任务",
  task_create: "创建后台任务",
  task_manage: "管理后台任务",
  personal_record: "更新个人记录",
  personal_rhythm: "更新个人节奏",
};
const PERMISSION_OPTIONS: Array<{
  value: PermissionMode;
  label: string;
  description: string;
  icon: typeof Hand;
}> = [
  { value: "request_approval", label: "请求批准", description: "写文件、联网和外部操作前先询问", icon: Hand },
  { value: "auto_approve", label: "替我审批", description: "仅对删除、外部路径等风险操作询问", icon: SquareTerminal },
  { value: "full_access", label: "完全访问权限", description: "不限制访问互联网和电脑文件", icon: ShieldCheck },
];

function isPermissionMode(value: unknown): value is PermissionMode {
  return value === "request_approval" || value === "auto_approve" || value === "full_access";
}

function initialPermissionMode(): PermissionMode {
  try {
    const stored = window.localStorage.getItem(PERMISSION_STORAGE_KEY);
    return isPermissionMode(stored) ? stored : "request_approval";
  } catch {
    return "request_approval";
  }
}

function clockTime(value?: string): string {
  const date = value ? new Date(value) : new Date();
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", hour12: false });
}

function runMessageId(runId: string, role: "user" | "assistant"): string {
  return `run:${runId}:${role}`;
}

function historicalMessage(item: MessageRow): ChatMessage {
  const details = item as MessageRow & { interrupted?: boolean; reasoning_content?: unknown; media?: unknown };
  const interrupted = details.interrupted === true || item.content === "[interrupted]";
  const attachments = Array.isArray(details.media)
    ? details.media.map((value) => ({ name: String(value).split(/[\\/]/).pop() || "附件" }))
    : undefined;
  return {
    id: item.id,
    role: item.role as "user" | "assistant",
    content: item.content === "[interrupted]" ? "" : item.content,
    timestamp: clockTime(item.timestamp),
    thinking: typeof details.reasoning_content === "string" ? details.reasoning_content : undefined,
    state: interrupted ? "stopped" : undefined,
    attachments,
  };
}

function PermissionSelector(props: {
  value: PermissionMode;
  onChange: (value: PermissionMode) => void;
  disabled: boolean;
}): React.ReactElement {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const selected = PERMISSION_OPTIONS.find((option) => option.value === props.value) ?? PERMISSION_OPTIONS[0];
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
  return (
    <div className={`permission-control mode-${props.value}`} ref={rootRef}>
      <button type="button" className="permission-trigger" onClick={() => setOpen((current) => !current)} disabled={props.disabled} aria-haspopup="menu" aria-expanded={open} title={`操作权限：${selected.label}`}>
        <ShieldAlert size={17} />
        <span>{selected.label}</span>
        <ChevronDown size={12} />
      </button>
      {open ? (
        <div className="permission-menu" role="menu" aria-label="AI 操作权限">
          <div className="permission-menu-head"><strong>AI 操作权限</strong><small>只影响之后发送的新任务</small></div>
          {PERMISSION_OPTIONS.map((option) => {
            const Icon = option.icon;
            return (
              <button type="button" role="menuitemradio" aria-checked={option.value === props.value} key={option.value} onClick={() => { props.onChange(option.value); setOpen(false); }}>
                <span className="permission-option-icon"><Icon size={18} /></span>
                <span className="permission-option-copy"><strong>{option.label}</strong><small>{option.description}</small></span>
                {option.value === props.value ? <Check size={17} className="permission-check" /> : null}
              </button>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

function ApprovalCard(props: {
  approval: PendingApproval;
  responding: boolean;
  onRespond: (approved: boolean) => void;
}): React.ReactElement {
  return (
    <section className="approval-card" aria-live="assertive">
      <div className="approval-card-icon"><ShieldAlert size={18} /></div>
      <div className="approval-card-body">
        <div className="approval-card-kicker"><span>需要你的批准</span><small>{TOOL_ACTION_LABELS[props.approval.toolName] ?? props.approval.toolName}</small><i>{props.approval.risk === "high" ? "高风险" : "需确认"}</i></div>
        <h3>{props.approval.title}</h3>
        <p>{props.approval.description}</p>
        {props.approval.preview ? <pre>{props.approval.preview}</pre> : null}
        <div className="approval-card-actions">
          <button type="button" className="approval-deny" disabled={props.responding} onClick={() => props.onRespond(false)}>拒绝</button>
          <button type="button" className="approval-allow" disabled={props.responding} onClick={() => props.onRespond(true)}>{props.responding ? "处理中…" : "批准这一次"}</button>
        </div>
      </div>
    </section>
  );
}

function CompactComposer(props: {
  input: string;
  setInput: React.Dispatch<React.SetStateAction<string>>;
  send: () => void;
  stop: () => void;
  busy: boolean;
  disabled: boolean;
  permissionMode: PermissionMode;
  setPermissionMode: (mode: PermissionMode) => void;
  attachments: UploadedAttachment[];
  uploadingCount: number;
  onChooseFiles: (files: FileList | File[]) => void;
  onRemoveAttachment: (attachment: UploadedAttachment) => void;
}): React.ReactElement {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canSend = Boolean(props.input.trim() || props.attachments.length);
  return (
    <div className={`compact-composer${props.attachments.length || props.uploadingCount ? " has-attachments" : ""}`}>
      {props.attachments.length || props.uploadingCount ? <div className="attachment-tray" aria-label="待发送附件">
        {props.attachments.map((attachment) => <span className="attachment-chip" key={attachment.id}><FileText size={15} /><span><strong>{attachment.name}</strong><small>{formatFileSize(attachment.size)}</small></span><button type="button" onClick={() => props.onRemoveAttachment(attachment)} aria-label={`移除 ${attachment.name}`} title="移除附件"><X size={13} /></button></span>)}
        {props.uploadingCount ? <span className="attachment-chip uploading"><LoaderCircle size={15} className="spin" /><span><strong>正在添加文件</strong><small>{props.uploadingCount} 个</small></span></span> : null}
      </div> : null}
      <div className="compact-composer-row">
        <input ref={fileInputRef} className="attachment-input" type="file" accept={ATTACHMENT_ACCEPT} multiple onChange={(event) => { if (event.target.files?.length) props.onChooseFiles(event.target.files); event.currentTarget.value = ""; }} />
        <button type="button" className="compact-plus" onClick={() => fileInputRef.current?.click()} title="添加文件" aria-label="添加文件"><Plus size={22} /></button>
        <PermissionSelector value={props.permissionMode} onChange={props.setPermissionMode} disabled={props.busy} />
        <textarea value={props.input} onChange={(event) => props.setInput(event.target.value)} onPaste={(event) => { const images = clipboardImageFiles(event.clipboardData); if (!images.length) return; event.preventDefault(); props.onChooseFiles(images); }} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); if (!props.busy && canSend && !props.uploadingCount) props.send(); } }} placeholder={props.busy ? "当前回复生成中，可先输入下一条消息" : props.attachments.length ? "告诉小满要如何处理这些文件" : "问问小满"} rows={1} />
        <ChatModelSelector disabled={props.busy} />
        {props.busy ? <button type="button" className="compact-stop" onClick={props.stop} title="停止生成" aria-label="停止生成"><Square size={14} fill="currentColor" /></button> : canSend ? <button type="button" className="compact-send" onClick={props.send} disabled={props.disabled || Boolean(props.uploadingCount)} title="发送" aria-label="发送"><Send size={17} /></button> : null}
      </div>
    </div>
  );
}

function formatFileSize(size: number): string {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${Math.ceil(size / 1024)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function RunProgress(props: { run: ActiveRun; onStop: () => void }): React.ReactElement {
  const [elapsed, setElapsed] = useState(0);
  const [expanded, setExpanded] = useState(false);
  useEffect(() => {
    const update = (): void => setElapsed(Math.max(0, Math.floor((Date.now() - new Date(props.run.startedAt).getTime()) / 1000)));
    update();
    const timer = window.setInterval(update, 1000);
    return () => window.clearInterval(timer);
  }, [props.run.startedAt]);
  const stopping = props.run.status === "stopping";
  const awaitingApproval = props.run.status === "approval";
  return (
    <section className={`run-trace${stopping ? " stopping" : ""}`} aria-live="polite">
      <div className="run-trace-line">
        <button type="button" className="run-trace-toggle" aria-expanded={expanded} onClick={() => setExpanded((current) => !current)}>
          {awaitingApproval ? <ShieldAlert size={16} /> : <LoaderCircle size={16} className={stopping ? "" : "spin"} />}
          <span>{stopping ? "正在停止" : awaitingApproval ? "等待批准" : "正在处理"}</span>
          <small>·&nbsp; {elapsed} 秒</small>
          <ChevronRight size={14} className={expanded ? "expanded" : ""} />
        </button>
        <span className="run-trace-divider" />
        <button type="button" className="run-trace-stop" onClick={props.onStop} disabled={stopping}><Square size={10} fill="currentColor" />{stopping ? "停止中" : "停止"}</button>
      </div>
      {expanded ? <div className="run-trace-details"><span><SquareTerminal size={13} /></span><p>{props.run.thinking || "正在理解任务并规划下一步…"}</p></div> : null}
    </section>
  );
}

export function ChatView(props: {
  chatId: string;
  onStarted?: (chatId: string, firstMessage: string) => void;
  onRunState?: (chatId: string, running: boolean, firstMessage?: string) => void;
  onActivity?: () => void;
}): React.ReactElement {
  const { chatId, onStarted, onRunState, onActivity } = props;
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [connected, setConnected] = useState(false);
  const [activeRun, setActiveRun] = useState<ActiveRun | null>(null);
  const [notice, setNotice] = useState("");
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [permissionMode, setPermissionMode] = useState<PermissionMode>(initialPermissionMode);
  const [pendingApproval, setPendingApproval] = useState<PendingApproval | null>(null);
  const [approvalResponding, setApprovalResponding] = useState(false);
  const [attachments, setAttachments] = useState<UploadedAttachment[]>([]);
  const [uploadingCount, setUploadingCount] = useState(0);
  const [dragActive, setDragActive] = useState(false);
  const [historyPage, setHistoryPage] = useState(1);
  const [hasOlderMessages, setHasOlderMessages] = useState(false);
  const [loadingOlderMessages, setLoadingOlderMessages] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const dragDepthRef = useRef(0);

  const ensureRunMessages = useCallback((runId: string, prompt: string, content = "", runAttachments: UploadedAttachment[] = []): void => {
    setMessages((current) => {
      const userId = runMessageId(runId, "user");
      const assistantId = runMessageId(runId, "assistant");
      const hasUser = current.some((message) => message.id === userId);
      const hasAssistant = current.some((message) => message.id === assistantId);
      const next = current.map((message) => message.id === assistantId ? { ...message, content: content || message.content, pending: true } : message);
      if (!hasUser) next.push({ id: userId, role: "user", content: prompt, timestamp: clockTime(), attachments: runAttachments });
      if (!hasAssistant) next.push({ id: assistantId, role: "assistant", content, pending: true, timestamp: clockTime() });
      return next;
    });
  }, []);

  const updateAssistant = useCallback((runId: string, update: (message: ChatMessage) => ChatMessage): void => {
    const assistantId = runMessageId(runId, "assistant");
    setMessages((current) => current.map((message) => message.id === assistantId ? update(message) : message));
  }, []);

  const connect = useCallback(() => {
    socketRef.current?.close();
    setNotice("");
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${protocol}://${window.location.host}/api/dashboard/chat/${chatId}`);
    socketRef.current = socket;
    socket.onopen = () => { if (socketRef.current === socket) setConnected(true); };
    socket.onclose = () => { if (socketRef.current === socket) setConnected(false); };
    socket.onmessage = (event) => {
      let payload: ChatEvent;
      try {
        payload = JSON.parse(String(event.data)) as ChatEvent;
      } catch {
        setNotice("收到无法识别的运行消息");
        return;
      }
      const runId = payload.run_id ?? "";
      if (payload.type === "status" && runId && payload.prompt) {
        const status = payload.status === "stopping" ? "stopping" : "thinking";
        const run = { id: runId, prompt: payload.prompt, startedAt: payload.started_at ?? new Date().toISOString(), status, thinking: payload.thinking ?? "" } satisfies ActiveRun;
        setActiveRun(run);
        if (isPermissionMode(payload.permission_mode)) setPermissionMode(payload.permission_mode);
        ensureRunMessages(runId, payload.prompt, payload.content ?? "", payload.attachments ?? []);
        onRunState?.(chatId, true, payload.prompt);
      } else if (payload.type === "content_delta" && runId && payload.delta) {
        updateAssistant(runId, (message) => ({ ...message, content: message.content + payload.delta }));
      } else if (payload.type === "thinking_delta" && runId && payload.delta) {
        setActiveRun((current) => current?.id === runId ? { ...current, thinking: current.thinking + payload.delta } : current);
      } else if (payload.type === "approval_request" && payload.approval_id) {
        setPendingApproval({
          id: payload.approval_id,
          toolName: payload.tool_name ?? "tool",
          risk: payload.risk === "high" ? "high" : payload.risk === "low" ? "low" : "medium",
          title: payload.title ?? "允许执行这项操作？",
          description: payload.description ?? "这项操作需要你的批准。",
          preview: payload.preview ?? "",
        });
        setApprovalResponding(false);
        setActiveRun((current) => current ? { ...current, status: "approval" } : current);
      } else if (payload.type === "approval_resolved" && payload.approval_id) {
        setPendingApproval((current) => current?.id === payload.approval_id ? null : current);
        setApprovalResponding(false);
        setActiveRun((current) => current?.status === "approval" ? { ...current, status: "thinking" } : current);
      } else if (payload.type === "final" && runId) {
        updateAssistant(runId, (message) => ({ ...message, content: payload.content ?? message.content, thinking: payload.thinking, pending: false }));
        setActiveRun(null);
        setPendingApproval(null);
        setApprovalResponding(false);
        onRunState?.(chatId, false);
        onActivity?.();
      } else if (payload.type === "cancelled" && runId) {
        updateAssistant(runId, (message) => ({ ...message, content: payload.content ?? message.content, thinking: payload.thinking, pending: false, state: "stopped" }));
        setActiveRun(null);
        setPendingApproval(null);
        setApprovalResponding(false);
        onRunState?.(chatId, false);
        onActivity?.();
      } else if (payload.type === "error") {
        if (runId) {
          updateAssistant(runId, (message) => ({ ...message, content: message.content || payload.message || "请求失败", pending: false, state: "error" }));
          setActiveRun(null);
          setPendingApproval(null);
          setApprovalResponding(false);
          onRunState?.(chatId, false);
          onActivity?.();
        } else {
          setNotice(payload.message ?? "请求失败");
          setActiveRun(null);
          setPendingApproval(null);
          setApprovalResponding(false);
          onRunState?.(chatId, false);
        }
      } else if (payload.type === "push" && payload.content) {
        setMessages((current) => [...current, { id: crypto.randomUUID(), role: "assistant", content: payload.content ?? "", timestamp: clockTime() }]);
      }
    };
  }, [chatId, ensureRunMessages, onActivity, onRunState, updateAssistant]);

  useEffect(() => {
    let disposed = false;
    setMessages([]);
    setActiveRun(null);
    setPendingApproval(null);
    setApprovalResponding(false);
    setConnected(false);
    setAttachments([]);
    setUploadingCount(0);
    setDragActive(false);
    setHistoryPage(1);
    setHasOlderMessages(false);
    setLoadingOlderMessages(false);
    dragDepthRef.current = 0;
    void api<PageResult<MessageRow>>(`/api/dashboard/messages?session_key=dashboard:${chatId}&page=1&page_size=100&sort_by=seq&sort_order=desc`)
      .then((payload) => {
        if (!disposed) {
          const visible = payload.items
            .filter((item) => item.role === "user" || item.role === "assistant")
            .reverse()
            .map(historicalMessage);
          setMessages(visible);
          setHasOlderMessages(payload.total > payload.items.length);
        }
      })
      .catch(() => { if (!disposed) setNotice("聊天记录加载失败，可重新连接后再试"); })
      .finally(() => { if (!disposed) connect(); });
    return () => { disposed = true; socketRef.current?.close(); };
  }, [chatId, connect]);

  const loadOlderMessages = useCallback(async (): Promise<void> => {
    if (!hasOlderMessages || loadingOlderMessages) return;
    setLoadingOlderMessages(true);
    setNotice("");
    try {
      const nextPage = historyPage + 1;
      const payload = await api<PageResult<MessageRow>>(`/api/dashboard/messages?session_key=dashboard:${chatId}&page=${nextPage}&page_size=100&sort_by=seq&sort_order=desc`);
      const older = payload.items
        .filter((item) => item.role === "user" || item.role === "assistant")
        .reverse()
        .map(historicalMessage);
      setMessages((current) => {
        const existing = new Set(current.map((message) => message.id));
        return [...older.filter((message) => !existing.has(message.id)), ...current];
      });
      setHistoryPage(nextPage);
      setHasOlderMessages(nextPage * 100 < payload.total);
    } catch (reason) {
      setNotice(reason instanceof Error ? reason.message : "更早的聊天记录加载失败");
    } finally {
      setLoadingOlderMessages(false);
    }
  }, [chatId, hasOlderMessages, historyPage, loadingOlderMessages]);

  useEffect(() => {
    try { window.localStorage.setItem(PERMISSION_STORAGE_KEY, permissionMode); } catch { /* local storage can be unavailable in hardened browsers */ }
  }, [permissionMode]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, activeRun?.thinking]);

  const uploadFiles = useCallback(async (files: FileList | File[]): Promise<void> => {
    const available = Math.max(0, MAX_ATTACHMENTS - attachments.length);
    const selected = Array.from(files).slice(0, available);
    if (!selected.length) {
      setNotice(`每条消息最多添加 ${MAX_ATTACHMENTS} 个文件`);
      return;
    }
    const oversized = selected.find((file) => file.size > MAX_ATTACHMENT_BYTES);
    if (oversized) {
      setNotice(`${oversized.name} 超过 128 MB 上限`);
      return;
    }
    setUploadingCount((current) => current + selected.length);
    setNotice("");
    const results = await Promise.allSettled(selected.map(async (file): Promise<UploadedAttachment> => {
      const response = await fetch(`/api/dashboard/chat/${encodeURIComponent(chatId)}/attachments?filename=${encodeURIComponent(file.name)}`, {
        method: "POST",
        headers: { "Content-Type": file.type || "application/octet-stream" },
        body: file,
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({})) as { detail?: string };
        throw new Error(payload.detail || `${file.name} 添加失败`);
      }
      return response.json() as Promise<UploadedAttachment>;
    }));
    const uploaded = results.flatMap((result) => result.status === "fulfilled" ? [result.value] : []);
    const failed = results.find((result): result is PromiseRejectedResult => result.status === "rejected");
    setAttachments((current) => [...current, ...uploaded].slice(0, MAX_ATTACHMENTS));
    setUploadingCount((current) => Math.max(0, current - selected.length));
    if (failed) setNotice(failed.reason instanceof Error ? failed.reason.message : "文件添加失败");
  }, [attachments.length, chatId]);

  const removeAttachment = useCallback((attachment: UploadedAttachment): void => {
    setAttachments((current) => current.filter((item) => item.id !== attachment.id));
    void fetch(`/api/dashboard/chat/${encodeURIComponent(chatId)}/attachments/${encodeURIComponent(attachment.id)}`, { method: "DELETE" });
  }, [chatId]);

  const send = (): void => {
    const content = input.trim() || (attachments.length ? "请阅读并分析这些附件。" : "");
    if (!content || uploadingCount || activeRun || socketRef.current?.readyState !== WebSocket.OPEN) return;
    const requestId = crypto.randomUUID();
    const startedAt = new Date().toISOString();
    ensureRunMessages(requestId, content, "", attachments);
    setActiveRun({ id: requestId, prompt: content, startedAt, status: "thinking", thinking: "" });
    setInput("");
    setNotice("");
    onStarted?.(chatId, content);
    onRunState?.(chatId, true, content);
    socketRef.current.send(JSON.stringify({ type: "message", content, request_id: requestId, permission_mode: permissionMode, attachment_ids: attachments.map((attachment) => attachment.id) }));
    setAttachments([]);
  };

  const stop = (): void => {
    if (!activeRun || socketRef.current?.readyState !== WebSocket.OPEN) return;
    setActiveRun((current) => current ? { ...current, status: "stopping" } : current);
    socketRef.current.send(JSON.stringify({ type: "stop" }));
  };

  const respondToApproval = (approved: boolean): void => {
    if (!pendingApproval || approvalResponding || socketRef.current?.readyState !== WebSocket.OPEN) return;
    setApprovalResponding(true);
    socketRef.current.send(JSON.stringify({ type: "approval_response", approval_id: pendingApproval.id, decision: approved ? "approve" : "deny" }));
  };

  const copyMessage = async (message: ChatMessage): Promise<void> => {
    await navigator.clipboard.writeText(message.content);
    setCopiedId(message.id);
    window.setTimeout(() => setCopiedId((current) => current === message.id ? null : current), 1400);
  };

  const composerProps = {
    input,
    setInput,
    send,
    stop,
    busy: Boolean(activeRun),
    disabled: !connected,
    permissionMode,
    setPermissionMode,
    attachments,
    uploadingCount,
    onChooseFiles: (files: FileList | File[]): void => { void uploadFiles(files); },
    onRemoveAttachment: removeAttachment,
  };

  return (
    <div className={`chat-page${messages.length ? " has-conversation" : " new-conversation"}${dragActive ? " is-dragging" : ""}`} onDragEnter={(event) => { if (!event.dataTransfer.types.includes("Files")) return; event.preventDefault(); dragDepthRef.current += 1; setDragActive(true); }} onDragOver={(event) => { if (!event.dataTransfer.types.includes("Files")) return; event.preventDefault(); event.dataTransfer.dropEffect = "copy"; }} onDragLeave={(event) => { if (!event.dataTransfer.types.includes("Files")) return; event.preventDefault(); dragDepthRef.current = Math.max(0, dragDepthRef.current - 1); if (!dragDepthRef.current) setDragActive(false); }} onDrop={(event) => { event.preventDefault(); dragDepthRef.current = 0; setDragActive(false); if (event.dataTransfer.files.length) void uploadFiles(event.dataTransfer.files); }}>
      {dragActive ? <div className="chat-drop-zone"><span><Upload size={22} /></span><strong>松开即可添加文件</strong><small>支持 PDF、Word、Excel、PowerPoint、文本和图片</small></div> : null}
      {!connected || notice ? <div className={`connection-banner${notice ? " error" : ""}`}><CircleAlert size={15} /><span>{notice || "正在连接小满"}</span><button type="button" onClick={connected ? () => setNotice("") : connect}>{connected ? "关闭" : "重新连接"}</button></div> : null}
      <div className={`chat-scroll${messages.length ? " has-messages" : ""}`}>
        {hasOlderMessages ? <button type="button" className="load-older-messages" disabled={loadingOlderMessages} onClick={() => void loadOlderMessages()}>{loadingOlderMessages ? "正在加载…" : "加载更早消息"}</button> : null}
        {messages.length === 0 ? (
          <div className="chat-welcome minimal-welcome">
            <div className="welcome-identity"><span><img src="/assets/assets/xiaoman-avatar.png" alt="小满" /><i /></span><small>小满在这里</small></div>
            <h2>今天想一起完成什么？</h2>
            <p>聊聊近况、整理想法，或者把一件需要完成的事交给我。</p>
            <div className="welcome-starters" aria-label="常用开始方式">
              <button type="button" onClick={() => setInput("帮我看看今天最值得优先处理什么")}>安排今天</button>
              <button type="button" onClick={() => setInput("帮我把下面这个零散想法整理清楚：")}>整理想法</button>
              <button type="button" onClick={() => setInput("请帮我调研并给出一份清晰的结论：")}>开始调研</button>
              <button type="button" onClick={() => setInput("我想聊聊最近的状态")}>聊聊近况</button>
            </div>
            <div className="compact-entry"><div className="compact-aura" aria-hidden="true"><i /><i /><i /></div><CompactComposer {...composerProps} /></div>
          </div>
        ) : messages.map((message) => (
          <div className={`chat-row ${message.role}`} key={message.id}>
            <div className={`message-wrap${expandedId === message.id ? " expanded" : ""}`}>
              <div className={`message-surface${message.pending ? " pending" : ""}`}>
                {message.role === "user" && message.attachments?.length ? <div className="message-attachments">{message.attachments.map((attachment, index) => <span key={`${attachment.name}:${index}`}><FileText size={14} /><span>{attachment.name}</span></span>)}</div> : null}
                {message.role === "assistant" && message.pending && activeRun?.id === message.id.split(":")[1] ? <RunProgress run={activeRun} onStop={stop} /> : null}
                {message.role === "assistant" && message.pending && activeRun?.id === message.id.split(":")[1] && pendingApproval ? <ApprovalCard approval={pendingApproval} responding={approvalResponding} onRespond={respondToApproval} /> : null}
                {message.content ? <div className="answer-content"><MarkdownView content={message.content} /></div> : null}
                {message.role === "assistant" && !message.pending && message.thinking ? <details className="completed-reasoning"><summary><SquareTerminal size={13} /><span>思考过程</span><ChevronRight size={13} /></summary><div>{message.thinking}</div></details> : null}
                {message.state === "stopped" ? <div className="message-state stopped"><Square size={10} fill="currentColor" />已停止生成</div> : null}
                {message.state === "error" ? <div className="message-state error"><CircleAlert size={12} />本轮执行失败</div> : null}
              </div>
              {message.role === "assistant" && message.content ? <div className="message-actions"><button aria-label="复制回复" title="复制回复" onClick={() => void copyMessage(message)}>{copiedId === message.id ? <Check size={14} /> : <Copy size={14} />}</button><button aria-label="展开回复" title="展开回复" onClick={() => setExpandedId((current) => current === message.id ? null : message.id)}><Maximize2 size={14} /></button><span>{message.timestamp}</span></div> : null}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
      {messages.length ? <div className="composer-shell conversation-mode"><CompactComposer {...composerProps} /></div> : null}
    </div>
  );
}
