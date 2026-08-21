import React from "react";
import { CircleAlert, LoaderCircle, X, type LucideIcon } from "lucide-react";

export function Badge(props: { children: React.ReactNode; tone?: "green" | "blue" | "amber" | "red" | "gray" }): React.ReactElement {
  return <span className={`badge badge-${props.tone ?? "gray"}`}>{props.children}</span>;
}

export function IconButton(props: { icon: LucideIcon; label: string; onClick?: () => void; disabled?: boolean; danger?: boolean }): React.ReactElement {
  const Icon = props.icon;
  return (
    <button className={`icon-button${props.danger ? " danger" : ""}`} title={props.label} aria-label={props.label} onClick={props.onClick} disabled={props.disabled}>
      <Icon size={17} />
    </button>
  );
}

export function EmptyState(props: { icon: LucideIcon; title: string; text: string }): React.ReactElement {
  const Icon = props.icon;
  return (
    <div className="empty-state-new">
      <div className="empty-icon"><Icon size={22} /></div>
      <strong>{props.title}</strong>
      <p>{props.text}</p>
    </div>
  );
}

export function LoadingState(): React.ReactElement {
  return <div className="loading-state"><LoaderCircle size={18} className="spin" /> 正在载入</div>;
}

export function ErrorBanner(props: { message: string }): React.ReactElement | null {
  if (!props.message) return null;
  return <div className="error-banner"><CircleAlert size={16} />{props.message}</div>;
}

export function PageIntro(props: { title: string; description: string; actions?: React.ReactNode }): React.ReactElement {
  return (
    <div className="page-intro">
      <div><h1>{props.title}</h1><p>{props.description}</p></div>
      {props.actions ? <div className="page-actions">{props.actions}</div> : null}
    </div>
  );
}

export function Modal(props: { title: string; description?: string; children: React.ReactNode; onClose: () => void }): React.ReactElement {
  return (
    <div className="modal-layer" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && props.onClose()}>
      <div className="dialog" role="dialog" aria-modal="true">
        <div className="dialog-head">
          <div><h2>{props.title}</h2>{props.description ? <p>{props.description}</p> : null}</div>
          <IconButton icon={X} label="关闭" onClick={props.onClose} />
        </div>
        {props.children}
      </div>
    </div>
  );
}
