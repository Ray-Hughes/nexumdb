/** Small shared pieces: badges, states, pagers. */

import type { ReactNode } from "react";
import type { EdgeType, NodeKind } from "../types";
import { WarningIcon } from "./Icons";

/** Node kind, colour-coded to match the CLI. */
export function KindBadge({ kind }: { kind: NodeKind }) {
  return (
    <span className={`badge kind-${kind.toLowerCase()}`}>
      <span className="dot" />
      {kind}
    </span>
  );
}

/** The colour family an edge type belongs to. */
export function edgeClassOf(edge: EdgeType): "structural" | "semantic" | "provenance" {
  switch (edge) {
    case "PART_OF":
    case "PRECEDES":
    case "FOLLOWS":
    case "SUPERSEDES":
      return "structural";
    case "MENTIONS":
    case "RELATES_TO":
    case "SIMILAR_TO":
      return "semantic";
    default:
      return "provenance";
  }
}

export function Spinner() {
  return <span className="spinner" aria-hidden />;
}

export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div className="loading-row" role="status">
      <Spinner />
      <span>{label}…</span>
    </div>
  );
}

/**
 * An error the user can act on.
 *
 * Always offers a retry: most failures here are "the server went away", which
 * is exactly the case where trying again is the right move.
 */
export function ErrorBanner({
  message,
  onRetry,
}: {
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div className="error-banner" role="alert">
      <WarningIcon />
      <span style={{ flex: 1 }}>{message}</span>
      {onRetry && (
        <button className="button subtle" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

/** An empty state that says what to do next, not just that nothing is here. */
export function Empty({
  title,
  hint,
  action,
}: {
  title: string;
  hint?: string;
  action?: ReactNode;
}) {
  return (
    <div className="empty">
      <div className="empty-title">{title}</div>
      {hint && <div className="empty-hint">{hint}</div>}
      {action}
    </div>
  );
}

export function Pager({
  offset,
  limit,
  total,
  onChange,
  unit = "rows",
}: {
  offset: number;
  limit: number;
  total: number;
  onChange: (offset: number) => void;
  unit?: string;
}) {
  if (total <= limit) {
    return (
      <div className="pager">
        <span className="mono">
          {total.toLocaleString()} {unit}
        </span>
      </div>
    );
  }
  const first = total === 0 ? 0 : offset + 1;
  const last = Math.min(offset + limit, total);
  return (
    <div className="pager">
      <button
        className="button subtle"
        disabled={offset === 0}
        onClick={() => onChange(Math.max(0, offset - limit))}
      >
        ← Previous
      </button>
      <span className="mono">
        {first.toLocaleString()}–{last.toLocaleString()} of{" "}
        {total.toLocaleString()} {unit}
      </span>
      <button
        className="button subtle"
        disabled={last >= total}
        onClick={() => onChange(offset + limit)}
      >
        Next →
      </button>
    </div>
  );
}

/** A monospace ID that copies on click. */
export function CopyableId({ id, short }: { id: string; short?: boolean }) {
  const shown = short ? id.replace(/-/g, "").slice(-8) : id;
  return (
    <span
      className="mono faint"
      title={`${id} — click to copy`}
      style={{ cursor: "copy" }}
      onClick={(event) => {
        event.stopPropagation();
        void navigator.clipboard?.writeText(id);
      }}
    >
      {shown}
    </span>
  );
}
