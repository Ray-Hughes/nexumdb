/** Presentation helpers shared across views. */

import type { NodeKind, EdgeClass } from "../types";

/**
 * Short form of a node ID.
 *
 * Node IDs are UUIDv7, whose leading bytes encode a millisecond timestamp — so
 * every node written in one batch shares a prefix. The entropy is in the tail,
 * which is what gets shown, matching what the CLI prints.
 */
export function shortId(id: string): string {
  const hex = id.replace(/-/g, "");
  return hex.slice(-8);
}

export function formatBytes(count: number): string {
  if (count < 1024) return `${count} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = count / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(1)} ${units[unit]}`;
}

export function formatNumber(value: number): string {
  return value.toLocaleString();
}

/** Absolute timestamp, in the viewer's locale. */
export function formatDate(millis: number): string {
  return new Date(millis).toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Relative time, for "last ingested" columns where exactness is noise. */
export function formatRelative(millis: number): string {
  const seconds = Math.round((Date.now() - millis) / 1000);
  if (seconds < 60) return "just now";
  const steps: [number, Intl.RelativeTimeFormatUnit][] = [
    [60, "minute"],
    [60, "hour"],
    [24, "day"],
    [7, "week"],
    [4.348, "month"],
    [12, "year"],
  ];
  let value = seconds;
  let unit: Intl.RelativeTimeFormatUnit = "second";
  for (const [divisor, nextUnit] of steps) {
    if (Math.abs(value) < divisor) break;
    value /= divisor;
    unit = nextUnit;
  }
  return new Intl.RelativeTimeFormat(undefined, { numeric: "auto" }).format(
    -Math.round(value),
    unit,
  );
}

export function truncate(text: string, width: number): string {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length <= width ? clean : `${clean.slice(0, width - 1)}…`;
}

/**
 * Colour tokens per node kind.
 *
 * Deliberately the same assignment the CLI uses, so someone moving between
 * terminal and window is not relearning the palette.
 */
export const KIND_COLOR: Record<NodeKind, string> = {
  Document: "var(--kind-document)",
  Chunk: "var(--kind-chunk)",
  Entity: "var(--kind-entity)",
  PipelineRun: "var(--kind-run)",
};

export const EDGE_CLASS_COLOR: Record<EdgeClass, string> = {
  structural: "var(--edge-structural)",
  semantic: "var(--edge-semantic)",
  provenance: "var(--edge-provenance)",
};

/** Raw hex, for canvas and WebGL which cannot resolve CSS variables. */
export const KIND_HEX: Record<NodeKind, string> = {
  Document: "#4cc4e8",
  Chunk: "#5fd39a",
  Entity: "#c48ce8",
  PipelineRun: "#e8b45f",
};

export const EDGE_CLASS_HEX: Record<EdgeClass, string> = {
  structural: "#3a6b80",
  semantic: "#6b4a80",
  provenance: "#806340",
};

/** The label a node shows in lists and on the graph. */
export function nodeLabel(node: {
  kind: NodeKind;
  title?: string;
  text?: string;
  name?: string;
  pipeline_version?: string;
  embedding_model?: string;
}): string {
  switch (node.kind) {
    case "Document":
      return node.title ?? "(untitled)";
    case "Chunk":
      return truncate(node.text ?? "", 80);
    case "Entity":
      return node.name ?? "(unnamed)";
    case "PipelineRun":
      return `run ${node.pipeline_version ?? ""} (${node.embedding_model ?? ""})`;
  }
}
