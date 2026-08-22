/**
 * The node inspector — also the provenance inspector the spec asks for.
 *
 * Everything a node knows about itself is here, and crucially so is its
 * lineage: which pipeline run embedded it, with which model, under which
 * config hash. That chain is the point of the provenance edges, and this is
 * where it becomes readable rather than merely stored.
 */

import { useMemo } from "react";
import { api } from "../lib/api";
import { formatDate, truncate } from "../lib/format";
import { useAsync } from "../lib/useAsync";
import type { GraphNodeRecord, NodeDetail, PipelineRunNode } from "../types";
import { CopyableId, edgeClassOf, ErrorBanner, KindBadge, Loading } from "./common";
import { CloseIcon, GraphIcon } from "./Icons";

interface Props {
  nodeId: string;
  onSelect: (id: string) => void;
  onClose: () => void;
  onExplore: (id: string) => void;
}

export function Inspector({ nodeId, onSelect, onClose, onExplore }: Props) {
  const detail = useAsync((signal) => api.node(nodeId, signal), [nodeId]);

  if (detail.loading && !detail.data) {
    return (
      <aside className="inspector">
        <Loading label="Loading node" />
      </aside>
    );
  }
  if (detail.error) {
    return (
      <aside className="inspector">
        <ErrorBanner message={detail.error} onRetry={detail.reload} />
      </aside>
    );
  }
  if (!detail.data) return <aside className="inspector" />;

  return (
    <aside className="inspector">
      <InspectorBody
        detail={detail.data}
        onSelect={onSelect}
        onClose={onClose}
        onExplore={onExplore}
      />
    </aside>
  );
}

function InspectorBody({
  detail,
  onSelect,
  onClose,
  onExplore,
}: {
  detail: NodeDetail;
  onSelect: (id: string) => void;
  onClose: () => void;
  onExplore: (id: string) => void;
}) {
  const node = detail.node;

  // The run that produced this node, found through its provenance edges.
  const runId = useMemo(() => {
    const provenance = detail.outgoing.find(
      (e) => e.edge.edge_type === "EMBEDDED_BY" || e.edge.edge_type === "EXTRACTED_BY",
    );
    if (provenance) return provenance.other_id;
    if (node.kind === "Document") return node.run_id ?? null;
    return null;
  }, [detail, node]);

  return (
    <>
      <header className="inspector-header">
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <KindBadge kind={node.kind} />
          <div style={{ flex: 1 }} />
          <button
            className="button subtle"
            title="Centre the graph explorer on this node"
            onClick={() => onExplore(node.id)}
          >
            <GraphIcon />
          </button>
          <button className="button subtle" title="Close" onClick={onClose}>
            <CloseIcon />
          </button>
        </div>
        <div style={{ marginTop: 8, fontSize: 13, lineHeight: 1.5 }}>
          <NodeHeadline node={node} />
        </div>
        <div style={{ marginTop: 4 }}>
          <CopyableId id={node.id} />
        </div>
      </header>

      <NodeProperties node={node} onSelect={onSelect} />

      {node.kind === "Chunk" && (
        <section className="inspector-section">
          <h3 className="section-title">Text</h3>
          <div style={{ fontSize: 12.5, lineHeight: 1.65, whiteSpace: "pre-wrap" }}>
            {node.text}
          </div>
        </section>
      )}

      {runId && <Provenance runId={runId} onSelect={onSelect} />}

      <EdgeList
        title={`Outgoing (${detail.outgoing.length})`}
        edges={detail.outgoing}
        onSelect={onSelect}
      />
      <EdgeList
        title={`Incoming (${detail.incoming.length})`}
        edges={detail.incoming}
        onSelect={onSelect}
      />
    </>
  );
}

function NodeHeadline({ node }: { node: GraphNodeRecord }) {
  switch (node.kind) {
    case "Document":
      return <strong>{node.title}</strong>;
    case "Chunk":
      return <span className="muted">{truncate(node.text, 120)}</span>;
    case "Entity":
      return <strong>{node.name}</strong>;
    case "PipelineRun":
      return <strong>{node.pipeline_version}</strong>;
  }
}

function NodeProperties({
  node,
  onSelect,
}: {
  node: GraphNodeRecord;
  onSelect: (id: string) => void;
}) {
  const rows: [string, React.ReactNode][] = [];

  switch (node.kind) {
    case "Document":
      rows.push(["Source", <span key="s">{node.source_uri}</span>]);
      rows.push(["Version", <span key="v">v{node.version}</span>]);
      rows.push(["Created", <span key="c">{formatDate(node.created_at)}</span>]);
      rows.push(["Content hash", <span key="h">{node.content_hash.slice(0, 16)}</span>]);
      if (node.supersedes_id) {
        rows.push([
          "Supersedes",
          <LinkId key="p" id={node.supersedes_id} onSelect={onSelect} />,
        ]);
      }
      break;
    case "Chunk":
      rows.push([
        "Document",
        <LinkId key="d" id={node.document_id} onSelect={onSelect} />,
      ]);
      rows.push(["Index", <span key="i">{node.chunk_index}</span>]);
      rows.push(["Tokens (est.)", <span key="t">{node.token_count}</span>]);
      rows.push(["Created", <span key="c">{formatDate(node.created_at)}</span>]);
      break;
    case "Entity":
      rows.push(["Type", <span key="t">{node.entity_type}</span>]);
      if (node.canonical_id) {
        rows.push([
          "Alias of",
          <LinkId key="a" id={node.canonical_id} onSelect={onSelect} />,
        ]);
      }
      rows.push(["Created", <span key="c">{formatDate(node.created_at)}</span>]);
      break;
    case "PipelineRun":
      rows.push(["Model", <span key="m">{node.embedding_model}</span>]);
      rows.push(["Chunker", <span key="k">{node.chunker}</span>]);
      rows.push(["Run at", <span key="r">{formatDate(node.run_at)}</span>]);
      rows.push(["Config hash", <span key="h">{node.config_hash.slice(0, 16)}</span>]);
      break;
  }

  const embeddings = "embeddings" in node ? node.embeddings : [];
  const metadata = Object.entries(node.metadata ?? {});

  return (
    <section className="inspector-section">
      <h3 className="section-title">Properties</h3>
      <dl className="kv">
        {rows.map(([label, value]) => (
          <ItemRow key={label} label={label}>
            {value}
          </ItemRow>
        ))}
      </dl>

      {embeddings.length > 0 && (
        <>
          <h3 className="section-title" style={{ marginTop: 16 }}>
            Embeddings
          </h3>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {embeddings.map((embedding) => (
              <span
                key={`${embedding.model}:${embedding.dim}`}
                className="badge model"
                title={`Embedded ${formatDate(embedding.embedded_at)}`}
              >
                {embedding.model} · {embedding.dim}d
              </span>
            ))}
          </div>
        </>
      )}

      {metadata.length > 0 && (
        <>
          <h3 className="section-title" style={{ marginTop: 16 }}>
            Metadata
          </h3>
          <dl className="kv">
            {metadata.map(([key, value]) => (
              <ItemRow key={key} label={key}>
                {typeof value === "object" ? JSON.stringify(value) : String(value)}
              </ItemRow>
            ))}
          </dl>
        </>
      )}
    </section>
  );
}

function ItemRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{children}</dd>
    </>
  );
}

function LinkId({ id, onSelect }: { id: string; onSelect: (id: string) => void }) {
  return (
    <button
      className="button subtle mono"
      style={{ padding: "0 4px", fontSize: 11 }}
      onClick={() => onSelect(id)}
    >
      {id.replace(/-/g, "").slice(-8)}
    </button>
  );
}

/** The pipeline run that produced this node — the provenance inspector. */
function Provenance({
  runId,
  onSelect,
}: {
  runId: string;
  onSelect: (id: string) => void;
}) {
  const run = useAsync((signal) => api.node(runId, signal), [runId]);
  const node = run.data?.node;
  if (!node || node.kind !== "PipelineRun") return null;
  const record = node as PipelineRunNode;

  return (
    <section className="inspector-section">
      <h3 className="section-title">Provenance</h3>
      <dl className="kv">
        <ItemRow label="Run">
          <LinkId id={record.id} onSelect={onSelect} />
        </ItemRow>
        <ItemRow label="Pipeline">{record.pipeline_version}</ItemRow>
        <ItemRow label="Model">{record.embedding_model}</ItemRow>
        <ItemRow label="Chunker">{record.chunker}</ItemRow>
        <ItemRow label="Run at">{formatDate(record.run_at)}</ItemRow>
        <ItemRow label="Config hash">{record.config_hash.slice(0, 16)}</ItemRow>
      </dl>
      <p className="faint" style={{ fontSize: 11, marginTop: 10, marginBottom: 0 }}>
        Two nodes share comparable embeddings only if they share this config
        hash.
      </p>
    </section>
  );
}

function EdgeList({
  title,
  edges,
  onSelect,
}: {
  title: string;
  edges: NodeDetail["outgoing"];
  onSelect: (id: string) => void;
}) {
  if (edges.length === 0) return null;
  return (
    <section className="inspector-section">
      <h3 className="section-title">{title}</h3>
      <div>
        {edges.map((edge) => (
          <div
            key={`${edge.edge.edge_type}-${edge.other_id}`}
            className="edge-row"
            onClick={() => onSelect(edge.other_id)}
          >
            <span className={`edge-type ${edgeClassOf(edge.edge.edge_type)}`}>
              {edge.edge.edge_type}
            </span>
            <span className="edge-label">
              {edge.other_label ?? edge.other_id.slice(0, 8)}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}
