/**
 * Graph explorer.
 *
 * Centred on one node, expandable by edge type, with a hop-depth control. The
 * server returns every edge between nodes in the view — not only the ones the
 * traversal walked — because a graph drawn from traversal edges alone renders
 * as a tree and hides exactly the cross-links that make it a graph.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { api } from "../lib/api";
import { EDGE_CLASS_HEX, KIND_HEX, truncate } from "../lib/format";
import { useAsync } from "../lib/useAsync";
import { Empty, ErrorBanner, Loading } from "../components/common";
import type { EdgeType, GraphViewNode, NodeKind } from "../types";

const EDGE_OPTIONS: EdgeType[] = [
  "PART_OF",
  "PRECEDES",
  "SUPERSEDES",
  "MENTIONS",
  "RELATES_TO",
  "SIMILAR_TO",
  "DERIVED_FROM",
  "EMBEDDED_BY",
  "EXTRACTED_BY",
];

interface Props {
  centerId: string | null;
  onSelect: (id: string) => void;
  onCenter: (id: string) => void;
}

interface ForceNode extends GraphViewNode {
  x?: number;
  y?: number;
}

export function Graph({ centerId, onSelect, onCenter }: Props) {
  const [hops, setHops] = useState(2);
  const [edges, setEdges] = useState<EdgeType[]>([]);
  const [hovered, setHovered] = useState<ForceNode | null>(null);
  const [pointer, setPointer] = useState({ x: 0, y: 0 });
  const wrapRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 800, height: 600 });

  // Fall back to a document when nothing is selected, so the view is never
  // an empty canvas on first open.
  const fallback = useAsync(
    (signal) => (centerId ? Promise.resolve(null) : api.documents({ limit: 1 }, signal)),
    [centerId],
  );
  const effectiveCenter = centerId ?? fallback.data?.items[0]?.id ?? null;

  const graph = useAsync(
    (signal) =>
      effectiveCenter
        ? api.graph(
            effectiveCenter,
            { hops, limit: 400, ...(edges.length ? { edges } : {}) },
            signal,
          )
        : Promise.resolve(null),
    [effectiveCenter, hops, edges.join(",")],
  );

  // The force layout needs explicit pixel dimensions, so track the container.
  useEffect(() => {
    const element = wrapRef.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      if (!entry) return;
      setSize({
        width: Math.max(1, entry.contentRect.width),
        height: Math.max(1, entry.contentRect.height),
      });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  // react-force-graph mutates the objects it is given, so hand it fresh copies
  // on every load — otherwise stale x/y positions leak across centre changes.
  const data = useMemo(() => {
    if (!graph.data) return { nodes: [], links: [] };
    return {
      nodes: graph.data.nodes.map((node) => ({ ...node })),
      links: graph.data.links.map((link) => ({ ...link })),
    };
  }, [graph.data]);

  const paintNode = useCallback(
    (node: ForceNode, ctx: CanvasRenderingContext2D, scale: number) => {
      const isCenter = node.id === effectiveCenter;
      // Centre largest, then decreasing with distance: the eye should find
      // the focus without reading any labels.
      const radius = isCenter ? 7 : Math.max(3, 6 - node.hops);
      const color = KIND_HEX[node.kind as NodeKind] ?? "#888";

      ctx.beginPath();
      ctx.arc(node.x ?? 0, node.y ?? 0, radius, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();

      if (isCenter) {
        ctx.strokeStyle = "#dde3ec";
        ctx.lineWidth = 2 / scale;
        ctx.stroke();
      }

      // Labels only once zoomed in, and only for near nodes — drawing every
      // label at every zoom turns a dense graph into unreadable soup.
      if (scale > 1.4 && (isCenter || node.hops <= 1)) {
        const label = truncate(node.label, 26);
        ctx.font = `${11 / scale}px ui-monospace, monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillStyle = "rgba(221, 227, 236, 0.85)";
        ctx.fillText(label, node.x ?? 0, (node.y ?? 0) + radius + 2 / scale);
      }
    },
    [effectiveCenter],
  );

  return (
    <>
      <div className="toolbar">
        <h2 className="toolbar-title">Graph explorer</h2>
        <label className="field">
          Hops
          <input
            className="input"
            type="number"
            min={1}
            max={5}
            value={hops}
            style={{ width: 56 }}
            onChange={(event) => setHops(Number(event.target.value) || 1)}
          />
        </label>
        <select
          className="select"
          multiple
          size={1}
          value={edges}
          style={{ minWidth: 190 }}
          title="Edge types to follow. Select none to follow all."
          onChange={(event) =>
            setEdges(
              Array.from(event.target.selectedOptions).map(
                (option) => option.value as EdgeType,
              ),
            )
          }
        >
          {EDGE_OPTIONS.map((edge) => (
            <option key={edge} value={edge}>
              {edge}
            </option>
          ))}
        </select>
        {edges.length > 0 && (
          <button className="button subtle" onClick={() => setEdges([])}>
            All edge types
          </button>
        )}
        <div className="toolbar-spacer" />
        {graph.data && (
          <span className="toolbar-meta">
            {graph.data.nodes.length} nodes · {graph.data.links.length} edges
            {graph.data.truncated && " · truncated"}
          </span>
        )}
      </div>

      {graph.error && <ErrorBanner message={graph.error} onRetry={graph.reload} />}

      <div className="canvas-wrap" ref={wrapRef}>
        {!effectiveCenter ? (
          <Empty
            title="Nothing to explore yet"
            hint="Select a node anywhere in the app, or ingest documents to build a graph."
          />
        ) : graph.loading && !graph.data ? (
          <Loading label="Building graph" />
        ) : (
          <>
            <div className="canvas-overlay">
              <div className="legend">
                {(Object.keys(KIND_HEX) as NodeKind[]).map((kind) => (
                  <span className="legend-item" key={kind}>
                    <span className="dot" style={{ color: KIND_HEX[kind] }} />
                    {kind}
                  </span>
                ))}
              </div>
              {graph.data?.truncated && (
                <div className="legend" style={{ color: "var(--warning)" }}>
                  Showing the first 400 nodes — narrow the edge types or reduce
                  hops to see the rest.
                </div>
              )}
            </div>

            <ForceGraph2D
              width={size.width}
              height={size.height}
              graphData={data}
              backgroundColor="rgba(0,0,0,0)"
              nodeRelSize={5}
              nodeCanvasObject={paintNode as never}
              nodePointerAreaPaint={((
                node: ForceNode,
                color: string,
                ctx: CanvasRenderingContext2D,
              ) => {
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(node.x ?? 0, node.y ?? 0, 8, 0, 2 * Math.PI);
                ctx.fill();
              }) as never}
              linkColor={((link: { class: keyof typeof EDGE_CLASS_HEX }) =>
                EDGE_CLASS_HEX[link.class] ?? "#333") as never}
              linkWidth={1}
              linkDirectionalArrowLength={3}
              linkDirectionalArrowRelPos={1}
              cooldownTicks={90}
              onNodeClick={((node: ForceNode) => onSelect(node.id)) as never}
              onNodeRightClick={((node: ForceNode) => onCenter(node.id)) as never}
              onNodeHover={((node: ForceNode | null) => setHovered(node)) as never}
              onBackgroundClick={() => setHovered(null)}
            />

            <div
              onMouseMove={(event) => {
                const rect = wrapRef.current?.getBoundingClientRect();
                setPointer({
                  x: event.clientX - (rect?.left ?? 0),
                  y: event.clientY - (rect?.top ?? 0),
                });
              }}
              style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
            />

            {hovered && (
              <div
                className="tooltip"
                style={{
                  left: Math.min(pointer.x + 14, size.width - 330),
                  top: Math.min(pointer.y + 14, size.height - 90),
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span className="dot" style={{ color: KIND_HEX[hovered.kind] }} />
                  <strong>{hovered.kind}</strong>
                  <span className="faint mono">
                    {hovered.hops === 0 ? "centre" : `${hovered.hops} hop`}
                  </span>
                </div>
                <div style={{ marginTop: 4 }}>{truncate(hovered.label, 160)}</div>
                <div className="faint" style={{ marginTop: 4, fontSize: 10.5 }}>
                  Click to inspect · right-click to recentre
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </>
  );
}
