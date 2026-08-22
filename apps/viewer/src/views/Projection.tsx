/**
 * Embedding projection: a 2D scatter of chunk vectors, coloured by document.
 *
 * The point is spotting duplicates and outliers — a tight cluster of points
 * from different documents is near-duplicate content, and a lone point far
 * from everything is usually a chunking accident. Hovering names the chunk so
 * that suspicion can be checked immediately rather than noted for later.
 *
 * The projection method is labelled, because PCA and the neighbourhood
 * refinement say different things and a plot that hid which one ran would be
 * misleading.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../lib/api";
import { colorForKey, createScatter, type ScatterHandle } from "../lib/scatter";
import { truncate } from "../lib/format";
import { useAsync } from "../lib/useAsync";
import { Empty, ErrorBanner, Loading } from "../components/common";
import type { ProjectionMethod, ProjectionPoint } from "../types";

interface Props {
  onSelect: (id: string) => void;
}

interface View {
  panX: number;
  panY: number;
  zoom: number;
}

const INITIAL_VIEW: View = { panX: 0, panY: 0, zoom: 1 };

export function Projection({ onSelect }: Props) {
  const [method, setMethod] = useState<ProjectionMethod>("neighborhood");
  const [includeSuperseded, setIncludeSuperseded] = useState(false);
  const [view, setView] = useState<View>(INITIAL_VIEW);
  const [hovered, setHovered] = useState<ProjectionPoint | null>(null);
  const [pointer, setPointer] = useState({ x: 0, y: 0 });
  const [webglFailed, setWebglFailed] = useState(false);

  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const scatterRef = useRef<ScatterHandle | null>(null);
  const sizeRef = useRef({ width: 1, height: 1 });
  const dragRef = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);

  const projection = useAsync(
    (signal) => api.projection({ method, includeSuperseded, limit: 5000 }, signal),
    [method, includeSuperseded],
  );

  const points = projection.data?.points ?? [];

  // Assign each document a colour once, so the legend and the plot agree and
  // the mapping does not shift when the projection is recomputed.
  const documentColors = useMemo(() => {
    const map = new Map<string, [number, number, number]>();
    let index = 0;
    for (const point of points) {
      if (!map.has(point.document_id)) {
        map.set(point.document_id, colorForKey(index));
        index += 1;
      }
    }
    return map;
  }, [points]);

  const geometry = useMemo(() => {
    const positions = new Float32Array(points.length * 2);
    const colors = new Float32Array(points.length * 3);
    points.forEach((point, i) => {
      positions[i * 2] = point.x;
      positions[i * 2 + 1] = point.y;
      const color = documentColors.get(point.document_id) ?? [0.6, 0.6, 0.6];
      colors[i * 3] = color[0];
      colors[i * 3 + 1] = color[1];
      colors[i * 3 + 2] = color[2];
    });
    return { positions, colors };
  }, [points, documentColors]);

  // Set up the renderer once.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const scatter = createScatter(canvas);
    if (!scatter) {
      setWebglFailed(true);
      return;
    }
    scatterRef.current = scatter;
    return () => {
      scatter.dispose();
      scatterRef.current = null;
    };
  }, []);

  const render = useCallback(() => {
    const scatter = scatterRef.current;
    if (!scatter) return;
    // Points shrink as the cloud grows, so a dense plot stays readable.
    const pointSize = Math.max(3, Math.min(9, 900 / Math.sqrt(scatter.count || 1)));
    scatter.draw({ ...view, pointSize: pointSize * Math.min(view.zoom, 2) });
  }, [view]);

  useEffect(() => {
    const element = wrapRef.current;
    const scatter = scatterRef.current;
    if (!element || !scatter) return;
    const observer = new ResizeObserver(([entry]) => {
      if (!entry) return;
      const { width, height } = entry.contentRect;
      sizeRef.current = { width, height };
      scatter.resize(width, height, window.devicePixelRatio || 1);
      render();
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [render]);

  useEffect(() => {
    const scatter = scatterRef.current;
    if (!scatter) return;
    scatter.setData(geometry.positions, geometry.colors);
    render();
  }, [geometry, render]);

  useEffect(render, [render]);

  /** Screen position of a world-space point, for hit testing and tooltips. */
  const toScreen = useCallback(
    (x: number, y: number) => {
      const { width, height } = sizeRef.current;
      const scale = Math.min(width, height) * 0.45 * view.zoom;
      return {
        x: x * scale + view.panX + width / 2,
        y: y * scale + view.panY + height / 2,
      };
    },
    [view],
  );

  const findNearest = useCallback(
    (screenX: number, screenY: number): ProjectionPoint | null => {
      let best: ProjectionPoint | null = null;
      let bestDistance = 14; // pixels
      for (const point of points) {
        const screen = toScreen(point.x, point.y);
        const distance = Math.hypot(screen.x - screenX, screen.y - screenY);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = point;
        }
      }
      return best;
    },
    [points, toScreen],
  );

  const documentCount = documentColors.size;

  return (
    <>
      <div className="toolbar">
        <h2 className="toolbar-title">Embedding projection</h2>
        <select
          className="select"
          value={method}
          onChange={(event) => setMethod(event.target.value as ProjectionMethod)}
        >
          <option value="neighborhood">Neighbourhood (clusters)</option>
          <option value="pca">PCA (linear)</option>
        </select>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={includeSuperseded}
            onChange={(event) => setIncludeSuperseded(event.target.checked)}
          />
          Include superseded
        </label>
        <button className="button subtle" onClick={() => setView(INITIAL_VIEW)}>
          Reset view
        </button>
        <div className="toolbar-spacer" />
        {projection.data && (
          <span className="toolbar-meta">
            {points.length.toLocaleString()} chunks · {documentCount} documents ·{" "}
            {projection.data.dimensions}d
            {projection.data.explained_variance != null &&
              ` · ${(projection.data.explained_variance * 100).toFixed(0)}% variance`}
            {projection.data.truncated && " · truncated"}
          </span>
        )}
      </div>

      {projection.error && (
        <ErrorBanner message={projection.error} onRetry={projection.reload} />
      )}

      <div
        className="canvas-wrap"
        ref={wrapRef}
        onMouseDown={(event) => {
          dragRef.current = {
            x: event.clientX,
            y: event.clientY,
            panX: view.panX,
            panY: view.panY,
          };
        }}
        onMouseMove={(event) => {
          const rect = wrapRef.current?.getBoundingClientRect();
          const localX = event.clientX - (rect?.left ?? 0);
          const localY = event.clientY - (rect?.top ?? 0);
          setPointer({ x: localX, y: localY });

          const drag = dragRef.current;
          if (drag) {
            setView((current) => ({
              ...current,
              panX: drag.panX + (event.clientX - drag.x),
              panY: drag.panY + (event.clientY - drag.y),
            }));
            return;
          }
          setHovered(findNearest(localX, localY));
        }}
        onMouseUp={() => {
          dragRef.current = null;
        }}
        onMouseLeave={() => {
          dragRef.current = null;
          setHovered(null);
        }}
        onWheel={(event) => {
          // Zoom toward the cursor, which is what every map interaction does
          // and what the hand expects.
          const rect = wrapRef.current?.getBoundingClientRect();
          const localX = event.clientX - (rect?.left ?? 0);
          const localY = event.clientY - (rect?.top ?? 0);
          setView((current) => {
            const factor = Math.exp(-event.deltaY * 0.0015);
            const zoom = Math.min(40, Math.max(0.25, current.zoom * factor));
            const ratio = zoom / current.zoom;
            const { width, height } = sizeRef.current;
            const cx = localX - width / 2;
            const cy = localY - height / 2;
            return {
              zoom,
              panX: cx - (cx - current.panX) * ratio,
              panY: cy - (cy - current.panY) * ratio,
            };
          });
        }}
        onClick={() => {
          if (hovered) onSelect(hovered.id);
        }}
        style={{ cursor: dragRef.current ? "grabbing" : hovered ? "pointer" : "grab" }}
      >
        <canvas ref={canvasRef} style={{ display: "block" }} />

        {webglFailed && (
          <Empty
            title="WebGL is unavailable"
            hint="This view needs WebGL2 to draw thousands of points at once. Everything else in the app works without it."
          />
        )}

        {!webglFailed && projection.loading && !projection.data && (
          <div style={{ position: "absolute", inset: 0 }}>
            <Loading label="Projecting embeddings" />
          </div>
        )}

        {!webglFailed && projection.data && points.length === 0 && (
          <div style={{ position: "absolute", inset: 0 }}>
            <Empty
              title="Nothing to project"
              hint="Ingest documents so there are embeddings to lay out."
            />
          </div>
        )}

        {!webglFailed && points.length > 0 && (
          <div className="canvas-overlay">
            <div className="legend">
              <span className="legend-item">
                Colour = document · drag to pan · scroll to zoom · click a point
                to inspect
              </span>
            </div>
            <div className="legend">
              <span className="legend-item">
                {method === "pca"
                  ? "PCA: linear, preserves global structure"
                  : "Neighbourhood: clusters separated, distances not to scale"}
              </span>
            </div>
          </div>
        )}

        {hovered && (
          <div
            className="tooltip"
            style={{
              left: Math.min(pointer.x + 14, sizeRef.current.width - 340),
              top: Math.min(pointer.y + 14, sizeRef.current.height - 80),
            }}
          >
            <div className="mono faint" style={{ fontSize: 10.5 }}>
              chunk {hovered.chunk_index} ·{" "}
              {hovered.document_id.replace(/-/g, "").slice(-8)}
            </div>
            <div style={{ marginTop: 3 }}>{truncate(hovered.preview, 200)}</div>
          </div>
        )}
      </div>
    </>
  );
}
