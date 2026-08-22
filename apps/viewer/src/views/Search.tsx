/**
 * Search sandbox.
 *
 * Type a query, see ranked results with similarity scores, then expand via the
 * graph to see what traversal adds. The expansion is shown inline and marked
 * with its hop count and the edge it arrived on, so the value the graph adds
 * over plain vector search is visible rather than asserted.
 */

import { useState } from "react";
import { api } from "../lib/api";
import { formatNumber, truncate } from "../lib/format";
import { useAsync } from "../lib/useAsync";
import {
  CopyableId,
  Empty,
  ErrorBanner,
  KindBadge,
  Loading,
  edgeClassOf,
} from "../components/common";
import type { EdgeType, ResultNode } from "../types";

const EXPANDABLE: EdgeType[] = [
  "MENTIONS",
  "RELATES_TO",
  "PART_OF",
  "PRECEDES",
  "SIMILAR_TO",
];

interface Props {
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export function Search({ selectedId, onSelect }: Props) {
  const [draft, setDraft] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [topK, setTopK] = useState(10);
  const [expand, setExpand] = useState(false);
  const [edgeTypes, setEdgeTypes] = useState<EdgeType[]>(["MENTIONS", "RELATES_TO"]);
  const [hops, setHops] = useState(1);
  const [latestOnly, setLatestOnly] = useState(true);

  const results = useAsync(
    (signal) =>
      submitted.trim()
        ? api.search(
            {
              query: submitted,
              top_k: topK,
              latest_only: latestOnly,
              ...(expand
                ? {
                    expand: {
                      edge_types: edgeTypes,
                      max_hops: hops,
                      direction: "both" as const,
                    },
                  }
                : {}),
            },
            signal,
          )
        : Promise.resolve(null),
    [submitted, topK, latestOnly, expand, edgeTypes.join(","), hops],
  );

  const direct = results.data?.results.filter((r) => r.score != null) ?? [];
  const expanded = results.data?.results.filter((r) => r.score == null) ?? [];
  const best = direct[0]?.score ?? 1;

  return (
    <>
      <form
        className="search-bar"
        onSubmit={(event) => {
          event.preventDefault();
          setSubmitted(draft);
        }}
      >
        <input
          className="input search-input"
          placeholder="Search chunks by meaning…"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          autoFocus
        />
        <button className="button primary" type="submit" disabled={!draft.trim()}>
          Search
        </button>
        <label className="field">
          Top
          <input
            className="input"
            type="number"
            min={1}
            max={100}
            value={topK}
            style={{ width: 64 }}
            onChange={(event) => setTopK(Number(event.target.value) || 10)}
          />
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={latestOnly}
            onChange={(event) => setLatestOnly(event.target.checked)}
          />
          Current versions only
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={expand}
            onChange={(event) => setExpand(event.target.checked)}
          />
          Expand via graph
        </label>
        {expand && (
          <>
            <select
              className="select"
              multiple
              size={1}
              value={edgeTypes}
              style={{ minWidth: 170 }}
              onChange={(event) =>
                setEdgeTypes(
                  Array.from(event.target.selectedOptions).map(
                    (option) => option.value as EdgeType,
                  ),
                )
              }
            >
              {EXPANDABLE.map((edge) => (
                <option key={edge} value={edge}>
                  {edge}
                </option>
              ))}
            </select>
            <label className="field">
              Hops
              <input
                className="input"
                type="number"
                min={1}
                max={4}
                value={hops}
                style={{ width: 56 }}
                onChange={(event) => setHops(Number(event.target.value) || 1)}
              />
            </label>
          </>
        )}
      </form>

      {results.error && (
        <ErrorBanner message={results.error} onRetry={results.reload} />
      )}

      <div className="content">
        {!submitted.trim() ? (
          <Empty
            title="Search the collection"
            hint="Queries are embedded with the same model the documents were, then matched by vector similarity. Turn on “Expand via graph” to follow edges out from the matches and see what traversal adds."
          />
        ) : results.loading ? (
          <Loading label="Searching" />
        ) : direct.length === 0 ? (
          <Empty
            title="No matches"
            hint={
              latestOnly
                ? "Only current document versions were searched. Turn off “Current versions only” to include superseded ones."
                : "Nothing in this database is close to that query."
            }
          />
        ) : (
          <>
            <ResultList
              results={direct}
              best={best}
              selectedId={selectedId}
              onSelect={onSelect}
            />
            {expanded.length > 0 && (
              <>
                <div
                  className="section-title"
                  style={{ padding: "16px 16px 8px", borderTop: "1px solid var(--border)" }}
                >
                  Added by graph expansion ({expanded.length})
                </div>
                <ResultList
                  results={expanded}
                  best={best}
                  selectedId={selectedId}
                  onSelect={onSelect}
                />
              </>
            )}
            {results.data && (
              <div className="pager">
                <span className="mono">
                  {formatNumber(direct.length)} matches
                  {expanded.length > 0 && ` · ${formatNumber(expanded.length)} expanded`}
                  {" · "}
                  {formatNumber(results.data.stats.edges_traversed)} edges walked
                  {" · via "}
                  {results.data.query_model}
                </span>
              </div>
            )}
          </>
        )}
      </div>
    </>
  );
}

function ResultList({
  results,
  best,
  selectedId,
  onSelect,
}: {
  results: ResultNode[];
  best: number;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div>
      {results.map((result) => {
        const node = result.node;
        const score = result.score;
        return (
          <div
            key={node.id}
            className={`result${node.id === selectedId ? " selected" : ""}`}
            onClick={() => onSelect(node.id)}
          >
            <div className="result-score">
              {score != null ? (
                <>
                  {score.toFixed(3)}
                  <div className="score-bar">
                    <div
                      className="score-bar-fill"
                      style={{
                        width: `${Math.max(2, (score / Math.max(best, 1e-6)) * 100)}%`,
                      }}
                    />
                  </div>
                </>
              ) : (
                <span className="faint">+{result.hops}</span>
              )}
            </div>
            <div style={{ minWidth: 0 }}>
              <div className="result-text">
                {node.kind === "Chunk"
                  ? truncate(node.text, 320)
                  : node.kind === "Document"
                    ? node.title
                    : node.kind === "Entity"
                      ? node.name
                      : node.pipeline_version}
              </div>
              <div className="result-meta">
                <KindBadge kind={node.kind} />
                <CopyableId id={node.id} short />
                {node.kind === "Chunk" && <span>chunk {node.chunk_index}</span>}
                {node.kind === "Entity" && <span>{node.entity_type}</span>}
                {result.via_edge && (
                  <span>
                    reached via{" "}
                    <span className={`edge-type ${edgeClassOf(result.via_edge)}`}>
                      {result.via_edge}
                    </span>
                  </span>
                )}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
