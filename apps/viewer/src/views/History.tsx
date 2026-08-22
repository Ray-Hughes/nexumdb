/**
 * Version history: the supersession chain, and what changed between versions.
 *
 * Because documents are immutable and a re-ingest writes a fresh set of
 * chunks, "what changed" is an alignment over chunk texts rather than a text
 * diff — so the diff is shown at chunk granularity, which is also the
 * granularity retrieval works at.
 */

import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";
import { diffChunks } from "../lib/diff";
import { formatDate, truncate } from "../lib/format";
import { useAsync } from "../lib/useAsync";
import { CopyableId, Empty, ErrorBanner, Loading } from "../components/common";
import type { DocumentSummary } from "../types";

interface Props {
  documentId: string | null;
  onSelect: (id: string) => void;
  onPickDocument: (id: string) => void;
}

export function History({ documentId, onSelect, onPickDocument }: Props) {
  const documents = useAsync(
    (signal) => api.documents({ includeSuperseded: false, limit: 500 }, signal),
    [],
  );

  useEffect(() => {
    if (!documentId && documents.data?.items.length) {
      onPickDocument(documents.data.items[0]!.id);
    }
  }, [documentId, documents.data, onPickDocument]);

  const history = useAsync(
    (signal) => (documentId ? api.history(documentId, signal) : Promise.resolve(null)),
    [documentId],
  );

  const versions = history.data ?? [];
  const [comparison, setComparison] = useState<[number, number] | null>(null);

  // Default to comparing the two newest versions — the change someone opening
  // this view almost always wants to see.
  useEffect(() => {
    setComparison(versions.length >= 2 ? [versions.length - 2, versions.length - 1] : null);
  }, [versions.length, documentId]);

  const options: DocumentSummary[] = documents.data?.items ?? [];

  return (
    <>
      <div className="toolbar">
        <h2 className="toolbar-title">Version history</h2>
        <select
          className="select"
          style={{ maxWidth: 420 }}
          value={documentId ?? ""}
          onChange={(event) => onPickDocument(event.target.value)}
        >
          {options.length === 0 && <option value="">No documents</option>}
          {options.map((doc) => (
            <option key={doc.id} value={doc.id}>
              {truncate(doc.title, 60)} — v{doc.version}
            </option>
          ))}
        </select>
        <div className="toolbar-spacer" />
        {versions.length > 0 && (
          <span className="toolbar-meta">
            {versions.length} version{versions.length === 1 ? "" : "s"}
          </span>
        )}
      </div>

      {history.error && (
        <ErrorBanner message={history.error} onRetry={history.reload} />
      )}

      <div className="content padded">
        {!documentId ? (
          <Empty title="No document selected" />
        ) : history.loading && !history.data ? (
          <Loading label="Tracing version chain" />
        ) : versions.length === 0 ? (
          <Empty title="No history for this document" />
        ) : (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: versions.length > 1 ? "320px 1fr" : "1fr",
              gap: 16,
              alignItems: "start",
            }}
          >
            <section className="panel">
              <header className="panel-header">Timeline</header>
              <div className="panel-body">
                <div className="timeline">
                  {versions.map((version, index) => {
                    const isCurrent = index === versions.length - 1;
                    const inComparison =
                      comparison?.[0] === index || comparison?.[1] === index;
                    return (
                      <div
                        key={version.id}
                        className={`timeline-item${isCurrent ? " current" : ""}`}
                      >
                        <span className="timeline-dot" />
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 8,
                            flexWrap: "wrap",
                          }}
                        >
                          <strong className="mono">v{version.version}</strong>
                          {isCurrent && <span className="badge success">current</span>}
                          {inComparison && (
                            <span className="badge neutral">comparing</span>
                          )}
                        </div>
                        <div className="faint" style={{ fontSize: 11, marginTop: 2 }}>
                          {formatDate(version.created_at)}
                        </div>
                        <div style={{ marginTop: 4, fontSize: 12 }}>
                          {truncate(version.title, 42)}
                        </div>
                        <div style={{ marginTop: 4, display: "flex", gap: 8 }}>
                          <CopyableId id={version.id} short />
                          <button
                            className="button subtle"
                            style={{ padding: "0 6px", fontSize: 11 }}
                            onClick={() => onSelect(version.id)}
                          >
                            Inspect
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </section>

            {versions.length > 1 && comparison && (
              <Diff
                beforeId={versions[comparison[0]]!.id}
                afterId={versions[comparison[1]]!.id}
                beforeVersion={versions[comparison[0]]!.version}
                afterVersion={versions[comparison[1]]!.version}
                versions={versions.map((v) => v.version)}
                comparison={comparison}
                onChange={setComparison}
              />
            )}
          </div>
        )}
      </div>
    </>
  );
}

function Diff({
  beforeId,
  afterId,
  beforeVersion,
  afterVersion,
  versions,
  comparison,
  onChange,
}: {
  beforeId: string;
  afterId: string;
  beforeVersion: number;
  afterVersion: number;
  versions: number[];
  comparison: [number, number];
  onChange: (next: [number, number]) => void;
}) {
  const before = useAsync(
    (signal) => api.chunks(beforeId, { limit: 1000 }, signal),
    [beforeId],
  );
  const after = useAsync(
    (signal) => api.chunks(afterId, { limit: 1000 }, signal),
    [afterId],
  );

  const summary = useMemo(() => {
    if (!before.data || !after.data) return null;
    return diffChunks(
      before.data.items.map((c) => c.text),
      after.data.items.map((c) => c.text),
    );
  }, [before.data, after.data]);

  return (
    <section className="panel">
      <header className="panel-header">
        <span>Changes</span>
        <div style={{ flex: 1 }} />
        <select
          className="select"
          value={comparison[0]}
          onChange={(event) => onChange([Number(event.target.value), comparison[1]])}
        >
          {versions.map((version, index) => (
            <option key={version} value={index}>
              v{version}
            </option>
          ))}
        </select>
        <span className="faint">→</span>
        <select
          className="select"
          value={comparison[1]}
          onChange={(event) => onChange([comparison[0], Number(event.target.value)])}
        >
          {versions.map((version, index) => (
            <option key={version} value={index}>
              v{version}
            </option>
          ))}
        </select>
      </header>
      <div className="panel-body">
        {before.loading || after.loading ? (
          <Loading label="Comparing chunks" />
        ) : !summary ? (
          <Empty title="Could not load chunks" />
        ) : (
          <>
            <div style={{ display: "flex", gap: 12, marginBottom: 12, flexWrap: "wrap" }}>
              <span className="badge success">+{summary.added} added</span>
              <span className="badge danger">−{summary.removed} removed</span>
              <span className="badge neutral">{summary.unchanged} unchanged</span>
              <span className="faint" style={{ fontSize: 11, alignSelf: "center" }}>
                v{beforeVersion} → v{afterVersion}, by chunk
              </span>
            </div>
            {summary.added === 0 && summary.removed === 0 ? (
              <p className="muted" style={{ margin: 0 }}>
                These versions produced identical chunks. The document was
                re-ingested, but its content did not change in a way chunking
                could see.
              </p>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                {summary.lines.map((line, index) => (
                  <div key={index} className={`diff-line ${line.op}`}>
                    <span className="diff-marker">
                      {line.op === "added" ? "+" : line.op === "removed" ? "−" : " "}
                    </span>
                    <span>{truncate(line.text, 400)}</span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </section>
  );
}
