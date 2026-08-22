/**
 * Chunk table: paginated, with text preview, metadata, and a model badge.
 *
 * The model badge is not decoration. A chunk can carry embeddings from several
 * models, and which one produced a given vector determines whether a search
 * score means anything — so it belongs in the table, not buried in a detail
 * pane.
 */

import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { formatNumber, truncate } from "../lib/format";
import { useAsync } from "../lib/useAsync";
import { CopyableId, Empty, ErrorBanner, Loading, Pager } from "../components/common";
import type { DocumentSummary } from "../types";

interface Props {
  documentId: string | null;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onPickDocument: (id: string) => void;
}

const PAGE_SIZE = 50;

export function Chunks({
  documentId,
  selectedId,
  onSelect,
  onPickDocument,
}: Props) {
  const [offset, setOffset] = useState(0);
  const documents = useAsync(
    (signal) => api.documents({ limit: 500 }, signal),
    [],
  );

  // Default to the first document so the view is never blank on arrival.
  useEffect(() => {
    if (!documentId && documents.data?.items.length) {
      onPickDocument(documents.data.items[0]!.id);
    }
  }, [documentId, documents.data, onPickDocument]);

  useEffect(() => setOffset(0), [documentId]);

  const page = useAsync(
    (signal) =>
      documentId
        ? api.chunks(documentId, { offset, limit: PAGE_SIZE }, signal)
        : Promise.resolve(null),
    [documentId, offset],
  );

  const options: DocumentSummary[] = documents.data?.items ?? [];

  return (
    <>
      <div className="toolbar">
        <h2 className="toolbar-title">Chunks</h2>
        <select
          className="select"
          style={{ maxWidth: 380 }}
          value={documentId ?? ""}
          onChange={(event) => onPickDocument(event.target.value)}
        >
          {options.length === 0 && <option value="">No documents</option>}
          {options.map((doc) => (
            <option key={doc.id} value={doc.id}>
              {truncate(doc.title, 60)} — v{doc.version} ({doc.chunk_count} chunks)
            </option>
          ))}
        </select>
        <div className="toolbar-spacer" />
        {page.data && (
          <span className="toolbar-meta">
            {formatNumber(page.data.total)} chunks
          </span>
        )}
      </div>

      {page.error && <ErrorBanner message={page.error} onRetry={page.reload} />}

      <div className="content">
        {!documentId ? (
          <Empty
            title="No document selected"
            hint="Pick a document above, or ingest one to get started."
          />
        ) : page.loading && !page.data ? (
          <Loading label="Loading chunks" />
        ) : (page.data?.items.length ?? 0) === 0 ? (
          <Empty title="This document has no chunks" />
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th className="num" style={{ width: 44 }}>
                  #
                </th>
                <th>Text</th>
                <th className="num" style={{ width: 70 }}>
                  Tokens
                </th>
                <th style={{ width: 190 }}>Embeddings</th>
                <th style={{ width: 90 }}>ID</th>
              </tr>
            </thead>
            <tbody>
              {page.data?.items.map((chunk) => (
                <tr
                  key={chunk.id}
                  className={chunk.id === selectedId ? "selected" : ""}
                  onClick={() => onSelect(chunk.id)}
                >
                  <td className="num muted">{chunk.chunk_index}</td>
                  <td style={{ maxWidth: 0 }}>
                    <div style={{ lineHeight: 1.6 }}>{truncate(chunk.text, 220)}</div>
                  </td>
                  <td className="num muted">{chunk.token_count}</td>
                  <td>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                      {chunk.embeddings.length === 0 ? (
                        <span className="badge warning" title="This chunk is not searchable">
                          none
                        </span>
                      ) : (
                        chunk.embeddings.map((embedding) => (
                          <span
                            key={`${embedding.model}:${embedding.dim}`}
                            className="badge model"
                            title={`${embedding.model}, ${embedding.dim} dimensions`}
                          >
                            {truncate(embedding.model, 22)}
                          </span>
                        ))
                      )}
                    </div>
                  </td>
                  <td>
                    <CopyableId id={chunk.id} short />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {page.data && (
        <Pager
          offset={offset}
          limit={PAGE_SIZE}
          total={page.data.total}
          onChange={setOffset}
          unit="chunks"
        />
      )}
    </>
  );
}
