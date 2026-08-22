/**
 * Collection browser: documents, versions, chunk counts, last ingested.
 */

import { useState } from "react";
import { api } from "../lib/api";
import { formatNumber, formatRelative, truncate } from "../lib/format";
import { useAsync } from "../lib/useAsync";
import { CopyableId, Empty, ErrorBanner, Loading, Pager } from "../components/common";

interface Props {
  selectedId: string | null;
  onSelect: (id: string) => void;
  onOpenHistory: (id: string) => void;
  onOpenChunks: (id: string) => void;
}

const PAGE_SIZE = 50;

export function Collection({
  selectedId,
  onSelect,
  onOpenHistory,
  onOpenChunks,
}: Props) {
  const [includeSuperseded, setIncludeSuperseded] = useState(false);
  const [offset, setOffset] = useState(0);
  const [filter, setFilter] = useState("");

  const page = useAsync(
    (signal) =>
      api.documents({ includeSuperseded, offset, limit: PAGE_SIZE }, signal),
    [includeSuperseded, offset],
  );

  const documents = (page.data?.items ?? []).filter((doc) => {
    if (!filter.trim()) return true;
    const needle = filter.toLowerCase();
    return (
      doc.title.toLowerCase().includes(needle) ||
      doc.source_uri.toLowerCase().includes(needle)
    );
  });

  return (
    <>
      <div className="toolbar">
        <h2 className="toolbar-title">Collection</h2>
        <input
          className="input"
          style={{ width: 220 }}
          placeholder="Filter by title or source"
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
        />
        <label className="checkbox">
          <input
            type="checkbox"
            checked={includeSuperseded}
            onChange={(event) => {
              setIncludeSuperseded(event.target.checked);
              setOffset(0);
            }}
          />
          Show superseded versions
        </label>
        <div className="toolbar-spacer" />
        <button className="button subtle" onClick={page.reload}>
          Refresh
        </button>
      </div>

      {page.error && <ErrorBanner message={page.error} onRetry={page.reload} />}

      <div className="content">
        {page.loading && !page.data ? (
          <Loading label="Loading documents" />
        ) : documents.length === 0 ? (
          <Empty
            title={filter ? "Nothing matches that filter" : "No documents yet"}
            hint={
              filter
                ? "Try a shorter search term."
                : "Run `nexum ingest <path>` against this database and refresh."
            }
          />
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Title</th>
                <th style={{ width: 90 }}>ID</th>
                <th className="num" style={{ width: 60 }}>
                  Version
                </th>
                <th className="num" style={{ width: 70 }}>
                  Chunks
                </th>
                <th style={{ width: 130 }}>Ingested</th>
                <th style={{ width: 90 }} />
              </tr>
            </thead>
            <tbody>
              {documents.map((doc) => (
                <tr
                  key={doc.id}
                  className={doc.id === selectedId ? "selected" : ""}
                  onClick={() => onSelect(doc.id)}
                >
                  <td>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span>{truncate(doc.title, 60)}</span>
                      {!doc.is_latest && (
                        <span className="badge neutral" title="A newer version exists">
                          superseded
                        </span>
                      )}
                    </div>
                    <div className="faint" style={{ fontSize: 11, marginTop: 2 }}>
                      {truncate(doc.source_uri, 90)}
                    </div>
                  </td>
                  <td>
                    <CopyableId id={doc.id} short />
                  </td>
                  <td className="num">v{doc.version}</td>
                  <td className="num">{formatNumber(doc.chunk_count)}</td>
                  <td className="muted" title={new Date(doc.created_at).toISOString()}>
                    {formatRelative(doc.created_at)}
                  </td>
                  <td>
                    <div style={{ display: "flex", gap: 4 }}>
                      <button
                        className="button subtle"
                        title="Show chunks"
                        onClick={(event) => {
                          event.stopPropagation();
                          onOpenChunks(doc.id);
                        }}
                      >
                        Chunks
                      </button>
                      {(doc.version > 1 || !doc.is_latest) && (
                        <button
                          className="button subtle"
                          title="Show version history"
                          onClick={(event) => {
                            event.stopPropagation();
                            onOpenHistory(doc.id);
                          }}
                        >
                          History
                        </button>
                      )}
                    </div>
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
          unit="documents"
        />
      )}
    </>
  );
}
