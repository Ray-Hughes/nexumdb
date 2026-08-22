/** Database overview: what is in here, and what produced it. */

import { api } from "../lib/api";
import { formatBytes, formatDate, formatNumber } from "../lib/format";
import { useAsync } from "../lib/useAsync";
import { Empty, ErrorBanner, Loading, edgeClassOf } from "../components/common";
import type { EdgeType } from "../types";

export function Overview({ onOpenDocuments }: { onOpenDocuments: () => void }) {
  const stats = useAsync((signal) => api.stats(signal), []);
  const config = useAsync((signal) => api.config(signal), []);

  if (stats.loading && !stats.data) return <Loading label="Reading database" />;
  if (stats.error) return <ErrorBanner message={stats.error} onRetry={stats.reload} />;
  if (!stats.data) return null;

  const s = stats.data;
  const empty = s.documents === 0;

  return (
    <div className="content padded">
      {empty ? (
        <Empty
          title="This database is empty"
          hint="Ingest documents with `nexum ingest <path>` and they will appear here. The viewer reads the same database the CLI writes."
        />
      ) : (
        <>
          <div className="stat-grid">
            <Stat
              label="Documents"
              value={formatNumber(s.latest_documents)}
              sub={
                s.documents > s.latest_documents
                  ? `${formatNumber(s.documents - s.latest_documents)} superseded`
                  : "all current"
              }
              onClick={onOpenDocuments}
            />
            <Stat label="Chunks" value={formatNumber(s.chunks)} />
            <Stat label="Entities" value={formatNumber(s.entities)} />
            <Stat label="Edges" value={formatNumber(s.edges)} />
            <Stat
              label="Pipeline runs"
              value={formatNumber(s.pipeline_runs)}
              sub="ingestion history"
            />
            <Stat
              label="On disk"
              value={formatBytes(s.store_bytes + s.wal_bytes)}
              sub={`${formatBytes(s.wal_bytes)} log`}
            />
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
              gap: 16,
              marginTop: 16,
            }}
          >
            <section className="panel">
              <header className="panel-header">Embedding models</header>
              <div className="panel-body">
                {Object.keys(s.namespaces).length === 0 ? (
                  <span className="faint">No embeddings yet.</span>
                ) : (
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Namespace</th>
                        <th className="num">Dim</th>
                        <th className="num">Vectors</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(s.namespaces).map(([namespace, info]) => (
                        <tr key={namespace} style={{ cursor: "default" }}>
                          <td className="mono">{namespace}</td>
                          <td className="num">{info.dim}</td>
                          <td className="num">{formatNumber(info.count)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
                {config.data && (
                  <p className="faint" style={{ fontSize: 11, marginBottom: 0, marginTop: 12 }}>
                    Queries are embedded with{" "}
                    <span className="mono">{config.data.embedding_model}</span>.
                    Scores from different models are not comparable.
                  </p>
                )}
              </div>
            </section>

            <section className="panel">
              <header className="panel-header">Edges by type</header>
              <div className="panel-body">
                <table className="table">
                  <tbody>
                    {Object.entries(s.edges_by_type)
                      .sort((a, b) => b[1] - a[1])
                      .map(([type, count]) => (
                        <tr key={type} style={{ cursor: "default" }}>
                          <td>
                            <span
                              className={`edge-type ${edgeClassOf(type as EdgeType)}`}
                            >
                              {type}
                            </span>
                          </td>
                          <td className="num">{formatNumber(count)}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </section>
          </div>

          <p className="faint" style={{ fontSize: 11, marginTop: 16 }}>
            Created {formatDate(s.created_at)} · log position {s.applied_lsn} ·{" "}
            <span className="mono">{s.path}</span>
          </p>
        </>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  sub,
  onClick,
}: {
  label: string;
  value: string;
  sub?: string;
  onClick?: () => void;
}) {
  return (
    <div
      className="stat"
      onClick={onClick}
      style={onClick ? { cursor: "pointer" } : undefined}
    >
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}
