/**
 * Application shell.
 *
 * Holds the connection to a database, the current view, and the selected node.
 * Selection is app-wide on purpose: clicking a chunk in search, a point in the
 * projection, or a node in the graph should all put the same thing in the
 * inspector, so moving between views never loses your place.
 */

import { useCallback, useEffect, useState, type ReactElement } from "react";
import { api, getBaseUrl, setBaseUrl } from "./lib/api";
import {
  closeDatabase,
  currentDatabase,
  isDesktop,
  openDatabase,
  pickFolder,
  recentDatabases,
  rememberDatabase,
} from "./lib/tauri";
import { Inspector } from "./components/Inspector";
import { ErrorBanner, Spinner } from "./components/common";
import {
  BrandMark,
  ChunkIcon,
  CollectionIcon,
  FolderIcon,
  GraphIcon,
  HistoryIcon,
  OverviewIcon,
  ScatterIcon,
  SearchIcon,
} from "./components/Icons";
import { Chunks } from "./views/Chunks";
import { Collection } from "./views/Collection";
import { Graph } from "./views/Graph";
import { History } from "./views/History";
import { Overview } from "./views/Overview";
import { Projection } from "./views/Projection";
import { Search } from "./views/Search";
import type { ApiInfo } from "./types";

type View =
  | "overview"
  | "collection"
  | "chunks"
  | "search"
  | "graph"
  | "projection"
  | "history";

const NAV: {
  view: View;
  label: string;
  Icon: (props: { className?: string }) => ReactElement;
}[] = [
  { view: "overview", label: "Overview", Icon: OverviewIcon },
  { view: "collection", label: "Collection", Icon: CollectionIcon },
  { view: "chunks", label: "Chunks", Icon: ChunkIcon },
  { view: "search", label: "Search", Icon: SearchIcon },
  { view: "graph", label: "Graph", Icon: GraphIcon },
  { view: "projection", label: "Projection", Icon: ScatterIcon },
  { view: "history", label: "History", Icon: HistoryIcon },
];

export function App() {
  const [connection, setConnection] = useState<ApiInfo | null>(null);
  const [connecting, setConnecting] = useState(true);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [recents, setRecents] = useState<string[]>([]);

  const [view, setView] = useState<View>("overview");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [documentId, setDocumentId] = useState<string | null>(null);
  const [graphCenter, setGraphCenter] = useState<string | null>(null);

  // On start, reconnect to whatever the shell already had open; in a browser,
  // probe the default `nexum serve` port so development needs no extra step.
  useEffect(() => {
    let live = true;
    (async () => {
      try {
        const existing = await currentDatabase();
        if (!live) return;
        if (existing) {
          setBaseUrl(existing.base_url);
          setConnection(existing);
        } else if (!isDesktop()) {
          const health = await api.health();
          const config = await api.config();
          if (!live) return;
          setConnection({
            base_url: getBaseUrl(),
            database: health.database,
            embedding_model: config.embedding_model,
            embedding_dimensions: config.embedding_dimensions,
            engine_version: health.engine_version,
          });
        }
      } catch {
        // Nothing open yet; the welcome screen handles it.
      } finally {
        if (live) setConnecting(false);
      }
      if (live) setRecents(await recentDatabases());
    })();
    return () => {
      live = false;
    };
  }, []);

  const connect = useCallback(async (path: string, create: boolean) => {
    setConnecting(true);
    setConnectionError(null);
    try {
      const info = await openDatabase(path, create);
      setBaseUrl(info.base_url);
      setConnection(info);
      setSelectedId(null);
      setDocumentId(null);
      setGraphCenter(null);
      setView("overview");
      setRecents(await rememberDatabase(path));
    } catch (cause) {
      setConnectionError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setConnecting(false);
    }
  }, []);

  const disconnect = useCallback(async () => {
    await closeDatabase();
    setConnection(null);
    setSelectedId(null);
  }, []);

  const select = useCallback((id: string) => setSelectedId(id), []);

  const explore = useCallback((id: string) => {
    setGraphCenter(id);
    setView("graph");
  }, []);

  if (!connection) {
    return (
      <Welcome
        connecting={connecting}
        error={connectionError}
        recents={recents}
        onOpen={connect}
      />
    );
  }

  return (
    <div className={`app${selectedId ? " with-inspector" : ""}`}>
      <nav className="sidebar">
        <div className="brand">
          <div className="brand-name">
            <BrandMark className="brand-mark" />
            NexumDB
          </div>
          <div className="brand-db" title={connection.database}>
            {connection.database}
          </div>
        </div>

        <div className="nav">
          <div className="nav-section">Browse</div>
          {NAV.slice(0, 3).map(({ view: item, label, Icon }) => (
            <NavItem
              key={item}
              active={view === item}
              label={label}
              icon={<Icon />}
              onClick={() => setView(item)}
            />
          ))}
          <div className="nav-section">Explore</div>
          {NAV.slice(3).map(({ view: item, label, Icon }) => (
            <NavItem
              key={item}
              active={view === item}
              label={label}
              icon={<Icon />}
              onClick={() => setView(item)}
            />
          ))}
        </div>

        <div className="sidebar-footer">
          <span title="Queries are embedded with this model">
            {connection.embedding_model} · {connection.embedding_dimensions}d
          </span>
          <span>engine {connection.engine_version}</span>
          <button
            className="button subtle"
            style={{ marginTop: 4, justifyContent: "flex-start" }}
            onClick={disconnect}
          >
            <FolderIcon />
            Close database
          </button>
        </div>
      </nav>

      <main className="main">
        {view === "overview" && (
          <>
            <div className="toolbar">
              <h2 className="toolbar-title">Overview</h2>
            </div>
            <Overview onOpenDocuments={() => setView("collection")} />
          </>
        )}
        {view === "collection" && (
          <Collection
            selectedId={selectedId}
            onSelect={select}
            onOpenHistory={(id) => {
              setDocumentId(id);
              setView("history");
            }}
            onOpenChunks={(id) => {
              setDocumentId(id);
              setView("chunks");
            }}
          />
        )}
        {view === "chunks" && (
          <Chunks
            documentId={documentId}
            selectedId={selectedId}
            onSelect={select}
            onPickDocument={setDocumentId}
          />
        )}
        {view === "search" && <Search selectedId={selectedId} onSelect={select} />}
        {view === "graph" && (
          <Graph
            centerId={graphCenter ?? selectedId}
            onSelect={select}
            onCenter={setGraphCenter}
          />
        )}
        {view === "projection" && <Projection onSelect={select} />}
        {view === "history" && (
          <History
            documentId={documentId}
            onSelect={select}
            onPickDocument={setDocumentId}
          />
        )}
      </main>

      {selectedId && (
        <Inspector
          nodeId={selectedId}
          onSelect={select}
          onClose={() => setSelectedId(null)}
          onExplore={explore}
        />
      )}
    </div>
  );
}

function NavItem({
  active,
  label,
  icon,
  onClick,
}: {
  active: boolean;
  label: string;
  icon: ReactElement;
  onClick: () => void;
}) {
  return (
    <button
      className={`nav-item${active ? " active" : ""}`}
      onClick={onClick}
      aria-current={active ? "page" : undefined}
    >
      {icon}
      {label}
    </button>
  );
}

function Welcome({
  connecting,
  error,
  recents,
  onOpen,
}: {
  connecting: boolean;
  error: string | null;
  recents: string[];
  onOpen: (path: string, create: boolean) => void;
}) {
  const choose = async (create: boolean) => {
    const path = await pickFolder(
      create ? "Choose a folder for the new database" : "Choose a NexumDB database folder",
    );
    if (path) onOpen(path, create);
  };

  return (
    <div className="welcome">
      <BrandMark className="welcome-mark" />
      <h1 className="welcome-title">NexumDB</h1>
      <p className="welcome-sub">
        A graph-native vector database for RAG. Open a database to browse its
        collection, inspect provenance, explore the graph, and search.
      </p>

      {error && <ErrorBanner message={error} />}

      <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
        <button
          className="button primary"
          disabled={connecting}
          onClick={() => choose(false)}
        >
          {connecting ? <Spinner /> : <FolderIcon />}
          Open database
        </button>
        <button className="button" disabled={connecting} onClick={() => choose(true)}>
          Create new
        </button>
      </div>

      {recents.length > 0 && (
        <div className="recent-list">
          <div className="section-title" style={{ marginBottom: 4 }}>
            Recent
          </div>
          {recents.map((path) => (
            <button
              key={path}
              className="recent-item"
              onClick={() => onOpen(path, false)}
              title={path}
            >
              <FolderIcon />
              <span
                style={{
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  direction: "rtl",
                  textAlign: "left",
                }}
              >
                {path}
              </span>
            </button>
          ))}
        </div>
      )}

      {!isDesktop() && (
        <p className="faint" style={{ fontSize: 11, marginTop: 16, textAlign: "center" }}>
          Running in a browser. Start <span className="mono">nexum serve</span>{" "}
          and reload, or launch the desktop app.
        </p>
      )}
    </div>
  );
}
