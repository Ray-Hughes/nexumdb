/**
 * Typed client for the NexumDB HTTP API.
 *
 * The base URL is discovered at runtime: under Tauri the shell reports the
 * loopback port it bound; in a browser dev session it falls back to the
 * default `nexum serve` port so the UI can be worked on without the desktop
 * shell in the loop.
 */

import type {
  DbStats,
  DocumentSummary,
  ChunkNode,
  DocumentNode,
  GraphView,
  GraphNodeRecord,
  IngestReport,
  NodeDetail,
  Page,
  ProjectionMethod,
  ProjectionResponse,
  QueryResult,
  SearchResults,
  ServerConfigInfo,
  EdgeType,
  Direction,
} from "../types";

/** An API call that failed, carrying the server's own explanation. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly kind: string,
  ) {
    super(message);
    this.name = "ApiError";
  }

  /** Whether retrying unchanged could plausibly succeed. */
  get retryable(): boolean {
    return this.status === 0 || this.status >= 500;
  }
}

let baseUrl = "http://127.0.0.1:8080";

export function setBaseUrl(url: string): void {
  baseUrl = url.replace(/\/+$/, "");
}

export function getBaseUrl(): string {
  return baseUrl;
}

async function request<T>(
  path: string,
  init?: RequestInit & { signal?: AbortSignal },
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${baseUrl}${path}`, {
      ...init,
      headers: {
        "content-type": "application/json",
        ...(init?.headers ?? {}),
      },
    });
  } catch (cause) {
    // A network-level failure has no status; surface it as one the UI can
    // distinguish from a rejected request.
    if (cause instanceof DOMException && cause.name === "AbortError") throw cause;
    throw new ApiError(
      `Could not reach the database at ${baseUrl}. Is it still open?`,
      0,
      "unreachable",
    );
  }

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    let kind = "unknown";
    try {
      const body = await response.json();
      if (typeof body?.error === "string") message = body.error;
      if (typeof body?.kind === "string") kind = body.kind;
    } catch {
      // Body was not the JSON error shape; the status line is what we have.
    }
    throw new ApiError(message, response.status, kind);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

function query(params: Record<string, string | number | boolean | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === "") continue;
    search.set(key, String(value));
  }
  const text = search.toString();
  return text ? `?${text}` : "";
}

export const api = {
  health: (signal?: AbortSignal) =>
    request<{ status: string; engine_version: string; database: string }>(
      "/health",
      { signal },
    ),

  config: (signal?: AbortSignal) =>
    request<ServerConfigInfo>("/api/config", { signal }),

  stats: (signal?: AbortSignal) => request<DbStats>("/api/stats", { signal }),

  documents: (
    options: { includeSuperseded?: boolean; offset?: number; limit?: number } = {},
    signal?: AbortSignal,
  ) =>
    request<Page<DocumentSummary>>(
      `/api/documents${query({
        include_superseded: options.includeSuperseded,
        offset: options.offset,
        limit: options.limit,
      })}`,
      { signal },
    ),

  document: (id: string, signal?: AbortSignal) =>
    request<DocumentNode>(`/api/documents/${id}`, { signal }),

  history: (id: string, signal?: AbortSignal) =>
    request<DocumentNode[]>(`/api/documents/${id}/history`, { signal }),

  chunks: (
    documentId: string,
    options: { offset?: number; limit?: number } = {},
    signal?: AbortSignal,
  ) =>
    request<Page<ChunkNode>>(
      `/api/documents/${documentId}/chunks${query({
        offset: options.offset,
        limit: options.limit,
      })}`,
      { signal },
    ),

  node: (id: string, signal?: AbortSignal) =>
    request<NodeDetail>(`/api/nodes/${id}`, { signal }),

  neighbors: (
    id: string,
    options: { edges?: EdgeType[]; direction?: Direction } = {},
    signal?: AbortSignal,
  ) =>
    request<GraphNodeRecord[]>(
      `/api/nodes/${id}/neighbors${query({
        edges: options.edges?.join(","),
        direction: options.direction,
      })}`,
      { signal },
    ),

  graph: (
    id: string,
    options: { hops?: number; edges?: EdgeType[]; limit?: number } = {},
    signal?: AbortSignal,
  ) =>
    request<GraphView>(
      `/api/graph/${id}${query({
        hops: options.hops,
        edges: options.edges?.join(","),
        limit: options.limit,
      })}`,
      { signal },
    ),

  search: (
    body: {
      query?: string;
      vector?: number[];
      top_k?: number;
      latest_only?: boolean;
      model?: string;
      expand?: { edge_types: EdgeType[]; max_hops: number; direction: Direction };
    },
    signal?: AbortSignal,
  ) =>
    request<SearchResults>("/api/search", {
      method: "POST",
      body: JSON.stringify(body),
      signal,
    }),

  traverse: (
    body: {
      start_ids: string[];
      edge_types?: EdgeType[];
      max_hops?: number;
      direction?: Direction;
    },
    signal?: AbortSignal,
  ) =>
    request<QueryResult>("/api/traverse", {
      method: "POST",
      body: JSON.stringify(body),
      signal,
    }),

  projection: (
    options: {
      model?: string;
      method?: ProjectionMethod;
      limit?: number;
      includeSuperseded?: boolean;
    } = {},
    signal?: AbortSignal,
  ) =>
    request<ProjectionResponse>(
      `/api/projection${query({
        model: options.model,
        method: options.method,
        limit: options.limit,
        include_superseded: options.includeSuperseded,
      })}`,
      { signal },
    ),

  ingestText: (
    body: { source_uri: string; title: string; text: string },
    signal?: AbortSignal,
  ) =>
    request<IngestReport[]>("/api/ingest", {
      method: "POST",
      body: JSON.stringify(body),
      signal,
    }),

  ingestPath: (
    body: { path: string; recursive: boolean },
    signal?: AbortSignal,
  ) =>
    request<IngestReport[]>("/api/ingest", {
      method: "POST",
      body: JSON.stringify(body),
      signal,
    }),
};
