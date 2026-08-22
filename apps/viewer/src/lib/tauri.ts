/**
 * Bridge to the desktop shell.
 *
 * Everything here degrades to a browser fallback so the UI can be developed
 * against a plain `nexum serve` without launching the whole desktop app. The
 * fallback is not a stub of the data — it is the real API at its default port,
 * so what you see in the browser is what the window shows.
 */

import type { ApiInfo } from "../types";

/** Whether the app is running inside the Tauri shell. */
export function isDesktop(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

async function invoke<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
  return tauriInvoke<T>(command, args);
}

/** The database the shell currently has open, if any. */
export async function currentDatabase(): Promise<ApiInfo | null> {
  if (!isDesktop()) {
    // In the browser, assume a `nexum serve` on the default port and let the
    // first API call report it if nothing is listening.
    return null;
  }
  return invoke<ApiInfo | null>("current_database");
}

export async function openDatabase(path: string, create = false): Promise<ApiInfo> {
  return invoke<ApiInfo>("open_database", { path, create });
}

export async function closeDatabase(): Promise<void> {
  if (!isDesktop()) return;
  await invoke<void>("close_database");
}

export async function recentDatabases(): Promise<string[]> {
  if (!isDesktop()) return [];
  return invoke<string[]>("recent_databases");
}

export async function rememberDatabase(path: string): Promise<string[]> {
  if (!isDesktop()) return [];
  return invoke<string[]>("remember_database", { path });
}

/** Native folder picker. Returns null when the user cancels. */
export async function pickFolder(title: string): Promise<string | null> {
  if (!isDesktop()) {
    const typed = window.prompt(`${title}\n\nPath to the database directory:`);
    return typed?.trim() || null;
  }
  const { open } = await import("@tauri-apps/plugin-dialog");
  const selected = await open({ directory: true, multiple: false, title });
  return typeof selected === "string" ? selected : null;
}
