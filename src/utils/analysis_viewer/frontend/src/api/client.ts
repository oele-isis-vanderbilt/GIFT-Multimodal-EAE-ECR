// Typed fetch helpers for the FastAPI sidecar.
//
// Base-URL resolution priority:
//   1. Tauri runtime  → invoke('get_sidecar_port') (the Rust shell knows the
//      port it bound the Python sidecar to) → http://127.0.0.1:<port>
//   2. window.__SIDECAR_PORT__ (best-effort eval injection from Tauri)
//   3. import.meta.env.VITE_SIDECAR_PORT (Vite dev override)
//   4. SAME-ORIGIN (relative URLs) — the LAN web-server mode, where the FastAPI
//      backend serves this built frontend itself. A remote browser must talk to
//      the host it loaded the page from, never 127.0.0.1.

import type { AnalysisSession, CompareResult, RunSummary } from '@/types/models';

let cachedPort: number | null = null;
let portResolved = false;

function isTauri(): boolean {
  // Tauri 2 exposes both `__TAURI_INTERNALS__` and `__TAURI__`. Either is fine.
  return (
    typeof window !== 'undefined' &&
    ('__TAURI_INTERNALS__' in window || '__TAURI__' in window)
  );
}

/**
 * Resolves the sidecar port asynchronously and caches it. Call once at app
 * startup before any other API helper, since the rest are sync. Returns null
 * when no explicit port source exists — that's the same-origin web mode.
 */
export async function initSidecarPort(): Promise<number | null> {
  if (portResolved) return cachedPort;
  portResolved = true;

  if (isTauri()) {
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      const port = await invoke<number | null>('get_sidecar_port');
      if (typeof port === 'number' && port > 0) cachedPort = port;
    } catch {
      // fall through to other resolvers
    }
  }

  if (cachedPort == null && typeof window !== 'undefined' && typeof window.__SIDECAR_PORT__ === 'number') {
    cachedPort = window.__SIDECAR_PORT__;
  }

  if (cachedPort == null) {
    const env = import.meta.env.VITE_SIDECAR_PORT;
    if (env) cachedPort = Number(env);
  }

  return cachedPort;
}

export function sidecarBaseUrl(): string {
  // All backend endpoints live under /api/* (the bare /analysis and /compare
  // paths serve the SPA shell, so the data API is namespaced out of their way).
  // Desktop / dev: an explicit port means the backend is on localhost.
  if (cachedPort != null && cachedPort > 0) return `http://127.0.0.1:${cachedPort}/api`;
  // Web mode: backend serves this page — same origin via a root-relative /api.
  return '/api';
}

async function getJson<T>(path: string): Promise<T> {
  const url = `${sidecarBaseUrl()}${path}`;
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText} — ${body || url}`);
  }
  return (await res.json()) as T;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const url = `${sidecarBaseUrl()}${path}`;
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const bodyText = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText} — ${bodyText || url}`);
  }
  return (await res.json()) as T;
}

export function fetchSession(jsonPath: string): Promise<AnalysisSession> {
  return getJson<AnalysisSession>(`/session?path=${encodeURIComponent(jsonPath)}`);
}

/** Move a flag into the instructor bin. Returns the re-merged session. */
export function deleteFlag(sessionPath: string, flagId: string): Promise<AnalysisSession> {
  return postJson<AnalysisSession>('/flags/delete', { path: sessionPath, flag_id: flagId });
}

/** Restore a binned flag. Returns the re-merged session. */
export function restoreFlag(sessionPath: string, flagId: string): Promise<AnalysisSession> {
  return postJson<AnalysisSession>('/flags/restore', { path: sessionPath, flag_id: flagId });
}

/**
 * Set (frame) or reset (null) the instructor POD-establishment frame.
 * The backend recomputes members/sectors/scores/flags from run caches and
 * returns the re-merged session.
 */
export function setPodFrame(sessionPath: string, frame: number | null): Promise<AnalysisSession> {
  return postJson<AnalysisSession>('/pod/frame', { path: sessionPath, frame });
}

/**
 * Set (frame) or reset (null) the instructor drill end. The backend re-runs
 * the window-dependent metrics (wall flags included) and POD detection from
 * run caches and returns the re-merged session.
 */
export function setDrillEnd(sessionPath: string, frame: number | null): Promise<AnalysisSession> {
  return postJson<AnalysisSession>('/drill/end', { path: sessionPath, frame });
}

export function fetchHealth(): Promise<{ ok: boolean; service: string }> {
  return getJson(`/health`);
}

export interface ServerConfig {
  outputs_root: string | null;
  serve_mode: boolean;
}

/** Runtime server config. In web mode, `outputs_root` is the folder Compare can browse. */
export function fetchConfig(): Promise<ServerConfig> {
  return getJson<ServerConfig>(`/config`);
}

export function videoUrl(videoPath: string, sessionPath: string): string {
  const params = new URLSearchParams({ path: videoPath, session: sessionPath });
  return `${sidecarBaseUrl()}/video?${params.toString()}`;
}

export function imageUrl(imagePath: string, sessionPath: string): string {
  const params = new URLSearchParams({ path: imagePath, session: sessionPath });
  return `${sidecarBaseUrl()}/image?${params.toString()}`;
}

export function fetchRuns(outputsRoot: string): Promise<RunSummary[]> {
  return getJson<RunSummary[]>(`/runs?outputs_root=${encodeURIComponent(outputsRoot)}`);
}

export function fetchComparison(
  currentRun: string,
  otherRun: string,
  metricId: string,
  sessionPath: string,
): Promise<CompareResult> {
  const params = new URLSearchParams({
    current_run: currentRun,
    other_run: otherRun,
    metric_id: metricId,
    session: sessionPath,
  });
  return getJson<CompareResult>(`/compare?${params.toString()}`);
}
