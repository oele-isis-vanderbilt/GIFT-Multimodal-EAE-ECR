<template>
  <div class="left-panel" :class="{ 'pod-adjust-locked': ui.podAdjustActive }">
    <header>
      <div class="title-row">
        <h3>Analysis Viewer</h3>
        <button
          v-if="canOpenDialog"
          class="open-btn"
          :disabled="opening"
          @click="onOpenClick"
        >
          {{ opening ? 'Opening…' : 'Open…' }}
        </button>
      </div>
      <p v-if="session.session" class="muted small">
        {{ session.session.session.video_basename }}
      </p>
      <!-- Browser-only fallback: no native file dialog, so accept a typed path. -->
      <form
        v-if="!canOpenDialog"
        class="browser-load"
        @submit.prevent="onBrowserLoad"
      >
        <input
          v-model="browserPath"
          type="text"
          placeholder="path/to/run_folder"
          class="path-input"
          spellcheck="false"
        />
        <button type="submit" class="open-btn">Load</button>
      </form>
    </header>

    <section v-if="session.loading">
      <p class="muted">Loading session…</p>
    </section>

    <section v-else-if="session.error" class="error">
      <strong>Failed to load session</strong>
      <p>{{ session.error }}</p>
    </section>

    <template v-else-if="session.session">
      <CompareSetupPanel v-if="ui.viewMode === 'compare'" />
      <template v-else>
        <section>
          <h4>Metrics</h4>
          <MetricTogglePanel />
        </section>

        <section>
          <h4>Flags</h4>
          <p v-if="session.mutationError" class="mutation-error">
            ⚠ {{ session.mutationError }}
            <button class="dismiss" @click="session.mutationError = null">✕</button>
          </p>
          <div v-for="(n, i) in notices" :key="i" class="notice">⚠ {{ n }}</div>
          <ul v-if="sortedFlags.length" class="flag-list">
            <li
              v-for="f in sortedFlags"
              :key="f.flag_id"
              class="flag-row"
              :class="{ selected: ui.selectedFlagId === f.flag_id }"
              @click="onFlagClick(f.flag_id, f.frame)"
            >
              <span
                class="flag-swatch"
                :class="{ diamond: f.severity === 'info' }"
                :style="{ background: swatchColor(f) }"
              />
              <span class="severity" :class="f.severity">{{ f.severity }}</span>
              <span class="title">{{ f.title }}</span>
              <template v-if="adjustActionFor(f)">
                <button
                  class="flag-adjust"
                  :class="{ 'drill-kind': f.type === 'drill_end' }"
                  :title="adjustActionFor(f)!.title"
                  :disabled="ui.podAdjustActive || session.mutating"
                  @click.stop="adjustActionFor(f)!.run()"
                >
                  {{ adjustActionFor(f)!.label }}
                </button>
                <button
                  v-if="adjustActionFor(f)!.reset"
                  class="flag-adjust reset"
                  title="Revert to the automatic result"
                  :disabled="ui.podAdjustActive || session.mutating"
                  @click.stop="adjustActionFor(f)!.reset!()"
                >
                  ⟲ Reset
                </button>
              </template>
              <button
                v-if="f.severity !== 'info'"
                class="flag-del"
                title="Delete flag (moves to bin; the metric's score recalculates)"
                :disabled="session.mutating"
                @click.stop="onDeleteFlag(f.flag_id)"
              >
                🗑
              </button>
            </li>
          </ul>
          <p v-else class="muted small">No flags raised.</p>

          <template v-if="session.session.flag_bin && session.session.flag_bin.length">
            <button class="bin-toggle" @click="binOpen = !binOpen">
              {{ binOpen ? '▾' : '▸' }} Deleted flags ({{ session.session.flag_bin.length }})
            </button>
            <ul v-if="binOpen" class="flag-list bin-list">
              <li v-for="f in sortedBin" :key="f.flag_id" class="flag-row binned">
                <span
                  class="flag-swatch"
                  :class="{ diamond: f.severity === 'info' }"
                  :style="{ background: swatchColor(f) }"
                />
                <span class="severity" :class="f.severity">{{ f.severity }}</span>
                <span class="title">{{ f.title }}</span>
                <button
                  class="flag-del restore"
                  title="Restore flag (its metric's score recalculates)"
                  :disabled="session.mutating"
                  @click.stop="onRestoreFlag(f.flag_id)"
                >
                  ↩
                </button>
              </li>
            </ul>
          </template>
        </section>
      </template>
    </template>

    <section v-else>
      <p class="muted">
        No run loaded. Click <strong>Open…</strong> and pick a run folder, or
        launch with
        <code>python -m src.utils.analysis_viewer &lt;run_folder&gt;</code>.
      </p>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';
import { metricColor } from '@/theme/tokens';
import type { FlagRecord } from '@/types/models';
import { useSessionStore } from '@/stores/session';
import { usePlaybackStore } from '@/stores/playback';
import { useUIStore } from '@/stores/ui';
import CompareSetupPanel from './CompareSetupPanel.vue';
import MetricTogglePanel from './MetricTogglePanel.vue';

const session = useSessionStore();
const playback = usePlaybackStore();
const ui = useUIStore();

// The native file picker requires the Tauri runtime. In the browser the
// dialog button is hidden and the typed-path form below takes over instead.
const canOpenDialog =
  typeof window !== 'undefined' &&
  ('__TAURI_INTERNALS__' in window || '__TAURI__' in window);

const opening = ref(false);

// Browser-only path input. Auto-fills with the URL's ?session= or with the
// last successfully-loaded path so the next reload only needs one click.
const RECENT_KEY = 'Vanderbilt-GIFT.AnalysisViewer.recentSession';
const browserPath = ref<string>('');

onMounted(() => {
  const params = new URLSearchParams(window.location.search);
  const fromUrl = params.get('session');
  const fromStorage = (() => {
    try {
      return localStorage.getItem(RECENT_KEY) ?? '';
    } catch {
      return '';
    }
  })();
  browserPath.value = fromUrl || fromStorage;
});

watch(
  () => session.session?.session_json_path,
  (path) => {
    if (!path) return;
    try {
      localStorage.setItem(RECENT_KEY, path);
    } catch {
      // localStorage may be disabled — silent fail
    }
  },
);

async function onBrowserLoad(): Promise<void> {
  const path = browserPath.value.trim();
  if (!path) return;
  await session.load(path);
}

async function onOpenClick(): Promise<void> {
  if (opening.value) return;
  opening.value = true;
  try {
    const { open } = await import('@tauri-apps/plugin-dialog');
    const selection = await open({
      title: 'Open Run Folder',
      multiple: false,
      directory: true,
    });
    if (typeof selection === 'string' && selection) {
      await session.load(selection);
    }
  } catch (err) {
    console.error('[analysis-viewer] folder dialog failed:', err);
  } finally {
    opening.value = false;
  }
}

function onFlagClick(flagId: string, frame: number | null): void {
  ui.selectedFlagId = flagId;
  ui.selectedItemId = null;
  ui.selectedMetricId = null;
  if (frame != null) playback.requestSeek(frame);
}

// --- flag swatches -----------------------------------------------------------

function swatchColor(f: FlagRecord): string {
  if (f.type === 'drill_start' || f.type === 'drill_end') return '#86efac';
  // The POD establishment mark is session-level (metric_id null) but keeps
  // the POD purple so it reads with the timeline diamonds.
  if (f.type === 'pod_establishment') return '#c084fc';
  return metricColor(f.metric_id);
}

// --- Temporal ordering: flags + info marks in frame sequence ----------------

const sortedFlags = computed(() => {
  const flags = session.session?.flags ?? [];
  return [...flags].sort(
    (a, b) => (a.frame ?? Number.POSITIVE_INFINITY) - (b.frame ?? Number.POSITIVE_INFINITY),
  );
});

const sortedBin = computed(() => {
  const flags = session.session?.flag_bin ?? [];
  return [...flags].sort(
    (a, b) => (a.frame ?? Number.POSITIVE_INFINITY) - (b.frame ?? Number.POSITIVE_INFINITY),
  );
});

const notices = computed(() => session.session?.overrides?.notices ?? []);

// --- Per-flag adjustment actions (POD mark + drill end) ----------------------

type AdjustAction = {
  label: string;
  title: string;
  run: () => void;
  reset?: () => void;
};

function podFrameOf(): number | null {
  const m = session.metricById.get('pod_sector_coverage');
  if (!m || m.metric_id !== 'pod_sector_coverage') return null;
  return m.summary.pod_frame;
}

function enterAdjust(frame: number | null, kind: 'pod' | 'drill_end'): void {
  // The adjustment always happens against the main (original) video view.
  playback.mode = 'original';
  const f = Math.max(1, Math.round(frame ?? playback.currentFrame ?? 1));
  ui.enterPodAdjust(f, kind);
  playback.requestSeek(f);
}

function adjustActionFor(f: FlagRecord): AdjustAction | null {
  if (f.type === 'pod_establishment') {
    const established = podFrameOf() != null;
    return {
      label: established ? 'Adjust' : 'Set…',
      title: established
        ? 'Drag the POD mark to a different frame (metrics recompute)'
        : 'Mark the POD frame manually (metrics recompute)',
      run: () => enterAdjust(podFrameOf(), 'pod'),
      reset:
        session.session?.overrides?.pod_frame_override != null
          ? () => void session.adjustPodFrame(null)
          : undefined,
    };
  }
  if (f.type === 'drill_end') {
    return {
      label: 'Adjust',
      title: 'Drag the drill end to a different frame (window-dependent metrics recompute)',
      run: () => enterAdjust(session.session?.drill_window?.end_frame ?? null, 'drill_end'),
      reset:
        session.session?.overrides?.drill_end_override != null
          ? () => void session.adjustDrillEnd(null)
          : undefined,
    };
  }
  return null;
}

// --- Flag bin ---------------------------------------------------------------

const binOpen = ref(false);

async function onDeleteFlag(flagId: string): Promise<void> {
  if (ui.selectedFlagId === flagId) ui.selectedFlagId = null;
  await session.removeFlag(flagId);
}

async function onRestoreFlag(flagId: string): Promise<void> {
  await session.unbinFlag(flagId);
}
</script>

<style scoped>
.left-panel.pod-adjust-locked {
  pointer-events: none;
  opacity: 0.55;
}

.left-panel {
  padding: 14px 14px 24px;
  height: 100%;
  overflow: auto;
}
header {
  margin-bottom: 12px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--color-border);
}
h3 {
  margin: 0;
  font-size: 1.05em;
  font-weight: 600;
}
.title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}
.open-btn {
  background: var(--color-bg-elev);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  padding: 4px 10px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.82em;
}
.open-btn:hover:not(:disabled) {
  background: var(--color-border);
}
.open-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.browser-load {
  display: flex;
  gap: 6px;
  margin-top: 8px;
  align-items: stretch;
}
.path-input {
  flex: 1;
  background: var(--color-bg);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  padding: 4px 8px;
  font-size: 0.82em;
  min-width: 0;
}
.path-input:focus {
  outline: none;
  border-color: var(--color-accent);
}
h4 {
  margin: 16px 0 8px;
  font-size: 0.78em;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--color-muted);
}
.muted {
  color: var(--color-muted);
}
.small {
  font-size: 0.85em;
}
.error {
  background: var(--color-danger-bg);
  border: 1px solid var(--color-border-strong);
  border-radius: var(--radius-md);
  padding: var(--space-sm) 10px;
  color: var(--color-danger-text);
}
.error strong {
  display: block;
  margin-bottom: 4px;
}
.error p {
  margin: 0;
  font-size: 0.85em;
  word-break: break-word;
}
ul {
  list-style: none;
  padding: 0;
  margin: 0;
}
.notice {
  background: rgba(253, 186, 116, 0.12);
  border: 1px solid rgba(253, 186, 116, 0.5);
  border-radius: 5px;
  color: #fdba74;
  font-size: 0.82em;
  padding: 6px 8px;
  margin: 6px 0;
  word-break: break-word;
}
.flag-swatch {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  flex: none;
  border: 1px solid rgba(0, 0, 0, 0.4);
}
.flag-swatch.diamond {
  border-radius: 2px;
  transform: rotate(45deg);
}
.mutation-error {
  display: flex;
  align-items: center;
  gap: 6px;
  background: rgba(239, 68, 68, 0.12);
  border: 1px solid rgba(239, 68, 68, 0.4);
  border-radius: 5px;
  color: #ef4444;
  font-size: 0.82em;
  padding: 6px 8px;
  margin: 6px 0;
  word-break: break-word;
}
.mutation-error .dismiss {
  margin-left: auto;
  background: transparent;
  border: none;
  color: inherit;
  cursor: pointer;
  padding: 0 2px;
}

.flag-adjust {
  margin-left: auto;
  background: transparent;
  border: 1px solid #c084fc;
  color: #c084fc;
  border-radius: 4px;
  font-size: 0.78em;
  font-weight: 600;
  padding: 1px 8px;
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.12s;
  flex: none;
}
.flag-adjust.drill-kind {
  border-color: #86efac;
  color: #86efac;
}
.flag-adjust.reset {
  margin-left: 4px;
  border-color: #fdba74;
  color: #fdba74;
  padding: 1px 8px;
  /* Always visible once an instructor override is active — it is the only
     way back to the automatic result, so it must not hide behind hover. */
  opacity: 1;
}
.flag-adjust.reset:hover:not(:disabled) {
  background: rgba(253, 186, 116, 0.12);
}
.flag-row:hover .flag-adjust,
.flag-row.selected .flag-adjust {
  opacity: 1;
}
.flag-adjust:disabled {
  opacity: 0.35;
  cursor: default;
}
.flag-adjust + .flag-del {
  margin-left: 4px;
}

.flag-del {
  margin-left: auto;
  background: transparent;
  border: none;
  color: var(--color-muted);
  cursor: pointer;
  font-size: 0.9em;
  padding: 0 4px;
  opacity: 0;
  transition: opacity 0.12s;
}
.flag-row:hover .flag-del {
  opacity: 0.8;
}
.flag-del:hover:not(:disabled) {
  opacity: 1;
  color: #ef4444;
}
.flag-del.restore:hover:not(:disabled) {
  color: #22c55e;
}
.flag-del:disabled {
  opacity: 0.3;
  cursor: default;
}
.bin-toggle {
  margin-top: 8px;
  background: transparent;
  border: none;
  color: var(--color-muted);
  cursor: pointer;
  font-size: 0.85em;
  padding: 2px 0;
}
.bin-list .flag-row.binned {
  opacity: 0.6;
  cursor: default;
}
.bin-list .flag-row.binned .flag-del {
  opacity: 0.8;
}

.flag-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 6px;
  margin: 0 -6px;
  font-size: 0.88em;
  border-radius: 4px;
  cursor: pointer;
}
.flag-row:hover {
  background: var(--color-border);
}
.flag-row.selected {
  background: var(--color-accent-bg);
  outline: 1px solid rgba(59, 130, 246, 0.6);
}
.flag-row .severity {
  font-size: 0.72em;
  text-transform: uppercase;
  padding: 1px 6px;
  border-radius: 3px;
  font-weight: 600;
  letter-spacing: 0.04em;
}
.flag-row .severity.warning {
  background: var(--color-warning-bg);
  color: var(--color-warning);
}
.flag-row .title {
  flex: 1;
  word-break: break-word;
}
code {
  background: var(--color-border);
  padding: 1px 5px;
  border-radius: 3px;
  font-size: 0.85em;
}
</style>
