<template>
  <div class="compare-setup">
    <section class="setup-section">
      <h4 class="section-title">Compare setup</h4>
      <div class="kv">
        <div class="row">
          <span class="key">Outputs root</span>
          <div class="value path-row">
            <span class="path-text" :title="compare.outputsRoot ?? ''">
              {{ compare.outputsRoot ?? '(no folder selected)' }}
            </span>
            <button class="link-btn" @click="onChangeOutputsRoot">Change…</button>
          </div>
        </div>
        <div class="row">
          <span class="key">Compare with</span>
          <select
            class="select"
            :value="compare.selectedOtherRunPath ?? ''"
            @change="onSelectOther(($event.target as HTMLSelectElement).value)"
          >
            <option value="">— pick a run —</option>
            <option
              v-for="r in otherRunsSorted"
              :key="r.path"
              :value="r.path"
            >
              {{ r.title }}{{ r.role === 'expert' ? '  [EXPERT]' : '' }}
            </option>
          </select>
        </div>
      </div>
    </section>

    <section class="setup-section">
      <h4 class="section-title">Metrics</h4>
      <p v-if="!compare.selectedOtherRunPath" class="muted small">
        Pick a run above to compare against.
      </p>
      <p v-else-if="!metricsToShow.length" class="muted small">
        No metrics available for this session.
      </p>
      <ul v-else class="metric-list">
        <li
          v-for="m in metricsToShow"
          :key="m.metric_id_upper"
          :class="{
            selected: compare.selectedMetricId === m.metric_id_upper,
            loading:
              compare.loading && compare.selectedMetricId === m.metric_id_upper,
          }"
          @click="onSelectMetric(m.metric_id_upper)"
        >
          <div class="metric-name">{{ m.label }}</div>
          <div
            v-if="compare.selectedMetricId === m.metric_id_upper && compare.result"
            class="metric-scores"
          >
            <div class="side">
              <span class="side-label">{{ currentSideLabel }}</span>
              <span class="score-num">{{ formatScore(compare.result.current.score) }}</span>
              <span
                v-if="relativeBand"
                class="band"
                :class="relativeBand"
                :title="BAND_TOOLTIP[relativeBand]"
              >
                {{ BAND_LABEL[relativeBand] }}
              </span>
            </div>
            <div class="side">
              <span class="side-label">{{ otherSideLabel }}</span>
              <span class="score-num">{{ formatScore(compare.result.other.score) }}</span>
            </div>
          </div>
          <div
            v-else-if="
              compare.selectedMetricId === m.metric_id_upper && compare.loading
            "
            class="muted small"
          >
            Computing…
          </div>
        </li>
      </ul>
    </section>

    <section class="setup-section legend-section">
      <h4 class="section-title">Legend</h4>
      <ul class="legend">
        <li v-for="b in LEGEND_BANDS" :key="b.id">
          <span class="band" :class="b.id">{{ b.label }}</span>
          <span class="legend-criteria">{{ b.criteria }}</span>
        </li>
      </ul>
      <p class="muted small legend-note">
        <strong>Reference</strong> = the run you picked above to compare
        against<template v-if="selectedOtherTitle">
          (currently <em>{{ selectedOtherTitle }}</em>)</template>.
        Grading is relative to that run — not an absolute threshold — because
        what counts as "good" varies by room.
      </p>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, watch } from 'vue';

import { useCompareStore } from '@/stores/compare';
import { useSessionStore } from '@/stores/session';
import { formatScore, relativeAssessment, type RelativeBand } from '@/utils/formatters';

const compare = useCompareStore();
const session = useSessionStore();

onMounted(async () => {
  compare.initFromEnv();
  if (compare.outputsRoot && !compare.runs.length) {
    await compare.refreshRuns();
  }
});

watch(
  () => session.session?.run_dir,
  async (runDir) => {
    if (!runDir) return;
    compare.initFromEnv();
    if (compare.outputsRoot) await compare.refreshRuns();
  },
);

// Auto-fetch the comparison whenever both "compare with" and "metric" are set.
watch(
  () => [compare.selectedOtherRunPath, compare.selectedMetricId] as const,
  ([o, m]) => {
    if (o && m) void compare.fetchComparisonResult();
  },
);

const otherRunsSorted = computed(() => {
  const out = compare.runs.filter((r) => r.path !== compare.currentRunPath);
  out.sort((a, b) => {
    if (a.role !== b.role) return a.role === 'expert' ? -1 : 1;
    return (b.created_at ?? '').localeCompare(a.created_at ?? '');
  });
  return out;
});

const selectedOtherTitle = computed(() => {
  const path = compare.selectedOtherRunPath;
  if (!path) return null;
  const match = compare.runs.find((r) => r.path === path);
  return match?.title ?? null;
});

// Metric IDs in the analysis JSON are lower-snake; the engine's compare
// dispatch table uses upper-snake. Hard-coded mapping for the canonical
// viewer-visible metrics; unknown ids round-trip via upper-snake.
const ENGINE_ID_MAP: Record<string, string> = {
  entrance_vectors: 'ENTRANCE_VECTORS',
  entrance_hesitation: 'ENTRANCE_HESITATION',
  total_time_of_entry: 'TOTAL_TIME_OF_ENTRY',
  move_along_wall: 'STAY_ALONG_WALL',
};

function toEngineMetricId(id: string): string {
  return ENGINE_ID_MAP[id] ?? id.toUpperCase();
}

// POD metrics have no expertCompare implementation yet — hide them from the
// compare picker rather than surfacing a backend error.
const COMPARE_EXCLUDED = new Set(['pod_sector_coverage', 'pod_mutual_facing']);

const metricsToShow = computed(() => {
  return (session.session?.metrics ?? [])
    .filter((m) => !COMPARE_EXCLUDED.has(m.metric_id))
    .map((m) => ({
      metric_id: m.metric_id,
      metric_id_upper: toEngineMetricId(m.metric_id),
      label: m.label,
    }));
});

const currentSideLabel = computed(() => {
  const role = compare.result?.current.role;
  return role === 'expert' ? 'You (expert)' : 'You';
});

const otherSideLabel = computed(() => {
  const role = compare.result?.other.role;
  return role === 'expert' ? 'Expert' : 'Reference';
});

const relativeBand = computed<RelativeBand | null>(() =>
  relativeAssessment(compare.result?.current.score, compare.result?.other.score),
);

const BAND_LABEL: Record<RelativeBand, string> = {
  above: 'above',
  at: 'at',
  near: 'near',
  below: 'below',
};

const BAND_TOOLTIP: Record<RelativeBand, string> = {
  above: 'At or above the reference (≥100%)',
  at: 'Within 10% of the reference (90–100%)',
  near: 'Approaching the reference (70–90%)',
  below: 'Below the reference (<70%)',
};

const LEGEND_BANDS: { id: RelativeBand; label: string; criteria: string }[] = [
  { id: 'above', label: 'above', criteria: '≥ 100% of reference' },
  { id: 'at', label: 'at', criteria: '90 – 100% of reference' },
  { id: 'near', label: 'near', criteria: '70 – 90% of reference' },
  { id: 'below', label: 'below', criteria: '< 70% of reference' },
];

async function onChangeOutputsRoot(): Promise<void> {
  let next: string | null = null;
  try {
    const dialog = await import('@tauri-apps/plugin-dialog');
    const picked = await dialog.open({ directory: true, multiple: false });
    next = typeof picked === 'string' ? picked : null;
  } catch {
    next = window.prompt('Outputs root (absolute path):', compare.outputsRoot ?? '') || null;
  }
  if (next) await compare.setOutputsRoot(next);
}

function onSelectOther(path: string): void {
  compare.selectOther(path || null);
}

function onSelectMetric(metricIdUpper: string): void {
  compare.selectMetric(metricIdUpper);
}
</script>

<style scoped>
.compare-setup {
  display: flex;
  flex-direction: column;
  gap: 18px;
  min-height: 100%;
}

.setup-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.section-title {
  margin: 0;
  font-size: 0.78em;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--color-muted);
}

.kv {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.kv .row {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.kv .key {
  color: var(--color-muted);
  font-size: 0.78em;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.kv .value {
  font-size: 0.9em;
  word-break: break-word;
}
.path-row {
  display: flex;
  align-items: center;
  gap: 6px;
}
.path-text {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.85em;
}
.link-btn {
  background: var(--color-bg-elev);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  padding: 3px 10px;
  border-radius: 3px;
  font-size: 0.82em;
  cursor: pointer;
  flex-shrink: 0;
}
.link-btn:hover {
  background: var(--color-border);
}

.select {
  width: 100%;
  background: var(--color-bg);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  padding: 4px 8px;
  font-size: 0.9em;
}

.metric-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.metric-list li {
  border: 1px solid var(--color-border);
  border-radius: 4px;
  padding: 8px 10px;
  background: var(--color-bg-elev);
  cursor: pointer;
  transition: background-color 80ms ease, border-color 80ms ease;
}
.metric-list li:hover:not(.selected) {
  background: var(--color-border);
}
.metric-list li.selected {
  border-color: var(--color-accent);
  background: var(--color-accent-bg);
}
.metric-list li.loading {
  opacity: 0.7;
}
.metric-name {
  font-size: 0.92em;
  font-weight: 500;
}
.metric-scores {
  margin-top: 6px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 0.85em;
}
.side {
  display: flex;
  align-items: center;
  gap: 8px;
}
.side-label {
  color: var(--color-muted);
  width: 70px;
  font-size: 0.82em;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.score-num {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  min-width: 38px;
}
.band {
  font-size: 0.74em;
  font-weight: 600;
  text-transform: uppercase;
  padding: 1px 6px;
  border-radius: 3px;
  background: var(--color-border);
  letter-spacing: 0.04em;
}
.band.above {
  background: var(--color-band-above-bg);
  color: var(--color-band-above);
}
.band.at {
  background: var(--color-success-bg);
  color: var(--color-success);
}
.band.near {
  background: var(--color-warning-bg);
  color: var(--color-warning);
}
.band.below {
  background: var(--color-danger-bg);
  color: var(--color-danger-text);
}

.legend-section {
  margin-top: auto;
  padding-top: 12px;
  border-top: 1px solid var(--color-border);
}
.legend {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.legend li {
  display: flex;
  align-items: center;
  gap: 8px;
}
.legend .band {
  flex-shrink: 0;
  width: 56px;
  text-align: center;
}
.legend-criteria {
  color: var(--color-muted);
  font-size: 0.82em;
}
.legend-note {
  margin: 8px 0 0;
  line-height: 1.4;
}

.muted {
  color: var(--color-muted);
}
.small {
  font-size: 0.85em;
}
</style>
