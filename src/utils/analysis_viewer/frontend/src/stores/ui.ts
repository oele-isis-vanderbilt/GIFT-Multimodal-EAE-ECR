import { defineStore } from 'pinia';
import { ref, watch } from 'vue';

const STORAGE_KEY = 'Vanderbilt-GIFT.AnalysisViewer';

type PersistedUI = {
  splitterLeft: number;
  splitterRight: number;
  rightCollapsed: boolean;
  metricVisibility: Record<string, boolean>;
  viewMode: 'analysis' | 'compare';
};

function loadPersisted(): Partial<PersistedUI> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

function persist(state: PersistedUI): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // localStorage may be full or disabled — silent fail
  }
}

export const useUIStore = defineStore('ui', () => {
  const persisted = loadPersisted();

  const splitterLeft = ref<number>(persisted.splitterLeft ?? 280);
  const splitterRight = ref<number>(persisted.splitterRight ?? 360);
  const rightCollapsed = ref<boolean>(persisted.rightCollapsed ?? false);
  const metricVisibility = ref<Record<string, boolean>>(persisted.metricVisibility ?? {});

  // Selection state — not persisted; resets on reload.
  const selectedItemId = ref<string | null>(null);
  const selectedMetricId = ref<string | null>(null);
  const selectedFlagId = ref<string | null>(null);

  // Top-level right-pane mode: 'analysis' shows the existing Summary/Metrics/Raw
  // tabs; 'compare' shows the Compare view (per-run side-by-side comparison).
  // Persisted so users who live in Compare mode don't re-flip the toggle each
  // launch. Defaults to 'analysis' when nothing is stored.
  const viewMode = ref<'analysis' | 'compare'>(persisted.viewMode ?? 'analysis');

  // --- POD-frame adjustment mode (not persisted) ---------------------------
  // While active: the video stage is forced to the main (original) view, the
  // POD marker on the timeline is draggable (seeking the video live), every
  // other interaction is locked, and Confirm / Cancel are the only exits.
  const podAdjustActive = ref(false);
  const podAdjustFrame = ref<number | null>(null);
  /** What the drag adjusts: the POD mark or the drill end. */
  const podAdjustKind = ref<'pod' | 'drill_end'>('pod');

  function enterPodAdjust(initialFrame: number, kind: 'pod' | 'drill_end' = 'pod'): void {
    podAdjustActive.value = true;
    podAdjustFrame.value = initialFrame;
    podAdjustKind.value = kind;
  }

  function exitPodAdjust(): void {
    podAdjustActive.value = false;
    podAdjustFrame.value = null;
  }

  function setMetricVisibility(metricId: string, visible: boolean): void {
    metricVisibility.value = { ...metricVisibility.value, [metricId]: visible };
  }

  function isMetricVisible(metricId: string): boolean {
    // Default to visible when no preference recorded yet.
    return metricVisibility.value[metricId] !== false;
  }

  function toggleRightCollapsed(): void {
    rightCollapsed.value = !rightCollapsed.value;
  }

  watch(
    [splitterLeft, splitterRight, rightCollapsed, metricVisibility, viewMode],
    () => {
      persist({
        splitterLeft: splitterLeft.value,
        splitterRight: splitterRight.value,
        rightCollapsed: rightCollapsed.value,
        metricVisibility: { ...metricVisibility.value },
        viewMode: viewMode.value,
      });
    },
    { deep: true },
  );

  return {
    splitterLeft,
    splitterRight,
    rightCollapsed,
    metricVisibility,
    selectedItemId,
    selectedMetricId,
    selectedFlagId,
    viewMode,
    podAdjustActive,
    podAdjustFrame,
    podAdjustKind,
    enterPodAdjust,
    exitPodAdjust,
    setMetricVisibility,
    isMetricVisible,
    toggleRightCollapsed,
  };
});
