import { defineStore } from 'pinia';
import { computed, ref } from 'vue';
import type {
  AnalysisSession,
  FlagRecord,
  MetricRecord,
  TimelineItem,
} from '@/types/models';
import { deleteFlag, fetchSession, restoreFlag, setDrillEnd, setPodFrame } from '@/api/client';

export const useSessionStore = defineStore('session', () => {
  const session = ref<AnalysisSession | null>(null);
  const loading = ref(false);
  const error = ref<string | null>(null);

  const itemById = computed(() => {
    const map = new Map<string, TimelineItem>();
    if (!session.value) return map;
    for (const item of session.value.timeline.items) {
      map.set(item.item_id, item);
    }
    return map;
  });

  const metricById = computed(() => {
    const map = new Map<string, MetricRecord>();
    if (!session.value) return map;
    for (const m of session.value.metrics) {
      map.set(m.metric_id, m);
    }
    return map;
  });

  const flagById = computed(() => {
    const map = new Map<string, FlagRecord>();
    if (!session.value) return map;
    for (const f of session.value.flags) {
      map.set(f.flag_id, f);
    }
    return map;
  });

  async function load(jsonPath: string): Promise<void> {
    loading.value = true;
    error.value = null;
    try {
      session.value = await fetchSession(jsonPath);
    } catch (e: unknown) {
      session.value = null;
      error.value = e instanceof Error ? e.message : String(e);
    } finally {
      loading.value = false;
    }
  }

  function clear(): void {
    session.value = null;
    error.value = null;
  }

  /** Current path key for adjustment endpoints (run dir preferred). */
  function sessionPathKey(): string | null {
    return session.value?.run_dir ?? session.value?.session_json_path ?? null;
  }

  const mutating = ref(false);
  // Adjustment failures land here, NOT in `error` — `error` means "no
  // session could be loaded" and blanks the panels; a failed flag/POD
  // mutation leaves the loaded session fully usable.
  const mutationError = ref<string | null>(null);

  async function _mutate(action: () => Promise<AnalysisSession>): Promise<boolean> {
    mutating.value = true;
    mutationError.value = null;
    try {
      session.value = await action();
      return true;
    } catch (e: unknown) {
      mutationError.value = e instanceof Error ? e.message : String(e);
      return false;
    } finally {
      mutating.value = false;
    }
  }

  async function removeFlag(flagId: string): Promise<boolean> {
    const key = sessionPathKey();
    if (!key) return false;
    return _mutate(() => deleteFlag(key, flagId));
  }

  async function unbinFlag(flagId: string): Promise<boolean> {
    const key = sessionPathKey();
    if (!key) return false;
    return _mutate(() => restoreFlag(key, flagId));
  }

  async function adjustPodFrame(frame: number | null): Promise<boolean> {
    const key = sessionPathKey();
    if (!key) return false;
    return _mutate(() => setPodFrame(key, frame));
  }

  async function adjustDrillEnd(frame: number | null): Promise<boolean> {
    const key = sessionPathKey();
    if (!key) return false;
    return _mutate(() => setDrillEnd(key, frame));
  }

  return {
    session,
    loading,
    error,
    mutating,
    mutationError,
    itemById,
    metricById,
    flagById,
    load,
    clear,
    removeFlag,
    unbinFlag,
    adjustPodFrame,
    adjustDrillEnd,
  };
});
