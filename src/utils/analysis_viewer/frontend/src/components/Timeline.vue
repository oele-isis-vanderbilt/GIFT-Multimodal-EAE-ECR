<template>
  <div ref="rootEl" class="timeline">
    <div class="timeline-body">
      <!-- Fixed-width label gutter — sits beside the timeline canvas, not on
           top of it. Vertical scroll is mirrored from the canvas container
           so labels always line up with their metric rows; horizontal scroll
           leaves the gutter untouched. -->
      <div class="label-column">
        <div ref="labelRailEl" class="label-rail">
          <div
            v-for="m in metricLabels"
            :key="m.metricId"
            class="metric-label"
            :style="{ top: `${m.startY + m.height / 2}px` }"
          >
            <span class="swatch" :style="{ background: m.color }"></span>
            <span class="name">{{ m.name }}</span>
          </div>
        </div>
      </div>
      <div ref="canvasContainer" class="canvas-container always-scroll"></div>
    </div>
    <span
      v-if="ripple"
      :key="ripple.key"
      class="tap-ripple"
      :style="{ left: `${ripple.x}px`, top: `${ripple.y}px` }"
    ></span>
    <div
      v-if="tooltip"
      class="tooltip"
      :class="{ below: tooltip.below }"
      :style="{ left: `${tooltip.x}px`, top: `${tooltip.y}px` }"
    >
      {{ tooltip.label }}
    </div>
    <div class="bar">
      <div class="zoom-controls">
        <button class="zoom-btn" title="Jump to first artifact" @click="onJumpToFirst">
          <span aria-hidden="true">⇤</span>
        </button>
        <button class="zoom-btn" title="Zoom out" @click="onZoomOut">−</button>
        <button class="zoom-btn" title="Fit to range" @click="onFit">Fit</button>
        <button class="zoom-btn" title="Zoom in" @click="onZoomIn">+</button>
      </div>
      <template v-if="ui.podAdjustActive">
        <span class="hint pod-adjust-hint" :class="{ 'drill-kind': ui.podAdjustKind === 'drill_end' }">
          {{ ui.podAdjustKind === 'drill_end' ? 'Adjusting drill end' : 'Adjusting POD' }}
          · drag the marker or tap the track · frame {{ ui.podAdjustFrame ?? '—' }}
        </span>
        <button
          class="zoom-btn pod-confirm"
          :disabled="session.mutating || ui.podAdjustFrame == null"
          @click="onPodConfirm"
        >
          {{ session.mutating ? 'Saving…' : 'Confirm' }}
        </button>
        <button class="zoom-btn pod-cancel" :disabled="session.mutating" @click="onPodCancel">
          Cancel
        </button>
      </template>
      <span v-else class="hint muted">Swipe, drag, or scroll to navigate · tap an empty track to seek</span>
      <span class="range-readout muted">{{ rangeLabel }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { useSessionStore } from '@/stores/session';
import { usePlaybackStore } from '@/stores/playback';
import { useUIStore } from '@/stores/ui';
import type { TimelineItem } from '@/types/models';
import {
  TimelineScene,
  type HoverPayload,
  type MetricLabelInfo,
} from './TimelineScene';

const session = useSessionStore();
const playback = usePlaybackStore();
const ui = useUIStore();

const rootEl = ref<HTMLDivElement | null>(null);
const canvasContainer = ref<HTMLDivElement | null>(null);
const labelRailEl = ref<HTMLDivElement | null>(null);
let scene: TimelineScene | null = null;

const tooltip = ref<{ x: number; y: number; label: string; below: boolean } | null>(null);
const metricLabels = ref<MetricLabelInfo[]>([]);

// Subtle tap confirmation: a small ring blooms at the press point and fades.
// It's transient + pointer-events:none, so it never obstructs other events.
const ripple = ref<{ x: number; y: number; key: number } | null>(null);
let rippleSeq = 0;
let rippleTimer: number | null = null;
function showRipple(clientX: number, clientY: number): void {
  if (!rootEl.value) return;
  const rect = rootEl.value.getBoundingClientRect();
  ripple.value = { x: clientX - rect.left, y: clientY - rect.top, key: ++rippleSeq };
  if (rippleTimer != null) clearTimeout(rippleTimer);
  rippleTimer = window.setTimeout(() => {
    ripple.value = null;
  }, 450);
}

// Drag-to-pan for touch/pen. Desktop mouse keeps its existing wheel/scrollbar
// scrolling untouched (we ignore mouse here). A tap (no drag) shows the ripple;
// Pixi separately handles the seek/selection for that tap.
// Matches the timeline's tap tolerance: a finger must move more than this
// before it pans, so small jitter during a tap still seeks instead of panning.
const PAN_TOLERANCE = 5;
let panId: number | null = null;
let panStartX = 0;
let panStartY = 0;
let panScrollL = 0;
let panScrollT = 0;
let panMoved = false;
let panCanScroll = false;

function onTlPointerDown(ev: PointerEvent): void {
  // POD-adjust mode: the marker drag owns the pointer — suppress
  // drag-to-pan so touch/pen dragging doesn't pan and drag simultaneously.
  // (Wheel / scrollbar navigation stays available.)
  if (ui.podAdjustActive) return;
  if (ev.pointerType === 'mouse' && ev.button !== 0) return;
  panId = ev.pointerId;
  panStartX = ev.clientX;
  panStartY = ev.clientY;
  panMoved = false;
  panCanScroll = ev.pointerType !== 'mouse';
  const c = canvasContainer.value;
  if (panCanScroll && c) {
    panScrollL = c.scrollLeft;
    panScrollT = c.scrollTop;
  }
}
function onTlPointerMove(ev: PointerEvent): void {
  if (ev.pointerId !== panId) return;
  const dx = ev.clientX - panStartX;
  const dy = ev.clientY - panStartY;
  if (!panMoved && Math.hypot(dx, dy) < PAN_TOLERANCE) return;
  panMoved = true;
  const c = canvasContainer.value;
  if (panCanScroll && c) {
    c.scrollLeft = panScrollL - dx;
    c.scrollTop = panScrollT - dy;
  }
}
function onTlPointerUp(ev: PointerEvent): void {
  if (ev.pointerId !== panId) return;
  panId = null;
  if (!panMoved) showRipple(ev.clientX, ev.clientY);
}

// Visible range follows the active video mode.
const visibleRange = computed(() => {
  if (!session.session) return { start: 0, end: 1 };
  const total = session.session.video.total_frames;
  if (playback.mode === 'original') return { start: 0, end: total };
  const dw = session.session.drill_window;
  if (!dw) return { start: 0, end: total };
  return { start: dw.start_frame, end: dw.end_frame };
});

const rangeLabel = computed(() => {
  const { start, end } = visibleRange.value;
  return `frames ${start}–${end}`;
});

onMounted(async () => {
  if (!canvasContainer.value) return;
  scene = new TimelineScene();
  await scene.init(canvasContainer.value, {
    onItemClicked: (itemId, seekFrame) => {
      ui.selectedItemId = itemId;
      ui.selectedFlagId = null;
      ui.selectedMetricId = null;
      playback.requestSeek(seekFrame);
    },
    onFlagClicked: (flagId) => {
      ui.selectedFlagId = flagId;
      ui.selectedItemId = null;
      ui.selectedMetricId = null;
      const flag = session.flagById.get(flagId);
      if (flag?.frame != null) playback.requestSeek(flag.frame);
    },
    onEmptyClicked: (frame) => {
      ui.selectedItemId = null;
      ui.selectedFlagId = null;
      playback.requestSeek(frame);
    },
    onHover: (payload) => updateTooltip(payload),
    onLayoutChanged: (labels) => {
      metricLabels.value = labels;
    },
    onPodDragPreview: (frame) => {
      ui.podAdjustFrame = frame;
      throttledSeek(frame);
    },
  });

  // The label gutter is a sibling of the canvas container, not a child.
  // Mirror the canvas's vertical scroll into the rail so labels always line
  // up with their metric rows. Horizontal scroll is independent: the gutter
  // never moves sideways, so labels can't overlap canvas content.
  canvasContainer.value.addEventListener('scroll', onCanvasScroll, { passive: true });

  // Drag-to-pan (touch/pen) + tap-ripple, on the scroll container.
  canvasContainer.value.addEventListener('pointerdown', onTlPointerDown);
  canvasContainer.value.addEventListener('pointermove', onTlPointerMove);
  canvasContainer.value.addEventListener('pointerup', onTlPointerUp);
  canvasContainer.value.addEventListener('pointercancel', onTlPointerUp);

  pushData();
  scene.setRange(visibleRange.value.start, visibleRange.value.end);
  scene.setMetricVisibility(ui.metricVisibility);
  scene.setSelectedMetric(ui.selectedMetricId);
  scene.setSelectedFlag(ui.selectedFlagId);
  scene.setSelectedItem(ui.selectedItemId);
  scene.setCurrentFrame(playback.currentFrame);
});

function onCanvasScroll(): void {
  if (!canvasContainer.value || !labelRailEl.value) return;
  labelRailEl.value.style.transform = `translateY(${-canvasContainer.value.scrollTop}px)`;
}

function pushData(): void {
  if (!scene) return;
  if (!session.session) {
    scene.setData({ items: [], flags: [], drillWindow: null, totalFrames: 1 });
    return;
  }
  const dw = session.session.drill_window;
  scene.setData({
    items: session.session.timeline.items,
    flags: session.session.flags,
    drillWindow: dw
      ? {
          start_frame: dw.start_frame,
          end_frame: dw.end_frame,
          end_uncertain: dw.end_uncertain,
        }
      : null,
    totalFrames: session.session.video.total_frames,
    metricIds: session.session.metrics.map((m) => m.metric_id),
  });
}

// Threshold (px from top of timeline div) below which the tooltip flips
// below the cursor instead of above it. Picked to comfortably clear the
// flag-row + tooltip height (~30 px).
const TOOLTIP_FLIP_THRESHOLD = 56;

function updateTooltip(payload: HoverPayload): void {
  if (!payload || !rootEl.value) {
    tooltip.value = null;
    return;
  }
  const label =
    payload.kind === 'item'
      ? itemTooltipLabel(session.itemById.get(payload.id))
      : payload.kind === 'flag'
        ? flagTooltipLabel(payload.id)
        : drillTooltipLabel(payload.id);
  if (!label) {
    tooltip.value = null;
    return;
  }
  // Position relative to the timeline root so the tooltip follows the cursor
  // regardless of the canvas-container's scroll offset. Auto-flip below the
  // cursor when hovering near the top so the tooltip never escapes upward.
  const rect = rootEl.value.getBoundingClientRect();
  const x = payload.clientX - rect.left;
  const y = payload.clientY - rect.top;
  tooltip.value = {
    x,
    y,
    label,
    below: y < TOOLTIP_FLIP_THRESHOLD,
  };
}

function itemTooltipLabel(item: TimelineItem | undefined): string | null {
  if (!item) return null;
  if (item.kind === 'entry') {
    return `${item.label} · track ${item.data.track_id} · ${item.time_sec.toFixed(2)}s`;
  }
  if (item.kind === 'vector') {
    return `${item.label} · ${item.data.direction_label} · ${item.time_sec.toFixed(2)}s`;
  }
  if (item.kind === 'pair_gap') {
    const tag = item.data.violates_time_limit ? ' · ⚠ violation' : '';
    return `${item.label} · gap ${item.data.gap_sec.toFixed(2)}s / allowed ${item.data.allowed_gap_sec.toFixed(2)}s${tag}`;
  }
  if (item.kind === 'duration') {
    const allowed = item.data.derived_allowed_duration_sec;
    const allowedStr = typeof allowed === 'number' ? allowed.toFixed(2) : '—';
    const tag = item.data.violates_total_entry_limit ? ' · ⚠ violation' : '';
    return `${item.label} · ${item.data.duration_sec.toFixed(2)}s / allowed ${allowedStr}s${tag}`;
  }
  if (item.kind === 'wall_excursion') {
    const human = item.data.label_kind === 'too_close' ? 'too close' : 'too far';
    return `${item.label} · ${human} for ${item.data.duration_sec.toFixed(2)}s · ${item.time_sec.toFixed(2)}s`;
  }
  if (item.kind === 'pod_establishment') {
    const src = item.data.source === 'instructor' ? 'instructor-set' : 'auto-detected';
    const t = item.time_sec != null ? ` · ${item.time_sec.toFixed(2)}s` : '';
    return `POD established · frame ${item.frame}${t} · ${src}`;
  }
  return null;
}

function flagTooltipLabel(flagId: string): string | null {
  const flag = session.flagById.get(flagId);
  if (!flag) return null;
  return `${flag.title} (${flag.severity})`;
}

function drillTooltipLabel(id: string): string | null {
  const dw = session.session?.drill_window;
  if (!dw) return null;
  if (id === 'drill_start') {
    return `Drill start · frame ${dw.start_frame} · ${dw.start_time_sec.toFixed(2)}s`;
  }
  if (id === 'drill_end') {
    const tag = dw.end_uncertain ? ' (uncertain)' : '';
    return `Drill end · frame ${dw.end_frame} · ${dw.end_time_sec.toFixed(2)}s${tag}`;
  }
  return null;
}

watch(() => session.session, pushData);
watch(visibleRange, (range) => {
  scene?.setRange(range.start, range.end);
});
watch(
  () => ({ ...ui.metricVisibility }),
  (vis) => scene?.setMetricVisibility(vis),
  { deep: true },
);
watch(
  () => ui.selectedMetricId,
  (id) => {
    scene?.setSelectedMetric(id);
    if (id) scene?.scrollToMetric(id);
  },
);
watch(
  () => ui.selectedFlagId,
  (id) => {
    const flag = id ? session.flagById.get(id) : null;
    const isInfo = flag?.severity === 'info';
    if (id && flag && !isInfo) {
      // Force the flag's owning metric visible — otherwise the flag (and the
      // row that hosts it) isn't in the layout, and the auto-scroll below
      // would land on nothing. Matches user intent: picking a flag is an
      // implicit "show me this". Info flags (POD establishment, drill
      // start/end) deliberately do NOT toggle metric rows on.
      if (flag.metric_id && !ui.isMetricVisible(flag.metric_id)) {
        ui.setMetricVisibility(flag.metric_id, true);
        // The visibility watcher applies pre-flush on the NEXT tick; scrolling
        // now would measure the stale layout (row not there yet) and no-op.
        scene?.setSelectedFlag(id);
        void nextTick(() => {
          if (ui.selectedFlagId === id) scene?.scrollToFlag(id);
        });
        return;
      }
    }
    scene?.setSelectedFlag(id);
    if (id && !isInfo) scene?.scrollToFlag(id);
  },
);
watch(
  () => ui.selectedItemId,
  (id) => scene?.setSelectedItem(id),
);
watch(
  () => playback.currentFrame,
  (f) => scene?.setCurrentFrame(f),
);

// --- POD-frame adjustment ---------------------------------------------------

// Seeking the <video> on every pointermove overwhelms it. Leading+trailing
// throttle: the first drag update seeks immediately (responsive), and while
// dragging the latest position is re-issued every 80 ms so the main video
// scrubs smoothly with the marker.
const SEEK_THROTTLE_MS = 80;
let seekTimer: number | null = null;
let pendingSeekFrame: number | null = null;
let lastSeekIssued: number | null = null;

function issuePendingSeek(): void {
  seekTimer = null;
  if (pendingSeekFrame != null && pendingSeekFrame !== lastSeekIssued) {
    lastSeekIssued = pendingSeekFrame;
    playback.requestSeek(pendingSeekFrame);
    pendingSeekFrame = null;
    seekTimer = window.setTimeout(issuePendingSeek, SEEK_THROTTLE_MS);
  } else {
    pendingSeekFrame = null;
  }
}

function throttledSeek(frame: number): void {
  if (seekTimer == null) {
    lastSeekIssued = frame;
    playback.requestSeek(frame);
    seekTimer = window.setTimeout(issuePendingSeek, SEEK_THROTTLE_MS);
  } else {
    pendingSeekFrame = frame;
  }
}

let podAdjustWasActive = false;
watch(
  () => [ui.podAdjustActive, ui.podAdjustFrame, ui.podAdjustKind] as const,
  ([active, frame, kind]) => {
    if (active && frame != null) {
      scene?.setPodAdjust({ frame, kind });
      if (!podAdjustWasActive) {
        // Entering adjust: bring the mark into comfortable view (matters
        // when zoomed in and the mark sits at the far end of the timeline).
        scene?.centerOnFrame(frame);
      }
    } else {
      scene?.setPodAdjust(null);
    }
    podAdjustWasActive = Boolean(active);
  },
);

// window.confirm is a no-op (always false) inside the Tauri webview — use
// the dialog plugin there, falling back to the browser dialog on the web.
async function confirmDialog(message: string): Promise<boolean> {
  const isTauri =
    typeof window !== 'undefined' &&
    ('__TAURI_INTERNALS__' in window || '__TAURI__' in window);
  if (isTauri) {
    try {
      const { confirm } = await import('@tauri-apps/plugin-dialog');
      return await confirm(message, { kind: 'warning' });
    } catch {
      return true; // dialog plugin unavailable: don't block the action
    }
  }
  return window.confirm(message);
}

async function onPodConfirm(): Promise<void> {
  const frame = ui.podAdjustFrame;
  if (frame == null) return;
  if (ui.podAdjustKind === 'drill_end') {
    const podFrame = podFrameOf();
    if (
      podFrame != null &&
      frame <= podFrame &&
      !(await confirmDialog(
        `The new drill end (frame ${frame}) is at/before the established POD ` +
          `(frame ${podFrame}). POD will be re-detected inside the new window ` +
          `and may become not-established. Continue?`,
      ))
    ) {
      return;
    }
    const ok = await session.adjustDrillEnd(frame);
    if (ok) {
      ui.exitPodAdjust();
      playback.requestSeek(frame);
    }
    return;
  }
  const ok = await session.adjustPodFrame(frame);
  if (ok) {
    ui.exitPodAdjust();
    playback.requestSeek(frame);
  }
}

function podFrameOf(): number | null {
  const m = session.metricById.get('pod_sector_coverage');
  if (!m || m.metric_id !== 'pod_sector_coverage') return null;
  return m.summary.pod_frame;
}

function onPodCancel(): void {
  ui.exitPodAdjust();
}

function onZoomIn(): void {
  scene?.zoomBy(1.4);
}
function onZoomOut(): void {
  scene?.zoomBy(1 / 1.4);
}
function onFit(): void {
  scene?.fit();
}
function onJumpToFirst(): void {
  if (!scene) return;
  const f = scene.firstArtifactFrame();
  if (f == null) return;
  playback.requestSeek(f);
  scene.scrollToFrame(f, 'start');
}

onBeforeUnmount(() => {
  const c = canvasContainer.value;
  if (c) {
    c.removeEventListener('scroll', onCanvasScroll);
    c.removeEventListener('pointerdown', onTlPointerDown);
    c.removeEventListener('pointermove', onTlPointerMove);
    c.removeEventListener('pointerup', onTlPointerUp);
    c.removeEventListener('pointercancel', onTlPointerUp);
  }
  if (rippleTimer != null) clearTimeout(rippleTimer);
  if (seekTimer != null) clearTimeout(seekTimer);
  scene?.destroy();
  scene = null;
});
</script>

<style scoped>
.timeline {
  position: relative;
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--color-bg-elev);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  overflow: hidden;
}

/* Body holds the label gutter and the canvas side-by-side so the gutter
   never overlaps canvas content. */
.timeline-body {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: row;
  overflow: hidden;
}

.canvas-container {
  flex: 1;
  min-width: 0;
  min-height: 0;
  /* The actual scrollbar styling lives in theme.css under .always-scroll —
   * applied via class on the template above so the ::-webkit-scrollbar
   * pseudo-elements bind reliably (no Vue scoped-css dance). */
}

/* Fixed-width left gutter for metric labels. Vertical scroll is mirrored
   from the canvas container via JS; horizontal scroll never affects this
   column, so labels can't overlap timeline content. */
.label-column {
  flex: 0 0 130px;
  position: relative;
  overflow: hidden;
  border-right: 1px solid var(--color-border);
  background: var(--color-bg-elev);
}
.label-rail {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  /* `transform: translateY` is set imperatively in onCanvasScroll. */
  will-change: transform;
}
.metric-label {
  position: absolute;
  left: 6px;
  right: 6px;
  /* Vertically centred on its metric region — `top` is set to the region's
     midpoint, transform shifts the label up by half its own height. */
  transform: translateY(-50%);
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 2px 6px;
  background: transparent;
  border: 0;
  border-radius: 3px;
  font-size: 0.72em;
  color: var(--color-text);
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.4;
  font-variant-numeric: tabular-nums;
}
.metric-label .swatch {
  width: 8px;
  height: 8px;
  border-radius: 1.5px;
  display: inline-block;
  flex-shrink: 0;
}
.metric-label .name {
  letter-spacing: 0.01em;
}

.tooltip {
  position: absolute;
  background: rgba(15, 17, 21, 0.96);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  padding: 4px 8px;
  font-size: 0.78em;
  pointer-events: none;
  white-space: nowrap;
  /* Default: appear above the cursor */
  transform: translate(-50%, calc(-100% - 10px));
  z-index: 5;
  font-variant-numeric: tabular-nums;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
}
/* Flip below the cursor when hovering near the top of the timeline so the
   tooltip never escapes the visible area. */
.tooltip.below {
  transform: translate(-50%, 14px);
}

.bar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 4px 10px;
  border-top: 1px solid var(--color-border);
  background: var(--color-bg);
  font-size: 0.75em;
}
.zoom-controls {
  display: inline-flex;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  overflow: hidden;
}
.pod-adjust-hint {
  color: #c084fc;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}
.pod-adjust-hint.drill-kind {
  color: #86efac;
}
.pod-confirm {
  border: 1px solid #c084fc !important;
  border-radius: 4px;
  color: #c084fc !important;
  font-weight: 600;
}
.pod-cancel {
  border: 1px solid var(--color-border) !important;
  border-radius: 4px;
}
.pod-confirm:disabled,
.pod-cancel:disabled {
  opacity: 0.5;
  cursor: default;
}

.zoom-btn {
  background: transparent;
  color: var(--color-text);
  border: none;
  border-right: 1px solid var(--color-border);
  padding: 2px 10px;
  font-size: 1em;
  cursor: pointer;
  min-width: 28px;
  font-variant-numeric: tabular-nums;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}
.zoom-btn:last-child {
  border-right: none;
}
.zoom-btn:hover,
.zoom-btn:active {
  background: var(--color-border);
}
.hint {
  flex: 1;
}

/* Touch: enlarge the zoom/jump buttons to comfortable tap targets and let the
 * bottom bar wrap instead of clipping the hint on a narrow screen. */
@media (hover: none) {
  .zoom-btn {
    min-width: 42px;
    padding: 9px 12px;
  }
  .bar {
    flex-wrap: wrap;
    row-gap: 4px;
  }
}
.range-readout {
  font-variant-numeric: tabular-nums;
}
.muted {
  color: var(--color-muted);
}

/* Tap confirmation ring — blooms once at the press point, then fades. Sits
 * above the canvas but is pointer-transparent and gone in <0.5s, so it never
 * hides neighbouring events. */
.tap-ripple {
  position: absolute;
  width: 16px;
  height: 16px;
  margin: -8px 0 0 -8px;
  border-radius: 50%;
  border: 2px solid var(--color-accent);
  background: var(--color-accent-bg);
  pointer-events: none;
  z-index: 6;
  animation: tl-tap-ripple 0.45s ease-out forwards;
}
@keyframes tl-tap-ripple {
  0% {
    transform: scale(0.4);
    opacity: 0.95;
  }
  100% {
    transform: scale(2.3);
    opacity: 0;
  }
}

/* Touch: the container is panned via JS (Timeline.vue pointer handlers), so
 * stop the browser from also trying to scroll/zoom it. Desktop is unaffected. */
@media (hover: none) {
  .canvas-container {
    touch-action: none;
  }
}
</style>
