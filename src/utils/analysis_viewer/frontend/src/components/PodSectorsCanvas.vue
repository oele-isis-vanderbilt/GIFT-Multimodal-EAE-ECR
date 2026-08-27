<template>
  <!--
    POD sector overlay on the aux (map) video cell — same host/transform
    pattern as MapBandCanvas. Selection-driven only: sectors appear when a
    POD metric, the POD timeline item, or a POD flag is clicked, and
    disappear when the selection is cleared. Merely playing through the POD
    frame never shows them.
  -->
  <div ref="hostEl" class="pod-sectors-host" aria-hidden="true">
    <canvas ref="canvasEl" class="pod-sectors-canvas" />
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { useSessionStore } from '@/stores/session';
import { useUIStore } from '@/stores/ui';
import { usePlaybackStore } from '@/stores/playback';
import { colorForTrackId } from '@/theme/tokens';
import type { PodSectorCoverageSummary } from '@/types/models';

const session = useSessionStore();
const ui = useUIStore();
const playback = usePlaybackStore();

const hostEl = ref<HTMLDivElement | null>(null);
const canvasEl = ref<HTMLCanvasElement | null>(null);

let videoEl: HTMLVideoElement | null = null;
let metaListener: (() => void) | null = null;
let resizeObserver: ResizeObserver | null = null;

function findVideo(): HTMLVideoElement | null {
  const host = hostEl.value;
  if (!host) return null;
  const cell = host.closest('.video-cell');
  return (cell?.querySelector('video') as HTMLVideoElement) ?? null;
}

function attachToVideo(): void {
  let attempts = 0;
  const tryOnce = (): void => {
    attempts += 1;
    const next = findVideo();
    if (!next) {
      if (attempts < 12) requestAnimationFrame(tryOnce);
      return;
    }
    if (videoEl === next) {
      ensureCanvasMatchesVideo();
      draw();
      return;
    }
    detachFromVideo();
    videoEl = next;
    metaListener = (): void => {
      ensureCanvasMatchesVideo();
      draw();
    };
    videoEl.addEventListener('loadedmetadata', metaListener);
    videoEl.addEventListener('emptied', metaListener);
    if (resizeObserver === null && typeof ResizeObserver !== 'undefined') {
      resizeObserver = new ResizeObserver(() => {
        ensureCanvasMatchesVideo();
        draw();
      });
    }
    if (resizeObserver) resizeObserver.observe(videoEl);
    ensureCanvasMatchesVideo();
    draw();
  };
  queueMicrotask(tryOnce);
}

function detachFromVideo(): void {
  if (videoEl && metaListener) {
    videoEl.removeEventListener('loadedmetadata', metaListener);
    videoEl.removeEventListener('emptied', metaListener);
  }
  if (resizeObserver && videoEl) resizeObserver.unobserve(videoEl);
  videoEl = null;
  metaListener = null;
}

function ensureCanvasMatchesVideo(): void {
  const c = canvasEl.value;
  const v = videoEl;
  if (!c || !v) return;
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const cssW = Math.max(1, Math.floor(v.clientWidth));
  const cssH = Math.max(1, Math.floor(v.clientHeight));
  if (c.style.width !== `${cssW}px`) c.style.width = `${cssW}px`;
  if (c.style.height !== `${cssH}px`) c.style.height = `${cssH}px`;
  const bufW = Math.max(1, Math.round(cssW * dpr));
  const bufH = Math.max(1, Math.round(cssH * dpr));
  if (c.width !== bufW) c.width = bufW;
  if (c.height !== bufH) c.height = bufH;
}

// ---- Selection gate -------------------------------------------------
//
// What renders depends on WHAT was clicked:
//   sector-coverage flag       -> sectors only
//   pod_flagging flag          -> ALL sectors + that pair's red line + circles
//   POD establishment          -> member circles only ("just the circles")
//   coverage metric selected   -> sectors + circles
//   facing metric selected     -> circles
// Anything else -> nothing (clicking away always clears the overlay).

const POD_IDS = new Set(['pod_sector_coverage', 'pod_mutual_facing']);

type PodOverlayMode = {
  sectors: boolean;
  members: boolean;
  pairLine: { from: number; to: number } | null;
} | null;

function podOverlayMode(): PodOverlayMode {
  if (ui.selectedFlagId) {
    const flag = session.flagById.get(ui.selectedFlagId);
    if (flag?.type === 'pod_flagging' && flag.track_id != null && flag.target_id != null) {
      return { sectors: true, members: true, pairLine: { from: flag.track_id, to: flag.target_id } };
    }
    if (flag?.type === 'pod_coverage') {
      return { sectors: true, members: false, pairLine: null };
    }
    if (flag?.type === 'pod_establishment') {
      return { sectors: false, members: true, pairLine: null };
    }
    if (flag && flag.metric_id != null && POD_IDS.has(flag.metric_id)) {
      return { sectors: false, members: true, pairLine: null };
    }
    return null;
  }
  if (ui.selectedItemId) {
    const item = session.itemById.get(ui.selectedItemId);
    return item?.kind === 'pod_establishment'
      ? { sectors: false, members: true, pairLine: null }
      : null;
  }
  if (ui.selectedMetricId === 'pod_sector_coverage') {
    return { sectors: true, members: true, pairLine: null };
  }
  if (ui.selectedMetricId === 'pod_mutual_facing') {
    return { sectors: false, members: true, pairLine: null };
  }
  return null;
}

function podSummary(): PodSectorCoverageSummary | null {
  const metric = session.metricById.get('pod_sector_coverage');
  if (!metric || metric.metric_id !== 'pod_sector_coverage') return null;
  const s = metric.summary;
  return s.pod_frame != null ? s : null;
}

function buildTransform(): { scale: number; offX: number; offY: number } | null {
  const v = videoEl;
  const c = canvasEl.value;
  if (!v || !c) return null;
  const vw = v.videoWidth;
  const vh = v.videoHeight;
  const cssW = v.clientWidth;
  const cssH = v.clientHeight;
  if (vw <= 0 || vh <= 0 || cssW <= 0 || cssH <= 0) return null;
  const aspectV = vw / vh;
  const aspectC = cssW / cssH;
  let scale: number;
  let offX = 0;
  let offY = 0;
  if (aspectV > aspectC) {
    scale = cssW / vw;
    offY = (cssH - vh * scale) / 2;
  } else {
    scale = cssH / vh;
    offX = (cssW - vw * scale) / 2;
  }
  return { scale, offX, offY };
}

function draw(): void {
  const c = canvasEl.value;
  if (!c) return;
  const ctx = c.getContext('2d');
  if (!ctx) return;

  const dpr = Math.max(1, window.devicePixelRatio || 1);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, c.width / dpr, c.height / dpr);

  // Motion mode only (gaze artifacts have their own baked overlays), and
  // strictly selection-driven.
  if (playback.mode !== 'motion') return;
  const mode = podOverlayMode();
  if (!mode) return;
  const summary = podSummary();
  if (!summary) return;
  const t = buildTransform();
  if (!t) return;

  const px = (p: [number, number]): [number, number] => [
    p[0] * t.scale + t.offX,
    p[1] * t.scale + t.offY,
  ];

  if (mode.sectors) {
    for (const [tidStr, polys] of Object.entries(summary.sectors ?? {})) {
      const color = colorForTrackId(Number(tidStr));
      for (const poly of polys) {
        if (!poly || poly.length < 3) continue;
        const path = new Path2D();
        const [x0, y0] = px(poly[0]);
        path.moveTo(x0, y0);
        for (let i = 1; i < poly.length; i++) {
          const [x, y] = px(poly[i]);
          path.lineTo(x, y);
        }
        path.closePath();
        ctx.globalAlpha = 0.28;
        ctx.fillStyle = color;
        ctx.fill(path);
        ctx.globalAlpha = 0.9;
        ctx.lineWidth = 1;
        ctx.strokeStyle = 'rgba(0,0,0,0.8)';
        ctx.stroke(path);
      }
    }
  }

  // The red dashed sector-of-fire line: ONLY for the clicked pair flag.
  if (mode.pairLine) {
    const memberById = new Map(summary.members.map((m) => [m.id, m]));
    const a = memberById.get(mode.pairLine.from);
    const b = memberById.get(mode.pairLine.to);
    if (a && b) {
      const [ax, ay] = px(a.pos);
      const [bx, by] = px(b.pos);
      ctx.globalAlpha = 0.95;
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // Member dots + short bearing ticks + id labels.
  if (!mode.members) {
    ctx.globalAlpha = 1.0;
    return;
  }
  for (const m of summary.members) {
    const [x, y] = px(m.pos);
    const color = colorForTrackId(m.id);
    const len = 22 * t.scale;
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + m.bearing[0] * len, y + m.bearing[1] * len);
    ctx.stroke();
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = 'rgba(0,0,0,0.9)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = 'rgba(0,0,0,0.85)';
    ctx.font = '600 11px system-ui, sans-serif';
    ctx.fillText(String(m.id), x + 7, y - 6);
  }
  ctx.globalAlpha = 1.0;
}

onMounted(() => {
  attachToVideo();
});

onBeforeUnmount(() => {
  detachFromVideo();
  if (resizeObserver) {
    resizeObserver.disconnect();
    resizeObserver = null;
  }
});

watch(() => playback.mode, () => {
  attachToVideo();
});
watch(() => session.session, () => {
  attachToVideo();
});
watch(
  () => [ui.selectedFlagId, ui.selectedItemId, ui.selectedMetricId] as const,
  () => draw(),
  { flush: 'post' },
);
</script>

<style scoped>
.pod-sectors-host {
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 5;
}
.pod-sectors-canvas {
  position: absolute;
  inset: 0;
  pointer-events: none;
}
</style>
