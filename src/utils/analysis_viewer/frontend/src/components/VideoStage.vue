<template>
  <div class="video-stage">
    <header class="stage-header">
      <div class="mode-switch" v-if="availableModes.length > 1">
        <button
          v-for="m in availableModes"
          :key="m.value"
          :class="{ active: playback.mode === m.value }"
          :disabled="ui.podAdjustActive"
          @click="playback.mode = m.value"
        >
          {{ m.label }}
        </button>
      </div>
      <div v-else class="mode-label muted">{{ availableModes[0]?.label ?? 'Video' }}</div>
    </header>

    <div class="video-grid" :class="`grid-${currentLayout.length || 1}`">
      <div
        v-for="player in currentLayout"
        :key="`${playback.mode}-${player.role}`"
        class="video-cell"
        :class="player.role"
      >
        <VideoPlayer
          :src="player.src"
          :role="player.role"
          :muted="player.muted"
          :frame-rate="frameRate"
          :sync-frame="player.role === 'aux' ? auxSyncFrame : undefined"
          :sync-min-delta="2"
          :ref="(instance) => onPlayerRef(player.role, instance)"
          @frame="onPrimaryFrame"
          @ready="player.role === 'primary' ? onPrimaryReady() : undefined"
          @play="onPlay"
          @pause="onPause"
        />
        <SubtitleOverlay v-if="player.role === 'primary'" />
        <MapBandCanvas v-if="player.role === 'aux'" />
        <PodSectorsCanvas v-if="player.role === 'aux'" />
      </div>
    </div>

    <footer class="transport" :class="{ locked: ui.podAdjustActive }">
      <button class="play-btn" :disabled="!hasPrimary || ui.podAdjustActive" @click="togglePlay">
        {{ playback.isPlaying ? '⏸' : '▶' }}
      </button>
      <input
        class="scrub"
        type="range"
        :min="playbackRange.start"
        :max="playbackRange.end"
        :value="playback.currentFrame"
        :disabled="!hasPrimary || ui.podAdjustActive"
        @input="onScrub"
      />
      <span class="readout">
        <span class="frame">{{ playback.currentFrame }} / {{ playbackRange.end }}</span>
        <span class="time">{{ playback.currentTime.toFixed(2) }}s</span>
      </span>
      <select
        class="speed"
        :value="playback.speed"
        :disabled="!hasPrimary || ui.podAdjustActive"
        @change="onSpeedChange"
      >
        <option v-for="s in speedOptions" :key="s" :value="s">{{ s }}×</option>
      </select>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { computed, watch } from 'vue';
import { useSessionStore } from '@/stores/session';
import { usePlaybackStore, type VideoMode } from '@/stores/playback';
import { useUIStore } from '@/stores/ui';
import { videoUrl } from '@/api/client';
import VideoPlayer from './VideoPlayer.vue';
import SubtitleOverlay from './SubtitleOverlay.vue';
import MapBandCanvas from './MapBandCanvas.vue';
import PodSectorsCanvas from './PodSectorsCanvas.vue';

type PlayerRole = 'primary' | 'aux';
type PlayerInstance = InstanceType<typeof VideoPlayer>;
type PlayerSpec = { role: PlayerRole; src: string | null; muted: boolean };

const session = useSessionStore();
const playback = usePlaybackStore();
const ui = useUIStore();

// Pause playback whenever we leave analysis mode. VideoStage stays mounted
// (v-show) behind the Compare view, so without this the video would keep
// playing — audio and all — while the user is comparing. Only acts if it was
// actually playing, so an already-paused video is left untouched.
watch(
  () => ui.viewMode,
  (mode) => {
    if (mode !== 'analysis' && playback.isPlaying) {
      playerRefs.get('primary')?.pause();
      playerRefs.get('aux')?.pause();
      playback.isPlaying = false;
    }
  },
);

// Entering POD/drill-end adjust mode must (a) stop playback — the marker is
// placed against a specific frame, so the video must not drift under it —
// and (b) land the primary player on the adjust frame. The seek is issued
// post-flush: enterAdjust also forces mode='original', which remounts the
// primary (keyed cell), so a seek issued in the same tick would be consumed
// by the outgoing player and lost.
watch(
  () => ui.podAdjustActive,
  (active) => {
    if (!active) return;
    resumeAfterModeChange = false; // the mode watcher may have latched a resume
    playerRefs.get('primary')?.pause();
    playerRefs.get('aux')?.pause();
    playback.isPlaying = false;
    if (ui.podAdjustFrame != null) playback.requestSeek(ui.podAdjustFrame);
  },
  { flush: 'post' },
);

const speedOptions = [0.25, 0.5, 1, 1.5, 2, 4];
const playerRefs = new Map<PlayerRole, PlayerInstance>();

function onPlayerRef(role: PlayerRole, instance: unknown): void {
  if (instance && typeof instance === 'object') {
    playerRefs.set(role, instance as PlayerInstance);
  } else {
    playerRefs.delete(role);
  }
}

const sessionPath = computed(() => session.session?.session_json_path ?? '');
const frameRate = computed(() => session.session?.video.frame_rate ?? 60);
const totalFrames = computed(() => session.session?.video.total_frames ?? 0);

// In motion/gaze mode the artifact videos are trimmed to the drill window.
// Their internal time-zero corresponds to the original video's drill_start
// frame. We translate between artifact-local and original-video coordinates
// here so the rest of the app (timeline, scrubber, store) only ever deals
// with original-video frames.
const videoOffsetFrames = computed(() => {
  if (playback.mode === 'original') return 0;
  return session.session?.drill_window?.start_frame ?? 0;
});

const playbackRange = computed(() => {
  if (playback.mode === 'original') {
    return { start: 0, end: Math.max(0, totalFrames.value - 1) };
  }
  const dw = session.session?.drill_window;
  if (!dw) return { start: 0, end: Math.max(0, totalFrames.value - 1) };
  return { start: dw.start_frame, end: dw.end_frame };
});

// Aux's syncFrame prop is artifact-local (each aux video starts at its own
// time-zero). Translate playback.currentFrame from original-video coords.
const auxSyncFrame = computed(() => playback.currentFrame - videoOffsetFrames.value);

const availableModes = computed(() => {
  const out: { value: VideoMode; label: string }[] = [];
  const v = session.session?.artifacts.videos ?? {};
  if (v.original) out.push({ value: 'original', label: 'Original' });
  if (v.motion_camera || v.motion_map) out.push({ value: 'motion', label: 'Motion' });
  if (v.gaze_camera || v.gaze_map) out.push({ value: 'gaze', label: 'Gaze' });
  return out;
});

const currentLayout = computed<PlayerSpec[]>(() => {
  const v = session.session?.artifacts.videos ?? {};
  const path = sessionPath.value;
  const url = (p?: string): string | null => (p ? videoUrl(p, path) : null);

  if (playback.mode === 'motion') {
    return [
      { role: 'primary', src: url(v.motion_camera), muted: false },
      { role: 'aux', src: url(v.motion_map), muted: true },
    ];
  }
  if (playback.mode === 'gaze') {
    return [
      { role: 'primary', src: url(v.gaze_camera), muted: false },
      { role: 'aux', src: url(v.gaze_map), muted: true },
    ];
  }
  return [{ role: 'primary', src: url(v.original), muted: false }];
});

const hasPrimary = computed(() => Boolean(currentLayout.value.find((p) => p.role === 'primary')?.src));

function onPrimaryFrame(frameIdx: number, timeSec: number): void {
  // The VideoPlayer reports the playback position in artifact-local frames.
  // Translate to original-video frames so the timeline + scrubber stay in
  // sync regardless of which mode is active.
  const offset = videoOffsetFrames.value;
  const originalFrame = frameIdx + offset;
  const originalTime = timeSec + offset / frameRate.value;
  playback.setFrame(originalFrame, originalTime);
}

function onPlay(): void {
  playback.isPlaying = true;
  playerRefs.get('aux')?.play();
}

function onPause(): void {
  playback.isPlaying = false;
  playerRefs.get('aux')?.pause();
}

function togglePlay(): void {
  const primary = playerRefs.get('primary');
  if (!primary) return;
  if (playback.isPlaying) {
    primary.pause();
    playerRefs.get('aux')?.pause();
  } else {
    primary.play();
    playerRefs.get('aux')?.play();
  }
}

function onScrub(ev: Event): void {
  const frame = Number((ev.target as HTMLInputElement).value);
  if (Number.isNaN(frame)) return;
  // The slider's value is already in original-video frames; clamp to the
  // active mode's playable range.
  const range = playbackRange.value;
  const clamped = Math.max(range.start, Math.min(range.end, frame));
  const offset = videoOffsetFrames.value;
  const localTime = (clamped - offset) / frameRate.value;
  playerRefs.get('primary')?.seek(localTime);
  // Aux follows primary via the syncFrame watcher inside VideoPlayer.
  playback.setFrame(clamped, localTime + offset / frameRate.value);
}

function onSpeedChange(ev: Event): void {
  const rate = Number((ev.target as HTMLSelectElement).value);
  if (Number.isNaN(rate)) return;
  playback.speed = rate;
  playerRefs.get('primary')?.setRate(rate);
  playerRefs.get('aux')?.setRate(rate);
}

// Default the mode to whatever's available; if the user's prior choice
// (e.g. "motion") is no longer present, snap to the first option.
watch(
  availableModes,
  (modes) => {
    if (modes.length === 0) return;
    if (!modes.find((m) => m.value === playback.mode)) {
      playback.mode = modes[0].value;
    }
  },
  { immediate: true },
);

// When the user changes mode, snap currentFrame into the new playable range
// so the new primary's load doesn't leave the scrubber out of bounds. Also
// remember whether playback was running so we can resume after the new
// primary video finishes loading (otherwise switching mode while playing
// leaves the UI showing "playing" but the new player paused).
let resumeAfterModeChange = false;

watch(
  () => playback.mode,
  () => {
    if (playback.isPlaying) {
      resumeAfterModeChange = true;
    }
    const range = playbackRange.value;
    const now = playback.currentFrame;
    if (now < range.start || now > range.end) {
      const target = Math.max(range.start, Math.min(range.end, now));
      playback.requestSeek(target);
    }
  },
);

function onPrimaryReady(): void {
  if (resumeAfterModeChange) {
    resumeAfterModeChange = false;
    playerRefs.get('primary')?.play();
    playerRefs.get('aux')?.play();
  }
}

// Honour seek requests from elsewhere (timeline clicks, keyboard, etc.).
// Seeks arrive in original-video frame coordinates; clamp to the active
// mode's playable range and translate to artifact-local time.
watch(
  () => playback.seekRequest,
  (req) => {
    if (!req) return;
    const range = playbackRange.value;
    const clamped = Math.max(range.start, Math.min(range.end, req.frame));
    const offset = videoOffsetFrames.value;
    const localTime = (clamped - offset) / frameRate.value;
    playerRefs.get('primary')?.seek(localTime);
    playback.setFrame(clamped, localTime + offset / frameRate.value);
  },
);
</script>

<style scoped>
.video-stage {
  display: flex;
  flex-direction: column;
  height: 100%;
  gap: 8px;
}

.stage-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 4px 4px 0;
}
.mode-switch {
  display: inline-flex;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  overflow: hidden;
}
.mode-switch button {
  background: transparent;
  color: var(--color-muted);
  border: none;
  padding: 5px 12px;
  font-size: 0.85em;
  cursor: pointer;
  border-right: 1px solid var(--color-border);
}
.mode-switch button:last-child {
  border-right: none;
}
.mode-switch button.active {
  background: var(--color-accent);
  color: #fff;
}
.mode-switch button:hover:not(.active) {
  background: var(--color-bg-elev);
  color: var(--color-text);
}
.mode-label {
  font-size: 0.85em;
}

.video-grid {
  flex: 1;
  display: grid;
  gap: 4px;
  background: #000;
  border-radius: 4px;
  min-height: 200px;
  overflow: hidden;
}
.video-grid.grid-1 {
  grid-template-columns: 1fr;
}
.video-grid.grid-2 {
  grid-template-columns: 1fr 1fr;
}
.video-cell {
  position: relative;
  min-width: 0;
  min-height: 0;
}

/* Phones / small tablets: keep the two videos side by side like desktop —
 * stacking them just wastes vertical space and shrinks each one. Only relax the
 * fixed min-height so the grid fits the bounded viewer cell. */
@media (max-width: 1000px) {
  .video-grid {
    min-height: 0;
  }
}

/* POD-adjust mode: the timeline drag owns seeking — lock the transport so
   nothing else can move the video out from under the marker. */
.transport.locked {
  pointer-events: none;
  opacity: 0.45;
}

.transport {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 4px 4px;
}
.play-btn {
  background: var(--color-bg-elev);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  padding: 4px 14px;
  border-radius: 4px;
  cursor: pointer;
  min-width: 44px;
  font-size: 1em;
}
.play-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.play-btn:hover:not(:disabled) {
  background: var(--color-border);
}
.scrub {
  flex: 1;
  accent-color: var(--color-accent);
}
.readout {
  display: inline-flex;
  flex-direction: column;
  align-items: flex-end;
  font-variant-numeric: tabular-nums;
  font-size: 0.78em;
  color: var(--color-muted);
  min-width: 100px;
}
.readout .frame {
  color: var(--color-text);
  font-weight: 500;
}
.speed {
  background: var(--color-bg-elev);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  padding: 4px 6px;
  border-radius: 4px;
  font-size: 0.85em;
  cursor: pointer;
}
</style>
