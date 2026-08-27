<template>
  <div class="center-panel">
    <button
      v-if="session.session"
      class="compare-toggle"
      :class="{ active: ui.viewMode === 'compare' }"
      :disabled="ui.podAdjustActive"
      :title="ui.viewMode === 'compare' ? 'Switch to analysis mode' : 'Switch to compare mode'"
      @click="ui.viewMode = ui.viewMode === 'compare' ? 'analysis' : 'compare'"
    >
      {{ ui.viewMode === 'compare' ? 'Analysis' : 'Compare' }}
    </button>

    <div v-if="!session.session" class="empty">
      <p class="muted">Open a session to begin.</p>
    </div>
    <template v-else>
      <div class="video-region">
        <VideoStage v-show="ui.viewMode === 'analysis'" />
        <CompareStage v-if="ui.viewMode === 'compare'" />
      </div>
      <div v-if="ui.viewMode === 'analysis'" class="timeline-region">
        <Timeline />
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { useSessionStore } from '@/stores/session';
import { useUIStore } from '@/stores/ui';
import VideoStage from './VideoStage.vue';
import CompareStage from './CompareStage.vue';
import Timeline from './Timeline.vue';
const session = useSessionStore();
const ui = useUIStore();
</script>

<style scoped>
.center-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  padding: 12px;
  gap: 10px;
  position: relative;
}
.compare-toggle {
  position: absolute;
  top: 12px;
  right: 12px;
  z-index: 10;
  background: var(--color-bg-elev);
  color: var(--color-muted);
  border: 1px solid var(--color-border);
  padding: 5px 14px;
  font-size: 0.82em;
  border-radius: 6px;
  cursor: pointer;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  font-weight: 600;
}
.compare-toggle:hover:not(.active) {
  color: var(--color-text);
  border-color: var(--color-text);
}
.compare-toggle.active {
  background: var(--color-accent);
  color: var(--color-bg);
  border-color: var(--color-accent);
}
.empty {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
}
.video-region {
  flex: 4;
  min-height: 240px;
}
.timeline-region {
  height: 160px;
  flex-shrink: 0;
}
.muted {
  color: var(--color-muted);
}

/* Phones / small tablets: the Viewer fills its (bounded) side-by-side cell, so
 * keep the desktop fill behaviour but with smaller floors and a shorter
 * timeline so the narrower video still has room. Desktop is unchanged. */
@media (max-width: 1000px) {
  .video-region {
    min-height: 110px;
  }
  .timeline-region {
    height: 150px;
  }
}
</style>
