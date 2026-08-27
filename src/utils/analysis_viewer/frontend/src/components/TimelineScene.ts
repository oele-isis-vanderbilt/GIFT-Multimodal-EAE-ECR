// PixiJS 8 timeline scene — owns the Application + scene graph.
//
// Layout
// ------
// Foundational artifacts (entries, drill-window markers) live BELOW the
// neutral track. Metric overlays (vectors, pair-gap bars + labels, total-
// entry duration bar, AND each metric's flags) live ABOVE in dynamically-
// stacked metric groups. Each metric occupies a CONTIGUOUS region; if its
// flags collide horizontally they stack into multiple flag sub-rows within
// that same region rather than spilling into a separate top band.
//
// Per-metric region (top to bottom):
//   1. flag sub-rows (0..N depending on how flag x-positions collide)
//   2. label row     (only when the metric uses labels — entrance_hesitation)
//   3. main row      (dots / bar)
//
// The currently-selected metric jumps to the top of the stack. Hidden
// metrics drop out and the rest collapse to fill. The track Y, entry-dot
// Y, drill-marker Y, and total canvas height are all derived from where
// the stack ends.

import {
  Application,
  Circle,
  Container,
  FederatedPointerEvent,
  Graphics,
  Rectangle,
  Text,
  TextStyle,
} from 'pixi.js';

import type {
  DurationTimelineItem,
  FlagRecord,
  PairGapTimelineItem,
  TimelineItem,
  WallExcursionTimelineItem,
} from '@/types/models';
import {
  BASELINE_ENTRY_COLOR_INT,
  metricColor,
  metricColorInt,
} from '@/theme/tokens';

export type HoverKind = 'item' | 'flag' | 'drill';

export type HoverPayload = {
  kind: HoverKind;
  id: string;
  clientX: number;
  clientY: number;
} | null;

export type MetricLabelInfo = {
  metricId: string;
  name: string;
  /** CSS-ready hex string, e.g. "#5eead4". */
  color: string;
  /** Top edge of the metric's region in canvas-content coordinates. */
  startY: number;
  /** Total height (px) of the metric's region (flag rows + main rows). */
  height: number;
};

export type TimelineCallbacks = {
  onItemClicked?: (itemId: string, seekFrame: number) => void;
  onFlagClicked?: (flagId: string) => void;
  onEmptyClicked?: (frame: number) => void;
  onHover?: (payload: HoverPayload) => void;
  /** Fires after every redraw with the visible metrics' label positions. */
  onLayoutChanged?: (labels: MetricLabelInfo[]) => void;
  /** POD-adjust mode: fires continuously while the POD marker is dragged. */
  onPodDragPreview?: (frame: number) => void;
};

export type TimelineSceneData = {
  items: TimelineItem[];
  flags: FlagRecord[];
  drillWindow: {
    start_frame: number;
    end_frame: number;
    end_uncertain: boolean;
  } | null;
  totalFrames: number;
  /** Metric ids present in the session — rows are only laid out for these,
   *  so legacy sessions without e.g. POD metrics don't render empty rows. */
  metricIds?: string[];
};

const COLOR = {
  track: 0x2a2d31,
  outline: 0x1a1d20,
  white: 0xffffff,
  drillStart: 0x86efac,
  drillEndUncertain: 0xfdba74,
  playhead: 0xef4444,
  labelText: 0xcbd5e1,
};

type MetricLayoutSpec = {
  /** Number of "main" rows (dots/bar/label) the metric needs. */
  mainRows: number;
  /** When true the metric's first main row is for textual labels (e.g. P1/P2/P3). */
  hasLabels: boolean;
};

const METRIC_LAYOUTS: Record<string, MetricLayoutSpec> = {
  entrance_vectors: { mainRows: 1, hasLabels: false },
  entrance_hesitation: { mainRows: 2, hasLabels: true },
  total_time_of_entry: { mainRows: 1, hasLabels: false },
  move_along_wall: { mainRows: 1, hasLabels: false },
  // Flags-only rows: no main content row (the POD mark itself lives on the
  // main timeline), so their breadth is exactly their flag sub-rows.
  pod_sector_coverage: { mainRows: 0, hasLabels: false },
  pod_mutual_facing: { mainRows: 0, hasLabels: false },
};

const DEFAULT_METRIC_ORDER: string[] = [
  'entrance_vectors',
  'entrance_hesitation',
  'total_time_of_entry',
  'move_along_wall',
  'pod_sector_coverage',
  'pod_mutual_facing',
];

const METRIC_DISPLAY_NAMES: Record<string, string> = {
  entrance_vectors: 'Vectors',
  entrance_hesitation: 'Hesitation',
  total_time_of_entry: 'Total Entry',
  move_along_wall: 'Wall',
  pod_sector_coverage: 'POD',
  pod_mutual_facing: 'POD Facing',
};

type MetricRenderPos = {
  startY: number;
  /** Y of each flag sub-row centre (badges drawn here for this metric). */
  flagRowYs: number[];
  /** Y baseline for the label row (defined only when hasLabels is true). */
  labelBaselineY?: number;
  /** Y for the main bar/dot row. For hesitation this is the bar's top Y;
   *  for single-row metrics this is the centre of the dot/bar. */
  mainY: number;
  /** Convenience: vertical centre of the metric's main content row. */
  mainCenterY: number;
};

const FLAG_RADIUS = 8;
const VECTOR_RADIUS = 5;

// Invisible hit-area radii. Visual sizes stay compact for mouse precision;
// the hit polygon is enlarged for reliable tap targets on touch surfaces.
const FLAG_HIT_RADIUS = 13;
const DOT_HIT_RADIUS = 11;
const DRILL_HIT_HALF = 12;
const BAR_HIT_VPAD = 4;

// Max pointer movement (in px) between pointerdown and pointerup that still
// counts as a tap rather than a drag-pan. Matches MapperCanvas's threshold.
const TAP_TOLERANCE_PX = 5;

const METRIC_STACK_START_Y = 8;
const ROW_HEIGHT = 16;
const METRIC_GAP = 4;
const TRACK_GAP = 8;

const TRACK_HEIGHT = 18;
const TRACK_TO_ENTRY_GAP = 8;
const ENTRY_RADIUS = 4;
const ENTRY_TO_DRILL_GAP = 14;
const DRILL_DIAMOND = 7;
const BOTTOM_PADDING = 10;

const BAR_HEIGHT = 6;

/** Minimum horizontal separation between two flag badges before we bump
 *  the second one into a new flag sub-row of the same metric. */
const FLAG_MIN_SPACING_PX = FLAG_RADIUS * 2 + 4;

const MIN_PPF = 0.05;
const MAX_PPF = 32.0;

const PAIR_LABEL_STYLE = new TextStyle({
  fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif',
  fontSize: 9,
  fontWeight: '600',
  fill: COLOR.labelText,
});

type ComputedLayout = {
  positions: Map<string, MetricRenderPos>;
  /** Map flag_id → y centre at which the flag should render (within its metric's group). */
  flagYByFlagId: Map<string, number>;
  trackY: number;
  entryY: number;
  drillY: number;
  totalHeight: number;
};

export class TimelineScene {
  private readonly app = new Application();
  private readonly root = new Container();
  private readonly trackBar = new Graphics();
  private readonly drillOverlay = new Graphics();
  private readonly rangesLayer = new Container();
  private readonly entriesLayer = new Container();
  private readonly vectorsLayer = new Container();
  private readonly flagsLayer = new Container();
  private readonly podLayer = new Container();
  private readonly drillMarkersLayer = new Container();
  // Drawn above the markers: a persistent ring around the selected event.
  private readonly selectionLayer = new Container();
  private readonly playhead = new Graphics();

  private container: HTMLElement | null = null;
  private callbacks: TimelineCallbacks = {};
  private resizeObserver: ResizeObserver | null = null;

  private items: TimelineItem[] = [];
  private flagsList: FlagRecord[] = [];
  private drillWindow: TimelineSceneData['drillWindow'] = null;
  private totalFrames = 1;
  private rangeStart = 0;
  private rangeEnd = 1;
  private hasUserZoomed = false;

  private pixelsPerFrame = 1;
  private currentFrame = 0;
  private metricVisibility: Record<string, boolean> = {};
  private selectedMetricId: string | null = null;
  private selectedFlagId: string | null = null;
  // The directly-clicked event (entry/vector/bar/flag). Drives the persistent
  // selection ring. Cleared when an empty spot or another event is clicked.
  private selectedItemId: string | null = null;
  // id -> shape geometry, rebuilt every redraw, used to draw the selection ring.
  private selGeom = new Map<
    string,
    | { kind: 'circle'; cx: number; cy: number; r: number }
    | { kind: 'rect'; x: number; y: number; w: number; h: number }
  >();
  /** Pointer-down state for click-vs-pan discrimination on the empty track.
   *  A seek only fires on pointerup when total movement is ≤ TAP_TOLERANCE_PX. */
  private stagePointerDown: { x: number; y: number; frame: number } | null = null;
  /** POD-frame adjustment mode: while set, the POD marker renders at
   *  ``frame``, is the only interactive element, and drags horizontally
   *  (reporting previews via onPodDragPreview). */
  private podAdjust: { frame: number; kind: 'pod' | 'drill_end' } | null = null;
  private podDragging = false;
  /** Metric ids present in the session data; null = accept all (legacy). */
  private presentMetricIds: Set<string> | null = null;
  /** flag_id → linked_item_id, rebuilt on each setData(). Used to gate
   *  pair_gap / wall_excursion bars: by design those bars are only
   *  rendered when the user has clicked their associated flag. */
  private flagLinkedItem: Map<string, string> = new Map();
  private layout: ComputedLayout = this.computeEmptyLayout();
  private initialized = false;

  async init(container: HTMLElement, callbacks: TimelineCallbacks): Promise<void> {
    this.container = container;
    this.callbacks = callbacks;

    await this.app.init({
      // Pin WebGL. Pixi v8 otherwise auto-selects WebGPU when the browser
      // exposes it (Chrome on localhost/127.0.0.1, desktop Safari 17+), and
      // WebGPU can silently render nothing (blank canvas, no error) in served /
      // remote / Safari contexts. WebGL is universally supported and reliable.
      preference: 'webgl',
      backgroundAlpha: 0,
      antialias: true,
      autoDensity: true,
      resolution: window.devicePixelRatio || 1,
      width: 100,
      height: this.layout.totalHeight,
    });

    container.appendChild(this.app.canvas);
    this.app.canvas.style.display = 'block';
    this.app.canvas.style.cursor = 'pointer';
    // Disable the browser's own touch gestures on the canvas; Timeline.vue
    // implements explicit drag-to-pan on the scroll container for touch/pen
    // (native overflow scrolling is unreliable here because Pixi intercepts the
    // touch stream). Taps still register as Pixi pointer events for seek/select.
    this.app.canvas.style.touchAction = 'none';

    this.app.stage.addChild(this.root);
    this.root.addChild(this.trackBar);
    // Drill-window overlay sits on the track but below interactive layers,
    // so clicks still hit the items / empty-track stage handler above.
    this.root.addChild(this.drillOverlay);
    this.root.addChild(this.rangesLayer);
    this.root.addChild(this.entriesLayer);
    this.root.addChild(this.vectorsLayer);
    this.root.addChild(this.flagsLayer);
    this.root.addChild(this.podLayer);
    this.root.addChild(this.drillMarkersLayer);
    this.root.addChild(this.selectionLayer);
    this.root.addChild(this.playhead);

    this.app.stage.eventMode = 'static';
    this.app.stage.on('pointerdown', this.onStagePointerDown);
    this.app.stage.on('pointerup', this.onStagePointerUp);
    this.app.stage.on('pointerupoutside', this.onStagePointerUp);

    container.addEventListener('mouseleave', this.onContainerLeave);

    this.resizeObserver = new ResizeObserver(() => this.onContainerResize());
    this.resizeObserver.observe(container);

    this.initialized = true;
    this.fitToWidth();
    this.redraw();
  }

  destroy(): void {
    if (!this.initialized) return;
    this.initialized = false;
    this.onWindowPodDragEnd();
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
    if (this.container) {
      this.container.removeEventListener('mouseleave', this.onContainerLeave);
    }
    this.app.destroy(true, { children: true, texture: true });
    this.container = null;
  }

  setData(data: TimelineSceneData): void {
    this.items = data.items;
    this.flagsList = data.flags;
    this.drillWindow = data.drillWindow;
    this.totalFrames = Math.max(1, data.totalFrames);
    this.presentMetricIds = data.metricIds ? new Set(data.metricIds) : null;
    // Rebuild the flag→item index used to gate segment-bar rendering.
    // Cheap (O(N_flags)); rebuilt only when the dataset changes.
    this.flagLinkedItem.clear();
    for (const flag of this.flagsList) {
      if (flag.flag_id && flag.linked_item_id) {
        this.flagLinkedItem.set(flag.flag_id, flag.linked_item_id);
      }
    }
    if (!this.hasUserZoomed) this.fitToWidth();
    this.redraw();
  }

  setRange(start: number, end: number): void {
    const clean = end > start ? { start, end } : { start, end: start + 1 };
    if (clean.start === this.rangeStart && clean.end === this.rangeEnd) return;
    this.rangeStart = clean.start;
    this.rangeEnd = clean.end;
    this.hasUserZoomed = false;
    this.fitToWidth();
    this.redraw();
  }

  setMetricVisibility(visibility: Record<string, boolean>): void {
    this.metricVisibility = visibility;
    this.redraw();
  }

  setSelectedMetric(metricId: string | null): void {
    if (this.selectedMetricId === metricId) return;
    this.selectedMetricId = metricId;
    this.redraw();
  }

  /**
   * Track the currently-selected flag. Pair-gap and wall-excursion bars
   * are hidden by default and revealed only when their linked flag is
   * selected — see ``drawRanges`` for the gating logic.
   */
  setSelectedFlag(flagId: string | null): void {
    if (this.selectedFlagId === flagId) return;
    this.selectedFlagId = flagId;
    this.redraw();
  }

  /** Track the directly-clicked event so it gets a persistent selection ring. */
  setSelectedItem(itemId: string | null): void {
    if (this.selectedItemId === itemId) return;
    this.selectedItemId = itemId;
    this.redraw();
  }

  setCurrentFrame(frame: number): void {
    this.currentFrame = frame;
    this.updatePlayheadPosition();
  }

  /** Multiplies pixelsPerFrame by `factor` (anchored at the visible centre). */
  zoomBy(factor: number): void {
    if (!this.container) return;
    const containerW = this.container.clientWidth;
    const scrollLeft = this.container.scrollLeft;
    const anchorCanvasX = scrollLeft + containerW / 2;
    const oldPpf = this.pixelsPerFrame;
    const newPpf = clamp(oldPpf * factor, MIN_PPF, MAX_PPF);
    if (newPpf === oldPpf) return;

    const frameUnderAnchor = this.rangeStart + anchorCanvasX / oldPpf;
    this.pixelsPerFrame = newPpf;
    this.hasUserZoomed = true;
    this.resizeCanvas();
    this.redraw();
    const newAnchorX = (frameUnderAnchor - this.rangeStart) * newPpf;
    this.container.scrollLeft = Math.max(0, newAnchorX - containerW / 2);
  }

  /** Reset zoom + pan to fit the full visible range. */
  fit(): void {
    this.hasUserZoomed = false;
    this.fitToWidth();
    this.redraw();
    if (this.container) this.container.scrollLeft = 0;
  }

  /** Lowest frame across visible items in the current range, or null. */
  firstArtifactFrame(): number | null {
    let best: number | null = null;
    for (const item of this.items) {
      const f = 'frame' in item ? item.frame : (item as PairGapTimelineItem).start_frame;
      if (f < this.rangeStart || f > this.rangeEnd) continue;
      if (best == null || f < best) best = f;
    }
    return best;
  }

  /** Scroll the parent container so `frame` lands near the left edge / centre. */
  scrollToFrame(frame: number, anchor: 'start' | 'center' = 'start'): void {
    if (!this.container) return;
    const x = this.xOf(frame);
    const containerW = this.container.clientWidth;
    const target = anchor === 'start' ? x - 24 : x - containerW / 2;
    this.container.scrollLeft = Math.max(0, target);
  }

  /**
   * Scroll the parent container so the flag's badge is centred horizontally
   * AND its owning metric row is in view vertically. Used by Timeline.vue
   * when ui.selectedFlagId changes — so clicking a flag in any panel always
   * brings it on-screen even when the timeline is zoomed in or panned away.
   */
  scrollToFlag(flagId: string): void {
    if (!this.container) return;
    const flag = this.flagsList.find((f) => f.flag_id === flagId);
    if (!flag || flag.frame == null) return;
    this.scrollToFrame(flag.frame, 'center');
    if (flag.metric_id) this.scrollMetricIntoView(flag.metric_id);
  }

  /** Vertical-only scroll so the metric's row is in the visible window. */
  scrollToMetric(metricId: string): void {
    this.scrollMetricIntoView(metricId);
  }

  private scrollMetricIntoView(metricId: string): void {
    if (!this.container) return;
    const pos = this.layout.positions.get(metricId);
    const spec = METRIC_LAYOUTS[metricId];
    if (!pos || !spec) return;
    const totalRows = pos.flagRowYs.length + spec.mainRows;
    const startY = pos.startY;
    const endY = startY + totalRows * ROW_HEIGHT;
    const containerH = this.container.clientHeight;
    const scrollTop = this.container.scrollTop;
    const margin = 8;
    if (startY < scrollTop + margin) {
      this.container.scrollTop = Math.max(0, startY - margin);
    } else if (endY > scrollTop + containerH - margin) {
      this.container.scrollTop = Math.max(0, endY - containerH + margin);
    }
  }

  // -- layout ---------------------------------------------------------------

  private computeEmptyLayout(): ComputedLayout {
    const trackY = METRIC_STACK_START_Y + TRACK_GAP;
    const entryY = trackY + TRACK_HEIGHT + TRACK_TO_ENTRY_GAP;
    const drillY = entryY + ENTRY_TO_DRILL_GAP;
    return {
      positions: new Map(),
      flagYByFlagId: new Map(),
      trackY,
      entryY,
      drillY,
      totalHeight: drillY + DRILL_DIAMOND + BOTTOM_PADDING,
    };
  }

  private computeLayout(): ComputedLayout {
    // Order: selected metric first, then others in default (data) order.
    // Only metrics actually present in the session get a row.
    const visible = DEFAULT_METRIC_ORDER.filter(
      (id) =>
        this.metricVisibility[id] !== false &&
        (this.presentMetricIds == null || this.presentMetricIds.has(id)),
    );
    const ordered =
      this.selectedMetricId && visible.includes(this.selectedMetricId)
        ? [this.selectedMetricId, ...visible.filter((id) => id !== this.selectedMetricId)]
        : visible;

    // Bucket flags by their owning metric for the per-metric flag-stacking
    // pass. Flags whose metric is not visible never enter the layout.
    const flagsByMetric = new Map<string, FlagRecord[]>();
    for (const flag of this.flagsList) {
      if (flag.frame == null) continue;
      if (!flag.metric_id) continue;
      if (this.metricVisibility[flag.metric_id] === false) continue;
      const arr = flagsByMetric.get(flag.metric_id) ?? [];
      arr.push(flag);
      flagsByMetric.set(flag.metric_id, arr);
    }

    const positions = new Map<string, MetricRenderPos>();
    const flagYByFlagId = new Map<string, number>();
    let cursor = METRIC_STACK_START_Y;

    for (const metricId of ordered) {
      const spec = METRIC_LAYOUTS[metricId];
      if (!spec) continue;

      const flagsForMetric = flagsByMetric.get(metricId) ?? [];
      const flagRowAssignment = this.assignFlagRows(flagsForMetric);
      const flagRowCount = flagRowAssignment.rowCount;

      // A flags-only metric with no flags at all has nothing to show —
      // drop the row entirely instead of reserving empty breadth. (Flags are
      // bucketed regardless of the visible range so row layout stays stable
      // when switching between full-video and drill-window views.)
      if (flagRowCount + spec.mainRows === 0) continue;

      const startY = cursor;
      const flagRowYs: number[] = [];
      for (let i = 0; i < flagRowCount; i++) {
        flagRowYs.push(startY + i * ROW_HEIGHT + ROW_HEIGHT / 2);
      }

      const mainStartY = startY + flagRowCount * ROW_HEIGHT;
      let labelBaselineY: number | undefined;
      let mainY: number;
      let mainCenterY: number;
      if (spec.hasLabels) {
        labelBaselineY = mainStartY + ROW_HEIGHT - 3;
        // Bar in the second main row (label row above + bar row below).
        mainY = mainStartY + ROW_HEIGHT + (ROW_HEIGHT - BAR_HEIGHT) / 2;
        mainCenterY = mainStartY + ROW_HEIGHT + ROW_HEIGHT / 2;
      } else {
        mainY = mainStartY + (ROW_HEIGHT - BAR_HEIGHT) / 2;
        mainCenterY = mainStartY + ROW_HEIGHT / 2;
      }

      positions.set(metricId, {
        startY,
        flagRowYs,
        labelBaselineY,
        mainY,
        mainCenterY,
      });

      // Resolve each flag's y from the row index assigned above.
      for (const { flag, rowIndex } of flagRowAssignment.placements) {
        flagYByFlagId.set(flag.flag_id, flagRowYs[rowIndex]);
      }

      cursor += (flagRowCount + spec.mainRows) * ROW_HEIGHT + METRIC_GAP;
    }

    if (positions.size === 0) cursor = METRIC_STACK_START_Y;

    const trackY = cursor + TRACK_GAP;
    const entryY = trackY + TRACK_HEIGHT + TRACK_TO_ENTRY_GAP;
    const drillY = entryY + ENTRY_TO_DRILL_GAP;
    const totalHeight = drillY + DRILL_DIAMOND + BOTTOM_PADDING;

    return { positions, flagYByFlagId, trackY, entryY, drillY, totalHeight };
  }

  /**
   * Greedy left-to-right packing: walks flags in frame order and places each
   * in the lowest-numbered row whose previous flag is far enough to the
   * left to not overlap horizontally. Returns the number of rows used and
   * each flag's assigned row index.
   */
  private assignFlagRows(flags: FlagRecord[]): {
    rowCount: number;
    placements: Array<{ flag: FlagRecord; rowIndex: number }>;
  } {
    if (flags.length === 0) return { rowCount: 0, placements: [] };

    const sorted = flags.slice().sort((a, b) => (a.frame ?? 0) - (b.frame ?? 0));
    const rowsLastX: number[] = [];
    const placements: Array<{ flag: FlagRecord; rowIndex: number }> = [];

    for (const flag of sorted) {
      if (flag.frame == null) continue;
      const x = this.xOf(flag.frame);
      let rowIndex = 0;
      while (
        rowIndex < rowsLastX.length &&
        rowsLastX[rowIndex] + FLAG_MIN_SPACING_PX > x
      ) {
        rowIndex++;
      }
      rowsLastX[rowIndex] = x;
      placements.push({ flag, rowIndex });
    }
    return { rowCount: rowsLastX.length, placements };
  }

  // -- internal -------------------------------------------------------------

  private onContainerResize(): void {
    if (!this.initialized) return;
    if (!this.hasUserZoomed) this.fitToWidth();
    this.resizeCanvas();
    this.redraw();
  }

  private fitToWidth(): void {
    if (!this.initialized || !this.container) return;
    const w = this.container.clientWidth;
    const span = this.rangeEnd - this.rangeStart;
    if (w <= 0 || span <= 0) return;
    this.pixelsPerFrame = clamp(w / span, MIN_PPF, MAX_PPF);
    this.resizeCanvas();
  }

  private resizeCanvas(): void {
    if (!this.container) return;
    const containerW = this.container.clientWidth;
    const span = this.rangeEnd - this.rangeStart;
    const contentW = Math.max(containerW, Math.ceil(span * this.pixelsPerFrame));
    this.app.renderer.resize(contentW, this.layout.totalHeight);
    this.app.stage.hitArea = this.app.screen;
  }

  private redraw(): void {
    this.layout = this.computeLayout();
    this.selGeom.clear();
    this.resizeCanvas();
    this.drawTrack();
    this.drawDrillOverlay();
    this.drawRanges();
    this.drawEntries();
    this.drawVectors();
    this.drawFlags();
    this.drawPodMarker();
    this.drawDrillMarkers();
    this.drawSelection();
    this.drawPlayhead();
    this.callbacks.onLayoutChanged?.(this.buildMetricLabels());
  }

  /** Enter/leave mark-adjust mode (POD or drill end) with a preview frame. */
  setPodAdjust(state: { frame: number; kind?: 'pod' | 'drill_end' } | null): void {
    const kind = state?.kind ?? 'pod';
    const changed =
      (state == null) !== (this.podAdjust == null) ||
      (state != null && this.podAdjust != null &&
        (state.frame !== this.podAdjust.frame || kind !== this.podAdjust.kind));
    this.podAdjust = state == null ? null : { frame: state.frame, kind };
    if (state == null) this.onWindowPodDragEnd();
    if (changed) this.redraw();
  }

  /**
   * The POD-establishment / drill-end markers on the main persistent
   * timeline row (there is deliberately no duplicate in the POD metric row).
   * Normal mode: a clickable purple diamond at the established POD frame.
   * Adjust mode: rendered at the preview frame with a full-height guide line
   * and horizontal-drag behavior (the only interactive element while
   * adjusting) — for both the POD mark and the drill end.
   */
  private drawPodMarker(): void {
    this.podLayer.removeChildren();
    const adjusting = this.podAdjust != null;
    const adjustKind = this.podAdjust?.kind ?? 'pod';
    const item = this.items.find((it) => it.kind === 'pod_establishment');
    const podColor = metricColorInt('pod_sector_coverage');

    // --- adjust-mode marker (POD or drill end) ---------------------------
    if (adjusting) {
      const frame = this.podAdjust!.frame;
      if (!this.inRange(frame)) return;
      const color = adjustKind === 'drill_end' ? COLOR.drillStart : podColor;
      // Both marks are adjusted on the main persistent timeline row — the
      // full-height guide line still shows the frame against every band.
      const y = this.layout.drillY;
      const g = new Graphics();
      g.rect(-0.75, 0, 1.5, this.layout.totalHeight).fill({ color, alpha: 0.5 });
      g.circle(0, y, FLAG_RADIUS + 4).fill({ color, alpha: 0.25 });
      const size = DRILL_DIAMOND + 3;
      g.poly([0, y - size, size, y, 0, y + size, -size, y])
        .fill({ color })
        .stroke({ color: COLOR.outline, width: 1.5 });
      g.x = this.edgeX(frame);
      g.eventMode = 'static';
      g.cursor = 'ew-resize';
      g.hitArea = new Circle(0, y, FLAG_HIT_RADIUS + 6);
      g.on('pointerdown', (ev: FederatedPointerEvent) => {
        ev.stopPropagation();
        this.startPodDrag();
      });
      this.podLayer.addChild(g);
      return;
    }

    // --- static POD markers ---------------------------------------------
    if (!item || !('frame' in item)) return;
    const frame = (item as { frame: number }).frame;
    if (!this.inRange(frame)) return;
    const x = this.edgeX(frame);

    // Persistent mark on the always-visible drill row (main timeline).
    {
      const y = this.layout.drillY;
      const size = DRILL_DIAMOND;
      const g = new Graphics();
      const selected =
        this.selectedItemId === item.item_id ||
        this.selectedFlagId === 'pod_establishment_flag';
      if (selected) {
        // The selGeom registry keys one shape per id (used by the row
        // marker), so this second location draws its own ring.
        const rr = size + 3.5;
        g.circle(0, y, rr).stroke({ color: 0x0b0e14, width: 4, alpha: 0.55 });
        g.circle(0, y, rr).stroke({ color: 0xffffff, width: 2, alpha: 0.95 });
      }
      g.poly([0, y - size, size, y, 0, y + size, -size, y])
        .fill({ color: podColor })
        .stroke({ color: COLOR.outline, width: 1 });
      g.x = x;
      g.eventMode = 'static';
      g.cursor = 'pointer';
      g.hitArea = new Circle(0, y, FLAG_HIT_RADIUS);
      this.attachClickAndHover(g, item.item_id, frame, 'item');
      this.podLayer.addChild(g);
    }

    // No duplicate marker in the POD metric row — the persistent diamond on
    // the main timeline is the single POD mark; the metric row carries only
    // its flag badges (sector-coverage / facing).
  }

  // -- selection ring -------------------------------------------------------

  private regCircle(id: string, cx: number, cy: number, r: number): void {
    this.selGeom.set(id, { kind: 'circle', cx, cy, r });
  }
  private regRect(id: string, x: number, y: number, w: number, h: number): void {
    this.selGeom.set(id, { kind: 'rect', x, y, w, h });
  }

  /** Draw a persistent ring around the selected event, matching its shape.
   * A reserved white ring (with a dark halo for contrast on any background) is
   * used so it never collides with metric colours. */
  private drawSelection(): void {
    this.selectionLayer.removeChildren();
    const id = this.selectedItemId ?? this.selectedFlagId;
    if (!id) return;
    const geom = this.selGeom.get(id);
    if (!geom) return; // selected event is off-range or its metric is hidden

    const g = new Graphics();
    const PAD = 3.5;
    if (geom.kind === 'circle') {
      const rr = geom.r + PAD;
      g.circle(geom.cx, geom.cy, rr).stroke({ color: 0x0b0e14, width: 4, alpha: 0.55 });
      g.circle(geom.cx, geom.cy, rr).stroke({ color: 0xffffff, width: 2, alpha: 0.95 });
    } else {
      const x = geom.x - PAD;
      const y = geom.y - PAD;
      const w = geom.w + PAD * 2;
      const h = geom.h + PAD * 2;
      g.roundRect(x, y, w, h, 3).stroke({ color: 0x0b0e14, width: 4, alpha: 0.55 });
      g.roundRect(x, y, w, h, 3).stroke({ color: 0xffffff, width: 2, alpha: 0.95 });
    }
    this.selectionLayer.addChild(g);
  }

  private buildMetricLabels(): MetricLabelInfo[] {
    const out: MetricLabelInfo[] = [];
    for (const [metricId, pos] of this.layout.positions.entries()) {
      const spec = METRIC_LAYOUTS[metricId];
      if (!spec) continue;
      const totalRows = pos.flagRowYs.length + spec.mainRows;
      out.push({
        metricId,
        name: METRIC_DISPLAY_NAMES[metricId] ?? metricId,
        color: metricColor(metricId),
        startY: pos.startY,
        height: totalRows * ROW_HEIGHT,
      });
    }
    return out;
  }

  private xOf(frame: number): number {
    return (frame - this.rangeStart) * this.pixelsPerFrame;
  }

  /** Marker-centre x clamped inside the drawable area — a mark sitting at
   *  the exact range edge (e.g. a drill end at the last frame) would
   *  otherwise be half- or fully clipped by the canvas border. */
  private edgeX(frame: number, margin: number = DRILL_DIAMOND + 2): number {
    return clamp(this.xOf(frame), margin, Math.max(margin, this.app.screen.width - margin));
  }

  private inRange(frame: number): boolean {
    return frame >= this.rangeStart && frame <= this.rangeEnd;
  }

  private rangeOverlap(start: number, end: number): { x: number; w: number } | null {
    const s = Math.max(start, this.rangeStart);
    const e = Math.min(end, this.rangeEnd);
    if (e <= s) return null;
    return {
      x: this.xOf(s),
      w: Math.max(2, (e - s) * this.pixelsPerFrame),
    };
  }

  private drawTrack(): void {
    const w = (this.rangeEnd - this.rangeStart) * this.pixelsPerFrame;
    this.trackBar.clear();
    this.trackBar.rect(0, this.layout.trackY, w, TRACK_HEIGHT).fill({ color: COLOR.track });
  }

  /** Translucent green band on the track from drill_start → drill_end. */
  private drawDrillOverlay(): void {
    this.drillOverlay.clear();
    if (!this.drillWindow) return;
    const overlap = this.rangeOverlap(
      this.drillWindow.start_frame,
      this.drillWindow.end_frame,
    );
    if (!overlap) return;
    this.drillOverlay
      .rect(overlap.x, this.layout.trackY, overlap.w, TRACK_HEIGHT)
      .fill({ color: COLOR.drillStart, alpha: 0.18 });
  }

  private drawRanges(): void {
    this.rangesLayer.removeChildren();
    const hesPos = this.layout.positions.get('entrance_hesitation');
    const durPos = this.layout.positions.get('total_time_of_entry');
    const wallPos = this.layout.positions.get('move_along_wall');

    // pair_gap and wall_excursion bars are gated by the currently-selected
    // flag: only the bar whose item_id is linked from the selected flag is
    // rendered. With no flag selected, none are rendered. Always allow
    // ``duration`` items (the master "Total Entry" bar is unconditional).
    const selectedItemId = this.selectedFlagId
      ? this.flagLinkedItem.get(this.selectedFlagId) ?? null
      : null;
    const itemMatchesSelectedFlag = (id: string | undefined): boolean =>
      selectedItemId !== null && id === selectedItemId;

    for (const item of this.items) {
      if (item.kind === 'pair_gap' && hesPos) {
        if (!itemMatchesSelectedFlag(item.item_id)) continue;
        const c = this.makePairGapContainer(item, hesPos);
        if (c) this.rangesLayer.addChild(c);
      } else if (item.kind === 'duration' && durPos) {
        const c = this.makeDurationContainer(item, durPos);
        if (c) this.rangesLayer.addChild(c);
      } else if (item.kind === 'wall_excursion' && wallPos) {
        if (!itemMatchesSelectedFlag(item.item_id)) continue;
        const c = this.makeWallExcursionContainer(item, wallPos);
        if (c) this.rangesLayer.addChild(c);
      }
    }
  }

  private makePairGapContainer(
    item: PairGapTimelineItem,
    pos: MetricRenderPos,
  ): Container | null {
    const overlap = this.rangeOverlap(item.start_frame, item.end_frame);
    if (!overlap) return null;
    const color = metricColorInt(item.metric_id);
    const barY = pos.mainY;
    const labelY = pos.labelBaselineY ?? barY - 3;

    const c = new Container();
    c.label = item.metric_id;

    const bar = new Graphics();
    bar
      .rect(0, barY, overlap.w, BAR_HEIGHT)
      .fill({ color, alpha: 0.85 })
      .stroke({ color: COLOR.outline, width: 0.5, alpha: 0.7 });
    bar.x = overlap.x;
    bar.eventMode = 'static';
    bar.cursor = 'pointer';
    bar.hitArea = new Rectangle(0, barY - BAR_HIT_VPAD, overlap.w, BAR_HEIGHT + BAR_HIT_VPAD * 2);
    this.attachClickAndHover(bar, item.item_id, item.start_frame, 'item');
    this.regRect(item.item_id, overlap.x, barY, overlap.w, BAR_HEIGHT);
    c.addChild(bar);

    const pairTag = `P${item.data.pair_number}`;
    if (this.inRange(item.start_frame)) {
      c.addChild(this.makePairLabel(pairTag, this.xOf(item.start_frame), 'start', item, labelY));
    }
    if (this.inRange(item.end_frame)) {
      c.addChild(this.makePairLabel(pairTag, this.xOf(item.end_frame), 'end', item, labelY));
    }
    return c;
  }

  private makePairLabel(
    text: string,
    x: number,
    side: 'start' | 'end',
    item: PairGapTimelineItem,
    labelBaselineY: number,
  ): Text {
    const label = new Text({ text, style: PAIR_LABEL_STYLE });
    label.anchor.set(side === 'start' ? 0 : 1, 1);
    label.x = x + (side === 'start' ? 2 : -2);
    label.y = labelBaselineY;
    label.eventMode = 'static';
    label.cursor = 'pointer';
    this.attachClickAndHover(
      label,
      item.item_id,
      side === 'start' ? item.start_frame : item.end_frame,
      'item',
    );
    return label;
  }

  private makeDurationContainer(
    item: DurationTimelineItem,
    pos: MetricRenderPos,
  ): Container | null {
    const overlap = this.rangeOverlap(item.start_frame, item.end_frame);
    if (!overlap) return null;
    const color = metricColorInt(item.metric_id);
    const barY = pos.mainY;

    const c = new Container();
    c.label = item.metric_id;

    const bar = new Graphics();
    bar
      .rect(0, barY, overlap.w, BAR_HEIGHT)
      .fill({ color, alpha: 0.85 })
      .stroke({ color: COLOR.outline, width: 0.5, alpha: 0.7 });
    bar.x = overlap.x;
    bar.eventMode = 'static';
    bar.cursor = 'pointer';
    bar.hitArea = new Rectangle(0, barY - BAR_HIT_VPAD, overlap.w, BAR_HEIGHT + BAR_HIT_VPAD * 2);
    this.attachClickAndHover(bar, item.item_id, item.start_frame, 'item');
    this.regRect(item.item_id, overlap.x, barY, overlap.w, BAR_HEIGHT);
    c.addChild(bar);
    return c;
  }

  private makeWallExcursionContainer(
    item: WallExcursionTimelineItem,
    pos: MetricRenderPos,
  ): Container | null {
    const overlap = this.rangeOverlap(item.start_frame, item.end_frame);
    if (!overlap) return null;
    // Per-spec: bar uses the metric color. The two excursion kinds (too_close
    // / too_far) share the same row; the kind is conveyed through the tooltip
    // and through the linked flag list. Safe segments are rendered as gaps —
    // we never emit an item for them, so nothing is drawn.
    const color = metricColorInt(item.metric_id);
    const barY = pos.mainY;
    const alpha = item.data.label_kind === 'too_close' ? 0.95 : 0.65;

    const c = new Container();
    c.label = item.metric_id;

    const bar = new Graphics();
    const drawnW = Math.max(overlap.w, 1);
    bar
      .rect(0, barY, drawnW, BAR_HEIGHT)
      .fill({ color, alpha })
      .stroke({ color: COLOR.outline, width: 0.5, alpha: 0.7 });
    bar.x = overlap.x;
    bar.eventMode = 'static';
    bar.cursor = 'pointer';
    bar.hitArea = new Rectangle(0, barY - BAR_HIT_VPAD, drawnW, BAR_HEIGHT + BAR_HIT_VPAD * 2);
    this.attachClickAndHover(bar, item.item_id, item.start_frame, 'item');
    this.regRect(item.item_id, overlap.x, barY, drawnW, BAR_HEIGHT);
    c.addChild(bar);
    return c;
  }

  private drawEntries(): void {
    this.entriesLayer.removeChildren();
    const y = this.layout.entryY;
    for (const item of this.items) {
      if (item.kind !== 'entry') continue;
      if (!this.inRange(item.frame)) continue;
      const g = new Graphics();
      g.circle(0, y, ENTRY_RADIUS).fill({ color: BASELINE_ENTRY_COLOR_INT });
      g.x = this.xOf(item.frame);
      g.label = '';
      g.eventMode = 'static';
      g.cursor = 'pointer';
      g.hitArea = new Circle(0, y, DOT_HIT_RADIUS);
      this.attachClickAndHover(g, item.item_id, item.frame, 'item');
      this.regCircle(item.item_id, this.xOf(item.frame), y, ENTRY_RADIUS);
      this.entriesLayer.addChild(g);
    }
  }

  private drawVectors(): void {
    this.vectorsLayer.removeChildren();
    const pos = this.layout.positions.get('entrance_vectors');
    if (!pos) return;
    for (const item of this.items) {
      if (item.kind !== 'vector') continue;
      if (!this.inRange(item.frame)) continue;
      const color = metricColorInt(item.metric_id);
      const g = new Graphics();
      g.circle(0, pos.mainCenterY, VECTOR_RADIUS).fill({ color });
      g.x = this.xOf(item.frame);
      g.label = item.metric_id;
      g.eventMode = 'static';
      g.cursor = 'pointer';
      g.hitArea = new Circle(0, pos.mainCenterY, DOT_HIT_RADIUS);
      this.attachClickAndHover(g, item.item_id, item.frame, 'item');
      this.regCircle(item.item_id, this.xOf(item.frame), pos.mainCenterY, VECTOR_RADIUS);
      this.vectorsLayer.addChild(g);
    }
  }

  private drawFlags(): void {
    this.flagsLayer.removeChildren();
    for (const flag of this.flagsList) {
      if (flag.frame == null) continue;
      if (!this.inRange(flag.frame)) continue;
      const y = this.layout.flagYByFlagId.get(flag.flag_id);
      // Flag's owning metric isn't in the layout (hidden, or not declared)
      // → don't render. This matches the "flags belong to their metric" rule.
      if (y == null) continue;
      const color = metricColorInt(flag.metric_id);

      const g = new Graphics();
      g.circle(0, y, FLAG_RADIUS)
        .fill({ color })
        .stroke({ color: COLOR.outline, width: 1 });
      g.rect(-1.2, y - FLAG_RADIUS * 0.55, 2.4, FLAG_RADIUS * 0.7)
        .fill({ color: COLOR.white });
      g.circle(0, y + FLAG_RADIUS * 0.45, 1.4)
        .fill({ color: COLOR.white });

      g.x = this.xOf(flag.frame);
      g.eventMode = 'static';
      g.cursor = 'pointer';
      g.hitArea = new Circle(0, y, FLAG_HIT_RADIUS);
      this.attachClickAndHover(g, flag.flag_id, flag.frame, 'flag');
      this.regCircle(flag.flag_id, this.xOf(flag.frame), y, FLAG_RADIUS);
      this.flagsLayer.addChild(g);
    }
  }

  private drawDrillMarkers(): void {
    this.drillMarkersLayer.removeChildren();
    if (!this.drillWindow) return;
    const dw = this.drillWindow;
    const y = this.layout.drillY;

    const drillHit = new Rectangle(
      -DRILL_HIT_HALF,
      -DRILL_HIT_HALF,
      DRILL_HIT_HALF * 2,
      DRILL_HIT_HALF * 2,
    );

    // Clicking a drill diamond selects its synthesized info flag (so the
    // detail panel, flag-list highlight, and selection ring all engage) and
    // seeks — falling back to a plain hover label if the flag is absent.
    const hasStartFlag = this.flagsList.some((f) => f.flag_id === 'drill_start_info');
    const hasEndFlag = this.flagsList.some((f) => f.flag_id === 'drill_end_info');

    if (this.inRange(dw.start_frame)) {
      const start = this.makeDiamond(DRILL_DIAMOND, COLOR.drillStart, false);
      start.y = y;
      start.x = this.edgeX(dw.start_frame);
      start.eventMode = 'static';
      start.cursor = 'pointer';
      start.hitArea = drillHit;
      if (hasStartFlag) {
        this.attachClickAndHover(start, 'drill_start_info', dw.start_frame, 'flag');
        this.regCircle('drill_start_info', this.edgeX(dw.start_frame), y, DRILL_DIAMOND + 1);
      } else {
        this.attachClickAndHover(start, 'drill_start', dw.start_frame, 'drill');
      }
      this.drillMarkersLayer.addChild(start);
    }
    if (this.inRange(dw.end_frame)) {
      const end = this.makeDiamond(
        DRILL_DIAMOND,
        dw.end_uncertain ? COLOR.drillEndUncertain : COLOR.drillStart,
        dw.end_uncertain,
      );
      end.y = y;
      end.x = this.edgeX(dw.end_frame);
      end.eventMode = 'static';
      end.cursor = 'pointer';
      end.hitArea = drillHit;
      if (hasEndFlag) {
        this.attachClickAndHover(end, 'drill_end_info', dw.end_frame, 'flag');
        this.regCircle('drill_end_info', this.edgeX(dw.end_frame), y, DRILL_DIAMOND + 1);
      } else {
        this.attachClickAndHover(end, 'drill_end', dw.end_frame, 'drill');
      }
      this.drillMarkersLayer.addChild(end);
    }
  }

  private makeDiamond(size: number, color: number, hollow: boolean): Graphics {
    const g = new Graphics();
    const points = [0, -size, size, 0, 0, size, -size, 0];
    if (hollow) {
      g.poly(points).stroke({ color, width: 2 });
    } else {
      g.poly(points).fill({ color });
    }
    return g;
  }

  private drawPlayhead(): void {
    this.playhead.clear();
    this.playhead.rect(-1, 0, 2, this.layout.totalHeight).fill({ color: COLOR.playhead });
    this.updatePlayheadPosition();
  }

  private updatePlayheadPosition(): void {
    this.playhead.x = this.xOf(this.currentFrame);
  }

  // -- pointer / hover ------------------------------------------------------

  private attachClickAndHover(
    target: Container | Graphics | Text,
    id: string,
    seekFrame: number,
    kind: HoverKind,
  ): void {
    target.on('pointerdown', (ev: FederatedPointerEvent) => {
      ev.stopPropagation();
      // POD-adjust mode locks every interaction except the marker drag.
      if (this.podAdjust != null) return;
      if (kind === 'item') {
        this.callbacks.onItemClicked?.(id, seekFrame);
      } else if (kind === 'flag') {
        this.callbacks.onFlagClicked?.(id);
      } else {
        this.callbacks.onEmptyClicked?.(seekFrame);
      }
    });
    target.on('pointerover', (ev: FederatedPointerEvent) => {
      this.callbacks.onHover?.({
        kind,
        id,
        clientX: ev.client.x,
        clientY: ev.client.y,
      });
    });
    target.on('pointerout', () => {
      this.callbacks.onHover?.(null);
    });
  }

  private onContainerLeave = (): void => {
    this.callbacks.onHover?.(null);
  };

  private onStagePointerDown = (ev: FederatedPointerEvent): void => {
    if (this.podAdjust != null) {
      // Adjust mode: remember the press so a tap (no drag) can place the
      // marker at the tapped frame — much easier than grabbing a marker
      // parked at the far edge of the timeline.
      this.stagePointerDown = {
        x: ev.client.x,
        y: ev.client.y,
        frame: this.rangeStart + Math.round(ev.global.x / this.pixelsPerFrame),
      };
      return;
    }
    const localX = ev.global.x;
    const frame = this.rangeStart + Math.round(localX / this.pixelsPerFrame);
    // Defer the seek until pointerup so a drag-to-pan doesn't trigger one.
    // Empty-track clicks from interactive shapes never reach this handler
    // (their handlers stopPropagation), so this only fires on bare clicks.
    this.stagePointerDown = { x: ev.client.x, y: ev.client.y, frame };
  };

  /**
   * POD-marker drag uses NATIVE window listeners, not Pixi stage events:
   * Pixi v8's stage `pointermove` is hit-tested (it only fires over drawn
   * display objects), so a drag would stall over empty track areas and stop
   * entirely outside the canvas. Window-level tracking is continuous.
   */
  private startPodDrag(): void {
    if (this.podAdjust == null || this.podDragging) return;
    this.podDragging = true;
    window.addEventListener('pointermove', this.onWindowPodDragMove);
    window.addEventListener('pointerup', this.onWindowPodDragEnd, { once: true });
    window.addEventListener('pointercancel', this.onWindowPodDragEnd, { once: true });
  }

  private clampAdjustFrame(rawFrame: number): number {
    // The drill end can never sit at/before the drill start.
    const minFrame = this.podAdjust?.kind === 'drill_end' && this.drillWindow
      ? Math.max(1, this.drillWindow.start_frame + 1)
      : 1;
    return clamp(
      rawFrame,
      Math.max(minFrame, Math.floor(this.rangeStart)),
      Math.min(this.totalFrames, Math.ceil(this.rangeEnd)),
    );
  }

  private setAdjustPreview(frame: number): void {
    if (this.podAdjust == null || frame === this.podAdjust.frame) return;
    this.podAdjust = { frame, kind: this.podAdjust.kind };
    this.redraw();
    this.callbacks.onPodDragPreview?.(frame);
  }

  /** Horizontally center the view on a frame (no-op when nothing scrolls). */
  centerOnFrame(frame: number): void {
    const c = this.container;
    if (!c) return;
    const x = (frame - this.rangeStart) * this.pixelsPerFrame;
    c.scrollLeft = Math.max(0, x - c.clientWidth / 2);
  }

  private onWindowPodDragMove = (ev: PointerEvent): void => {
    if (!this.podDragging || this.podAdjust == null) return;
    const rect = this.app.canvas.getBoundingClientRect();
    // autoDensity keeps CSS px == stage px, so canvas-local x maps directly.
    const localX = ev.clientX - rect.left;
    this.setAdjustPreview(
      this.clampAdjustFrame(this.rangeStart + Math.round(localX / this.pixelsPerFrame)),
    );
  };

  private onWindowPodDragEnd = (): void => {
    this.podDragging = false;
    window.removeEventListener('pointermove', this.onWindowPodDragMove);
  };

  private onStagePointerUp = (ev: FederatedPointerEvent): void => {
    if (this.podDragging) {
      return;
    }
    if (this.podAdjust != null) {
      const down = this.stagePointerDown;
      this.stagePointerDown = null;
      if (down) {
        const dx = ev.client.x - down.x;
        const dy = ev.client.y - down.y;
        if (Math.hypot(dx, dy) <= TAP_TOLERANCE_PX) {
          this.setAdjustPreview(this.clampAdjustFrame(down.frame));
        }
      }
      return;
    }
    const down = this.stagePointerDown;
    this.stagePointerDown = null;
    if (!down) return;
    const dx = ev.client.x - down.x;
    const dy = ev.client.y - down.y;
    if (Math.hypot(dx, dy) > TAP_TOLERANCE_PX) return;
    if (this.inRange(down.frame)) {
      this.callbacks.onEmptyClicked?.(down.frame);
    }
  };
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}
