// Single source of truth for tokens that appear in BOTH the PixiJS canvas
// (which wants 24-bit ints) and CSS (which wants hex strings).
// Keep in lock-step with src/theme/theme.css (the CSS-side tokens consumed
// via custom properties).

const HEX = {
  metricVectors: '#5eead4', // teal — distinct from per-direction chevron colors
  metricHesitation: '#60a5fa',
  metricTotalEntry: '#a78bfa',
  metricMoveAlongWall: '#f472b6', // pink — visually distinct from the other three
  metricUnknown: '#9ca3af',

  baselineEntry: '#e5e7eb',

  severityWarning: '#fdba74',
  severityError: '#ef4444',
  severityInfo: '#93c5fd',
} as const;

export const METRIC_COLOR: Record<string, string> = {
  entrance_vectors: HEX.metricVectors,
  entrance_hesitation: HEX.metricHesitation,
  total_time_of_entry: HEX.metricTotalEntry,
  move_along_wall: HEX.metricMoveAlongWall,
  pod_sector_coverage: '#c084fc',
  pod_mutual_facing: '#fbbf24', // amber — the wall metric already owns the pink
};

export function metricColor(metricId: string | null | undefined): string {
  if (!metricId) return HEX.metricUnknown;
  return METRIC_COLOR[metricId] ?? HEX.metricUnknown;
}

export function metricColorInt(metricId: string | null | undefined): number {
  return hexToInt(metricColor(metricId));
}

export const SEVERITY_COLOR: Record<string, string> = {
  warning: HEX.severityWarning,
  error: HEX.severityError,
  info: HEX.severityInfo,
};

export const BASELINE_ENTRY_COLOR_INT = hexToInt(HEX.baselineEntry);
export const NEUTRAL_DOT_COLOR_INT = hexToInt(HEX.baselineEntry);

// Per-track palette — must mirror the engine's track palette in
// ``helper_functions._build_track_color_cache`` so the analysis-viewer
// overlays match the colors users already see in motion_map / gaze_map
// artifact videos. The engine stores cv2 BGR tuples and indexes by
// ``track_id % len``; we mirror that here as CSS hex (RGB).
//
// Engine BGR → CSS hex mapping (do not reorder — tracks are colored
// by index):
//   (255,   0,   0) → #0000ff  blue
//   (  0, 255,   0) → #00ff00  green
//   (  0,   0, 255) → #ff0000  red
//   (255, 255,   0) → #00ffff  cyan
//   (255,   0, 255) → #ff00ff  magenta
//   (  0, 255, 255) → #ffff00  yellow
//   (128,   0,   0) → #000080  dark blue
//   (  0, 128,   0) → #008000  dark green
//   (  0,   0, 128) → #800000  dark red
//   (128, 128,   0) → #008080  teal
//   (128,   0, 128) → #800080  purple
//   (  0, 128, 128) → #808000  olive
export const TRACK_COLOR_PALETTE: ReadonlyArray<string> = [
  '#0000ff', // 0: blue
  '#00ff00', // 1: green
  '#ff0000', // 2: red
  '#00ffff', // 3: cyan
  '#ff00ff', // 4: magenta
  '#ffff00', // 5: yellow
  '#000080', // 6: dark blue
  '#008000', // 7: dark green
  '#800000', // 8: dark red
  '#008080', // 9: teal
  '#800080', // 10: purple
  '#808000', // 11: olive
];

/**
 * Track colors in the engine match ``track_id % palette.length``.
 * In-room IDs are drawn white in artifacts; pass ``isInRoom = true`` to
 * surface that here too. Falsy / non-finite track ids fall back to a
 * neutral grey.
 */
export function colorForTrackId(
  trackId: number | null | undefined,
  isInRoom = false,
): string {
  if (isInRoom) return '#ffffff';
  if (trackId == null || !Number.isFinite(trackId)) {
    return HEX.metricUnknown;
  }
  const idx = Math.abs(Math.trunc(trackId)) % TRACK_COLOR_PALETTE.length;
  return TRACK_COLOR_PALETTE[idx];
}

export function hexToInt(hex: string): number {
  return parseInt(hex.replace('#', ''), 16);
}

// -- Spacing / radius / font tokens (mirrors theme.css custom properties) --

export const SPACING = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  '2xl': 32,
} as const;

export const RADIUS = {
  sm: 3,
  md: 4,
  lg: 6,
  xl: 8,
} as const;

export const FONT_SIZE = {
  xs: 10,    // body 14px * 0.72 ≈ 10px (tooltip / chip)
  sm: 12,    // body 14px * 0.85 ≈ 12px
  md: 14,
  lg: 16,
} as const;
