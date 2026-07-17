# Pose Model Backends

The pose stage of the pipeline is configurable. Detection is always the same
first step (the project's fine-tuned **RTMDet-m** person detector); the pose
model that runs on the detected boxes is selected per run via the config.
Everything downstream — tracking, positioning, metrics, caches, overlays, the
Analysis Viewer — behaves identically for every backend because all backends
emit the same **canonical Halpe-26 keypoint layout** (see
[Canonical keypoint contract](#canonical-keypoint-contract)).

All models are from one family — **RTMDet + RTMPose (SimCC head, CSPNeXt
backbone)** — served by the in-repo, OpenMMLab-free `libs/giftpose` runtime.
One backbone implementation, one head style, one codec, shared preprocessing
and export tooling; no new dependencies.

## Config keys

```json
{
  "pose_backend": "body2d",          // "body2d" | "wholebody" | "pose3d"
  "pose_model_size": "x",            // size within the backend (see tables)
  "auto_download_models": false,      // fetch official checkpoints on demand
  "pose_backend_weights": null,       // optional explicit weights override
  "facing_metadata": true             // KGF facing sidecar (metadata only)
}
```

Omitting every key reproduces the legacy behavior exactly (fine-tuned
RTMPose-x Halpe26). `pose2d_config` / `pose2d_weights` remain the explicit
legacy override for the default backend and always win when the new keys are
absent.

## Backends and registered architectures

### `body2d` (default) — 26-kp Halpe26 body pose

| Size | Tag | Input | Checkpoint | Disk |
|---|---|---|---|---|
| **x (default)** | `rtmpose-x-halpe26-384x288` | 384×288 | **fine-tuned `models/pose.pth`** (project data: helmets, camo, room geometry) | 191 MB |
| x (official) | same tag via auto-download | 384×288 | `rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288` | 191 MB |
| l | `rtmpose-l-halpe26-256x192` | 256×192 | `rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192` | 108 MB |
| m | `rtmpose-m-halpe26-256x192` | 256×192 | `rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192` | 53 MB |
| s | `rtmpose-s-halpe26-256x192` | 256×192 | `rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192` | 22 MB |
| t | `rtmpose-t-halpe26-256x192` | 256×192 | `rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192` | 13 MB |

### `wholebody` — 133-kp COCO-WholeBody (RTMW)

Adds 6 feet + 68 face + 42 hand keypoints in the **same single pass** (no
extra models, no extra detection). The canonical 26-kp block is derived by
the adapter; the full 133-kp set is carried per object in
`keypoints_wb` / `keypoint_scores_wb`.

| Size | Tag | Input | Checkpoint | Disk | Params |
|---|---|---|---|---|---|
| x | `rtmw-x-cocktail14-133` | 384×288 | `rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288` | 353 MB | 92 M |
| l | `rtmw-l-cocktail14-133` | 384×288 | `rtmw-dw-x-l_simcc-cocktail14_270e-384x288` (distilled) | 220 MB | — |
| m | `rtmw-m-cocktail14-133` | 256×192 | `rtmw-dw-l-m_simcc-cocktail14_270e-256x192` (distilled) | 124 MB | — |

### `pose3d` — 133-kp 3D (RTMW3D)

Single-frame top-down 3D: same architecture as RTMW plus a third SimCC
classifier for depth. Emits the canonical 2D block exactly like `wholebody`,
plus per-keypoint **root-normalized metric z** in `keypoints_z`
(z = (bins/(D/2) − 1) × 2.1744869, hips-root convention from upstream).
Map positioning is unchanged (ankles + homography remain the floor-plane
source of truth); z is informational metadata.

| Size | Tag | Input | Checkpoint | Disk | Params |
|---|---|---|---|---|---|
| l | `rtmw3d-l-cocktail14-133` | 384×288 (z bins 288) | `rtmw3d-l_8xb64_cocktail14-384x288` | 220 MB | 58 M |

Implementation notes:
- Detections are **batched per frame** exactly like the 2D backends (all
  person crops in one stacked forward).
- Keypoint scores use the upstream convention `min(max_x, max_y)` — the z
  branch is excluded (its softmax-trained logits are an order of magnitude
  smaller and would starve downstream confidence thresholds).
- `pose3d` runs additionally save `{basename}_Pose3D_Skeletons.mp4`: a 3D
  matplotlib plot video of every tracked skeleton, color-coded by track id
  (x = image x, y = metric depth z, z = height). Inspection artifact, not an
  AAR overlay; disable with `"annotate_pose3d_plot": false`.

## Auto-download

With `"auto_download_models": true`, any registered architecture whose
weights are missing is fetched once from `download.openmmlab.com` into
`models/` (atomic write, size-checked) and reused afterwards. Without it, a
missing checkpoint raises an error containing the exact `curl` command. The
fine-tuned project weights (`models/pose.pth`, `models/detect-best-mAP.pth`)
are never downloaded — they ship with the project. URLs were verified live
on 2026-07-16; the registry (`libs/giftpose/registry.py`) is the single
source of truth.

## Canonical keypoint contract

Immediately after pose inference — before the tracker — every backend's
output is adapted to the Halpe-26 layout (`libs/giftpose/adapters.py`):

- `body2d`: identity (zero cost — the default path is unchanged).
- `wholebody` / `pose3d` (COCO-WholeBody 133):
  - body 0–16 → identical indices (both layouts use COCO body order);
  - Halpe 17 head / 18 neck / 19 hip synthesized (face centre, shoulder
    midpoint, hip midpoint; score = min of parents);
  - feet remapped: WB 17,18,19,20,21,22 (LBigToe, LSmallToe, LHeel,
    RBigToe, RSmallToe, RHeel) → Halpe 20,22,24,21,23,25.

Consequences: the tracker's `keypoint_indices [15,16]` (ankles), the gaze
`gaze_keypoint_map` (0–4), skeleton edges, shoulder/elbow/wrist metric
indices and every `== 26` gate hold for all backends **without any config
change**. Score stays in column 2 of the canonical block everywhere; z is
never interleaved. Extended data rides in optional per-object TrackerOutput
fields (`keypoints_wb`, `keypoint_scores_wb`, `keypoints_z`) written only
when the backend produces them.

## Facing metadata (KGF)

For every backend, a per-frame per-track facing estimate is computed from
the canonical keypoints (`src/facing.py`, the KGF v2 estimator validated in
the July 2026 study) and saved as **metadata only** — no overlay, metric, or
analysis change:

- `{basename}_FacingCache.txt`: `frame,id,yaw_cam,back,map_bearing,door_rel,conf`
- per-object `facing: [yaw_cam, back, map_bearing, door_rel, conf]`

`yaw_cam` is camera-relative (0 = toward camera). `map_bearing` is in map
coordinates via the homography — **camera-placement invariant**. `door_rel`
is the cosine against the door's inward normal (+1 = facing into the room,
−1 = facing back out the door) — **invariant to camera-vs-door orientation**.
Disable with `"facing_metadata": false`.

## Visualization policy (AAR)

Rendered overlay videos follow the selected model: the default `body2d`
backend draws the canonical-26 skeleton exactly as before, while
`wholebody` / `pose3d` runs draw the **full 133-keypoint skeleton** in the
tracking overlay — body + feet + hand finger links (edges from the
COCO-WholeBody metainfo, hands attached at the wrists) with the 68 face
landmarks as fine dots. Gaze triangles and map renderings are identical
across backends. `pose3d` additionally saves the 3D skeleton plot video.

## Exports / runtime backends

Runtime autoselect order is unchanged (TensorRT → ONNX → TorchScript →
PyTorch per device). Export tools accept the registry tag:

```bash
python -m libs.giftpose.export.torchscript_export --pose-tag rtmw-x-cocktail14-133 \
    --pose-weights models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288.pth --device cpu --verify
python -m libs.giftpose.export.onnx_export --pose-tag rtmw-l-cocktail14-133 ...
```

Notes:
- `pose3d` (3 SimCC outputs) runs on the PyTorch backend by default; ONNX/TS
  export emits the third `pred_z` output, but the artifact runtimes currently
  decode 2 outputs — pass `prefer_backend="pytorch"` (default behavior for
  3D specs) unless you extend them.
- TensorRT builds are CUDA-only; on macOS autoselect falls back gracefully.

## Choosing a backend

- **Assessments / production**: keep the default (`body2d` x, fine-tuned).
  It is the only checkpoint trained on this project's footage (helmets,
  camo, these rooms) and drives all validated metric scores.
- **Faster experiments / weaker hardware**: `body2d` with `m` or `s` sizes.
- **Face/hand keypoint research**: `wholebody` (one pass, 133 kp saved per
  object). Zero-shot — expect lower body-keypoint quality than the
  fine-tuned default on this footage.
- **Depth-aware research**: `pose3d` (adds metric z). Same zero-shot caveat;
  slowest option.
- Fine-tuning RTMW/RTMW3D on project data is the upgrade path if wholebody
  or 3D ever needs production quality; plug the result in via
  `pose_backend_weights`.

## Measured on this Mac (Apple Silicon) — full-benchmark, July 16 2026

Pipeline loop fps (decode + detect + pose + track), test video 3-TrimmedV2.
PyTorch/TorchScript on MPS; ONNX on CPU (autoselect skips ONNX on MPS).

| Config | PyTorch-MPS | TorchScript-MPS | ONNX-CPU |
|---|---|---|---|
| body2d x (fine-tuned default) | 10.0 | 11.3 | 3.6 |
| body2d l / m / s / t | 21.0 / 22.9 / 20.6 / 12.9 | 23.7 / 25.3 / 26.7 / 13.8 | 5.3 / 5.9 / 6.5 / 6.6 |
| wholebody m / l / x | 18.9 / 16.7 / 10.4 | 11.6 / 9.4 / 9.3 | 5.2 / 3.4 / 2.7 |
| pose3d l | 8.9 | 17.9 | 3.4 |

Original EAE (pre-branch code, default config): 9.7 fps — the branch default
matches it within noise and reproduces its metric scores exactly.
TorchScript-MPS is the fastest deployment for most configs. Zero-shot
caveat: the biggest zero-shot models (wholebody-x, pose3d-l) degrade
tracking-derived metric scores on full-length runs of this footage —
fine-tuning via ``pose_backend_weights`` is the production path for them.

## Regression guarantees

The default configuration is regression-gated: engine runs on
`input/test.vmeta.xml` must reproduce the frozen baseline metric scores
(ENTRANCE_VECTORS 1.0, ENTRANCE_HESITATION 1.0, TOTAL_TIME_OF_ENTRY 1.0,
STAY_ALONG_WALL 0.93 at the baseline commit) with identical tracking output.
Every non-default backend is smoke-gated end-to-end (engine → tracker →
metrics → overlays → viewer-loadable run folder).
