# GIFT Multimodal EAE

Python integration engine for processing session videos using metadata in a `.vmeta.xml` file. Person detection (RTMDet-m) + pose estimation (RTMPose-x, Halpe-26) run from the in-repo [`libs/giftpose/`](libs/giftpose/) library — pure PyTorch / numpy / opencv, no OpenMMLab dependencies at runtime.

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Desktop tools — the setup pipeline](#desktop-tools--the-setup-pipeline)
  - [1. Scaled Point Viewer](#1-scaled-point-viewer)
  - [2. Mapper tool](#2-mapper-tool)
  - [3. Config Builder tool](#3-config-builder-tool)
  - [5. Analysis Viewer](#5-analysis-viewer)
- [Usage](#usage)
- [Vmeta format](#vmetaxml-format)
- [Config file](#config-file-spaceenvironment-metadata)
- [Running the engine](#running-the-engine-backend)
- [Running locally](#running-the-local-engine-without-gift)

## Requirements

- **Conda (recent version)** — Anaconda, Miniconda, or Miniforge (recommended ≥ 24.x).

## Installation

```bash
git clone https://github.com/Surya-Rayala/GIFT-Multimodal-EAE-ECR.git
cd GIFT-Multimodal-EAE-ECR
conda env create --file environment.yml            # macOS / Linux
conda env create --file environment.windows.yml    # Windows (Python stack from pip; conda only for python/git/ffmpeg/rust — avoids win-64 DLL clashes)
conda activate gift-meae
```

> If `Solving environment` hangs:
> ```bash
> conda install -n base conda-libmamba-solver -y
> conda config --set solver libmamba
> ```

Drop your fine-tuned `models/pose.pth` and `models/detect-best-mAP.pth` (or any other names you point at via the vmeta config) into `models/`. Mmengine optimizer / EMA / metadata blobs are stripped in-memory at load time — no preprocessing step is required.

### Hugging Face token

The transcription pipeline pulls the NeMo Parakeet-TDT checkpoint (`nvidia/parakeet-tdt-0.6b-v2`, ~2.4 GB) from Hugging Face on first run, so a token is required. See [TRANSCRIPTION.md](TRANSCRIPTION.md) for the ASR pipeline details.

Generate one at <https://huggingface.co/settings/tokens> (read scope is enough), then store it once with the CLI:

```bash
conda activate gift-meae
hf auth login
# paste the hf_xxxxxxxxxxxxxxxxxxxxxxxx token when prompted
```

This writes `~/.cache/huggingface/token`; every subsequent run picks it up automatically. Verify:

```bash
python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"
```

Prints your Hugging Face username + email when the token is valid.

### CUDA users

The conda env's PyTorch is **not guaranteed to be a CUDA build**, so verify and fix it.

1. Make sure the CUDA Toolkit is installed, then check its version:

   ```bash
   nvcc --version    # the installed CUDA Toolkit version, e.g. "release 12.6"
   ```

   If `nvcc` isn't found, install the CUDA Toolkit from <https://developer.nvidia.com/cuda-downloads>. (Note: `nvidia-smi` shows only the *driver's* max-supported CUDA, not the installed toolkit — pick your `cu###` wheel below by `nvcc`, and it must not exceed the driver's CUDA in `nvidia-smi`.)

   **Preferred: a CUDA 12.x toolkit (12.6, 12.8, or 12.9)** — torch 2.8 ships wheels exactly for these (`cu126` / `cu128` / `cu129`). If your toolkit is **newer than 12.9** (e.g. CUDA 13.x), there's no matching torch 2.8 wheel — install the **nearest lower** one, `cu129`. (It runs fine: the wheel bundles its own CUDA runtime and the driver is backward-compatible.)

2. Confirm PyTorch actually sees the GPU:

   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
   ```

   If this prints `True` and a CUDA version, you're done — skip to ONNX/TensorRT below.

3. If it prints `False` / `None`, install the CUDA wheel **for the version this repo pins** — `torch 2.8.*` / `torchvision 0.23.*` / `torchaudio 2.8.*` (keep this version; a mismatch breaks the torchvision ABI, and torch 2.8 satisfies NeMo's `torch>=2.6`). Uninstall the current build, then from <https://pytorch.org/get-started/previous-versions/> grab the **v2.8.0** command whose `cu###` matches your `nvcc --version` toolkit:

   Pick the `cu###` for your toolkit: **12.6 → `cu126`, 12.8 → `cu128`, 12.9 (or anything newer, e.g. 13.x) → `cu129`** (the highest torch 2.8 ships).

   ```bash
   pip uninstall -y torch torchvision torchaudio
   # swap cu129 below for cu126 / cu128 to match YOUR `nvcc --version`:
   pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
   ```

For the optional ONNX-CUDA backend, swap `onnxruntime` for `onnxruntime-gpu`:

```bash
pip install --force-reinstall onnxruntime-gpu
```

For the TensorRT backend (`*.engine`), install `tensorrt` matching your CUDA version per NVIDIA's pip guide: <https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html>.

> **Precision:** prefer **fp32** engines (`trt_build --no-fp16`, see below) — fp16 trades accuracy for speed and has been observed to produce worse detections/pose on this model. If you *do* opt into fp16 on **TensorRT 10.12+**, it needs `onnxconverter-common` (newer TensorRT dropped the `FP16` builder flag for *strong typing*, so the fp16 engine is built from an fp16 ONNX); without it the build logs `Building an fp32 engine instead` and falls back to fp32 anyway:
> ```bash
> pip install onnxconverter-common
> ```

### Optional — accelerated runtime artifacts

The engine runs straight off `*.pth`. To upgrade to a faster backend, export once:

```bash
# ONNX 
python -m libs.giftpose.export.onnx_export

# TorchScript 
python -m libs.giftpose.export.torchscript_export --device cpu  --verify
python -m libs.giftpose.export.torchscript_export --device mps  --verify
python -m libs.giftpose.export.torchscript_export --device cuda --verify   # NVIDIA

# TensorRT (NVIDIA only, additional speedup) — auto-exports ONNX from .pth if the .onnx artifacts aren't already present.
# Build fp32 engines (--no-fp16): fp16 can degrade detection/pose quality (spurious or
# misplaced boxes). Drop --no-fp16 only if you've verified fp16 accuracy on your data.
python -m libs.giftpose.export.trt_build --no-fp16
```

The runtime auto-selects per device:

| device | order |
|---|---|
| cuda | TensorRT (`*.engine`) → ONNX-CUDA-EP → TorchScript → PyTorch |
| cpu  | ONNX-CPU-EP → TorchScript → PyTorch |
| mps  | TorchScript-MPS → PyTorch (ONNX/CoreML skipped — graph fragments) |

Confirm which backend gets picked **for a given device** — the choice is
device-specific, so pass the device you actually run on (`cpu`, `cuda`, or `mps`):

```bash
# replace `mps` with `cpu` or `cuda` for your machine
python -c "import sys; from libs.giftpose.runtime.autoselect import select_backend; print(type(select_backend('models/detect-best-mAP.pth', 'models/pose.pth', device=sys.argv[1])).__name__)" mps
```

## Desktop tools — the setup pipeline

Authoring a scenario is a short pipeline. Run the tools in this order; each one
feeds the next:

1. **Scaled Point Viewer** — `python -m src.utils.scaled_point_viewer`
   Place real-world reference points on the room map from wall measurements.
2. **Mapper tool** — `python -m src.utils.mapper_viewer`
   Align the camera to the map, and mark entry zones, POD targets, and the room boundary.
3. **Config Builder** — `python -m src.utils.config_builder_app`
   Bundle the mapper outputs + scoring knobs into a ready-to-use `config.json`.
4. **Run the engine** — `run_engine_local.py` (local) / `run_engine.py` (server) — see [Usage](#usage).
5. **Analysis Viewer** — `python -m src.utils.analysis_viewer`
   View and compare the results.

### Node.js is auto-installed

The three desktop apps (Scaled Point Viewer, Mapper, Analysis Viewer) are
Tauri + Vue apps. **You don't need to install Node.js yourself** — on first
launch each app reuses a system Node ≥ 20 if present, otherwise downloads a
small portable Node into `~/.cache/gift-meae/node/` and installs the frontend
packages once. The first launch can take a minute (download + build); later
launches start in seconds. The three apps share a single hoisted `node_modules`
and one Rust build cache (`.build/`) so they don't each consume gigabytes.

### What gets committed to git

Only source is tracked. The generated artifacts — `node_modules/`, `dist/`,
`src-tauri/target/`, `.build/`, and the portable-Node cache — are git-ignored and
rebuilt on first launch. Data and weights are not shared either: `input/` ships
empty and `output/`, `models/`, `work_dirs/`, and `tests/` are ignored. A fresh
clone therefore stays lightweight; the desktop apps regenerate what they need.

## 1. Scaled Point Viewer

Standalone desktop app (Tauri + Vue 3 + FastAPI sidecar) that places **accurate
reference points on a map** when all you know is each point's **real-world
distance to its two nearest walls** and those **walls' real lengths**. The saved
points are later used to set up the camera↔map homography in the Mapper.

```bash
conda activate gift-meae
python -m src.utils.scaled_point_viewer
```

### How it works (in plain English)

You don't enter pixel coordinates — you draw the walls and type real measurements:

1. **Setup** — pick the map image, a save folder, a project name, and your unit (ft/m/in).
2. **Draw Walls** — click two points to draw each wall, then type its real length.
   The tool turns all your walls into **one consistent map scale** (pixels per
   real unit). Using several walls — and longer ones — makes that scale accurate,
   so a point stays correct even when one wall is short or **cut by a corridor**.
3. **Place Points** — click a wall (Wall A), then the adjacent wall at the same
   corner (Wall B), and type the point's distance from each. An orange preview
   shows where it lands; **Add point** commits it. Oblique (non-90°) corners are
   handled correctly, not just square ones.

It saves three files (named with your project prefix): `<name>_scaled_points.txt`
(map-pixel coordinates), `<name>_scaled_points.png` (the map with points drawn),
and `<name>_scaler_project.json` (reload/edit later). It replaces the older PyQt5
`scaled_point_mapper_app`, fixing the corridor / short-wall scaling problems.

### How to take the measurements

```
                 wall A  (e.g. 20 ft long)
        ┌───────────────────────────────────
        │              ·  ← your point P
        │              ╎
 wall B │   distFromB  ╎ distFromA      distFromA = straight-line (perpendicular)
 (16 ft)│  ┄┄┄┄┄┄┄┄┄┄┄┄·                  distance from P to wall A
        │              ╎                 distFromB = perpendicular distance from
        │              ╎                  P to wall B
      corner where wall A and wall B meet
```

- For each point, use its **two nearest adjacent walls** — the two walls of the
  corner it sits closest to.
- Each distance is the **perpendicular (shortest, straight-out) distance** from the
  point to that wall — measured at a right angle to the wall. It is **not** the
  diagonal to the corner, and not measured along the floor at an angle.
- **Keep units consistent.** Use the same unit (ft, m, or in) for every wall
  length and every distance.
- Stand inside the room; distances are positive. The two walls **don't** have to
  meet at 90° — oblique corners work, as long as you give each wall's true length
  and the two perpendicular distances.

## 2. Mapper tool

Cross-platform desktop app that helps you author the four per-room files the config builder needs (Tauri shell + Vue 3 frontend + FastAPI Python sidecar). Walks you through Setup → Align Camera to Map → Mark Entry Zones → Mark POD Targets → Outline the Room as a 5-step wizard, with contextual help on each step.

The four output files (named with your project prefix):

- `*_mapping.txt` — camera↔map point pairs (homography) → `point_mapping_path`
- `*_entry_polygons.txt` — entry/door regions on the map → `entry_polys_path`
- `*_POD_points.txt` — single points on the map → `POD`
- `*_room_boundary.txt` — outline of the walkable/trackable area → `Boundary`

### What it’s for (in plain English)

The engine takes people detected in the **camera video** and places them onto a **top‑down map** of the room. This mapper teaches the system how to translate between those two views by:

1) picking matching points in the camera view and the map (so the system knows how to line them up), and
2) drawing the important regions on the map (where the room boundary is, where entries are, and POD locations).

### Run

```bash
conda activate gift-meae
python -m src.utils.mapper_viewer
```

A native window opens. **Node.js is auto-installed** on first launch (see
[Node.js is auto-installed](#nodejs-is-auto-installed)) and the frontend packages
install once into the shared workspace `node_modules`; later launches start in
seconds. The Rust shell spawns the Python sidecar internally and tears it down on
window close — no other terminals to manage.

### Basic workflow

1) **Setup** — pick the map image, a camera reference (a video — then scrub to the frame you want — or a single frame image), the save folder, and a short project name.

2) **Align Camera to Map** — alternate clicks: a recognizable spot in the **Camera** view, then the matching spot on the **Map**. Aim for 4+ well-spread pairs. Drag any existing point to fine-tune. **Save this step** writes `<project>_mapping.txt`.

3) **Mark Entry Zones** — on the **Map**, click vertices around each door/entry. Press **Confirm polygon** to close (don’t draw the closing edge yourself). Press **New polygon** for additional zones. **Save this step** writes `<project>_entry_polygons.txt`.

4) **Mark POD Targets** — on the **Map**, click to drop each POD pin. **Save this step** writes `<project>_POD_points.txt`.

5) **Outline the Room** — on the **Map**, click vertices along the walls (CW or CCW). Press **Confirm boundary** when finished. **Save this step** writes `<project>_room_boundary.txt`.

You can navigate freely between steps via the progress bar at the top. The Next button is gated only on Setup (mandatory fields). Each canvas supports pan (drag), zoom (Ctrl + scroll), and point drag.

## 3. Config Builder tool

This repository also includes a **GUI config builder** that helps you assemble a valid `config.json` for a scenario without hand-editing JSON.

It is designed to work together with the **Mapper tool**:

- Mapper produces the `.txt` files (POD points, boundary polygon, entry polygons, and point mapping)
- Config Builder loads those files, lets you adjust the main scoring/behavior knobs, previews everything on the map, and saves a ready-to-use `config.json`

### What it’s for (in plain English)

If the mapper tool is where you *draw* and *measure* the room, the config builder is where you *bundle those results* into one clean “settings file” (`config.json`) that the engine can read.

### How to open

```bash
python -m src.utils.config_builder_app
```

### What you need before using it

You’ll typically generate these with the **Mapper tool** and keep them in your Input Folder (often under `Misc-Data/`):

- POD points file (`*_map_points.txt` or similar)
- Room boundary file (`*_room_boundary.txt`)
- Entry polygons file (`*_entry_polygons.txt`)
- Point mapping file (`*_mapping.txt`)
- Map image (`.png/.jpg`) used for overlays

### Required inputs (cannot save without these)

The app will not enable **Save** until all required items are selected:

- `POD` points (loaded from a points `.txt`)
- `Boundary` polygon (loaded from a boundary `.txt`)
- `MapPath` (map image)
- `point_mapping_path` (mapping `.txt`)
- `entry_polys_path` (entry polygons `.txt`)
- **Project Root folder** (where `config.json` will be saved)

### Basic workflow

1) **Set Project Root**
   - This is the folder where `config.json` will be saved.
   - The builder will try to store selected file paths as **relative paths** when possible.

2) **Load core inputs (Required)**
   - Pick the POD points file, boundary file, map image, point mapping file, and entry polys file.

3) **Adjust main knobs (Main tab)**
   - These control scoring/behavior such as visual angle, threat interaction time, entry time threshold, POD working radius, POD capture threshold, coverage time threshold, and the stay-along-wall setting.
   - **Per‑POD time limits** auto-expand to match the number of PODs.

4) **Use the live preview**
   - The map preview shows:
     - boundary polygon
     - POD points with labels
     - POD working-radius circles
     - a “wall band” visualization related to the stay‑along‑wall metric
   - You can drag POD points in the preview; the builder will update its internal values (and may also rewrite the source POD `.txt` file if it can).

5) **Advanced tab (optional)**
   - Change model paths, thresholds, device selection, gaze keypoints, enemy tracking settings, etc.
   - Defaults are pre-filled so most users don’t need to edit this.

6) **Save `config.json`**
   - Click **Save config.json…** once all required inputs are provided.
   - The saved JSON includes an `_comments` section explaining each field.

### Tips

- Keep the coordinate systems consistent:
  - POD points + boundary + entry polygons must be in the same pixel coordinate system as the map image.
  - The point mapping file must map from **video pixels** to **map pixels** for that same map image.
- If you move your Input Folder, re-open the builder and re-save so relative paths remain valid.


## 5. Analysis Viewer

Cross-platform desktop viewer for `{basename}_Analysis.json` (Tauri shell + Vue 3 frontend + FastAPI Python sidecar). Replaces the legacy PyQt5 viewer.

A **Compare** toggle in the central pane swaps the video stage for a side-by-side comparison view: pick any other run from the same outputs folder (expert runs are flagged automatically), pick a metric, and see your score vs the reference, the saved per-metric visualizations, and a plain-English summary. Grading is relative to the chosen reference run — not an absolute threshold — because what counts as "good" varies room-to-room.

> **Node.js is auto-installed** on first launch — see [Node.js is auto-installed](#nodejs-is-auto-installed). No manual Node setup is needed.

### Run

```bash
conda activate gift-meae

# Open the viewer (use the in-app Open... button to pick a run folder)
python -m src.utils.analysis_viewer

# Or auto-load a run folder on startup
python -m src.utils.analysis_viewer output/trainee_Test_Video_20260521_165915/
```

Each run folder is one of the per-run directories the engine writes under `output/` (it contains a `RunInfo.json` manifest plus the Analysis.json, caches, and rendered videos). The Compare-mode dropdown lists every sibling run folder in the same outputs root.

A native window opens. Node.js is auto-installed on first launch and frontend packages install once into the shared workspace `node_modules`; later launches start in seconds. The Rust shell spawns the Python sidecar internally and tears it down on window close — no other terminals to manage.

### Host it on the network (multiple viewers, one machine)

Instead of the desktop window, you can host the viewer from the machine where the
output folders live, and let people open sessions from their own laptops/tablets
on the same network — each viewing a different session at the same time.

```bash
conda activate gift-meae
python -m src.utils.analysis_viewer --serve --outputs-root /path/to/output
# options: --host (default 0.0.0.0) · --port (default 8000) · --rebuild
```

This builds the frontend once and runs a web server (no desktop window). It prints
the URLs to share. To open a specific session, append its **host-machine folder
path** to the URL.

#### Reading the URL

```
http://<host-ip>:8000/analysis/?run=/path/to/output/run1
└─┬─┘  └──┬───┘ └─┬┘ └───┬────┘└─┬─┘└────────┬─────────┘
scheme   host    port   view    query   run folder path (on host)
```

| Part | Meaning |
|---|---|
| `http://` | plain HTTP (no certificate) — intended for a trusted LAN |
| `<host-ip>` | the **serving machine's** IP on the network (the `--serve` command prints it, e.g. `192.168.1.42`). From the host itself you can also use `127.0.0.1`. |
| `:8000` | the `--port` the server is listening on |
| `view` | which view to open: nothing (just `/`) or `/analysis/` → **Analysis**; `/compare/` → **Compare** (lands with the reference dropdown ready). A deep-linked view wins over the last-used mode. |
| `?run=…` | which session to open. The value is the **absolute path to the run folder on the host**, exactly as it exists on the serving machine's disk — *not* a path on the viewer's own device. |

So the same link works from any device on the network; the path is always resolved
on the host. You can also point `?run=` at the `..._Analysis.json` file inside a run
folder — either form works. Omitting `?run=` opens the app empty (use **Open…** to
type a path). Each browser/tab is independent, so multiple people can view different
sessions at once.

#### What people can and can't reach

`--outputs-root` decides which **sessions** can be opened — not every file they reference:

- A `?run=` path is accepted **only if it resolves inside `--outputs-root`** (by real
  path, so symlinks can't escape); anything else — `?run=/etc/passwd`, another user's
  home folder — is refused with `403 Forbidden`. **Compare** likewise only lists and
  reads sibling runs **under the root**.
- The files a loaded session **points to** — its videos and images, including the
  **original source video** in your input folder — are then served from *wherever they
  live on the host*, even outside the root. Only files an opened session actually
  references are served, so the server still can't be coaxed into reading an unrelated
  arbitrary path.
- There is **no login or password** — anyone who can reach `http://<host-ip>:8000` can
  view every session under the root (and the files those sessions reference). Host it
  only on a trusted network, and point `--outputs-root` at just the folder you mean to
  share (e.g. one project's `output/`, not your whole home directory).

#### Finding the host IP and connecting

`<host-ip>` is the address of the **machine running `--serve`** — printed on startup,
or found with: macOS `ipconfig getifaddr en0` · Linux `hostname -I` · Windows
`ipconfig`. Other devices must use this LAN IP; `127.0.0.1`/`localhost` only works on
the host itself.

Devices must be on the **same Wi-Fi/subnet** with the host **firewall** allowing the
port. If a device still can't connect, it's usually **client isolation** (common on
guest/enterprise Wi-Fi — blocks device-to-device traffic; use a private network or
hotspot). Test from another device with `http://<host-ip>:8000/health` → `{"ok": true}`.


## Usage


The GIFT Integration Engine can query a directory and process a video based on a provided environment file.

### Input/output folder structure

When running the engine, organize your data on disk using an **Input Folder** and an **Output Folder**.

#### Input Folder

The Input Folder must contain:

- **Configuration file** (the "config file")
  - Controls runtime behavior of the pipeline (you will provide this alongside the `.vmeta.xml`).

- **Vmeta file** (`.vmeta.xml`)
  - Describes the session and references the associated videos.

- **`videos/` folder**
  - Contains the input video files.
  - **Important:** the filenames/titles in `videos/` must match what is referenced inside the `.vmeta.xml`.
  - If you rename videos, you must update the corresponding references in the `.vmeta.xml`.

- **`Misc-Data/` folder**
  - Contains additional supporting assets used by certain modules:

  - **Map image**
    - Used for mapping/visualization.

  - **`point_mappings`**
    - Used to initialize the mapper.

  - **`entry_polys`**
    - Used to define and restrict entry to a specific entry region.

#### Output Folder

The Output Folder is where generated outputs are written (videos, logs, metrics, and any other artifacts produced by the pipeline).

### `.vmeta.xml` format

The engine is driven by a **Vmeta** file (`.vmeta.xml`). This XML contains metadata about a single session/video and points the engine to:

- the video file to process
- basic identifiers such as title, start time, and offset
- the configuration file used for spatial/environment metadata

Below is a minimal example:

```xml
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<lom xmlns="http://ltsc.ieee.org/xsd/LOM">
    <technical>
        <location>videos/Scenario 1/SQ19-2.mp4</location>
    </technical>
    <general>
        <identifier>
            <catalog>start_time</catalog>
            <entry>1631202028000</entry>
        </identifier>
        <identifier>
            <catalog>offset</catalog>
            <entry>0</entry>
        </identifier>
        <identifier>
            <catalog>title</catalog>
            <entry>Test_Video</entry>
        </identifier>
        <identifier>
            <catalog>space_metadata_file</catalog>
            <entry>config.json</entry>
        </identifier>
    </general>
</lom>
```

> Note: Paths are case-sensitive on most systems. Make sure the folder name in the XML (e.g., `videos/`) matches your on-disk folder name exactly.

#### Fields

- `<technical><location>...` 
  - Relative (or absolute) path to the input video.
  - This must match the on-disk video filename. If you move/rename a video, update this path.

- `<general><identifier><catalog>start_time</catalog><entry>...` 
  - Session start time as a Unix epoch timestamp in **milliseconds**.

- `<general><identifier><catalog>offset</catalog><entry>...` 
  - Time offset (in milliseconds) applied relative to `start_time` (often `0`).

- `<general><identifier><catalog>title</catalog><entry>...` 
  - A human-readable session title used for labeling outputs.

- `<general><identifier><catalog>space_metadata_file</catalog><entry>...` 
  - Name/path of the config file that provides spatial/environment metadata used by the pipeline.
  - Typically this points to a JSON file located in the Input Folder.


#### Notes

- The root element is `<lom>` with an IEEE LOM XML namespace; keep this structure as shown.
- The engine expects the `identifier/catalog` values above (especially `start_time`, `offset`, `title`, and `space_metadata_file`).

> A tool for creating environment files is currently under development.

### Config file (space/environment metadata)

The config file referenced by the Vmeta field `space_metadata_file` (for example `config.json`) provides the **spatial layout**, **mapping assets**, and **model/runtime settings** used by the pipeline.

A typical config looks like this (comments shown here for explanation; your actual JSON may omit them):

```json
{
  "POD": [[75, 263], [75, 53], [290, 263], [397, 263]],
  "Boundary": [[22, 0], [450, 0], [450, 316], [22, 316]],

  "point_mapping_path": "Misc-Data/point_mapping_SQ19.txt",
  "entry_polys_path": "Misc-Data/generated_entry_polys.txt",
  "MapPath": "Misc-Data/map_image.png",

  "det_model": "rtmdet-m-person-640",
  "det_weights": "models/detect-best-mAP.pth",
  "det_cat_ids": [0],

  "pose2d_config": "rtmpose-x-halpe26-384x288",
  "pose2d_weights": "models/pose.pth",

  "box_conf_threshold": 0.3,
  "pose_conf_threshold": 0.3,

  "keypoint_indices": [15, 16],
  "device": "cpu",

  "boundary_pad_pct": 0.05,
  "track_enemy": true,
  "enemy_ids": [99],

  "visual_angle_degrees": 20.0,
  "min_threat_interaction_time_sec": 1.0,
  "entry_time_threshold_sec": 2.0,

  "pod_working_radius": 40.0,
  "pod_capture_threshold_sec": 0.1,
  "pod_time_limits": [1, 3, 1.5, 2],

  "coverage_time_threshold": 3.0,
  "stay_along_wall_pWall": 0.2,

  "gaze_keypoint_map": {"NOSE": 0, "LEYE": 1, "REYE": 2, "LEAR": 3, "REAR": 4}
}
```

#### Path resolution

- Paths such as `Misc-Data/...` are typically **relative to the Input Folder**.
- `det_model` / `pose2d_config` are **architecture tags** consumed by the in-repo runtime (`rtmdet-m-person-640` / `rtmpose-x-halpe26-384x288`). The legacy `libs/mmpose/...` config-string paths are still accepted (deprecated; resolves to the same tags).
- Model weights such as `models/detect-best-mAP.pth` and `models/pose.pth` are typically **relative to the repository root**. Drop new fine-tuned `.pth` checkpoints in directly — the runtime strips optimizer/EMA/meta blobs in-memory at load time.

If you reorganize folders, update these paths accordingly.

#### Field reference

##### Spatial layout and mapping assets

- `POD`: list of **POD points** on the map image, each as `[x, y]` in **map pixel coordinates**.
  - Used for POD assignment and POD capture analysis, and to render POD overlays on map videos.

- `Boundary`: polygon defining the **room boundary**, as a list of `[x, y]` vertices in **map pixel coordinates**.
  - Used to clamp/project mapped positions and to support boundary-aware computations (coverage/gaze).

- `point_mapping_path`: path to a pixel→map coordinate mapping file (used by the mapper).
  - This is typically a text file under `Misc-Data/`.

- `entry_polys_path`: path to entry-region polygons.
  - Used to allow entry points near doors even if slightly outside the main boundary.

- `MapPath`: path to the static room map image used for map-based overlays and coverage.

##### Models and inference

- `det_model`: detector **architecture tag** — currently `rtmdet-m-person-640`. The legacy MMDetection config-string path also works (deprecated).
- `det_weights`: checkpoint weights for the detector.
- `det_cat_ids`: detector category IDs to keep (commonly `[0]` for COCO person; the deployed person detector is single-class so this is informational).

- `pose2d_config`: pose **architecture tag** — currently `rtmpose-x-halpe26-384x288`.
- `pose2d_weights`: checkpoint weights for the 2D pose model.

- `pose_backend` / `pose_model_size` / `auto_download_models` / `pose_backend_weights`: optional pose-backend selection — switch the pose stage between the default fine-tuned 26-kp model, official size variants (t/s/m/l/x, auto-downloadable), 133-kp whole-body (RTMW), or 3D (RTMW3D), with identical downstream behavior via the canonical keypoint adapter. **See [MODELS.md](MODELS.md) for the full model documentation.**

- `box_conf_threshold`: minimum bounding-box confidence to accept a detection (post-NMS filter).
- `pose_conf_threshold`: minimum keypoint confidence to accept keypoints and render gaze/triangles.

- `device`: compute device for the PyTorch backend (e.g., `cpu`, `cuda`, `mps`). The runtime upgrades to ONNX or TensorRT automatically when `models/*.onnx` / `models/*.engine` artifacts are present alongside the weights — see [Step 5 — Optional Export](#step-5--optional-export-accelerated-runtime-artifacts) for how to produce them.

##### Tracking and boundary behavior

- `keypoint_indices`: keypoint indices used by the tracker for keypoint-based positioning logic.
- `boundary_pad_pct`: extra padding (fraction) around the boundary used when validating positions.
- `track_enemy`: enable/disable enemy tracking behaviors.

- `enemy_ids`: track IDs considered enemies (used for fall detection, gaze/coverage filtering, threat clearance).

##### Metrics/scoring parameters

- `visual_angle_degrees`: full field-of-view angle (degrees) used for gaze triangles, map gaze/coverage, and threat-clearance.
- `min_threat_interaction_time_sec`: minimum interaction time (seconds) required to count a threat as cleared.
- `entry_time_threshold_sec`: max allowed team entry span (seconds) for full score in `TOTAL_TIME_OF_ENTRY`.

- `pod_working_radius`: radius (map pixels) around each POD used to compute work areas for POD capture analysis.
- `pod_capture_threshold_sec`: seconds required inside a POD work area to count as captured.
- `pod_time_limits`: per-POD time limits (seconds) for `POD_CAPTURE_TIME` scoring.
  - If the scenario has fewer POD time limits than PODs, the engine may extend/reuse defaults depending on implementation.

- `coverage_time_threshold`: seconds of sustained coverage needed for full score in `TOTAL_FLOOR_COVERAGE_TIME`.
- `stay_along_wall_pWall`: sensitivity/threshold for the `STAY_ALONG_WALL` metric (higher is typically stricter wall adherence).

##### Gaze keypoints

- `gaze_keypoint_map`: mapping of named facial keypoints to indices (Halpe26 indices are commonly used).
  - Used to compute gaze direction (nose/eyes/ears).

#### Tips

- Start with `device: "cpu"` until your environment is verified.
- Keep map coordinate systems consistent:
  - `POD` and `Boundary` must be in the same pixel coordinate system as `MapPath`.
  - `point_mapping_path` must map from video pixels to this same map coordinate system.

### Running the engine (backend)

Start the backend engine (XMLRPC server):

```bash
python run_engine.py [options]
```

Once started, the engine listens for **XMLRPC** traffic on the specified port to begin processing.

#### Arguments

- `-p`, `--port` (int, default: `8000`)
  - Port to run the XMLRPC server on.

- `-f`, `--force_transcode` (flag)
  - Force re-encoding of videos before processing. Can help resolve issues with video format incompatibilities or corruption.

- `-v`, `--verbose` (flag)
  - Enable detailed logging (DEBUG). Without this flag, logging is limited to errors.

#### Examples

```bash
# Run on the default port (8000)
python run_engine.py

# Run on a custom port with verbose logs
python run_engine.py --port 9000 --verbose

# Force transcode before processing
python run_engine.py --force_transcode
```

### Running the local engine (without GIFT)

To run the engine locally for testing:

```bash
python run_engine_local.py <path/to/session.vmeta.xml> [options]
```

#### Required argument

- `vmeta` (positional)
  - Path to the `.vmeta.xml` file describing the session video.

#### Options

- `-f`, `--force_transcode` (flag)
  - Force re-encoding of videos before processing. Can help resolve issues with video format incompatibilities or corruption.

- `-v`, `--verbose` (flag)
  - Enable detailed logging (DEBUG). Without this flag, logging is limited to errors.

- `-o`, `--output_path` (str, default: `output/`)
  - This run's output folder — all artifacts (incl. `RunInfo.json`) are written flat into it. Give each run its own folder (e.g. `output/run1/`) so they sit as siblings under one parent for Compare.

#### Notes

- For legal reasons, example `.vmeta.xml` files are not included in the repository.
- Processing may take some time; a progress bar is displayed during execution.
- Outputs (videos/logs/metrics) are written to `--output_path` (default: `output/`).

#### Examples

```bash
# Basic run
python run_engine_local.py input/test.vmeta.xml

# Verbose logs
python run_engine_local.py input/test.vmeta.xml --verbose

# Force transcode and write this run's outputs into its own folder
python run_engine_local.py input/test.vmeta.xml --force_transcode --output_path ./output/run1/
```
