# GIFT Multimodal EAE

GIFT Multimodal EAE is a Python-based integration engine for processing session videos using metadata provided in a `.vmeta.xml` file.

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Mapper tool](#mapper-tool-generate-mapping--polygons-for-config)
- [Config Builder tool](#config-builder-tool-create-configjson)
- [Usage](#usage)
- [Vmeta format](#vmetaxml-format)
- [Config file](#config-file-spaceenvironment-metadata)
- [Running the engine](#running-the-engine-backend)
- [Running locally](#running-the-local-engine-without-gift)

## Requirements

- **Python 3.10**
- **Conda** (Anaconda or Miniconda)

> All required environment details are specified in `environment.yml`. If you need additional packages, add them to `environment.yml` so the environment remains reproducible.

## Installation

### Step 0 — Install Conda

Install either **Anaconda** or **Miniconda**:

- Anaconda: https://www.anaconda.com/products/individual
- Miniconda: https://docs.conda.io/en/latest/miniconda.html

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/Surya-Rayala/GIFT-Multimodal-EAE-ECR.git
cd GIFT-Multimodal-EAE-ECR
```

---

### Step 2 — Create the Conda environment

Create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
```

---

### Step 3 — Activate the environment

```bash
conda activate gift-meae
```

#### Windows: Install system dependencies (required)

On **Windows**, install FFmpeg and related libraries via conda **after** activating the environment:

```bash
conda install -c conda-forge ffmpeg libx264 opencv
```

On **macOS**, the project typically works without this step.

---

### Step 4 — Install the OpenMMLab libraries (one-time setup)

With the environment active, install the OpenMMLab stack using `mim` (provided by the `openmim` package in the environment).

> Versions are pinned to avoid known compatibility issues.

```bash
mim install "mmengine>=0.7.1,<1.0.0"
mim install "mmcv>=2.0.0rc4,<2.2.0"
mim install "mmdet>=3.1.0,<3.3.0"
```

---

### Step 5 — Install MMPose

```bash
mim install "mmpose>=1.1.0"
```

#### macOS (Apple Silicon) note

On **macOS**, this project is supported in **CPU mode only** by default. MPS/GPU acceleration is **not officially supported** because MMCV’s NMS op does not provide an MPS implementation and will error at runtime (e.g., `nms_impl: implementation for device mps:0 not found`). 

##### Optional (advanced) — Enable MPS by patching MMCV NMS

If you still want to run with `device=mps`, you can patch the installed MMCV NMS implementation to force the NMS call onto CPU (your exact workaround). This is a **local environment hack** and may be overwritten if you reinstall MMCV.

1) Run the pipeline once with `device=mps`. When it crashes, read the traceback and find:

- the **file path** under `site-packages/mmcv/ops/` (often `mmcv/ops/nms.py`)
- the **exact line number** where it fails
- the **op name** mentioned in the failing call (examples: `ext_module.nms`, `ext_module.nms_rotated`, `ext_module.nms_quadri`, `ext_module.softnms`)

Example (yours will differ):

```text
File "/opt/anaconda3/envs/gift-meae/lib/python3.8/site-packages/mmcv/ops/nms.py", line 27, in forward
    inds = ext_module.nms(
```

2) Open **Finder** → press **⌘⇧G** (Go to Folder…) → paste the folder path from the traceback (everything up to `/mmcv/ops/`).

3) Open the file from the traceback (for example `nms.py`) and jump to the **exact line number** reported.

4) Patch the failing `ext_module.<op>(...)` call by forcing its tensor inputs to CPU.

- If it is **standard NMS** (`ext_module.nms`), you will typically see something like this at the error line:

```python
inds = ext_module.nms(
            bboxes, scores, iou_threshold=float(iou_threshold), offset=offset)
```

Change it to the following (exactly):

```python
inds = ext_module.nms(
            bboxes.cpu(), scores.cpu(), iou_threshold=float(iou_threshold), offset=offset)
inds = inds.to(bboxes.device)
```

- If the traceback shows a **different op**, apply the similar pattern.

That is the only change needed to avoid the MPS NMS crash.

---

### Optional — Verify installation

```bash
python -c "import torch; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available())"
python -c "import mmcv; print('mmcv', mmcv.__version__)"
python -c "import mmdet; print('mmdet', mmdet.__version__)"
python -c "import mmpose; print('mmpose', mmpose.__version__)"
```

## Mapper tool

This repository includes a small **GUI helper app** that lets you click on images to generate the data files that the config builder needs:

- **Homography mapping** (camera pixels → map pixels)
- **Entry polygons** (door/entry regions on the map)
- **Map points** (single points on the map; can be used for PODs or other reference points)
- **Room boundary** (the valid walkable/trackable area on the map)

### What it’s for (in plain English)

Think of the engine as taking people detected in the **camera video** and placing them onto a **top‑down map** of the room.

This mapper app helps you “teach” the system how to translate between those two views by:

1) picking matching points in the camera view and the map (so the system knows how to line them up), and
2) drawing the important regions on the map (where the room boundary is, where entries are, and key points like POD locations).

### How to open

```bash
python -m src.utils.mapper_app
```

(Depending on your repo layout, the module may also be named `src.utils.mapper`.)

### Basic workflow

1) **Load Map Image**
   - Choose the static map image (the same file you reference as `MapPath` in the config).

2) **Load Video** (or **Load Frame Image**)
   - Choose a representative video for the scenario, or a single frame image.

3) **Mapping tab (Homography)**
   - Click a recognizable point in the **Camera** view (green), then click the matching point on the **Map** (red).
   - Repeat for multiple point pairs (more is better; aim for >=5 spread across the room).
   - Click **Save Mapping TXT** to export a `*_mapping.txt` file.

4) **Entry Regions tab**
   - Click around a door/entry area on the **Map** to create a polygon.
   - Click **Confirm Polygon**, then **New Polygon** to add more.
   - Click **Save Regions TXT** to export a `*_entry_polygons.txt` file.

5) **Map Points tab**
   - Click points on the **Map** (one click = one point).
   - Click **Save Points TXT** to export a `*_map_points.txt` file.

6) **Room Boundary tab**
   - Click around the room outline on the **Map** to create the boundary polygon.
   - Click **Confirm Boundary**, then **Save Boundary TXT** to export a `*_room_boundary.txt` file.

### How these outputs map to the config

- `*_mapping.txt` → used to build `point_mapping_path`
- `*_entry_polygons.txt` → used to build `entry_polys_path`
- `*_map_points.txt` → often used to populate `POD` (or other reference points)
- `*_room_boundary.txt` → used to populate `Boundary`



## Config Builder tool

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

  "det_model": "libs/mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py",
  "det_weights": "models/detect.pth",
  "det_cat_ids": [0],

  "pose2d_config": "libs/mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py",
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
- The `det_model` / `pose2d_config` paths are typically **relative to the repository root** (they point into `libs/...`).
- Model weights such as `models/detect.pth` and `models/pose.pth` are typically **relative to the repository root** (or wherever you keep your weights).

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

- `det_model`: MMDetection config used by the pose inferencer for **person detection**.
- `det_weights`: checkpoint weights for the detector.
- `det_cat_ids`: detector category IDs to keep (commonly `[0]` for COCO person).

- `pose2d_config`: MMPose config for **2D pose**.
- `pose2d_weights`: checkpoint weights for the 2D pose model.

- `box_conf_threshold`: minimum bounding-box confidence to accept a detection.
- `pose_conf_threshold`: minimum keypoint confidence to accept keypoints and render gaze/triangles.

- `device`: compute device for inference (e.g., `cpu`, `cuda`, `mps`).
  - **macOS note:** MPS is not officially supported by default due to MMCV NMS limitations; see the macOS note in Installation.

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
  - Directory where outputs should be written.

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

# Force transcode and write outputs to a custom directory
python run_engine_local.py input/test.vmeta.xml --force_transcode --output_path ./my_outputs/
```
## Expert comparison (expert vs trainee)

The engine includes an **expert comparison** helper that compares a trainee run against a reference **expert** run for a single metric.

### What `compare_expert(...)` expects

`ProcessingEngine.compare_expert(metric_name, session_folder, expert_folder, vmeta_path)` expects:

- `metric_name` (string)
  - The name/ID of the metric you want to compare (example: `IDENTIFY_AND_CAPTURE_POD`).

- `session_folder` (string path)
  - Path to a **completed output folder** from running the engine on the **trainee** team.

- `expert_folder` (string path)
  - Path to a **completed output folder** from running the engine on the **expert** (reference) team.

- `vmeta_path` (string path)
  - Path to the **same** `.vmeta.xml` you used to run both sessions.

In other words: you first run the pipeline twice (once for expert, once for trainee) so you have two output directories. Then you pass those two output directories into `compare_expert`.


### Supported expert-comparison metrics

The `metric_name` argument must be one of the following supported metric IDs:

- `IDENTIFY_AND_CAPTURE_POD`
- `POD_CAPTURE_TIME`
- `STAY_ALONG_WALL`
- `ENTRANCE_VECTORS`
- `ENTRANCE_HESITATION`
- `THREAT_CLEARANCE`
- `TEAMMATE_COVERAGE`
- `THREAT_COVERAGE`
- `FLOOR_COVERAGE`
- `TOTAL_FLOOR_COVERAGE_TIME`

If a metric is not in this list, `compare_expert` may return an error artifact and a message indicating the metric is not implemented.

### How to test (simple step-by-step)

1) **Run the engine for the expert team** and save outputs to a dedicated folder:

```bash
python run_engine_local.py /path/to/session.vmeta.xml --output_path /path/to/expert_output --verbose
```

2) **Run the engine for the trainee team** and save outputs to a different folder:

```bash
python run_engine_local.py /path/to/session.vmeta.xml --output_path /path/to/trainee_output --verbose
```

3) **Run the comparison script** and point it at those two output folders.

Example `test_expert.py`:

```python
from src.processing_engine import ProcessingEngine

# Output folders produced by steps (1) and (2)
expert_folder = "/path/to/expert_output"
session_folder = "/path/to/trainee_output"

# Metric you want to compare
metric_name = "IDENTIFY_AND_CAPTURE_POD"

# The vmeta used for both runs
vmeta_path = "/path/to/session.vmeta.xml"

engine = ProcessingEngine()
result = engine.compare_expert(metric_name, session_folder, expert_folder, vmeta_path)
print(result)
```

Run it:

```bash
python test_expert.py
```


### Common gotchas

- The `expert_folder` and `session_folder` must be the **engine output folders** (the same paths you passed via `--output_path`).
- Make sure both runs are generated from the same scenario/config and the same `.vmeta.xml` so the comparison is meaningful.
- `metric_name` must match the metric ID used by the engine.

### Where expert-comparison artifacts are saved

When you call `compare_expert(...)`, any generated images or tables (such as side-by-side visualizations or difference overlays) are **saved into the session (trainee) output folder**—that is, under the `session_folder` argument you pass in.

- If you pass `--output_path` when running the engine, that is the root of your `session_folder`, so all expert-comparison artifacts will live somewhere under that folder.

### Returned elements (schema) from expert comparison

The result returned by `compare_expert(...)` is always a **JSON-serializable dict**.  
All expert-comparison artifacts (images and text files) are written into the **session (trainee) output folder** (`session_folder`).

There are three possible return shapes depending on the metric type.

---

#### 1) Side-by-side image metrics

Used by metrics such as:

- `IDENTIFY_AND_CAPTURE_POD`
- `STAY_ALONG_WALL`
- `ENTRANCE_VECTORS`

Return schema:

```json
{
  "Name": "<METRIC_ID>",
  "Type": "SideBySide",
  "ExpertImageLocation": "<session_folder>/<METRIC_ID>_Expert.jpg",
  "TraineeImageLocation": "<session_folder>/<METRIC_ID>_Trainee.jpg",
  "TxtLocation": "<session_folder>/<METRIC_ID>_Comparison.txt",
  "Text": "<structured summary + CSV block>"
}
```

Notes:

- Both expert and trainee images are written into the `session_folder`.
- `Text` contains a structured summary followed by a comma-separated table.

---

#### 2) Single-image metrics

Used by metrics such as:

- `ENTRANCE_HESITATION`
- `TOTAL_TIME_OF_ENTRY`
- `TOTAL_FLOOR_COVERAGE_TIME`

Return schema:

```json
{
  "Name": "<METRIC_ID>",
  "Type": "Single",
  "ImgLocation": "<session_folder>/<METRIC_ID>_Comparison.jpg",
  "TxtLocation": "<session_folder>/<METRIC_ID>_Comparison.txt",
  "Text": "<structured summary (and optional CSV block)>"
}
```

Notes:

- Only one comparison image is generated.
- `Text` may contain only summary lines (e.g., total-time metrics) or a summary followed by a CSV table (e.g., entrance hesitation).

---

### Structure of the `Text` field

For metrics that include detailed comparisons, the returned `Text` string follows this structure:

```
<High-level summary sentence(s)>

<CSV header line>
<CSV row 1>
<CSV row 2>
...
```

Important:

- The returned `Text` always uses **comma-separated rows** for machine readability.
- The saved `.txt` file contains a formatted (aligned) table version of the same data.
- Callers should not assume a fixed number of lines; instead, parse by splitting on newlines.

Callers should treat unknown keys in the returned dict as metric-specific and avoid hard-coding assumptions beyond the documented schema above.
