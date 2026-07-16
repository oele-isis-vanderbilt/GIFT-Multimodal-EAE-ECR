"""Architecture registry — the single source of truth for model tags.

Each supported architecture tag maps to an :class:`ArchSpec` describing how
to build the network (CSPNeXt scaling factors, head family, keypoint count,
SimCC split ratio, input size), which keypoint metadata to use (flip indices
for flip-test, sigmas, skeleton), and where the official checkpoint lives for
auto-download (``url`` — only for stock OpenMMLab releases; the fine-tuned
project weights in ``models/`` are never downloaded).

Tags are resolved by :func:`resolve_pose` / :func:`resolve_det`; legacy
mmpose config-path strings alias to the same specs so existing configs keep
working unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class ArchSpec:
    """Static description of one network architecture + checkpoint source."""

    tag: str
    kind: str                       # "det" | "pose"
    family: str                     # "rtmdet" | "rtmpose" | "rtmw" | "rtmw3d"
    deepen_factor: float
    widen_factor: float
    input_size: Tuple[int, int]     # (W, H)
    num_keypoints: int = 0          # 0 for detectors
    simcc_split_ratio: float = 2.0
    meta: str = "halpe26"           # meta module under libs/giftpose/meta
    has_neck: bool = False          # CSPNeXtPAFPN between backbone and head
    has_z: bool = False             # 3D head (extra SimCC z branch)
    z_input_size: int = 0           # z codec bins base (288 for RTMW3D)
    url: Optional[str] = None       # official checkpoint (auto-download)
    approx_mb: int = 0              # download size hint for logs
    default_filename: Optional[str] = None  # cache filename under models/

    @property
    def head_in_channels(self) -> int:
        # CSPNeXt P5 last stage = 1024 * widen_factor (no-neck heads read it
        # directly; PAFPN necks keep the channel count per level).
        return int(1024 * self.widen_factor)

    @property
    def in_featuremap_size(self) -> Tuple[int, int]:
        return (self.input_size[0] // 32, self.input_size[1] // 32)


_OMM = "https://download.openmmlab.com/mmpose/v1"


def _body2d(size: str, deepen: float, widen: float, wh: Tuple[int, int],
            url_file: str, mb: int) -> ArchSpec:
    w, h = wh
    return ArchSpec(
        tag=f"rtmpose-{size}-halpe26-{h}x{w}",
        kind="pose",
        family="rtmpose",
        deepen_factor=deepen,
        widen_factor=widen,
        input_size=(w, h),
        num_keypoints=26,
        meta="halpe26",
        url=f"{_OMM}/projects/rtmposev1/{url_file}",
        approx_mb=mb,
        default_filename=url_file,
    )


# ---------------------------------------------------------------------------
# Detector specs
# ---------------------------------------------------------------------------

DET_ARCHS: dict[str, ArchSpec] = {
    "rtmdet-m-person-640": ArchSpec(
        tag="rtmdet-m-person-640",
        kind="det",
        family="rtmdet",
        deepen_factor=0.67,
        widen_factor=0.75,
        input_size=(640, 640),
    ),
}

# ---------------------------------------------------------------------------
# Pose specs
# ---------------------------------------------------------------------------

POSE_ARCHS: dict[str, ArchSpec] = {
    # Current production architecture. NOTE: no ``url`` — the default weights
    # are the project's fine-tuned ``models/pose.pth``; the official body7
    # release for this geometry is registered separately below as size "x".
    "rtmpose-x-halpe26-384x288": ArchSpec(
        tag="rtmpose-x-halpe26-384x288",
        kind="pose",
        family="rtmpose",
        deepen_factor=1.33,
        widen_factor=1.25,
        input_size=(288, 384),
        num_keypoints=26,
        meta="halpe26",
        url=f"{_OMM}/projects/rtmposev1/"
            "rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.pth",
        approx_mb=191,
        default_filename="rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288.pth",
    ),
    **{s.tag: s for s in (
        _body2d("t", 0.167, 0.375, (192, 256),
                "rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605.pth", 13),
        _body2d("s", 0.33, 0.5, (192, 256),
                "rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.pth", 22),
        _body2d("m", 0.67, 0.75, (192, 256),
                "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth", 53),
        _body2d("l", 1.0, 1.0, (192, 256),
                "rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.pth", 108),
    )},
    # --- COCO-WholeBody 133-kp (RTMW, cocktail14). "l" and "m" are the
    # officially released distilled variants (rtmw-dw-x-l / rtmw-dw-l-m).
    "rtmw-x-cocktail14-133": ArchSpec(
        tag="rtmw-x-cocktail14-133",
        kind="pose",
        family="rtmw",
        deepen_factor=1.33,
        widen_factor=1.25,
        input_size=(288, 384),
        num_keypoints=133,
        meta="wholebody133",
        url=f"{_OMM}/projects/rtmw/"
            "rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth",
        approx_mb=353,
        default_filename="rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288.pth",
    ),
    "rtmw-l-cocktail14-133": ArchSpec(
        tag="rtmw-l-cocktail14-133",
        kind="pose",
        family="rtmw",
        deepen_factor=1.0,
        widen_factor=1.0,
        input_size=(288, 384),
        num_keypoints=133,
        meta="wholebody133",
        url=f"{_OMM}/projects/rtmw/"
            "rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth",
        approx_mb=220,
        default_filename="rtmw-dw-x-l_simcc-cocktail14_270e-384x288.pth",
    ),
    "rtmw-m-cocktail14-133": ArchSpec(
        tag="rtmw-m-cocktail14-133",
        kind="pose",
        family="rtmw",
        deepen_factor=0.67,
        widen_factor=0.75,
        input_size=(192, 256),
        num_keypoints=133,
        meta="wholebody133",
        url=f"{_OMM}/projects/rtmw/"
            "rtmw-dw-l-m_simcc-cocktail14_270e-256x192-20231122.pth",
        approx_mb=124,
        default_filename="rtmw-dw-l-m_simcc-cocktail14_270e-256x192.pth",
    ),
    # --- 3D whole-body (RTMW3D): single-frame top-down, SimCC x/y/z. The
    # z channel is root-normalized metric depth (see codecs/simcc3d.py).
    "rtmw3d-l-cocktail14-133": ArchSpec(
        tag="rtmw3d-l-cocktail14-133",
        kind="pose",
        family="rtmw3d",
        deepen_factor=1.0,
        widen_factor=1.0,
        input_size=(288, 384),
        num_keypoints=133,
        meta="wholebody133",
        has_z=True,
        z_input_size=288,
        url=f"{_OMM}/wholebody_3d_keypoint/rtmw3d/"
            "rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth",
        approx_mb=220,
        default_filename="rtmw3d-l_8xb64_cocktail14-384x288.pth",
    ),
}

# Legacy mmpose config-path strings accepted by older configs.
_DET_ALIASES = {
    "libs/mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py":
        "rtmdet-m-person-640",
}
_POSE_ALIASES = {
    "libs/mmpose/configs/body_2d_keypoint/rtmpose/body8/"
    "rtmpose-x_8xb256-700e_body8-halpe26-384x288.py":
        "rtmpose-x-halpe26-384x288",
}


def resolve_pose(tag: str) -> ArchSpec:
    key = _POSE_ALIASES.get(tag, tag)
    if key not in POSE_ARCHS:
        raise ValueError(
            f"Unsupported pose2d model identifier {tag!r}. "
            f"Supported values: {sorted(POSE_ARCHS) + sorted(_POSE_ALIASES)}"
        )
    return POSE_ARCHS[key]


def resolve_det(tag: str) -> ArchSpec:
    key = _DET_ALIASES.get(tag, tag)
    if key not in DET_ARCHS:
        raise ValueError(
            f"Unsupported det_model identifier {tag!r}. "
            f"Supported values: {sorted(DET_ARCHS) + sorted(_DET_ALIASES)}"
        )
    return DET_ARCHS[key]


# Default tags used when a config omits the architecture fields entirely.
DEFAULT_POSE_TAG = "rtmpose-x-halpe26-384x288"
DEFAULT_DET_TAG = "rtmdet-m-person-640"
