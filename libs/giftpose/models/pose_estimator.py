"""TopdownPoseEstimator — backbone + head wrapper for RTMPose.

Submodule names ``backbone`` and ``head`` mirror upstream so checkpoints with
the prefix layout ``backbone.*`` / ``head.*`` load via ``strict=True``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from libs.giftpose.models.cspnext import CSPNeXt
from libs.giftpose.models.cspnext_pafpn import CSPNeXtPAFPN
from libs.giftpose.models.rtmcc_head import RTMCCHead


class TopdownPoseEstimator(nn.Module):
    def __init__(self, backbone: CSPNeXt, head: RTMCCHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        return self.head(feats[-1])


class TopdownPoseEstimatorWithNeck(nn.Module):
    """Backbone + CSPNeXtPAFPN neck + multi-level head (RTMW / RTMW3D).

    Submodule names ``backbone`` / ``neck`` / ``head`` mirror upstream so
    official checkpoints load with ``strict=True``.
    """

    def __init__(self, backbone: CSPNeXt, neck: CSPNeXtPAFPN, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.head(feats)


def build_pose(spec) -> TopdownPoseEstimator:
    """Build a pose estimator from a :class:`libs.giftpose.registry.ArchSpec`.

    Families:
      - ``rtmpose``: CSPNeXt backbone (last stage only) + RTMCCHead. Covers
        every body2d size variant (t/s/m/l/x) — only the scaling factors,
        input size and keypoint count differ.
      - ``rtmw`` / ``rtmw3d``: built by their family-specific builders (see
        Stage-5/7 modules); dispatched here once available.
    """
    if spec.family == "rtmpose":
        backbone = CSPNeXt(
            deepen_factor=spec.deepen_factor,
            widen_factor=spec.widen_factor,
            out_indices=(4,),
            expand_ratio=0.5,
            channel_attention=True,
        )
        head = RTMCCHead(
            in_channels=spec.head_in_channels,
            num_keypoints=spec.num_keypoints,
            input_size=spec.input_size,
            in_featuremap_size=spec.in_featuremap_size,
            simcc_split_ratio=spec.simcc_split_ratio,
            final_layer_kernel_size=7,
            gau_hidden_dims=256,
            gau_s=128,
            gau_expansion_factor=2.0,
            gau_act_fn="SiLU",
        )
        return TopdownPoseEstimator(backbone, head)
    if spec.family in ("rtmw", "rtmw3d"):
        from libs.giftpose.models.rtmw_head import RTMWHead
        from libs.giftpose.models.rtmw3d_head import RTMW3DHead

        c = spec.head_in_channels  # deepest backbone level (1024 * widen)
        backbone = CSPNeXt(
            deepen_factor=spec.deepen_factor,
            widen_factor=spec.widen_factor,
            out_indices=(2, 3, 4),
            expand_ratio=0.5,
            channel_attention=True,
        )
        neck = CSPNeXtPAFPN(
            in_channels=[c // 4, c // 2, c],
            out_channels=None,
            num_csp_blocks=2,
            expand_ratio=0.5,
            out_indices=(1, 2),
        )
        head_common = dict(
            in_channels=c,
            out_channels=spec.num_keypoints,
            input_size=spec.input_size,
            in_featuremap_size=spec.in_featuremap_size,
            simcc_split_ratio=spec.simcc_split_ratio,
            final_layer_kernel_size=7,
            gau_hidden_dims=256,
            gau_s=128,
            gau_expansion_factor=2.0,
            gau_act_fn="SiLU",
        )
        if spec.family == "rtmw3d":
            head = RTMW3DHead(z_input_size=spec.z_input_size, **head_common)
        else:
            head = RTMWHead(**head_common)
        return TopdownPoseEstimatorWithNeck(backbone, neck, head)
    raise NotImplementedError(
        f"Pose family {spec.family!r} ({spec.tag}) is not available yet."
    )


def build_rtmpose_x_halpe26(input_size: tuple[int, int] = (288, 384)) -> TopdownPoseEstimator:
    """RTMPose-x configured for Halpe-26 at 288x384 — matches
    ``libs/mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py``.

    Kept as a thin wrapper over the registry spec so existing callers
    (export scripts, parity harness) keep working; the module graph is
    identical to the pre-registry version.
    """
    from libs.giftpose.registry import POSE_ARCHS

    spec = POSE_ARCHS["rtmpose-x-halpe26-384x288"]
    if tuple(input_size) != tuple(spec.input_size):
        from dataclasses import replace
        spec = replace(spec, input_size=tuple(input_size))
    return build_pose(spec)
