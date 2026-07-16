"""RTMDet wrapper — backbone + neck + head.

Submodule names ``backbone`` / ``neck`` / ``bbox_head`` match the upstream
mmengine state-dict prefix so legacy ``.pth`` checkpoints load via
``strict=True`` after the optimizer/EMA/meta blobs are stripped.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from libs.giftpose.models.cspnext import CSPNeXt
from libs.giftpose.models.cspnext_pafpn import CSPNeXtPAFPN
from libs.giftpose.models.rtmdet_head import RTMDetSepBNHead, decode_rtmdet


class RTMDet(nn.Module):
    def __init__(self, backbone: CSPNeXt, neck: CSPNeXtPAFPN, bbox_head: RTMDetSepBNHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.bbox_head = bbox_head

    def forward_features(self, x: torch.Tensor) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.bbox_head(feats)

    def detect(
        self,
        x: torch.Tensor,
        score_thr: float = 0.05,
        iou_threshold: float = 0.6,
        max_per_img: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cls_scores, bbox_preds = self.forward_features(x)
        return decode_rtmdet(
            cls_scores, bbox_preds, strides=self.bbox_head.strides,
            score_thr=score_thr, iou_threshold=iou_threshold, max_per_img=max_per_img,
        )

    def forward(self, x: torch.Tensor) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        # Plain forward returns the raw FPN cls_scores + bbox_preds (used by
        # ONNX export so the decoder lives in the graph wrapper).
        return self.forward_features(x)


def build_detector(spec) -> RTMDet:
    """Build a detector from a :class:`libs.giftpose.registry.ArchSpec`.

    Only the RTMDet-m person geometry is registered today; the function
    exists so backends resolve detectors through the registry the same way
    they resolve pose models.
    """
    if spec.family == "rtmdet" and (spec.deepen_factor, spec.widen_factor) == (0.67, 0.75):
        return build_rtmdet_m_person()
    raise NotImplementedError(f"Detector spec {spec.tag!r} is not available yet.")


def build_rtmdet_m_person() -> RTMDet:
    """RTMDet-m person detector — matches
    ``libs/mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py`` /
    base config ``rtmdet_m_8xb32-300e_coco.py`` with ``num_classes=1``.

    From the saved checkpoint:
      - widen=0.75 (192/256), deepen=0.67
      - neck: in=[192, 384, 768], out=192, num_csp_blocks=2
      - head: in=192, feat=192, stacked=2, single class, exp_on_reg=True
    """
    backbone = CSPNeXt(
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(2, 3, 4),
        expand_ratio=0.5,
        channel_attention=True,
    )
    neck = CSPNeXtPAFPN(
        in_channels=[192, 384, 768],
        out_channels=192,
        num_csp_blocks=2,
        expand_ratio=0.5,
    )
    head = RTMDetSepBNHead(
        num_classes=1,
        in_channels=192,
        feat_channels=192,
        stacked_convs=2,
        strides=(8, 16, 32),
        exp_on_reg=True,
        pred_kernel_size=1,
    )
    return RTMDet(backbone, neck, head)
