"""RTMW3D head — RTMW head plus a z-axis SimCC classifier.

Identical dual-branch structure to :class:`RTMWHead` (same upstream module
names, so checkpoints strict-load), with one extra linear ``cls_z`` decoding
the depth axis. Port of ``projects/rtmpose3d/rtmpose3d/rtmw3d_head.py``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from libs.giftpose.models.rtmw_head import RTMWHead


class RTMW3DHead(RTMWHead):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_size: tuple[int, int],          # (W, H)
        in_featuremap_size: tuple[int, int],
        z_input_size: int = 288,
        simcc_split_ratio: float = 2.0,
        final_layer_kernel_size: int = 7,
        gau_hidden_dims: int = 256,
        gau_s: int = 128,
        gau_expansion_factor: float = 2.0,
        gau_act_fn: str = "SiLU",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            input_size=input_size,
            in_featuremap_size=in_featuremap_size,
            simcc_split_ratio=simcc_split_ratio,
            final_layer_kernel_size=final_layer_kernel_size,
            gau_hidden_dims=gau_hidden_dims,
            gau_s=gau_s,
            gau_expansion_factor=gau_expansion_factor,
            gau_act_fn=gau_act_fn,
        )
        D = int(z_input_size * simcc_split_ratio)
        self.cls_z = nn.Linear(gau_hidden_dims, D, bias=False)

    def forward(self, feats) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc_b, enc_t = feats

        feats_t = self.final_layer(enc_t)
        feats_t = torch.flatten(feats_t, 2)
        feats_t = self.mlp(feats_t)

        dec_t = self.ps(enc_t)
        dec_t = self.conv_dec(dec_t)
        enc_b = torch.cat([dec_t, enc_b], dim=1)

        feats_b = self.final_layer2(enc_b)
        feats_b = torch.flatten(feats_b, 2)
        feats_b = self.mlp2(feats_b)

        out = torch.cat([feats_t, feats_b], dim=2)
        out = self.gau(out)
        return self.cls_x(out), self.cls_y(out), self.cls_z(out)
