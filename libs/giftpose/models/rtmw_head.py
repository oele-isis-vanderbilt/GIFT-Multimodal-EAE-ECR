"""RTMW head — port of mmpose ``RTMWHead`` (RTMPose-Wholebody, 2023).

Dual-branch SimCC head: the deepest neck level goes through
``final_layer``/``mlp``; a PixelShuffle-upscaled copy is fused with the
higher-resolution level through ``conv_dec``/``final_layer2``/``mlp2``; the
two halves are concatenated token-wise, run through one GAU block, and
classified into SimCC x/y vectors.

Module attribute names mirror upstream exactly (``conv_dec``, ``final_layer``,
``final_layer2``, ``mlp``, ``mlp2``, ``gau``, ``cls_x``, ``cls_y``) so official
checkpoints load with ``strict=True``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from libs.giftpose.models.csp_blocks import ConvBNAct
from libs.giftpose.models.rtmcc_block import RTMCCBlock, ScaleNorm

_BN = dict(type="BN")
_RELU = dict(type="ReLU")


class RTMWHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_size: tuple[int, int],          # (W, H)
        in_featuremap_size: tuple[int, int],  # (W/32, H/32)
        simcc_split_ratio: float = 2.0,
        final_layer_kernel_size: int = 7,
        gau_hidden_dims: int = 256,
        gau_s: int = 128,
        gau_expansion_factor: float = 2.0,
        gau_act_fn: str = "SiLU",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.simcc_split_ratio = simcc_split_ratio

        flatten_dims = in_featuremap_size[0] * in_featuremap_size[1]
        ps = 2
        self.ps = nn.PixelShuffle(ps)
        self.conv_dec = ConvBNAct(
            in_channels // ps**2,
            in_channels // 4,
            final_layer_kernel_size,
            padding=final_layer_kernel_size // 2,
            norm_cfg=_BN,
            act_cfg=_RELU,
        )
        self.final_layer = ConvBNAct(
            in_channels,
            out_channels,
            final_layer_kernel_size,
            padding=final_layer_kernel_size // 2,
            norm_cfg=_BN,
            act_cfg=_RELU,
        )
        self.final_layer2 = ConvBNAct(
            in_channels // ps + in_channels // 4,
            out_channels,
            final_layer_kernel_size,
            padding=final_layer_kernel_size // 2,
            norm_cfg=_BN,
            act_cfg=_RELU,
        )

        self.mlp = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, gau_hidden_dims // 2, bias=False),
        )
        self.mlp2 = nn.Sequential(
            ScaleNorm(flatten_dims * ps**2),
            nn.Linear(flatten_dims * ps**2, gau_hidden_dims // 2, bias=False),
        )

        W = int(input_size[0] * simcc_split_ratio)
        H = int(input_size[1] * simcc_split_ratio)

        self.gau = RTMCCBlock(
            gau_hidden_dims,
            gau_hidden_dims,
            s=gau_s,
            expansion_factor=gau_expansion_factor,
            act_fn=gau_act_fn,
        )
        self.cls_x = nn.Linear(gau_hidden_dims, W, bias=False)
        self.cls_y = nn.Linear(gau_hidden_dims, H, bias=False)

    def forward(self, feats: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # feats = (enc_b, enc_t): enc_b is the higher-resolution level
        # (in_channels // 2 channels), enc_t the deepest (in_channels).
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
        return self.cls_x(out), self.cls_y(out)
