"""CSPNeXt-PAFPN neck — port of mmdet ``CSPNeXtPAFPN``.

Module attribute names mirror upstream: ``reduce_layers``, ``top_down_blocks``,
``downsamples``, ``bottom_up_blocks``, ``out_convs``.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn

from libs.giftpose.models.csp_blocks import ConvBNAct, CSPLayer

# Upstream RTMDet neck uses ``norm_cfg=dict(type='SyncBN')`` (default PyTorch
# eps/momentum); revert_sync_batchnorm preserves those defaults at inference.
_DEFAULT_NORM = dict(type="BN")
_DEFAULT_ACT = dict(type="SiLU", inplace=True)


class CSPNeXtPAFPN(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int | None,
        num_csp_blocks: int = 3,
        expand_ratio: float = 0.5,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
        out_indices: Sequence[int] | None = None,
    ) -> None:
        """``out_channels=None`` preserves per-level channel counts (no
        ``out_convs``) — the RTMW/RTMW3D pose-neck configuration; upstream
        checkpoints for those models carry no ``out_convs`` keys.
        ``out_indices`` selects which pyramid levels are returned
        (``None`` = all levels — the detector's behavior).
        """
        super().__init__()
        norm_cfg = norm_cfg or _DEFAULT_NORM
        act_cfg = act_cfg or _DEFAULT_ACT
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.out_indices = tuple(out_indices) if out_indices is not None else None

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Top-down: highest-level reduces channels then concatenates with the
        # level below it after upsampling, then a CSPLayer fuses them.
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(self.in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvBNAct(self.in_channels[idx], self.in_channels[idx - 1], 1,
                          norm_cfg=norm_cfg, act_cfg=act_cfg)
            )
            self.top_down_blocks.append(
                CSPLayer(
                    self.in_channels[idx - 1] * 2,
                    self.in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_cspnext_block=True,
                    expand_ratio=expand_ratio,
                    norm_cfg=norm_cfg, act_cfg=act_cfg,
                )
            )

        # Bottom-up: each level downsamples and concatenates with the level
        # above, then a CSPLayer fuses them up to the next level's channel count.
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(self.in_channels) - 1):
            self.downsamples.append(
                ConvBNAct(self.in_channels[idx], self.in_channels[idx], 3,
                          stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            )
            self.bottom_up_blocks.append(
                CSPLayer(
                    self.in_channels[idx] * 2,
                    self.in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_cspnext_block=True,
                    expand_ratio=expand_ratio,
                    norm_cfg=norm_cfg, act_cfg=act_cfg,
                )
            )

        # Per-level 3x3 ConvBNAct collapsing each level to the shared neck
        # width. Skipped for ``out_channels=None`` (RTMW-style pose necks).
        if out_channels is not None:
            self.out_convs = nn.ModuleList([
                ConvBNAct(self.in_channels[i], out_channels, 3, padding=1,
                          norm_cfg=norm_cfg, act_cfg=act_cfg)
                for i in range(len(self.in_channels))
            ])
        else:
            self.out_convs = None

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        assert len(inputs) == len(self.in_channels)

        inner_outs: list[torch.Tensor] = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_high = self.reduce_layers[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = self.upsample(feat_high)
            inner = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], dim=1)
            )
            inner_outs.insert(0, inner)

        outs: list[torch.Tensor] = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](torch.cat([downsample, feat_high], dim=1))
            outs.append(out)

        if self.out_convs is not None:
            for idx, conv in enumerate(self.out_convs):
                outs[idx] = conv(outs[idx])
        if self.out_indices is not None:
            return tuple(outs[i] for i in self.out_indices)
        return tuple(outs)
