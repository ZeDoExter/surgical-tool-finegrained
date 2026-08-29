# -*- coding: utf-8 -*-
"""
edge_branch.py — Scharr Edge Fusion branch (SEF)

Small branch for processing edge maps computed from grayscale images using a Scharr filter,
then fused with the main embedding (DINOv2 + length) to emphasize shape/edges.

Architecture: 3x Conv2d(3x3) + BN + ReLU + GAP -> 64-dim vector
"""
import torch
import torch.nn as nn


class ScharrEdgeBranch(nn.Module):
    """
    Takes edge_map (B, 1, H, W) with float values in [0,1] — returns vector (B, 64)
    Used together with DeepFusionHead: concat(384 + 1 + 64) -> 384
    """
    def __init__(self, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(dropout),
        )
        self.out_dim = out_dim

    def forward(self, edge_map: torch.Tensor) -> torch.Tensor:
        """
        edge_map: (B, 1, H, W) — H, W may vary (GAP reduces to 1x1)
        return: (B, 64)
        """
        x = self.net(edge_map)  # (B, 64, 1, 1)
        return x.flatten(1)     # (B, 64)
