# -*- coding: utf-8 -*-
"""
edge_branch.py — Scharr Edge Fusion branch (SEF)

สาขาเล็กสำหรับประมวลผล edge map ที่คำนวณจากภาพ grayscale ด้วย Scharr filter
แล้วรวมกับ embedding หลัก (DINOv2 + length) เพื่อเน้นรูปทรง/ขอบ

Architecture: 3× Conv2d(3×3) + BN + ReLU + GAP → 64-dim vector
"""
import torch
import torch.nn as nn


class ScharrEdgeBranch(nn.Module):
    """
    รับ edge_map (B, 1, H, W) ค่า float [0,1] — คืน vector (B, 64)
    ใช้ร่วมกับ DeepFusionHead: concat(384 + 1 + 64) → 384
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
        edge_map: (B, 1, H, W) — ถ้า H,W ต่างกันก็ได้ (GAP จะย่อเหลือ 1×1)
        return: (B, 64)
        """
        x = self.net(edge_map)  # (B, 64, 1, 1)
        return x.flatten(1)     # (B, 64)
