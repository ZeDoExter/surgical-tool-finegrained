# -*- coding: utf-8 -*-
"""
det_model.py — DINOv2 detector WITHOUT YOLO: backbone + light segmentation head

Idea (why this works with 441 images):
    DINOv2 patch features are famously strong for dense prediction "out of the
    box" (linear-probe segmentation). We attach a tiny conv decoder on the
    40×40 patch grid that predicts, per patch, (1 + num_classes) channels:
      ch0  = foreground (instrument) probability
      ch1: = per-class probability
    → binary mask + per-patch class → connected components → instance
      bbox + label + confidence (like YOLO output, but from segmentation).

    Bonus over YOLO for this project: the instance MASK lets us measure the
    true instrument length with minAreaRect (much more accurate than
    max(w,h) of an axis-aligned bbox when the tool lies diagonally), which
    feeds the classifier's length fusion + cm calibration.

Decoder cost: ~1.2M params (vs 21M backbone) — keeps realtime on Pi 5
(one 560×560 forward per frame, then pure numpy/cv2 post-processing).
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Dinov2Model

try:
    from peft import LoraConfig, TaskType, get_peft_model
    _PEFT_AVAILABLE = True
    _PEFT_IMPORT_ERROR = ""
except ImportError as e:
    _PEFT_AVAILABLE = False
    _PEFT_IMPORT_ERROR = str(e)
except Exception as e:
    _PEFT_AVAILABLE = False
    _PEFT_IMPORT_ERROR = str(e)


class ConvBNAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3, dropout: float = 0.0):
        super().__init__(
            nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )


class SegDecoder(nn.Module):
    """
    Light conv decoder on the ViT patch grid.

    Input : patch tokens (B, 40*40, D) [+ optional mid-layer tokens for detail]
    Output: (B, 1+C, 560, 560) — sigmoid semantics, trained with BCE-with-logits
            (ch0 = instrument-foreground, ch1.. = per-class).

    Uses 2× pixel-shuffle upsampling steps 40→80→160→320→560-ish, then a final
    bilinear resize to exactly img_size. All convs are narrow (192-256 ch) so
    the head stays ~1.2M params and exports cleanly to ONNX (Conv, PixelShuffle,
    GroupNorm, SiLU — no exotic ops).
    """

    def __decoder_block(self, cin: int, cout: int, up: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            ConvBNAct(cin, cout, 3, dropout),
            nn.Conv2d(cout, cout * up * up, 1, bias=False),
            nn.PixelShuffle(up),
        )

    def __init__(self, embed_dim: int, num_classes: int, img_size: int,
                 mid_dim: int = 256, decoder_dim: int = 192,
                 dropout: float = 0.1, use_mid_feats: bool = True,
                 mid_dims: Optional[List[int]] = None):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.use_mid_feats = use_mid_feats

        # optional fusion of mid-layer features (layers 6 & 9 = 1/2 and 3/4 depth)
        if use_mid_feats:
            mid_dims = mid_dims or [embed_dim, embed_dim]
            self.mid_proj = nn.ModuleList([
                nn.Linear(d, decoder_dim) for d in mid_dims
            ])
            in_ch = embed_dim + decoder_dim * len(mid_dims)
        else:
            self.mid_proj = None
            in_ch = embed_dim

        self.fuse = ConvBNAct(in_ch, mid_dim, 1, dropout)
        # 40 → 80 → 160, then bilinear to img_size (keeps Pi/1650 VRAM reasonable)
        self.up1 = self.__decoder_block(mid_dim, decoder_dim, 2, dropout)
        self.up2 = self.__decoder_block(decoder_dim, decoder_dim, 2, dropout)
        self.head = nn.Conv2d(decoder_dim, 1 + num_classes, 1, bias=True)

    def forward(self, tokens: torch.Tensor, mid_tokens: Optional[List[torch.Tensor]] = None,
                grid: Optional[int] = None) -> torch.Tensor:
        B, N, D = tokens.shape
        g = grid or int(N ** 0.5)
        x = tokens.transpose(1, 2).reshape(B, D, g, g)

        feats = [x]
        if self.use_mid_feats and self.mid_proj is not None and mid_tokens is not None:
            for proj, mt in zip(self.mid_proj, mid_tokens):
                m = proj(mt).transpose(1, 2).reshape(B, -1, g, g)
                feats.append(m)

        y = self.fuse(torch.cat(feats, dim=1))
        y = self.up1(y)
        y = self.up2(y)
        y = self.head(y)
        if y.shape[-1] != self.img_size:
            y = F.interpolate(y, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return y


class SurgicalDinoDetector(nn.Module):
    """DINOv2 (+LoRA) + SegDecoder — YOLO-free single-stage detector."""

    def __init__(self,
                 backbone_name: str = "facebook/dinov2-small",
                 finetune_mode: str = "lora",
                 lora_r: int = 16,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.1,
                 partial_last_blocks: int = 2,
                 num_classes: int = 14,
                 img_size: int = 560,
                 decoder_dim: int = 192,
                 decoder_mid_dim: int = 256,
                 decoder_dropout: float = 0.1,
                 use_mid_feats: bool = True):
        super().__init__()
        assert finetune_mode in ("frozen", "partial", "lora"), f"invalid mode: {finetune_mode}"
        self.backbone = Dinov2Model.from_pretrained(backbone_name)
        self.embed_dim = self.backbone.config.hidden_size
        self.num_classes = num_classes
        self.img_size = img_size
        self.finetune_mode = finetune_mode

        if finetune_mode == "lora":
            if not _PEFT_AVAILABLE:
                hint = f" (detail: {_PEFT_IMPORT_ERROR[:120]})" if _PEFT_IMPORT_ERROR else ""
                raise ImportError(
                    "finetune_mode='lora' requires `peft` but import failed" + hint +
                    ". Use finetune_mode='frozen'/'partial' to avoid LoRA."
                )
            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=["query", "value"], bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_cfg)
        elif finetune_mode == "partial":
            self._freeze_partial(partial_last_blocks)
        else:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.decoder = SegDecoder(
            self.embed_dim, num_classes, img_size,
            mid_dim=decoder_mid_dim, decoder_dim=decoder_dim,
            dropout=decoder_dropout, use_mid_feats=use_mid_feats,
        )

    def _freeze_partial(self, k: int) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        for block in self.backbone.encoder.layer[-k:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in self.backbone.layernorm.parameters():
                p.requires_grad = True

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: (B,3,H,W) ImageNet-normalized
        return: (B, 1+C, H, W) logits (ch0 fg / ch1.. class)
        """
        out = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        last = out.last_hidden_state[:, 1:, :]        # drop CLS → patch tokens
        mid_tokens = None
        if self.decoder.use_mid_feats:
            # hidden_states: (embeddings, layer1..layer12) → 6th & 9th block outputs
            hs = out.hidden_states
            mid_tokens = [hs[6][:, 1:, :], hs[9][:, 1:, :]]
        return self.decoder(last, mid_tokens, grid=self.grid)

    @property
    def grid(self) -> int:
        return self.img_size // self.backbone.config.patch_size

    def param_groups(self, lr_head: float, lr_backbone: float = None) -> List[dict]:
        """Differential LRs: decoder (new, higher) vs backbone adapters (lower)."""
        dec = [{"params": [p for p in self.decoder.parameters() if p.requires_grad], "lr": lr_head}]
        bb = [p for p in self.backbone.parameters() if p.requires_grad]
        if bb:
            dec.append({"params": bb, "lr": lr_backbone if lr_backbone is not None else lr_head})
        return dec


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_detector(ckpt_path: str, device: Optional[torch.device] = None) -> dict:
    """Load detector checkpoint → dict(model, cfg, classes, device) (mirrors evaluate.load_bundle)."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    from config import DetectorConfig
    cfg = DetectorConfig(**ckpt["cfg"])
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SurgicalDinoDetector(
        backbone_name=cfg.backbone_name, finetune_mode=cfg.finetune_mode,
        lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        partial_last_blocks=cfg.partial_last_blocks,
        num_classes=len(ckpt["classes"]), img_size=cfg.img_size,
        decoder_dim=cfg.decoder_dim, decoder_mid_dim=cfg.decoder_mid_dim,
        decoder_dropout=cfg.decoder_dropout, use_mid_feats=cfg.use_mid_feats,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()
    return {"model": model, "cfg": cfg, "classes": ckpt["classes"],
            "device": device, "val_iou": ckpt.get("val_iou"), "epoch": ckpt.get("epoch")}
