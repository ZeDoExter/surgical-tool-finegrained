# -*- coding: utf-8 -*-
"""
model.py — DINOv2 backbone + fusion head that fuses "length from mask" into the embedding

Architecture (v2 — improved):
    image -> DINOv2 ViT-S/14 -> all tokens (257 tokens, 384-dim)
    -> AttentionPooling: multi-head attention over patch tokens only -> weighted sum -> 384-dim
    mask -> measure_length_px() -> normalize(mean,std of train) -> scalar (1-dim)
    concat(384+1) -> DeepFusionHead (LN->Linear->GELU->Dropout->Linear->GELU->Dropout) -> 384-dim
    -> ArcFace loss (L2-normalize both embedding and class weights)

    Background: green cloth with shadows (not silver tray) — shadows make bounding-box
    detection harder; image size 504 and bbox 0.15 are defaults chosen from experiments.

v2 changes:
  - AttentionPooling instead of CLS-only: retains spatial detail from all patch tokens
  - DeepFusionHead: 2-layer MLP with LayerNorm for deeper fusion
  - Supports mask_aux_features: width/height/area/ratio from mask (optional, adds auxiliary info)

Why fuse length: resizing the image to 224x224 loses the true "real scale" information
but some class pairs differ only by length — so the length measured from the mask is fed in
directly to help.
"""
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Dinov2Model

try:
    from edge_branch import ScharrEdgeBranch
    _EDGE_AVAILABLE = True
except ImportError:
    ScharrEdgeBranch = None  # type: ignore
    _EDGE_AVAILABLE = False

try:
    from peft import LoraConfig, TaskType, get_peft_model
    _PEFT_AVAILABLE = True
    _PEFT_IMPORT_ERROR = ""
except ImportError as e:
    # Colab has torchao 0.10.0 but peft 0.17+ requires >=0.16.0
    # Fall back to non-LoRA modes; user can fix via: !pip install -U torchao  (then restart runtime)
    _PEFT_AVAILABLE = False
    _PEFT_IMPORT_ERROR = str(e)
except Exception as e:
    _PEFT_AVAILABLE = False
    _PEFT_IMPORT_ERROR = str(e)


class AttentionPooling(nn.Module):
    """
    Multi-head attention pooling over ViT patch tokens

    Instead of using the CLS token alone — learn attention weights
    over patch tokens only (excluding CLS) to retain spatial detail
    needed for fine-grained classification
    """

    def __init__(self, embed_dim: int, num_heads: int = 6, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        # learnable query — 1 token that attends to all patch tokens
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, num_tokens, embed_dim) — tokens from ViT (including CLS)
        return: (B, embed_dim) — pooled embedding
        """
        B = x.shape[0]
        # Separate patch tokens (index 1:) from CLS (index 0)
        patch_tokens = x[:, 1:, :]   # (B, 256, 384)
        q = self.query.expand(B, -1, -1)  # (B, 1, 384)
        attn_out, _ = self.attn(q, patch_tokens, patch_tokens)  # (B, 1, 384)
        attn_out = attn_out.squeeze(1)   # (B, 384)
        return self.norm(attn_out)


class DeepFusionHead(nn.Module):
    """
    Deep fusion head: concat visual embedding + length scalar -> project back to 384-dim

    v1: Linear(385->384) single layer — fast but shallow fusion
    v2: LN -> Linear(385->768) -> GELU -> Dropout -> Linear(768->384) -> Dropout
    """

    def __init__(self, embed_dim: int, aux_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim + aux_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim + aux_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, emb: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """
        emb: (B, embed_dim) | aux: (B, aux_dim)
        return: (B, embed_dim)
        """
        x = torch.cat([emb, aux.to(emb.dtype)], dim=1)  # (B, embed_dim+aux_dim)
        x = self.ln(x)
        return self.net(x)


class SurgicalDinoFusion(nn.Module):
    """
    DINOv2 backbone (+LoRA/partial unfreeze) + AttentionPooling + DeepFusionHead

    finetune_mode:
      - "lora":    Wrap backbone with LoRA (train only adapters, very few parameters)
      - "partial": Freeze entire backbone then unfreeze last N blocks + final LayerNorm
      - "frozen":  Freeze entire backbone (use as feature extractor only)
    """

    def __init__(self,
                 backbone_name: str = "facebook/dinov2-small",
                 finetune_mode: str = "lora",
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.1,
                 partial_last_blocks: int = 2,
                 head_dropout: float = 0.1,
                 use_attention_pool: bool = True,
                 use_sef: bool = False,
                 sef_out_dim: int = 64):
        super().__init__()
        assert finetune_mode in ("frozen", "partial", "lora"), f"invalid mode: {finetune_mode}"

        self.backbone = Dinov2Model.from_pretrained(backbone_name)
        self.embed_dim = self.backbone.config.hidden_size  # 384 for dinov2-small
        self.use_attention_pool = use_attention_pool
        self.use_sef = use_sef

        if finetune_mode == "lora":
            if not _PEFT_AVAILABLE:
                hint = f" (detail: {_PEFT_IMPORT_ERROR[:120]})" if _PEFT_IMPORT_ERROR else ""
                raise ImportError(
                    "finetune_mode='lora' requires `peft` but import failed" + hint +
                    ". Fix in Colab: !pip install -U torchao  then Runtime -> Restart session, "
                    "or use finetune_mode='frozen'/'partial' to avoid LoRA."
                )
            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["query", "value"],  # LoRA only on attention query/value projections
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_cfg)
        elif finetune_mode == "partial":
            self._freeze_partial(partial_last_blocks)
        else:  # frozen
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Attention pooling: aggregate patch tokens -> 384-dim
        self.attn_pool = AttentionPooling(self.embed_dim, num_heads=6, dropout=head_dropout) if use_attention_pool else None
        # SEF branch (if enabled)
        self.edge_branch = None
        if use_sef:
            if not _EDGE_AVAILABLE:
                raise ImportError("use_sef=True requires edge_branch.py")
            self.edge_branch = ScharrEdgeBranch(out_dim=sef_out_dim, dropout=head_dropout)
        # Deep fusion head: concat(384 + 1 + 64_if_sef) -> 384
        aux_dim = 1 + (sef_out_dim if use_sef else 0)
        self.fusion = DeepFusionHead(self.embed_dim, aux_dim=aux_dim, dropout=head_dropout)

    def _freeze_partial(self, k: int) -> None:
        """Freeze entire backbone then unfreeze only last k blocks + final layernorm"""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for block in self.backbone.encoder.layer[-k:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in self.backbone.layernorm.parameters():
            p.requires_grad = True

    def forward(self, pixel_values: torch.Tensor, length_feat: torch.Tensor,
                edge_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        pixel_values: (B,3,H,W) normalized | length_feat: (B,) normalized length
        edge_map: (B,1,H,W) Scharr edge [0,1] if use_sef=True — if None, zeros are used instead
        return: embedding (B, embed_dim=384) for ArcFace loss
        """
        out = self.backbone(pixel_values=pixel_values).last_hidden_state  # (B, tokens, 384)
        # Attention pooling: attend only to patch tokens instead of CLS-only
        if self.attn_pool is not None:
            e = self.attn_pool(out)        # (B, 384)
        else:
            e = out[:, 0]                   # CLS token fallback
        # Fusion: concat visual embedding + length scalar (+ 64 edge features if SEF)
        if self.use_sef and self.edge_branch is not None and edge_map is not None:
            ef = self.edge_branch(edge_map)  # (B, 64)
            aux = torch.cat([length_feat.unsqueeze(1), ef], dim=1)  # (B, 65)
        else:
            aux = length_feat.unsqueeze(1)      # (B, 1)
        return self.fusion(e, aux)          # (B, 384)

    def param_groups(self, lr_head: float, lr_backbone: Optional[float] = None) -> List[dict]:
        """
        Split parameter groups for differential learning rates:
          - head (attention_pool + fusion + edge_branch): lr_head
          - backbone where requires_grad=True (LoRA adapter or unfrozen blocks): lr_backbone
        """
        head_params = []
        if self.attn_pool is not None:
            head_params += list(self.attn_pool.parameters())
        head_params += list(self.fusion.parameters())
        if self.edge_branch is not None:
            head_params += list(self.edge_branch.parameters())
        groups = [{"params": [p for p in head_params if p.requires_grad], "lr": lr_head}]
        bb_trainable = [p for p in self.backbone.parameters() if p.requires_grad]
        if bb_trainable:
            groups.append({"params": bb_trainable, "lr": lr_backbone if lr_backbone is not None else lr_head})
        return groups

def arcface_logits(loss_fn, embeddings: torch.Tensor) -> torch.Tensor:
    """
    Logits at evaluate/inference: ``s · cos(θ)`` (no margin — margin is used only during training)

    Uses ``loss_fn.get_cosine()`` from pytorch-metric-learning directly — CosineSimilarity
    of the library normalizes both embedding and weight (W stored as shape (emb_dim, num_classes))
    itself, so it is correct on the "angle" for every version of the library
    """
    cos = loss_fn.get_cosine(embeddings)  # (B, num_classes), cosine of angle between vectors
    return cos * loss_fn.scale


class AdaptiveArcFaceLoss(torch.nn.Module):
    """
    Per-class margin ArcFace (for LGMS)

    margin_per_class: list/array of size num_classes in degrees
    Each sample uses the margin of its own label
    """
    def __init__(self, num_classes: int, embedding_size: int,
                 margin_per_class: List[float], scale: float = 64.0):
        super().__init__()
        from pytorch_metric_learning.distances import CosineSimilarity
        from pytorch_metric_learning.utils import common_functions as c_f
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.scale = scale
        self.margin_per_class = np.array(margin_per_class, dtype=np.float64)
        self.margins_rad = np.radians(self.margin_per_class)
        self.W = torch.nn.Parameter(torch.Tensor(embedding_size, num_classes))
        # Xavier init like pytorch-metric-learning
        torch.nn.init.xavier_uniform_(self.W)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")
        self._cosine = CosineSimilarity()
        self._c_f = c_f

    def get_cosine(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Follows pytorch-metric-learning: normalize embeddings and W
        emb_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        W_norm = torch.nn.functional.normalize(self.W, p=2, dim=0)
        return torch.mm(emb_norm, W_norm)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Return mean loss — call compute_loss per-sample then mean
        so CAHM can weight per-sample afterwards (for per-sample loss call compute_loss_dict)
        """
        d = self.compute_loss_dict(embeddings, labels)
        return d["losses"].mean()

    def compute_loss_dict(self, embeddings: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Return dict {"losses": (B,) per-sample, "logits": (B,C)}
        for CAHM that needs per-sample weighting
        """
        dtype, device = embeddings.dtype, embeddings.device
        # Move W/margins to matching device/dtype
        W = self._c_f.to_device(self.W, device=device, dtype=dtype)
        margins = torch.as_tensor(self.margins_rad, device=device, dtype=dtype)  # (C,)
        # cosine (B, C)
        cosine = self.get_cosine(embeddings)  # internal uses self.W — need to ensure uses moved W; get_cosine uses normalized W directly
        # But get_cosine looks at self.W itself — cannot override to use moved W; do manual:
        emb_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        W_norm = torch.nn.functional.normalize(W, p=2, dim=0)
        cosine = torch.mm(emb_norm, W_norm)
        # mask
        B = labels.size(0)
        mask = torch.zeros(B, self.num_classes, dtype=dtype, device=device)
        mask[torch.arange(B, device=device), labels] = 1
        cosine_target = cosine[mask == 1]  # (B,)
        angles = torch.acos(torch.clamp(cosine_target, -1 + 1e-7, 1 - 1e-7))
        m_per_sample = margins[labels]  # (B,)
        # cos(theta + m) as in ArcFace
        cos_theta_plus_m = torch.cos(angles + m_per_sample)
        cos_theta = torch.cos(angles)
        # keep monotonically decreasing (same as ArcFace)
        # if theta + m > pi fall back
        cond = angles <= (np.pi - m_per_sample)
        # m in radians must be converted to tensor for sin
        modified = torch.where(cond, cos_theta_plus_m, cos_theta - m_per_sample * torch.sin(torch.as_tensor(m_per_sample)))
        diff = (modified - cosine_target).unsqueeze(1)  # (B,1)
        logits = cosine + (mask * diff)
        logits = logits * self.scale
        losses = self.cross_entropy(logits, labels)  # (B,)
        return {"losses": losses, "logits": logits, "cosine": cosine}

    # Like pytorch-metric-learning: has attributes W and scale for arcface_logits
    @property
    def W_t(self):
        return self.W.t()


def count_trainable(model: nn.Module) -> int:
    """Count number of trainable parameters (used to verify LoRA/frozen is working correctly)"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
