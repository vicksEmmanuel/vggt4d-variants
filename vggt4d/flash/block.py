"""
FlashBlockFor4D — VGGT4D block with FlashVGGT's kv_downfactor in attention.

Inherits VGGT4D's Q/K return pattern while using FlashVGGT-style
attention with optional spatial KV compression.

Frame attention blocks use kv_downfactor=1 (full resolution).
Global attention blocks use kv_downfactor from config (default 4).
"""

from typing import Tuple

import torch
from torch import Tensor, nn

from vggt.layers.drop_path import DropPath
from vggt.layers.layer_scale import LayerScale
from vggt.layers.mlp import Mlp

from vggt4d.flash.attention import FlashAttentionFor4D


class FlashBlockFor4D(nn.Module):
    """Block that merges FlashVGGT's kv_downfactor with VGGT4D's Q/K return."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer=None,
        norm_layer=None,
        attn_class=None,  # ignored — we use FlashAttentionFor4D
        ffn_layer=None,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
        kv_downfactor: int = 1,
    ):
        super().__init__()
        if act_layer is None:
            act_layer = nn.GELU
        if norm_layer is None:
            norm_layer = nn.LayerNorm
        if ffn_layer is None:
            ffn_layer = Mlp

        self.norm1 = norm_layer(dim)
        self.attn = FlashAttentionFor4D(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
            kv_downfactor=kv_downfactor,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop, bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(
        self,
        x: Tensor,
        pos=None,
        pH: int = None,
        pW: int = None,
        patch_start_idx: int = None,
        keyframe_indices: Tensor = None,
        is_frame_attn: bool = True,
        layer_id: int = 0,
        dyn_masks: Tensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns (output, q, k) — compatible with VGGT4D aggregator."""

        def attn_residual_func(x_in: Tensor, pos_in=None) -> Tuple[Tensor, Tensor, Tensor]:
            x_out, q_out, k_out = self.attn(
                self.norm1(x_in),
                pos=pos_in,
                pH=pH,
                pW=pW,
                patch_start_idx=patch_start_idx,
                keyframe_indices=keyframe_indices,
                is_frame_attn=is_frame_attn,
                layer_id=layer_id,
                dyn_masks=dyn_masks,
            )
            x_out = self.ls1(x_out)
            return x_out, q_out, k_out

        def ffn_residual_func(x_in: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x_in)))

        # Always use the non-drop-path path for inference (matching VGGT4D's BlockFor4D behavior)
        # This also ensures Q/K are always returned.
        attn_x, attn_q, attn_k = attn_residual_func(x, pos)
        x = x + attn_x
        x = x + ffn_residual_func(x)

        return x, attn_q, attn_k
