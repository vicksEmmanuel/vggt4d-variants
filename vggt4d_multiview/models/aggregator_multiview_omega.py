"""
AggregatorFor4DMultiViewOmega — Multi-view VGGT4D with VGGT-Omega architecture components.

Same 3-way attention decomposition as AggregatorFor4DMultiView (frame/temporal/crossview),
but uses VGGT-Omega's SelfAttentionBlock as the building block with register tokens,
Omega-style RoPE (normalize_coords="max"), and Omega-style DinoVisionTransformer patch embed.

::

    aa_order = ["frame", "temporal", "crossview"]

    Frame     (B*S, P, C)   — within-frame spatial attention
    Temporal  (B*V, T*P, C) — same view across time
    Crossview (B*T, V*P, C) — different views, same time

Weights are loaded from the VGGT-Omega checkpoint:
    frame_blocks     ← checkpoint aggregator.frame_blocks
    temporal_blocks  ← checkpoint aggregator.inter_frame_blocks
    crossview_blocks ← copy from temporal_blocks
"""

from __future__ import annotations

import gc
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from vggt.models.aggregator import slice_expand_and_flatten
from vggt_omega.models.layers import Mlp, RopePositionEmbedding, SelfAttentionBlock
from vggt_omega.models.layers.vision_transformer import DinoVisionTransformer

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class AggregatorFor4DMultiViewOmega(nn.Module):
    """Multi-view aggregator using VGGT-Omega architecture components.

    Parameters
    ----------
    patch_size : int
        Patch size for patch embedding (default 16, same as VGGT-Omega).
    embed_dim : int
        Token embedding dimension (default 1024).
    depth : int
        Number of blocks *per attention type* (default 24 — produces 72 total blocks).
    num_heads : int
        Number of attention heads (default 16).
    mlp_ratio : float
        Ratio of MLP hidden dim to embedding dim (default 4.0).
    num_register_tokens : int
        Number of register tokens (default 16, same as VGGT-Omega).
    cached_layer_indices : tuple[int, ...]
        Layer indices whose outputs are preserved (default (4, 11, 17, 23)).
    """

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 16,
        cached_layer_indices: tuple[int, ...] = (4, 11, 17, 23),
    ):
        super().__init__()

        self.patch_embed = _build_patch_embed(patch_size=patch_size, embed_dim=embed_dim)
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=100,
            normalize_coords="max",
            dtype=torch.float32,
        )

        # Frame blocks (within-frame spatial attention — Omega frame_blocks)
        self.frame_blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=embed_dim, num_heads=num_heads, ffn_ratio=mlp_ratio,
                qkv_bias=True, proj_bias=True, ffn_bias=True,
                ffn_layer=Mlp, init_values=1e-5, use_qk_norm=True, mask_k_bias=True,
            ) for _ in range(depth)
        ])

        # Temporal blocks (same view, different time — Omega inter_frame_blocks)
        self.temporal_blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=embed_dim, num_heads=num_heads, ffn_ratio=mlp_ratio,
                qkv_bias=True, proj_bias=True, ffn_bias=True,
                ffn_layer=Mlp, init_values=1e-5, use_qk_norm=True, mask_k_bias=True,
            ) for _ in range(depth)
        ])

        # Crossview blocks (different views, same time — copy of temporal after loading)
        self.crossview_blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=embed_dim, num_heads=num_heads, ffn_ratio=mlp_ratio,
                qkv_bias=True, proj_bias=True, ffn_bias=True,
                ffn_layer=Mlp, init_values=1e-5, use_qk_norm=True, mask_k_bias=True,
            ) for _ in range(depth)
        ])

        self.depth = depth
        self.patch_size = patch_size
        self.cached_layer_indices = set(cached_layer_indices)

        # Omega-style: first frame gets unique camera/register tokens, rest share
        self.camera_token = nn.Parameter(torch.empty(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.empty(1, 2, num_register_tokens, embed_dim))
        self.patch_token_start = 1 + num_register_tokens

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.camera_token, std=1e-3)
        nn.init.normal_(self.register_token, std=1e-3)

    def forward(
        self,
        images: torch.Tensor,
        dyn_masks: Optional[torch.Tensor] = None,
        enable_memory_saving: bool = True,
        view_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Optional[torch.Tensor]], int, dict, torch.Tensor]:
        """
        Parameters
        ----------
        images : (B, S, 3, H, W)  [0, 1]
            Interleaved sequence: [front_0, rear_0, front_1, rear_1, ...].
            S = V × T.
        dyn_masks : (B, S, H, W) or None
            Per-frame dynamic masks.
        enable_memory_saving : bool
            Offload non-preserved intermediates to save VRAM.
        view_ids : (B, S) or None
            View index per frame. If None, assumes V=1.

        Returns
        -------
        output_list : list[Tensor | None]
            Each element has shape (B, S, P, 3*C) — concatenated frame + temporal + crossview.
        patch_start_idx : int
        qk_dict : dict (empty — not used in Omega variant)
        patch_tokens : Tensor
        """
        B, S, C_in, H, W = images.shape
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        if view_ids is None:
            view_ids = torch.zeros(B, S, dtype=torch.long, device=images.device)
        V = int(view_ids.max().item()) + 1
        T = S // V
        assert V * T == S, f"S={S} is not divisible by V={V}"

        # ── Patch embedding ───────────────────────────────────────
        images_norm = (images - self._resnet_mean) / self._resnet_std
        images_flat = images_norm.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images_flat)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Dynamic mask
        if dyn_masks is not None:
            dyn_masks = F.max_pool2d(
                dyn_masks.float(), kernel_size=self.patch_size, stride=self.patch_size,
            )
            dyn_masks = rearrange(dyn_masks, "b s h w -> b s (h w)") > 0.5

        # Special tokens (Omega-style: first frame unique, rest share)
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        patch_grid_h = H // self.patch_size
        patch_grid_w = W // self.patch_size
        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(H=patch_grid_h, W=patch_grid_w)
            frame_rope = (
                rope_sin.to(device=tokens.device, dtype=torch.float32),
                rope_cos.to(device=tokens.device, dtype=torch.float32),
            )

        _, P_total, C = tokens.shape

        preserve_idx = [4, 11, 17, 23]

        frame_idx = temporal_idx = crossview_idx = 0
        output_list: List[Optional[torch.Tensor]] = [None] * (self.depth * B)

        for i in range(self.depth):
            # 1. Frame attention (within-frame spatial)
            tokens, frame_out = self._run_frame_attention(
                tokens, B, S, P_total, C, frame_idx, frame_rope,
            )
            frame_idx += 1

            # 2. Temporal attention (same view, different time)
            tokens, temp_out = self._run_temporal_attention(
                tokens, B, S, P_total, C, temporal_idx, V, T,
            )
            temporal_idx += 1

            # 3. Crossview attention (different views, same time)
            tokens, cross_out = self._run_crossview_attention(
                tokens, B, S, P_total, C, crossview_idx, V, T,
            )
            crossview_idx += 1

            # Concatenate frame + temporal + crossview → (B, S, P, 3*C)
            concat_inter = torch.cat([frame_out, temp_out, cross_out], dim=-1)
            output_list[i * B] = concat_inter

            if enable_memory_saving and i not in preserve_idx:
                for j in range(B):
                    output_list[i * B + j] = None
                del concat_inter, frame_out, temp_out, cross_out

        if enable_memory_saving:
            del tokens

        qk_dict = {
            "global_q": torch.empty(0), "global_k": torch.empty(0),
            "frame_q": torch.empty(0), "frame_k": torch.empty(0),
        }
        return output_list, self.patch_token_start, qk_dict, patch_tokens

    # ── Attention processors ──────────────────────────────────────────

    def _run_frame_attention(
        self, tokens: torch.Tensor, B: int, S: int, P: int, C: int,
        block_idx: int, rope_sincos: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Within-frame spatial attention (B*S, P, C) with Omega-style RoPE."""
        tokens = tokens.view(B * S, P, C)
        tokens = self.frame_blocks[block_idx](tokens, rope_sincos)
        frame_out = tokens.view(B, S, P, C)
        return tokens, frame_out

    def _run_temporal_attention(
        self, tokens: torch.Tensor, B: int, S: int, P: int, C: int,
        block_idx: int, V: int, T: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Same view, different time (B*V, T*P, C) — full global attention."""
        tokens_view = tokens.view(B, S, P, C)
        tokens_t = rearrange(tokens_view, "b (t v) p c -> b t v p c", t=T, v=V)
        tokens_t = rearrange(tokens_t, "b t v p c -> (b v) (t p) c")
        tokens_t = self.temporal_blocks[block_idx](tokens_t, None)
        tokens_out = rearrange(tokens_t, "(b v) (t p) c -> b t v p c", b=B, v=V, t=T)
        tokens_out = rearrange(tokens_out, "b t v p c -> b (t v) p c")
        temp_out = tokens_out.clone()
        tokens = tokens_out.reshape(B * S, P, C)
        return tokens, temp_out

    def _run_crossview_attention(
        self, tokens: torch.Tensor, B: int, S: int, P: int, C: int,
        block_idx: int, V: int, T: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Different views, same time (B*T, V*P, C) — full global attention."""
        tokens_view = tokens.view(B, S, P, C)
        tokens_c = rearrange(tokens_view, "b (t v) p c -> b t v p c", t=T, v=V)
        tokens_c = rearrange(tokens_c, "b t v p c -> (b t) (v p) c")
        tokens_c = self.crossview_blocks[block_idx](tokens_c, None)
        tokens_out = rearrange(tokens_c, "(b t) (v p) c -> b t v p c", b=B, t=T, v=V)
        tokens_out = rearrange(tokens_out, "b t v p c -> b (t v) p c")
        cross_out = tokens_out.clone()
        tokens = tokens_out.reshape(B * S, P, C)
        return tokens, cross_out

    def clear_inference_cache(self):
        gc.collect()
        torch.cuda.empty_cache()


def _build_patch_embed(patch_size: int, embed_dim: int) -> DinoVisionTransformer:
    model = DinoVisionTransformer(
        img_size=224,
        patch_size=patch_size,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="max",
        pos_embed_rope_dtype="fp32",
        embed_dim=embed_dim,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-5,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    )
    model.init_weights()
    return model
