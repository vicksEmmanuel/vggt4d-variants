"""
AggregatorFor4DMultiView — 3-way decomposed attention for VGGT4D.

Extends the base VGGT Aggregator with three structurally distinct
attention modes that decompose the original "global" attention into
temporal (same view, different time) and crossview (different views,
same time) pathways.

::

    aa_order = ["frame", "temporal", "crossview"]

    Frame     (B*S, P, C)  — within-frame spatial attention  ← checkpoint frame_blocks
    Temporal  (B*V, T*P, C) — same view across time           ← checkpoint global_blocks
    Crossview (B*T, V*P, C) — different views, same time      ← checkpoint global_blocks

Sequence layout
    Interleaved:  [f0, r0, f1, r1, f2, r2, ...]
    S = V × T   (V = num_views, T = temporal frames per view)

Weights
    frame_blocks     ← loaded from VGGT4D frame_blocks (indices 0-23)
    temporal_blocks  ← loaded from VGGT4D global_blocks (indices 0-23)
    crossview_blocks ← loaded from VGGT4D global_blocks (indices 0-23)
    The VGGT base model was trained on diverse camera configurations,
    so the same global-block weights encode both temporal smoothness
    and cross-view correspondence priors.
"""

from __future__ import annotations

import gc
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from vggt.models.aggregator import Aggregator, slice_expand_and_flatten
from vggt4d.layers.block import BlockFor4D


class AggregatorFor4DMultiView(Aggregator):
    """Multi-view VGGT4D aggregator with 3-way decomposed attention.

    Parameters
    ----------
    img_size : int
        Input image size in pixels (default 518).
    patch_size : int
        Patch size for PatchEmbed (default 14).
    embed_dim : int
        Token embedding dimension (default 1024).
    depth : int
        Number of blocks *per attention type* (default 24 — produces
        72 total blocks: 24 frame + 24 temporal + 24 crossview).
    num_heads : int
        Number of attention heads (default 16).
    mlp_ratio : float
        Ratio of MLP hidden dim to embedding dim (default 4.0).
    num_register_tokens : int
        Number of register tokens (default 4).
    qkv_bias : bool
        Bias in QKV projections.
    proj_bias : bool
        Bias in output projection.
    ffn_bias : bool
        Bias in MLP layers.
    patch_embed : str
        Patch embedding type (default "dinov2_vitl14_reg").
    aa_block_size : int
        Blocks per attention-type call (default 1).
    qk_norm : bool
        Apply QK normalisation (default True).
    rope_freq : int
        RoPE base frequency; -1 disables (default 100).
    init_values : float
        Init scale for LayerScale (default 0.01).
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 4,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        patch_embed: str = "dinov2_vitl14_reg",
        aa_block_size: int = 1,
        qk_norm: bool = True,
        rope_freq: int = 100,
        init_values: float = 0.01,
    ):
        # ── build the base aggregator with BlockFor4D blocks ──────────
        # We override aa_order to include all three types; the base
        # Aggregator.__init__ creates frame_blocks + global_blocks.
        # We keep frame_blocks, rename global_blocks → temporal_blocks,
        # and add crossview_blocks as copies.
        super_kwargs = dict(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_register_tokens=num_register_tokens,
            block_fn=BlockFor4D,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            patch_embed=patch_embed,
            aa_order=["frame", "temporal", "crossview"],
            aa_block_size=aa_block_size,
            qk_norm=qk_norm,
            rope_freq=rope_freq,
            init_values=init_values,
        )
        super().__init__(**super_kwargs)

        # ── Add crossview blocks (initialised from global weights) ────
        # The base class created self.frame_blocks and self.global_blocks.
        # We repurpose self.global_blocks as temporal_blocks and create
        # crossview_blocks separately.
        self.temporal_blocks = self.global_blocks
        del self.global_blocks  # clear confusing alias

        self.crossview_blocks = nn.ModuleList(
            [
                BlockFor4D(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.preserve_layer_idx = [4, 11, 17, 23]

    # ── Main forward ──────────────────────────────────────────────────

    def forward(
        self,
        images: torch.Tensor,
        dyn_masks: Optional[torch.Tensor] = None,
        enable_memory_saving: bool = True,
        view_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], int, dict, torch.Tensor]:
        """
        Parameters
        ----------
        images : (B, S, 3, H, W)  [0, 1]
            Interleaved sequence:  [front_0, rear_0, front_1, rear_1, ...].
            S = V × T.
        dyn_masks : (B, S, H, W) or None
            Per-frame dynamic masks.
        enable_memory_saving : bool
            Offload non-preserved intermediates to save VRAM.
        view_ids : (B, S) or None
            View index per frame.  If None, assumes V=1 (degenerates to
            standard VGGT4D).

        Returns
        -------
        output_list : list[Tensor]  length = aa_block_num × B
            Each element has shape (B, S, P, 3*C) — concatenated
            frame + temporal + crossview intermediates.
        patch_start_idx : int
        qk_dict : dict
        patch_tokens : Tensor
        """
        B, S, C_in, H, W = images.shape
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        if view_ids is None:
            # Degenerate: single view  →  V=1, T=S
            view_ids = torch.zeros(B, S, dtype=torch.long, device=images.device)
        V = int(view_ids.max().item()) + 1
        T = S // V
        assert V * T == S, f"S={S} is not divisible by V={V} (got view_ids max={V-1})"

        # ── Patch embedding ───────────────────────────────────────
        images_norm = (images - self._resnet_mean) / self._resnet_std
        images_flat = images_norm.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images_flat)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Dynamic mask downscaling (same as VGGT4D)
        if dyn_masks is not None:
            dyn_masks = F.max_pool2d(
                dyn_masks.float(), kernel_size=self.patch_size, stride=self.patch_size
            )
            dyn_masks = rearrange(dyn_masks, "b s h w -> b s (h w)") > 0.5

        # Special tokens
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        # Position encodings
        pos = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S, H // self.patch_size, W // self.patch_size, device=images.device
            )
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2, device=images.device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        _, P, C = tokens.shape  # P now includes special tokens

        frame_idx = 0
        temporal_idx = 0
        crossview_idx = 0
        output_list: List[Optional[torch.Tensor]] = [None] * (self.aa_block_num * B)

        # Q/K tracking (for dynamic mask extraction — match VGGT4D format)
        global_q_list: List[torch.Tensor] = []
        frame_q_list: List[torch.Tensor] = []
        global_k_list: List[torch.Tensor] = []
        frame_k_list: List[torch.Tensor] = []

        for i in range(self.aa_block_num):
            frame_intermediates = []
            temporal_intermediates = []
            crossview_intermediates = []
            attn_type: str
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, f_inter, f_q, f_k = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos, dyn_masks=dyn_masks,
                    )
                    frame_intermediates = f_inter
                    frame_q_list.append(f_q.detach().cpu())
                    frame_k_list.append(f_k.detach().cpu())
                    del f_q, f_k
                elif attn_type == "temporal":
                    tokens, temporal_idx, t_inter, t_q, t_k = self._process_temporal_attention(
                        tokens, B, S, P, C, temporal_idx, V, T, pos=pos, dyn_masks=dyn_masks,
                    )
                    temporal_intermediates = t_inter
                    # Temporal Q/K has shape (1, aa_block, 16, T*P, 64) — skip stacking
                    # since qk_dict is only needed for dynamic mask extraction (unused here).
                    del t_q, t_k
                elif attn_type == "crossview":
                    tokens, crossview_idx, c_inter, c_q, c_k = self._process_crossview_attention(
                        tokens, B, S, P, C, crossview_idx, V, T, pos=pos, dyn_masks=dyn_masks,
                    )
                    crossview_intermediates = c_inter
                    # Crossview Q/K has shape (1, aa_block, 16, V*P, 64) — differs from
                    # temporal Q/K shape, so cannot be stacked together. Not needed for
                    # multi-view pipeline (qk_dict unused).
                    del c_q, c_k
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            # Concat all three intermediates  →  (B, S, P, 3C)
            for j in range(len(frame_intermediates)):
                concat_inter = torch.cat(
                    [frame_intermediates[j], temporal_intermediates[j], crossview_intermediates[j]],
                    dim=-1,
                )
                output_list[i * B + j] = concat_inter

            if enable_memory_saving:
                if i not in self.preserve_layer_idx:
                    for j in range(B):
                        tmp = output_list[i * B + j]
                        output_list[i * B + j] = None
                        del tmp
                del concat_inter, frame_intermediates, temporal_intermediates, crossview_intermediates

        # Build qk_dict (compatible with existing dynamic mask extraction)
        if global_q_list:
            global_q = torch.stack(global_q_list, dim=0)
            global_k = torch.stack(global_k_list, dim=0)
            frame_q = torch.stack(frame_q_list, dim=0)
            frame_k = torch.stack(frame_k_list, dim=0)
        else:
            global_q = global_k = frame_q = frame_k = torch.empty(0)

        if enable_memory_saving:
            del tokens

        qk_dict = {
            "global_q": global_q,
            "global_k": global_k,
            "frame_q": frame_q,
            "frame_k": frame_k,
        }

        if enable_memory_saving:
            self.clear_inference_cache()

        return output_list, self.patch_start_idx, qk_dict, patch_tokens

    # ── Attention processors ──────────────────────────────────────────

    def _process_frame_attention(
        self, tokens, B, S, P, C, frame_idx, pos=None, dyn_masks=None,
    ):
        """Within-frame spatial attention  (B*S, P, C)."""
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)
        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates, attn_q, attn_k = [], [], []
        for _ in range(self.aa_block_size):
            if self.training:
                tokens, q, k = checkpoint(
                    self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant,
                )
            else:
                tokens, q, k = self.frame_blocks[frame_idx](
                    tokens, pos=pos, is_frame_attn=True, layer_id=frame_idx, dyn_masks=dyn_masks,
                )
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
            attn_q.append(q)
            attn_k.append(k)

        attn_q = torch.stack(attn_q, dim=0) if attn_q else torch.empty(0)
        attn_k = torch.stack(attn_k, dim=0) if attn_k else torch.empty(0)
        return tokens, frame_idx, intermediates, attn_q, attn_k

    def _process_temporal_attention(
        self, tokens, B, S, P, C, temporal_idx, V, T, pos=None, dyn_masks=None,
    ):
        """Same view, different time  (B*V, T*P, C)."""
        # Reshape:  (B*S, P, C)  →  (B, V, T, P, C)  →  (B*V, T*P, C)
        # Interleaved input:  [v0_t0, v1_t0, v0_t1, v1_t1, ...]
        tokens_view = tokens.view(B, S, P, C)  # (B, S, P, C)
        tokens_t = rearrange(tokens_view, "b (t v) p c -> b t v p c", t=T, v=V)
        tokens_t = rearrange(tokens_t, "b t v p c -> (b v) (t p) c")

        pos_t = None
        if pos is not None:
            pos_view = pos.view(B, S, P, 2)
            pos_t = rearrange(pos_view, "b (t v) p d -> b t v p d", t=T, v=V)
            pos_t = rearrange(pos_t, "b t v p d -> (b v) (t p) d")

        intermediates, attn_q, attn_k = [], [], []
        for _ in range(self.aa_block_size):
            if self.training:
                tokens_t, q, k = checkpoint(
                    self.temporal_blocks[temporal_idx], tokens_t, pos_t,
                    use_reentrant=self.use_reentrant,
                )
            else:
                tokens_t, q, k = self.temporal_blocks[temporal_idx](
                    tokens_t, pos=pos_t, is_frame_attn=False, layer_id=temporal_idx,
                    dyn_masks=None,  # no dynamic mask for temporal attention
                )
            temporal_idx += 1
            # Reshape back:  (B*V, T*P, C)  →  (B, V, T, P, C)  →  (B, T*V, P, C)
            inter = rearrange(tokens_t, "(b v) (t p) c -> b t v p c", b=B, v=V, t=T)
            inter = rearrange(inter, "b t v p c -> b (t v) p c")
            intermediates.append(inter)
            attn_q.append(q)
            attn_k.append(k)

        attn_q = torch.stack(attn_q, dim=0) if attn_q else torch.empty(0)
        attn_k = torch.stack(attn_k, dim=0) if attn_k else torch.empty(0)
        # Return tokens in (B*S, P, C) shape for next attention type
        tokens = rearrange(tokens_t, "(b v) (t p) c -> b t v p c", b=B, v=V, t=T)
        tokens = rearrange(tokens, "b t v p c -> b (t v) p c")
        tokens = tokens.reshape(B * S, P, C)
        return tokens, temporal_idx, intermediates, attn_q, attn_k

    def _process_crossview_attention(
        self, tokens, B, S, P, C, crossview_idx, V, T, pos=None, dyn_masks=None,
    ):
        """Different views, same time  (B*T, V*P, C)."""
        tokens_view = tokens.view(B, S, P, C)  # (B, S, P, C)
        tokens_c = rearrange(tokens_view, "b (t v) p c -> b t v p c", t=T, v=V)
        tokens_c = rearrange(tokens_c, "b t v p c -> (b t) (v p) c")

        pos_c = None
        if pos is not None:
            pos_view = pos.view(B, S, P, 2)
            pos_c = rearrange(pos_view, "b (t v) p d -> b t v p d", t=T, v=V)
            pos_c = rearrange(pos_c, "b t v p d -> (b t) (v p) d")

        intermediates, attn_q, attn_k = [], [], []
        for _ in range(self.aa_block_size):
            if self.training:
                tokens_c, q, k = checkpoint(
                    self.crossview_blocks[crossview_idx], tokens_c, pos_c,
                    use_reentrant=self.use_reentrant,
                )
            else:
                tokens_c, q, k = self.crossview_blocks[crossview_idx](
                    tokens_c, pos=pos_c, is_frame_attn=False, layer_id=crossview_idx,
                    dyn_masks=None,  # no dynamic mask for crossview attention
                )
            crossview_idx += 1
            inter = rearrange(tokens_c, "(b t) (v p) c -> b t v p c", b=B, t=T, v=V)
            inter = rearrange(inter, "b t v p c -> b (t v) p c")
            intermediates.append(inter)
            attn_q.append(q)
            attn_k.append(k)

        attn_q = torch.stack(attn_q, dim=0) if attn_q else torch.empty(0)
        attn_k = torch.stack(attn_k, dim=0) if attn_k else torch.empty(0)
        tokens = rearrange(tokens_c, "(b t) (v p) c -> b t v p c", b=B, t=T, v=V)
        tokens = rearrange(tokens, "b t v p c -> b (t v) p c")
        tokens = tokens.reshape(B * S, P, C)
        return tokens, crossview_idx, intermediates, attn_q, attn_k

    def clear_inference_cache(self):
        if hasattr(self, "rope") and self.rope is not None:
            if hasattr(self.rope, "frequency_cache"):
                self.rope.frequency_cache.clear()
        if hasattr(self, "position_getter") and self.position_getter is not None:
            if hasattr(self.position_getter, "position_cache"):
                self.position_getter.position_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()
