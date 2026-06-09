"""
AggregatorFor4DMultiViewFlash — Multi-view VGGT4D with FlashVGGT KV compression.

Same 3-way attention decomposition as AggregatorFor4DMultiView (frame/temporal/crossview),
but uses FlashBlockFor4D with kv_downfactor for temporal and crossview attention.

This reduces the memory of temporal attention from O(T·P)² to O(T·P·P/kv_downfactor).

Weights are loaded from the FlashVGGT checkpoint (frame_blocks ← checkpoint frame_blocks,
temporal_blocks ← checkpoint global_blocks, crossview_blocks ← copy from temporal).
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
from vggt4d.flash.block import FlashBlockFor4D


class AggregatorFor4DMultiViewFlash(Aggregator):
    """Multi-view VGGT4D with FlashVGGT KV compression in temporal/crossview blocks.

    Parameters
    ----------
    kv_downfactor : int
        Spatial compression factor for temporal and crossview attention (default 4).
    temporal_kv_downfactor : int
        Separate factor for temporal blocks (default: kv_downfactor).
    crossview_kv_downfactor : int
        Separate factor for crossview blocks (default: kv_downfactor).
    """

    def __init__(
        self,
        kv_downfactor: int = 4,
        temporal_kv_downfactor: int | None = None,
        crossview_kv_downfactor: int | None = None,
        **kwargs,
    ):
        self.flash_kv_downfactor = kv_downfactor
        self.flash_temporal_kdf = temporal_kv_downfactor if temporal_kv_downfactor is not None else kv_downfactor
        self.flash_crossview_kdf = crossview_kv_downfactor if crossview_kv_downfactor is not None else kv_downfactor

        # Save builder params
        self._fdim = kwargs.get("embed_dim", 1024)
        self._fheads = kwargs.get("num_heads", 16)
        self._fmlp = kwargs.get("mlp_ratio", 4.0)
        self._fqkv_b = kwargs.get("qkv_bias", True)
        self._fproj_b = kwargs.get("proj_bias", True)
        self._ffffn_b = kwargs.get("ffn_bias", True)
        self._finit = kwargs.get("init_values", 0.01)
        self._fqkn = kwargs.get("qk_norm", True)

        # Build via base Aggregator (not AggregatorFor4DMultiView — we build blocks ourselves)
        kwargs["aa_order"] = ["frame", "temporal", "crossview"]
        kwargs["block_fn"] = FlashBlockFor4D  # placeholder
        super().__init__(**kwargs)

        # Replace blocks with properly configured FlashBlockFor4D blocks
        self._build_flash_blocks()

        # Now global_blocks is FlashBlockFor4D. Alias as temporal_blocks.
        self.temporal_blocks = self.global_blocks
        # Create crossview blocks (copy weights from temporal_blocks)
        self.crossview_blocks = nn.ModuleList(
            [FlashBlockFor4D(
                dim=self._fdim, num_heads=self._fheads,
                mlp_ratio=self._fmlp,
                qkv_bias=self._fqkv_b, proj_bias=self._fproj_b,
                ffn_bias=self._ffffn_b,
                init_values=self._finit, qk_norm=self._fqkn, rope=self.rope,
                kv_downfactor=1,  # crossview uses reshaped layout (B*T, V*P, C) — not compatible with _downscale_kv
            ) for _ in range(self.depth)]
        )
        # Copy weights from temporal → crossview
        for i in range(self.depth):
            for src_p, dst_p in zip(self.temporal_blocks[i].parameters(),
                                    self.crossview_blocks[i].parameters()):
                dst_p.data.copy_(src_p.data)

        self.preserve_layer_idx = [4, 11, 17, 23]

    def _build_flash_blocks(self):
        """Replace frame_blocks and global_blocks with FlashBlockFor4D."""
        depth = self.depth
        frame_blocks = nn.ModuleList()
        global_blocks = nn.ModuleList()

        for idx in range(depth):
            frame_blocks.append(FlashBlockFor4D(
                dim=self._fdim, num_heads=self._fheads,
                mlp_ratio=self._fmlp,
                qkv_bias=self._fqkv_b, proj_bias=self._fproj_b,
                ffn_bias=self._ffffn_b,
                init_values=self._finit, qk_norm=self._fqkn, rope=self.rope,
                kv_downfactor=1,  # frame blocks always full res
            ))
            global_blocks.append(FlashBlockFor4D(
                dim=self._fdim, num_heads=self._fheads,
                mlp_ratio=self._fmlp,
                qkv_bias=self._fqkv_b, proj_bias=self._fproj_b,
                ffn_bias=self._ffffn_b,
                init_values=self._finit, qk_norm=self._fqkn, rope=self.rope,
                kv_downfactor=1,  # temporal uses reshaped layout (B*V, T*P, C) — not compatible with _downscale_kv
            ))

        # Copy weights from old blocks (built by parent with BlockFor4D)
        # to new FlashBlockFor4D blocks (same internal param shapes)
        for i in range(depth):
            self._copy_params(self.frame_blocks[i], frame_blocks[i])
            self._copy_params(self.frame_blocks[i], global_blocks[i])  # temporal = frame init, will be overwritten by checkpoint

        self.frame_blocks = frame_blocks
        self.global_blocks = global_blocks

    def _copy_params(self, src, dst):
        for sp, dp in zip(src.parameters(), dst.parameters()):
            dp.data.copy_(sp.data)
        for sb, db in zip(src.buffers(), dst.buffers()):
            db.data.copy_(sb.data)

    # ── Attention helpers ────────────────────────────────────────────────

    def _get_patch_dims(self, P: int) -> Tuple[int, int, int]:
        """Estimate pH, pW from token count."""
        ps_idx = self.patch_start_idx
        pt = P - ps_idx
        side = int(pt ** 0.5)
        if side * side != pt:
            side = 37  # default for 518px square
        return side, side, ps_idx

    def _process_frame_attention(
        self, tokens, B, S, P, C, frame_idx, pos=None, dyn_masks=None,
    ):
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)
        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates, attn_q, attn_k = [], [], []
        for _ in range(self.aa_block_size):
            tokens, q, k = self.frame_blocks[frame_idx](
                tokens, pos=pos,
                is_frame_attn=True, layer_id=frame_idx, dyn_masks=dyn_masks,
            )
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
            attn_q.append(q)
            attn_k.append(k)
        return tokens, frame_idx, intermediates, \
            torch.stack(attn_q, dim=0) if attn_q else torch.empty(0), \
            torch.stack(attn_k, dim=0) if attn_k else torch.empty(0)

    def _process_temporal_attention(
        self, tokens, B, S, P, C, temporal_idx, V, T, pos=None, dyn_masks=None,
    ):
        """Same view, different time — with KV compression."""
        tokens_view = tokens.view(B, S, P, C)
        tokens_t = rearrange(tokens_view, "b (t v) p c -> b t v p c", t=T, v=V)
        tokens_t = rearrange(tokens_t, "b t v p c -> (b v) (t p) c")

        pos_t = None
        if pos is not None:
            pos_view = pos.view(B, S, P, 2)
            pos_t = rearrange(pos_view, "b (t v) p d -> b t v p d", t=T, v=V)
            pos_t = rearrange(pos_t, "b t v p d -> (b v) (t p) d")

        pH, pW, ps_idx = self._get_patch_dims(P)
        intermediates, attn_q, attn_k = [], [], []
        for _ in range(self.aa_block_size):
            tokens_t, q, k = self.temporal_blocks[temporal_idx](
                tokens_t, pos=pos_t,
                pH=pH, pW=pW, patch_start_idx=ps_idx,
                keyframe_indices=None,
                is_frame_attn=False, layer_id=temporal_idx, dyn_masks=None,
            )
            temporal_idx += 1
            inter = rearrange(tokens_t, "(b v) (t p) c -> b t v p c", b=B, v=V, t=T)
            inter = rearrange(inter, "b t v p c -> b (t v) p c")
            intermediates.append(inter)
            attn_q.append(q)
            attn_k.append(k)

        tokens = rearrange(tokens_t, "(b v) (t p) c -> b t v p c", b=B, v=V, t=T)
        tokens = rearrange(tokens, "b t v p c -> b (t v) p c")
        tokens = tokens.reshape(B * S, P, C)
        return tokens, temporal_idx, intermediates, \
            torch.stack(attn_q, dim=0) if attn_q else torch.empty(0), \
            torch.stack(attn_k, dim=0) if attn_k else torch.empty(0)

    def _process_crossview_attention(
        self, tokens, B, S, P, C, crossview_idx, V, T, pos=None, dyn_masks=None,
    ):
        """Different views, same time — with KV compression."""
        tokens_view = tokens.view(B, S, P, C)
        tokens_c = rearrange(tokens_view, "b (t v) p c -> b t v p c", t=T, v=V)
        tokens_c = rearrange(tokens_c, "b t v p c -> (b t) (v p) c")

        pos_c = None
        if pos is not None:
            pos_view = pos.view(B, S, P, 2)
            pos_c = rearrange(pos_view, "b (t v) p d -> b t v p d", t=T, v=V)
            pos_c = rearrange(pos_c, "b t v p d -> (b t) (v p) d")

        pH, pW, ps_idx = self._get_patch_dims(P)
        intermediates, attn_q, attn_k = [], [], []
        for _ in range(self.aa_block_size):
            tokens_c, q, k = self.crossview_blocks[crossview_idx](
                tokens_c, pos=pos_c,
                pH=pH, pW=pW, patch_start_idx=ps_idx,
                keyframe_indices=None,
                is_frame_attn=False, layer_id=crossview_idx, dyn_masks=None,
            )
            crossview_idx += 1
            inter = rearrange(tokens_c, "(b t) (v p) c -> b t v p c", b=B, t=T, v=V)
            inter = rearrange(inter, "b t v p c -> b (t v) p c")
            intermediates.append(inter)
            attn_q.append(q)
            attn_k.append(k)

        tokens = rearrange(tokens_c, "(b t) (v p) c -> b t v p c", b=B, t=T, v=V)
        tokens = rearrange(tokens, "b t v p c -> b (t v) p c")
        tokens = tokens.reshape(B * S, P, C)
        return tokens, crossview_idx, intermediates, \
            torch.stack(attn_q, dim=0) if attn_q else torch.empty(0), \
            torch.stack(attn_k, dim=0) if attn_k else torch.empty(0)

    def forward(
        self, images: torch.Tensor, dyn_masks: Optional[torch.Tensor] = None,
        enable_memory_saving: bool = True, view_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], int, dict, torch.Tensor]:
        """Same forward interface as AggregatorFor4DMultiView.

        Returns (output_list, patch_start_idx, qk_dict, patch_tokens).
        """
        B, S, C_in, H, W = images.shape
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        if view_ids is None:
            view_ids = torch.zeros(B, S, dtype=torch.long, device=images.device)
        V = int(view_ids.max().item()) + 1
        T = S // V
        assert V * T == S

        # Patch embed (same as base Aggregator)
        images_norm = (images - self._resnet_mean) / self._resnet_std
        patch_tokens = self.patch_embed(images_norm.view(B * S, C_in, H, W))
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        if dyn_masks is not None:
            dyn_masks = F.max_pool2d(dyn_masks.float(), kernel_size=self.patch_size, stride=self.patch_size)
            dyn_masks = rearrange(dyn_masks, "b s h w -> b s (h w)") > 0.5

        # Special tokens
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2, device=images.device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        _, P, C = tokens.shape

        frame_idx = temporal_idx = crossview_idx = 0
        output_list = [None] * (self.aa_block_num * B)

        for i in range(self.aa_block_num):
            frame_intermediates = temporal_intermediates = crossview_intermediates = []

            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, f_int, f_q, f_k = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos, dyn_masks=dyn_masks,
                    )
                    frame_intermediates = f_int
                    del f_q, f_k
                elif attn_type == "temporal":
                    tokens, temporal_idx, t_int, t_q, t_k = self._process_temporal_attention(
                        tokens, B, S, P, C, temporal_idx, V, T, pos=pos, dyn_masks=dyn_masks,
                    )
                    temporal_intermediates = t_int
                    del t_q, t_k
                elif attn_type == "crossview":
                    tokens, crossview_idx, c_int, c_q, c_k = self._process_crossview_attention(
                        tokens, B, S, P, C, crossview_idx, V, T, pos=pos, dyn_masks=dyn_masks,
                    )
                    crossview_intermediates = c_int
                    del c_q, c_k

            for j in range(len(frame_intermediates)):
                concat_inter = torch.cat(
                    [frame_intermediates[j], temporal_intermediates[j], crossview_intermediates[j]],
                    dim=-1,
                )
                output_list[i * B + j] = concat_inter

            if enable_memory_saving and i not in self.preserve_layer_idx:
                for j in range(B):
                    output_list[i * B + j] = None
                del concat_inter, frame_intermediates, temporal_intermediates, crossview_intermediates

        if enable_memory_saving:
            del tokens

        qk_dict = {
            "global_q": torch.empty(0),
            "global_k": torch.empty(0),
            "frame_q": torch.empty(0),
            "frame_k": torch.empty(0),
        }
        return output_list, self.patch_start_idx, qk_dict, patch_tokens
