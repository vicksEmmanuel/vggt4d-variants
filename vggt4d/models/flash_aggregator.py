"""
AggregatorFor4DFlash — VGGT4D aggregator with FlashVGGT's kv_downfactor.

Inherits from the base VGGT Aggregator (not AggregatorFor4D) to avoid
the complex BlockFor4D/Q-K-tuple override chain. Directly uses
FlashBlockFor4D which returns Q/K tuples for dynamic mask extraction.

Architecture:
  - frame_blocks: FlashBlockFor4D, kv_downfactor=1 (full resolution)
  - global_blocks: FlashBlockFor4D, kv_downfactor=N (layers 9-19 compressed)

Checkpoint Compatibility:
  FlashVGGT checkpoint has identical weight keys to VGGT:
    qkv.weight, q_norm.weight, k_norm.weight, norm1.weight, mlp.*, etc.
  kv_downfactor is runtime-only (no extra params), so weights load cleanly.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from vggt.models.aggregator import Aggregator, slice_expand_and_flatten
from vggt4d.flash.block import FlashBlockFor4D


class AggregatorFor4DFlash(Aggregator):
    """VGGT4D aggregator with FlashVGGT KV compression in global attention.

    Parameters
    ----------
    kv_downfactor : int
        Spatial compression factor for global K/V (default 4).
    global_start_idx : int
        First global block index to compress (default 9).
    global_end_idx : int
        Last global block index to compress (default 19).
    """

    def __init__(
        self,
        kv_downfactor: int = 4,
        global_start_idx: int = 9,
        global_end_idx: int = 19,
        **kwargs,
    ):
        self.flash_kv_downfactor = kv_downfactor
        self.flash_global_start_idx = global_start_idx
        self.flash_global_end_idx = global_end_idx

        # Build blocks — use FlashBlockFor4D as block_fn
        kwargs["block_fn"] = FlashBlockFor4D
        super().__init__(**kwargs)

        # Replace blocks with per-block kv_downfactor
        self._rebuild_blocks()

        # Preserve layer indices for memory saving (same as VGGT4D)
        self.preserve_layer_idx = [4, 11, 17, 23]

    def _rebuild_blocks(self):
        """Replace blocks with FlashBlockFor4D configured per-block kv_downfactor.

        Copy weights from the originally-built blocks (BlockFor4D) to ensure
        weight compatibility with VGGT4D checkpoint loading.
        """
        depth = self.depth
        embed_dim = self.frame_blocks[0].attn.qkv.weight.shape[1]  # infer dim

        # Save old blocks for weight copying
        old_frame = self.frame_blocks
        old_global = self.global_blocks

        new_frame = nn.ModuleList()
        new_global = nn.ModuleList()

        for idx in range(depth):
            is_compressed = (self.flash_global_start_idx <= idx <= self.flash_global_end_idx)
            kdf = self.flash_kv_downfactor if is_compressed else 1

            new_frame.append(FlashBlockFor4D(
                dim=embed_dim, num_heads=old_frame[0].attn.num_heads,
                mlp_ratio=4.0, qkv_bias=True, proj_bias=True, ffn_bias=True,
                init_values=0.01, qk_norm=True, rope=self.rope,
                kv_downfactor=1,
            ))
            new_global.append(FlashBlockFor4D(
                dim=embed_dim, num_heads=old_global[0].attn.num_heads,
                mlp_ratio=4.0, qkv_bias=True, proj_bias=True, ffn_bias=True,
                init_values=0.01, qk_norm=True, rope=self.rope,
                kv_downfactor=kdf,
            ))

        # Copy weights old → new
        for idx in range(depth):
            self._copy_params(old_frame[idx], new_frame[idx])
            self._copy_params(old_global[idx], new_global[idx])

        self.frame_blocks = new_frame
        self.global_blocks = new_global

    def _copy_params(self, src, dst):
        for sp, dp in zip(src.parameters(), dst.parameters()):
            dp.data.copy_(sp.data)
        for sb, db in zip(src.buffers(), dst.buffers()):
            db.data.copy_(sb.data)

    # ── Forward pass (parallels AggregatorFor4D but with FlashBlockFor4D blocks) ──

    def forward(self, images: torch.Tensor,
                dyn_masks: Optional[torch.Tensor] = None,
                enable_memory_saving: bool = True) -> Tuple[List[torch.Tensor], int, dict, torch.Tensor]:
        """Same interface as AggregatorFor4D.forward().

        Returns (output_list, patch_start_idx, qk_dict, patch_tokens).
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize and patch embed
        images_norm = (images - self._resnet_mean) / self._resnet_std
        images_flat = images_norm.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images_flat)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Dynamic masks
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

        frame_idx = global_idx = 0
        output_list = [None] * (self.aa_block_num * B)
        global_q_list, frame_q_list = [], []
        global_k_list, frame_k_list = [], []

        pH, pW = H // self.patch_size, W // self.patch_size

        for i in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, f_int, f_q, f_k = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos, dyn_masks=dyn_masks,
                    )
                    frame_q_list.append(f_q.detach().cpu())
                    frame_k_list.append(f_k.detach().cpu())
                    del f_q, f_k
                elif attn_type == "global":
                    tokens, global_idx, g_int, g_q, g_k = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos, dyn_masks=dyn_masks,
                        pH=pH, pW=pW,
                    )
                    global_q_list.append(g_q.detach().cpu())
                    global_k_list.append(g_k.detach().cpu())
                    del g_q, g_k
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for j in range(len(f_int)):
                concat_inter = torch.cat([f_int[j], g_int[j]], dim=-1)
                output_list[i * B + j] = concat_inter

            if enable_memory_saving and i not in self.preserve_layer_idx:
                for j in range(B):
                    output_list[i * B + j] = None
                del concat_inter, f_int, g_int

        if enable_memory_saving:
            del tokens

        qk_dict = {
            "global_q": torch.stack(global_q_list, dim=0) if global_q_list else torch.empty(0),
            "global_k": torch.stack(global_k_list, dim=0) if global_k_list else torch.empty(0),
            "frame_q": torch.stack(frame_q_list, dim=0) if frame_q_list else torch.empty(0),
            "frame_k": torch.stack(frame_k_list, dim=0) if frame_k_list else torch.empty(0),
        }

        if enable_memory_saving:
            self.clear_inference_cache()

        return output_list, self.patch_start_idx, qk_dict, patch_tokens

    # ── Attention processors ──

    def _process_frame_attention(
        self, tokens, B, S, P, C, frame_idx, pos=None, dyn_masks=None,
    ):
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)
        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates, attn_q, attn_k = [], [], []
        for _ in range(self.aa_block_size):
            if self.training:
                tokens, q, k = checkpoint(
                    self.frame_blocks[frame_idx], tokens, pos,
                    use_reentrant=self.use_reentrant,
                )
            else:
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

    def _process_global_attention(
        self, tokens, B, S, P, C, global_idx, pos=None, dyn_masks=None,
        pH: int = None, pW: int = None,
    ):
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)
        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates, attn_q, attn_k = [], [], []
        ps_idx = self.patch_start_idx

        for _ in range(self.aa_block_size):
            if self.training:
                tokens, q, k = checkpoint(
                    self.global_blocks[global_idx], tokens, pos, pH=pH, pW=pW,
                    patch_start_idx=ps_idx,
                    use_reentrant=self.use_reentrant,
                )
            else:
                tokens, q, k = self.global_blocks[global_idx](
                    tokens, pos=pos,
                    pH=pH, pW=pW, patch_start_idx=ps_idx,
                    keyframe_indices=None,
                    is_frame_attn=False, layer_id=global_idx, dyn_masks=dyn_masks,
                )
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
            attn_q.append(q)
            attn_k.append(k)

        return tokens, global_idx, intermediates, \
            torch.stack(attn_q, dim=0) if attn_q else torch.empty(0), \
            torch.stack(attn_k, dim=0) if attn_k else torch.empty(0)

    def clear_inference_cache(self):
        if hasattr(self, "rope") and self.rope is not None:
            if hasattr(self.rope, "frequency_cache"):
                self.rope.frequency_cache.clear()
        if hasattr(self, "position_getter") and self.position_getter is not None:
            if hasattr(self.position_getter, "position_cache"):
                self.position_getter.position_cache.clear()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
