"""
FlashAttentionFor4D — FlashVGGT-style KV-compressed attention with Q/K return.

Combines:
  - FlashVGGT's `kv_downfactor` spatial downsampling (for global attention)
  - VGGT4D's Q/K tuple return (for dynamic mask extraction)

Key design:
  - Frame attention: kv_downfactor=1 (no compression), returns Q/K for masks
  - Temporal/Crossview attention: kv_downfactor=N (spatial compression), returns Q/K
  - Memory: compressed K/V reduces global attention from O(S·P)² to O(S·P/N)²
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

# Reuse FlashVGGT's Attention as the base — it has the kv_downfactor logic
from flashvggt.layers.attention import Attention as FlashVGGTAttention


class FlashAttentionFor4D(FlashVGGTAttention):
    """FlashVGGT attention adapted for 4D: adds Q/K return + dynamic mask support.

    Forward() returns (output, q, k) tuples for compatibility with
    VGGT4D's dynamic mask extraction pipeline.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        """
        Returns
        -------
        output : Tensor  — attention output (B, N, C)
        q : Tensor       — query tensor for dynamic mask extraction
        k : Tensor       — key tensor for dynamic mask extraction
        """
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.kv_downfactor > 1:
            if pH is None or pW is None or patch_start_idx is None:
                raise ValueError("pH, pW, and patch_start_idx must be provided when kv_downfactor > 1")

            k = self._downscale_kv(k, pH, pW, patch_start_idx, keyframe_indices)
            v = self._downscale_kv(v, pH, pW, patch_start_idx, keyframe_indices)
        elif self.kv_downfactor < 1:
            raise ValueError("kv_downfactor must be >= 1")

        # Dynamic masking (same as VGGT4D's AttentionFor4D)
        if dyn_masks is not None and layer_id in range(0, 5):
            output = self._masked_attention(q, k, v, is_frame_attn, layer_id, dyn_masks)
        else:
            output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
            )

        output = output.transpose(1, 2).reshape(B, N, C)
        output = self.proj(output)
        output = self.proj_drop(output)

        return output, q, k

    def _downscale_kv(
        self,
        tensor: Tensor,
        pH: int,
        pW: int,
        patch_start_idx: int,
        keyframe_indices: Tensor = None,
    ) -> Tensor:
        """Spatially downsample K or V using nearest interpolation.

        Same logic as FlashVGGT's Attention.forward() downscale.
        """
        B_t, H_t, N_seq, D_t = tensor.shape
        P = patch_start_idx + pH * pW
        S = N_seq // P

        tensor = tensor.view(B_t, H_t, S, P, D_t)
        prefix = tensor[:, :, :, :patch_start_idx, :]
        spatial = tensor[:, :, :, patch_start_idx:, :]

        spatial = spatial.reshape(B_t * H_t * S, pH, pW, D_t).permute(0, 3, 1, 2)

        new_pH, new_pW = pH // self.kv_downfactor, pW // self.kv_downfactor

        spatial = F.interpolate(
            spatial.float(), size=(new_pH, new_pW),
            mode="nearest"
        ).to(spatial.dtype)

        spatial = spatial.permute(0, 2, 3, 1).reshape(B_t, H_t, S, -1, D_t)
        out = torch.cat([prefix, spatial], dim=3).reshape(B_t, H_t, -1, D_t)

        if keyframe_indices is not None:
            B_idx = torch.arange(B_t, device=tensor.device).unsqueeze(1)
            tensor_hs = tensor.transpose(1, 2)
            reference = tensor_hs[B_idx, keyframe_indices, :, patch_start_idx:, :]
            reference = reference.transpose(1, 2).reshape(B_t, H_t, -1, D_t)
            out = torch.cat([reference, out], dim=2)

        return out

    def _masked_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_frame_attn: bool,
        layer_id: int,
        dyn_masks: Tensor,
    ) -> Tensor:
        """Attention with dynamic masking applied (same as VGGT4D pattern)."""
        B, H, S_len, dk = q.shape
        _, _, _, dv = v.shape

        # dyn_masks: (B_img, S_img, HW) — align with token layout
        B_img, S_img, HW = dyn_masks.shape
        pad = torch.zeros(B_img, S_img, 5, dtype=torch.bool, device=q.device)
        dyn_masks = torch.cat([pad, dyn_masks], dim=-1)

        O = torch.empty_like(v)
        if is_frame_attn:
            dyn_masks = rearrange(dyn_masks, "b s n -> (b s) n")
        else:
            dyn_masks = rearrange(dyn_masks, "b s n -> b (s n)")

        for b in range(B):
            qb = q[b:b + 1]
            kb = k[b:b + 1].contiguous()
            vb = v[b:b + 1].contiguous()

            dm = dyn_masks[b] if not is_frame_attn else dyn_masks[b]
            if not is_frame_attn:
                # global attn: one dyn_mask per batch
                pass
            else:
                # frame attn: dyn_masks already flattened to (B*S, N)
                pass

            non_dyn_idx = (~dm).nonzero(as_tuple=True)[0]
            if non_dyn_idx.numel() == 0:
                # All tokens are dynamic — should not happen, but guard
                O[b:b + 1] = F.scaled_dot_product_attention(qb, kb, vb)
            else:
                nk = kb[..., non_dyn_idx, :].contiguous()
                nv = vb[..., non_dyn_idx, :].contiguous()
                O[b:b + 1] = F.scaled_dot_product_attention(qb, nk, nv)

        return O
