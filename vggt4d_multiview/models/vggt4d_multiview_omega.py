"""
VGGTFor4DMultiViewOmega — Multi-view VGGT4D with VGGT-Omega architecture.

Same 3-way attention decomposition and weight mapping as VGGTFor4DMultiView,
but uses VGGT-Omega's SelfAttentionBlock as the building block with register tokens,
Omega-style RoPE, and Omega-style CameraHead + DenseHead.

Loads the VGGT-Omega checkpoint and maps:
    aggregator.frame_blocks       ← checkpoint aggregator.frame_blocks
    aggregator.temporal_blocks    ← checkpoint aggregator.inter_frame_blocks
    aggregator.crossview_blocks   ← copy from temporal_blocks
    camera_head                   ← checkpoint camera_head
    dense_head                    ← checkpoint dense_head
    point_head / track_head       ← freshly initialised (not in Omega checkpoint)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt_omega.models.heads import CameraHead, DenseHead

from vggt4d_multiview.models.aggregator_multiview_omega import AggregatorFor4DMultiViewOmega

logger = logging.getLogger(__name__)


class VGGTFor4DMultiViewOmega(nn.Module):
    """Multi-view VGGT4D with VGGT-Omega architecture components.

    Parameters
    ----------
    img_size : int
        Input image size in pixels (default 518).
    patch_size : int
        Patch size (default 16 — matches VGGT-Omega).
    embed_dim : int
        Token embedding dimension (default 1024).
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 16,
        embed_dim: int = 1024,
    ):
        super().__init__()

        self.aggregator = AggregatorFor4DMultiViewOmega(
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        # Projection layers: 3C → 2C (frame + temporal + crossview → 2C for heads)
        self.down_embed_camera = nn.Linear(3 * embed_dim, 2 * embed_dim)
        self.down_embed_depth = nn.Linear(3 * embed_dim, 2 * embed_dim)
        self.down_embed_point = nn.Linear(3 * embed_dim, 2 * embed_dim)
        self.down_embed_track = nn.Linear(3 * embed_dim, 2 * embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.dense_head = DenseHead(dim_in=2 * embed_dim, patch_size=patch_size)
        self.point_head = DPTHead(
            dim_in=2 * embed_dim, output_dim=4, patch_size=patch_size,
            activation="inv_log", conf_activation="expp1",
        )
        self.track_head = TrackHead(
            dim_in=2 * embed_dim, patch_size=patch_size,
        )

    def load_omega_checkpoint(
        self, checkpoint_path: Union[str, Path], device: str = "cpu",
    ) -> Dict[str, Any]:
        """Load VGGT-Omega checkpoint and map to multi-view architecture.

        Mapping::

            checkpoint aggregator.frame_blocks       →  self.aggregator.frame_blocks
            checkpoint aggregator.inter_frame_blocks →  self.aggregator.temporal_blocks
            checkpoint aggregator.inter_frame_blocks →  self.aggregator.crossview_blocks (copy)
            checkpoint aggregator.patch_embed        →  self.aggregator.patch_embed
            checkpoint aggregator.rope_embed         →  self.aggregator.rope_embed
            checkpoint aggregator.camera_token       →  self.aggregator.camera_token
            checkpoint aggregator.register_token     →  self.aggregator.register_token
            checkpoint camera_head                   →  self.camera_head
            checkpoint dense_head                    →  self.dense_head
        """
        ckpt_path = Path(checkpoint_path)
        if ckpt_path.suffix == ".safetensors":
            import safetensors.torch
            state = safetensors.torch.load_file(str(ckpt_path), device=device)
        else:
            state = torch.load(str(ckpt_path), map_location=device, weights_only=True)

        # Unwrap container wrappers
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state and isinstance(state["model"], dict):
                state = state["model"]

        renamed: Dict[str, torch.Tensor] = {}

        for key, value in state.items():
            # Map aggregator keys
            if key.startswith("aggregator.inter_frame_blocks."):
                renamed[key.replace("aggregator.inter_frame_blocks.", "aggregator.temporal_blocks.")] = value
            elif key.startswith("aggregator."):
                renamed[key] = value
            elif key.startswith("camera_head."):
                renamed[key] = value
            elif key.startswith("dense_head."):
                renamed[key] = value
            else:
                renamed[key] = value

        # Copy temporal → crossview blocks
        for i in range(24):
            prefix_t = f"aggregator.temporal_blocks.{i}."
            for key in list(renamed.keys()):
                if key.startswith(prefix_t):
                    cv_key = key.replace(
                        f"aggregator.temporal_blocks.{i}.",
                        f"aggregator.crossview_blocks.{i}.",
                    )
                    if cv_key not in renamed:
                        renamed[cv_key] = renamed[key].clone()

        missing, unexpected = self.load_state_dict(renamed, strict=False)

        proj_keys = [k for k in missing if "down_embed" in k]
        if proj_keys:
            logger.info(
                "VGGT4DMultiViewOmega: %d fresh projection keys initialized: %s",
                len(proj_keys), proj_keys,
            )
        other_missing = len(missing) - len(proj_keys)
        if other_missing > 0:
            logger.info(
                "VGGT4DMultiViewOmega: %d additional missing keys: %s",
                other_missing, [k for k in missing if "down_embed" not in k][:3],
            )
        if unexpected:
            logger.debug("VGGT4DMultiViewOmega: unexpected keys: %s", unexpected)

        self._init_projection_layers()

        return {"missing": missing, "unexpected": unexpected}

    def _init_projection_layers(self):
        """Init each down_embed layer as identity for [0:2C] and zero for [2C:3C]."""
        C3 = 3 * 1024
        C2 = 2 * 1024
        for name in ("down_embed_camera", "down_embed_depth",
                     "down_embed_point", "down_embed_track"):
            layer: nn.Linear = getattr(self, name)
            with torch.no_grad():
                layer.weight.zero_()
                for i in range(C2):
                    layer.weight[i, i] = 1.0
                layer.weight[:, C2:] = 0.0
                layer.bias.zero_()

    def forward(
        self,
        images: torch.Tensor,
        dyn_masks: Optional[torch.Tensor] = None,
        query_points: Optional[torch.Tensor] = None,
        view_ids: Optional[torch.Tensor] = None,
        enable_point_head: bool = True,
    ):
        """
        Parameters
        ----------
        images : (B, S, 3, H, W) or (S, 3, H, W)  [0, 1]
        dyn_masks : (B, S, H, W) or None
        query_points : (B, N, 2) or None
            Pixel coordinates for point tracking.
        view_ids : (B, S) or None
        enable_point_head : bool
            If False, skip point_head and track_head (saves ~40% decoder
            compute when only depth/pose are needed).

        Returns
        -------
        predictions : dict
            pose_enc, depth, depth_conf,
            world_points, world_points_conf (if enable_point_head),
            images, track (if query_points provided and enable_point_head),
            vis (if query_points provided and enable_point_head),
            conf (if query_points provided and enable_point_head)
        qk_dict : dict (empty)
        enc_feat : Tensor
            Patch tokens (encoder features).
        agg_tokens_list : list[Tensor | None]
            Aggregated token list (post-projection).
        """
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if dyn_masks is not None and len(dyn_masks.shape) == 3:
            dyn_masks = dyn_masks.unsqueeze(0)
        if view_ids is not None and len(view_ids.shape) == 1:
            view_ids = view_ids.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx, qk_dict, enc_feat = self.aggregator(
            images, dyn_masks=dyn_masks, view_ids=view_ids,
        )

        # Project 3C → 2C
        agg_for_heads = []
        for tokens in aggregated_tokens_list:
            if tokens is None:
                agg_for_heads.append(None)
            else:
                B, S, P, C3 = tokens.shape
                t2d = tokens.reshape(B * S * P, C3)
                t2c = self.down_embed_camera(t2d).to(dtype=t2d.dtype)
                agg_for_heads.append(t2c.reshape(B, S, P, -1))

        predictions: Dict[str, Any] = {}

        with torch.amp.autocast("cuda", enabled=False):
            if self.camera_head is not None:
                pose_enc = self.camera_head(
                    agg_for_heads, patch_token_start=patch_start_idx,
                )
                predictions["pose_enc"] = pose_enc

            if self.dense_head is not None:
                depth, depth_conf = self.dense_head(
                    agg_for_heads, images=images, patch_token_start=patch_start_idx,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if enable_point_head and self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    agg_for_heads, images=images, patch_start_idx=patch_start_idx,
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if enable_point_head and self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                agg_for_heads, images=images,
                patch_start_idx=patch_start_idx, query_points=query_points,
            )
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        predictions["images"] = images
        return predictions, qk_dict, enc_feat, agg_for_heads
