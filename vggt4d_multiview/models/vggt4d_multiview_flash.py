"""
VGGTFor4DMultiViewFlash — Multi-view VGGT4D with FlashVGGT KV compression.

Same 3-way attention decomposition and weight mapping as VGGTFor4DMultiView,
but uses FlashBlockFor4D with kv_downfactor for temporal and crossview blocks.

Loads the FlashVGGT checkpoint (flashvggt.pt) and maps frame_blocks → frame_blocks,
global_blocks → temporal_blocks, global_blocks (copy) → crossview_blocks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead

from vggt4d_multiview.models.aggregator_multiview_flash import AggregatorFor4DMultiViewFlash

logger = logging.getLogger(__name__)


class VGGTFor4DMultiViewFlash(nn.Module):
    """Multi-view VGGT4D with FlashVGGT KV-compressed attention.

    Parameters
    ----------
    img_size : int
    patch_size : int
    embed_dim : int
    kv_downfactor : int
        KV compression factor for temporal/crossview attention (default 4).
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        kv_downfactor: int = 4,
    ):
        super().__init__()

        self.aggregator = AggregatorFor4DMultiViewFlash(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            kv_downfactor=kv_downfactor,
            temporal_kv_downfactor=kv_downfactor,
            crossview_kv_downfactor=kv_downfactor,
        )

        # Projection layers: 3C → 2C (same as VGGTFor4DMultiView)
        self.down_embed_camera = nn.Linear(3 * embed_dim, 2 * embed_dim)
        self.down_embed_depth = nn.Linear(3 * embed_dim, 2 * embed_dim)
        self.down_embed_point = nn.Linear(3 * embed_dim, 2 * embed_dim)
        self.down_embed_track = nn.Linear(3 * embed_dim, 2 * embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(
            dim_in=2 * embed_dim, output_dim=4,
            activation="inv_log", conf_activation="expp1",
        )
        self.depth_head = DPTHead(
            dim_in=2 * embed_dim, output_dim=2,
            activation="exp", conf_activation="expp1",
        )
        self.track_head = TrackHead(
            dim_in=2 * embed_dim, patch_size=patch_size,
        )

    def load_flash_checkpoint(self, checkpoint_path: Union[str, Path], device: str = "cpu") -> Dict[str, Any]:
        """Load FlashVGGT checkpoint (flashvggt.pt) and map to multi-view architecture.

        Mapping:
            checkpoint aggregator.frame_blocks  →  self.aggregator.frame_blocks
            checkpoint aggregator.global_blocks  →  self.aggregator.temporal_blocks
            checkpoint aggregator.global_blocks  →  self.aggregator.crossview_blocks (copy)
            checkpoint aggregator.camera_token  →  self.aggregator.camera_token
            checkpoint aggregator.register_token → self.aggregator.register_token
            checkpoint aggregator.patch_embed   →  self.aggregator.patch_embed
            checkpoint camera_head   →  self.camera_head
            checkpoint depth_head    →  self.depth_head

        point_head is NOT in the FlashVGGT checkpoint (FlashVGGT doesn't have it).
        track_head is NOT in the FlashVGGT checkpoint.
        """
        ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
        # Unwrap container wrappers (matching FlashVGGT's own load_ckpt)
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                state = ckpt["state_dict"]
            elif "model" in ckpt and isinstance(ckpt["model"], dict):
                state = ckpt["model"]
            else:
                state = ckpt
        else:
            state = ckpt

        renamed: Dict[str, torch.Tensor] = {}

        for key, value in state.items():
            if key.startswith("aggregator.global_blocks."):
                renamed[key.replace("aggregator.global_blocks.", "aggregator.temporal_blocks.")] = value
            elif key.startswith("aggregator."):
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
        # Only log projection layer keys (8 fresh Linear layers); point/track heads
        # are not in the FlashVGGT checkpoint and are freshly initialized — expected.
        proj_keys = [k for k in missing if "down_embed" in k]
        if proj_keys:
            logger.info("VGGT4DMultiViewFlash: %d fresh projection keys initialized: %s",
                        len(proj_keys), proj_keys)
        other_missing = len(missing) - len(proj_keys)
        if other_missing > 0:
            logger.info("VGGT4DMultiViewFlash: %d additional missing keys (point/track heads — expected): %s ...",
                        other_missing, [k for k in missing if "down_embed" not in k][:3])
        if unexpected:
            logger.debug("VGGT4DMultiViewFlash: unexpected keys: %s", unexpected)

        # Init projection layers (identity for first 2C, zero for crossview)
        self._init_projection_layers()

        return {"missing": missing, "unexpected": unexpected}

    def _init_projection_layers(self):
        """Same identity-init as VGGTFor4DMultiView."""
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

    def forward(self, images, dyn_masks=None, query_points=None, view_ids=None):
        """Same forward interface as VGGTFor4DMultiView."""
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if dyn_masks is not None and len(dyn_masks.shape) == 3:
            dyn_masks = dyn_masks.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)
        if view_ids is not None and len(view_ids.shape) == 1:
            view_ids = view_ids.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx, qk_dict, enc_feat = self.aggregator(
            images, dyn_masks=dyn_masks, view_ids=view_ids,
        )

        # Project 3C → 2C
        # Cast projection output back to input dtype to avoid float32/bfloat16 mismatch
        # with pretrained head layers that expect bfloat16.
        agg_for_heads = []
        for tokens in aggregated_tokens_list:
            if tokens is None:
                agg_for_heads.append(None)
            else:
                B, S, P, C3 = tokens.shape
                t2d = tokens.reshape(B * S * P, C3)
                t2c = self.down_embed_camera(t2d).to(dtype=t2d.dtype)
                agg_for_heads.append(t2c.reshape(B, S, P, -1))

        predictions = {}
        with torch.amp.autocast("cuda", enabled=False):
            if self.camera_head is not None:
                last_tokens = agg_for_heads[-1]
                camera_list = [last_tokens] if last_tokens is not None else [agg_for_heads[-1]]
                predictions["pose_enc"] = self.camera_head(camera_list)[-1]

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    agg_for_heads, images=images, patch_start_idx=patch_start_idx,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    agg_for_heads, images=images, patch_start_idx=patch_start_idx,
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                agg_for_heads, images=images,
                patch_start_idx=patch_start_idx, query_points=query_points,
            )
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        predictions["images"] = images
        return predictions, qk_dict, enc_feat, agg_for_heads
