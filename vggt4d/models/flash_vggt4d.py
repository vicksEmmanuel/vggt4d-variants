"""
VGGTFor4DFlash — VGGT4D model using FlashVGGT's KV-compressed attention.

Architecture:
  - AggregatorFor4DFlash (FlashBlockFor4D blocks with kv_downfactor)
  - CameraHead, DPTHead (depth), TrackHead from VGGT (same dims)
  - Loads FlashVGGT checkpoint (flashvggt.pt) with strict=False

The FlashVGGT checkpoint has the same weight shapes as VGGT (embed_dim=1024,
num_heads=16, patch_embed=dinov2_vitl14_reg, 24 blocks, camera+point+depth heads),
so the state dict maps seamlessly — only the attention forward pass differs
(spatial KV compression in global blocks + Q/K return).
"""

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from vggt4d.models.flash_aggregator import AggregatorFor4DFlash
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead


class VGGTFor4DFlash(nn.Module):
    """VGGT4D with FlashVGGT-style KV-compressed global attention.

    Parameters
    ----------
    img_size : int
        Input image size (default 518).
    patch_size : int
        Patch size (default 14).
    embed_dim : int
        Token embedding dimension (default 1024).
    kv_downfactor : int
        Spatial compression factor for global K/V (default 4).
    global_start_idx : int
        First global block to compress (default 9).
    global_end_idx : int
        Last global block to compress (default 19).
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        kv_downfactor: int = 4,
        global_start_idx: int = 9,
        global_end_idx: int = 19,
    ):
        super().__init__()

        self.aggregator = AggregatorFor4DFlash(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            kv_downfactor=kv_downfactor,
            global_start_idx=global_start_idx,
            global_end_idx=global_end_idx,
        )

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

    def load_checkpoint(
        self,
        ckpt_path: Union[str, Path],
        device: str = "cpu",
        strict: bool = False,
    ) -> dict:
        """Load checkpoint (FlashVGGT or VGGT) into FlashVGGT4D model.

        The FlashVGGT checkpoint has identical weight shapes to VGGT:
          aggregator.frame_blocks.*.attn.qkv.weight  →  self.aggregator.frame_blocks.*.attn.qkv.weight
          aggregator.global_blocks.*.attn.qkv.weight  →  self.aggregator.global_blocks.*.attn.qkv.weight
          camera_head.*, depth_head.*, point_head.*, track_head.*
          patch_embed.*

        Returns dict of missing/unexpected keys for diagnostics.
        """
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        # Handle wrapped checkpoints (containing "state_dict" or "model" keys)
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif "model" in ckpt and isinstance(ckpt["model"], dict):
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        return {"missing": missing, "unexpected": unexpected}

    def forward(
        self,
        images: torch.Tensor,
        dyn_masks: Optional[torch.Tensor] = None,
        query_points: Optional[torch.Tensor] = None,
    ):
        """Forward pass — same interface as VGGTFor4D.

        Returns
        -------
        predictions : dict
        qk_dict : dict
        enc_feat : Tensor
        aggregated_tokens_list : list[Tensor]
        """
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if dyn_masks is not None and len(dyn_masks.shape) == 3:
            dyn_masks = dyn_masks.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx, qk_dict, enc_feat = self.aggregator(
            images, dyn_masks,
        )

        predictions = {}

        with torch.amp.autocast("cuda", enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images,
                patch_start_idx=patch_start_idx, query_points=query_points,
            )
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        predictions["images"] = images
        return predictions, qk_dict, enc_feat, aggregated_tokens_list
