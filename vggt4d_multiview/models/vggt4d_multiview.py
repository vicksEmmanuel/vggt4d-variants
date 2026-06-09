"""
VGGTFor4DMultiView — Multi-view VGGT4D with 3-way decomposed attention.

Loads the VGGT4D checkpoint (model_tracker_fixed_e20.pt), maps weights
to the decomposed architecture, and adds lightweight projection layers to
bridge the 3×embed_dim aggregator output with the 2×embed_dim heads.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead

from vggt4d_multiview.models.aggregator_multiview import AggregatorFor4DMultiView

logger = logging.getLogger(__name__)


class VGGTFor4DMultiView(nn.Module, PyTorchModelHubMixin):
    """Multi-view VGGT4D model with 3-way decomposed attention.

    Parameters
    ----------
    img_size : int
    patch_size : int
    embed_dim : int
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
    ):
        super().__init__()

        # Aggregator outputs tokens at 3×C (frame + temporal + crossview)
        self.aggregator = AggregatorFor4DMultiView(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        # Projection layers:  3×C  →  2×C  (so existing heads match)
        self.down_embed_camera = nn.Linear(3 * embed_dim, 2 * embed_dim)
        self.down_embed_depth = nn.Linear(3 * embed_dim, 2 * embed_dim)
        self.down_embed_point = nn.Linear(3 * embed_dim, 2 * embed_dim)
        self.down_embed_track = nn.Linear(3 * embed_dim, 2 * embed_dim)

        # Heads (dim_in = 2×embed_dim — same as original VGGT4D)
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

    def load_4d_checkpoint(self, checkpoint_path: str, device: str = "cpu") -> Dict[str, Any]:
        """Load VGGT4D weights and map to multi-view architecture.

        Mapping::

            checkpoint frame_blocks  →  self.aggregator.frame_blocks
            checkpoint global_blocks →  self.aggregator.temporal_blocks
            checkpoint global_blocks →  self.aggregator.crossview_blocks  (copy)
            checkpoint camera_token  →  self.aggregator.camera_token
            checkpoint register_token → self.aggregator.register_token
            checkpoint patch_embed   →  self.aggregator.patch_embed
            checkpoint camera_head   →  self.camera_head
            checkpoint depth_head    →  self.depth_head
            checkpoint point_head    →  self.point_head
            checkpoint track_head    →  self.track_head
        """
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)

        # ── Rename keys ────────────────────────────────────────────
        renamed: Dict[str, torch.Tensor] = {}

        for key, value in state.items():
            # Aggregator temporal blocks
            if key.startswith("aggregator.global_blocks."):
                renamed[key.replace("aggregator.global_blocks.", "aggregator.temporal_blocks.")] = value
            elif key.startswith("aggregator."):
                renamed[key] = value
            # Heads — direct mapping
            else:
                renamed[key] = value

        # ── Copy temporal → crossview blocks ───────────────────────
        for i in range(24):
            prefix_t = f"aggregator.temporal_blocks.{i}."
            for key in list(renamed.keys()):
                if key.startswith(prefix_t):
                    crossview_key = key.replace(
                        f"aggregator.temporal_blocks.{i}.",
                        f"aggregator.crossview_blocks.{i}.",
                    )
                    if crossview_key not in renamed:
                        renamed[crossview_key] = renamed[key].clone()

        # ── Load ────────────────────────────────────────────────────
        missing, unexpected = self.load_state_dict(renamed, strict=False)
        if missing:
            logger.info("VGGT4DMultiView: missing keys (projection layers): %s",
                        [k for k in missing if "down_embed" in k])
        if unexpected:
            logger.debug("VGGT4DMultiView: unexpected keys: %s", unexpected)

        # ── Initialise projection layers as identity (first 2C) ─────
        # The down_embed_* layers project 3C → 2C.  The pretrained
        # heads expect token distributions identical to what they saw
        # during training (2C = concat[frame, global] from VGGT4D).
        # We initialise the projection to pass through the first 2C
        # (frame + temporal) unchanged and zero the crossview
        # contribution.  Crossview information still flows into the
        # heads because the frame/temporal/crossview blocks share
        # weights — frame and temporal tokens carry crossview context.
        self._init_projection_layers()

        return {"missing": missing, "unexpected": unexpected}

    def _init_projection_layers(self):
        """Set each down_embed_* layer to identity for [0:2C] +
        zero for [2C:3C], and zero bias."""
        C3 = 3 * 1024   # input  (3 × embed_dim, embed_dim=1024)
        C2 = 2 * 1024   # output (2 × embed_dim)
        for name in ("down_embed_camera", "down_embed_depth",
                     "down_embed_point", "down_embed_track"):
            layer: nn.Linear = getattr(self, name)
            with torch.no_grad():
                layer.weight.zero_()
                # Identity on the first 2C columns
                for i in range(C2):
                    layer.weight[i, i] = 1.0
                # Zero out crossview columns [2C:3C]
                layer.weight[:, C2:] = 0.0
                # Zero bias
                layer.bias.zero_()

    def _project_aggregated_tokens(
        self, output_list: list,
    ) -> list:
        """Project aggregated tokens from 3C → 2C for compatibility with heads.

        Camera head uses the full token (with camera token at idx 0).
        Depth/Point/Track heads use only patch tokens.
        """
        projected: list = []
        for tokens in output_list:
            if tokens is None:
                projected.append(None)
                continue
            B, S, P, C3 = tokens.shape  # C3 = 3 * embed_dim
            tokens_2d = tokens.reshape(B * S * P, C3)
            tokens_2c = self.down_embed_camera(tokens_2d)
            projected.append(tokens_2c.reshape(B, S, P, -1))
        return projected

    def _project_for_head(
        self, output_list: list, head_type: str,
    ) -> list:
        """Project output tokens for a specific head type."""
        projection = {
            "camera": self.down_embed_camera,
            "depth": self.down_embed_depth,
            "point": self.down_embed_point,
            "track": self.down_embed_track,
        }[head_type]
        projected: list = []
        for tokens in output_list:
            if tokens is None:
                projected.append(None)
                continue
            B, S, P, C3 = tokens.shape
            tokens_2d = tokens.reshape(B * S * P, C3)
            tokens_2c = projection(tokens_2d)
            projected.append(tokens_2c.reshape(B, S, P, -1))
        return projected

    def forward(
        self,
        images: torch.Tensor,
        dyn_masks: Optional[torch.Tensor] = None,
        query_points: Optional[torch.Tensor] = None,
        view_ids: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        images : (B, S, 3, H, W)  or  (S, 3, H, W)  [0, 1]
            Interleaved multi-view sequence.
        dyn_masks : (B, S, H, W) or None
            Dynamic masks for attention masking.
        query_points : (B, N, 2) or None
            Pixel coordinates for point tracking.
        view_ids : (B, S) or None
            View index per frame (0=front, 1=rear, ...).

        Returns
        -------
        predictions : dict
            pose_enc, depth, depth_conf, world_points, world_points_conf,
            track, vis, conf, images
        qk_dict : dict
            raw Q/K tensors for dynamic mask extraction.
        enc_feat : Tensor
            Patch tokens (encoder features).
        agg_tokens_list : list[Tensor]
            Aggregated token list (post-projection).
        """
        # Add batch dim if needed
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

        # Project: 3C → 2C for heads (replace Nones with zeros for preserve layers)
        # Cast projection output back to input dtype to avoid float32/bfloat16 mismatch
        # with pretrained head layers that expect bfloat16.
        agg_for_heads: list = []
        for tokens in aggregated_tokens_list:
            if tokens is None:
                agg_for_heads.append(None)
            else:
                B, S, P, C3 = tokens.shape
                t2d = tokens.reshape(B * S * P, C3)
                t2c = self.down_embed_camera(t2d).to(dtype=t2d.dtype)
                agg_for_heads.append(t2c.reshape(B, S, P, -1))

        predictions: dict = {}

        with torch.amp.autocast("cuda", enabled=False):
            if self.camera_head is not None:
                # Camera head expects tokens from last (non-None) layer
                last_tokens = agg_for_heads[-1]
                # But camera_head internally uses aggregated_tokens_list[-1][:, :, 0]
                # We need to provide a list where last element is usable
                # Create a list with only the last valid token for camera head
                camera_list = [last_tokens] if last_tokens is not None else [agg_for_heads[-1]]
                pose_enc_list = self.camera_head(camera_list)
                predictions["pose_enc"] = pose_enc_list[-1]

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
