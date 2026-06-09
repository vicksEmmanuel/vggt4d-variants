"""
VGGT4D — Mining Motion Cues in Visual Geometry Transformers for 4D Scene Reconstruction.

This package bundles multiple VGGT variants as a single installable unit:

- **vggt**          — base VGGT-1B (Facebook Research)
- **flashvggt**     — FlashVGGT accelerated single-forward
- **flashvggt_stream** — FlashVGGT streaming for long sequences
- **vggt_omega**    — VGGT-Omega with text alignment
- **vggt4d**        — 4D dynamic scene reconstruction
- **vggt4d_multiview** — multi-view temporal 4D reconstruction
"""

__version__ = "1.0.0"

# Bootstrap vendored third-party packages (flashvggt, vggt_omega, vggt)
# so they are importable as top-level modules.
from vggt4d.third_party import _bootstrap  # noqa: F401
_bootstrap()
