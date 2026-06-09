"""
VGGT4D third-party dependency bootstrap.

Adds vendored packages to sys.path so that ``import flashvggt``,
``import vggt_omega``, etc. resolve correctly when VGGT4D is installed
as a single package.
"""
import os as _os
import sys as _sys


def _bootstrap():
    """Add vendored third-party directories to sys.path (idempotent)."""
    # VGGT4D root (where vggt4d/, vggt/, third_party/ live)
    vggt4d_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

    # vggt — root-level vendored copy (models/, layers/, heads/, utils/, visual_util.py)
    vggt_dir = _os.path.join(vggt4d_root, "vggt")
    if vggt_dir not in _sys.path:
        _sys.path.insert(0, vggt_dir)

    # flashvggt / flashvggt_stream  (from FlashVGGT)
    flashvggt_dir = _os.path.join(vggt4d_root, "third_party", "FlashVGGT")
    if flashvggt_dir not in _sys.path:
        _sys.path.insert(0, flashvggt_dir)

    # vggt_omega  (from vggt-omega)
    omega_dir = _os.path.join(vggt4d_root, "third_party", "vggt-omega")
    if omega_dir not in _sys.path:
        _sys.path.insert(0, omega_dir)
