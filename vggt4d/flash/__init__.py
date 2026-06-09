# FlashVGGT-adapted VGGT4D components
# Bridges FlashVGGT's kv_downfactor compression with VGGT4D's 4D architecture.
# Only re-export the primitives needed by flash aggregators.
# The flash aggregator itself lives at vggt4d.models.flash_aggregator
# and vggt4d_multiview.models.aggregator_multiview_flash respectively.
from vggt4d.flash.attention import FlashAttentionFor4D
from vggt4d.flash.block import FlashBlockFor4D
