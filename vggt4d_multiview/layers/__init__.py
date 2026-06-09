# Re-export existing VGGT4D layer types (no modifications needed —
# the same weights handle decomposed temporal/crossview attention).
from vggt4d.layers.block import BlockFor4D
from vggt4d.layers.attention import AttentionFor4D
