# vggt4d_multiview — Multi-View VGGT4D with 3-way decomposed attention
#
# Extends VGGT4D's alternating attention (frame ↔ global) into three
# structurally distinct attention modes:
#   1. FRAME attention    — within-frame spatial  (B*S, P, C)
#   2. TEMPORAL attention — same view, different time  (B*V, T*P, C)
#   3. CROSSVIEW attention — different views, same time  (B*T, V*P, C)
#
# Reuses VGGT4D's BlockFor4D and AttentionFor4D weights (from the
# model_tracker_fixed_e20.pt checkpoint).  The crossview blocks are
# initialised from the global block weights, which were trained on
# diverse camera configurations and naturally support cross-view
# correspondence learning.
