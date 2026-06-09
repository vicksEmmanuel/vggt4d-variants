#!/bin/bash
# VGGT4D Setup Script
# Downloads all checkpoints and installs the package
# Usage: ./setup.sh [--rebuild]

REBUILD=false
for arg in "$@"; do
    [ "$arg" = "--rebuild" ] && REBUILD=true
done

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_EXE=$(which python3)
[ -z "$PYTHON_EXE" ] && PYTHON_EXE="python"

CKPT_DIR_VGGT="$SCRIPT_DIR/ckpts/vggt"
CKPT_DIR_FLASH="$SCRIPT_DIR/ckpts/flashvggt"
CKPT_DIR_OMEGA="$SCRIPT_DIR/ckpts/vggt-omega"

echo "📦 Installing VGGT4D package..."
cd "$SCRIPT_DIR"
$PYTHON_EXE -m pip install -e .

mkdir -p "$CKPT_DIR_VGGT" "$CKPT_DIR_FLASH" "$CKPT_DIR_OMEGA"

# VGGT-1B
if [ ! -f "$CKPT_DIR_VGGT/model.pt" ] || [ "$REBUILD" = true ]; then
    echo "⬇️  Downloading VGGT-1B checkpoint..."
    wget -c "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt" -O "$CKPT_DIR_VGGT/model.pt"
else
    echo "✅ VGGT-1B already exists."
fi

# VGGT4D tracker
if [ ! -f "$CKPT_DIR_FLASH/model_tracker_fixed_e20.pt" ] || [ "$REBUILD" = true ]; then
    echo "⬇️  Downloading VGGT4D tracker checkpoint..."
    wget -c "https://huggingface.co/facebook/VGGT_tracker_fixed/resolve/main/model_tracker_fixed_e20.pt?download=true" -O "$CKPT_DIR_FLASH/model_tracker_fixed_e20.pt"
else
    echo "✅ VGGT4D tracker already exists."
fi

# FlashVGGT
if [ ! -f "$CKPT_DIR_FLASH/flashvggt.pt" ] || [ "$REBUILD" = true ]; then
    echo "⬇️  Downloading FlashVGGT checkpoint..."
    $PYTHON_EXE -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='ZipW/FlashVGGT', filename='flashvggt.pt', local_dir='$CKPT_DIR_FLASH')
"
else
    echo "✅ FlashVGGT already exists."
fi

# FlashVGGT-Stream
if [ ! -f "$CKPT_DIR_FLASH/flashvggt_stream.pt" ] || [ "$REBUILD" = true ]; then
    echo "⬇️  Downloading FlashVGGT-Stream checkpoint..."
    $PYTHON_EXE -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='ZipW/FlashVGGT', filename='flashvggt_stream.pt', local_dir='$CKPT_DIR_FLASH')
"
else
    echo "✅ FlashVGGT-Stream already exists."
fi

# VGGT-Omega (gated)
if [ ! -f "$CKPT_DIR_OMEGA/vggt_omega_1b_512.pt" ] || [ "$REBUILD" = true ]; then
    echo "⬇️  Downloading VGGT-Omega checkpoint (gated)..."
    $PYTHON_EXE -c "
from huggingface_hub import hf_hub_download
import os
token = os.environ.get('HF_TOKEN', None)
hf_hub_download(repo_id='facebook/VGGT-Omega', filename='vggt_omega_1b_512.pt', local_dir='$CKPT_DIR_OMEGA', token=token)
" || {
        echo "⚠️  VGGT-Omega download failed. This repo is gated."
        echo "   1. Request access: https://huggingface.co/facebook/VGGT-Omega"
        echo "   2. Set HF_TOKEN: export HF_TOKEN=hf_your_token_here"
        echo "   3. Run: ./setup.sh"
    }
else
    echo "✅ VGGT-Omega already exists."
fi

echo ""
echo "✅ Setup complete!"
echo "   VGGT checkpoints:      $CKPT_DIR_VGGT/"
echo "   FlashVGGT checkpoints: $CKPT_DIR_FLASH/"
echo "   VGGT-Omega checkpoint: $CKPT_DIR_OMEGA/"
