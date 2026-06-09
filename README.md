<div align="center">

<h2>VGGT4D Variants: Mining Motion Cues in Visual Geometry Transformers for 4D Scene Reconstruction</h2>

<!-- Badges -->
<p>
  <a href="https://3dagentworld.github.io/vggt4d/">
    <img src="https://img.shields.io/badge/Project-Page-blue?logo=web&logoColor=white" alt="Project Page">
  </a>
  <a href="https://arxiv.org/abs/2511.19971">
    <img src="https://img.shields.io/badge/arXiv-2511.19971-B31B1B?logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://github.com/vicksEmmanuel/vggt4d-variants">
    <img src="https://img.shields.io/badge/Code-GitHub-black?logo=github" alt="Code">
  </a>
</p>

</div>


## VGGT4D Variants

VGGT4D is a **standalone** package that bundles all VGGT variants into a single installable unit. It includes:

| Variant | Package | Description |
|---------|---------|-------------|
| **VGGT-1B** | `vggt` | Base visual geometry transformer (CVPR 2025 Best Paper) |
| **FlashVGGT** | `flashvggt` | Accelerated single-forward via compressed descriptor attention |
| **FlashVGGT-Stream** | `flashvggt_stream` | Streaming variant for long image sequences |
| **VGGT-Omega** | `vggt_omega` | Omega architecture with text alignment head |
| **VGGT4D** | `vggt4d` | 4D dynamic scene reconstruction from VGGT |
| **VGGT4D-MV** | `vggt4d_multiview` | Multi-view temporal 4D (frame/temporal/crossview attention) |

All variants are importable after a single `pip install`:

```python
from vggt.models.vggt import VGGT                         # base VGGT
from flashvggt.models.flash_vggt import FlashVGGT          # FlashVGGT
from flashvggt_stream.models.flash_vggt import FlashVGGT   # streaming
from vggt_omega.models import VGGTOmega                    # VGGT-Omega
from vggt4d.models.vggt4d import VGGTFor4D                 # VGGT4D
from vggt4d_multiview.models import VGGTFor4DMultiView     # VGGT4D-MV
```


## Quick Start

### 1. Install

```bash
# Clone
git clone <repo-url> VGGT4D
cd VGGT4D

# Create environment
python -m venv .venv && source .venv/bin/activate
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124

# Install VGGT4D (installs ALL variants)
pip install -e .

# Or install with demo dependencies
pip install -e ".[demo]"
```

### 2. Download Checkpoints

```bash
# One command downloads everything:
./setup.sh

# Or skip existing checkpoints with --rebuild:
./setup.sh --rebuild
```

This downloads:
- **VGGT-1B** → `ckpts/vggt/model.pt`
- **VGGT4D tracker** → `ckpts/flashvggt/model_tracker_fixed_e20.pt`
- **FlashVGGT** → `ckpts/flashvggt/flashvggt.pt`
- **FlashVGGT-Stream** → `ckpts/flashvggt/flashvggt_stream.pt`
- **VGGT-Omega** → `ckpts/vggt-omega/vggt_omega_1b_512.pt` (gated — requires `HF_TOKEN`)

### 3. Run Demo

```bash
# VGGT4D demo
python demo_vggt4d.py --input_dir <images_dir> --output_dir <output_dir>

# Use individual variants directly
python -c "
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
# ... your pipeline
"
```


## Usage by Variant

### VGGT-1B (base)

```python
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

model = VGGT()
state = torch.load("ckpts/vggt/model.pt", map_location="cuda", weights_only=True)
model.load_state_dict(state)
model = model.cuda().eval()

images = load_and_preprocess_images(["img1.jpg", "img2.jpg"]).cuda()
with torch.no_grad():
    predictions = model(images)
ext, int_ = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
```

### FlashVGGT

```python
import torch
from flashvggt.models.flash_vggt import FlashVGGT

model = FlashVGGT()
model.load_ckpt("ckpts/flashvggt/flashvggt.pt")
model = model.cuda().eval()

from flashvggt.utils.load_fn import load_and_preprocess_images
images = load_and_preprocess_images(["img1.jpg", "img2.jpg"], mode="pad").cuda()
with torch.no_grad():
    predictions = model(images)
```

### FlashVGGT-Stream (long sequences)

```python
import torch
from flashvggt_stream.models.flash_vggt import FlashVGGT

model = FlashVGGT()
model.load_ckpt("ckpts/flashvggt/flashvggt_stream.pt")
model = model.cuda().eval()

from flashvggt_stream.utils.load_fn import load_and_preprocess_images
images = load_and_preprocess_images(image_paths, mode="pad").cuda()
with torch.no_grad():
    predictions = model(images)
```

### VGGT-Omega

```python
import torch
from vggt_omega.models import VGGTOmega

model = VGGTOmega()
state = torch.load("ckpts/vggt-omega/vggt_omega_1b_512.pt", map_location="cuda", weights_only=True)
model.load_state_dict(state, strict=False)
model = model.cuda().eval()

from vggt_omega.utils.load_fn import load_and_preprocess_images
images = load_and_preprocess_images(image_paths, mode="balanced", image_resolution=512).unsqueeze(0).cuda()
with torch.no_grad():
    predictions = model(images)
```

### VGGT4D (4D dynamic scenes)

```python
import torch
from vggt4d.models.vggt4d import VGGTFor4D

model = VGGTFor4D()
state = torch.load("ckpts/flashvggt/model_tracker_fixed_e20.pt", map_location="cuda", weights_only=True)
model.load_state_dict(state, strict=False)
model = model.cuda().eval()

from vggt.utils.load_fn import load_and_preprocess_images
images = load_and_preprocess_images(image_paths, mode="crop").cuda()
with torch.no_grad():
    predictions, qk_dict, enc_feat, agg_tokens = model(images)
```

### VGGT4D-MV (multi-view temporal)

#### With base VGGT

```python
import torch
from vggt4d_multiview.models import VGGTFor4DMultiView

model = VGGTFor4DMultiView()
model.load_4d_checkpoint("ckpts/flashvggt/model_tracker_fixed_e20.pt", device="cuda")
model = model.cuda().eval()

view_ids = torch.tensor([[0, 0, 0, 1, 1, 1]], device="cuda")  # 3 front + 3 rear
images = load_and_preprocess_images(interleaved_frame_paths, mode="crop").cuda()
with torch.no_grad():
    predictions = model(images, view_ids=view_ids)
```

#### With FlashVGGT

```python
import torch
from vggt4d_multiview.models import VGGTFor4DMultiViewFlash

model = VGGTFor4DMultiViewFlash(kv_downfactor=4)
model.load_flash_checkpoint("ckpts/flashvggt/flashvggt.pt", device="cuda")
model = model.cuda().eval()

view_ids = torch.tensor([[0, 0, 0, 1, 1, 1]], device="cuda")  # 3 front + 3 rear
images = load_and_preprocess_images(interleaved_frame_paths, mode="crop").cuda()
with torch.no_grad():
    predictions = model(images, view_ids=view_ids)
```

#### With VGGT-Omega

```python
import torch
from vggt4d_multiview.models.vggt4d_multiview_omega import VGGTFor4DMultiViewOmega

model = VGGTFor4DMultiViewOmega()
model.load_omega_checkpoint("ckpts/vggt-omega/vggt_omega_1b_512.pt", device="cuda")
model = model.cuda().eval()

view_ids = torch.tensor([[0, 0, 0, 1, 1, 1]], device="cuda")  # 3 front + 3 rear
from vggt_omega.utils.load_fn import load_and_preprocess_images
images = load_and_preprocess_images(interleaved_frame_paths, mode="balanced", image_resolution=512).unsqueeze(0).cuda()
with torch.no_grad():
    predictions = model(images, view_ids=view_ids, enable_point_head=False)
```


## Demo Videos

| Video | Description |
|-------|-------------|
| [cam_0002_zbuffer.mp4](demo/cam_0002_zbuffer.mp4) | Z-Buffer visualization |
| [cam_0003.mp4](demo/cam_0003.mp4) | 4D reconstruction output |


## Project Structure

```
VGGT4D/
├── pyproject.toml              # Unified pip-installable package
├── __init__.py                 # Bootstrap + version
├── requirements.txt            # All dependencies merged
├── vggt/                       # VGGT core (models, layers, heads, utils, visual_util)
├── vggt4d/                     # VGGT4D model + layers + masks + flash attention
├── vggt4d_multiview/           # Multi-view temporal extension
├── training/                   # Training code (Hydra + DDP)
├── third_party/                # Vendored dependencies (auto-added to sys.path)
│   ├── __init__.py             # sys.path bootstrap
│   ├── FlashVGGT/              # FlashVGGT source
│   │   ├── flashvggt/          #   single-forward package
│   │   └── flashvggt_stream/   #   streaming package
│   └── vggt-omega/             # VGGT-Omega source
├── demo_vggt4d.py              # Inference demo
├── eval_mask.py                # DAVIS mask evaluation
└── vis_vggt4d.py               # 4D visualization
```


## TODO

- [x] Release code
- [x] Standalone installation with all VGGT variants
- [ ] Data preprocess scripts
- [ ] Evaluation scripts
- [ ] Long sequence implementation

## Acknowledgements

We thank the authors of [VGGT](https://github.com/facebookresearch/vggt), [FlashVGGT](https://github.com/wzpscott/FlashVGGT), [VGGT-Omega](https://github.com/facebookresearch/vggt-omega), [DUSt3R](https://github.com/naver/dust3r), and [Easi3R](https://github.com/Inception3D/Easi3R) for releasing their models and code.

## License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@misc{hu2025vggt4d,
      title={VGGT4D: Mining Motion Cues in Visual Geometry Transformers for 4D Scene Reconstruction}, 
      author={Yu Hu and Chong Cheng and Sicheng Yu and Xiaoyang Guo and Hao Wang},
      year={2025},
      eprint={2511.19971},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.19971}, 
}
```
