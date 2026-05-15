# BASG-Net: Background-Aware Spatial Gating Network for Cervical Cell Segmentation

This repository provides the **model architecture implementation** of BASG-Net, a background-aware spatial gating network designed for cervical cell segmentation in Pap smear images.

> **Note:** This repository releases the model architecture only. It does **not** include datasets, local paths, training scripts, evaluation scripts, logs, checkpoints, prediction masks, Grad-CAM visualizations, or experimental results.

---

## Introduction

<!-- INTRO_START -->
Accurate cervical cell segmentation is an important step for automated cytology image analysis. However, Pap smear images often contain complex non-cellular background interference, including red blood cells, staining residues, mucus, cell debris, and uneven background regions. These factors may introduce false-positive responses and affect the continuity of predicted cell boundaries.

BASG-Net is designed to improve feature propagation in an encoder-decoder segmentation framework by introducing a background-aware auxiliary branch. Instead of acting as an independent detector for specific artifact categories, this branch estimates a coarse background/non-cell probability map from intermediate encoder features. The complementary spatial response is then used as a keep gate to modulate multi-level skip connections, allowing foreground-related features to be retained while reducing the influence of background responses during feature fusion.
<!-- INTRO_END -->

---

## Architecture Overview

<!-- FIGURE_START -->
![BASG-Net architecture overview](assets/basgnet_overview.png)
<!-- FIGURE_END -->

**Figure 1.** Overview of BASG-Net. The model follows a U-Net-style encoder-decoder structure. An auxiliary background-aware branch is attached to the H/4 encoder feature map and generates a background probability map. Its complementary response is used as a spatial keep gate to modulate skip-connection features before decoder fusion.

---

## Model Design

The released model keeps the architecture aligned with the original implementation:

- U-Net-style encoder-decoder backbone
- 5 × 5 convolutional blocks
- auxiliary branch attached to the H/4 encoder feature map
- complementary spatial keep gate for multi-level skip connections
- segmentation head with 5 × 5 convolution plus 1 × 1 projection

The auxiliary branch should be interpreted as a coarse background/non-cell probability estimation branch used for skip-connection modulation, **not** as an independent detector for red blood cells, staining residues, mucus, or other specific artifact categories.

---

## Repository Scope

This repository includes:

```text
basg_net/
├── __init__.py
└── model.py
```

This repository does **not** include:

```text
datasets
training scripts
evaluation scripts
trained weights
local file paths
experimental logs
prediction masks
visualization results
Grad-CAM results
```

---

## Usage

```python
import torch
from basg_net import BASGNet

model = BASGNet(in_channels=3, num_classes=1, base_c=64)

x = torch.randn(2, 3, 224, 224)
seg_logits, artifact_logits = model(x)

print(seg_logits.shape)      # [2, 1, 224, 224]
print(artifact_logits.shape) # [2, 1, 56, 56]
```

---

## Output

The model returns two tensors:

```python
seg_logits, artifact_logits = model(x)
```

- `seg_logits`: segmentation logits with the same spatial size as the input image
- `artifact_logits`: auxiliary background/non-cell logits generated from the H/4 encoder feature map

The final segmentation mask can be obtained by applying a sigmoid function and a threshold to `seg_logits`.

---

## Requirements

```text
torch
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Citation

If you use this architecture or find this repository helpful, please cite the corresponding paper once available.

```bibtex
@article{basgnet,
  title   = {BASG-Net: A Background-Aware Spatial Gating Network for Cervical Cell Segmentation in Pap Smear Images},
  author  = {Author Name},
  journal = {Journal Name},
  year    = {Year}
}
```

---

## License

This repository is released for academic and research use only.
