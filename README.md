[README.md](https://github.com/user-attachments/files/27796914/README.md)
# BASG-Net: Model Architecture Only

This repository releases only the model architecture for cervical cell segmentation.
It does **not** include datasets, local paths, training scripts, evaluation scripts,
logs, checkpoints, prediction masks, Grad-CAM visualizations, or experimental results.

The released model keeps the architecture aligned with the original implementation:

- U-Net-style encoder-decoder backbone
- 5 x 5 convolutional blocks
- auxiliary branch attached to the H/4 encoder feature map
- complementary spatial keep gate for multi-level skip connections
- segmentation head with 5 x 5 convolution plus 1 x 1 projection

The auxiliary branch should be interpreted as a coarse background/non-cell probability
estimation branch used for skip-connection modulation, not as an independent detector
for specific artifact categories.

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
