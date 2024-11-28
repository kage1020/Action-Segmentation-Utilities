# Video Feature Extractor for ASU

This module provides the feature extraction tools for action segmentation.

## Extraction models

The following models are contained in this module:

- `i3d.py`: [I3D](https://arxiv.org/abs/1705.07750) implemented by [piergiai](https://github.com/piergiaj/pytorch-i3d) and trained on [Kinetics](https://arxiv.org/abs/1705.06950) or [ImageNet](https://arxiv.org/abs/1409.0575).
  - Pretrained weights are available [here](https://github.com/piergiaj/pytorch-i3d/tree/master/models). It contains RGB and Flow for each dataset.
- `s3d_howto100m.py`: [S3D](https://arxiv.org/pdf/1712.04851) implemented by [antoine77340](https://github.com/antoine77340/S3D_HowTo100M) and trained on [HowTo100M](https://arxiv.org/abs/1906.03327). Training code is inspired by [antoine77340](https://github.com/antoine77340/MIL-NCE_HowTo100M).
  - Pretrained weights are available [here](https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy).
- `s3d_kinetics.py`: [S3D](https://arxiv.org/pdf/1712.04851) implemented by [kylemin](https://github.com/kylemin/S3D) and trained on [Kinetics](https://arxiv.org/abs/1705.06950).
  - Pretrained weights are available [here](https://drive.google.com/file/d/1IysdsJI7UuWazoxFyYvrbO97Lz_EP5H-/view?usp=sharing).

You can also use pytorch-implemented models from torchvision like below:

```python
from torchvision.models.video.s3d import S3D_Weights, s3d

model = s3d(weights=S3D_Weights.KINETICS400_V1)
```
