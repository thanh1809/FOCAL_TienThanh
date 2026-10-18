# DINOv3 Encoder Update

This document describes the encoder update for the FOCAL multimodal
time-series sensing project.

## What was updated

The original feature encoder has been replaced with a DINOv3-based vision
encoder through an adapter designed for time-frequency sensor tensors.

The adapter performs these steps for every sensor modality:

1. Projects the input channels to three channels with a learnable 1x1
   convolution.
2. Resizes the tensor to `224 x 224`.
3. Extracts visual features with `vit_small_patch16_dinov3`.
4. Maps each modality feature into the FOCAL embedding space.

## Updated files

- `src/models/DINOv3.py`: DINOv3 encoder and sensor-input adapter.
- `src/train_utils/model_selection.py`: registers the `DINOv3` model name.
- `src/data/<dataset>.yaml`: DINOv3 architecture and optimizer settings.
- `web_app/inference.py`: loads DINOv3 for web inference.
- `requirements.txt`: requires `timm>=1.0.20`.
- `requirements-render.txt`: requires `timm>=1.0.20`.

## Training commands

Run these commands from the `src/` directory:

```bash
python train.py -dataset=<DATASET> -model=DINOv3 -learn_framework=FOCAL -stage=pretrain -gpu=0 -batch_size=16
python train.py -dataset=<DATASET> -model=DINOv3 -learn_framework=FOCAL -stage=finetune -gpu=0 -batch_size=16
python test.py -dataset=<DATASET> -model=DINOv3 -learn_framework=FOCAL -stage=finetune -gpu=0 -batch_size=16
```

## Pretrained weights

The configuration uses `pretrained: False` by default so local and Render
inference can run without downloading a separate foundation-model checkpoint.
To request pretrained DINOv3 weights in a compatible training environment,
set the following variable before training:

```bash
DINOV3_PRETRAINED=1
```

The official DINOv3 implementation is available at:

<https://github.com/facebookresearch/dinov3>

## Deployment

After training, place the fine-tuned checkpoint directly in the root
`weights/` directory using the filename expected by `web_app/inference.py`,
then commit the code and checkpoint before triggering a new deployment.
