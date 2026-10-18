"""DINOv3 backbone adapter for FOCAL time-frequency sensor inputs.

DINOv3 is a vision backbone.  PAMAP2 tensors are therefore projected from
their sensor channels to three channels and resized to an image-like tensor
before entering the shared DINOv3 encoder.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DINOv3(nn.Module):
    """Use a DINOv3 ViT as the modality encoder inside FOCAL."""

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.config = args.dataset_config["DINOv3"]
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        self.device = args.device

        model_name = self.config.get("model_name", "vit_small_patch16_dinov3")
        input_size = int(self.config.get("input_size", 224))
        self.input_size = input_size

        # Keep deployment deterministic/offline by default.  Set
        # DINOV3_PRETRAINED=1 when the official pretrained weights are
        # available in the training environment.
        use_pretrained = bool(self.config.get("pretrained", False))
        use_pretrained = use_pretrained or os.getenv("DINOV3_PRETRAINED", "0") == "1"
        self.backbone = timm.create_model(
            model_name,
            pretrained=use_pretrained,
            num_classes=0,
            img_size=input_size,
        )
        self.backbone_dim = int(self.backbone.num_features)

        # PAMAP2 has six frequency channels per modality (real/imaginary
        # channels after FFT). Each modality gets its own 1x1 input adapter.
        self.input_adapters = nn.ModuleDict()
        self.feature_adapters = nn.ModuleDict()
        for loc in self.locations:
            for mod in self.modalities:
                key = self._key(loc, mod)
                in_channels = args.dataset_config["loc_mod_in_freq_channels"][loc][mod]
                self.input_adapters[key] = nn.Conv2d(in_channels, 3, kernel_size=1)
                self.feature_adapters[key] = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.config["loc_out_channels"]),
                    nn.LayerNorm(self.config["loc_out_channels"]),
                    nn.GELU(),
                    nn.Dropout(self.config.get("dropout_ratio", 0.2)),
                )

        if self.config.get("freeze_backbone", False):
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        out_dim = args.dataset_config["FOCAL"]["emb_dim"]
        self.mod_projectors = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Linear(self.config["loc_out_channels"], out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )
            for mod in self.modalities
        })

        self.sample_dim = self.config["loc_out_channels"] * len(self.modalities)
        num_classes = args.dataset_config[args.task]["num_classes"]
        if args.train_mode == "supervised" or self.config["pretrained_head"] == "linear":
            self.class_layer = nn.Linear(self.sample_dim, num_classes)
        else:
            self.class_layer = nn.Sequential(
                nn.Linear(self.sample_dim, self.config["fc_dim"]),
                nn.GELU(),
                nn.Linear(self.config["fc_dim"], num_classes),
            )

    @staticmethod
    def _key(loc, mod):
        return f"{loc}__{mod}"

    def _prepare_input(self, tensor, adapter):
        """Convert a sensor spectrogram to a normalized 3-channel image."""
        tensor = tensor.float()
        tensor = adapter(tensor)
        tensor = F.interpolate(
            tensor,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = tensor.mean(dim=(2, 3), keepdim=True)
        std = tensor.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (tensor - mean) / std

    def _backbone_features(self, tensor):
        features = self.backbone(tensor)
        if isinstance(features, dict):
            features = features.get("x_norm_clstoken", features.get("x_norm_patchtokens"))
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim == 3:
            features = features[:, 0]
        return features

    def forward_encoder(self, freq_x, class_head=True, proj_head=False):
        modality_features = {}
        for mod in self.modalities:
            location_features = []
            for loc in self.locations:
                key = self._key(loc, mod)
                sensor_image = self._prepare_input(freq_x[loc][mod], self.input_adapters[key])
                dino_features = self._backbone_features(sensor_image)
                location_features.append(self.feature_adapters[key](dino_features))
            modality_features[mod] = torch.stack(location_features, dim=0).mean(dim=0)

        if not class_head:
            if proj_head:
                return {
                    mod: self.mod_projectors[mod](modality_features[mod])
                    for mod in self.modalities
                }
            return modality_features

        sample_features = torch.cat(
            [modality_features[mod] for mod in self.modalities], dim=1
        )
        return self.class_layer(sample_features)

    def forward(self, freq_x, class_head=True, proj_head=False):
        return self.forward_encoder(freq_x, class_head, proj_head)
