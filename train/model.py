from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torchvision.models.segmentation import lraspp_mobilenet_v3_large


class LRASPPWrapper(nn.Module):
    def __init__(self, num_classes: int = 3, pretrained_backbone: bool = True) -> None:
        super().__init__()
        weights_backbone = "DEFAULT" if pretrained_backbone else None
        self.model = lraspp_mobilenet_v3_large(weights=None, weights_backbone=weights_backbone, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: Dict[str, torch.Tensor] = self.model(x)
        return out["out"]


def build_model(num_classes: int = 3, pretrained_backbone: bool = True) -> nn.Module:
    return LRASPPWrapper(num_classes=num_classes, pretrained_backbone=pretrained_backbone)


def load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device) -> Dict:
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def save_checkpoint(state: Dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
