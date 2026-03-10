from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ObstacleInfo:
    has_obstacle: bool
    position: str
    confidence: float


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dims)
        cardinality = torch.sum(probs + one_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


def compute_iou(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> List[float]:
    preds = torch.argmax(logits, dim=1)
    ious = []
    for cls in range(num_classes):
        pred_mask = preds == cls
        gt_mask = targets == cls
        intersection = (pred_mask & gt_mask).sum().item()
        union = (pred_mask | gt_mask).sum().item()
        iou = intersection / union if union > 0 else float("nan")
        ious.append(iou)
    return ious


def summarize_iou(iou_values: List[List[float]]) -> Dict[str, float]:
    arr = np.array(iou_values, dtype=np.float32)
    cls_iou = np.nanmean(arr, axis=0)
    out = {f"class_{idx}_iou": float(v) for idx, v in enumerate(cls_iou)}
    out["miou"] = float(np.nanmean(cls_iou))
    return out


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    color_map = {
        0: np.array([0, 0, 0]),
        1: np.array([255, 255, 0]),
        2: np.array([255, 0, 0]),
    }
    overlay = np.zeros_like(image_rgb)
    for cls, color in color_map.items():
        overlay[mask == cls] = color
    return cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)


def obstacle_from_mask(mask: np.ndarray, obstacle_prob: np.ndarray | None = None) -> Dict:
    obstacle_bin = (mask == 2).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(obstacle_bin, connectivity=8)
    if num_labels <= 1:
        return ObstacleInfo(False, "none", 0.0).__dict__

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = int(np.argmax(areas)) + 1
    cx = float(centroids[best_idx][0])
    w = mask.shape[1]
    if cx < w / 3:
        pos = "left"
    elif cx < 2 * w / 3:
        pos = "center"
    else:
        pos = "right"

    confidence = 0.5
    if obstacle_prob is not None:
        comp = labels == best_idx
        confidence = float(np.clip(obstacle_prob[comp].mean(), 0.0, 1.0))

    return ObstacleInfo(True, pos, confidence).__dict__
