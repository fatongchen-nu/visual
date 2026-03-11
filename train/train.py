from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TactileSegDataset, create_splits
from export import export_all
from model import build_model, load_checkpoint, save_checkpoint
from utils import DiceLoss, compute_iou, summarize_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train/config.yaml")
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, ce_loss, dice_loss, device, num_classes):
    model.train()
    losses = []
    all_ious = []
    for images, masks in tqdm(loader, desc="train", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = ce_loss(logits, masks) + dice_loss(logits, masks, num_classes)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        all_ious.append(compute_iou(logits.detach(), masks, num_classes))

    metrics = summarize_iou(all_ious)
    metrics["loss"] = float(sum(losses) / max(len(losses), 1))
    return metrics


@torch.no_grad()
def validate(model, loader, ce_loss, dice_loss, device, num_classes):
    model.eval()
    losses = []
    all_ious = []
    for images, masks in tqdm(loader, desc="val", leave=False):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss = ce_loss(logits, masks) + dice_loss(logits, masks, num_classes)
        losses.append(loss.item())
        all_ious.append(compute_iou(logits, masks, num_classes))

    metrics = summarize_iou(all_ious)
    metrics["loss"] = float(sum(losses) / max(len(losses), 1))
    return metrics


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    torch.manual_seed(cfg["seed"])

    num_classes = cfg["model"]["num_classes"]
    train_imgs, train_masks, val_imgs, val_masks = create_splits(
        root=cfg["dataset"]["root"],
        image_dir=cfg["dataset"]["image_dir"],
        mask_dir=cfg["dataset"]["mask_dir"],
        val_split=cfg["training"]["val_split"],
        seed=cfg["seed"],
    )

    train_set = TactileSegDataset(train_imgs, train_masks, image_size=cfg["training"]["image_size"], is_train=True)
    val_set = TactileSegDataset(val_imgs, val_masks, image_size=cfg["training"]["image_size"], is_train=False)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        shuffle=False,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=num_classes, pretrained_backbone=cfg["model"]["pretrained_backbone"]).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()

    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    best_path = ckpt_dir / cfg["training"]["best_model_name"]
    last_path = ckpt_dir / cfg["training"]["last_model_name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_miou = -1.0

    if args.resume:
        checkpoint = load_checkpoint(model, args.resume, device)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint.get("best_miou", best_miou)
        print(f"Resumed from {args.resume} @ epoch {start_epoch}")

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        tr = train_one_epoch(model, train_loader, optimizer, ce_loss, dice_loss, device, num_classes)
        va = validate(model, val_loader, ce_loss, dice_loss, device, num_classes)
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{cfg['training']['epochs']} | "
            f"train loss={tr['loss']:.4f} miou={tr['miou']:.4f} | "
            f"val loss={va['loss']:.4f} miou={va['miou']:.4f} | "
            f"cls IoU: bg={va['class_0_iou']:.4f} tactile={va['class_1_iou']:.4f} obstacle={va['class_2_iou']:.4f}"
        )

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_miou": best_miou,
            "config": cfg,
        }
        save_checkpoint(state, str(last_path))

        if va["miou"] > best_miou:
            best_miou = va["miou"]
            state["best_miou"] = best_miou
            save_checkpoint(state, str(best_path))
            print(f"[best] saved -> {best_path} (mIoU={best_miou:.4f})")

    export_all(config=cfg, checkpoint_path=str(best_path), device=device)


if __name__ == "__main__":
    main()
