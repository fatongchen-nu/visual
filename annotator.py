from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import Image

DATASET = Path("dataset")
IMAGES_DIR = DATASET / "images"
MASKS_DIR = DATASET / "masks"
DISCARDED_DIR = DATASET / "discarded"
META_FILE = DATASET / "meta.json"

CLASS_COLORS = {
    0: np.array([0, 0, 0], dtype=np.uint8),
    1: np.array([255, 255, 0], dtype=np.uint8),  # tactile: yellow
    2: np.array([255, 0, 0], dtype=np.uint8),    # obstacle: red
}


class AnnotationManager:
    def __init__(self) -> None:
        MASKS_DIR.mkdir(parents=True, exist_ok=True)
        DISCARDED_DIR.mkdir(parents=True, exist_ok=True)

        self.meta: List[Dict] = json.loads(META_FILE.read_text(encoding="utf-8")) if META_FILE.exists() else []
        if not self.meta:
            image_files = sorted([p for p in IMAGES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            self.meta = [
                {"file_name": p.name, "annotated": False, "auto_annotated": False, "discarded": False}
                for p in image_files
            ]

        for item in self.meta:
            item.setdefault("annotated", False)
            item.setdefault("auto_annotated", False)
            item.setdefault("discarded", False)

        self.active_indices: List[int] = []
        self.index = 0
        self.cache_auto: Dict[str, Tuple[np.ndarray, bool, float]] = {}
        self.refresh_active_indices()

    def refresh_active_indices(self) -> None:
        self.active_indices = [
            i
            for i, item in enumerate(self.meta)
            if not item.get("discarded", False) and (IMAGES_DIR / item["file_name"]).exists()
        ]
        self.index = min(self.index, max(len(self.active_indices) - 1, 0))

    def has_items(self) -> bool:
        return len(self.active_indices) > 0

    def current_meta_idx(self) -> int:
        if not self.has_items():
            raise RuntimeError("No active images available")
        return self.active_indices[self.index]

    def image_path(self, meta_idx: int) -> Path:
        return IMAGES_DIR / self.meta[meta_idx]["file_name"]

    def current_image_path(self) -> Path:
        return self.image_path(self.current_meta_idx())

    def save_meta(self) -> None:
        META_FILE.write_text(json.dumps(self.meta, indent=2, ensure_ascii=False), encoding="utf-8")

    def read_image(self, meta_idx: int) -> np.ndarray:
        return np.array(Image.open(self.image_path(meta_idx)).convert("RGB"))

    def current_image(self) -> np.ndarray:
        return self.read_image(self.current_meta_idx())

    def mask_path(self, meta_idx: int) -> Path:
        name = Path(self.meta[meta_idx]["file_name"]).stem
        return MASKS_DIR / f"{name}.png"

    def region_auto_label(self, img: np.ndarray) -> np.ndarray:
        """Auto pre-label using color + edge cues + kmeans region partition."""
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 80, 160)

        yellow = cv2.inRange(hsv, np.array([15, 60, 60]), np.array([40, 255, 255]))

        z = img.reshape((-1, 3)).astype(np.float32)
        k = min(5, max(3, (h * w) // (256 * 256) + 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.2)
        _ret, labels, centers = cv2.kmeans(z, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        labels = labels.reshape((h, w))

        mask = np.zeros((h, w), dtype=np.uint8)
        for cid in range(k):
            region = labels == cid
            if not np.any(region):
                continue
            yellow_ratio = float((yellow[region] > 0).mean())
            edge_ratio = float((edges[region] > 0).mean())

            if yellow_ratio > 0.18:
                mask[region] = 1
            elif edge_ratio > 0.12:
                mask[region] = 2

        # refine obstacle with grabcut foreground priors
        obstacle_seed = ((edges > 0) & (yellow == 0)).astype(np.uint8) * 255
        gc_mask = np.full(gray.shape, cv2.GC_PR_BGD, np.uint8)
        gc_mask[yellow > 0] = cv2.GC_BGD
        gc_mask[obstacle_seed > 0] = cv2.GC_PR_FGD
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        rect = (1, 1, max(1, w - 2), max(1, h - 2))
        cv2.grabCut(img, gc_mask, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_MASK)
        obstacle = ((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD))
        mask[obstacle & (mask != 1)] = 2

        return mask

    def get_auto_for_meta_idx(self, meta_idx: int) -> Tuple[np.ndarray, bool, float]:
        key = self.meta[meta_idx]["file_name"]
        if key not in self.cache_auto:
            img = self.read_image(meta_idx)
            auto_mask = self.region_auto_label(img)
            tactile_ratio = float((auto_mask == 1).mean())
            confidence = 1.0 - min(abs(tactile_ratio - 0.2) / 0.2, 1.0)
            high_conf = 0.05 <= tactile_ratio <= 0.4
            self.cache_auto[key] = (auto_mask, high_conf, confidence)
        return self.cache_auto[key]

    def load_current(self):
        if not self.has_items():
            return None

        meta_idx = self.current_meta_idx()
        img = self.read_image(meta_idx)
        auto_mask, high_conf, conf = self.get_auto_for_meta_idx(meta_idx)
        saved_path = self.mask_path(meta_idx)
        mask = np.array(Image.open(saved_path), dtype=np.uint8) if saved_path.exists() else auto_mask
        return img, mask, high_conf, conf

    def save_mask(self, mask: np.ndarray, auto_annotated: bool):
        if not self.has_items():
            return
        meta_idx = self.current_meta_idx()
        Image.fromarray(mask.astype(np.uint8)).save(self.mask_path(meta_idx))
        self.meta[meta_idx]["annotated"] = True
        self.meta[meta_idx]["auto_annotated"] = auto_annotated
        self.meta[meta_idx]["discarded"] = False
        self.save_meta()

    def discard_current(self) -> str:
        if not self.has_items():
            return "No active image to discard"

        meta_idx = self.current_meta_idx()
        file_name = self.meta[meta_idx]["file_name"]
        src = IMAGES_DIR / file_name
        dst = DISCARDED_DIR / file_name
        if src.exists():
            shutil.move(str(src), str(dst))

        mask_path = self.mask_path(meta_idx)
        if mask_path.exists():
            mask_path.unlink()

        self.meta[meta_idx]["annotated"] = False
        self.meta[meta_idx]["auto_annotated"] = False
        self.meta[meta_idx]["discarded"] = True
        self.save_meta()

        self.cache_auto.pop(file_name, None)
        self.refresh_active_indices()
        return f"Discarded: {file_name}"

    def skip(self):
        if self.has_items():
            self.index = min(self.index + 1, len(self.active_indices) - 1)

    def next(self):
        if self.has_items():
            self.index = min(self.index + 1, len(self.active_indices) - 1)

    def progress_text(self) -> str:
        return f"{self.index + 1}/{len(self.active_indices)}" if self.has_items() else "0/0"

    def bulk_accept_high_conf(self) -> str:
        saved = 0
        for pos in range(len(self.active_indices)):
            self.index = pos
            loaded = self.load_current()
            if loaded is None:
                continue
            _, mask, high_conf, _ = loaded
            if high_conf:
                self.save_mask(mask, auto_annotated=True)
                saved += 1
        self.index = 0
        return f"Auto-saved {saved} high-confidence masks"


manager = AnnotationManager()


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, rgb in CLASS_COLORS.items():
        color[mask == cls] = rgb
    return color


def color_to_mask(color_img: np.ndarray) -> np.ndarray:
    if color_img.ndim == 2:
        return color_img.astype(np.uint8)
    if color_img.shape[-1] == 4:
        color_img = color_img[..., :3]

    h, w = color_img.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    colors = np.stack([CLASS_COLORS[0], CLASS_COLORS[1], CLASS_COLORS[2]], axis=0).astype(np.int16)
    pixels = color_img.astype(np.int16).reshape(-1, 1, 3)
    dists = np.sum((pixels - colors[None, ...]) ** 2, axis=2)
    cls = np.argmin(dists, axis=1).astype(np.uint8)
    out[:] = cls.reshape(h, w)
    return out


def extract_editor_rgb(editor_value) -> np.ndarray:
    if isinstance(editor_value, dict):
        img = editor_value.get("composite") or editor_value.get("background")
    else:
        img = editor_value
    arr = np.array(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def make_editor_value(mask_rgb: np.ndarray) -> dict:
    """Build a stable ImageEditor value payload for Gradio 4.x."""
    return {
        "background": mask_rgb,
        "layers": [],
        "composite": mask_rgb,
    }


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = image_rgb.copy()
    overlay[mask == 1] = np.array([255, 255, 0], dtype=np.uint8)
    overlay[mask == 2] = np.array([255, 0, 0], dtype=np.uint8)
    return cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)


def load_item():
    loaded = manager.load_current()
    if loaded is None:
        empty = np.zeros((512, 512, 3), dtype=np.uint8)
        return empty, make_editor_value(empty), empty, "0/0", "no image", refresh_thumbs()

    img, mask, high_conf, conf = loaded
    mask_rgb = mask_to_color(mask)
    return (
        img,
        make_editor_value(mask_rgb),
        overlay_mask(img, mask),
        manager.progress_text(),
        f"{conf:.2f} ({'high' if high_conf else 'low'})",
        refresh_thumbs(),
    )


def preview_overlay(editor_value):
    loaded = manager.load_current()
    if loaded is None:
        empty = np.zeros((512, 512, 3), dtype=np.uint8)
        return empty
    img, _, _, _ = loaded
    mask = color_to_mask(extract_editor_rgb(editor_value))
    return overlay_mask(img, mask)


def save_auto():
    loaded = manager.load_current()
    if loaded is not None:
        _, auto_mask, _, _ = loaded
        manager.save_mask(auto_mask, auto_annotated=True)
    manager.next()
    return load_item()


def save_manual(editor_value):
    mask = color_to_mask(extract_editor_rgb(editor_value))
    manager.save_mask(mask, auto_annotated=False)
    manager.next()
    return load_item()


def reset_to_auto():
    loaded = manager.load_current()
    if loaded is None:
        empty = np.zeros((512, 512, 3), dtype=np.uint8)
        return make_editor_value(empty), empty
    img, auto_mask, _, _ = loaded
    auto_rgb = mask_to_color(auto_mask)
    return make_editor_value(auto_rgb), overlay_mask(img, auto_mask)


def skip_item():
    manager.skip()
    return load_item()


def discard_item():
    msg = manager.discard_current()
    loaded = load_item()
    return (*loaded, msg)


def refresh_thumbs():
    items = []
    for pos, meta_idx in enumerate(manager.active_indices[:64]):
        p = manager.image_path(meta_idx)
        img = np.array(Image.open(p).convert("RGB").resize((128, 96)))
        # Avoid running heavy auto-labeling for dozens of images during initial UI load.
        # Only trust cache if it's already available; otherwise skip confidence highlight.
        cached = manager.cache_auto.get(manager.meta[meta_idx]["file_name"])
        high = bool(cached[1]) if cached is not None else False
        if high:
            img[:6, :, :] = np.array([0, 255, 0])
        if pos == manager.index:
            img[:, :4, :] = np.array([255, 255, 255])
        items.append((img, p.name))
    return items


with gr.Blocks() as demo:
    gr.Markdown(
        "## Tactile Path Segmentation Annotator\n"
        "- 中间画布支持涂鸦（画笔颜色：黑=背景，黄=盲道，红=障碍物）\n"
        "- 可先用自动分区结果，再手工修正"
    )
    with gr.Row():
        orig = gr.Image(label="原图", type="numpy")
        editor = gr.ImageEditor(
            label="Mask 可编辑画布（可涂鸦）",
            type="numpy",
            brush=gr.Brush(colors=["#000000", "#FFFF00", "#FF0000"], color_mode="fixed", default_size=10),
            eraser=gr.Eraser(default_size=20),
        )
        over = gr.Image(label="叠加预览", type="numpy")

    with gr.Row():
        progress = gr.Textbox(label="进度")
        conf = gr.Textbox(label="置信度")
        batch_result = gr.Textbox(label="状态")

    with gr.Row():
        preview_btn = gr.Button("刷新叠加预览")
        reset_btn = gr.Button("重置为自动预标注")
        accept_auto = gr.Button("接受自动标注")
        save_btn = gr.Button("手动修改后保存")
        skip_btn = gr.Button("跳过")
        discard_btn = gr.Button("Discard 当前图片")
        batch_btn = gr.Button("自动处理全部高置信度")

    thumbs = gr.Gallery(label="缩略图队列(高置信度高亮)", columns=8, rows=1, height=140)

    demo.load(load_item, outputs=[orig, editor, over, progress, conf, thumbs])

    preview_btn.click(preview_overlay, inputs=[editor], outputs=[over])
    reset_btn.click(reset_to_auto, outputs=[editor, over])
    accept_auto.click(save_auto, outputs=[orig, editor, over, progress, conf, thumbs])
    save_btn.click(save_manual, inputs=[editor], outputs=[orig, editor, over, progress, conf, thumbs])
    skip_btn.click(skip_item, outputs=[orig, editor, over, progress, conf, thumbs])
    discard_btn.click(discard_item, outputs=[orig, editor, over, progress, conf, thumbs, batch_result])
    batch_btn.click(lambda: manager.bulk_accept_high_conf(), outputs=[batch_result])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
