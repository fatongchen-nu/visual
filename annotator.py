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
        if self.active_indices:
            self.index = min(self.index, len(self.active_indices) - 1)
        else:
            self.index = 0

    def has_items(self) -> bool:
        return len(self.active_indices) > 0

    def current_meta_idx(self) -> int:
        if not self.has_items():
            raise RuntimeError("No active images available")
        return self.active_indices[self.index]

    def current_image_path(self) -> Path:
        return IMAGES_DIR / self.meta[self.current_meta_idx()]["file_name"]

    def save_meta(self) -> None:
        META_FILE.write_text(json.dumps(self.meta, indent=2, ensure_ascii=False), encoding="utf-8")

    def current_image(self) -> np.ndarray:
        return np.array(Image.open(self.current_image_path()).convert("RGB"))

    def mask_path(self, meta_idx: int) -> Path:
        name = Path(self.meta[meta_idx]["file_name"]).stem
        return MASKS_DIR / f"{name}.png"

    def heuristic_auto_label(self, img: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        yellow1 = np.array([15, 60, 60])
        yellow2 = np.array([40, 255, 255])
        tactile = cv2.inRange(hsv, yellow1, yellow2)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        obstacle_seed = ((edges > 0) & (tactile == 0)).astype(np.uint8) * 255

        gc_mask = np.full(gray.shape, cv2.GC_PR_BGD, np.uint8)
        gc_mask[tactile > 0] = cv2.GC_BGD
        gc_mask[obstacle_seed > 0] = cv2.GC_PR_FGD
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        rect = (1, 1, img.shape[1] - 2, img.shape[0] - 2)
        cv2.grabCut(img, gc_mask, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_MASK)

        obstacle = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        obstacle[tactile > 0] = 0

        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[tactile > 0] = 1
        mask[obstacle > 0] = 2

        tactile_ratio = float((mask == 1).mean())
        confidence = 1.0 - min(abs(tactile_ratio - 0.2) / 0.2, 1.0)
        high_conf = 0.05 <= tactile_ratio <= 0.4
        return mask, high_conf, confidence

    def load_current(self):
        if not self.has_items():
            return None

        img = self.current_image()
        key = self.current_image_path().name
        if key not in self.cache_auto:
            self.cache_auto[key] = self.heuristic_auto_label(img)
        auto_mask, high_conf, conf = self.cache_auto[key]

        meta_idx = self.current_meta_idx()
        saved_path = self.mask_path(meta_idx)
        mask = np.array(Image.open(saved_path)) if saved_path.exists() else auto_mask
        return img, mask, high_conf, conf

    def save_mask(self, mask: np.ndarray, auto_annotated: bool):
        if not self.has_items():
            return
        meta_idx = self.current_meta_idx()
        mask = mask.astype(np.uint8)
        Image.fromarray(mask).save(self.mask_path(meta_idx))
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
        if not self.has_items():
            return "0/0"
        return f"{self.index + 1}/{len(self.active_indices)}"

    def bulk_accept_high_conf(self) -> str:
        saved = 0
        for active_pos in range(len(self.active_indices)):
            self.index = active_pos
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


def composite(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = img.copy()
    overlay[mask == 1] = (255, 255, 0)
    overlay[mask == 2] = (255, 0, 0)
    return cv2.addWeighted(img, 0.55, overlay, 0.45, 0)


def load_item():
    loaded = manager.load_current()
    if loaded is None:
        empty = np.zeros((512, 512, 3), dtype=np.uint8)
        mask = np.zeros((512, 512), dtype=np.uint8)
        return empty, empty, mask, "0/0", "no image"

    img, mask, high_conf, conf = loaded
    return img, composite(img, mask), mask, manager.progress_text(), f"{conf:.2f} ({'high' if high_conf else 'low'})"


def paint(mask: np.ndarray, evt: gr.SelectData, label: int, brush_size: int):
    x, y = int(evt.index[0]), int(evt.index[1])
    cv2.circle(mask, (x, y), brush_size, int(label), -1)
    img = manager.current_image()
    return composite(img, mask), mask


def save_auto(mask):
    manager.save_mask(mask, auto_annotated=True)
    manager.next()
    return load_item()


def save_manual(mask):
    manager.save_mask(mask, auto_annotated=False)
    manager.next()
    return load_item()


def skip_item():
    manager.skip()
    return load_item()


def discard_item():
    msg = manager.discard_current()
    return (*load_item(), msg, refresh_thumbs())


with gr.Blocks() as demo:
    gr.Markdown("## Tactile Path Segmentation Annotator")
    with gr.Row():
        orig = gr.Image(label="原图", type="numpy")
        over = gr.Image(label="Mask叠加", type="numpy")
        with gr.Column():
            progress = gr.Textbox(label="进度")
            conf = gr.Textbox(label="置信度")
            cls = gr.Radio([0, 1, 2], value=1, label="类别 (0背景/1盲道/2障碍物)")
            brush = gr.Slider(1, 40, value=8, step=1, label="画笔大小")
            erase = gr.Button("橡皮擦(设为背景)")
            accept_auto = gr.Button("接受自动标注")
            save_btn = gr.Button("手动修改后保存")
            skip_btn = gr.Button("跳过")
            discard_btn = gr.Button("Discard 当前图片")
            batch_btn = gr.Button("自动处理全部高置信度")
            batch_result = gr.Textbox(label="批处理结果")

    thumbs = gr.Gallery(label="缩略图队列(高置信度建议优先)", columns=8, rows=1, height=140)
    mask_state = gr.State()

    def refresh_thumbs():
        items = []
        for pos, meta_idx in enumerate(manager.active_indices[:64]):
            p = IMAGES_DIR / manager.meta[meta_idx]["file_name"]
            img = np.array(Image.open(p).convert("RGB").resize((128, 96)))
            high = False
            if pos == manager.index and manager.has_items():
                loaded = manager.load_current()
                if loaded is not None:
                    _, _, high, _ = loaded
            if high:
                img[:6, :, :] = np.array([0, 255, 0])
            items.append((img, p.name))
        return items

    demo.load(load_item, outputs=[orig, over, mask_state, progress, conf])
    demo.load(refresh_thumbs, outputs=[thumbs])

    over.select(paint, inputs=[mask_state, cls, brush], outputs=[over, mask_state])
    erase.click(lambda: 0, outputs=cls)
    accept_auto.click(save_auto, inputs=[mask_state], outputs=[orig, over, mask_state, progress, conf])
    save_btn.click(save_manual, inputs=[mask_state], outputs=[orig, over, mask_state, progress, conf])
    skip_btn.click(skip_item, outputs=[orig, over, mask_state, progress, conf])
    discard_btn.click(discard_item, outputs=[orig, over, mask_state, progress, conf, batch_result, thumbs])
    batch_btn.click(lambda: manager.bulk_accept_high_conf(), outputs=[batch_result])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
