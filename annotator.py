from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import Image

DATASET = Path("dataset")
IMAGES_DIR = DATASET / "images"
MASKS_DIR = DATASET / "masks"
META_FILE = DATASET / "meta.json"


class AnnotationManager:
    def __init__(self) -> None:
        MASKS_DIR.mkdir(parents=True, exist_ok=True)
        self.meta: List[Dict] = json.loads(META_FILE.read_text(encoding="utf-8")) if META_FILE.exists() else []
        self.images = [IMAGES_DIR / m["file_name"] for m in self.meta] if self.meta else sorted(IMAGES_DIR.glob("*.jpg"))
        if not self.meta:
            self.meta = [{"file_name": p.name, "annotated": False, "auto_annotated": False} for p in self.images]
        self.index = 0
        self.cache_auto: Dict[str, Tuple[np.ndarray, bool, float]] = {}

    def save_meta(self) -> None:
        META_FILE.write_text(json.dumps(self.meta, indent=2, ensure_ascii=False), encoding="utf-8")

    def current_image(self) -> np.ndarray:
        return np.array(Image.open(self.images[self.index]).convert("RGB"))

    def mask_path(self, idx: int) -> Path:
        return MASKS_DIR / f"{self.images[idx].stem}.png"

    def heuristic_auto_label(self, img: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        yellow1 = np.array([15, 60, 60])
        yellow2 = np.array([40, 255, 255])
        tactile = cv2.inRange(hsv, yellow1, yellow2)

        # obstacle heuristic: non-yellow strong edges + grabcut fg
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
        img = self.current_image()
        key = self.images[self.index].name
        if key not in self.cache_auto:
            self.cache_auto[key] = self.heuristic_auto_label(img)
        auto_mask, high_conf, conf = self.cache_auto[key]

        saved_path = self.mask_path(self.index)
        mask = np.array(Image.open(saved_path)) if saved_path.exists() else auto_mask
        return img, mask, high_conf, conf

    def save_mask(self, mask: np.ndarray, auto_annotated: bool):
        mask = mask.astype(np.uint8)
        Image.fromarray(mask).save(self.mask_path(self.index))
        self.meta[self.index]["annotated"] = True
        self.meta[self.index]["auto_annotated"] = auto_annotated
        self.save_meta()

    def skip(self):
        self.index = min(self.index + 1, len(self.images) - 1)

    def next(self):
        self.index = min(self.index + 1, len(self.images) - 1)

    def progress_text(self) -> str:
        return f"{self.index + 1}/{len(self.images)}"

    def bulk_accept_high_conf(self) -> str:
        saved = 0
        for i in range(len(self.images)):
            self.index = i
            img, mask, high_conf, _ = self.load_current()
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
    img, mask, high_conf, conf = manager.load_current()
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
            batch_btn = gr.Button("自动处理全部高置信度")
            batch_result = gr.Textbox(label="批处理结果")

    thumbs = gr.Gallery(label="缩略图队列(高置信度建议优先)", columns=8, rows=1, height=140)
    mask_state = gr.State()

    def refresh_thumbs():
        items = []
        for i, p in enumerate(manager.images[:64]):
            img = np.array(Image.open(p).convert("RGB").resize((128, 96)))
            _, _, high, _ = manager.load_current() if i == manager.index else (None, None, False, None)
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
    batch_btn.click(lambda: manager.bulk_accept_high_conf(), outputs=[batch_result])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
