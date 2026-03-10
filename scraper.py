from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from zipfile import ZIP_DEFLATED, ZipFile


def ensure_packages() -> None:
    required = ["icrawler", "Pillow", "imagehash", "tqdm"]
    for pkg in required:
        try:
            __import__(pkg if pkg != "Pillow" else "PIL")
        except Exception:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


ensure_packages()

import imagehash
import numpy as np
from icrawler.builtin import BaiduImageCrawler, BingImageCrawler, GoogleImageCrawler
from PIL import Image
from tqdm import tqdm

QUERIES = [
    "tactile paving",
    "tactile path sidewalk",
    "blindo path obstacle",
    "盲道",
    "盲道障碍物",
    "导盲砖",
    "tactile ground surface indicator",
    "TGSI path",
    "yellow tactile tiles street",
]


class LoggingCrawler:
    def __init__(self, crawler_cls, engine: str, storage_root: Path):
        self.engine = engine
        self.storage_root = storage_root
        self.crawler = crawler_cls(storage={"root_dir": str(storage_root)}, log_level="ERROR")

    def crawl(self, keyword: str, max_num: int) -> List[Path]:
        before = set(self.storage_root.glob("*"))
        self.crawler.crawl(keyword=keyword, max_num=max_num, overwrite=False)
        after = set(self.storage_root.glob("*"))
        return sorted(after - before)


def is_valid_image(path: Path) -> Tuple[bool, Dict]:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size
            if w < 300 or h < 300:
                return False, {}
            arr = np.asarray(img)
            if float(arr.std()) < 10.0:
                return False, {}
            return True, {"width": w, "height": h}
    except Exception:
        return False, {}


def build_dataset(max_images: int) -> None:
    dataset_root = Path("dataset")
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    raw_dir = dataset_root / "_raw"
    for p in [images_dir, masks_dir, raw_dir]:
        p.mkdir(parents=True, exist_ok=True)

    crawlers = [
        LoggingCrawler(GoogleImageCrawler, "google", raw_dir / "google"),
        LoggingCrawler(BingImageCrawler, "bing", raw_dir / "bing"),
        LoggingCrawler(BaiduImageCrawler, "baidu", raw_dir / "baidu"),
    ]
    for c in crawlers:
        c.storage_root.mkdir(parents=True, exist_ok=True)

    seen_hashes = set()
    meta = []
    saved = 0

    iterator = [(q, c) for q in QUERIES for c in crawlers]
    for query, crawler in tqdm(iterator, desc="crawling"):
        if saved >= max_images:
            break
        files = crawler.crawl(query, max_num=30)
        for file in files:
            if saved >= max_images:
                break
            ok, info = is_valid_image(file)
            if not ok:
                continue

            with Image.open(file) as img:
                ph = str(imagehash.phash(img.convert("RGB")))
                if ph in seen_hashes:
                    continue
                seen_hashes.add(ph)

                out_name = f"img_{saved + 1:05d}.jpg"
                out_path = images_dir / out_name
                img.convert("RGB").save(out_path, quality=95)

            meta.append(
                {
                    "file_name": out_name,
                    "source_engine": crawler.engine,
                    "source_file": str(file),
                    "query": query,
                    "width": info["width"],
                    "height": info["height"],
                    "annotated": False,
                    "auto_annotated": False,
                }
            )
            saved += 1

    (dataset_root / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (dataset_root / "README.txt").write_text(
        "Mask format: single-channel PNG, pixel values: 0=background, 1=tactile paving, 2=obstacle.\n"
        "Mask file name must match image stem.\n",
        encoding="utf-8",
    )

    zip_name = f"tactile_dataset_{datetime.now().strftime('%Y%m%d')}.zip"
    with ZipFile(zip_name, "w", ZIP_DEFLATED) as zf:
        for p in dataset_root.rglob("*"):
            if p.is_file() and "_raw" not in p.parts:
                zf.write(p, p.relative_to(Path(".")))

    print(f"Saved {saved} images. Dataset zip: {zip_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_images", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(max_images=args.max_images)
