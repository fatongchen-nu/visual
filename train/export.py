from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from model import build_model, load_checkpoint
from utils import obstacle_from_mask


def _iter_representative_samples(config: Dict) -> Iterable[list[np.ndarray]]:
    root = Path(config["dataset"]["root"]) / config["dataset"]["image_dir"]
    paths = [p for p in sorted(root.glob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    limit = min(len(paths), int(config["export"]["representative_dataset_size"]))
    for p in paths[:limit]:
        import cv2

        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (config["training"]["image_size"], config["training"]["image_size"]))
        img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        yield [np.expand_dims(img, axis=0)]


def postprocess_prediction(logits: np.ndarray) -> Dict:
    probs = F.softmax(torch.from_numpy(logits), dim=1).numpy()[0]
    mask = np.argmax(probs, axis=0).astype(np.uint8)
    info = obstacle_from_mask(mask, obstacle_prob=probs[2])
    return {
        "mask": mask,
        "result_json": json.dumps(info, ensure_ascii=False),
    }


def export_coreml(model: torch.nn.Module, config: Dict, export_dir: Path) -> Path:
    import coremltools as ct

    model.eval()
    example = torch.randn(*config["export"]["input_size"])
    traced = torch.jit.trace(model.cpu(), example)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(shape=example.shape, name="input")],
        outputs=[ct.TensorType(name="logits")],
    )
    mlmodel.user_defined_metadata["postprocess"] = (
        "Run argmax on logits => mask, connected-components on class=2 to output "
        '{"has_obstacle": bool, "position": "left|center|right|none", "confidence": float}'
    )
    out_path = export_dir / config["export"]["coreml_name"]
    mlmodel.save(str(out_path))
    return out_path


def export_tflite_int8(model: torch.nn.Module, config: Dict, export_dir: Path) -> Path:
    import onnx
    import onnx_tf
    import tensorflow as tf

    model.eval()
    input_size = tuple(config["export"]["input_size"])
    dummy = torch.randn(*input_size)
    onnx_path = export_dir / "model.onnx"
    tf_path = export_dir / "tf_saved_model"

    torch.onnx.export(
        model.cpu(),
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=13,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    onnx_model = onnx.load(str(onnx_path))
    tf_rep = onnx_tf.backend.prepare(onnx_model)
    tf_rep.export_graph(str(tf_path))

    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _iter_representative_samples(config)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    out_path = export_dir / config["export"]["tflite_name"]
    out_path.write_bytes(tflite_model)
    return out_path


def export_all(config: Dict, checkpoint_path: str, device: torch.device | str = "cpu") -> None:
    export_dir = Path(config["export"]["output_dir"])
    export_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(
        num_classes=config["model"]["num_classes"],
        pretrained_backbone=False,
    )
    load_checkpoint(model, checkpoint_path, torch.device(device))

    print("Exporting CoreML...")
    try:
        coreml_path = export_coreml(model, config, export_dir)
        print(f"CoreML exported: {coreml_path}")
    except Exception as e:
        print(f"CoreML export skipped: {e}")

    print("Exporting TFLite INT8...")
    try:
        tflite_path = export_tflite_int8(model, config, export_dir)
        print(f"TFLite exported: {tflite_path}")
    except Exception as e:
        print(f"TFLite export skipped: {e}")

    (export_dir / "postprocess_spec.json").write_text(
        json.dumps(
            {
                "description": "Connected-component based obstacle localization",
                "output": {"has_obstacle": "bool", "position": "left|center|right|none", "confidence": "float"},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
