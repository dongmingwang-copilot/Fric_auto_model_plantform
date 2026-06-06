from __future__ import annotations

import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app.core.config import Settings
from app.ml.image_ops import load_uint8_gray
from app.services.checkpoints import CheckpointService
from app.services.datasets import DatasetService
from app.services.inference import InferenceService


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _annotation_path(item: dict, label_id: str) -> str | None:
    annotation = item.get("annotations", {}).get(label_id)
    if annotation and annotation.get("path"):
        return annotation["path"]
    if label_id == "spall" and item.get("annotation_path"):
        return item["annotation_path"]
    return None


def _overlay(base: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.42) -> Image.Image:
    arr = np.array(base.convert("RGB"), dtype=np.float32)
    hit = mask.astype(bool)
    arr[hit] = arr[hit] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode="RGB")


def _diff_overlay(base: Image.Image, gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    arr = np.array(base.convert("RGB"), dtype=np.float32)
    fn = gt & ~pred
    fp = pred & ~gt
    tp = gt & pred
    arr[tp] = arr[tp] * 0.72 + np.array([80, 170, 255], dtype=np.float32) * 0.28
    arr[fn] = arr[fn] * 0.25 + np.array([255, 45, 35], dtype=np.float32) * 0.75
    arr[fp] = arr[fp] * 0.25 + np.array([30, 210, 95], dtype=np.float32) * 0.75
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode="RGB")


def _fit_width(img: Image.Image, width: int) -> Image.Image:
    if img.width == width:
        return img
    height = max(1, round(img.height * (width / img.width)))
    return img.resize((width, height), Image.Resampling.BILINEAR)


def _triptych(original: Image.Image, gt_overlay: Image.Image, diff: Image.Image, title: str) -> Image.Image:
    cell_w = 360
    header_h = 30
    gap = 8
    cells = [_fit_width(img.convert("RGB"), cell_w) for img in (original, gt_overlay, diff)]
    cell_h = max(img.height for img in cells)
    out = Image.new("RGB", (cell_w * 3 + gap * 2, cell_h + header_h), (248, 250, 251))
    draw = ImageDraw.Draw(out)
    labels = ["原图", "人工GT", "差异: 红=漏检 绿=多检 蓝=命中"]
    for idx, img in enumerate(cells):
        x = idx * (cell_w + gap)
        draw.text((x + 6, 8), labels[idx], fill=(20, 28, 35))
        out.paste(img, (x, header_h))
    draw.text((6, cell_h + header_h - 18), title[:90], fill=(20, 28, 35))
    return out


class ModelTestService:
    def __init__(self, settings: Settings, datasets: DatasetService, checkpoints: CheckpointService, inference: InferenceService):
        self.settings = settings
        self.datasets = datasets
        self.checkpoints = checkpoints
        self.inference = inference

    def list(self, dataset_id: str) -> list[dict]:
        root = self.settings.tests_dir / dataset_id
        if not root.exists():
            return []
        rows = []
        for path in sorted(root.glob("*/manifest.json"), reverse=True):
            try:
                rows.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return rows

    def get(self, dataset_id: str, run_id: str) -> dict:
        path = self.settings.tests_dir / dataset_id / run_id / "manifest.json"
        if not path.exists():
            raise KeyError(f"Unknown model test: {run_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def run(
        self,
        dataset_id: str,
        checkpoint_id: str,
        label_id: str,
        threshold: float,
        tile: int,
        stride: int,
        sample_count: int,
        seed: int | None,
    ) -> dict:
        self.checkpoints.get(checkpoint_id)
        items = self.datasets.list_items(dataset_id)
        reviewed = [item for item in items if _annotation_path(item, label_id)]
        if not reviewed:
            raise RuntimeError("当前数据集没有可测试的已 Review GT。")
        rng = random.Random(seed if seed is not None else datetime.now().timestamp())
        selected = list(reviewed)
        rng.shuffle(selected)
        selected = selected[:min(sample_count, len(selected))]

        run_id = f"test-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        out_dir = self.settings.tests_dir / dataset_id / run_id
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        total_tp = total_fp = total_fn = 0
        for idx, item in enumerate(selected, start=1):
            image = load_uint8_gray(Path(item["original_path"]))
            gt_img = Image.open(_annotation_path(item, label_id)).convert("L")
            if gt_img.size != image.size:
                gt_img = gt_img.resize(image.size, Image.Resampling.NEAREST)
            gt = np.array(gt_img) > 0
            prob = self.inference.predict_probability(checkpoint_id, image, tile, stride)
            pred = prob >= threshold
            tp = int((pred & gt).sum())
            fp = int((pred & ~gt).sum())
            fn = int((~pred & gt).sum())
            total_tp += tp
            total_fp += fp
            total_fn += fn
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            dice = (2 * tp) / max(2 * tp + fp + fn, 1)

            stem = f"{idx:03d}_{item['id']}"
            original = image.convert("RGB")
            gt_overlay = _overlay(original, gt, (255, 55, 40), alpha=0.38)
            diff = _diff_overlay(original, gt, pred)
            triptych = _triptych(original, gt_overlay, diff, f"{item.get('original_name')}  D {dice:.4f}  R {recall:.4f}  P {precision:.4f}")
            row_file = f"{stem}_row.png"
            triptych.save(out_dir / row_file)
            rows.append({
                "item_id": item["id"],
                "original_name": item.get("original_name"),
                "row": row_file,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "dice": dice,
            })

        total_precision = total_tp / max(total_tp + total_fp, 1)
        total_recall = total_tp / max(total_tp + total_fn, 1)
        total_dice = (2 * total_tp) / max(2 * total_tp + total_fp + total_fn, 1)
        manifest = {
            "id": run_id,
            "dataset_id": dataset_id,
            "checkpoint_id": checkpoint_id,
            "label_id": label_id,
            "threshold": threshold,
            "tile": tile,
            "stride": stride,
            "sample_count": len(rows),
            "available_reviewed": len(reviewed),
            "seed": seed,
            "created_at": _now(),
            "metrics": {
                "precision": total_precision,
                "recall": total_recall,
                "dice": total_dice,
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
            },
            "rows": rows,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return manifest
