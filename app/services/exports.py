from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.core.config import Settings
from app.core.json_store import write_json
from app.services.datasets import DatasetService


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class ExportService:
    def __init__(self, settings: Settings, datasets: DatasetService):
        self.settings = settings
        self.datasets = datasets

    def export(self, dataset_id: str, export_format: str, label_id: str = "spall", include_predictions: bool = True) -> dict:
        if export_format == "active_learning":
            return self.export_active_learning(dataset_id, label_id, include_predictions)
        if export_format == "coco":
            return self.export_coco(dataset_id, label_id)
        if export_format == "fiftyone":
            return self.export_fiftyone(dataset_id, label_id, include_predictions)
        raise ValueError(f"Unsupported export format: {export_format}")

    def _export_root(self, dataset_id: str, export_format: str) -> Path:
        path = self.settings.exports_dir / dataset_id / f"{_run_id()}_{export_format}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _annotation_for(self, item: dict, label_id: str) -> dict | None:
        annotation = item.get("annotations", {}).get(label_id)
        if annotation:
            return annotation
        if label_id == "spall" and item.get("annotation_path"):
            return {
                "path": item["annotation_path"],
                "mask_px": item.get("annotation", {}).get("mask_px"),
                "source": item.get("annotation", {}).get("source", "legacy"),
            }
        return None

    def _prediction_for(self, item: dict) -> dict | None:
        latest = item.get("latest_prediction")
        if not latest:
            return None
        return item.get("predictions", {}).get(latest)

    def export_active_learning(self, dataset_id: str, label_id: str, include_predictions: bool) -> dict:
        meta = self.datasets.get_meta(dataset_id)
        root = self._export_root(dataset_id, "active_learning")
        samples = []
        for item in meta.get("items", []):
            annotation = self._annotation_for(item, label_id)
            prediction = self._prediction_for(item) if include_predictions else None
            samples.append({
                "id": item["id"],
                "original_name": item.get("original_name"),
                "image_path": item.get("image_path"),
                "original_path": item.get("original_path"),
                "metadata_path": item.get("metadata_path"),
                "status": item.get("status"),
                "source_format": item.get("source_format"),
                "bit_depth": item.get("bit_depth"),
                "dtype": item.get("dtype"),
                "sha256": item.get("sha256"),
                "label_id": label_id,
                "annotation": annotation,
                "prediction": prediction,
            })
        manifest = {
            "type": "metalwear.active_learning.dataset",
            "schema_version": "1.0",
            "created_at": _now(),
            "dataset": {
                "id": meta.get("id"),
                "name": meta.get("name"),
                "schema_version": meta.get("schema_version"),
                "labels": meta.get("labels", []),
            },
            "summary": self.datasets.summary(dataset_id),
            "samples": samples,
        }
        manifest_path = root / "manifest.json"
        write_json(manifest_path, manifest)
        return {"format": "active_learning", "path": str(root), "manifest": str(manifest_path), "n_samples": len(samples)}

    def export_coco(self, dataset_id: str, label_id: str) -> dict:
        meta = self.datasets.get_meta(dataset_id)
        root = self._export_root(dataset_id, "coco")
        images_dir = root / "images"
        masks_dir = root / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        images = []
        annotations = []
        ann_id = 1
        image_id = 1
        for item in meta.get("items", []):
            annotation = self._annotation_for(item, label_id)
            if not annotation or not annotation.get("path"):
                continue
            image_src = Path(item["image_path"])
            mask_src = Path(annotation["path"])
            if not image_src.exists() or not mask_src.exists():
                continue
            image_dst = images_dir / f"{item['id']}.png"
            mask_dst = masks_dir / f"{item['id']}_{label_id}.png"
            shutil.copy2(image_src, image_dst)
            shutil.copy2(mask_src, mask_dst)
            width, height = int(item.get("width", 0)), int(item.get("height", 0))
            images.append({
                "id": image_id,
                "file_name": str(image_dst.relative_to(root)).replace("\\", "/"),
                "width": width,
                "height": height,
                "item_id": item["id"],
            })

            mask = np.array(Image.open(mask_src).convert("L")) > 0
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour) < 3:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                polygon = contour.reshape(-1, 2).astype(float).ravel().tolist()
                if len(polygon) < 6:
                    continue
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": [polygon],
                    "area": float(cv2.contourArea(contour)),
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "iscrowd": 0,
                    "mask_file": str(mask_dst.relative_to(root)).replace("\\", "/"),
                })
                ann_id += 1
            image_id += 1

        coco = {
            "info": {
                "description": "Metalwear active learning export",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": _now(),
            },
            "licenses": [],
            "categories": [{"id": 1, "name": label_id, "supercategory": "defect"}],
            "images": images,
            "annotations": annotations,
        }
        annotations_path = root / "annotations.json"
        write_json(annotations_path, coco)
        return {
            "format": "coco",
            "path": str(root),
            "annotations": str(annotations_path),
            "n_images": len(images),
            "n_annotations": len(annotations),
        }

    def export_fiftyone(self, dataset_id: str, label_id: str, include_predictions: bool) -> dict:
        active = self.export_active_learning(dataset_id, label_id, include_predictions)
        root = self._export_root(dataset_id, "fiftyone")
        shutil.copy2(active["manifest"], root / "manifest.json")
        loader_path = root / "load_fiftyone.py"
        script = f'''from __future__ import annotations

import json
from pathlib import Path

import fiftyone as fo

manifest_path = Path(__file__).resolve().parent / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
dataset_name = "metalwear_" + manifest["dataset"]["id"].replace("-", "_")

if fo.dataset_exists(dataset_name):
    fo.delete_dataset(dataset_name)

dataset = fo.Dataset(dataset_name)
dataset.persistent = True

for row in manifest["samples"]:
    sample = fo.Sample(filepath=row["image_path"])
    sample["item_id"] = row["id"]
    sample["status"] = row.get("status")
    sample["source_format"] = row.get("source_format")
    sample["bit_depth"] = row.get("bit_depth")
    sample["sha256"] = row.get("sha256")
    if row.get("annotation") and row["annotation"].get("path"):
        sample["review_{label_id}"] = fo.Segmentation(mask_path=row["annotation"]["path"])
    if row.get("prediction") and row["prediction"].get("mask_path"):
        sample["prediction_{label_id}"] = fo.Segmentation(mask_path=row["prediction"]["mask_path"])
    dataset.add_sample(sample)

print(dataset)
session = fo.launch_app(dataset)
session.wait()
'''
        loader_path.write_text(script, encoding="utf-8")
        (root / "README.md").write_text(
            "Run with a Python environment that has FiftyOne installed:\n\n"
            f"python {loader_path.name}\n",
            encoding="utf-8",
        )
        return {
            "format": "fiftyone",
            "path": str(root),
            "manifest": str(root / "manifest.json"),
            "n_samples": active["n_samples"],
            "loader": str(loader_path),
        }
