from __future__ import annotations

from pathlib import Path

from app.ml.image_ops import decode_mask_data_url, rgba_to_binary_mask, save_binary_mask
from app.services.datasets import DatasetService


class AnnotationService:
    def __init__(self, datasets: DatasetService):
        self.datasets = datasets

    def save_review_mask(self, dataset_id: str, item_id: str, mask_png_base64: str, reviewer: str, source: str, label_id: str = "spall") -> dict:
        item = self.datasets.get_item(dataset_id, item_id)
        mask_img = decode_mask_data_url(mask_png_base64)
        mask = rgba_to_binary_mask(mask_img)
        root = self.datasets.dataset_root(dataset_id)
        out_path = root / "annotations" / label_id / f"{item_id}_review_mask.png"
        synced_path = root / "masks" / "reviewed" / label_id / f"{item_id}.png"
        save_binary_mask(mask, out_path)
        save_binary_mask(mask, synced_path)
        annotations = dict(item.get("annotations", {}))
        annotations[label_id] = {
            "path": str(out_path),
            "synced_mask_path": str(synced_path),
            "reviewer": reviewer,
            "source": source,
            "mask_px": int(mask.sum()),
            "based_on_prediction": item.get("latest_prediction"),
        }
        patch = {
            "status": "reviewed",
            "annotations": annotations,
        }
        if label_id == "spall":
            patch["annotation_path"] = str(out_path)
            patch["annotation"] = annotations[label_id]
        saved = self.datasets.update_item(dataset_id, item_id, {
            **patch,
        })
        self.datasets.record_event(dataset_id, "review_mask_sync", {
            "item_id": item_id,
            "mask_path": str(out_path),
            "synced_mask_path": str(synced_path),
            "mask_px": int(mask.sum()),
            "reviewer": reviewer,
            "source": source,
        }, label_id=label_id)
        return saved
