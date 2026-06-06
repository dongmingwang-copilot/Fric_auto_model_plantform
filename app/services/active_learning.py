from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.ml.image_ops import load_uint8_gray
from app.services.datasets import DatasetService
from app.services.inference import InferenceService


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ActiveLearningService:
    def __init__(self, datasets: DatasetService, inference: InferenceService):
        self.datasets = datasets
        self.inference = inference

    def batch_predict(
        self,
        dataset_id: str,
        checkpoint_id: str,
        threshold: float,
        tile: int,
        stride: int,
        limit: int = 0,
        only_unreviewed: bool = True,
        force: bool = False,
    ) -> dict:
        items = self.datasets.list_items(dataset_id)
        done = []
        skipped = 0
        for item in items:
            if only_unreviewed and (item.get("annotation_path") or item.get("annotations")):
                skipped += 1
                continue
            if item.get("predictions", {}).get(checkpoint_id) and not force:
                skipped += 1
                continue
            pred = self.inference.predict(dataset_id, item["id"], checkpoint_id, threshold, tile, stride)
            done.append({"item_id": item["id"], "pred_px": pred["pred_px"]})
            if limit and len(done) >= limit:
                break
        return {"predicted": len(done), "skipped": skipped, "force": force, "items": done}

    def rank(
        self,
        dataset_id: str,
        checkpoint_id: str,
        label_id: str,
        threshold: float,
        tile: int,
        stride: int,
        top_k: int,
        predict_missing: bool = False,
        create_batch: bool = True,
    ) -> dict:
        meta = self.datasets.get_meta(dataset_id)
        items = meta.get("items", [])
        reviewed_hists = self._reviewed_histograms(items, label_id)
        reference_hist = np.mean(reviewed_hists, axis=0) if reviewed_hists else None
        rows = []
        updated = 0

        for item in items:
            if item.get("annotations", {}).get(label_id) or (label_id == "spall" and item.get("annotation_path")):
                continue
            pred = item.get("predictions", {}).get(checkpoint_id)
            if not pred and predict_missing:
                pred = self.inference.predict(dataset_id, item["id"], checkpoint_id, threshold, tile, stride)
                item = self.datasets.get_item(dataset_id, item["id"])
            if not pred:
                continue
            prob_path = Path(pred["probability_path"])
            if not prob_path.exists():
                continue
            prob = np.load(prob_path).astype(np.float32)
            score = self._score_item(item, prob, threshold, reference_hist)
            score["item_id"] = item["id"]
            score["original_name"] = item.get("original_name")
            score["status"] = item.get("status")
            score["checkpoint_id"] = checkpoint_id
            score["label_id"] = label_id
            rows.append(score)
            active = dict(item.get("active_learning", {}))
            active[checkpoint_id] = score
            self.datasets.update_item(dataset_id, item["id"], {"active_learning": active})
            updated += 1

        rows.sort(key=lambda r: r["score"], reverse=True)
        selected = rows[:top_k]
        meta = self.datasets.get_meta(dataset_id)
        batch = self._create_batch(meta, dataset_id, checkpoint_id, label_id, threshold, selected) if create_batch and selected else None
        return {
            "dataset_id": dataset_id,
            "checkpoint_id": checkpoint_id,
            "label_id": label_id,
            "updated": updated,
            "ranked": len(rows),
            "top_k": top_k,
            "batch": batch,
            "items": selected,
        }

    def list_batches(self, dataset_id: str) -> list[dict]:
        meta = self.datasets.get_meta(dataset_id)
        batches = meta.get("active_learning_batches", [])
        return [self._refresh_batch_status(batch, meta) for batch in reversed(batches)]

    def create_initial_review_queue(self, dataset_id: str, label_id: str, top_k: int) -> dict:
        meta = self.datasets.get_meta(dataset_id)
        rows = []
        for item in meta.get("items", []):
            if item.get("annotations", {}).get(label_id):
                continue
            rows.append({
                "score": 1.0,
                "uncertain_ratio": 0.0,
                "entropy_mean": 0.0,
                "margin_uncertainty": 0.0,
                "pred_ratio": 0.0,
                "pred_px": 0,
                "components": 0,
                "component_score": 0.0,
                "diversity": 0.0,
                "item_id": item["id"],
                "original_name": item.get("original_name"),
                "checkpoint_id": "manual-initial-review",
                "label_id": label_id,
            })
            if len(rows) >= top_k:
                break
        batch = self._create_batch(meta, dataset_id, "manual-initial-review", label_id, 0.5, rows, strategy="initial_manual_review_v1") if rows else None
        return {
            "dataset_id": dataset_id,
            "checkpoint_id": "manual-initial-review",
            "label_id": label_id,
            "updated": 0,
            "ranked": len(rows),
            "top_k": top_k,
            "batch": batch,
            "items": rows,
        }

    def get_batch(self, dataset_id: str, batch_id: str) -> dict:
        meta = self.datasets.get_meta(dataset_id)
        for batch in meta.get("active_learning_batches", []):
            if batch.get("id") == batch_id:
                return self._refresh_batch_status(batch, meta)
        raise KeyError(f"Unknown active learning batch: {batch_id}")

    def mark_item_reviewed(self, dataset_id: str, batch_id: str, item_id: str) -> dict:
        meta = self.datasets.get_meta(dataset_id)
        for batch in meta.get("active_learning_batches", []):
            if batch.get("id") != batch_id:
                continue
            for row in batch.get("items", []):
                if row.get("item_id") == item_id:
                    row["status"] = "reviewed"
                    row["reviewed_at"] = _now()
            batch["updated_at"] = _now()
            self.datasets.save_meta(dataset_id, meta)
            return self._refresh_batch_status(batch, meta)
        raise KeyError(f"Unknown active learning batch: {batch_id}")

    def _create_batch(self, meta: dict, dataset_id: str, checkpoint_id: str, label_id: str, threshold: float, rows: list[dict], strategy: str = "uncertainty_entropy_area_components_diversity_v1") -> dict:
        batch_id = f"al-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        batch = {
            "id": batch_id,
            "created_at": _now(),
            "updated_at": _now(),
            "strategy": strategy,
            "checkpoint_id": checkpoint_id,
            "label_id": label_id,
            "threshold": threshold,
            "n_items": len(rows),
            "n_reviewed": 0,
            "n_pending": len(rows),
            "items": [
                {
                    **row,
                    "rank": idx + 1,
                    "status": "pending",
                }
                for idx, row in enumerate(rows)
            ],
        }
        meta.setdefault("active_learning_batches", []).append(batch)
        self.datasets.save_meta(dataset_id, meta)
        return batch

    def _refresh_batch_status(self, batch: dict, meta: dict) -> dict:
        items_by_id = {item["id"]: item for item in meta.get("items", [])}
        reviewed = 0
        pending = 0
        refreshed_items = []
        for row in batch.get("items", []):
            item = items_by_id.get(row.get("item_id"), {})
            label_id = batch.get("label_id", "spall")
            is_reviewed = bool(item.get("annotations", {}).get(label_id) or (label_id == "spall" and item.get("annotation_path")))
            current = dict(row)
            current["status"] = "reviewed" if is_reviewed else current.get("status", "pending")
            if current["status"] == "reviewed":
                reviewed += 1
            else:
                pending += 1
            refreshed_items.append(current)
        out = dict(batch)
        out["items"] = refreshed_items
        out["n_reviewed"] = reviewed
        out["n_pending"] = pending
        out["n_items"] = len(refreshed_items)
        return out

    def _reviewed_histograms(self, items: list[dict], label_id: str) -> list[np.ndarray]:
        hists = []
        for item in items:
            has_annotation = item.get("annotations", {}).get(label_id) or (label_id == "spall" and item.get("annotation_path"))
            if not has_annotation:
                continue
            try:
                hists.append(self._image_histogram(Path(item["image_path"])))
            except Exception:
                continue
        return hists

    def _image_histogram(self, image_path: Path) -> np.ndarray:
        image = load_uint8_gray(image_path)
        arr = np.array(image, dtype=np.uint8)
        hist, _ = np.histogram(arr, bins=32, range=(0, 255), density=False)
        hist = hist.astype(np.float32)
        return hist / max(float(hist.sum()), 1.0)

    def _score_item(self, item: dict, prob: np.ndarray, threshold: float, reference_hist: np.ndarray | None) -> dict:
        eps = 1e-6
        p = np.clip(prob, eps, 1.0 - eps)
        entropy = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
        entropy_mean = float(entropy.mean())
        uncertain_ratio = float(((p >= 0.35) & (p <= 0.65)).mean())
        margin_uncertainty = float((1.0 - np.abs(p - 0.5) * 2.0).mean())
        pred = p >= threshold
        pred_ratio = float(pred.mean())
        n_components = 0
        if pred.any():
            n_components = int(cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)[0] - 1)
        component_score = min(np.log1p(max(n_components, 0)) / np.log(64), 1.0)
        diversity = 0.0
        if reference_hist is not None:
            hist = self._image_histogram(Path(item["image_path"]))
            diversity = float(0.5 * np.abs(hist - reference_hist).sum())
        score = (
            0.34 * uncertain_ratio
            + 0.24 * entropy_mean
            + 0.16 * margin_uncertainty
            + 0.10 * min(pred_ratio / 0.12, 1.0)
            + 0.08 * component_score
            + 0.08 * min(diversity, 1.0)
        )
        return {
            "score": float(score),
            "uncertain_ratio": uncertain_ratio,
            "entropy_mean": entropy_mean,
            "margin_uncertainty": margin_uncertainty,
            "pred_ratio": pred_ratio,
            "pred_px": int(pred.sum()),
            "components": n_components,
            "component_score": float(component_score),
            "diversity": diversity,
        }
