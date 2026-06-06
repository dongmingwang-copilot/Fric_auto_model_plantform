from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.core.config import Settings
from app.ml.image_ops import load_uint8_gray, refine_mask, save_binary_mask, save_overlay, save_probability, tile_positions
from app.ml.unet import UNet
from app.services.checkpoints import CheckpointService
from app.services.datasets import DatasetService


class InferenceService:
    def __init__(self, settings: Settings, checkpoints: CheckpointService, datasets: DatasetService):
        self.settings = settings
        self.checkpoints = checkpoints
        self.datasets = datasets
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_cache: dict[str, torch.nn.Module] = {}

    def _load_model(self, checkpoint_id: str) -> torch.nn.Module:
        if checkpoint_id in self._model_cache:
            return self._model_cache[checkpoint_id]
        record = self.checkpoints.get(checkpoint_id)
        checkpoint = torch.load(record["path"], map_location=self.device, weights_only=False)
        args = checkpoint.get("args", {})
        model = UNet(base=int(args.get("base_ch", 32)), dropout=float(args.get("dropout", 0.1))).to(self.device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        self._model_cache[checkpoint_id] = model
        return model

    @torch.no_grad()
    def _predict_tiled(self, model: torch.nn.Module, image: Image.Image, tile: int, stride: int, batch_size: int = 4) -> np.ndarray:
        arr = np.array(image, dtype=np.float32) / 255.0
        h, w = arr.shape
        pad_h = max(h, tile)
        pad_w = max(w, tile)
        padded = np.zeros((pad_h, pad_w), dtype=np.float32)
        padded[:h, :w] = arr
        coords = [(y, x) for y in tile_positions(pad_h, tile, stride) for x in tile_positions(pad_w, tile, stride)]
        prob_sum = np.zeros((pad_h, pad_w), dtype=np.float32)
        weight = np.zeros((pad_h, pad_w), dtype=np.float32)
        for start in range(0, len(coords), batch_size):
            chunk = coords[start:start + batch_size]
            tiles = np.stack([padded[y:y + tile, x:x + tile] for y, x in chunk], axis=0)
            x_tensor = torch.from_numpy(tiles[:, None]).float().to(self.device)
            probs = torch.sigmoid(model(x_tensor))[:, 0].detach().cpu().numpy()
            for (y0, x0), prob in zip(chunk, probs):
                prob_sum[y0:y0 + tile, x0:x0 + tile] += prob
                weight[y0:y0 + tile, x0:x0 + tile] += 1.0
        return (prob_sum / np.maximum(weight, 1e-6))[:h, :w]

    def predict_probability(self, checkpoint_id: str, image: Image.Image, tile: int, stride: int) -> np.ndarray:
        model = self._load_model(checkpoint_id)
        return self._predict_tiled(model, image, tile=tile, stride=stride)

    def predict(self, dataset_id: str, item_id: str, checkpoint_id: str, threshold: float, tile: int, stride: int) -> dict:
        item = self.datasets.get_item(dataset_id, item_id)
        image = load_uint8_gray(Path(item["original_path"]))
        prob = self.predict_probability(checkpoint_id, image, tile, stride)
        mask = refine_mask(prob >= threshold)

        pred_root = self.datasets.dataset_root(dataset_id) / "predictions" / checkpoint_id
        prob_path = pred_root / "probability" / f"{item_id}.npy"
        mask_path = pred_root / "masks" / f"{item_id}.png"
        overlay_path = pred_root / "overlays" / f"{item_id}.png"
        synced_mask_path = self.datasets.dataset_root(dataset_id) / "masks" / "predicted" / checkpoint_id / f"{item_id}.png"
        save_probability(prob, prob_path)
        save_binary_mask(mask, mask_path)
        save_binary_mask(mask, synced_mask_path)
        save_overlay(image, mask, overlay_path)

        record = {
            "checkpoint_id": checkpoint_id,
            "threshold": threshold,
            "tile": tile,
            "stride": stride,
            "probability_path": str(prob_path),
            "mask_path": str(mask_path),
            "synced_mask_path": str(synced_mask_path),
            "overlay_path": str(overlay_path),
            "pred_px": int(mask.sum()),
        }
        predictions = dict(item.get("predictions", {}))
        predictions[checkpoint_id] = record
        reviewed = bool(item.get("annotation_path") or item.get("annotations"))
        self.datasets.update_item(dataset_id, item_id, {
            "status": "reviewed" if reviewed else "predicted",
            "predictions": predictions,
            "latest_prediction": checkpoint_id,
        })
        self.datasets.record_event(dataset_id, "prediction_mask_sync", {
            "item_id": item_id,
            "checkpoint_id": checkpoint_id,
            "mask_path": str(mask_path),
            "synced_mask_path": str(synced_mask_path),
            "pred_px": int(mask.sum()),
        })
        return record
