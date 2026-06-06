from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from app.ml.image_ops import load_uint8_gray
from app.ml.unet import UNet


@dataclass(frozen=True)
class ReviewedSample:
    item_id: str
    image_path: Path
    mask_path: Path
    width: int
    height: int
    group: str
    has_spall: bool
    spall_pixels: int


class ReviewCropDataset(Dataset):
    def __init__(self, samples: list[ReviewedSample], length: int, crop_size: int = 512, augment: bool = True):
        self.samples = samples
        self.length = length
        self.crop_size = crop_size
        self.augment = augment
        self.cache = [self._load_sample(sample) for sample in samples]
        self.positive = [row for row in self.cache if row["sample"].has_spall]

    def _load_sample(self, sample: ReviewedSample) -> dict:
        image = load_uint8_gray(sample.image_path)
        mask = Image.open(sample.mask_path).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        return {
            "sample": sample,
            "image": np.array(image, dtype=np.float32) / 255.0,
            "mask": (np.array(mask) > 0).astype(np.float32),
        }

    def __len__(self) -> int:
        return self.length

    def _choose_sample(self) -> dict:
        if self.positive and random.random() < 0.75:
            return random.choice(self.positive)
        return random.choice(self.cache)

    def __getitem__(self, index: int):
        row = self._choose_sample()
        image_arr = row["image"]
        mask_arr = row["mask"]
        h, w = image_arr.shape
        crop = self.crop_size
        pad_h = max(0, crop - h)
        pad_w = max(0, crop - w)
        if pad_h or pad_w:
            image_arr = np.pad(image_arr, ((0, pad_h), (0, pad_w)), mode="constant")
            mask_arr = np.pad(mask_arr, ((0, pad_h), (0, pad_w)), mode="constant")
            h, w = image_arr.shape

        if mask_arr.any() and random.random() < 0.65:
            ys, xs = np.where(mask_arr > 0)
            center_i = random.randrange(len(xs))
            cx, cy = int(xs[center_i]), int(ys[center_i])
            x0 = min(max(cx - random.randint(crop // 4, crop * 3 // 4), 0), max(w - crop, 0))
            y0 = min(max(cy - random.randint(crop // 4, crop * 3 // 4), 0), max(h - crop, 0))
        else:
            x0 = random.randint(0, max(w - crop, 0))
            y0 = random.randint(0, max(h - crop, 0))

        image_arr = image_arr[y0:y0 + crop, x0:x0 + crop]
        mask_arr = mask_arr[y0:y0 + crop, x0:x0 + crop]

        if self.augment:
            if random.random() < 0.5:
                image_arr = np.fliplr(image_arr).copy()
                mask_arr = np.fliplr(mask_arr).copy()
            if random.random() < 0.5:
                image_arr = np.flipud(image_arr).copy()
                mask_arr = np.flipud(mask_arr).copy()
            k = random.randint(0, 3)
            if k:
                image_arr = np.rot90(image_arr, k).copy()
                mask_arr = np.rot90(mask_arr, k).copy()
            image_arr = np.clip(image_arr * random.uniform(0.90, 1.12) + random.uniform(-0.04, 0.04), 0, 1)
            if random.random() < 0.25:
                image_arr = np.clip(image_arr + np.random.normal(0, random.uniform(0.004, 0.018), image_arr.shape), 0, 1)

        return (
            torch.from_numpy(np.ascontiguousarray(image_arr[None])).float(),
            torch.from_numpy(np.ascontiguousarray(mask_arr[None])).float(),
        )


def _gpu_progress(device: torch.device) -> dict:
    if device.type != "cuda":
        return {}
    return {
        "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated(device) / (1024 * 1024), 1),
        "gpu_memory_reserved_mb": round(torch.cuda.memory_reserved(device) / (1024 * 1024), 1),
        "gpu_max_memory_reserved_mb": round(torch.cuda.max_memory_reserved(device) / (1024 * 1024), 1),
    }


def recall_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.3, beta: float = 0.7, pos_weight: float = 1.0) -> torch.Tensor:
    weight = torch.tensor(pos_weight, device=logits.device) if pos_weight and pos_weight != 1.0 else None
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
    prob = torch.sigmoid(logits)
    tp = (prob * target).sum(dim=(1, 2, 3))
    fp = (prob * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - prob) * target).sum(dim=(1, 2, 3))
    tversky = (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)
    return bce + (1 - tversky.mean())


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> dict:
    model.eval()
    tp = fp = fn = tn = 0
    loss_sum = 0.0
    batches = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss_sum += float(recall_loss(logits, y).item())
        pred = torch.sigmoid(logits) >= threshold
        truth = y >= 0.5
        tp += int((pred & truth).sum())
        fp += int((pred & ~truth).sum())
        fn += int((~pred & truth).sum())
        tn += int((~pred & ~truth).sum())
        batches += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    dice = (2 * tp) / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    beta2 = 4.0
    f2 = ((1 + beta2) * precision * recall) / max(beta2 * precision + recall, 1e-12)
    return {
        "loss": loss_sum / max(batches, 1),
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
        "specificity": specificity,
        "f2": f2,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def collect_reviewed_samples(items: list[dict], label_id: str = "spall") -> list[ReviewedSample]:
    samples: list[ReviewedSample] = []
    for item in items:
        annotation = item.get("annotations", {}).get(label_id)
        annotation_path = annotation.get("path") if annotation else item.get("annotation_path") if label_id == "spall" else None
        if not annotation_path:
            continue
        mask_path = Path(annotation_path)
        image_path = Path(item["original_path"])
        if not image_path.exists() or not mask_path.exists():
            continue
        mask = Image.open(mask_path).convert("L")
        mask_arr = np.array(mask) > 0
        spall_pixels = int(mask_arr.sum())
        samples.append(ReviewedSample(
            item_id=item["id"],
            image_path=image_path,
            mask_path=mask_path,
            width=int(item["width"]),
            height=int(item["height"]),
            group=sample_group(item.get("original_name") or item["id"]),
            has_spall=spall_pixels > 0,
            spall_pixels=spall_pixels,
        ))
    return samples


def sample_group(name: str) -> str:
    stem = Path(str(name)).stem
    parts = stem.split("__")
    if len(parts) >= 3:
        return parts[1]
    prefix = stem.rsplit("-", 1)
    if len(prefix) == 2 and prefix[1].isdigit():
        return prefix[0]
    if len(prefix) == 2 and len(prefix[1]) == 8:
        return prefix[0]
    return stem


def _split_stats(samples: list[ReviewedSample]) -> dict:
    return {
        "count": len(samples),
        "positive": sum(sample.has_spall for sample in samples),
        "negative": sum(not sample.has_spall for sample in samples),
        "spall_pixels": sum(sample.spall_pixels for sample in samples),
        "groups": sorted({sample.group for sample in samples}),
    }


def tile_positions(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


@torch.no_grad()
def predict_tiled(model: torch.nn.Module, image: Image.Image, device: torch.device, tile: int = 512, stride: int = 384) -> np.ndarray:
    arr = np.array(image, dtype=np.float32) / 255.0
    h, w = arr.shape
    pad_h = max(h, tile)
    pad_w = max(w, tile)
    padded = np.zeros((pad_h, pad_w), dtype=np.float32)
    padded[:h, :w] = arr
    prob_sum = np.zeros((pad_h, pad_w), dtype=np.float32)
    weight = np.zeros((pad_h, pad_w), dtype=np.float32)
    coords = [(y, x) for y in tile_positions(pad_h, tile, stride) for x in tile_positions(pad_w, tile, stride)]
    for start in range(0, len(coords), 4):
        chunk = coords[start:start + 4]
        tiles = np.stack([padded[y:y + tile, x:x + tile] for y, x in chunk], axis=0)
        x_tensor = torch.from_numpy(tiles[:, None]).float().to(device)
        probs = torch.sigmoid(model(x_tensor))[:, 0].detach().cpu().numpy()
        for (y0, x0), prob in zip(chunk, probs):
            prob_sum[y0:y0 + tile, x0:x0 + tile] += prob
            weight[y0:y0 + tile, x0:x0 + tile] += 1.0
    return (prob_sum / np.maximum(weight, 1e-6))[:h, :w]


def overlay_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int] = (255, 55, 40), alpha: float = 0.34) -> Image.Image:
    base = np.array(image.convert("RGB"), dtype=np.uint8)
    out = base.astype(np.float32)
    hit = mask.astype(bool)
    out[hit] = out[hit] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha
    return Image.fromarray(out.clip(0, 255).astype(np.uint8), mode="RGB")


def save_test_visualizations(model: torch.nn.Module, samples: list[ReviewedSample], output_dir: Path, device: torch.device, threshold: float = 0.5, limit: int = 24) -> list[dict]:
    viz_dir = output_dir / "test_visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    model.eval()
    for idx, sample in enumerate(samples[:limit], start=1):
        image = load_uint8_gray(sample.image_path)
        gt_img = Image.open(sample.mask_path).convert("L")
        if gt_img.size != image.size:
            gt_img = gt_img.resize(image.size, Image.Resampling.NEAREST)
        gt = np.array(gt_img) > 0
        prob = predict_tiled(model, image, device)
        pred = prob >= threshold
        stem = f"{idx:03d}_{sample.item_id}"
        original_path = viz_dir / f"{stem}_original.png"
        gt_path = viz_dir / f"{stem}_gt.png"
        pred_path = viz_dir / f"{stem}_pred.png"
        image.convert("RGB").save(original_path)
        overlay_mask(image, gt).save(gt_path)
        overlay_mask(image, pred).save(pred_path)
        tp = int((pred & gt).sum())
        fp = int((pred & ~gt).sum())
        fn = int((~pred & gt).sum())
        tn = int((~pred & ~gt).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        dice = (2 * tp) / max(2 * tp + fp + fn, 1)
        iou = tp / max(tp + fp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        rows.append({
            "item_id": sample.item_id,
            "original": original_path.name,
            "gt": gt_path.name,
            "pred": pred_path.name,
            "precision": precision,
            "recall": recall,
            "dice": dice,
            "iou": iou,
            "specificity": specificity,
        })
    (viz_dir / "manifest.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def split_samples(samples: list[ReviewedSample], seed: int) -> tuple[list[ReviewedSample], list[ReviewedSample], dict]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    if len(shuffled) < 2:
        return shuffled, shuffled, {"strategy": "single_sample_fallback"}
    val_n = max(1, round(len(shuffled) * 0.2))
    return shuffled[val_n:], shuffled[:val_n], {
        "strategy": "random_image_split",
        "train": _split_stats(shuffled[val_n:]),
        "val": _split_stats(shuffled[:val_n]),
    }


def grouped_split_samples(samples: list[ReviewedSample], seed: int) -> tuple[list[ReviewedSample], list[ReviewedSample], dict]:
    groups: dict[str, list[ReviewedSample]] = {}
    for sample in samples:
        groups.setdefault(sample.group, []).append(sample)
    items = sorted(groups.items(), key=lambda item: -len(item[1]))
    if len(items) < 3:
        train, val, meta = split_samples(samples, seed)
        meta["fallback_reason"] = f"Need >=3 groups, have {len(items)}"
        return train, val, meta

    rng = random.Random(seed)
    n_val = max(1, round(len(items) * 0.2))
    n_train = len(items) - n_val
    totals = _split_stats(samples)

    def flatten(assignment: dict[str, list[tuple[str, list[ReviewedSample]]]], key: str) -> list[ReviewedSample]:
        return [sample for _, group_samples in assignment[key] for sample in group_samples]

    def score(assignment: dict[str, list[tuple[str, list[ReviewedSample]]]]) -> float:
        value = 0.0
        targets = {"train": 0.8, "val": 0.2}
        for split_name in ("train", "val"):
            split_samples_for_score = flatten(assignment, split_name)
            stats = _split_stats(split_samples_for_score)
            target = targets[split_name]
            value += abs(stats["count"] / max(totals["count"], 1) - target) * 4
            value += abs(stats["positive"] / max(totals["positive"], 1) - target) * 2
            value += abs(stats["negative"] / max(totals["negative"], 1) - target) * 2
            value += abs(stats["spall_pixels"] / max(totals["spall_pixels"], 1) - target) * 2
            if split_name == "val":
                if stats["positive"] == 0:
                    value += 10
                if stats["negative"] == 0:
                    value += 3
        return value

    best: dict[str, list[tuple[str, list[ReviewedSample]]]] | None = None
    best_score = float("inf")
    for _ in range(4000):
        rng.shuffle(items)
        candidate = {"train": items[:n_train], "val": items[n_train:n_train + n_val]}
        candidate_score = score(candidate)
        if candidate_score < best_score:
            best = {key: list(value) for key, value in candidate.items()}
            best_score = candidate_score

    assert best is not None
    train = sorted(flatten(best, "train"), key=lambda sample: sample.item_id)
    val = sorted(flatten(best, "val"), key=lambda sample: sample.item_id)
    return train, val, {
        "strategy": "group_aware_split",
        "score": best_score,
        "train": _split_stats(train),
        "val": _split_stats(val),
    }


def _metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    dice = (2 * tp) / max(2 * tp + fp + fn, 1)
    beta2 = 4.0
    return {
        "dice": dice,
        "iou": tp / max(tp + fp + fn, 1),
        "precision": precision,
        "recall": recall,
        "specificity": tn / max(tn + fp, 1),
        "f2": ((1 + beta2) * precision * recall) / max(beta2 * precision + recall, 1e-12),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


@torch.no_grad()
def threshold_sweep(model: torch.nn.Module, samples: list[ReviewedSample], device: torch.device, thresholds: list[float] | None = None) -> dict:
    model.eval()
    thresholds = thresholds or [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    probs: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    for sample in samples:
        image = load_uint8_gray(sample.image_path)
        mask = Image.open(sample.mask_path).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        probs.append(predict_tiled(model, image, device))
        truths.append(np.array(mask) > 0)

    results = []
    for threshold in thresholds:
        tp = fp = fn = tn = 0
        for prob, truth in zip(probs, truths):
            pred = prob >= threshold
            tp += int((pred & truth).sum())
            fp += int((pred & ~truth).sum())
            fn += int((~pred & truth).sum())
            tn += int((~pred & ~truth).sum())
        row = {"threshold": threshold, **_metrics_from_counts(tp, fp, fn, tn)}
        results.append(row)

    best = max(results, key=lambda row: (row["dice"], row["recall"], -row["fp"])) if results else {"threshold": 0.5}
    return {"sweep": results, "best": best}


def train_review_model(
    *,
    items: list[dict],
    base_checkpoint_path: Path,
    output_dir: Path,
    epochs: int,
    samples_per_epoch: int,
    batch_size: int,
    learning_rate: float,
    label_id: str = "spall",
    status_callback: Callable[[dict], None] | None = None,
    seed: int = 20260605,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_reviewed_samples(items, label_id=label_id)
    if len(samples) < 2:
        raise RuntimeError("至少需要 2 张已 Review 并保存 mask 的图像才能开始训练。")
    if not any(sample.has_spall for sample in samples):
        raise RuntimeError("至少需要 1 张包含 Spall mask 的 Review 图像才能开始训练。")

    train_samples, val_samples, split_meta = grouped_split_samples(samples, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(base_checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})
    model = UNet(base=int(args.get("base_ch", 32)), dropout=float(args.get("dropout", 0.1))).to(device)
    model.load_state_dict(checkpoint["model"])

    if status_callback:
        status_callback({
            "phase": "prepare",
            "epoch": 0,
            "epochs": epochs,
            "batch": 0,
            "batches": 0,
            "message": "Loading reviewed samples into memory",
        })
    train_dataset = ReviewCropDataset(train_samples, samples_per_epoch, augment=True)
    val_dataset = ReviewCropDataset(val_samples, max(64, min(512, len(val_samples) * 64)), augment=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_score = -1.0
    best_path = output_dir / "best.pt"
    history = []
    loss_config = {"name": "BCE + Tversky", "alpha": 0.3, "beta": 0.7, "pos_weight": 1.0}
    total_batches = len(train_loader)
    progress_every = max(1, min(10, total_batches // 10 or 1))

    if status_callback:
        status_callback({
            "phase": "ready",
            "epoch": 0,
            "epochs": epochs,
            "batch": 0,
            "batches": total_batches,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "samples_per_epoch": samples_per_epoch,
            "batch_size": batch_size,
            "device": str(device),
            **_gpu_progress(device),
        })

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum = 0.0
        batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = recall_loss(model(x), y, **{key: loss_config[key] for key in ("alpha", "beta", "pos_weight")})
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item())
            batches += 1
            if status_callback and (batches == 1 or batches == total_batches or batches % progress_every == 0):
                elapsed = max(time.perf_counter() - epoch_start, 1e-6)
                status_callback({
                    "phase": "batch",
                    "epoch": epoch,
                    "epochs": epochs,
                    "batch": batches,
                    "batches": total_batches,
                    "train_loss": loss_sum / max(batches, 1),
                    "batches_per_second": batches / elapsed,
                    "elapsed_seconds": elapsed,
                    "best_score": best_score,
                    "device": str(device),
                    **_gpu_progress(device),
                })

        if status_callback:
            status_callback({
                "phase": "evaluate",
                "epoch": epoch,
                "epochs": epochs,
                "batch": total_batches,
                "batches": total_batches,
                "train_loss": loss_sum / max(batches, 1),
                "message": "Evaluating validation split",
                "device": str(device),
                **_gpu_progress(device),
            })
        val = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(batches, 1),
            "val": val,
            "elapsed_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        score = 0.45 * val["f2"] + 0.35 * val["dice"] + 0.20 * val["recall"]
        if score > best_score:
            best_score = score
            torch.save({
                "model": model.state_dict(),
                "args": {
                    "base_ch": int(args.get("base_ch", 32)),
                    "dropout": float(args.get("dropout", 0.1)),
                    "source": "Plantform_v1_review_training",
                    "epochs": epochs,
                    "samples_per_epoch": samples_per_epoch,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "loss": loss_config,
                },
                "val": val,
                "selection_score": score,
                "selection_rule": "0.45*F2 + 0.35*Dice + 0.20*Recall",
            }, best_path)
        if status_callback:
            status_callback({
                "phase": "epoch_end",
                "epoch": epoch,
                "epochs": epochs,
                "train_loss": row["train_loss"],
                "val": val,
                "best_score": best_score,
                "elapsed_seconds": row["elapsed_seconds"],
                "device": str(device),
                **_gpu_progress(device),
            })

    metrics = {
        "device": str(device),
        "label_id": label_id,
        "training_scope": "all_reviewed_annotations",
        "n_reviewed": len(samples),
        "n_train_images": len(train_samples),
        "n_val_images": len(val_samples),
        "split": split_meta,
        "loss": loss_config,
        "best_path": str(best_path),
        "best_selection_score": best_score,
        "selection_rule": "0.45*F2 + 0.35*Dice + 0.20*Recall",
        "history": history,
    }
    if best_path.exists():
        best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model"])
    else:
        best_checkpoint = {}
    sweep = threshold_sweep(model, val_samples, device)
    best_threshold = float((sweep.get("best") or {}).get("threshold", 0.5))
    metrics["threshold_sweep"] = sweep.get("sweep", [])
    metrics["best_threshold"] = best_threshold
    metrics["best_threshold_metrics"] = sweep.get("best", {})
    if best_path.exists():
        best_checkpoint["threshold_sweep"] = sweep.get("sweep", [])
        best_checkpoint["best_threshold"] = best_threshold
        best_checkpoint["best_threshold_metrics"] = sweep.get("best", {})
        best_checkpoint["split"] = split_meta
        torch.save(best_checkpoint, best_path)
    metrics["test_visualizations"] = save_test_visualizations(model, val_samples, output_dir, device, threshold=best_threshold)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics
