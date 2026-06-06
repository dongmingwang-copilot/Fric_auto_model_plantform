from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from app.core.config import Settings
from app.core.json_store import read_json, write_json
from app.ml.unet import UNet


def _display_label(label_id: str | None) -> str | None:
    if not label_id:
        return None
    return " ".join(part.capitalize() for part in str(label_id).replace("_", "-").split("-") if part)


def _normalize_record(record: dict) -> dict:
    current = dict(record)
    if current.get("id") == CheckpointService.BASELINE_ID:
        current.setdefault("label_id", "spall")
        current.setdefault("label_name", "Spall")
        current.setdefault("label_color", "#ff3728")
        current.setdefault("project_type", "optimization")
        current.setdefault("model_stage", "curated_baseline")
        current["name"] = "Spall 基线模型 | recall-v1"
        current["description"] = "当前主基线，作为模型升级工作台的默认起点。"
    if current.get("id") == CheckpointService.SCRATCH_ID:
        current["name"] = "UNet-32 Scratch 初始权重"
        current["description"] = "新缺陷类别小样本生成的初始权重。"
    return current


@dataclass(frozen=True)
class CheckpointRecord:
    id: str
    name: str
    role: str
    model_type: str
    path: str
    threshold_default: float
    created_at: str
    readonly: bool


class CheckpointService:
    BASELINE_ID = "baseline-spall-unet-recall-v1"
    BASELINE_FILE = "spall_unet_recall_baseline.pt"
    SCRATCH_ID = "scratch-unet32"
    SCRATCH_FILE = "unet32_scratch.pt"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.registry_path = settings.checkpoints_dir / "registry.json"
        self._bootstrap_baseline()
        self._bootstrap_scratch()

    def _bootstrap_baseline(self) -> None:
        records = read_json(self.registry_path, [])
        if any(row.get("id") == self.BASELINE_ID for row in records):
            return
        baseline_path = self.settings.baseline_dir / self.BASELINE_FILE
        if not baseline_path.exists():
            return
        record = CheckpointRecord(
            id=self.BASELINE_ID,
            name="Spall 高召回基线模型",
            role="baseline",
            model_type="unet",
            path=str(baseline_path),
            threshold_default=self.settings.default_threshold,
            created_at=datetime.now(timezone.utc).isoformat(),
            readonly=True,
        )
        records.append(asdict(record))
        write_json(self.registry_path, records)

    def _bootstrap_scratch(self) -> None:
        records = read_json(self.registry_path, [])
        scratch_path = self.settings.baseline_dir / self.SCRATCH_FILE
        if not scratch_path.exists():
            model = UNet(base=32, dropout=0.1)
            torch.save({
                "model": model.state_dict(),
                "args": {
                    "base_ch": 32,
                    "dropout": 0.1,
                    "source": "Plantform_v1_scratch_unet32",
                },
                "val": {},
            }, scratch_path)
        if any(row.get("id") == self.SCRATCH_ID for row in records):
            return
        records.append(asdict(CheckpointRecord(
            id=self.SCRATCH_ID,
            name="UNet-32 Scratch 初始权重",
            role="foundation",
            model_type="unet",
            path=str(scratch_path),
            threshold_default=self.settings.default_threshold,
            created_at=datetime.now(timezone.utc).isoformat(),
            readonly=True,
        )))
        write_json(self.registry_path, records)

    def _resolve_checkpoint_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.settings.root / path

    def _with_resolved_path(self, record: dict) -> dict:
        current = dict(record)
        if current.get("id") == self.BASELINE_ID:
            current["path"] = str(self.settings.baseline_dir / self.BASELINE_FILE)
        elif current.get("id") == self.SCRATCH_ID:
            current["path"] = str(self.settings.baseline_dir / self.SCRATCH_FILE)
        elif current.get("path"):
            current["path"] = str(self._resolve_checkpoint_path(current["path"]))
        return current

    def list(self) -> list[dict]:
        records = [self._with_resolved_path(_normalize_record(row)) for row in read_json(self.registry_path, [])]
        known = {row["path"] for row in records if "path" in row}
        for path in sorted(self.settings.run_checkpoints_dir.glob("**/*.pt")):
            if str(path) in known:
                continue
            job = read_json(self.settings.training_jobs_dir / f"{path.parent.name}.json", {})
            if job.get("status") != "completed":
                continue
            label_id = job.get("label_id")
            project_type = job.get("project_type")
            dataset_id = job.get("dataset_id")
            if project_type == "generation" and dataset_id and not (self.settings.datasets_dir / dataset_id / "dataset.json").exists():
                continue
            label_name = (job.get("label") or {}).get("name") or _display_label(label_id)
            if project_type == "generation":
                if job.get("generation_complete"):
                    name = f"{label_name or 'Unknown'} 基线模型 | {path.parent.name}"
                    model_stage = "generated_baseline"
                else:
                    name = f"{label_name or 'Unknown'} 生成训练模型 | {path.parent.name}"
                    model_stage = "generation_run"
            else:
                name = f"{label_name or 'Unknown'} 优化模型 | {path.parent.name}"
                model_stage = "optimization_run"
            record = asdict(CheckpointRecord(
                id=f"run-{path.parent.name}-{path.stem}",
                name=name,
                role="training_run",
                model_type="unet",
                path=str(path),
                threshold_default=float((job.get("metrics") or {}).get("best_threshold") or self.settings.default_threshold),
                created_at=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                readonly=False,
            ))
            if label_id:
                record["label_id"] = label_id
            if label_name:
                record["label_name"] = label_name
            if (job.get("label") or {}).get("color"):
                record["label_color"] = job["label"]["color"]
            if project_type:
                record["project_type"] = project_type
            record["model_stage"] = model_stage
            if dataset_id:
                record["dataset_id"] = dataset_id
            if job.get("base_checkpoint_id"):
                record["base_checkpoint_id"] = job.get("base_checkpoint_id")
            if (job.get("metrics") or {}).get("best_threshold_metrics"):
                record["best_threshold_metrics"] = job["metrics"]["best_threshold_metrics"]
            records.append(record)
        return records

    def get(self, checkpoint_id: str) -> dict:
        for record in self.list():
            if record["id"] == checkpoint_id:
                path = Path(record["path"])
                if not path.exists():
                    raise FileNotFoundError(f"Checkpoint missing: {path}")
                return record
        raise KeyError(f"Unknown checkpoint: {checkpoint_id}")
