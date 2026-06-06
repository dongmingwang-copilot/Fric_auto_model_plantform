from __future__ import annotations

import threading
from typing import Any
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import Settings
from app.core.json_store import read_json, write_json
from app.ml.training import train_review_model
from app.services.checkpoints import CheckpointService
from app.services.datasets import DatasetService


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrainingJobService:
    def __init__(self, settings: Settings, checkpoints: CheckpointService, datasets: DatasetService, active_learning: Any | None = None):
        self.settings = settings
        self.checkpoints = checkpoints
        self.datasets = datasets
        self.active_learning = active_learning
        self._threads: dict[str, threading.Thread] = {}
        self._recover_interrupted_jobs()

    def _recover_interrupted_jobs(self) -> None:
        for path in self.settings.training_jobs_dir.glob("*.json"):
            job = read_json(path, {})
            if job.get("status") not in {"queued", "running"}:
                continue
            job["status"] = "failed"
            job["error"] = "Training worker was interrupted before completion."
            job["progress"] = None
            job["checkpoint_available"] = False
            self._save(job)

    def list(self) -> list[dict]:
        return [
            self._normalize_job(read_json(path, {}))
            for path in sorted(self.settings.training_jobs_dir.glob("*.json"), reverse=True)
        ]

    def get(self, job_id: str) -> dict:
        job = read_json(self.settings.training_jobs_dir / f"{job_id}.json", None)
        if job is None:
            raise KeyError(f"Unknown training job: {job_id}")
        return self._normalize_job(job)

    def _normalize_job(self, job: dict) -> dict:
        if not job:
            return job
        current = dict(job)
        current["checkpoint_available"] = bool(
            current.get("checkpoint_available")
            or (
                current.get("status") == "completed"
                and Path(current.get("expected_best_model_path") or "").exists()
            )
        )
        current.setdefault("progress_history", [])
        return current

    def _save(self, job: dict) -> None:
        job["updated_at"] = _now()
        write_json(self.settings.training_jobs_dir / f"{job['id']}.json", job)

    def create(
        self,
        dataset_id: str,
        base_checkpoint_id: str,
        label_id: str,
        active_batch_id: str | None,
        epochs: int,
        samples_per_epoch: int,
        batch_size: int,
        learning_rate: float,
        note: str = "",
    ) -> dict:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = self.settings.run_checkpoints_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_meta = self.datasets.get_meta(dataset_id)
        label_meta = next((label for label in dataset_meta.get("labels", []) if label.get("id") == label_id), None)
        snapshot = self.datasets.create_snapshot(
            dataset_id,
            name=f"train-{run_id}",
            note=f"Training job {run_id}; active_batch={active_batch_id or 'none'}",
        )
        job = {
            "id": run_id,
            "status": "queued",
            "dataset_id": dataset_id,
            "dataset_name": dataset_meta.get("name"),
            "project_type": dataset_meta.get("project_type", "optimization"),
            "base_checkpoint_id": base_checkpoint_id,
            "label_id": label_id,
            "label": label_meta or {"id": label_id, "name": label_id},
            "active_batch_id": active_batch_id,
            "dataset_snapshot_id": snapshot["id"],
            "training_scope": "all_reviewed_annotations",
            "params": {
                "epochs": epochs,
                "samples_per_epoch": samples_per_epoch,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            },
            "output_dir": str(output_dir),
            "expected_best_model_path": str(output_dir / "best.pt"),
            "note": note,
            "created_at": _now(),
            "updated_at": _now(),
            "runner": "in_process_worker",
            "progress": None,
            "progress_history": [],
            "checkpoint_available": False,
            "error": None,
        }
        self._save(job)
        thread = threading.Thread(target=self._run_job, args=(run_id,), daemon=True)
        self._threads[run_id] = thread
        thread.start()
        return job

    def _run_job(self, job_id: str) -> None:
        job = self.get(job_id)
        try:
            job["status"] = "running"
            self._save(job)
            checkpoint = self.checkpoints.get(job["base_checkpoint_id"])
            items = self.datasets.list_items(job["dataset_id"])

            def on_progress(progress: dict) -> None:
                current = self.get(job_id)
                current["status"] = "running"
                current["progress"] = progress
                history = list(current.get("progress_history") or [])
                if progress.get("phase") == "epoch_end":
                    history.append(progress)
                current["progress_history"] = history[-500:]
                self._save(current)

            metrics = train_review_model(
                items=items,
                base_checkpoint_path=Path(checkpoint["path"]),
                output_dir=self.settings.run_checkpoints_dir / job_id,
                epochs=int(job["params"]["epochs"]),
                samples_per_epoch=int(job["params"]["samples_per_epoch"]),
                batch_size=int(job["params"]["batch_size"]),
                learning_rate=float(job["params"]["learning_rate"]),
                label_id=job.get("label_id", "spall"),
                status_callback=on_progress,
            )
            job = self.get(job_id)
            completion = self._dataset_completion(job["dataset_id"], job.get("label_id", "spall"))
            job["status"] = "completed"
            job["generation_complete"] = job.get("project_type") == "generation" and completion["all_reviewed"]
            job["checkpoint_available"] = True
            job["metrics"] = {
                "training_scope": metrics["training_scope"],
                "n_reviewed": metrics["n_reviewed"],
                "n_train_images": metrics["n_train_images"],
                "n_val_images": metrics["n_val_images"],
                "best_path": metrics["best_path"],
                "best_selection_score": metrics.get("best_selection_score"),
                "selection_rule": metrics.get("selection_rule"),
                "best_threshold": metrics.get("best_threshold"),
                "best_threshold_metrics": metrics.get("best_threshold_metrics"),
                "test_visualizations": metrics.get("test_visualizations", []),
                "dataset_completion": completion,
            }
            job["progress"] = None
            self._save(job)
            self._advance_active_learning_cycle(job_id)
        except Exception as exc:
            job = self.get(job_id)
            job["status"] = "failed"
            job["error"] = str(exc)
            self._save(job)

    def _dataset_completion(self, dataset_id: str, label_id: str) -> dict:
        items = self.datasets.list_items(dataset_id)
        reviewed = 0
        for item in items:
            annotation = item.get("annotations", {}).get(label_id)
            if annotation and annotation.get("path"):
                reviewed += 1
            elif label_id == "spall" and item.get("annotation_path"):
                reviewed += 1
        total = len(items)
        return {
            "label_id": label_id,
            "total": total,
            "reviewed": reviewed,
            "remaining": max(0, total - reviewed),
            "all_reviewed": total > 0 and reviewed >= total,
        }

    def _advance_active_learning_cycle(self, job_id: str) -> None:
        if self.active_learning is None:
            return
        job = self.get(job_id)
        if job.get("status") != "completed":
            return
        checkpoint_id = f"run-{job_id}-best"
        try:
            self.checkpoints.get(checkpoint_id)
        except Exception as exc:
            job["next_cycle"] = {"status": "failed", "error": f"new checkpoint not available: {exc}", "updated_at": _now()}
            self._save(job)
            return

        dataset_id = job["dataset_id"]
        label_id = job.get("label_id", "spall")
        project_type = job.get("project_type", "optimization")
        threshold = float((job.get("metrics") or {}).get("best_threshold") or self.settings.default_threshold)
        top_k = 20

        if project_type == "generation" and job.get("generation_complete"):
            prediction = self.active_learning.batch_predict(
                dataset_id=dataset_id,
                checkpoint_id=checkpoint_id,
                threshold=threshold,
                tile=self.settings.default_tile,
                stride=self.settings.default_stride,
                limit=0,
                only_unreviewed=False,
                force=True,
            )
            job["next_cycle"] = {
                "status": "baseline_promoted",
                "checkpoint_id": checkpoint_id,
                "predicted": prediction.get("predicted", 0),
                "skipped": prediction.get("skipped", 0),
                "threshold": threshold,
                "message": "Generation dataset is fully reviewed; best model is available as optimization baseline.",
                "updated_at": _now(),
            }
            self._save(job)
            self.datasets.record_event(
                dataset_id,
                "generated_baseline_promoted",
                {
                    "checkpoint_id": checkpoint_id,
                    "best_path": (job.get("metrics") or {}).get("best_path") or job.get("expected_best_model_path"),
                    "best_threshold": threshold,
                    "predicted": prediction.get("predicted", 0),
                    "job_id": job_id,
                },
                label_id=label_id,
            )
            return

        active_batch_id = job.get("active_batch_id")
        if active_batch_id:
            try:
                batch = self.active_learning.get_batch(dataset_id, active_batch_id)
                threshold = float(batch.get("threshold", threshold))
                top_k = max(1, int(batch.get("n_items", top_k)))
            except Exception:
                pass

        try:
            existing = self.active_learning.list_batches(dataset_id)
            existing_next = next((batch for batch in existing if batch.get("checkpoint_id") == checkpoint_id), None)
            if existing_next:
                job["next_cycle"] = {
                    "status": "exists",
                    "checkpoint_id": checkpoint_id,
                    "batch_id": existing_next.get("id"),
                    "predicted": 0,
                    "ranked": existing_next.get("n_items", 0),
                    "updated_at": _now(),
                }
                self._save(job)
                return

            prediction = self.active_learning.batch_predict(
                dataset_id=dataset_id,
                checkpoint_id=checkpoint_id,
                threshold=threshold,
                tile=self.settings.default_tile,
                stride=self.settings.default_stride,
                limit=0,
                only_unreviewed=False,
                force=True,
            )
            ranked = self.active_learning.rank(
                dataset_id=dataset_id,
                checkpoint_id=checkpoint_id,
                label_id=label_id,
                threshold=threshold,
                tile=self.settings.default_tile,
                stride=self.settings.default_stride,
                top_k=top_k,
                predict_missing=False,
                create_batch=True,
            )
            job["next_cycle"] = {
                "status": "created" if ranked.get("batch") else "no_candidates",
                "checkpoint_id": checkpoint_id,
                "batch_id": ranked.get("batch", {}).get("id") if ranked.get("batch") else None,
                "predicted": prediction.get("predicted", 0),
                "skipped": prediction.get("skipped", 0),
                "ranked": ranked.get("ranked", 0),
                "top_k": top_k,
                "updated_at": _now(),
            }
            self._save(job)
        except Exception as exc:
            job["next_cycle"] = {"status": "failed", "checkpoint_id": checkpoint_id, "error": str(exc), "updated_at": _now()}
            self._save(job)
