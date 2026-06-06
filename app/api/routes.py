from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.config import Settings
from app.domain.schemas import ActiveLearningRankRequest, BatchPredictRequest, DatasetCreate, DatasetCreateImportRequest, DatasetExportRequest, DatasetSnapshotRequest, DatasetUpdate, DeploymentPackageRequest, ImportImagesRequest, ModelTestRequest, PredictRequest, SaveAnnotationRequest, TrainJobRequest
from app.services.active_learning import ActiveLearningService
from app.services.annotations import AnnotationService
from app.services.audit import AuditService
from app.services.checkpoints import CheckpointService
from app.services.datasets import DatasetService
from app.services.deployments import DeploymentService
from app.services.exports import ExportService
from app.services.inference import InferenceService
from app.services.model_tests import ModelTestService
from app.services.training_jobs import TrainingJobService


def create_router(
    checkpoints: CheckpointService,
    datasets: DatasetService,
    exports: ExportService,
    inference: InferenceService,
    annotations: AnnotationService,
    training_jobs: TrainingJobService,
    active_learning: ActiveLearningService,
    model_tests: ModelTestService,
    deployments: DeploymentService,
    audit: AuditService,
    settings: Settings,
) -> APIRouter:
    router = APIRouter(prefix="/api")

    def _verification_flag(*values: object) -> bool:
        haystack = " ".join(str(value or "") for value in values).lower()
        return "verification" in haystack or "验证" in haystack

    def _record_audit(action: str, resource_type: str, resource_id: str, payload: dict | None = None) -> None:
        audit.record(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            payload=payload or {},
        )

    @router.get("/health")
    def health() -> dict:
        return {"ok": True, "device": str(inference.device)}

    @router.get("/settings")
    def platform_settings() -> dict:
        return {
            "inference": {
                "threshold": settings.default_threshold,
                "tile": settings.default_tile,
                "stride": settings.default_stride,
            }
        }

    @router.get("/checkpoints")
    def list_checkpoints() -> list[dict]:
        return checkpoints.list()

    @router.get("/deployments")
    def list_deployments() -> list[dict]:
        return deployments.list()

    @router.post("/deployments")
    def create_deployment(payload: DeploymentPackageRequest) -> dict:
        try:
            return deployments.create(payload.checkpoint_id, payload.target, payload.note)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.delete("/deployments/{deployment_id}")
    def delete_deployment(deployment_id: str) -> dict:
        try:
            return deployments.delete(deployment_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/audit/events")
    def list_audit_events(resource_type: str | None = None, action: str | None = None, limit: int = 50) -> list[dict]:
        return audit.list(resource_type=resource_type, action=action, limit=limit)

    @router.delete("/audit/verification-events")
    def clear_verification_audit_events() -> dict:
        return audit.clear_verification_events()

    @router.get("/datasets")
    def list_datasets(project_type: str | None = None, label_id: str | None = None) -> list[dict]:
        return datasets.list(project_type, label_id)

    @router.get("/datasets/catalog")
    def dataset_catalog(project_type: str | None = None, label_id: str | None = None) -> list[dict]:
        return datasets.catalog(project_type, label_id)

    @router.get("/datasets/events")
    def dataset_events(project_type: str | None = None, label_id: str | None = None, dataset_id: str | None = None, limit: int = 80) -> list[dict]:
        return datasets.events(project_type, label_id, dataset_id, limit)

    @router.delete("/datasets/verification-events")
    def clear_verification_dataset_events() -> dict:
        return datasets.clear_verification_events()

    @router.post("/datasets")
    def create_dataset(payload: DatasetCreate) -> dict:
        result = datasets.create(payload.name, payload.defect_class, payload.label_id, payload.label_name, payload.label_color, payload.project_type)
        _record_audit("dataset.create", "dataset", result["id"], {
            "name": result.get("name"),
            "defect_class": result.get("defect_class"),
            "project_type": result.get("project_type"),
            "label_id": payload.label_id,
            "label_name": payload.label_name,
            "verification": _verification_flag(payload.name, payload.defect_class),
        })
        return result

    @router.post("/datasets/create-and-import")
    def create_and_import_dataset(payload: DatasetCreateImportRequest) -> dict:
        try:
            result = datasets.create_and_import(
                payload.name,
                payload.defect_class,
                payload.label_id,
                payload.label_name,
                payload.label_color,
                payload.project_type,
                Path(payload.source_dir) if payload.source_dir else None,
            )
            dataset = result["dataset"]
            imported = result.get("import") or {}
            _record_audit("dataset.create_import", "dataset", dataset["id"], {
                "name": dataset.get("name"),
                "defect_class": dataset.get("defect_class"),
                "project_type": dataset.get("project_type"),
                "source_dir": payload.source_dir,
                "imported_count": imported.get("imported") or imported.get("count") or dataset.get("n_items") or len(dataset.get("items", [])),
                "verification": _verification_flag(payload.name, payload.defect_class, payload.source_dir),
            })
            return result
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.patch("/datasets/{dataset_id}")
    def update_dataset(dataset_id: str, payload: DatasetUpdate) -> dict:
        try:
            result = datasets.update_dataset(
                dataset_id,
                payload.name,
                payload.defect_class,
                payload.label_id,
                payload.label_name,
                payload.label_color,
            )
            _record_audit("dataset.update", "dataset", dataset_id, {
                "name": result.get("name"),
                "defect_class": result.get("defect_class"),
                "label_id": payload.label_id,
                "label_name": payload.label_name,
                "label_color": payload.label_color,
                "verification": _verification_flag(payload.name, payload.defect_class),
            })
            return result
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/dataset-archives")
    def list_dataset_archives(project_type: str | None = None, label_id: str | None = None) -> list[dict]:
        return datasets.list_archives(project_type, label_id)

    @router.post("/dataset-archives/{archive_id}/restore")
    def restore_dataset_archive(archive_id: str, project_type: str | None = None) -> dict:
        try:
            result = datasets.restore_archive(archive_id, project_type)
            _record_audit("dataset_archive.restore", "dataset_archive", archive_id, {
                "restored_dataset_id": result.get("id") or result.get("dataset_id"),
                "project_type": project_type or result.get("project_type"),
                "verification": _verification_flag(archive_id),
            })
            return result
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/datasets/{dataset_id}/import")
    def import_images(dataset_id: str, payload: ImportImagesRequest) -> dict:
        try:
            result = datasets.import_images(dataset_id, Path(payload.source_dir))
            _record_audit("dataset.import", "dataset", dataset_id, {
                "source_dir": payload.source_dir,
                "imported_count": result.get("imported") or result.get("count"),
                "verification": _verification_flag(dataset_id, payload.source_dir),
            })
            return result
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/datasets/{dataset_id}/archive")
    def archive_dataset(dataset_id: str) -> dict:
        try:
            result = datasets.archive(dataset_id)
            _record_audit("dataset.archive", "dataset", dataset_id, {
                "archive_id": result.get("archive_id") or result.get("id"),
                "verification": _verification_flag(dataset_id),
            })
            return result
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.delete("/datasets/{dataset_id}")
    def delete_dataset(dataset_id: str) -> dict:
        try:
            result = datasets.delete(dataset_id)
            _record_audit("dataset.delete", "dataset", dataset_id, {
                "deleted": result.get("deleted", True),
                "verification": _verification_flag(dataset_id),
            })
            return result
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/datasets/{dataset_id}/items")
    def list_items(dataset_id: str) -> list[dict]:
        try:
            return datasets.list_items(dataset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/datasets/{dataset_id}/summary")
    def dataset_summary(dataset_id: str) -> dict:
        try:
            return datasets.summary(dataset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/datasets/{dataset_id}/snapshots")
    def create_dataset_snapshot(dataset_id: str, payload: DatasetSnapshotRequest) -> dict:
        try:
            result = datasets.create_snapshot(dataset_id, payload.name, payload.note)
            _record_audit("dataset.snapshot", "dataset", dataset_id, {
                "snapshot_id": result.get("id") or result.get("name"),
                "name": payload.name,
                "verification": _verification_flag(dataset_id, payload.name, payload.note),
            })
            return result
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/datasets/{dataset_id}/rebuild-metadata")
    def rebuild_dataset_metadata(dataset_id: str) -> dict:
        try:
            result = datasets.rebuild_metadata(dataset_id)
            _record_audit("dataset.rebuild_metadata", "dataset", dataset_id, {
                "item_count": result.get("item_count") or result.get("n_items"),
                "verification": _verification_flag(dataset_id),
            })
            return result
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/datasets/{dataset_id}/exports")
    def export_dataset(dataset_id: str, payload: DatasetExportRequest) -> dict:
        try:
            result = exports.export(dataset_id, payload.format, payload.label_id, payload.include_predictions)
            _record_audit("dataset.export", "dataset", dataset_id, {
                "format": payload.format,
                "label_id": payload.label_id,
                "include_predictions": payload.include_predictions,
                "export_path": result.get("path") or result.get("export_path"),
                "verification": _verification_flag(dataset_id),
            })
            return result
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/datasets/{dataset_id}/items/{item_id}")
    def get_item(dataset_id: str, item_id: str) -> dict:
        try:
            return datasets.get_item(dataset_id, item_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/datasets/{dataset_id}/items/{item_id}/image")
    def get_item_image(dataset_id: str, item_id: str) -> FileResponse:
        try:
            item = datasets.get_item(dataset_id, item_id)
            return FileResponse(item["image_path"])
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/datasets/{dataset_id}/items/{item_id}/prediction-mask")
    def get_prediction_mask(dataset_id: str, item_id: str, checkpoint_id: str) -> FileResponse:
        try:
            item = datasets.get_item(dataset_id, item_id)
            pred = item.get("predictions", {}).get(checkpoint_id)
            if not pred:
                raise KeyError(f"No prediction for {checkpoint_id}")
            return FileResponse(pred["mask_path"])
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/datasets/{dataset_id}/items/{item_id}/overlay")
    def get_overlay(dataset_id: str, item_id: str, checkpoint_id: str) -> FileResponse:
        try:
            item = datasets.get_item(dataset_id, item_id)
            pred = item.get("predictions", {}).get(checkpoint_id)
            if not pred:
                raise KeyError(f"No prediction for {checkpoint_id}")
            return FileResponse(pred["overlay_path"])
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/datasets/{dataset_id}/items/{item_id}/annotation")
    def get_annotation(dataset_id: str, item_id: str, label_id: str = "spall") -> FileResponse:
        try:
            item = datasets.get_item(dataset_id, item_id)
            annotation = item.get("annotations", {}).get(label_id)
            path = annotation.get("path") if annotation else item.get("annotation_path") if label_id == "spall" else None
            if not path:
                raise KeyError("No annotation saved")
            return FileResponse(path)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/datasets/{dataset_id}/items/{item_id}/predict")
    def predict(dataset_id: str, item_id: str, payload: PredictRequest) -> dict:
        try:
            result = inference.predict(
                dataset_id=dataset_id,
                item_id=item_id,
                checkpoint_id=payload.checkpoint_id,
                threshold=payload.threshold,
                tile=payload.tile,
                stride=payload.stride,
            )
            _record_audit("prediction.create", "dataset_item", f"{dataset_id}:{item_id}", {
                "dataset_id": dataset_id,
                "item_id": item_id,
                "checkpoint_id": payload.checkpoint_id,
                "threshold": payload.threshold,
                "tile": payload.tile,
                "stride": payload.stride,
                "verification": _verification_flag(dataset_id, item_id),
            })
            return result
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/datasets/{dataset_id}/active-learning/batch-predict")
    def active_learning_batch_predict(dataset_id: str, payload: BatchPredictRequest) -> dict:
        try:
            result = active_learning.batch_predict(
                dataset_id,
                payload.checkpoint_id,
                payload.threshold,
                payload.tile,
                payload.stride,
                payload.limit,
                payload.only_unreviewed,
                payload.force,
            )
            _record_audit("active_learning.batch_predict", "dataset", dataset_id, {
                "checkpoint_id": payload.checkpoint_id,
                "threshold": payload.threshold,
                "limit": payload.limit,
                "only_unreviewed": payload.only_unreviewed,
                "force": payload.force,
                "predicted_count": result.get("predicted") or result.get("count"),
                "verification": _verification_flag(dataset_id),
            })
            return result
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/datasets/{dataset_id}/active-learning/rank")
    def active_learning_rank(dataset_id: str, payload: ActiveLearningRankRequest) -> dict:
        try:
            result = active_learning.rank(
                dataset_id,
                payload.checkpoint_id,
                payload.label_id,
                payload.threshold,
                payload.tile,
                payload.stride,
                payload.top_k,
                payload.predict_missing,
                payload.create_batch,
            )
            _record_audit("active_learning.rank", "dataset", dataset_id, {
                "checkpoint_id": payload.checkpoint_id,
                "label_id": payload.label_id,
                "top_k": payload.top_k,
                "predict_missing": payload.predict_missing,
                "create_batch": payload.create_batch,
                "batch_id": result.get("batch_id") or result.get("id"),
                "ranked_count": len(result.get("items", []) or result.get("ranked", []) or []),
                "verification": _verification_flag(dataset_id),
            })
            return result
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/datasets/{dataset_id}/active-learning/initial-review-queue")
    def active_learning_initial_review_queue(dataset_id: str, payload: ActiveLearningRankRequest) -> dict:
        try:
            result = active_learning.create_initial_review_queue(
                dataset_id,
                payload.label_id,
                payload.top_k,
            )
            _record_audit("active_learning.initial_review_queue", "dataset", dataset_id, {
                "label_id": payload.label_id,
                "top_k": payload.top_k,
                "batch_id": result.get("batch_id") or result.get("id"),
                "verification": _verification_flag(dataset_id),
            })
            return result
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/datasets/{dataset_id}/active-learning/batches")
    def active_learning_batches(dataset_id: str) -> list[dict]:
        try:
            return active_learning.list_batches(dataset_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/datasets/{dataset_id}/active-learning/batches/{batch_id}")
    def active_learning_batch(dataset_id: str, batch_id: str) -> dict:
        try:
            return active_learning.get_batch(dataset_id, batch_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/datasets/{dataset_id}/active-learning/batches/{batch_id}/items/{item_id}/reviewed")
    def active_learning_mark_reviewed(dataset_id: str, batch_id: str, item_id: str) -> dict:
        try:
            result = active_learning.mark_item_reviewed(dataset_id, batch_id, item_id)
            _record_audit("active_learning.item_reviewed", "dataset_item", f"{dataset_id}:{item_id}", {
                "dataset_id": dataset_id,
                "batch_id": batch_id,
                "item_id": item_id,
                "verification": _verification_flag(dataset_id, batch_id, item_id),
            })
            return result
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/datasets/{dataset_id}/items/{item_id}/annotations")
    def save_annotation(dataset_id: str, item_id: str, payload: SaveAnnotationRequest) -> dict:
        try:
            result = annotations.save_review_mask(
                dataset_id=dataset_id,
                item_id=item_id,
                mask_png_base64=payload.mask_png_base64,
                reviewer=payload.reviewer,
                source=payload.source,
                label_id=payload.label_id,
            )
            _record_audit("annotation.save", "dataset_item", f"{dataset_id}:{item_id}", {
                "dataset_id": dataset_id,
                "item_id": item_id,
                "label_id": payload.label_id,
                "source": payload.source,
                "reviewer": payload.reviewer,
                "mask_path": result.get("path") or result.get("mask_path"),
                "verification": _verification_flag(dataset_id, item_id, payload.reviewer, payload.source),
            })
            return result
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/training/jobs")
    def create_training_job(payload: TrainJobRequest) -> dict:
        result = training_jobs.create(
            payload.dataset_id,
            payload.base_checkpoint_id,
            payload.label_id,
            payload.active_batch_id,
            payload.epochs,
            payload.samples_per_epoch,
            payload.batch_size,
            payload.learning_rate,
            payload.note,
        )
        _record_audit("training_job.create", "training_job", result.get("id") or result.get("job_id"), {
            "dataset_id": payload.dataset_id,
            "base_checkpoint_id": payload.base_checkpoint_id,
            "label_id": payload.label_id,
            "active_batch_id": payload.active_batch_id,
            "epochs": payload.epochs,
            "samples_per_epoch": payload.samples_per_epoch,
            "batch_size": payload.batch_size,
            "learning_rate": payload.learning_rate,
            "verification": _verification_flag(payload.dataset_id, payload.note),
        })
        return result

    @router.get("/training/jobs")
    def list_training_jobs() -> list[dict]:
        return training_jobs.list()

    @router.get("/training/jobs/{job_id}")
    def get_training_job(job_id: str) -> dict:
        try:
            return training_jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/training/jobs/{job_id}/metrics")
    def get_training_metrics(job_id: str) -> dict:
        try:
            job = training_jobs.get(job_id)
            metrics_path = Path(job["output_dir"]) / "metrics.json"
            live = {
                "status": job.get("status"),
                "progress": job.get("progress"),
                "history": job.get("progress_history") or [],
                "checkpoint_available": bool(job.get("checkpoint_available")),
                "params": job.get("params") or {},
                "training_scope": job.get("training_scope"),
            }
            if not metrics_path.exists():
                return live
            import json
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["status"] = job.get("status")
            metrics["progress"] = job.get("progress")
            metrics["checkpoint_available"] = bool(job.get("checkpoint_available"))
            if not metrics.get("history"):
                metrics["history"] = job.get("progress_history") or []
            return metrics
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/training/jobs/{job_id}/visualizations")
    def list_training_visualizations(job_id: str) -> list[dict]:
        try:
            job = training_jobs.get(job_id)
            manifest = Path(job["output_dir"]) / "test_visualizations" / "manifest.json"
            if not manifest.exists():
                return []
            import json
            return json.loads(manifest.read_text(encoding="utf-8"))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/training/jobs/{job_id}/visualizations/{filename}")
    def get_training_visualization(job_id: str, filename: str) -> FileResponse:
        try:
            job = training_jobs.get(job_id)
            viz_dir = Path(job["output_dir"]) / "test_visualizations"
            path = (viz_dir / filename).resolve()
            if not str(path).lower().startswith(str(viz_dir.resolve()).lower()) or not path.exists():
                raise KeyError(f"Unknown visualization: {filename}")
            return FileResponse(path)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/datasets/{dataset_id}/model-tests")
    def create_model_test(dataset_id: str, payload: ModelTestRequest) -> dict:
        try:
            result = model_tests.run(
                dataset_id=dataset_id,
                checkpoint_id=payload.checkpoint_id,
                label_id=payload.label_id,
                threshold=payload.threshold,
                tile=payload.tile,
                stride=payload.stride,
                sample_count=payload.sample_count,
                seed=payload.seed,
            )
            _record_audit("model_test.create", "model_test", result.get("id") or result.get("run_id"), {
                "dataset_id": dataset_id,
                "checkpoint_id": payload.checkpoint_id,
                "label_id": payload.label_id,
                "sample_count": payload.sample_count,
                "seed": payload.seed,
                "verification": _verification_flag(dataset_id),
            })
            return result
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/datasets/{dataset_id}/model-tests")
    def list_model_tests(dataset_id: str) -> list[dict]:
        try:
            return model_tests.list(dataset_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/datasets/{dataset_id}/model-tests/{run_id}")
    def get_model_test(dataset_id: str, run_id: str) -> dict:
        try:
            return model_tests.get(dataset_id, run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/datasets/{dataset_id}/model-tests/{run_id}/files/{filename}")
    def get_model_test_file(dataset_id: str, run_id: str, filename: str) -> FileResponse:
        try:
            root = model_tests.settings.tests_dir / dataset_id / run_id
            path = (root / filename).resolve()
            if not str(path).lower().startswith(str(root.resolve()).lower()) or not path.exists():
                raise KeyError(f"Unknown model test file: {filename}")
            return FileResponse(path)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return router
