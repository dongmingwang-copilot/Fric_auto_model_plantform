from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import create_router
from app.core.config import ensure_layout, get_settings
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


settings = get_settings()
ensure_layout(settings)

checkpoint_service = CheckpointService(settings)
audit_service = AuditService(settings)
dataset_service = DatasetService(settings)
export_service = ExportService(settings, dataset_service)
inference_service = InferenceService(settings, checkpoint_service, dataset_service)
annotation_service = AnnotationService(dataset_service)
active_learning_service = ActiveLearningService(dataset_service, inference_service)
training_job_service = TrainingJobService(settings, checkpoint_service, dataset_service, active_learning_service)
model_test_service = ModelTestService(settings, dataset_service, checkpoint_service, inference_service)
deployment_service = DeploymentService(settings, checkpoint_service, audit_service)

app = FastAPI(title="Plantform v1", version="0.1.0")
app.include_router(create_router(
    checkpoint_service,
    dataset_service,
    export_service,
    inference_service,
    annotation_service,
    training_job_service,
    active_learning_service,
    model_test_service,
    deployment_service,
    audit_service,
    settings,
))
app.mount("/static", StaticFiles(directory=settings.web_dir), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(settings.web_dir / "index.html")


@app.get("/optimize")
def optimize_page() -> FileResponse:
    return FileResponse(settings.web_dir / "optimize.html")


@app.get("/generate")
def generate_page() -> FileResponse:
    return FileResponse(settings.web_dir / "generation.html")


@app.get("/datasets")
def datasets_page() -> FileResponse:
    return FileResponse(settings.web_dir / "datasets.html")


@app.get("/training")
def training_page() -> FileResponse:
    return FileResponse(settings.web_dir / "training.html")


@app.get("/testing")
def testing_page() -> FileResponse:
    return FileResponse(settings.web_dir / "testing.html")
