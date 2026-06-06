from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.core.config import DEFAULT_STRIDE, DEFAULT_THRESHOLD, DEFAULT_TILE


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=96)
    defect_class: str = Field(default="Spall", min_length=1, max_length=64)
    label_id: str | None = Field(default=None, min_length=1, max_length=64)
    label_name: str | None = Field(default=None, min_length=1, max_length=64)
    label_color: str = Field(default="#ff8a80", min_length=4, max_length=16)
    project_type: Literal["optimization", "generation"] = "optimization"


class ImportImagesRequest(BaseModel):
    source_dir: str


class DatasetCreateImportRequest(DatasetCreate):
    source_dir: str | None = None


class DatasetUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=96)
    defect_class: str | None = Field(default=None, min_length=1, max_length=64)
    label_id: str | None = Field(default=None, min_length=1, max_length=64)
    label_name: str | None = Field(default=None, min_length=1, max_length=64)
    label_color: str | None = Field(default=None, min_length=4, max_length=16)


class DatasetSnapshotRequest(BaseModel):
    name: str = "manual-snapshot"
    note: str = ""


class DatasetExportRequest(BaseModel):
    format: Literal["active_learning", "coco", "fiftyone"] = "active_learning"
    label_id: str = "spall"
    include_predictions: bool = True


class DeploymentPackageRequest(BaseModel):
    checkpoint_id: str = "baseline-spall-unet-recall-v1"
    target: Literal["torch_package", "openvino_next"] = "torch_package"
    note: str = ""


class PredictRequest(BaseModel):
    checkpoint_id: str = "baseline-spall-unet-recall-v1"
    threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0.01, le=0.99)
    tile: int = Field(default=DEFAULT_TILE, ge=128, le=2048)
    stride: int = Field(default=DEFAULT_STRIDE, ge=64, le=2048)


class BatchPredictRequest(BaseModel):
    checkpoint_id: str = "baseline-spall-unet-recall-v1"
    threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0.01, le=0.99)
    tile: int = Field(default=DEFAULT_TILE, ge=128, le=2048)
    stride: int = Field(default=DEFAULT_STRIDE, ge=64, le=2048)
    limit: int = Field(default=0, ge=0, le=100000)
    only_unreviewed: bool = True
    force: bool = False


class ActiveLearningRankRequest(BaseModel):
    checkpoint_id: str = "baseline-spall-unet-recall-v1"
    label_id: str = "spall"
    threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0.01, le=0.99)
    tile: int = Field(default=DEFAULT_TILE, ge=128, le=2048)
    stride: int = Field(default=DEFAULT_STRIDE, ge=64, le=2048)
    top_k: int = Field(default=20, ge=1, le=1000)
    predict_missing: bool = False
    create_batch: bool = True


class SaveAnnotationRequest(BaseModel):
    mask_png_base64: str
    label_id: str = "spall"
    source: Literal["manual_review", "imported_mask", "system"] = "manual_review"
    reviewer: str = "local"


class TrainJobRequest(BaseModel):
    dataset_id: str
    base_checkpoint_id: str = "baseline-spall-unet-recall-v1"
    label_id: str = "spall"
    active_batch_id: str | None = None
    epochs: int = Field(default=8, ge=1, le=200)
    samples_per_epoch: int = Field(default=512, ge=32, le=20000)
    batch_size: int = Field(default=8, ge=1, le=32)
    learning_rate: float = Field(default=2e-4, gt=0, le=0.01)
    note: str = ""


class ModelTestRequest(BaseModel):
    checkpoint_id: str = "baseline-spall-unet-recall-v1"
    label_id: str = "spall"
    threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0.01, le=0.99)
    tile: int = Field(default=DEFAULT_TILE, ge=128, le=2048)
    stride: int = Field(default=DEFAULT_STRIDE, ge=64, le=2048)
    sample_count: int = Field(default=20, ge=1, le=200)
    seed: int | None = None
