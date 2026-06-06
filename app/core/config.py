from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_TILE = 512
DEFAULT_STRIDE = 384
DEFAULT_THRESHOLD = 0.50


@dataclass(frozen=True)
class Settings:
    root: Path
    checkpoints_dir: Path
    baseline_dir: Path
    run_checkpoints_dir: Path
    storage_dir: Path
    datasets_dir: Path
    categories_dir: Path
    training_jobs_dir: Path
    exports_dir: Path
    tests_dir: Path
    archives_dir: Path
    audit_dir: Path
    database_path: Path
    web_dir: Path
    default_tile: int = DEFAULT_TILE
    default_stride: int = DEFAULT_STRIDE
    default_threshold: float = DEFAULT_THRESHOLD


def get_settings() -> Settings:
    root = Path(__file__).resolve().parents[2]
    checkpoints_dir = root / "checkpoints"
    storage_dir = root / "storage"
    return Settings(
        root=root,
        checkpoints_dir=checkpoints_dir,
        baseline_dir=checkpoints_dir / "baseline",
        run_checkpoints_dir=checkpoints_dir / "runs",
        storage_dir=storage_dir,
        datasets_dir=storage_dir / "datasets",
        categories_dir=storage_dir / "categories",
        training_jobs_dir=storage_dir / "training_jobs",
        exports_dir=storage_dir / "exports",
        tests_dir=storage_dir / "model_tests",
        archives_dir=storage_dir / "dataset_archives",
        audit_dir=storage_dir / "audit_logs",
        database_path=storage_dir / "platform.sqlite3",
        web_dir=root / "web",
    )


def ensure_layout(settings: Settings) -> None:
    for path in (
        settings.baseline_dir,
        settings.run_checkpoints_dir,
        settings.datasets_dir,
        settings.categories_dir,
        settings.training_jobs_dir,
        settings.exports_dir,
        settings.tests_dir,
        settings.archives_dir,
        settings.audit_dir,
        settings.web_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
