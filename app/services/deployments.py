from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import Settings
from app.core.json_store import read_json, write_json
from app.services.audit import AuditService
from app.services.checkpoints import CheckpointService


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-") or "deployment"


class DeploymentService:
    def __init__(self, settings: Settings, checkpoints: CheckpointService, audit: AuditService | None = None):
        self.settings = settings
        self.checkpoints = checkpoints
        self.audit = audit
        self.root = settings.exports_dir / "deployments"
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[dict]:
        rows = []
        for manifest_path in sorted(self.root.glob("*/manifest.json"), reverse=True):
            manifest = read_json(manifest_path, {})
            if manifest:
                rows.append(manifest)
        return rows

    def create(self, checkpoint_id: str, target: str = "torch_package", note: str = "") -> dict:
        checkpoint = self.checkpoints.get(checkpoint_id)
        source_path = Path(checkpoint["path"])
        if not source_path.exists():
            raise FileNotFoundError(f"Checkpoint missing: {source_path}")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        deployment_id = f"{stamp}-{_slug(checkpoint_id)}-{_slug(target)}"
        out_dir = self.root / deployment_id
        out_dir.mkdir(parents=True, exist_ok=False)
        model_dir = out_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_copy = model_dir / source_path.name
        shutil.copy2(source_path, model_copy)

        manifest = {
            "id": deployment_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_name": checkpoint.get("name"),
            "model_stage": checkpoint.get("model_stage"),
            "label_id": checkpoint.get("label_id"),
            "label_name": checkpoint.get("label_name"),
            "target": target,
            "status": "ready" if target == "torch_package" else "conversion_required",
            "model_file": str(model_copy),
            "source_checkpoint": str(source_path),
            "threshold_default": checkpoint.get("threshold_default"),
            "created_at": _now(),
            "note": note,
            "next_steps": [
                "Validate the package with a held-out dataset before production use.",
                "Convert to OpenVINO IR for Intel hardware deployment when the conversion toolchain is available.",
                "For OVMS, package model.xml/model.bin plus config.json after OpenVINO conversion.",
            ],
        }
        write_json(out_dir / "manifest.json", manifest)
        (out_dir / "README.md").write_text(self._readme(manifest), encoding="utf-8")
        write_json(out_dir / "ovms-next-config.json", {
            "model_config_list": [
                {
                    "config": {
                        "name": _slug(checkpoint.get("label_name") or checkpoint_id),
                        "base_path": "/opt/ml/model",
                        "target_device": "CPU",
                    }
                }
            ],
            "note": "Template only. Replace model/ with OpenVINO IR model.xml and model.bin before running OVMS.",
        })
        if self.audit:
            self.audit.record(
                action="deployment.create",
                resource_type="deployment",
                resource_id=deployment_id,
                payload={
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_name": checkpoint.get("name"),
                    "target": target,
                    "status": manifest["status"],
                    "verification": "verification" in note.lower(),
                },
            )
        return manifest

    def delete(self, deployment_id: str) -> dict:
        manifest = read_json(self.root / deployment_id / "manifest.json", {})
        target = (self.root / deployment_id).resolve()
        root = self.root.resolve()
        if not target.exists() or not str(target).lower().startswith(str(root).lower()):
            raise KeyError(f"Unknown deployment package: {deployment_id}")
        shutil.rmtree(target)
        if self.audit:
            self.audit.record(
                action="deployment.delete",
                resource_type="deployment",
                resource_id=deployment_id,
                payload={
                    "checkpoint_id": manifest.get("checkpoint_id"),
                    "checkpoint_name": manifest.get("checkpoint_name"),
                    "verification": "verification" in str(manifest.get("note", "")).lower(),
                },
            )
        return {"deleted": True, "deployment_id": deployment_id}

    def _readme(self, manifest: dict) -> str:
        return f"""# {manifest.get("checkpoint_name") or manifest["checkpoint_id"]} Deployment Package

Package ID: `{manifest["id"]}`

This package contains the selected PyTorch checkpoint and deployment metadata. It is a local deployment-preparation artifact for validation, registry review, and later OpenVINO conversion.

## Contents

- `manifest.json`: model identity, threshold, label, stage, and validation notes.
- `model/{Path(manifest["model_file"]).name}`: copied checkpoint file.
- `ovms-next-config.json`: template for the later OpenVINO Model Server package.

## Required Validation

1. Run model testing on a representative dataset.
2. Confirm Dice, Recall, IoU, and threshold selection.
3. Convert to OpenVINO IR before OpenVINO Model Server deployment.
4. Archive or delete temporary datasets after validation.
"""
