from __future__ import annotations

import json
import re
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import Settings
from app.core.json_store import read_json, write_json
from app.services.importers.image_importer import IMAGE_EXTS, ImageImporter
from app.services.importers.scientific_importer import SCIENTIFIC_EXTS, ScientificImageImporter


SUPPORTED_EXTS = IMAGE_EXTS | SCIENTIFIC_EXTS


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip().lower()).strip("-")
    return cleaned or "dataset"


class DatasetService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.importers = [ImageImporter(), ScientificImageImporter()]

    def list(self, project_type: str | None = None, label_id: str | None = None) -> list[dict]:
        datasets = []
        for meta_path in sorted(self.settings.datasets_dir.glob("*/dataset.json")):
            meta = read_json(meta_path, {})
            row_project_type = meta.get("project_type", "optimization")
            if project_type and row_project_type != project_type:
                continue
            labels = meta.get("labels", [])
            if label_id and not any(label.get("id") == label_id for label in labels):
                continue
            datasets.append({
                "id": meta.get("id"),
                "name": meta.get("name"),
                "defect_class": meta.get("defect_class"),
                "project_type": row_project_type,
                "labels": labels,
                "created_at": meta.get("created_at"),
                "updated_at": meta.get("updated_at"),
                "n_items": len(meta.get("items", [])),
                "schema_version": meta.get("schema_version", "legacy"),
            })
        return datasets

    def create(self, name: str, defect_class: str, label_id: str = "spall", label_name: str | None = None, label_color: str = "#ff8a80", project_type: str = "optimization") -> dict:
        project_prefix = "gen" if project_type == "generation" else "opt"
        dataset_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{project_prefix}-{_slug(name)}"
        root = self.dataset_root(dataset_id)
        self.ensure_dataset_layout(root)
        clean_label_id = _slug(label_id or defect_class)
        clean_label_name = label_name or defect_class or clean_label_id
        meta = {
            "schema_version": "metalwear.dataset.v2",
            "id": dataset_id,
            "name": name,
            "defect_class": defect_class,
            "project_type": project_type,
            "labels": [
                {"id": clean_label_id, "name": clean_label_name, "color": label_color, "enabled": True},
            ],
            "created_at": _now(),
            "updated_at": _now(),
            "imports": [],
            "versions": [],
            "items": [],
        }
        self.save_meta(dataset_id, meta)
        self._record_event("create", meta, {"label_id": clean_label_id})
        return meta

    def create_and_import(
        self,
        name: str,
        defect_class: str,
        label_id: str,
        label_name: str,
        label_color: str,
        project_type: str,
        source_dir: Path | None,
    ) -> dict:
        meta = self.create(name, defect_class, label_id, label_name, label_color, project_type)
        imported = None
        if source_dir:
            imported = self.import_images(meta["id"], source_dir)
            meta = self.get_meta(meta["id"])
        return {
            "dataset": meta,
            "import": imported,
            "catalog": self.catalog(project_type=project_type, label_id=label_id),
        }

    def update_dataset(
        self,
        dataset_id: str,
        name: str | None = None,
        defect_class: str | None = None,
        label_id: str | None = None,
        label_name: str | None = None,
        label_color: str | None = None,
    ) -> dict:
        meta = self.get_meta(dataset_id)
        before = {
            "name": meta.get("name"),
            "defect_class": meta.get("defect_class"),
            "labels": meta.get("labels", []),
        }
        if name is not None:
            meta["name"] = name.strip()
        if defect_class is not None:
            meta["defect_class"] = defect_class.strip()

        labels = list(meta.get("labels") or [])
        current = dict(labels[0] if labels else {})
        old_label_id = current.get("id")
        defect_changed = defect_class is not None
        if label_id is not None:
            current["id"] = _slug(label_id)
        elif defect_changed:
            current["id"] = _slug(meta.get("defect_class") or "defect")
        elif not current.get("id"):
            current["id"] = _slug(meta.get("defect_class") or "defect")
        if label_name is not None:
            current["name"] = label_name.strip()
        elif defect_changed:
            current["name"] = meta.get("defect_class") or current.get("id") or "defect"
        elif not current.get("name"):
            current["name"] = meta.get("defect_class") or current["id"]
        if label_color is not None:
            current["color"] = label_color
        elif not current.get("color"):
            current["color"] = "#ff9e94"
        current["enabled"] = bool(current.get("enabled", True))
        meta["labels"] = [current, *labels[1:]]

        new_label_id = current.get("id")
        if old_label_id and new_label_id and old_label_id != new_label_id:
            self._rename_item_label(meta, old_label_id, new_label_id)
            self._remove_category_indexes({**meta, "labels": [{"id": old_label_id}], "project_type": meta.get("project_type", "optimization")})

        self.save_meta(dataset_id, meta)
        self._record_event("update_dataset", meta, {
            "before": before,
            "after": {
                "name": meta.get("name"),
                "defect_class": meta.get("defect_class"),
                "labels": meta.get("labels", []),
            },
        })
        return meta

    def _rename_item_label(self, meta: dict, old_label_id: str, new_label_id: str) -> None:
        root = self.dataset_root(meta["id"])
        old_reviewed_dir = root / "masks" / "reviewed" / old_label_id
        new_reviewed_dir = root / "masks" / "reviewed" / new_label_id
        old_annotation_dir = root / "annotations" / old_label_id
        new_annotation_dir = root / "annotations" / new_label_id
        if old_reviewed_dir.exists() and not new_reviewed_dir.exists():
            new_reviewed_dir.parent.mkdir(parents=True, exist_ok=True)
            old_reviewed_dir.rename(new_reviewed_dir)
        if old_annotation_dir.exists() and not new_annotation_dir.exists():
            new_annotation_dir.parent.mkdir(parents=True, exist_ok=True)
            old_annotation_dir.rename(new_annotation_dir)
        for item in meta.get("items", []):
            annotations = item.setdefault("annotations", {})
            if old_label_id in annotations and new_label_id not in annotations:
                annotations[new_label_id] = annotations.pop(old_label_id)
                annotation = annotations[new_label_id]
                if annotation.get("path"):
                    annotation["path"] = str(Path(annotation["path"]).parent.parent / new_label_id / Path(annotation["path"]).name)
                if annotation.get("synced_mask_path"):
                    annotation["synced_mask_path"] = str(Path(annotation["synced_mask_path"]).parent.parent / new_label_id / Path(annotation["synced_mask_path"]).name)
                if old_label_id == "spall" or new_label_id == "spall":
                    item["annotation"] = annotation
                    item["annotation_path"] = annotation.get("path")

    def catalog(self, project_type: str | None = None, label_id: str | None = None) -> list[dict]:
        rows = self._all_catalog_rows()
        self._sync_catalog_table(rows)
        self._ensure_event_baseline(rows)
        self._sync_category_indexes()
        return [
            row
            for row in rows
            if (not project_type or row.get("project_type") == project_type)
            and (not label_id or row.get("label_id") == label_id)
        ]

    def events(self, project_type: str | None = None, label_id: str | None = None, dataset_id: str | None = None, limit: int = 80) -> list[dict]:
        self._ensure_event_baseline(self._all_catalog_rows())
        query = "SELECT id, dataset_id, project_type, label_id, event_type, payload_json, created_at FROM dataset_events"
        clauses = []
        params: list[object] = []
        if project_type:
            clauses.append("project_type = ?")
            params.append(project_type)
        if label_id:
            clauses.append("label_id = ?")
            params.append(label_id)
        if dataset_id:
            clauses.append("dataset_id = ?")
            params.append(dataset_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, min(int(limit or 80), 500)))
        with sqlite3.connect(self.settings.database_path) as conn:
            self._ensure_database(conn)
            rows = []
            for row in conn.execute(query, params).fetchall():
                payload_raw = row[5] or "{}"
                try:
                    payload = json.loads(payload_raw)
                except json.JSONDecodeError:
                    payload = {"raw": payload_raw}
                rows.append({
                    "id": row[0],
                    "dataset_id": row[1],
                    "project_type": row[2],
                    "label_id": row[3],
                    "event_type": row[4],
                    "payload": payload,
                    "created_at": row[6],
                })
            return rows

    def clear_verification_events(self) -> dict:
        with sqlite3.connect(self.settings.database_path) as conn:
            self._ensure_database(conn)
            cursor = conn.execute(
                """
                DELETE FROM dataset_events
                WHERE lower(dataset_id) LIKE '%verification%'
                   OR lower(label_id) LIKE '%verification%'
                   OR lower(payload_json) LIKE '%verification%'
                   OR dataset_id LIKE '%\u9a8c\u8bc1%'
                   OR label_id LIKE '%\u9a8c\u8bc1%'
                   OR payload_json LIKE '%\u9a8c\u8bc1%'
                """
            )
            deleted = int(cursor.rowcount or 0)
        return {"deleted": deleted}

    def _all_catalog_rows(self) -> list[dict]:
        rows = []
        for meta_path in sorted(self.settings.datasets_dir.glob("*/dataset.json")):
            meta = read_json(meta_path, {})
            if not meta:
                continue
            row_project_type = meta.get("project_type", "optimization")
            labels = meta.get("labels", [])
            for label in labels or [{"id": None, "name": meta.get("defect_class"), "color": None}]:
                rows.append(self._catalog_row(meta, label))
        return rows

    def _ensure_database(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_catalog (
                dataset_id TEXT NOT NULL,
                label_id TEXT NOT NULL,
                project_type TEXT NOT NULL,
                project_name TEXT,
                defect_class TEXT,
                label_name TEXT,
                label_color TEXT,
                count INTEGER NOT NULL,
                reviewed INTEGER NOT NULL,
                predicted INTEGER NOT NULL,
                status TEXT,
                format TEXT,
                formats_json TEXT,
                best_checkpoint_id TEXT,
                best_pt TEXT,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY (dataset_id, label_id, project_type)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT NOT NULL,
                project_type TEXT,
                label_id TEXT,
                event_type TEXT NOT NULL,
                payload_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            DELETE FROM dataset_events
            WHERE event_type = 'catalog_baseline'
              AND id NOT IN (
                SELECT MIN(id)
                FROM dataset_events
                WHERE event_type = 'catalog_baseline'
                GROUP BY dataset_id, label_id, event_type
              )
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS dataset_events_catalog_baseline_unique
            ON dataset_events(dataset_id, label_id, event_type)
            WHERE event_type = 'catalog_baseline'
            """
        )

    def _sync_catalog_table(self, rows: list[dict]) -> None:
        with sqlite3.connect(self.settings.database_path) as conn:
            self._ensure_database(conn)
            conn.execute("DELETE FROM dataset_catalog")
            conn.executemany(
                """
                INSERT OR REPLACE INTO dataset_catalog (
                    dataset_id, label_id, project_type, project_name, defect_class,
                    label_name, label_color, count, reviewed, predicted, status,
                    format, formats_json, best_checkpoint_id, best_pt, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row.get("dataset_id"),
                        row.get("label_id") or "",
                        row.get("project_type") or "",
                        row.get("project_name"),
                        row.get("defect_class"),
                        row.get("label_name"),
                        row.get("label_color"),
                        int(row.get("count") or 0),
                        int(row.get("reviewed") or 0),
                        int(row.get("predicted") or 0),
                        row.get("status"),
                        row.get("format"),
                        json.dumps(row.get("formats") or {}, ensure_ascii=False),
                        row.get("best_checkpoint_id"),
                        row.get("best_pt"),
                        row.get("created_at"),
                        row.get("updated_at"),
                    )
                    for row in rows
                ],
            )

    def _record_event(self, event_type: str, meta: dict, payload: dict | None = None) -> None:
        label = (meta.get("labels") or [{}])[0]
        with sqlite3.connect(self.settings.database_path) as conn:
            self._ensure_database(conn)
            conn.execute(
                """
                INSERT INTO dataset_events (
                    dataset_id, project_type, label_id, event_type, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    meta.get("id") or "",
                    meta.get("project_type", "optimization"),
                    label.get("id") or "",
                    event_type,
                    json.dumps(payload or {}, ensure_ascii=False),
                    _now(),
                ),
            )

    def _ensure_event_baseline(self, rows: list[dict]) -> None:
        with sqlite3.connect(self.settings.database_path) as conn:
            self._ensure_database(conn)
            for row in rows:
                existing = conn.execute(
                    "SELECT 1 FROM dataset_events WHERE dataset_id = ? AND label_id = ? LIMIT 1",
                    (row.get("dataset_id") or "", row.get("label_id") or ""),
                ).fetchone()
                if existing:
                    continue
                conn.execute(
                    """
                    INSERT OR IGNORE INTO dataset_events (
                        dataset_id, project_type, label_id, event_type, payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.get("dataset_id") or "",
                        row.get("project_type") or "",
                        row.get("label_id") or "",
                        "catalog_baseline",
                        json.dumps({
                            "project_name": row.get("project_name"),
                            "defect_class": row.get("defect_class"),
                            "count": row.get("count"),
                            "reviewed": row.get("reviewed"),
                            "predicted": row.get("predicted"),
                            "status": row.get("status"),
                            "best_checkpoint_id": row.get("best_checkpoint_id"),
                        }, ensure_ascii=False),
                        _now(),
                    ),
                )

    def record_event(self, dataset_id: str, event_type: str, payload: dict | None = None, label_id: str | None = None) -> None:
        meta = self.get_meta(dataset_id)
        event_payload = dict(payload or {})
        if label_id:
            labels = meta.get("labels") or []
            primary = next((label for label in labels if label.get("id") == label_id), None)
            if primary:
                meta = {**meta, "labels": [primary]}
            event_payload["label_id"] = label_id
        self._record_event(event_type, meta, event_payload)

    def _sync_indexes(self) -> None:
        self._sync_catalog_table(self._all_catalog_rows())
        self._sync_category_indexes()

    def _sync_category_indexes(self) -> None:
        active: set[Path] = set()
        for meta_path in sorted(self.settings.datasets_dir.glob("*/dataset.json")):
            meta = read_json(meta_path, {})
            if not meta:
                continue
            root = self.dataset_root(meta.get("id", meta_path.parent.name))
            self.ensure_dataset_layout(root)
            for label in meta.get("labels", []) or [{"id": _slug(meta.get("defect_class", "dataset")), "name": meta.get("defect_class")}]:
                label_id = _slug(label.get("id") or label.get("name") or meta.get("defect_class") or "dataset")
                index_root = self.settings.categories_dir / label_id / meta.get("project_type", "optimization") / meta.get("id", meta_path.parent.name)
                index_root.mkdir(parents=True, exist_ok=True)
                active.add(index_root.resolve())
                write_json(index_root / "dataset-index.json", {
                    "dataset_id": meta.get("id"),
                    "project_name": meta.get("name"),
                    "defect_class": meta.get("defect_class"),
                    "project_type": meta.get("project_type", "optimization"),
                    "label": label,
                    "dataset_root": str(root),
                    "images_dir": str(root / "images"),
                    "originals_dir": str(root / "originals"),
                    "predicted_masks_dir": str(root / "masks" / "predicted"),
                    "reviewed_masks_dir": str(root / "masks" / "reviewed"),
                    "updated_at": _now(),
                })
        categories_root = self.settings.categories_dir.resolve()
        for index_file in self.settings.categories_dir.glob("*/*/*/dataset-index.json"):
            index_root = index_file.parent.resolve()
            if index_root in active:
                continue
            if str(index_root).lower().startswith(str(categories_root).lower()):
                shutil.rmtree(index_root, ignore_errors=True)

    def _catalog_row(self, meta: dict, label: dict) -> dict:
        items = meta.get("items", [])
        label_id = label.get("id")
        reviewed = 0
        predicted = 0
        formats: dict[str, int] = {}
        for item in items:
            formats[item.get("source_format", "unknown")] = formats.get(item.get("source_format", "unknown"), 0) + 1
            if item.get("latest_prediction") or item.get("predictions"):
                predicted += 1
            if label_id and (item.get("annotations", {}).get(label_id) or (label_id == "spall" and item.get("annotation_path"))):
                reviewed += 1
        best = self._best_checkpoint(meta.get("id"), label_id)
        return {
            "dataset_id": meta.get("id"),
            "project_name": meta.get("name"),
            "defect_class": meta.get("defect_class"),
            "project_type": meta.get("project_type", "optimization"),
            "label_id": label_id,
            "label_name": label.get("name") or label_id,
            "label_color": label.get("color"),
            "count": len(items),
            "reviewed": reviewed,
            "predicted": predicted,
            "status": self._dataset_status(len(items), reviewed, predicted, meta.get("project_type", "optimization")),
            "formats": formats,
            "format": ", ".join(f"{name}:{count}" for name, count in sorted(formats.items())) or "-",
            "best_pt": best.get("best_pt"),
            "best_checkpoint_id": best.get("checkpoint_id"),
            "updated_at": meta.get("updated_at"),
            "created_at": meta.get("created_at"),
        }

    def _dataset_status(self, total: int, reviewed: int, predicted: int, project_type: str) -> str:
        if total == 0:
            return "empty"
        if reviewed >= total:
            return "baseline_ready" if project_type == "generation" else "review_complete"
        if reviewed > 0:
            return "reviewing"
        if predicted > 0:
            return "predicted"
        return "imported"

    def _best_checkpoint(self, dataset_id: str | None, label_id: str | None) -> dict:
        if not dataset_id:
            return {}
        best_job = None
        for job_path in sorted(self.settings.training_jobs_dir.glob("*.json"), reverse=True):
            job = read_json(job_path, {})
            if job.get("dataset_id") != dataset_id:
                continue
            if label_id and job.get("label_id") != label_id:
                continue
            if job.get("status") != "completed":
                continue
            best_job = job
            break
        if not best_job:
            return {}
        best_path = (best_job.get("metrics") or {}).get("best_path") or best_job.get("expected_best_model_path")
        return {
            "checkpoint_id": f"run-{best_job.get('id')}-best",
            "best_pt": best_path,
        }

    def dataset_root(self, dataset_id: str) -> Path:
        return self.settings.datasets_dir / dataset_id

    def ensure_dataset_layout(self, root: Path) -> None:
        for sub in ("originals", "images", "metadata", "predictions", "annotations", "masks/predicted", "masks/reviewed", "versions"):
            (root / sub).mkdir(parents=True, exist_ok=True)

    def meta_path(self, dataset_id: str) -> Path:
        return self.dataset_root(dataset_id) / "dataset.json"

    def get_meta(self, dataset_id: str) -> dict:
        meta = read_json(self.meta_path(dataset_id), None)
        if meta is None:
            raise KeyError(f"Unknown dataset: {dataset_id}")
        return meta

    def save_meta(self, dataset_id: str, meta: dict) -> None:
        meta["updated_at"] = _now()
        write_json(self.meta_path(dataset_id), meta)
        self._sync_indexes()

    def import_images(self, dataset_id: str, source_dir: Path) -> dict:
        if not source_dir.exists() or not source_dir.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        meta = self.get_meta(dataset_id)
        root = self.dataset_root(dataset_id)
        self.ensure_dataset_layout(root)
        imported = []
        existing_names = {item["original_name"] for item in meta["items"]}
        import_record = {
            "id": f"import-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "source_dir": str(source_dir),
            "created_at": _now(),
            "imported": 0,
            "skipped": 0,
            "errors": [],
        }
        for path in sorted(source_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
                continue
            if path.name in existing_names:
                import_record["skipped"] += 1
                continue
            item_id = f"{path.stem[:48]}-{uuid.uuid4().hex[:8]}"
            original_dst = root / "originals" / f"{item_id}{path.suffix.lower()}"
            preview_dst = root / "images" / f"{item_id}.png"
            metadata_dst = root / "metadata" / f"{item_id}.json"
            importer = self._select_importer(path)
            try:
                imported_asset = importer.import_asset(path, original_dst, preview_dst, metadata_dst)
            except Exception as exc:
                import_record["errors"].append({"file": str(path), "error": str(exc)})
                continue
            item = {
                "id": item_id,
                "original_name": path.name,
                "original_path": str(original_dst),
                "image_path": str(preview_dst),
                "metadata_path": str(metadata_dst),
                "importer": imported_asset.metadata.get("importer"),
                "source_format": imported_asset.metadata.get("source_format"),
                "media_type": imported_asset.metadata.get("media_type"),
                "sha256": imported_asset.metadata.get("sha256"),
                "bit_depth": imported_asset.metadata.get("bit_depth"),
                "dtype": imported_asset.metadata.get("dtype"),
                "pixel_size": imported_asset.metadata.get("pixel_size"),
                "axes": imported_asset.metadata.get("axes", []),
                "width": imported_asset.preview_width,
                "height": imported_asset.preview_height,
                "status": "imported",
                "predictions": {},
                "latest_prediction": None,
                "annotations": {},
                "annotation_path": None,
                "created_at": _now(),
                "updated_at": _now(),
            }
            meta["items"].append(item)
            imported.append(item)
            import_record["imported"] += 1
        meta.setdefault("imports", []).append(import_record)
        self.save_meta(dataset_id, meta)
        self._record_event("import", meta, import_record)
        return {"imported": len(imported), "items": imported, "import": import_record}

    def _select_importer(self, path: Path):
        for importer in self.importers:
            if importer.supports(path):
                return importer
        raise RuntimeError(f"Unsupported file format: {path.suffix}")

    def list_items(self, dataset_id: str) -> list[dict]:
        return self.get_meta(dataset_id)["items"]

    def get_item(self, dataset_id: str, item_id: str) -> dict:
        for item in self.get_meta(dataset_id)["items"]:
            if item["id"] == item_id:
                return item
        raise KeyError(f"Unknown item: {item_id}")

    def update_item(self, dataset_id: str, item_id: str, patch: dict) -> dict:
        meta = self.get_meta(dataset_id)
        for item in meta["items"]:
            if item["id"] == item_id:
                item.update(patch)
                item["updated_at"] = _now()
                self.save_meta(dataset_id, meta)
                return item
        raise KeyError(f"Unknown item: {item_id}")

    def summary(self, dataset_id: str) -> dict:
        meta = self.get_meta(dataset_id)
        items = meta.get("items", [])
        by_status: dict[str, int] = {}
        by_format: dict[str, int] = {}
        by_importer: dict[str, int] = {}
        reviewed_by_label: dict[str, int] = {}
        for item in items:
            by_status[item.get("status", "unknown")] = by_status.get(item.get("status", "unknown"), 0) + 1
            fmt = item.get("source_format", "unknown")
            by_format[fmt] = by_format.get(fmt, 0) + 1
            importer = item.get("importer", "legacy")
            by_importer[importer] = by_importer.get(importer, 0) + 1
            for label_id, annotation in item.get("annotations", {}).items():
                if annotation.get("path"):
                    reviewed_by_label[label_id] = reviewed_by_label.get(label_id, 0) + 1
            if item.get("annotation_path") and "spall" not in item.get("annotations", {}):
                reviewed_by_label["spall"] = reviewed_by_label.get("spall", 0) + 1
        return {
            "id": meta.get("id"),
            "name": meta.get("name"),
            "defect_class": meta.get("defect_class"),
            "project_type": meta.get("project_type", "optimization"),
            "schema_version": meta.get("schema_version", "legacy"),
            "labels": meta.get("labels", []),
            "n_items": len(items),
            "by_status": by_status,
            "by_format": by_format,
            "by_importer": by_importer,
            "reviewed_by_label": reviewed_by_label,
            "n_versions": len(meta.get("versions", [])),
            "latest_version": meta.get("versions", [])[-1] if meta.get("versions") else None,
        }

    def create_snapshot(self, dataset_id: str, name: str, note: str = "") -> dict:
        meta = self.get_meta(dataset_id)
        items = meta.get("items", [])
        version_id = f"v{len(meta.get('versions', [])) + 1:04d}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        snapshot = {
            "id": version_id,
            "name": name,
            "note": note,
            "created_at": _now(),
            "n_items": len(items),
            "item_ids": [item["id"] for item in items],
            "reviewed_item_ids": [item["id"] for item in items if item.get("annotation_path") or item.get("annotations")],
            "summary": self.summary(dataset_id),
        }
        meta.setdefault("versions", []).append(snapshot)
        self.save_meta(dataset_id, meta)
        write_json(self.dataset_root(dataset_id) / "versions" / f"{version_id}.json", snapshot)
        return snapshot

    def rebuild_metadata(self, dataset_id: str) -> dict:
        meta = self.get_meta(dataset_id)
        root = self.dataset_root(dataset_id)
        updated = 0
        errors = []
        meta.setdefault("schema_version", "metalwear.dataset.v2")
        meta.setdefault("project_type", "optimization")
        meta.setdefault("imports", [])
        meta.setdefault("versions", [])
        meta.setdefault("labels", [
            {"id": "spall", "name": "Spall", "color": "#ff3728", "enabled": True},
        ])
        for item in meta.get("items", []):
            original_path = Path(item["original_path"])
            if not original_path.exists():
                errors.append({"item_id": item["id"], "error": "original file missing"})
                continue
            metadata_dst = root / "metadata" / f"{item['id']}.json"
            importer = self._select_importer(original_path)
            try:
                imported_asset = importer.import_asset(original_path, original_path, Path(item["image_path"]), metadata_dst)
            except Exception as exc:
                errors.append({"item_id": item["id"], "error": str(exc)})
                continue
            item.update({
                "metadata_path": str(metadata_dst),
                "importer": imported_asset.metadata.get("importer"),
                "source_format": imported_asset.metadata.get("source_format"),
                "media_type": imported_asset.metadata.get("media_type"),
                "sha256": imported_asset.metadata.get("sha256"),
                "bit_depth": imported_asset.metadata.get("bit_depth"),
                "dtype": imported_asset.metadata.get("dtype"),
                "pixel_size": imported_asset.metadata.get("pixel_size"),
                "axes": imported_asset.metadata.get("axes", []),
                "width": imported_asset.preview_width,
                "height": imported_asset.preview_height,
            })
            item.setdefault("annotations", {})
            updated += 1
        self.save_meta(dataset_id, meta)
        return {"updated": updated, "errors": errors, "summary": self.summary(dataset_id)}

    def delete(self, dataset_id: str) -> dict:
        meta = self.get_meta(dataset_id)
        root = self.dataset_root(dataset_id).resolve()
        datasets_root = self.settings.datasets_dir.resolve()
        if not str(root).lower().startswith(str(datasets_root).lower()) or not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")
        deleted_jobs = 0
        deleted_run_dirs = 0
        for job_path in sorted(self.settings.training_jobs_dir.glob("*.json")):
            job = read_json(job_path, {})
            if job.get("dataset_id") != dataset_id:
                continue
            run_dir = Path(job.get("output_dir") or (self.settings.run_checkpoints_dir / job_path.stem)).resolve()
            runs_root = self.settings.run_checkpoints_dir.resolve()
            if run_dir.exists() and str(run_dir).lower().startswith(str(runs_root).lower()):
                shutil.rmtree(run_dir)
                deleted_run_dirs += 1
            job_resolved = job_path.resolve()
            jobs_root = self.settings.training_jobs_dir.resolve()
            if str(job_resolved).lower().startswith(str(jobs_root).lower()):
                job_path.unlink(missing_ok=True)
                deleted_jobs += 1
        deleted_tests = 0
        test_root = (self.settings.tests_dir / dataset_id).resolve()
        tests_root = self.settings.tests_dir.resolve()
        if test_root.exists() and str(test_root).lower().startswith(str(tests_root).lower()):
            shutil.rmtree(test_root)
            deleted_tests = 1
        shutil.rmtree(root)
        self._remove_category_indexes(meta)
        self._record_event("delete", meta, {
            "deleted_training_jobs": deleted_jobs,
            "deleted_run_dirs": deleted_run_dirs,
            "deleted_model_tests": deleted_tests,
        })
        self._sync_indexes()
        return {
            "deleted": True,
            "dataset_id": dataset_id,
            "name": meta.get("name"),
            "deleted_training_jobs": deleted_jobs,
            "deleted_run_dirs": deleted_run_dirs,
            "deleted_model_tests": deleted_tests,
        }

    def _remove_category_indexes(self, meta: dict) -> None:
        categories_root = self.settings.categories_dir.resolve()
        dataset_id = meta.get("id")
        if not dataset_id:
            return
        for label in meta.get("labels", []) or [{"id": _slug(meta.get("defect_class", "dataset"))}]:
            label_id = _slug(label.get("id") or label.get("name") or meta.get("defect_class") or "dataset")
            index_root = (self.settings.categories_dir / label_id / meta.get("project_type", "optimization") / dataset_id).resolve()
            if index_root.exists() and str(index_root).lower().startswith(str(categories_root).lower()):
                shutil.rmtree(index_root, ignore_errors=True)

    def archives_root(self) -> Path:
        return self.settings.archives_dir

    def archive(self, dataset_id: str) -> dict:
        meta = self.get_meta(dataset_id)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        archive_id = f"{dataset_id}-{stamp}"
        archive_root = self.archives_root() / archive_id
        if archive_root.exists():
            raise FileExistsError(f"Archive already exists: {archive_root}")
        archive_root.mkdir(parents=True, exist_ok=True)
        dataset_copy = archive_root / "dataset"
        shutil.copytree(self.dataset_root(dataset_id), dataset_copy)
        manifest = {
            "id": archive_id,
            "dataset_id": dataset_id,
            "name": meta.get("name"),
            "project_type": meta.get("project_type", "optimization"),
            "labels": meta.get("labels", []),
            "created_at": _now(),
            "dataset_copy": str(dataset_copy),
            "note": "集中归档：包含平台内原图副本、预览图、人工 GT/mask、prediction metadata 和 dataset.json；不触碰外部原始图片路径。",
        }
        write_json(archive_root / "archive.json", manifest)
        self._record_event("archive", meta, {"archive_id": archive_id})
        return manifest

    def list_archives(self, project_type: str | None = None, label_id: str | None = None) -> list[dict]:
        archives = []
        for path in sorted(self.archives_root().glob("*/archive.json"), reverse=True):
            archive = read_json(path, {})
            if self._is_internal_archive(archive):
                continue
            if project_type and archive.get("project_type") != project_type:
                continue
            labels = archive.get("labels", [])
            if label_id and not any(label.get("id") == label_id for label in labels):
                continue
            archives.append(archive)
        return archives

    def _is_internal_archive(self, archive: dict) -> bool:
        labels = archive.get("labels", []) or []
        haystack = " ".join([
            str(archive.get("id", "")),
            str(archive.get("dataset_id", "")),
            str(archive.get("name", "")),
            " ".join(f"{label.get('id', '')} {label.get('name', '')}" for label in labels),
        ]).lower()
        return "smoke" in haystack or "blank-defect" in haystack or "blank defect" in haystack

    def restore_archive(self, archive_id: str, project_type: str | None = None) -> dict:
        archive_root = self.archives_root() / archive_id
        manifest = read_json(archive_root / "archive.json", None)
        if manifest is None:
            raise KeyError(f"Unknown dataset archive: {archive_id}")
        source_root = archive_root / "dataset"
        if not source_root.exists():
            raise FileNotFoundError(f"Archive dataset copy missing: {source_root}")
        old_meta = read_json(source_root / "dataset.json", None)
        if old_meta is None:
            raise FileNotFoundError("Archived dataset.json missing")
        new_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-restored-{_slug(old_meta.get('name', archive_id))}"
        target_root = self.dataset_root(new_id)
        shutil.copytree(source_root, target_root)
        self.ensure_dataset_layout(target_root)
        meta = read_json(target_root / "dataset.json", {})
        old_root = Path(old_meta.get("id", "")).name
        meta["id"] = new_id
        meta["name"] = f"{meta.get('name', archive_id)}（恢复）"
        if project_type:
            meta["project_type"] = project_type
        meta["restored_from_archive"] = archive_id
        meta["created_at"] = _now()
        self._rewrite_item_paths(meta, target_root)
        self.save_meta(new_id, meta)
        self._record_event("restore", meta, {"archive_id": archive_id, "old_dataset_id": old_meta.get("id")})
        return meta

    def _rewrite_item_paths(self, meta: dict, root: Path) -> None:
        for item in meta.get("items", []):
            item_id = item.get("id")
            original = next((root / "originals").glob(f"{item_id}.*"), None) if item_id else None
            image = root / "images" / f"{item_id}.png" if item_id else None
            metadata = root / "metadata" / f"{item_id}.json" if item_id else None
            if original:
                item["original_path"] = str(original)
            if image and image.exists():
                item["image_path"] = str(image)
            if metadata and metadata.exists():
                item["metadata_path"] = str(metadata)
            for label_id, annotation in item.get("annotations", {}).items():
                mask_path = root / "annotations" / label_id / f"{item_id}_review_mask.png"
                synced_mask_path = root / "masks" / "reviewed" / label_id / f"{item_id}.png"
                if mask_path.exists():
                    annotation["path"] = str(mask_path)
                    if synced_mask_path.exists():
                        annotation["synced_mask_path"] = str(synced_mask_path)
                    if label_id == "spall":
                        item["annotation_path"] = str(mask_path)
                        item["annotation"] = annotation
            for checkpoint_id, pred in item.get("predictions", {}).items():
                pred_root = root / "predictions" / checkpoint_id
                candidates = {
                    "probability_path": (pred_root / "probability" / f"{item_id}.npy", pred_root / f"{item_id}_prob.npy"),
                    "mask_path": (pred_root / "masks" / f"{item_id}.png", pred_root / f"{item_id}_mask.png"),
                    "overlay_path": (pred_root / "overlays" / f"{item_id}.png", pred_root / f"{item_id}_overlay.png"),
                }
                for key, paths in candidates.items():
                    pred_path = next((path for path in paths if path.exists()), None)
                    if pred_path:
                        pred[key] = str(pred_path)
                synced_pred_path = root / "masks" / "predicted" / checkpoint_id / f"{item_id}.png"
                if synced_pred_path.exists():
                    pred["synced_mask_path"] = str(synced_pred_path)
