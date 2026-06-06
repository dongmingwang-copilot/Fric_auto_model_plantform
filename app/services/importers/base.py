from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ImportedAsset:
    preview_width: int
    preview_height: int
    metadata: dict


class Importer(Protocol):
    name: str

    def supports(self, path: Path) -> bool:
        ...

    def import_asset(self, source: Path, original_dst: Path, preview_dst: Path, metadata_dst: Path) -> ImportedAsset:
        ...


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def common_file_metadata(path: Path) -> dict:
    stat = path.stat()
    return {
        "original_name": path.name,
        "source_format": path.suffix.lower().lstrip(".") or "unknown",
        "file_size_bytes": stat.st_size,
        "sha256": sha256_file(path),
    }

