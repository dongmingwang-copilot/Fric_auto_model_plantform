from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from app.ml.image_ops import save_preview_png
from app.services.importers.base import ImportedAsset, common_file_metadata


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class ImageImporter:
    name = "pillow-image"

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in IMAGE_EXTS

    def import_asset(self, source: Path, original_dst: Path, preview_dst: Path, metadata_dst: Path) -> ImportedAsset:
        original_dst.parent.mkdir(parents=True, exist_ok=True)
        metadata_dst.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != original_dst.resolve():
            shutil.copy2(source, original_dst)
        width, height = save_preview_png(original_dst, preview_dst)
        with Image.open(original_dst) as img:
            arr = np.array(img)
            dtype = str(arr.dtype)
            bit_depth = int(arr.dtype.itemsize * 8) if hasattr(arr.dtype, "itemsize") else None
            mode = img.mode
        metadata = {
            **common_file_metadata(original_dst),
            "importer": self.name,
            "media_type": "image",
            "width": width,
            "height": height,
            "mode": mode,
            "dtype": dtype,
            "bit_depth": bit_depth,
            "axes": [
                {"name": "y", "size": height, "unit": "pixel"},
                {"name": "x", "size": width, "unit": "pixel"},
            ],
            "pixel_size": None,
            "scientific_metadata": {},
        }
        metadata_dst.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        return ImportedAsset(preview_width=width, preview_height=height, metadata=metadata)
