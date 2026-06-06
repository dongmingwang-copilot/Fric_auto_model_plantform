from __future__ import annotations

from pathlib import Path

from app.services.importers.base import ImportedAsset
from app.services.importers.image_importer import ImageImporter


SCIENTIFIC_EXTS = {".dm3", ".dm4", ".emd", ".ser", ".emi", ".h5", ".hdf5", ".zarr"}


class ScientificImageImporter:
    """Optional hook for HyperSpy/RosettaSciIO-style scientific microscopy files.

    The current MVP keeps this adapter non-invasive: common image formats still use
    Pillow, while unsupported scientific containers fail with a clear message until
    the dependency is installed and mapped.
    """

    name = "scientific-microscopy"

    def __init__(self):
        self._image_fallback = ImageImporter()

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in SCIENTIFIC_EXTS

    def import_asset(self, source: Path, original_dst: Path, preview_dst: Path, metadata_dst: Path) -> ImportedAsset:
        try:
            import hyperspy.api as hs  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "该科学显微格式需要安装 HyperSpy/RosettaSciIO 导入插件后才能读取。"
            ) from exc

        signal = hs.load(str(source), lazy=True)
        data = signal.data.compute() if hasattr(signal.data, "compute") else signal.data
        if data.ndim > 2:
            data = data.reshape((-1,) + data.shape[-2:])[0]

        import json
        import shutil
        import numpy as np
        from PIL import Image
        from app.ml.image_ops import save_preview_png
        from app.services.importers.base import common_file_metadata

        original_dst.parent.mkdir(parents=True, exist_ok=True)
        metadata_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, original_dst)
        arr = np.asarray(data, dtype=np.float32)
        lo, hi = np.percentile(arr, [0.5, 99.5])
        preview = ((arr - lo) / max(hi - lo, 1e-6) * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(preview, mode="L").save(preview_dst)
        width, height = save_preview_png(preview_dst, preview_dst)

        axes = []
        for axis in signal.axes_manager:
            axes.append({
                "name": axis.name,
                "size": int(axis.size),
                "scale": float(axis.scale),
                "unit": axis.units,
            })
        metadata = {
            **common_file_metadata(original_dst),
            "importer": self.name,
            "media_type": "scientific_image",
            "width": width,
            "height": height,
            "dtype": str(np.asarray(data).dtype),
            "bit_depth": int(np.asarray(data).dtype.itemsize * 8),
            "axes": axes,
            "pixel_size": None,
            "scientific_metadata": {
                "metadata": signal.metadata.as_dictionary() if hasattr(signal.metadata, "as_dictionary") else {},
                "original_metadata": signal.original_metadata.as_dictionary() if hasattr(signal.original_metadata, "as_dictionary") else {},
            },
        }
        metadata_dst.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        return ImportedAsset(preview_width=width, preview_height=height, metadata=metadata)

