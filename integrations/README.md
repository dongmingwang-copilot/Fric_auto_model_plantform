# Optional Integrations

These tools are optional. The active-learning platform does not depend on them at runtime.

## Create Environment

```powershell
conda env create -f C:\Users\TR\Desktop\Plantform_v1\integrations\conda-integrations.yml
conda activate metalwear-integrations
```

## FiftyOne

Use the platform button `生成 FiftyOne`. It writes:

- `manifest.json`
- `load_fiftyone.py`

Then run the generated loader inside the integration environment.

## Datumaro / COCO

Use `导出 COCO`. The export contains:

- `annotations.json`
- `images/`
- `masks/`

Datumaro can import COCO-style annotations from that folder.

