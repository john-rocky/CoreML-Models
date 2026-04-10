# models.json Manifest Schema

The CoreML-Models hub app fetches a single `models.json` file at launch (and
periodically) to know which models are available. This document defines the
schema.

## Top-level

```json
{
  "manifest_version": 1,
  "updated_at": "2026-04-10",
  "min_app_version": "1.0",
  "categories": [ ... ],
  "models": [ ... ]
}
```

| Field | Type | Description |
|---|---|---|
| `manifest_version` | int | Schema version. The app refuses to load manifests with a higher version than it knows about. |
| `updated_at` | string | ISO date, informational. |
| `min_app_version` | string | Hub app version needed to handle this manifest. Older app versions show "update required". |
| `categories` | array | Display order + metadata for category groupings. |
| `models` | array | The model entries. |

## Category entry

```json
{
  "id": "depth",
  "name": "Monocular Depth Estimation",
  "icon": "cube.transparent",
  "order": 7
}
```

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable identifier referenced by `Model.category_id`. |
| `name` | string | Display name shown in UI. |
| `icon` | string | SF Symbol name for the section header. |
| `order` | int | Display order in the model list (ascending). |

## Model entry

```json
{
  "id": "moge2_vitb_normal_504",
  "name": "MoGe-2 ViT-B (504×504)",
  "subtitle": "Microsoft, CVPR 2025",
  "category_id": "depth",
  "description_md": "Monocular geometry from a single image. Predicts metric depth, surface normals, and a confidence mask in one forward pass. The successor to MiDaS-style relative depth: depth comes out in real meters.\n\nBased on a DINOv2 ViT-B/14 backbone with three task heads.",
  "thumbnail_url": "https://huggingface.co/john-rocky/coreml-zoo/resolve/main/thumbnails/moge2.jpg",
  "demo": {
    "template": "depth_visualization",
    "config": {
      "input_size": 504,
      "output_keys": ["depth", "normal", "mask", "metric_scale"],
      "depth_unit": "meters"
    }
  },
  "files": [
    {
      "name": "MoGe2_ViTB_Normal_504.mlpackage",
      "url": "https://huggingface.co/john-rocky/coreml-zoo/resolve/main/moge2/MoGe2_ViTB_Normal_504.mlpackage.zip",
      "archive": "zip",
      "size_bytes": 209715200,
      "sha256": "abc123...",
      "compute_units": "all"
    }
  ],
  "requirements": {
    "min_ios": "17.0",
    "min_ram_mb": 600,
    "device_capabilities": ["arm64"]
  },
  "license": {
    "name": "MIT",
    "url": "https://github.com/microsoft/MoGe/blob/main/LICENSE"
  },
  "upstream": {
    "name": "microsoft/MoGe",
    "url": "https://github.com/microsoft/MoGe",
    "year": 2025
  },
  "credits_md": "Conversion: john-rocky\nUpstream: Wang et al., 2025"
}
```

### Field reference

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | ✓ | Stable snake_case identifier. Used as the on-disk directory name. Must match `^[a-z0-9_]+$`. |
| `name` | string | ✓ | Title shown in the model list. Keep it short; use the variant in parentheses if needed. |
| `subtitle` | string | – | Short attribution shown under the title (e.g. "Microsoft, CVPR 2025"). |
| `category_id` | string | ✓ | Must match a `categories[].id`. |
| `description_md` | string | ✓ | Long description shown on the model detail page. Markdown supported. |
| `thumbnail_url` | string | – | URL of a 512×512 (or so) preview image. Optional but strongly recommended. |
| `demo.template` | string | ✓ | One of the registered template ids — see "Demo templates" below. |
| `demo.config` | object | – | Template-specific configuration. Schema varies per template. |
| `files` | array | ✓ | List of files to download. At least one. |
| `requirements.min_ios` | string | ✓ | Minimum iOS version that can run the model (e.g. "17.0"). |
| `requirements.min_ram_mb` | int | ✓ | Approximate peak RAM needed during inference. The app refuses to load on devices with less. |
| `requirements.device_capabilities` | array | – | Reserved for future feature gating. |
| `license.name` | string | ✓ | SPDX-ish identifier (MIT, Apache-2.0, OpenRAIL-M, custom, etc.). |
| `license.url` | string | ✓ | URL to the full license text. |
| `upstream.name` | string | ✓ | Owner/repo of the upstream project. |
| `upstream.url` | string | ✓ | URL of the upstream repo or paper page. |
| `upstream.year` | int | – | First publication year. |
| `credits_md` | string | – | Extra credits shown on the detail page. |

### File entry

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | ✓ | Final filename after download/unpack. The app stores files at `Application Support/Models/{model_id}/{name}`. |
| `url` | string | ✓ | Direct HTTPS download URL. The app does no auth or token handling. |
| `archive` | string | – | One of `zip`, `tar.gz`, or omitted for raw files. The app unpacks archives in-place. |
| `size_bytes` | int | ✓ | Byte length of the downloaded file (for progress bar + free-space check). |
| `sha256` | string | ✓ | Hex-encoded SHA-256 of the downloaded file. The app refuses to use files that don't match. |
| `compute_units` | string | – | Hint passed to `MLModelConfiguration.computeUnits`. One of `all`, `cpuAndNeuralEngine`, `cpuAndGPU`, `cpuOnly`. Defaults to `all`. |
| `optional` | bool | – | If true, the model still works without this file (e.g. extra voice packs). Default false. |
| `kind` | string | – | Annotation: `model`, `vocab`, `voice`, `aux`. Default `model`. |

## Demo templates

The app ships with a fixed set of templates. Each template knows how to render
inputs/outputs and consume `demo.config`. v1 templates:

| Template id | Inputs | Outputs | Used by |
|---|---|---|---|
| `image_in_out` | Photo or camera | Single image | RMBG, DDColor, SinSR, EfficientAD |
| `image_detection` | Photo or camera | Boxes + labels | YOLOv9, YOLOv10, YOLO26 |
| `open_vocab_detection` | Photo or camera + text query | Boxes + labels | YOLO-World |
| `depth_visualization` | Photo | Depth heatmap, normal map, mask | MoGe-2 |
| `video_matting` | Video clip | Alpha-composited video | MatAnyone |
| `text_to_image` | Prompt | Image | Hyper-SD |
| `image_to_text` | Photo | Caption / answer | Florence-2 |
| `zero_shot_classify` | Photo + class list | Per-class scores | SigLIP |
| `face_compare` | Two photos | Similarity score | AdaFace |
| `face_3d` | Photo | 3D mesh viewer | 3DDFA_V2 |
| `audio_in_out` | Audio file or mic | Audio file (or multiple stems) | HTDemucs, OpenVoice, Diarization |
| `text_to_audio` | Prompt + voice/style | Audio | Kokoro, Stable Audio |
| `audio_to_score` | Audio file | MIDI / piano roll | Basic Pitch |

Each template defines its own `config` schema in code. The manifest only stores
the values, not the schema. Future templates can be added without changing the
manifest version (the app skips models whose template id it doesn't know).

## Hosting

All files are hosted on a single Hugging Face model repository:

```
https://huggingface.co/john-rocky/coreml-zoo/resolve/main/{model_id}/{name}
```

`models.json` itself lives at the root of that repo so the app can fetch it
without any other infrastructure.

## On-disk layout in the app

```
Application Support/
└── coreml-models/
    ├── manifest.json                  # cached copy of latest models.json
    ├── manifest.etag                  # etag for cheap revalidation
    └── models/
        └── moge2_vitb_normal_504/
            ├── MoGe2_ViTB_Normal_504.mlpackage
            └── .meta.json             # download timestamp, sha256, file list
```

The `Application Support` directory is excluded from iCloud backup
(`URLResourceKey.isExcludedFromBackupKey = true`) so model files don't bloat
the user's iCloud storage.

## Versioning

- The schema is forward-compatible: unknown fields are ignored by the app.
- Removing a field requires bumping `manifest_version` and `min_app_version`.
- Models can be removed from the manifest at any time; the app will keep
  already-downloaded copies until the user clears them, but won't show them
  in the catalog.
