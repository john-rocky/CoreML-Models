# YOLOWorldDemo

Open-vocabulary object detection on iOS using [YOLO-World](https://github.com/AILab-CVC/YOLO-World) + CLIP.

Type any text — "person", "red car", "coffee cup" — and detect it in real-time camera, photos, or videos. No fixed class list.

## Architecture

```
Text Input ──→ CLIP Text Encoder ──→ txt_feats [1,80,512]
                                          │
Camera/Image ──→ YOLO-World Detector ─────┤──→ boxes [1,4,8400]
                                          └──→ scores [1,80,8400] (sigmoid-calibrated)
                                                  │
                                              NMS + Filter ──→ Bounding Boxes
```

The CoreML detector includes the full BNContrastiveHead scoring pipeline internally. Scores are pre-computed — no external parameter files needed.

## Models

| Model | Size | Description |
|-------|------|-------------|
| `yoloworld_detector.mlpackage` | 25 MB | YOLO-World V2-S visual detector |
| `clip_text_encoder.mlpackage` | 121 MB | CLIP ViT-B/32 text encoder |
| `clip_vocab.json` | 1.6 MB | BPE vocabulary for tokenizer |

## Features

- **Camera**: Real-time open-vocabulary detection
- **Photo**: Pick from library, detect with any text query
- **Video**: Pick a video, detect frame-by-frame with overlay
- **Open-vocabulary**: Up to 80 simultaneous queries, any text

## Requirements

- iOS 16.0+
- Xcode 15.0+
- Physical device (camera + Neural Engine)

## Quick Start

1. Open `YOLOWorldDemo.xcodeproj` in Xcode
2. Select your development team
3. Build and run on a physical device

Models are pre-bundled. No additional setup required.

## Re-converting Models (Optional)

To convert with a different model size (m/l/x):

```bash
pip install ultralytics open_clip_torch coremltools torch==2.7.0
python convert_models.py --size l
```

Then replace the `.mlpackage` files in the Xcode project.

## Usage

1. Enter comma-separated object names in the text field (e.g., `person, dog, car`)
2. Tap the search button or press return
3. Switch between Camera / Photo / Video modes with the bottom buttons
4. In Photo/Video mode, tap the green (+) button to pick from library
