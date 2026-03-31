# YOLOv9Demo

Sample iOS app for **NMS-equipped** YOLO object detection models (YOLOv9, YOLO11).

## Supported Models

| Model | Size | Architecture | Year |
|-------|------|-------------|------|
| [YOLOv9s](https://github.com/john-rocky/CoreML-Models/releases/download/yolo-models-v1/yolov9s.mlpackage.zip) | 14 MB | PGI + GELAN, CoreML NMS pipeline | 2024 |
| [YOLO11s](https://github.com/john-rocky/CoreML-Models/releases/download/yolo-models-v1/yolo11s.mlpackage.zip) | 18 MB | Improved backbone/neck, CoreML NMS pipeline | 2024 |

These models include a built-in CoreML NMS pipeline and output `VNRecognizedObjectObservation` via the Vision framework.

## Features

- **Camera** — Real-time detection with FPS/latency stats
- **Photo** — Pick an image from library and run detection
- **Video** — Frame-by-frame detection with progress bar

## Setup

1. Download a model from the links above
2. Unzip and drag the `.mlpackage` into the Xcode project
3. Build and run on a physical device (iOS 16+)

## Output Format

Model output: `Confidence (0 x 80)` + `Coordinates (0 x 4)` via CoreML NMS pipeline. The Vision framework returns `VNRecognizedObjectObservation` with bounding boxes and class labels directly.
