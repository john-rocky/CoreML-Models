# YOLO26Demo

Sample iOS app for **NMS-free** YOLO object detection models (YOLO26, YOLOv10).

## Supported Models

| Model | Size | Architecture | Year |
|-------|------|-------------|------|
| [YOLO26s](https://github.com/john-rocky/CoreML-Models/releases/download/yolo-models-v1/yolo26s.mlpackage.zip) | 18 MB | End-to-end, NMS-free | 2026 |
| [YOLOv10s](https://github.com/john-rocky/CoreML-Models/releases/download/yolo-models-v1/yolov10s.mlpackage.zip) | 14 MB | Dual assignment, NMS-free | 2024 |

These models output direct predictions without requiring Non-Maximum Suppression post-processing.

## Features

- **Camera** — Real-time detection with FPS/latency stats
- **Photo** — Pick an image from library and run detection
- **Video** — Frame-by-frame detection with progress bar

## Setup

1. Download a model from the links above
2. Unzip and drag the `.mlpackage` into the Xcode project
3. Build and run on a physical device (iOS 16+)

## Output Format

Model output: `[1, 300, 6]` where each row is `[x1, y1, x2, y2, confidence, class_id]` in pixel coordinates (0-640). The app decodes these directly — no NMS needed.
