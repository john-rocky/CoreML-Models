# YOLO26Demo

Sample iOS app for **NMS-free** YOLO object detection models (YOLO26, YOLOv10), with on-device **multi-object tracking (ByteTrack)**.

## Supported Models

| Model | Size | Architecture | Year |
|-------|------|-------------|------|
| [YOLO26s](https://github.com/john-rocky/CoreML-Models/releases/download/yolo-models-v1/yolo26s.mlpackage.zip) | 18 MB | End-to-end, NMS-free | 2026 |
| [YOLOv10s](https://github.com/john-rocky/CoreML-Models/releases/download/yolo-models-v1/yolov10s.mlpackage.zip) | 14 MB | Dual assignment, NMS-free | 2024 |

These models output direct predictions without requiring Non-Maximum Suppression post-processing.

## Features

- **Camera** — Real-time detection + tracking with FPS/latency stats
- **Photo** — Pick an image from library and run detection
- **Video** — Frame-by-frame detection + tracking with progress bar
- **Track toggle** — Tap `Track` / `Raw` in Camera or Video to switch between tracked (persistent IDs, smoothed boxes) and raw detections

## Tracking

Tracking is implemented in pure Swift (`Tracker.swift`) using **ByteTrack**
(Zhang et al., ECCV 2022, [arxiv 2110.06864](https://arxiv.org/abs/2110.06864))
— currently the mobile sweet-spot for multi-object tracking:

- **No appearance network.** Pure motion (per-track 8D constant-velocity Kalman filter) + IoU association. No second neural net on the ANE/GPU.
- **Two-stage association.** High-confidence detections are matched first; low-confidence detections are then used to rescue tracks about to be lost, which keeps IDs stable through motion blur and brief occlusions.
- **Class-aware.** Only detections of the same class can inherit a track.
- **Track lifecycle.** Lost tracks survive 30 frames before being dropped; new tentative tracks need a second-frame confirmation before being drawn.

Each tracked object gets a persistent color (indexed by track ID) and its box label shows `#<id> <class> <conf>%`. Track IDs restart when tracking is toggled or when the Camera view re-appears.

## Setup

1. Download a model from the links above
2. Unzip and drag the `.mlpackage` into the Xcode project
3. Build and run on a physical device (iOS 16+)

## Output Format

Model output: `[1, 300, 6]` where each row is `[x1, y1, x2, y2, confidence, class_id]` in pixel coordinates (0-640). The app decodes these directly — no NMS needed.
