"""
Convert YOLOv10-N (Nano) to CoreML format.

Requirements:
    pip install ultralytics

The exported model will be saved alongside the .pt weights as
yolov10n.mlpackage.  Drag it into the Xcode project so the compiler
produces the bundled .mlmodelc at build time.

Usage:
    python convert_yolov10.py
"""

from ultralytics import YOLO

# Download (if needed) and load the pretrained YOLOv10-N weights
model = YOLO("yolov10n.pt")

# Export to CoreML
# - imgsz : input resolution expected by the model
# - half  : use float16 for smaller model size on device
# - nms   : disable built-in NMS (YOLOv10 is NMS-free by design)
model.export(format="coreml", imgsz=640, half=True, nms=False)

print("CoreML conversion complete. Look for yolov10n.mlpackage")
