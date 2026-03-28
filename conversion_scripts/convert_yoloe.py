# YOLOE-S -> CoreML conversion
# YOLOE: Real-Time Seeing Anything (ICCV 2025)
# https://github.com/THU-MIG/yoloe
# pip install ultralytics

from ultralytics import YOLO

# YOLOE-S with text prompt capability
model = YOLO("yoloe-11s-seg.pt")
model.export(format="coreml", imgsz=640, half=True)
print("Exported YOLOE-S to CoreML format")

# Alternative: Export with ONNX first then convert
# model.export(format="onnx", imgsz=640)
# Then use coremltools to convert ONNX -> CoreML
