# DWPose / RTMPose -> CoreML conversion
# DWPose uses RTMPose as backbone with distillation
# pip install torch coremltools onnx onnxruntime

import coremltools as ct
import onnx

# Download RTMPose ONNX model from:
# https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
# rtmpose-m_simcc-body7_pt-body7_420e-256x192.onnx

# For whole-body (133 keypoints):
# dwpose: rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288.onnx

onnx_path = "rtmpose-m_simcc-body7_pt-body7_420e-256x192.onnx"
onnx_model = onnx.load(onnx_path)

mlmodel = ct.converters.convert(
    onnx_model,
    inputs=[ct.ImageType(name="image", shape=(1, 3, 256, 192), scale=1/255.0)],
    minimum_deployment_target=ct.target.iOS16,
    convert_to="mlprogram",
)
mlmodel.save("DWPose.mlpackage")
print("Saved DWPose.mlpackage")
