# Depth Anything V2 Small -> CoreML conversion
# pip install torch torchvision coremltools transformers
import torch
import coremltools as ct
from transformers import AutoModelForDepthEstimation, AutoImageProcessor

model_name = "depth-anything/Depth-Anything-V2-Small-hf"
model = AutoModelForDepthEstimation.from_pretrained(model_name)
model.eval()

# Trace
dummy = torch.randn(1, 3, 518, 518)
traced = torch.jit.trace(model, dummy)

# Convert
mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(name="image", shape=(1, 3, 518, 518), scale=1/255.0, bias=[0, 0, 0])],
    outputs=[ct.TensorType(name="depth")],
    minimum_deployment_target=ct.target.iOS16,
    convert_to="mlprogram",
)
mlmodel.save("DepthAnythingV2Small.mlpackage")
print("Saved DepthAnythingV2Small.mlpackage")
