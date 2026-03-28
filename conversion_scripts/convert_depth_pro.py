# Depth Pro -> CoreML conversion
# Apple's official repo: https://github.com/apple/ml-depth-pro
# pip install depth-pro

import torch
import coremltools as ct
import depth_pro

# Load model
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Trace with dummy input
dummy = torch.randn(1, 3, 1536, 1536)

# Note: Depth Pro outputs both depth map and focal length
# For CoreML, we trace the model and convert
traced = torch.jit.trace(model, dummy)

mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(name="image", shape=(1, 3, 1536, 1536), scale=1/255.0)],
    outputs=[ct.TensorType(name="depth"), ct.TensorType(name="focallength")],
    minimum_deployment_target=ct.target.iOS16,
    convert_to="mlprogram",
)
mlmodel.save("DepthPro.mlpackage")
print("Saved DepthPro.mlpackage")
