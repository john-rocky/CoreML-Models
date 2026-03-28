# BiRefNet -> CoreML conversion
# pip install torch torchvision coremltools transformers
import torch
import coremltools as ct
from transformers import AutoModelForImageSegmentation

model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
model.eval()

dummy = torch.randn(1, 3, 1024, 1024)
traced = torch.jit.trace(model, dummy)

mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(name="image", shape=(1, 3, 1024, 1024), scale=1/255.0)],
    outputs=[ct.TensorType(name="mask")],
    minimum_deployment_target=ct.target.iOS16,
    convert_to="mlprogram",
)
mlmodel.save("BiRefNet.mlpackage")
