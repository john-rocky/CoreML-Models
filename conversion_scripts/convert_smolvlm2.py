# SmolVLM2-500M -> CoreML conversion
# pip install torch coremltools transformers accelerate

import torch
import coremltools as ct
from transformers import AutoProcessor, AutoModelForVision2Seq

model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.float32)
model.eval()

# Note: VLM conversion to CoreML is complex due to autoregressive generation.
# For production use, consider:
# 1. Export vision encoder separately
# 2. Export language model separately
# 3. Use MLX Swift for on-device inference (proven to work on iPhone)
#
# Vision Encoder conversion:
vision_encoder = model.model.vision_model
dummy_pixel = torch.randn(1, 3, 384, 384)
traced_vision = torch.jit.trace(vision_encoder, dummy_pixel)

vision_ml = ct.convert(
    traced_vision,
    inputs=[ct.ImageType(name="pixel_values", shape=(1, 3, 384, 384), scale=1/255.0)],
    outputs=[ct.TensorType(name="image_features")],
    minimum_deployment_target=ct.target.iOS16,
    convert_to="mlprogram",
)
vision_ml.save("SmolVLM2_VisionEncoder.mlpackage")
print("Saved SmolVLM2_VisionEncoder.mlpackage")

# For the full model, consider using GGUF format with llama.cpp or MLX Swift
# GGUF models available at: https://huggingface.co/ggml-org/SmolVLM2-500M-Video-Instruct-GGUF
