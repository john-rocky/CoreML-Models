# Kokoro-82M -> CoreML conversion
# Pre-converted CoreML model available at: https://huggingface.co/FluidInference/kokoro-82m-coreml
# iOS Swift package: https://github.com/mlalma/kokoro-ios
#
# Manual conversion:
# pip install torch coremltools kokoro

import torch
import coremltools as ct

# Kokoro has a two-stage pipeline: Duration Predictor + Decoder
# The model uses StyleTTS2-based architecture with ISTFTNet decoder

# Download from HuggingFace
from huggingface_hub import hf_hub_download
import json

# Load the model
repo_id = "hexgrad/Kokoro-82M"
model_path = hf_hub_download(repo_id, "kokoro-v1.0.onnx")

# Convert from ONNX to CoreML
mlmodel = ct.converters.convert(
    model_path,
    minimum_deployment_target=ct.target.iOS16,
    convert_to="mlprogram",
)
mlmodel.save("Kokoro82M.mlpackage")
print("Saved Kokoro82M.mlpackage")
