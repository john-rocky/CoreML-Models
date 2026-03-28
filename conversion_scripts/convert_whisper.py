# Whisper Tiny -> CoreML conversion
# Apple provides official conversion via whisperkittools
# pip install whisperkittools
# Alternatively, use huggingface optimum:
# pip install optimum[exporters]

# Method 1: Using whisperkit (recommended)
# python -m whisperkittools.generate_model --model openai/whisper-tiny --output-dir .

# Method 2: Manual conversion
import torch
import coremltools as ct
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model_name = "openai/whisper-tiny"
model = WhisperForConditionalGeneration.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)
model.eval()

# Convert encoder
encoder = model.get_encoder()
mel_input = torch.randn(1, 80, 3000)
traced_encoder = torch.jit.trace(encoder, mel_input)

encoder_ml = ct.convert(
    traced_encoder,
    inputs=[ct.TensorType(name="mel_input", shape=(1, 80, 3000))],
    outputs=[ct.TensorType(name="encoder_output")],
    minimum_deployment_target=ct.target.iOS16,
    convert_to="mlprogram",
)
encoder_ml.save("WhisperTinyEncoder.mlpackage")
print("Saved WhisperTinyEncoder.mlpackage")

# Note: Decoder conversion requires more complex handling for autoregressive generation.
# For production use, consider using WhisperKit or Apple's pre-converted models.
