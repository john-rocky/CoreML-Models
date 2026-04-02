# HTDemucs -> CoreML conversion
# pip install torch torchaudio coremltools demucs
#
# The model takes raw stereo audio and outputs 4 separated stems directly.
# All STFT/iSTFT/normalization is handled internally by the model.
#
# Input:  mix [1, 2, 343980] - stereo audio at 44100Hz (~7.8s)
# Output: sources [1, 4, 2, 343980] - 4 stems (drums, bass, other, vocals), stereo
#
# Uses Float32 to prevent overflow in the frequency branch.

import torch
import coremltools as ct
from demucs.pretrained import get_model

# Load HTDemucs
bag = get_model("htdemucs")
model = bag.models[0]
model.eval()

segment_samples = int(model.segment * model.samplerate)  # 343980
print(f"sources: {model.sources}")
print(f"segment_samples: {segment_samples}")
print(f"samplerate: {model.samplerate}")

# Wrapper to flatten output from [1,4,2,T] to [1,8,T] for CoreML compatibility
class HTDemucsExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mix):
        # mix: [1, 2, T]
        # output: [1, 4, 2, T] -> [1, 8, T]
        x = self.model(mix)
        B, S, C, T = x.shape
        return x.reshape(B, S * C, T)

wrapper = HTDemucsExport(model)
wrapper.eval()

# Export via ONNX to avoid coremltools int op conversion bug
print("Exporting to ONNX...")
dummy = torch.randn(1, 2, segment_samples)
onnx_path = "HTDemucs_F32.onnx"

with torch.no_grad():
    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["mix"],
        output_names=["sources"],
        opset_version=17,
        do_constant_folding=True,
    )
print(f"Saved ONNX: {onnx_path}")

# Convert ONNX to CoreML with Float32
print("Converting ONNX to CoreML (Float32)...")
mlmodel = ct.convert(
    onnx_path,
    inputs=[
        ct.TensorType(
            name="mix",
            shape=(1, 2, segment_samples),
        ),
    ],
    outputs=[
        ct.TensorType(name="sources"),
    ],
    minimum_deployment_target=ct.target.iOS16,
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT32,
)

mlmodel.author = "Meta Research (Demucs)"
mlmodel.license = "MIT License"
mlmodel.short_description = (
    "HTDemucs audio source separation. Input: stereo mix [1,2,343980] at 44.1kHz. "
    "Output: [1,8,343980] = 4 stems x 2ch. Order: drums, bass, other, vocals."
)
mlmodel.input_description["mix"] = "Stereo audio waveform [1, 2, 343980] at 44100 Hz (~7.8 seconds)"
mlmodel.output_description["sources"] = (
    "Separated stems [1, 8, 343980]. 8 channels = 4 sources x 2 stereo. "
    "Source order: drums(0,1), bass(2,3), other(4,5), vocals(6,7)"
)

mlmodel.save("HTDemucs_F32.mlpackage")
print("Saved HTDemucs_F32.mlpackage")
