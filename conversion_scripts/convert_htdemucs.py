# HTDemucs -> CoreML conversion
# pip install torch torchaudio coremltools demucs
#
# Converts the HTDemucs source separation model to CoreML.
# The model separates audio into 4 stems: drums, bass, vocals, other.
#
# Architecture: Hybrid Transformer Demucs
# - Frequency branch: processes STFT (complex as real/imag channels)
# - Time branch: processes raw waveform
# Both branches produce output that should be combined for best quality.
#
# NOTE: Use compute_precision=ct.precision.FLOAT32 to avoid Float16 overflow
# in the frequency branch (intermediate values exceed ±65504).

import torch
import coremltools as ct

# Load pretrained HTDemucs
from demucs.pretrained import get_model

model = get_model("htdemucs")
model.eval()

# Model parameters
segment_samples = int(model.segment * model.samplerate)  # ~343980 samples at 44.1kHz
n_fft = model.nfft          # 4096
hop_length = n_fft // 4     # 1024
n_freq = n_fft // 2         # 2048
n_frames = segment_samples // hop_length  # 336

print(f"segment_samples={segment_samples}, n_fft={n_fft}, hop={hop_length}, n_freq={n_freq}, n_frames={n_frames}")

# Create dummy inputs matching the model's expected format
# spectral_magnitude: complex STFT as real/imag channels [1, 4, n_freq, n_frames]
#   torch.cat([z.real, z.imag], dim=1) where z is [batch, 2, n_freq, n_frames] complex
dummy_spectral = torch.randn(1, 4, n_freq, n_frames)
# audio_waveform: raw stereo waveform [1, 2, segment_samples]
dummy_waveform = torch.randn(1, 2, segment_samples)

# Trace the model's core forward pass
# HTDemucs forward takes both spectral and waveform inputs
class HTDemucsWrapper(torch.nn.Module):
    """Wrapper to expose the hybrid encoder-decoder as a single forward pass."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, spectral_magnitude, audio_waveform):
        # The model internally splits spectral into real/imag and processes both branches
        # Reconstruct the complex STFT from real/imag channels
        B = spectral_magnitude.shape[0]
        # spectral_magnitude: [B, 4, F, T] -> [B, 2, F, T] complex
        half = spectral_magnitude.shape[1] // 2  # 2
        z_real = spectral_magnitude[:, :half]  # [B, 2, F, T]
        z_imag = spectral_magnitude[:, half:]  # [B, 2, F, T]
        z = torch.complex(z_real, z_imag)  # [B, 2, F, T] complex

        # Run the model's internal processing
        # The model has: frequency encoder, time encoder, cross-attention, decoders
        length = audio_waveform.shape[-1]

        # Use model's internal _spec and _ispec methods if available,
        # otherwise use the full forward pass
        x = audio_waveform

        # Encode frequency branch
        freq_encoded = self.model._domain_forward("freq", z)
        # Encode time branch
        time_encoded = self.model._domain_forward("time", x)

        # Cross-attention and decoder
        freq_out, time_out = self.model._decode(freq_encoded, time_encoded, length)

        # freq_out: [B, S*2C, F, T] where S=4 sources, C=2 stereo
        # Convert complex output to real/imag channels
        # freq_out is complex [B, S, 2, F, T]
        freq_real = freq_out.real  # [B, sources*2, F, T]
        freq_imag = freq_out.imag
        freq_output = torch.cat([freq_real, freq_imag], dim=2)  # or appropriate dim

        return freq_out, time_out

# Alternative simpler approach: trace the full model
class HTDemucsSimpleWrapper(torch.nn.Module):
    """Simple wrapper that takes spectral + waveform and returns separated stems."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, spectral_magnitude, audio_waveform):
        # Reconstruct complex STFT
        half = spectral_magnitude.shape[1] // 2
        z_real = spectral_magnitude[:, :half]
        z_imag = spectral_magnitude[:, half:]
        z = torch.complex(z_real, z_imag)

        length = audio_waveform.shape[-1]

        # Use the model's hybrid forward
        # freq_out: complex [B, S, C, F, T], time_out: [B, S, C, T]
        freq_out, time_out = self.model.forward_hybrid(z, audio_waveform, length)

        # Convert freq_out complex to real channels
        # freq_out shape: [B, 4, 2, n_freq, n_frames] complex
        # -> [B, 16, n_freq, n_frames] as real/imag interleaved per source
        S = freq_out.shape[1]  # 4 sources
        C = freq_out.shape[2]  # 2 channels
        freq_real = freq_out.real.reshape(1, S * C, n_freq, n_frames)
        freq_imag = freq_out.imag.reshape(1, S * C, n_freq, n_frames)
        freq_result = torch.cat([freq_real, freq_imag], dim=1)  # [1, 16, F, T]

        # time_out shape: [B, 4, 2, T] -> [1, 8, T]
        time_result = time_out.reshape(1, S * C, -1)

        return freq_result, time_result

wrapper = HTDemucsSimpleWrapper(model)
wrapper.eval()

# Try tracing
print("Tracing model...")
with torch.no_grad():
    traced = torch.jit.trace(wrapper, (dummy_spectral, dummy_waveform))

print("Converting to CoreML (Float32)...")
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(
            name="spectral_magnitude",
            shape=(1, 4, n_freq, n_frames),
        ),
        ct.TensorType(
            name="audio_waveform",
            shape=(1, 2, segment_samples),
        ),
    ],
    outputs=[
        ct.TensorType(name="freq_output"),
        ct.TensorType(name="time_output"),
    ],
    minimum_deployment_target=ct.target.iOS16,
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT32,  # Prevent Float16 overflow
)

mlmodel.author = "Meta Research (Demucs)"
mlmodel.license = "MIT License"
mlmodel.short_description = (
    "HTDemucs audio source separation - separates audio into 4 stems: "
    "drums, bass, vocals, other. Requires app-side STFT/iSTFT processing."
)

mlmodel.save("HTDemucs_SourceSeparation_F32.mlpackage")
print("Saved HTDemucs_SourceSeparation_F32.mlpackage")
print("Model uses Float32 to prevent frequency branch overflow.")
