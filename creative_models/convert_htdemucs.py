#!/usr/bin/env python3
"""
Convert HTDemucs v4 (Hybrid Transformer Demucs) to CoreML.

This script converts Meta's Demucs audio source separation model to CoreML format.
The model separates audio into 4 stems: drums, bass, vocals, other.

Key challenges solved:
1. STFT/iSTFT operations are NOT included in the CoreML model (must be done app-side)
2. Complex tensor operations replaced with complex-as-channels representation
3. einops rearrange operations replaced with native torch ops to avoid aten::Int issues
4. coremltools aten::Int handler monkey-patched to handle numpy ndarray scalars
5. Positional embeddings pre-computed and stored as model buffers
6. All dynamic shape operations replaced with fixed dimensions for the training segment length

Usage:
    source /tmp/coreml_env2/bin/activate
    python convert_htdemucs.py

Requirements:
    pip install demucs coremltools torch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from coremltools.converters.mil.frontend.torch import ops as torch_ops
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY
from demucs.pretrained import get_model
from demucs.transformer import create_sin_embedding, create_2d_sin_embedding
from einops import rearrange
import math


# =============================================================================
# Step 1: Monkey-patch coremltools to handle aten::Int with ndarray values
# =============================================================================
def patched_int(context, node):
    """Fixed aten::Int handler that properly converts numpy ndarrays to scalars."""
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]
    if x.can_be_folded_to_const():
        val = x.val
        if isinstance(val, np.ndarray):
            val = int(val.item()) if val.size == 1 else int(val.flat[0])
        else:
            val = int(val)
        res = mb.const(val=val, name=node.name)
    elif len(x.shape) > 0:
        x = mb.squeeze(x=x, name=node.name + "_item")
        res = mb.cast(x=x, dtype="int32", name=node.name)
    else:
        res = mb.cast(x=x, dtype="int32", name=node.name)
    context.add(res, node.name)


_TORCH_OPS_REGISTRY["int"] = patched_int
_TORCH_OPS_REGISTRY["aten::Int"] = patched_int
torch_ops._int = patched_int


# =============================================================================
# Step 2: Define the CoreML-compatible wrapper
# =============================================================================
class HTDemucsForCoreML(nn.Module):
    """
    HTDemucs neural network core with STFT/iSTFT removed.

    Takes pre-computed spectral magnitude and raw waveform as inputs.
    Outputs frequency-domain mask and time-domain signal for 4 sources.

    All einops operations replaced with native torch ops.
    All dynamic shape operations use fixed constants.
    Positional embeddings pre-computed and stored as buffers.
    """

    def __init__(self, model, pos_emb_2d, pos_emb_1d):
        super().__init__()
        # Encoder/decoder
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.tencoder = model.tencoder
        self.tdecoder = model.tdecoder
        self.freq_emb = model.freq_emb
        self.freq_emb_scale = model.freq_emb_scale

        # Channel up/down samplers
        self.channel_upsampler = model.channel_upsampler
        self.channel_downsampler = model.channel_downsampler
        self.channel_upsampler_t = model.channel_upsampler_t
        self.channel_downsampler_t = model.channel_downsampler_t

        # Cross-transformer components (flattened to avoid dynamic dispatch)
        ct = model.crosstransformer
        self.ct_norm_in = ct.norm_in
        self.ct_norm_in_t = ct.norm_in_t
        self.ct_layers = ct.layers
        self.ct_layers_t = ct.layers_t
        self.ct_weight_pos_embed = ct.weight_pos_embed

        # Pre-computed positional embeddings
        self.register_buffer("pos_emb_2d", pos_emb_2d)
        self.register_buffer("pos_emb_1d", pos_emb_1d)

    def forward(self, mag, mix):
        """
        Args:
            mag: [1, 4, 2048, 336] spectral magnitude (complex-as-channels)
            mix: [1, 2, 343980] raw stereo waveform

        Returns:
            freq_output: [1, 16, 2048, 336] frequency domain output
            time_output: [1, 8, 343980] time domain output
        """
        x = mag

        # Normalize frequency branch
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        # Normalize time branch
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        # ===== ENCODER =====
        saved = []
        saved_t = []

        for idx, encode in enumerate(self.encoder):
            inject = None
            if idx < len(self.tencoder):
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb
            saved.append(x)

        # ===== CHANNEL UPSAMPLING (fixed shapes, no einops) =====
        x = x.reshape(1, 384, 2688)
        x = self.channel_upsampler(x)
        x = x.view(1, 512, 8, 336)
        xt = self.channel_upsampler_t(xt)

        # ===== CROSS TRANSFORMER (native torch ops, no einops) =====
        # Freq: [1, 512, 8, 336] -> [1, 2688, 512]
        x = x.permute(0, 3, 2, 1).contiguous().view(1, 2688, 512)
        x = self.ct_norm_in(x)
        x = x + self.ct_weight_pos_embed * self.pos_emb_2d

        # Time: [1, 512, 1344] -> [1, 1344, 512]
        xt = xt.permute(0, 2, 1)
        xt = self.ct_norm_in_t(xt)
        xt = xt + self.ct_weight_pos_embed * self.pos_emb_1d

        # Unrolled 5 transformer layers (alternating self-attn / cross-attn)
        # Layer 0: self-attention
        x = self.ct_layers[0](x)
        xt = self.ct_layers_t[0](xt)
        # Layer 1: cross-attention
        old_x = x
        x = self.ct_layers[1](x, xt)
        xt = self.ct_layers_t[1](xt, old_x)
        # Layer 2: self-attention
        x = self.ct_layers[2](x)
        xt = self.ct_layers_t[2](xt)
        # Layer 3: cross-attention
        old_x = x
        x = self.ct_layers[3](x, xt)
        xt = self.ct_layers_t[3](xt, old_x)
        # Layer 4: self-attention
        x = self.ct_layers[4](x)
        xt = self.ct_layers_t[4](xt)

        # Freq: [1, 2688, 512] -> [1, 512, 8, 336]
        x = x.view(1, 336, 8, 512).permute(0, 3, 2, 1)
        # Time: [1, 1344, 512] -> [1, 512, 1344]
        xt = xt.permute(0, 2, 1)

        # ===== CHANNEL DOWNSAMPLING =====
        x = x.reshape(1, 512, 2688)
        x = self.channel_downsampler(x)
        x = x.view(1, 384, 8, 336)
        xt = self.channel_downsampler_t(xt)

        # ===== DECODER =====
        enc_lengths = [336, 336, 336, 336]
        enc_lengths_t = [343980, 85995, 21499, 5375]

        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            length_val = enc_lengths[3 - idx]
            x, pre = decode(x, skip, length_val)

            tdec = self.tdecoder[idx]
            length_t_val = enc_lengths_t[3 - idx]
            if tdec.empty:
                pre = pre[:, :, 0]
                xt, _ = tdec(pre, None, length_t_val)
            else:
                skip_t = saved_t.pop(-1)
                xt, _ = tdec(xt, skip_t, length_t_val)

        # ===== OUTPUT =====
        # Frequency output: denormalize and reshape
        x = x.view(1, 4, 4, 2048, 336)
        x = x * std[:, None] + mean[:, None]
        freq_out = x.reshape(1, 16, 2048, 336)

        # Time output: denormalize and reshape
        xt = xt.view(1, 4, 2, 343980)
        xt = xt * stdt[:, None] + meant[:, None]
        time_out = xt.reshape(1, 8, 343980)

        return freq_out, time_out


# =============================================================================
# Step 3: Load model, build wrapper, trace, and convert
# =============================================================================
def main():
    print("Loading HTDemucs model...")
    bag = get_model("htdemucs")
    orig_model = bag.models[0]
    orig_model.eval()

    training_length = int(orig_model.segment * orig_model.samplerate)
    print(f"Training segment: {training_length} samples ({training_length/44100:.2f}s)")

    # Pre-compute positional embeddings
    FREQ_SEQ = 2688  # 336 time frames * 8 freq bins at bottleneck
    TIME_SEQ = 1344

    pos_emb_2d = create_2d_sin_embedding(
        512, 8, 336, "cpu", orig_model.crosstransformer.max_period
    )
    pos_emb_2d = rearrange(pos_emb_2d, "b c fr t1 -> b (t1 fr) c")

    pos_emb_1d = create_sin_embedding(
        TIME_SEQ, 512, shift=0, device="cpu",
        max_period=orig_model.crosstransformer.max_period
    )
    pos_emb_1d = rearrange(pos_emb_1d, "t2 b c -> b t2 c")

    # Build wrapper
    wrapper = HTDemucsForCoreML(orig_model, pos_emb_2d, pos_emb_1d)
    wrapper.eval()

    # Prepare test inputs
    mix_test = torch.randn(1, 2, training_length)
    with torch.no_grad():
        z_test = orig_model._spec(mix_test)
        mag_test = orig_model._magnitude(z_test)

    # Verify wrapper matches original
    print("Verifying wrapper correctness...")
    with torch.no_grad():
        freq_out, time_out = wrapper(mag_test, mix_test)
        orig_out = orig_model(mix_test)

        freq_reshaped = freq_out.reshape(1, 4, 4, 2048, 336)
        zout = freq_reshaped.view(1, 4, 2, 2, 2048, 336).permute(0, 1, 2, 4, 5, 3)
        zout = torch.view_as_complex(zout.contiguous())
        freq_audio = orig_model._ispec(zout, training_length)
        time_reshaped = time_out.reshape(1, 4, 2, training_length)
        reconstructed = freq_audio + time_reshaped

        max_diff = (reconstructed - orig_out).abs().max().item()
        print(f"  Max diff vs original: {max_diff:.8f}")

    # Trace
    print("Tracing model...")
    traced = torch.jit.trace(wrapper, (mag_test, mix_test), check_trace=False)

    # Convert to CoreML
    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="spectral_magnitude", shape=(1, 4, 2048, 336)),
            ct.TensorType(name="audio_waveform", shape=(1, 2, 343980)),
        ],
        outputs=[
            ct.TensorType(name="freq_output"),
            ct.TensorType(name="time_output"),
        ],
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram",
    )

    # Add metadata
    mlmodel.author = "Meta Research (Demucs)"
    mlmodel.license = "MIT License"
    mlmodel.short_description = (
        "HTDemucs audio source separation - neural network core. "
        "Separates audio into 4 stems: drums, bass, vocals, other. "
        "Requires app-side STFT/iSTFT processing."
    )
    mlmodel.version = "4.0"

    mlmodel.input_description["spectral_magnitude"] = (
        "STFT magnitude with complex-as-channels [1, 4, 2048, 336]. "
        "4 channels = 2 stereo x 2 (real, imag). "
        "nfft=4096, hop=1024, hann window, normalized."
    )
    mlmodel.input_description["audio_waveform"] = (
        "Raw stereo waveform [1, 2, 343980] at 44100 Hz (~7.8s segment)."
    )
    mlmodel.output_description["freq_output"] = (
        "Frequency domain output [1, 16, 2048, 336]. "
        "16 = 4 sources x 2 stereo x 2 real/imag. Apply iSTFT app-side."
    )
    mlmodel.output_description["time_output"] = (
        "Time domain output [1, 8, 343980]. "
        "8 = 4 sources x 2 stereo. Add to iSTFT(freq_output) for final audio."
    )

    output_path = "/Users/majimadaisuke/Downloads/CoreML-Models/creative_models/HTDemucs_SourceSeparation.mlpackage"
    mlmodel.save(output_path)
    print(f"Saved to {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
