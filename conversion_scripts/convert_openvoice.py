"""
Convert OpenVoice V2 voice conversion models to CoreML.

Usage:
    python3 convert_openvoice.py

Output:
    sample_apps/OpenVoiceDemo/OpenVoiceDemo/OpenVoice_SpeakerEncoder.mlpackage
    sample_apps/OpenVoiceDemo/OpenVoiceDemo/OpenVoice_VoiceConverter.mlpackage

Model: OpenVoice V2 (MyShell AI)
  - SpeakerEncoder: mel spectrogram → 256-dim speaker embedding
  - VoiceConverter: spectrogram + src/tgt speaker embeddings → waveform
  - License: MIT
  - Repo: https://github.com/myshell-ai/OpenVoice
"""

import sys
sys.path.insert(0, '/tmp/openvoice_repo')

import torch
import torch.nn as nn
import json
import os
import coremltools as ct
from openvoice.models import SynthesizerTrn
from openvoice import utils

HF_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--myshell-ai--OpenVoiceV2/snapshots/"
)
# Find snapshot dir
snapshot = next(os.listdir(HF_DIR).__iter__())
HF_DIR = os.path.join(HF_DIR, snapshot)

CONVERTER_DIR = os.path.join(HF_DIR, "converter")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_apps", "OpenVoiceDemo", "OpenVoiceDemo")


def load_model():
    """Load the OpenVoice V2 converter model."""
    with open(os.path.join(CONVERTER_DIR, "config.json")) as f:
        config = json.load(f)

    hps = utils.HParams(**config)
    m = hps.model
    model = SynthesizerTrn(
        n_vocab=0,
        spec_channels=hps.data.filter_length // 2 + 1,  # 513
        inter_channels=m.inter_channels,
        hidden_channels=m.hidden_channels,
        filter_channels=m.filter_channels,
        n_heads=m.n_heads,
        n_layers=m.n_layers,
        kernel_size=m.kernel_size,
        p_dropout=m.p_dropout,
        resblock=m.resblock,
        resblock_kernel_sizes=m.resblock_kernel_sizes,
        resblock_dilation_sizes=m.resblock_dilation_sizes,
        upsample_rates=m.upsample_rates,
        upsample_initial_channel=m.upsample_initial_channel,
        upsample_kernel_sizes=m.upsample_kernel_sizes,
        n_speakers=0,
        gin_channels=m.gin_channels,
        zero_g=m.zero_g,
    )

    ckpt = torch.load(os.path.join(CONVERTER_DIR, "checkpoint.pth"),
                       map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Remove weight_norm for export
    model.dec.remove_weight_norm()
    for layer in model.flow.flows:
        if hasattr(layer, 'remove_weight_norm'):
            layer.remove_weight_norm()
    for layer in model.enc_q.enc.in_layers:
        torch.nn.utils.remove_weight_norm(layer)

    return model, hps


class SpeakerEncoderWrapper(nn.Module):
    """Wraps ref_enc to extract speaker embedding from mel spectrogram."""
    def __init__(self, ref_enc):
        super().__init__()
        self.ref_enc = ref_enc

    def forward(self, spec_t):
        # spec_t: [1, T, 513] (transposed spectrogram)
        se = self.ref_enc(spec_t)  # [1, 256]
        return se.unsqueeze(-1)  # [1, 256, 1]


class VoiceConverterWrapper(nn.Module):
    """Wraps enc_q + flow + dec for voice conversion."""
    def __init__(self, enc_q, flow, dec, zero_g=True):
        super().__init__()
        self.enc_q = enc_q
        self.flow = flow
        self.dec = dec
        self.zero_g = zero_g

    def forward(self, spec, spec_lengths, src_se, tgt_se):
        # spec: [1, 513, T]
        # spec_lengths: [1] (int, actual T value)
        # src_se, tgt_se: [1, 256, 1]

        g_src = src_se
        if self.zero_g:
            g_src_enc = torch.zeros_like(src_se)
        else:
            g_src_enc = src_se

        # Encode
        z, m_q, logs_q, mask = self.enc_q(spec, spec_lengths, g=g_src_enc, tau=0.3)

        # Flow: source → target
        z_p = self.flow(z, mask, g=src_se)
        z_hat = self.flow(z_p, mask, g=tgt_se, reverse=True)

        # Decode
        if self.zero_g:
            g_dec = torch.zeros_like(tgt_se)
        else:
            g_dec = tgt_se

        audio = self.dec(z_hat * mask, g=g_dec)
        return audio


def convert_speaker_encoder(model):
    print("\n=== Converting SpeakerEncoder ===")
    wrapper = SpeakerEncoderWrapper(model.ref_enc)
    wrapper.eval()

    # Input: [1, T, 513] - T is variable
    dummy = torch.randn(1, 100, 513)
    with torch.no_grad():
        out = wrapper(dummy)
    print(f"Output shape: {out.shape}")

    traced = torch.jit.trace(wrapper, dummy)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="spectrogram", shape=ct.Shape(
            shape=(1, ct.RangeDim(lower_bound=10, upper_bound=1000, default=100), 513)
        ))],
        outputs=[ct.TensorType(name="speaker_embedding")],
        minimum_deployment_target=ct.target.iOS16,
    )

    mlmodel.author = "CoreML-Models"
    mlmodel.short_description = "OpenVoice V2 Speaker Encoder: extracts 256-dim speaker embedding from spectrogram."
    mlmodel.license = "MIT"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "OpenVoice_SpeakerEncoder.mlpackage")
    mlmodel.save(path)
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fns in os.walk(path) for f in fns) / 1e6
    print(f"Saved to {path} ({size:.1f} MB)")


def convert_voice_converter(model, hps):
    print("\n=== Converting VoiceConverter ===")
    zero_g = getattr(hps.model, 'zero_g', True)
    wrapper = VoiceConverterWrapper(model.enc_q, model.flow, model.dec, zero_g=zero_g)
    wrapper.eval()

    T = 100
    dummy_spec = torch.randn(1, 513, T)
    dummy_lengths = torch.tensor([T], dtype=torch.long)
    dummy_src_se = torch.randn(1, 256, 1)
    dummy_tgt_se = torch.randn(1, 256, 1)

    with torch.no_grad():
        out = wrapper(dummy_spec, dummy_lengths, dummy_src_se, dummy_tgt_se)
    print(f"Output shape: {out.shape}")

    print("Tracing model...")
    traced = torch.jit.trace(wrapper, (dummy_spec, dummy_lengths, dummy_src_se, dummy_tgt_se))

    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="spectrogram", shape=ct.Shape(
                shape=(1, 513, ct.RangeDim(lower_bound=10, upper_bound=1000, default=100))
            )),
            ct.TensorType(name="spec_lengths", shape=(1,)),
            ct.TensorType(name="source_speaker", shape=(1, 256, 1)),
            ct.TensorType(name="target_speaker", shape=(1, 256, 1)),
        ],
        outputs=[ct.TensorType(name="audio")],
        minimum_deployment_target=ct.target.iOS16,
    )

    mlmodel.author = "CoreML-Models"
    mlmodel.short_description = "OpenVoice V2 Voice Converter: converts voice from source to target speaker."
    mlmodel.license = "MIT"

    path = os.path.join(OUTPUT_DIR, "OpenVoice_VoiceConverter.mlpackage")
    mlmodel.save(path)
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fns in os.walk(path) for f in fns) / 1e6
    print(f"Saved to {path} ({size:.1f} MB)")


def main():
    model, hps = load_model()
    convert_speaker_encoder(model)
    convert_voice_converter(model, hps)
    print("\nDone!")


if __name__ == "__main__":
    main()
