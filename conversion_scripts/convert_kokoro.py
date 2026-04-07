"""
Convert Kokoro-82M TTS to CoreML.

Strategy: split into 2 models because pred_dur creates dynamic length:
  1. KokoroPredictor: input_ids → (duration, hidden_d, t_en)
       Fixed input length (e.g. 256 phonemes max)
  2. KokoroDecoder: (en_aligned, F0_input_aligned, asr_aligned, ref_s) → audio
       Fixed output frame count (e.g. 1024 frames max)

Swift code:
  - Run Predictor → get pred_dur, d, t_en
  - Build alignment matrix from pred_dur (or use repeat_interleave)
  - Pad expanded features to max frames
  - Run Decoder
  - Trim audio to actual length (sum(pred_dur) * frame_rate)
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from kokoro import KModel
import coremltools as ct

# Patch coremltools int op (for shape ops)
from coremltools.converters.mil.frontend.torch import ops as _ct_ops
from coremltools.converters.mil import Builder as mb

def _patched_int(context, node):
    inputs = _ct_ops._get_inputs(context, node)
    x = inputs[0]
    if x.val is not None:
        val = x.val
        if isinstance(val, np.ndarray):
            val = int(val.item()) if val.ndim == 0 else int(val.flat[0])
        else:
            val = int(val)
        res = mb.const(val=np.int32(val), name=node.name)
    else:
        res = mb.cast(x=x, dtype="int32", name=node.name)
    context.add(res)

_ct_ops._TORCH_OPS_REGISTRY.register_func(_patched_int, torch_alias=["int"], override=True)

OUTPUT_DIR = os.path.expanduser("~/Downloads/CoreML-Models/conversion_scripts")
MAX_PHONEMES = 256       # Predictor input length (incl. BOS/EOS pad)
# Decoder bucket sizes (in predictor frames). Pick smallest bucket >= total
# frames at runtime, pad with zeros, then trim audio to actual length.
# Smaller padding ratio = fewer convolutional boundary artifacts.
DECODER_BUCKETS = [128, 256, 512]


# ---------- Wrapper 1: Duration predictor + features ----------
class PredictorWrapper(nn.Module):
    """Predicts duration and extracts features for alignment."""
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.bert = kmodel.bert
        self.bert_encoder = kmodel.bert_encoder
        self.predictor = kmodel.predictor
        self.text_encoder = kmodel.text_encoder

    def forward(self, input_ids, ref_s_style):
        # input_ids: [1, T] int32 (T = actual phoneme count, no padding)
        # ref_s_style: [1, 128] (second half of voice tensor)
        T = input_ids.shape[-1]
        # No padding → all positions are valid → mask is all False
        text_mask = torch.zeros(1, T, dtype=torch.bool)
        input_lengths = torch.tensor([T], dtype=torch.long)

        bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)  # [1, hidden, T]
        d = self.predictor.text_encoder(d_en, ref_s_style, input_lengths, text_mask)  # [1, T, hidden]
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)  # [1, T, max_dur]
        duration = torch.sigmoid(duration).sum(axis=-1)  # [1, T]

        # Pre-alignment features
        # d for F0 input (transposed): [1, hidden, T]
        d_for_align = d.transpose(-1, -2)
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)  # [1, channels, T]

        return duration, d_for_align, t_en


# ---------- Wrapper 2: F0/N + Decoder (with already-aligned features) ----------
class DecoderWrapper(nn.Module):
    """Generates audio from aligned features."""
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.predictor = kmodel.predictor
        self.decoder = kmodel.decoder

    def forward(self, en_aligned, asr_aligned, ref_s):
        # en_aligned: [1, hidden, frames] - aligned d
        # asr_aligned: [1, channels, frames] - aligned t_en
        # ref_s: [1, 256] full voice tensor
        s_style = ref_s[:, 128:]
        s_decoder = ref_s[:, :128]

        F0_pred, N_pred = self.predictor.F0Ntrain(en_aligned, s_style)
        audio = self.decoder(asr_aligned, F0_pred, N_pred, s_decoder).squeeze(1)
        return audio


def main():
    print("Loading Kokoro (disable_complex=True for CoreML)...")
    model = KModel(repo_id='hexgrad/Kokoro-82M', disable_complex=True).eval()

    # Test inputs
    print("\nTracing Predictor...")
    pred_wrapper = PredictorWrapper(model).eval()
    input_ids = torch.zeros(1, MAX_PHONEMES, dtype=torch.long)
    input_ids[0, 0] = 0  # BOS
    input_ids[0, 1:10] = torch.randint(1, 100, (9,))
    ref_s_style = torch.randn(1, 128)

    with torch.no_grad():
        d, d_align, t_en = pred_wrapper(input_ids, ref_s_style)
    print(f"  duration: {list(d.shape)}")
    print(f"  d_for_align: {list(d_align.shape)}")
    print(f"  t_en: {list(t_en.shape)}")

    with torch.no_grad():
        traced_pred = torch.jit.trace(pred_wrapper, (input_ids, ref_s_style))

    print("\nConverting Predictor to CoreML (flexible input length)...")
    # Flexible input length: 1..MAX_PHONEMES so LSTM doesn't see padding
    flex_len = ct.RangeDim(lower_bound=1, upper_bound=MAX_PHONEMES, default=MAX_PHONEMES)
    pred_ml = ct.convert(
        traced_pred,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, flex_len), dtype=np.int32),
            ct.TensorType(name="ref_s_style", shape=ref_s_style.shape, dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="duration"),
            ct.TensorType(name="d_for_align"),
            ct.TensorType(name="t_en"),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )
    pred_ml.save(f"{OUTPUT_DIR}/Kokoro_Predictor.mlpackage")
    print(f"  Saved Kokoro_Predictor.mlpackage")

    dec_wrapper = DecoderWrapper(model).eval()
    ref_s = torch.randn(1, 256)
    hidden_d = d_align.shape[1]
    hidden_t = t_en.shape[1]

    # Convert one fixed-shape decoder per bucket. Fixed shapes are proven to
    # work; flexible shapes hit upsample/conv shape issues with this network.
    for bucket in DECODER_BUCKETS:
        print(f"\n=== Decoder bucket: {bucket} frames ===")
        en_aligned = torch.randn(1, hidden_d, bucket)
        asr_aligned = torch.randn(1, hidden_t, bucket)

        with torch.no_grad():
            audio = dec_wrapper(en_aligned, asr_aligned, ref_s)
        print(f"  audio: {list(audio.shape)}")

        with torch.no_grad():
            traced_dec = torch.jit.trace(
                dec_wrapper, (en_aligned, asr_aligned, ref_s)
            )

        print("  Converting (FP32 — FP16 corrupts audio quality)...")
        dec_ml = ct.convert(
            traced_dec,
            inputs=[
                ct.TensorType(name="en_aligned", shape=en_aligned.shape, dtype=np.float32),
                ct.TensorType(name="asr_aligned", shape=asr_aligned.shape, dtype=np.float32),
                ct.TensorType(name="ref_s", shape=ref_s.shape, dtype=np.float32),
            ],
            outputs=[ct.TensorType(name="audio")],
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT32,
        )
        out_path = f"{OUTPUT_DIR}/Kokoro_Decoder_{bucket}.mlpackage"
        dec_ml.save(out_path)
        print(f"  Saved {os.path.basename(out_path)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
