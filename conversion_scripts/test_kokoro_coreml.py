"""
Test Kokoro CoreML pipeline end-to-end.

Pipeline:
  1. Run Predictor (CoreML, flexible input length)
  2. Compute durations → expand features via repeat_interleave
  3. Pick smallest decoder bucket >= total_frames; pad to bucket size
  4. Run Decoder bucket (CoreML)
  5. Trim audio to actual length

Compares against PyTorch reference.
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import coremltools as ct
import soundfile as sf
from kokoro import KModel, KPipeline

OUTPUT_DIR = os.path.expanduser("~/Downloads/CoreML-Models/conversion_scripts")
DECODER_BUCKETS = [64, 128, 256, 384, 512]
SAMPLE_RATE = 24000


def expand_features(d_for_align, t_en, pred_dur):
    """Repeat features per duration. d_for_align: [1,H,T], t_en: [1,C,T], pred_dur: [T]."""
    indices = torch.repeat_interleave(
        torch.arange(d_for_align.shape[-1]), pred_dur
    )
    en_aligned = d_for_align[:, :, indices]
    asr_aligned = t_en[:, :, indices]
    return en_aligned, asr_aligned


def pick_bucket(total_frames):
    for b in DECODER_BUCKETS:
        if b >= total_frames:
            return b
    return DECODER_BUCKETS[-1]


def main():
    print("Loading Kokoro PyTorch reference...")
    kmodel = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()
    pipe = KPipeline(lang_code="a", model=kmodel)  # 'a' = American English

    text = "Hello, this is a test of Kokoro text to speech running on CoreML."
    voice = "af_heart"
    print(f"\nText: {text!r}")
    print(f"Voice: {voice}")

    # PyTorch reference
    print("\n[PyTorch reference]")
    pt_audio = None
    pt_pred_dur = None
    pt_input_ids = None
    pt_ref_s = None
    for result in pipe(text, voice=voice):
        # Use first chunk
        pt_audio = result.audio.numpy()
        pt_pred_dur = result.tokens  # not what we need
        # The pipeline doesn't expose pred_dur directly; we need to call kmodel ourselves
        break
    # Re-run via kmodel directly to get pred_dur
    ps = result.phonemes
    print(f"  phonemes: {ps!r}")
    voice_tensor = pipe.voices[voice].clone()
    # voice tensor is per-length: shape [N, 1, 256]; Kokoro picks index based on phoneme count
    n_phonemes = len(ps) + 2  # +2 for BOS/EOS
    if voice_tensor.dim() == 3:
        ref_s = voice_tensor[len(ps) - 1] if len(ps) - 1 < voice_tensor.shape[0] else voice_tensor[-1]
    else:
        ref_s = voice_tensor
    if ref_s.dim() == 1:
        ref_s = ref_s.unsqueeze(0)
    print(f"  ref_s shape: {list(ref_s.shape)}")

    input_ids = list(filter(lambda i: i is not None, map(lambda p: kmodel.vocab.get(p), ps)))
    input_ids_t = torch.LongTensor([[0, *input_ids, 0]])
    torch.manual_seed(0)
    with torch.no_grad():
        pt_audio, pred_dur = kmodel.forward_with_tokens(input_ids_t, ref_s, speed=1.0)
    pt_audio = pt_audio.numpy()
    print(f"  pred_dur shape: {list(pred_dur.shape)}, sum: {int(pred_dur.sum())}")
    print(f"  pt_audio shape: {pt_audio.shape}, max: {pt_audio.max():.3f}, min: {pt_audio.min():.3f}")

    sf.write(f"{OUTPUT_DIR}/kokoro_pt_reference.wav", pt_audio, SAMPLE_RATE)
    print(f"  Saved kokoro_pt_reference.wav")

    # CoreML pipeline
    print("\n[CoreML pipeline]")
    print("  Loading Predictor...")
    predictor_ml = ct.models.MLModel(f"{OUTPUT_DIR}/Kokoro_Predictor.mlpackage")

    # Run predictor
    ref_s_style = ref_s[:, 128:].numpy().astype(np.float32)
    input_ids_np = input_ids_t.numpy().astype(np.int32)
    print(f"  input_ids: {input_ids_np.shape}, ref_s_style: {ref_s_style.shape}")

    pred_out = predictor_ml.predict({
        "input_ids": input_ids_np,
        "ref_s_style": ref_s_style,
    })
    duration_ml = pred_out["duration"]
    d_for_align_ml = pred_out["d_for_align"]
    t_en_ml = pred_out["t_en"]
    print(f"  duration: {duration_ml.shape}, d_for_align: {d_for_align_ml.shape}, t_en: {t_en_ml.shape}")

    # Compare predictor outputs to PyTorch
    pred_wrapper_pt_dur, pred_wrapper_pt_d, pred_wrapper_pt_t = None, None, None
    # Re-derive PyTorch values for the same wrapper
    from convert_kokoro import PredictorWrapper
    pw = PredictorWrapper(kmodel).eval()
    with torch.no_grad():
        pt_dur, pt_d_for_align, pt_t_en = pw(input_ids_t, ref_s[:, 128:])
    print(f"  duration max diff: {np.abs(pt_dur.numpy() - duration_ml).max():.5f}")
    print(f"  d_for_align max diff: {np.abs(pt_d_for_align.numpy() - d_for_align_ml).max():.5f}")
    print(f"  t_en max diff: {np.abs(pt_t_en.numpy() - t_en_ml).max():.5f}")

    # Convert duration → integer pred_dur (matching kmodel.forward_with_tokens)
    pred_dur_ml = np.round(duration_ml).clip(min=1).astype(np.int64).squeeze()
    print(f"  pred_dur_ml: {pred_dur_ml.shape}, sum: {pred_dur_ml.sum()}")
    print(f"  pred_dur_pt: {pred_dur.numpy().shape}, sum: {pred_dur.sum().item()}")

    # Use PT duration for fair quality comparison (avoid drift)
    use_dur = pred_dur.numpy()
    total_frames = int(use_dur.sum())
    print(f"  total_frames: {total_frames}")

    # Expand features
    indices = np.repeat(np.arange(input_ids_t.shape[-1]), use_dur)
    en_aligned = d_for_align_ml[:, :, indices]
    asr_aligned = t_en_ml[:, :, indices]
    print(f"  en_aligned: {en_aligned.shape}, asr_aligned: {asr_aligned.shape}")

    # Pick bucket and pad
    bucket = pick_bucket(total_frames)
    print(f"  picked bucket: {bucket} (total_frames={total_frames})")
    pad_amount = bucket - total_frames
    if pad_amount > 0:
        en_aligned = np.pad(en_aligned, ((0, 0), (0, 0), (0, pad_amount)))
        asr_aligned = np.pad(asr_aligned, ((0, 0), (0, 0), (0, pad_amount)))
    print(f"  padded en_aligned: {en_aligned.shape}")

    # Load decoder bucket
    print(f"  Loading Kokoro_Decoder_{bucket}.mlpackage...")
    decoder_ml = ct.models.MLModel(f"{OUTPUT_DIR}/Kokoro_Decoder_{bucket}.mlpackage")
    dec_out = decoder_ml.predict({
        "en_aligned": en_aligned.astype(np.float32),
        "asr_aligned": asr_aligned.astype(np.float32),
        "ref_s": ref_s.numpy().astype(np.float32),
    })
    audio_ml = dec_out["audio"].squeeze()
    print(f"  audio_ml: {audio_ml.shape}")

    # Trim to actual length
    # audio_length = total_frames * (samples per frame). PT audio gives the ratio.
    samples_per_frame = pt_audio.shape[0] // int(pred_dur.sum())
    actual_samples = total_frames * samples_per_frame
    audio_ml_trimmed = audio_ml[:actual_samples]
    print(f"  samples_per_frame: {samples_per_frame}, actual_samples: {actual_samples}")
    print(f"  audio_ml_trimmed: {audio_ml_trimmed.shape}")

    # Compare
    if audio_ml_trimmed.shape[0] == pt_audio.shape[0]:
        diff = np.abs(audio_ml_trimmed - pt_audio).max()
        print(f"  max audio diff (vs UNPADDED PT): {diff:.5f}")
    else:
        print(f"  shape mismatch: ml={audio_ml_trimmed.shape} vs pt={pt_audio.shape}")

    # Also compare CoreML to PyTorch run on the SAME padded input
    # This isolates: CoreML conversion error vs fundamental padding artifact
    print("\n[PyTorch decoder on SAME padded input]")
    from convert_kokoro import DecoderWrapper
    dw = DecoderWrapper(kmodel).eval()
    en_aligned_t = torch.from_numpy(en_aligned).float()
    asr_aligned_t = torch.from_numpy(asr_aligned).float()
    torch.manual_seed(0)
    with torch.no_grad():
        pt_padded_audio = dw(en_aligned_t, asr_aligned_t, ref_s).numpy().squeeze()
    pt_padded_trimmed = pt_padded_audio[:actual_samples]
    print(f"  pt_padded shape: {pt_padded_audio.shape}")

    diff_pt_padded_vs_unpadded = np.abs(pt_padded_trimmed - pt_audio).max()
    diff_ml_vs_pt_padded = np.abs(audio_ml_trimmed - pt_padded_trimmed).max()
    print(f"  max diff PT-padded vs PT-unpadded: {diff_pt_padded_vs_unpadded:.5f}")
    print(f"    (this measures the FUNDAMENTAL padding artifact)")
    print(f"  max diff CoreML vs PT-padded: {diff_ml_vs_pt_padded:.5f}")
    print(f"    (this measures the CoreML conversion error)")
    sf.write(f"{OUTPUT_DIR}/kokoro_pt_padded_bucket{bucket}.wav", pt_padded_trimmed, SAMPLE_RATE)

    sf.write(f"{OUTPUT_DIR}/kokoro_coreml_bucket{bucket}.wav", audio_ml_trimmed, SAMPLE_RATE)
    print(f"  Saved kokoro_coreml_bucket{bucket}.wav")

    print("\nDone!")


if __name__ == "__main__":
    main()
