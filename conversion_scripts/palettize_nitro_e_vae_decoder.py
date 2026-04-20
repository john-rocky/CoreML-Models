"""Palettize the FP32 DC-AE decoder to 8-bit (per-grouped-channel, group_size=32).

FP16 compute_precision is unsafe for the Sana linear-attention normalization
(FP16 overflows and produces smeared output). We keep the numerics in FP32
but drop weights to 8 bits post-training. Target: 608 MB -> ~160 MB.
"""

import argparse
import os
import time

import coremltools as ct
import coremltools.optimize as cto

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(THIS_DIR, "NitroE_VAEDecoder_FP32.mlpackage"))
    ap.add_argument("--dst", default=os.path.join(THIS_DIR, "NitroE_VAEDecoder_INT8.mlpackage"))
    ap.add_argument("--nbits", type=int, default=8)
    ap.add_argument("--group_size", type=int, default=32)
    args = ap.parse_args()

    print(f"[load] {args.src}")
    model = ct.models.MLModel(args.src)

    op_config = cto.coreml.OpPalettizerConfig(
        nbits=args.nbits,
        mode="kmeans",
        granularity="per_grouped_channel",
        group_size=args.group_size,
    )
    config = cto.coreml.OptimizationConfig(global_config=op_config)
    print(f"[palettize] nbits={args.nbits}, group_size={args.group_size}")
    t0 = time.time()
    q = cto.coreml.palettize_weights(model, config)
    dt = time.time() - t0
    print(f"[palettize] done in {dt:.1f}s")

    q.short_description = (
        f"Nitro-E DC-AE f32c32 decoder — compute FP32, weights palettized to "
        f"{args.nbits} bits. Latent [1,32,16,16] -> image [1,3,512,512]."
    )
    print(f"[save] -> {args.dst}")
    q.save(args.dst)


if __name__ == "__main__":
    main()
