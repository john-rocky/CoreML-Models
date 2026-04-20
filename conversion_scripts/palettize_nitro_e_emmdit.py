"""Palettize the E-MMDiT FP16 mlpackage to N-bit (per-grouped-channel)."""

import argparse
import os
import time

import coremltools as ct
import coremltools.optimize as cto

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(THIS_DIR, "NitroE_EMMDiT.mlpackage"))
    ap.add_argument("--dst", default=None)
    ap.add_argument("--nbits", type=int, default=8)
    ap.add_argument("--group_size", type=int, default=32)
    args = ap.parse_args()

    dst = args.dst or os.path.join(
        THIS_DIR, f"NitroE_EMMDiT_INT{args.nbits}.mlpackage"
    )

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
    print(f"[palettize] done in {time.time() - t0:.1f}s")

    q.short_description = (
        f"Nitro-E E-MMDiT denoiser (304M) — {args.nbits}-bit palettized. "
        f"latent [1,32,16,16], encoder_hidden_states [1,128,2048], timestep [1] → noise_pred [1,32,16,16]."
    )
    print(f"[save] -> {dst}")
    q.save(dst)


if __name__ == "__main__":
    main()
