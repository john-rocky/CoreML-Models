"""Palettize the Nitro-E Llama 3.2 1B text encoder mlpackage to INT4 (block-wise,
per-grouped-channel, group_size=32) — the same recipe Apple uses in their
"On Device Llama 3.1 with Core ML" blog post.

Input:  NitroE_TextEncoder.mlpackage   (~2.3 GB, FP16)
Output: NitroE_TextEncoder_INT4.mlpackage  (target ~0.6 GB)
"""

import argparse
import os
import time

import coremltools as ct
import coremltools.optimize as cto

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(THIS_DIR, "NitroE_TextEncoder.mlpackage"))
    ap.add_argument("--dst", default=os.path.join(THIS_DIR, "NitroE_TextEncoder_INT4.mlpackage"))
    ap.add_argument("--nbits", type=int, default=4)
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
    print(f"[palettize] nbits={args.nbits}, granularity=per_grouped_channel, group_size={args.group_size}")
    t0 = time.time()
    q = cto.coreml.palettize_weights(model, config)
    dt = time.time() - t0
    print(f"[palettize] done in {dt:.1f}s")

    q.short_description = (
        f"Llama 3.2 1B text encoder for Nitro-E — {args.nbits}-bit palettized "
        f"(group_size={args.group_size}). Inputs: input_ids [1,128], attention_mask [1,128]. "
        f"Output: last_hidden_state [1,128,2048]."
    )
    print(f"[save] -> {args.dst}")
    q.save(args.dst)


if __name__ == "__main__":
    main()
