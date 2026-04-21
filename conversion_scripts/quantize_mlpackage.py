"""
Post-hoc weight quantization for any existing `.mlpackage`.

Use this to shrink models that were converted with `ct.precision.FLOAT16`
(or even FP32) **without re-running the original PyTorch → CoreML
conversion**. Typical reductions (FP16 → 8-bit palettized):

  MatAnyone (5 packages, 111 MB)    →  ~28 MB
  Florence-2 text encoder (~90 MB)  →  ~23 MB
  Kokoro decoder_512 (~246 MB)      →  ~62 MB
  CLIP ViT-B/32 text enc (~121 MB)  →  ~31 MB

All reductions are lossy — verify parity against the FP16 reference on
representative inputs before shipping.

Examples:

  # Single file, default 8-bit palettize
  python quantize_mlpackage.py MatAnyone_encoder.mlpackage

  # Whole directory, place outputs next to inputs with `_int8` suffix
  python quantize_mlpackage.py --mode palettize --nbits 8 --recursive out/

  # 4-bit palettize (aggressive, risky for attention layers)
  python quantize_mlpackage.py --nbits 4 Kokoro_Decoder_512.mlpackage

  # Linear quantize instead of k-means palettize (faster to build,
  # sometimes better for conv-heavy models)
  python quantize_mlpackage.py --mode linear Kokoro_Predictor.mlpackage

Dependencies: `coremltools >= 8.0`.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

try:
    import coremltools as ct
    import coremltools.optimize.coreml as cto
except ImportError as e:
    print(f"coremltools >= 8.0 is required: {e}", file=sys.stderr)
    sys.exit(1)


def quantize(
    src: Path,
    dst: Path,
    *,
    mode: str = "palettize",
    nbits: int = 8,
    granularity: str = "per_grouped_channel",
    group_size: int = 16,
) -> None:
    """Load `src` as an MLPackage, quantize, save as `dst`.

    Parameters
    ----------
    src : Path
        Existing `.mlpackage` (or `.mlmodel`).
    dst : Path
        Destination `.mlpackage` path. Parent directory must exist.
    mode : {"palettize", "linear"}
        - "palettize": k-means lookup-table compression (nbits 4/6/8).
           Strong compression, preserves quality well for smooth weights.
        - "linear": affine/symmetric quantization. Faster to build,
           sometimes better for conv-heavy networks.
    nbits : int
        Effective bit-width. For palettize: 2/4/6/8. For linear: 8 only.
    granularity : str
        Palettize granularity. `per_grouped_channel` with small groups
        preserves accuracy on transformer attention better than per_tensor.
    group_size : int
        Group size for per-grouped-channel granularity. Smaller = more
        accurate, larger = more compression.
    """
    print(f"  loading {src.name}...")
    model = ct.models.MLModel(str(src))

    if mode == "palettize":
        op_config = cto.OpPalettizerConfig(
            nbits=nbits,
            mode="kmeans",
            granularity=granularity,
            group_size=group_size if granularity == "per_grouped_channel" else None,
        )
        config = cto.OptimizationConfig(global_config=op_config)
        print(f"  palettizing: nbits={nbits}, granularity={granularity}...")
        compressed = cto.palettize_weights(model, config)
    elif mode == "linear":
        op_config = cto.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
            granularity="per_channel",
        )
        config = cto.OptimizationConfig(global_config=op_config)
        print("  linear-quantizing to int8...")
        compressed = cto.linear_quantize_weights(model, config)
    else:
        raise ValueError(f"unknown mode: {mode}")

    # Overwrite destination if it already exists
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    compressed.save(str(dst))

    src_bytes = _dir_size(src)
    dst_bytes = _dir_size(dst)
    ratio = (1 - dst_bytes / src_bytes) * 100 if src_bytes > 0 else 0
    print(f"  {src.name}: {_fmt_bytes(src_bytes)} → {_fmt_bytes(dst_bytes)} "
          f"({ratio:.1f}% smaller) → {dst}")


def _dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _collect_targets(inputs: list[Path], recursive: bool) -> list[Path]:
    targets: list[Path] = []
    for inp in inputs:
        if inp.is_file() and inp.suffix in (".mlpackage", ".mlmodel"):
            targets.append(inp)
        elif inp.is_dir() and inp.suffix == ".mlpackage":
            targets.append(inp)
        elif inp.is_dir() and recursive:
            for child in inp.rglob("*.mlpackage"):
                if "int8" in child.name or "int4" in child.name:
                    continue  # skip already-quantized outputs
                targets.append(child)
    return targets


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("inputs", nargs="+", type=Path,
                    help=".mlpackage file(s) or directory containing them")
    ap.add_argument("--mode", choices=("palettize", "linear"),
                    default="palettize")
    ap.add_argument("--nbits", type=int, default=8,
                    choices=(2, 4, 6, 8),
                    help="bit-width for palettize mode (ignored for linear)")
    ap.add_argument("--granularity",
                    choices=("per_tensor", "per_channel",
                             "per_grouped_channel"),
                    default="per_grouped_channel",
                    help="palettize granularity (finer = better quality, "
                         "slightly larger file)")
    ap.add_argument("--group-size", type=int, default=16,
                    help="group size for per_grouped_channel granularity")
    ap.add_argument("--recursive", "-r", action="store_true",
                    help="recurse into directories looking for .mlpackage")
    ap.add_argument("--suffix", type=str, default="_int8",
                    help="output filename suffix before .mlpackage "
                         "(pass empty string to overwrite in place)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="optional directory to place outputs into; "
                         "default is next to each input")
    args = ap.parse_args()

    targets = _collect_targets(args.inputs, args.recursive)
    if not targets:
        print("no .mlpackage inputs found", file=sys.stderr)
        sys.exit(2)

    print(f"quantizing {len(targets)} model(s): mode={args.mode} "
          f"nbits={args.nbits} granularity={args.granularity}")

    for src in targets:
        stem = src.stem
        dst_name = f"{stem}{args.suffix}.mlpackage" if args.suffix else f"{stem}.mlpackage"
        dst_dir = args.out_dir if args.out_dir else src.parent
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / dst_name
        quantize(
            src, dst,
            mode=args.mode,
            nbits=args.nbits,
            granularity=args.granularity,
            group_size=args.group_size,
        )


if __name__ == "__main__":
    main()
