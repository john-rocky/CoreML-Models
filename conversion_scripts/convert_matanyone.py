"""
Convert MatAnyone (https://github.com/pq-yang/MatAnyone) to Core ML.

Splits the network into 5 mlpackages so the per-frame state machine can live
in Swift while CoreML handles the heavy compute:

  encoder       image                          -> ms feats + key/shrinkage/selection
  mask_encoder  image, pix_feat, sensory, mask -> mask_value, new_sensory, obj_summary
  read_first    first-frame memory readout (no attention)
  read          memory attention readout over a fixed-T ring buffer
  decoder       ms feats + memory_readout + sensory -> alpha matte

Resolution is fixed (default 768 x 432). Single object only. No flip-aug, no
long-term memory, no chunking. These match the official "matting" config.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent / "MatAnyone"
sys.path.insert(0, str(REPO))

# Monkey-patch helpers that use ops coremltools can't trace. We always run with
# num_objects = 1, so prod-over-objects collapses to identity.
import matanyone.utils.tensor_utils as _tu  # noqa: E402


def _aggregate_single_object(prob: torch.Tensor, dim: int) -> torch.Tensor:
    prob = prob.float()
    one_minus = 1.0 - prob  # equivalent to torch.prod(1-prob, dim=dim, keepdim=True) when dim has size 1
    new_prob = torch.cat([one_minus, prob], dim=dim).clamp(1e-7, 1 - 1e-7)
    return torch.log(new_prob / (1.0 - new_prob))


_tu.aggregate = _aggregate_single_object

# Patch the symbol everywhere it has already been imported.
import matanyone.model.matanyone as _ma  # noqa: E402
import matanyone.model.transformer.object_transformer as _ot  # noqa: E402

_ma.aggregate = _aggregate_single_object
_ot.aggregate = _aggregate_single_object


# Monkey-patch PixelFeatureFuser.forward to skip its `[:, i:i+chunk_size]`
# slice over the singleton num_objects dim. That slice is what trips Metal
# Performance Shaders on iOS GPU with `subRange.start = -1 vs length 1`,
# forcing the read / read_first models onto CPU. We hard-code the
# num_objects = 1, single_object = False (matting) fast path: one shot, no
# loop, no slicing on the singleton dim.
import matanyone.model.big_modules as _bm  # noqa: E402


def _pixel_feature_fuser_single_object_forward(
    self,
    pix_feat: torch.Tensor,
    pixel_memory: torch.Tensor,
    sensory_memory: torch.Tensor,
    last_mask: torch.Tensor,
    last_others: torch.Tensor,
    *,
    chunk_size: int = -1,
):
    # Matanyone matting config: single_object=False, num_objects=1.
    last_mask = torch.stack([last_mask, last_others], dim=2)
    sensory_readout = self.sensory_compress(torch.cat([sensory_memory, last_mask], 2))
    p16 = pixel_memory + sensory_readout
    p16 = self.fuser(pix_feat, p16)
    return p16


_bm.PixelFeatureFuser.forward = _pixel_feature_fuser_single_object_forward

from matanyone import InferenceCore  # noqa: E402
from matanyone.model.matanyone import MatAnyone  # noqa: E402

import coremltools as ct  # noqa: E402


# ----------------------------------------------------------------------------
# Wrapper modules
# ----------------------------------------------------------------------------


class EncoderWrapper(nn.Module):
    """encode_image + transform_key, returns the tensors needed by every other
    module per frame.
    """

    def __init__(self, net: MatAnyone):
        super().__init__()
        self.net = net

    def forward(self, image: torch.Tensor):
        # image: (1, 3, H, W) in [0, 1]
        n = self.net
        norm = (image - n.pixel_mean) / n.pixel_std
        f16, f8, f4, f2, f1 = n.pixel_encoder(norm)
        pix_feat = n.pix_feat_proj(f16)
        key, shrinkage, selection = n.key_proj(f16, need_s=True, need_e=True)
        return f16, f8, f4, f2, f1, pix_feat, key, shrinkage, selection


class MaskEncoderWrapper(nn.Module):
    """encode_mask, hard-coded for num_objects=1.

    Always runs the deep update path; the shallow-update use case in Swift can
    just discard ``new_sensory``.
    """

    def __init__(self, net: MatAnyone):
        super().__init__()
        self.net = net

    def forward(
        self,
        image: torch.Tensor,        # (1, 3, H, W) in [0, 1]
        pix_feat: torch.Tensor,     # (1, 256, h, w) — projected pix feat
        sensory: torch.Tensor,      # (1, 1, 256, h, w)
        mask: torch.Tensor,         # (1, 1, H, W) in [0, 1]
    ):
        n = self.net
        image = (image - n.pixel_mean) / n.pixel_std
        # single object so "others" is all zeros
        others = torch.zeros_like(mask)
        mask_value, new_sensory = n.mask_encoder(
            image, pix_feat, sensory, mask, others, deep_update=True, chunk_size=-1
        )
        obj_summary, _ = n.object_summarizer(mask, mask_value, need_weights=False)
        return mask_value, new_sensory, obj_summary


class ReadFirstWrapper(nn.Module):
    """First-frame readout: no memory attention, just pixel_fusion + readout_query."""

    def __init__(self, net: MatAnyone):
        super().__init__()
        self.net = net

    def forward(
        self,
        pix_feat: torch.Tensor,        # (1, 256, h, w)
        last_msk_value: torch.Tensor,  # (1, 1, 256, h, w)
        sensory: torch.Tensor,         # (1, 1, 256, h, w)
        last_mask: torch.Tensor,       # (1, 1, H, W) in [0, 1]
        obj_memory: torch.Tensor,      # (1, 1, 1, 16, 257)
    ):
        n = self.net
        pixel_readout = n.pixel_fusion(pix_feat, last_msk_value, sensory, last_mask)
        mem_readout, _ = n.object_transformer(
            pixel_readout, obj_memory, selector=None, need_weights=False, seg_pass=False
        )
        return mem_readout


class ReadWrapper(nn.Module):
    """Memory attention readout over a fixed-T ring buffer.

    Replicates ``MemoryManager.read`` for the single-object, no-long-term case
    plus the readout path that follows it inside ``InferenceCore._segment``.
    """

    def __init__(self, net: MatAnyone, top_k: int = 30):
        super().__init__()
        self.net = net
        self.top_k = top_k

    def forward(
        self,
        query_key: torch.Tensor,        # (1, 64, h, w)
        query_selection: torch.Tensor,  # (1, 64, h, w)
        pix_feat: torch.Tensor,         # (1, 256, h, w)
        sensory: torch.Tensor,          # (1, 1, 256, h, w)
        last_mask: torch.Tensor,        # (1, 1, H, W)
        last_pix_feat: torch.Tensor,    # (1, 256, h, w)
        last_msk_value: torch.Tensor,   # (1, 1, 256, h, w)
        mem_key: torch.Tensor,          # (1, 64,  T*h*w) — pre-flattened by Swift
        mem_shrinkage: torch.Tensor,    # (1, 1,   T*h*w)
        mem_msk_value: torch.Tensor,    # (1, 256, T*h*w)
        mem_valid: torch.Tensor,        # (1, T*h*w) — 1.0 for active slots, broadcast in Swift
        obj_memory: torch.Tensor,       # (1, 1, 1, 16, 257)
    ):
        n = self.net
        bs = query_key.shape[0]
        h, w = query_key.shape[-2:]
        HW = h * w

        mk = mem_key                                  # (B, 64, N)
        ms = mem_shrinkage                            # (B, 1, N)
        qk = query_key.reshape(bs, 64, HW)
        qe = query_selection.reshape(bs, 64, HW)

        # XMem-style anisotropic L2 similarity
        mk_t = mk.transpose(1, 2)                     # (B, N, 64)
        a_sq = mk_t.pow(2) @ qe                       # (B, N, HW)
        two_ab = 2.0 * (mk_t @ (qk * qe))             # (B, N, HW)
        b_sq = (qe * qk.pow(2)).sum(dim=1, keepdim=True)  # (B, 1, HW)
        similarity = -a_sq + two_ab - b_sq
        similarity = similarity * ms.transpose(1, 2) / (64 ** 0.5)

        # Mask out invalid memory slots — broadcast (B, N, 1) onto (B, N, HW)
        similarity = similarity + (1.0 - mem_valid.unsqueeze(-1)) * (-6.0e4)

        # Top-k softmax over the memory dimension (dim=1)
        top_vals, top_idx = torch.topk(similarity, k=self.top_k, dim=1)
        exp_vals = torch.exp(top_vals - top_vals.max(dim=1, keepdim=True)[0])
        exp_vals = exp_vals / exp_vals.sum(dim=1, keepdim=True)
        affinity = torch.zeros_like(similarity).scatter(1, top_idx, exp_vals)

        # Visual readout: (B, 256, N) @ (B, N, HW) -> (B, 256, HW) -> (B, 1, 256, h, w)
        visual_readout = (mem_msk_value @ affinity).reshape(bs, 1, 256, h, w)

        # Uncertainty. We deliberately use `.squeeze(1)` instead of `[:, 0]`
        # here — the singleton-dim index trips MPS on iOS GPU with the same
        # `subRange.start = -1` assertion that the chunk-loop slice did.
        diff = visual_readout.squeeze(1) - last_msk_value.squeeze(1)
        uncert = n.pred_uncertainty(last_pix_feat, pix_feat, last_mask, diff)
        uncert_prob = torch.sigmoid(uncert["logits"]).unsqueeze(1)  # (B, 1, 1, h, w)
        visual_readout = visual_readout * uncert_prob + last_msk_value * (1.0 - uncert_prob)

        pixel_readout = n.pixel_fusion(pix_feat, visual_readout, sensory, last_mask)
        mem_readout, _ = n.object_transformer(
            pixel_readout, obj_memory, selector=None, need_weights=False, seg_pass=False
        )
        return mem_readout


class DecoderWrapper(nn.Module):
    """MaskDecoder hard-coded for matting (seg_pass=False), num_objects=1, no last_mask residual."""

    def __init__(self, net: MatAnyone):
        super().__init__()
        self.net = net

    def forward(
        self,
        f16: torch.Tensor,           # (1, 1024, h, w)
        f8: torch.Tensor,
        f4: torch.Tensor,
        f2: torch.Tensor,
        f1: torch.Tensor,            # (1, 3, H, W) — already normalized in encoder
        mem_readout: torch.Tensor,   # (1, 1, 256, h, w)
        sensory: torch.Tensor,       # (1, 1, 256, h, w)
    ):
        new_sensory, logits = self.net.mask_decoder(
            [f16, f8, f4, f2, f1],
            mem_readout,
            sensory,
            chunk_size=-1,
            update_sensory=True,
            seg_pass=False,
            last_mask=None,
            sigmoid_residual=False,
        )
        # Matting: clamp & convert to alpha. logits is (1, 1, H, W) — already
        # the alpha matte for our single object.
        alpha = logits.clamp(0.0, 1.0)
        return new_sensory, alpha


# ----------------------------------------------------------------------------
# Conversion driver
# ----------------------------------------------------------------------------


def trace_and_convert(
    name: str,
    wrapper: nn.Module,
    sample_inputs: Tuple[torch.Tensor, ...],
    input_names,
    output_names,
    out_dir: Path,
):
    print(f"\n=== {name} ===")
    wrapper.eval()
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, sample_inputs, strict=False, check_trace=False)
    print(f"  traced. inputs={len(sample_inputs)}")

    inputs = [
        ct.TensorType(name=n, shape=t.shape, dtype=np.float32)
        for n, t in zip(input_names, sample_inputs)
    ]
    outputs = [ct.TensorType(name=n, dtype=np.float32) for n in output_names]

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )
    out_path = out_dir / f"MatAnyone_{name}.mlpackage"
    mlmodel.save(str(out_path))
    print(f"  saved -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=432)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--mem-frames", type=int, default=5,
                    help="ring-buffer length for working memory")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--only", type=str, default="all",
                    help="comma-separated subset of: encoder,mask_encoder,read_first,read,decoder")
    args = ap.parse_args()

    H, W = args.height, args.width
    assert H % 16 == 0 and W % 16 == 0, "H/W must be divisible by 16"
    h, w = H // 16, W // 16
    T = args.mem_frames

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading MatAnyone from HF...")
    core = InferenceCore("PeiqingYang/MatAnyone", device=torch.device("cpu"))
    net = core.network.eval().to("cpu")
    for p in net.parameters():
        p.requires_grad_(False)

    only = set(args.only.split(",")) if args.only != "all" else None

    def want(name):
        return only is None or name in only

    image = torch.rand(1, 3, H, W)
    mask = torch.rand(1, 1, H, W)
    sensory = torch.zeros(1, 1, 256, h, w)
    pix_feat = torch.rand(1, 256, h, w)
    f16 = torch.rand(1, 1024, h, w)
    f8 = torch.rand(1, 512, h * 2, w * 2)
    f4 = torch.rand(1, 256, h * 4, w * 4)
    f2 = torch.rand(1, 64, h * 8, w * 8)
    f1 = torch.rand(1, 3, H, W)
    key = torch.rand(1, 64, h, w)
    shrinkage = torch.rand(1, 1, h, w)
    selection = torch.rand(1, 64, h, w)
    mask_value = torch.rand(1, 1, 256, h, w)
    obj_memory = torch.rand(1, 1, 1, 16, 257)
    mem_readout = torch.rand(1, 1, 256, h, w)
    N = T * h * w
    mem_key = torch.rand(1, 64, N)
    mem_shrinkage = torch.rand(1, 1, N)
    mem_msk_value = torch.rand(1, 256, N)
    mem_valid = torch.ones(1, N)

    if want("encoder"):
        trace_and_convert(
            "encoder",
            EncoderWrapper(net),
            (image,),
            ["image"],
            ["f16", "f8", "f4", "f2", "f1", "pix_feat", "key", "shrinkage", "selection"],
            args.out,
        )

    if want("mask_encoder"):
        trace_and_convert(
            "mask_encoder",
            MaskEncoderWrapper(net),
            (image, pix_feat, sensory, mask),
            ["image", "pix_feat", "sensory", "mask"],
            ["mask_value", "new_sensory", "obj_summary"],
            args.out,
        )

    if want("read_first"):
        trace_and_convert(
            "read_first",
            ReadFirstWrapper(net),
            (pix_feat, mask_value, sensory, mask, obj_memory),
            ["pix_feat", "last_msk_value", "sensory", "last_mask", "obj_memory"],
            ["mem_readout"],
            args.out,
        )

    if want("read"):
        trace_and_convert(
            "read",
            ReadWrapper(net),
            (
                key, selection, pix_feat, sensory, mask, pix_feat, mask_value,
                mem_key, mem_shrinkage, mem_msk_value, mem_valid, obj_memory,
            ),
            [
                "query_key", "query_selection", "pix_feat", "sensory", "last_mask",
                "last_pix_feat", "last_msk_value",
                "mem_key", "mem_shrinkage", "mem_msk_value", "mem_valid", "obj_memory",
            ],
            ["mem_readout"],
            args.out,
        )

    if want("decoder"):
        trace_and_convert(
            "decoder",
            DecoderWrapper(net),
            (f16, f8, f4, f2, f1, mem_readout, sensory),
            ["f16", "f8", "f4", "f2", "f1", "mem_readout", "sensory"],
            ["new_sensory", "alpha"],
            args.out,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
