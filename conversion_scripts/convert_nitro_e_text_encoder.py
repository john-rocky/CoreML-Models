"""Convert Llama 3.2 1B to CoreML, exposing the final-layer hidden state used
by Nitro-E as the "text encoder" output. Fixed sequence length = 128.

Start with FP16 (~2.5GB); palettization to INT4 (~0.6GB) is a follow-up step.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

import coremltools as ct
from transformers import AutoTokenizer, AutoModelForCausalLM

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(THIS_DIR, "Nitro-E", "reference_dump")


def _patch_coremltools_cast():
    import numpy as np
    import coremltools.converters.mil.frontend.torch.ops as _ops
    from coremltools.converters.mil import Builder as mb

    _get_inputs = _ops._get_inputs

    def _cast(context, node, dtype, dtype_name):
        inputs = _get_inputs(context, node, expected=1)
        x = inputs[0]
        if not (len(x.shape) == 0 or np.all([d == 1 for d in x.shape])):
            raise ValueError("input to cast must be either a scalar or a length 1 tensor")
        if x.can_be_folded_to_const():
            if not isinstance(x.val, dtype):
                val = x.val
                if hasattr(val, "item"):
                    val = val.item()
                res = mb.const(val=dtype(val), name=node.name)
            else:
                res = x
        elif len(x.shape) > 0:
            x = mb.squeeze(x=x, name=node.name + "_item")
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
        else:
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
        context.add(res, node.name)

    _ops._cast = _ops._cast.__wrapped__ if hasattr(_ops._cast, "__wrapped__") else _cast
    _ops._cast = _cast


class LlamaEncoderWrapper(nn.Module):
    """Return the final-layer hidden state (post-norm), matching Nitro-E's
    ``hidden_states[-1]`` extraction."""

    def __init__(self, model: AutoModelForCausalLM):
        super().__init__()
        # LlamaForCausalLM wraps LlamaModel as `.model`; we only need the backbone
        self.backbone = model.model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return out.last_hidden_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "NitroE_TextEncoder.mlpackage"))
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args()

    _patch_coremltools_cast()

    print("[load] meta-llama/Llama-3.2-1B")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=torch.float32)
    model.eval()
    wrapper = LlamaEncoderWrapper(model).eval()

    # Parity check against the captured reference
    if os.path.exists(os.path.join(REF_DIR, "prompt_embeds.pt")):
        prompt = "a hot air balloon in the shape of a heart, grand canyon"  # lowercased like the pipeline does
        tokens = tokenizer(
            prompt,
            padding="max_length",
            max_length=args.seq_len,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask
        with torch.no_grad():
            our = wrapper(input_ids, attention_mask)
        ref = torch.load(os.path.join(REF_DIR, "prompt_embeds.pt"),
                         map_location="cpu", weights_only=True)
        diff = (our - ref).abs().max().item()
        print(f"[parity] wrapper vs reference prompt_embeds max abs = {diff:.6f}")
        if diff > 1e-3:
            print("  WARNING: wrapper differs from reference; check pipeline preprocessing")
    else:
        input_ids = torch.randint(0, tokenizer.vocab_size, (1, args.seq_len), dtype=torch.long)
        attention_mask = torch.ones(1, args.seq_len, dtype=torch.long)

    print("[trace]")
    traced = torch.jit.trace(wrapper, (input_ids, attention_mask))
    traced = torch.jit.freeze(traced.eval())

    print(f"[convert] precision={args.precision}")
    ct_precision = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, args.seq_len), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, args.seq_len), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="last_hidden_state", dtype=np.float32)],
        compute_precision=ct_precision,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
    )
    mlmodel.short_description = (
        f"Llama 3.2 1B final-layer hidden state as used by Nitro-E as the text encoder. "
        f"Fixed seq_len={args.seq_len}. Outputs last_hidden_state [1,{args.seq_len},2048]."
    )
    print(f"[save] -> {args.out}")
    mlmodel.save(args.out)


if __name__ == "__main__":
    main()
