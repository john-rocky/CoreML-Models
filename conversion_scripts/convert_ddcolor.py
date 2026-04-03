"""
Convert DDColor Tiny image colorization model to CoreML.

Hooks are replaced with explicit feature passing for correct tracing.
coremltools _int op is monkey-patched to handle non-scalar arrays.
"""

import sys
sys.path.insert(0, '/tmp/ddcolor_repo')

import torch
import torch.nn as nn
import os
import numpy as np


class DDColorFlat(nn.Module):
    """DDColor with hooks replaced by explicit feature passing."""

    def __init__(self, ddcolor):
        super().__init__()
        # Encoder components
        self.encoder_arch = ddcolor.encoder.arch
        # Decoder components
        self.decoder = ddcolor.decoder
        # Refine
        self.refine_net = ddcolor.refine_net
        # Normalization buffers
        self.register_buffer('mean', ddcolor.mean)
        self.register_buffer('std', ddcolor.std)

    def forward(self, x):
        # Normalize input
        x = (x - self.mean) / self.std

        # Run ConvNeXt encoder and capture features at each stage
        features = []
        h = x
        for i in range(4):
            h = self.encoder_arch.downsample_layers[i](h)
            h = self.encoder_arch.stages[i](h)
            norm_layer = getattr(self.encoder_arch, f'norm{i}')
            feat = norm_layer(h)
            features.append(feat)

        # Manually inject features into decoder hooks
        for i, hook in enumerate(self.decoder.hooks):
            hook.feature = features[i]

        # Run decoder
        out_feat = self.decoder()

        # Refine
        coarse_input = torch.cat([out_feat, x], dim=1)
        out = self.refine_net(coarse_input)
        return out


def main():
    import coremltools as ct
    import coremltools.converters.mil.frontend.torch.ops as torch_ops
    from coremltools.converters.mil import Builder as mb

    # Monkey-patch _int handler for non-scalar arrays
    def patched_cast(context, node, dtype, dtype_name):
        inputs = torch_ops._get_inputs(context, node, expected=1)
        x = inputs[0]
        if x.val is not None:
            val = x.val
            if hasattr(val, 'item'):
                try:
                    val = val.item()
                except (ValueError, RuntimeError):
                    val = int(np.asarray(val).flat[0])
            res = mb.const(val=dtype(val), name=node.name)
            context.add(res)
        else:
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
            context.add(res)
    torch_ops._cast = patched_cast

    from ddcolor.model import DDColor

    print("Loading DDColor Tiny model...")
    model = DDColor(
        encoder_name='convnext-t',
        decoder_name='MultiScaleColorDecoder',
        input_size=[512, 512],
        num_output_channels=2,
        last_norm='Spectral',
        do_normalize=False,
        num_queries=100,
        num_scales=3,
        dec_layers=9,
    )

    hf_snapshots = os.path.expanduser(
        "~/.cache/huggingface/hub/models--piddnad--ddcolor_paper_tiny/snapshots/")
    snapshot = os.listdir(hf_snapshots)[0]
    weights_path = os.path.join(hf_snapshots, snapshot, "pytorch_model.bin")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Create flat model (no hooks for data flow)
    flat = DDColorFlat(model)
    flat.eval()

    # Verify output matches original
    dummy = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out_orig = model(dummy)
        out_flat = flat(dummy)
    diff = (out_orig - out_flat).abs().max().item()
    print(f"Flat vs original diff: {diff:.8f}")
    print(f"Output shape: {out_flat.shape}")

    # Trace
    print("Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(flat, dummy, strict=False)

    # Verify trace
    with torch.no_grad():
        out_traced = traced(dummy)
    diff2 = (out_flat - out_traced).abs().max().item()
    print(f"Trace verification diff: {diff2:.8f}")

    # Second input to make sure it's not using cached values
    dummy2 = torch.rand(1, 3, 512, 512)  # different input
    with torch.no_grad():
        out_flat2 = flat(dummy2)
        out_traced2 = traced(dummy2)
    diff3 = (out_flat2 - out_traced2).abs().max().item()
    diff_inputs = (out_traced - out_traced2).abs().max().item()
    print(f"Second input trace diff: {diff3:.8f}")
    print(f"Different inputs produce different outputs: {diff_inputs > 0.01}")

    # Convert
    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name='image', shape=(1, 3, 512, 512))],
        outputs=[ct.TensorType(name='ab_channels')],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT32,
    )

    mlmodel.author = "CoreML-Models"
    mlmodel.short_description = "DDColor Tiny: Image colorization. Input: 512x512 RGB [0,1]. Output: AB channels in LAB."
    mlmodel.license = "Apache-2.0"

    output_dir = os.path.join(os.path.dirname(__file__), "..",
                               "sample_apps", "DDColorDemo", "DDColorDemo")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "DDColor_Tiny.mlpackage")
    mlmodel.save(output_path)

    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fns in os.walk(output_path) for f in fns) / 1e6
    print(f"\nSaved to {output_path} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
