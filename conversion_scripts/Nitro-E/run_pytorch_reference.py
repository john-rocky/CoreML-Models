"""Reference inference for Nitro-E on macOS (CPU/MPS), no flash-attn required.

Runs the 4-step distilled variant, saves the output image and the intermediate
text embedding / initial latent / final latent as .pt files so later CoreML
conversions can be numerically compared against this reference.
"""

import os
import sys
import time
import argparse

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from core.tools.inference_pipe import init_pipe  # noqa: E402


def _install_dump_hooks(pipe, dump_dir: str):
    """Capture prompt_embeds, initial latent, per-step latents, and VAE output."""
    os.makedirs(dump_dir, exist_ok=True)
    state = {"step": 0}

    orig_encode_prompt = pipe.encode_prompt

    def encode_prompt_wrapper(*args, **kwargs):
        out = orig_encode_prompt(*args, **kwargs)
        prompt_embeds, prompt_attn_mask, neg_embeds, neg_attn_mask = out
        torch.save(prompt_embeds.detach().cpu(), os.path.join(dump_dir, "prompt_embeds.pt"))
        torch.save(prompt_attn_mask.detach().cpu(), os.path.join(dump_dir, "prompt_attention_mask.pt"))
        if neg_embeds is not None:
            torch.save(neg_embeds.detach().cpu(), os.path.join(dump_dir, "negative_prompt_embeds.pt"))
            torch.save(neg_attn_mask.detach().cpu(), os.path.join(dump_dir, "negative_prompt_attention_mask.pt"))
        return out

    pipe.encode_prompt = encode_prompt_wrapper

    orig_transformer_call = pipe.transformer.__call__

    def transformer_wrapper(self, hidden_states, **kw):
        idx = state["step"]
        torch.save(hidden_states.detach().cpu(), os.path.join(dump_dir, f"latent_in_step{idx}.pt"))
        if "encoder_hidden_states" in kw:
            # only save once on first step — identical across steps
            if idx == 0:
                torch.save(kw["encoder_hidden_states"].detach().cpu(),
                           os.path.join(dump_dir, "transformer_encoder_hidden_states.pt"))
        if "timestep" in kw:
            torch.save(kw["timestep"].detach().cpu(), os.path.join(dump_dir, f"timestep_step{idx}.pt"))
        out = orig_transformer_call(hidden_states, **kw)
        noise_pred = out[0] if isinstance(out, tuple) else out.sample
        torch.save(noise_pred.detach().cpu(), os.path.join(dump_dir, f"noise_pred_step{idx}.pt"))
        state["step"] += 1
        return out

    import types
    pipe.transformer.__call__ = types.MethodType(transformer_wrapper, pipe.transformer)
    # DiffusionPipeline forwards to transformer.forward via Module.__call__; patch forward too
    orig_forward = pipe.transformer.forward

    def forward_wrapper(hidden_states, **kw):
        idx = state["step"]
        torch.save(hidden_states.detach().cpu(), os.path.join(dump_dir, f"latent_in_step{idx}.pt"))
        if idx == 0 and "encoder_hidden_states" in kw:
            torch.save(kw["encoder_hidden_states"].detach().cpu(),
                       os.path.join(dump_dir, "transformer_encoder_hidden_states.pt"))
        if "timestep" in kw:
            torch.save(kw["timestep"].detach().cpu(), os.path.join(dump_dir, f"timestep_step{idx}.pt"))
        out = orig_forward(hidden_states, **kw)
        noise_pred = out[0] if isinstance(out, tuple) else out.sample
        torch.save(noise_pred.detach().cpu(), os.path.join(dump_dir, f"noise_pred_step{idx}.pt"))
        state["step"] += 1
        return out

    pipe.transformer.forward = forward_wrapper

    orig_vae_decode = pipe.vae.decode

    def vae_decode_wrapper(z, **kw):
        torch.save(z.detach().cpu(), os.path.join(dump_dir, "vae_decode_input.pt"))
        out = orig_vae_decode(z, **kw)
        img = out.sample if hasattr(out, "sample") else out
        torch.save(img.detach().cpu(), os.path.join(dump_dir, "vae_decode_output.pt"))
        return out

    pipe.vae.decode = vae_decode_wrapper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="A hot air balloon in the shape of a heart, grand canyon")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default="Nitro-E-512px-dist.safetensors")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "reference_out.png"))
    ap.add_argument("--dump", default=None, help="If set, dump intermediate tensors to this dir")
    args = ap.parse_args()

    # fp32 on CPU is the safest: MPS + bf16 has known numerical issues for DiT.
    dtype = torch.float32 if args.device == "cpu" else torch.float16
    device = torch.device(args.device)

    print(f"[load] device={device}, dtype={dtype}")
    pipe = init_pipe(
        device=device,
        dtype=dtype,
        resolution=args.resolution,
        repo_name="amd/Nitro-E",
        ckpt_name=args.ckpt,
    )
    pipe.to(device)

    if args.dump is not None:
        _install_dump_hooks(pipe, args.dump)
        print(f"[dump] capturing intermediates to {args.dump}")

    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    t0 = time.time()
    out = pipe(
        prompt=args.prompt,
        width=args.resolution,
        height=args.resolution,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=gen,
    ).images[0]
    dt = time.time() - t0
    print(f"[done] {dt:.1f}s, saving -> {args.out}")
    out.save(args.out)


if __name__ == "__main__":
    main()
