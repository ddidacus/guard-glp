"""Ad-hoc probe: prove sample_on_manifold actually denoises (latent moves each step).

Loads a GLP, builds a few normalized layer-14 activations, and runs the same
noise_level=0.0 reconstruction path the eval uses — but instrumented to print, per
step, how far the latent moves and how large the denoiser's velocity prediction is.
If denoising is real, per-step deltas are non-zero and the reconstruction differs
from the input; if it were a no-op, deltas would be ~0.

    python scripts/detection/_debug_denoise.py \
        --glp=/lambdafs/public_artifacts/results/GLP/glp-llama1b-guardglpbenign-alltok-full-layer14-d3 \
        --layer=14
"""

from typing import cast

import fire
import torch

from glp import flow_matching
from glp.denoiser import load_glp


def main(glp: str, layer: int = 14, n: int = 8, num_timesteps: int = 50) -> None:
    device = "cuda:0"
    model = load_glp(glp, device=device, checkpoint="final")

    d = model.denoiser.model.d_input
    # a few random-but-fixed "activations" in raw space, then normalize as the eval does
    torch.manual_seed(0)
    raw = torch.randn(n, 1, d, device=device) * 3.0 + 1.0
    x0 = model.normalizer.normalize(raw, layer_idx=layer)

    # replicate fm_prepare at noise_level=0.0: no noise added, start at max timestep
    noise = torch.randn_like(x0)
    noisy, _, timesteps, _ = flow_matching.fm_prepare(
        model.scheduler, x0, noise, u=torch.zeros(n)
    )
    print(f"noise added? ||noisy - x0|| = {torch.norm(noisy - x0).item():.6e}  (expect ~0 at noise_level=0)")

    # instrumented denoise loop (mirrors sample_on_manifold)
    latents = noisy.clone()
    start = latents.clone()
    model.scheduler.set_timesteps(num_timesteps)
    print(f"{'step':>4} {'timestep':>10} {'||Δlatent||':>14} {'||velocity||':>14}")
    for i, t in enumerate(cast(torch.Tensor, model.scheduler.timesteps)):
        mask = timesteps[:, 0, 0] <= t
        latents[mask] = start[mask]
        prev = latents.clone()
        ts = t[None, ...].repeat(latents.shape[0], 1, 1)
        vel = model.denoiser(latents=latents, timesteps=ts, layer_idx=layer)
        latents = model.scheduler.step(
            cast("torch.FloatTensor", vel),
            cast("torch.FloatTensor", t),
            cast("torch.FloatTensor", latents),
            return_dict=False,
        )[0]
        if i < 5 or i == num_timesteps - 1:
            print(
                f"{i:>4} {float(t):>10.2f} "
                f"{torch.norm(latents - prev).item():>14.6e} "
                f"{torch.norm(vel).item():>14.6e}"
            )

    recon_err = torch.norm((latents - x0).reshape(n, -1), dim=1)
    print(f"\nfinal ||recon - x0|| per sample: {recon_err.tolist()}")
    print(f"total latent travel ||final - start||: {torch.norm(latents - start).item():.6e}")


if __name__ == "__main__":
    fire.Fire(main)
