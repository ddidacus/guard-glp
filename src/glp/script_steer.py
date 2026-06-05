from collections.abc import Callable
from typing import Any

import einops
import torch
import transformers
from baukit import TraceDict

from glp import flow_matching
from glp.denoiser import GLP


# =========================
#   Diffusion Functions
# =========================
def postprocess_on_manifold_wrapper(
    model: GLP,
    u: float = 0.5,
    num_timesteps: int = 20,
    layer_idx: int | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    scheduler = model.scheduler
    scheduler.set_timesteps(num_timesteps)

    def postprocess_on_manifold(acts_edit: torch.Tensor) -> torch.Tensor:
        has_seq_dim = len(acts_edit.shape) == 3
        b = acts_edit.shape[0]
        latents = acts_edit
        if has_seq_dim:
            latents = einops.rearrange(latents, "b s d -> (b s) 1 d")
        else:
            latents = einops.rearrange(latents, "b d -> b 1 d")
        latents = model.normalizer.normalize(latents, layer_idx=layer_idx)
        noise = torch.randn_like(latents)
        noisy_latents, _, timesteps, _ = flow_matching.fm_prepare(
            scheduler,
            latents,
            noise,
            u=torch.ones(latents.shape[0]) * u,
        )
        latents = flow_matching.sample_on_manifold(
            model,
            noisy_latents,
            start_timestep=timesteps[0].item(),
            num_timesteps=num_timesteps,
            layer_idx=layer_idx,
        )
        latents = model.normalizer.denormalize(latents, layer_idx=layer_idx)
        if has_seq_dim:
            latents = einops.rearrange(latents, "(b s) 1 d -> b s d", b=b)
        else:
            latents = einops.rearrange(latents, "b 1 d -> b d")
        latents = latents.to(device=acts_edit.device, dtype=acts_edit.dtype)
        return latents

    return postprocess_on_manifold


# =========================
#    Steering Functions
# =========================
def addition_intervention(
    w: torch.Tensor | None = None,
    alphas: torch.Tensor | None = None,
    postprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> Callable[[Any, str, Any], Any]:
    if postprocess_fn is None:

        def _identity(x: torch.Tensor) -> torch.Tensor:
            return x

        postprocess_fn = _identity

    def rep_act(output: Any, layer_name: str, inputs: Any) -> Any:
        nonlocal w, alphas
        use_tuple = isinstance(output, tuple)
        act = output[0] if use_tuple else output
        if w is not None:
            if alphas is None:
                raise ValueError("alphas must be provided when w is provided")
            # move to device
            w = w.to(device=act.device, dtype=act.dtype)
            alphas = alphas.to(device=act.device, dtype=act.dtype)
            # reshape based on if batched / unbatched
            if w.ndim == 1:
                w = w[None, None, :]
            elif w.ndim == 2:
                w = w[:, None, :]
            if alphas.ndim == 1:
                alphas = alphas[:, None, None]
            # only apply to every new generated token
            act[:, [-1], :] = postprocess_fn(act[:, [-1], :] + alphas * w)
        return (act, *output[1:]) if use_tuple else act

    return rep_act


def generate(
    model: Any,
    processor: Any,
    inputs: dict[str, Any],
    remove_input: bool = True,
    **generate_kwargs: Any,
) -> list[str]:
    with torch.no_grad():
        output = model.generate(**inputs, **generate_kwargs)
        if remove_input:
            input_len = inputs["input_ids"].shape[1]
            output = output[:, input_len:]
        output = processor.batch_decode(output, skip_special_tokens=True)
    return output


def generate_with_intervention_wrapper(
    seed: int | None = 42,
    generate_kwargs: dict[str, Any] | None = None,
) -> Callable[..., list[str]]:
    def generate_with_intervention(
        text: str | list[str],
        hf_model: Any,
        hf_processor: Any,
        generate_kwargs: dict[str, Any] | None = None,
        layers: list[str] | None = None,
        intervention_wrapper: Callable[..., Callable[..., Any]] | None = None,
        intervention_kwargs: dict[str, Any] | None = None,
        forward_only: bool = False,
    ) -> list[str]:
        generate_kwargs = generate_kwargs or {"max_new_tokens": 10}
        layers = layers or []
        intervention_kwargs = intervention_kwargs or {}
        if seed is not None:
            transformers.set_seed(seed)
        inputs = hf_processor(text, return_tensors="pt", padding=True).to(
            hf_model.device
        )
        if intervention_wrapper is not None:
            intervention_fn = intervention_wrapper(**intervention_kwargs)
        else:
            intervention_fn = None
        with TraceDict(hf_model, layers=layers, edit_output=intervention_fn):
            if forward_only:
                outputs = hf_model(**inputs)
                output_text = hf_processor.batch_decode(
                    outputs.logits.argmax(dim=-1), skip_special_tokens=True
                )
            else:
                output_text = generate(
                    hf_model, hf_processor, inputs, **generate_kwargs
                )
        return output_text

    return generate_with_intervention
