import math
import numpy as np
import torch
from types import SimpleNamespace

from diffusers import FlowMatchEulerDiscreteScheduler

# ==========================
#  Flow Matching Functions
# ==========================
def fm_scheduler():
    return FlowMatchEulerDiscreteScheduler()

def fm_prepare(scheduler, model_input, noise, u=None, generator=None):
    """
    Prepare inputs for flow matching training.
    Reference: https://github.com/huggingface/diffusers/blob/9f48394bf7ab75a43435d3ebb96649665e09c98b/examples/dreambooth/train_dreambooth_lora_flux.py#L1736
    """
    # sanity check
    assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler), "Only FlowMatchEulerDiscreteScheduler is supported"
    assert model_input.ndim == 3, f"Expected (batch, seq, dim), got shape {model_input.shape}"

    # random u -> random indices from the scheduler -> random noising steps (for learning)
    # scheduler.timesteps (1000,)
    # scheduler.sigmas (1000,)
    
    scheduler_sigmas = torch.linspace(0, 1.0, 1000)

    if u is None:
        batch_size = model_input.shape[0]
        u = torch.rand(size=(batch_size,), generator=generator)
        indices = (u * len(scheduler.timesteps)).long()
        sigmas = scheduler.sigmas[indices].flatten()

    # fixed u -> no scheduler, those are just the sigmas
    else:
        # sigmas = u.clone()
        indices = (u * len(scheduler.timesteps)).long()
        sigmas = scheduler_sigmas[indices].flatten()

    timesteps = scheduler.timesteps[indices]
    timesteps = timesteps.to(model_input.device)
    sigmas = sigmas.to(model_input.device)
    timesteps = timesteps[:, None, None]
    sigmas = sigmas[:, None, None]

    # interpolate between model_input and noise
    noisy_model_input = (1.0 - sigmas) * model_input.to(sigmas.dtype) + sigmas * noise
    noisy_model_input = noisy_model_input.to(model_input.dtype)
    
    # the target in flow matching is the "velocity"
    target = noise - model_input
    return noisy_model_input, target, timesteps, {"sigmas": sigmas, "noise": noise, "u": u}

def fm_clean_estimate(scheduler, latents, noise_pred, timesteps):
    assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler), "Only FlowMatchEulerDiscreteScheduler is supported"
    step_indices = [(scheduler.timesteps == t).nonzero().item() for t in timesteps]
    sigma = scheduler.sigmas[step_indices]
    sigma = sigma.to(device=latents.device, dtype=latents.dtype)
    pred_x0 = latents - sigma * noise_pred
    return pred_x0

# ==========================
#   Generic Sampling Code
# ==========================
@torch.no_grad()
def sample(
    model,
    latents,
    num_timesteps=20,
    **kwargs
):
    """
    Generate activations from pure noise.
    We recommend setting `num_timesteps` based on your priorities:
    - 20: moderate quality at fast speed
    - 100: good quality at reasonable speed
    - 1000: best quality for diffusion purists
    """
    model.scheduler.set_timesteps(num_timesteps)
    model.scheduler.timesteps = model.scheduler.timesteps.to(latents.device)
    for i, timestep in enumerate(model.scheduler.timesteps):
        timesteps = timestep.repeat(latents.shape[0], 1)
        noise_pred = model.denoiser(
            latents=latents,
            timesteps=timesteps,
            **kwargs
        )
        latents = model.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
    return latents

@torch.no_grad()
def sample_on_manifold(
    model, 
    latents, 
    num_timesteps=20, 
    start_timestep=None,
    **kwargs
):
    """
    Post-process activations into their on-manifold counterpart.
    See the `sample` function above for recommendations on `num_timesteps`.
    This is essentially the activation-space analogue of SDEdit (Meng et. al., 2022).
    """
    start_latents = latents.clone()
    model.scheduler.set_timesteps(num_timesteps)
    for i, timestep in enumerate(model.scheduler.timesteps):
        if start_timestep is not None and torch.is_tensor(start_timestep):
            # inject original latents until start_timestep
            timestep_mask = start_timestep[:, 0, 0] <= timestep
            latents[timestep_mask] = start_latents[timestep_mask]
        elif start_timestep is not None and timestep > start_timestep:
            continue
        timesteps = timestep[None, ...]
        noise_pred = model.denoiser(
            latents=latents,
            timesteps=timesteps.repeat(latents.shape[0], 1, 1),
            **kwargs
        )
        latents = model.scheduler.step(noise_pred, timesteps, latents, return_dict=False)[0]
    return latents


# ==========================
#   Density Estimation
# ==========================
def _log_prob_hutchinson(
    model,
    latents,
    num_steps=100,
    num_hutchinson_samples=1,
    layer_idx=None,
    generator=None,
    normalize=True,
):
    """CNF change-of-variables log p(x) via Hutchinson trace estimation."""
    device = latents.device
    dtype = latents.dtype
    b, s, d = latents.shape # B, L, D

    # normalize x_in
    if normalize:
        x = model.normalizer.normalize(latents, layer_idx=layer_idx)
        var = model.normalizer.get_layer_stat(model.normalizer.var, layer_idx)
        log_det_normalize = (-0.5 * torch.log(var).sum() * s).to(torch.float32)
    else:
        x = latents.clone()
        log_det_normalize = torch.tensor(0.0, device=device, dtype=torch.float32)

    model.scheduler.set_timesteps(model.scheduler.config.num_train_timesteps)

    sigma_min, sigma_max = 1e-5, 1.0 - 1e-5
    sigmas = torch.linspace(sigma_min, sigma_max, num_steps + 1, device=device)
    dt = (sigma_max - sigma_min) / num_steps

    log_det_flow = torch.zeros(b, device=device, dtype=torch.float32)

    for i in range(num_steps):
        sigma = sigmas[i].item()
        index = min(int(sigma * 1000), 999)
        timestep_val = model.scheduler.timesteps[index].item()
        timestep = torch.full((b, s), timestep_val, device=device, dtype=torch.long)

        trace_est = torch.zeros(b, device=device, dtype=torch.float32)
        x_in = x.detach().to(dtype).requires_grad_(True)
        v = model.denoiser(latents=x_in, timesteps=timestep, layer_idx=layer_idx)
        for j in range(num_hutchinson_samples):
            eps = torch.randn(b, s, d, device=device, dtype=dtype, generator=generator)
            v_dot_eps = (v * eps).sum()
            # Intuitively: the velocity is learned to be subtracted, 
            # hence we maximize the velocity in order to move back to x_in.
            # Average over multiple traces on the way back. These gradients are accumulated
            # as the determinant
            # formula: grad = grad_{x_in} (denoiser(x_s) * noise)
            grad = torch.autograd.grad(
                v_dot_eps, x_in,
                retain_graph=(j < num_hutchinson_samples - 1),
            )[0]
            trace_est = trace_est + (grad * eps).sum(dim=(-2, -1)).float()
        trace_est = trace_est / num_hutchinson_samples
        # progressively noise x_in with the denoiser's timestep noise
        x = x_in.detach() + dt * v.detach()
        # to compute the trace 
        log_det_flow = log_det_flow - dt * trace_est

    z = x.detach()
    # noisy sample logp
    log_p_base = -0.5 * (d * s * math.log(2 * math.pi) + (z.float() ** 2).sum(dim=(-2, -1)))
    log_prob_val = log_p_base + log_det_flow + log_det_normalize

    return SimpleNamespace(
        log_prob=log_prob_val,
        prob=log_prob_val.exp(),
        z=z,
        log_p_base=log_p_base,
        log_det_flow=log_det_flow,
        log_det_normalize=log_det_normalize,
    )


def dte_posterior(
    model,
    latents,
    reference_latents,
    K: int = 5,
    num_sigma_bins: int = 100,
    layer_idx=None,
    normalize: bool = True,
):
    """
    Non-parametric DTE-InverseGamma posterior p(sigma_t^2 | x) using KNN.

    Based on Livernoche et al. "On Diffusion Modeling for Anomaly Detection"
    (DTE-NP / DTE-IG variant): for a test point x with mean K-NN distance d,
    the InvGamma posterior over the noise variance is

        p(sigma_t^2 | x) proportional to InvGamma(sigma_t^2; a=0.5*D-1, scale=d^2/2)

    whose mode is d^2/D_eff, independent of absolute scale. The sigma grid is
    set adaptively from the reference set's own K-NN distances so the posterior
    is always well-supported.

    The classifier score p_clean is defined relative to the reference set:
        p_clean = exp(-(d_test / d_ref)^2)
    so in-distribution points (d_test ~ d_ref) score ~0.37 and OOD points score
    near 0. This is stable across all dimensionalities.

    Args:
        model: GLP instance (used for the normalizer when normalize=True).
        latents: (B, S, D) test activations.
        reference_latents: (N_ref, S, D) reference "clean" activations.
        K: number of nearest neighbors.
        num_sigma_bins: resolution of the sigma grid for the posterior.
        layer_idx: layer index for the normalizer (multi-layer models).
        normalize: whether to apply model.normalizer.normalize to both sets
                   before computing distances.

    Returns:
        SimpleNamespace with:
          posterior:      (B, num_sigma_bins) normalized p(sigma_t^2 | x)
          sigmas:         (num_sigma_bins,) sigma² grid (centered on ref mode)
          p_clean:        (B,) exp(-(d/d_ref)^2), in [0,1], higher = in-dist
          expected_sigma: (B,) E[sigma²] under posterior, normalized by ref mode
          mode_sigma:     (B,) argmax sigma² under posterior
          knn_dist:       (B,) mean K-NN distance to reference set
          ref_knn_dist:   scalar, mean K-NN distance within reference (LOO)
          log_prob:       (B,) log(p_clean) (uniform interface with Hutchinson)
          prob:           (B,) p_clean (uniform interface)
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.stats import invgamma

    assert latents.ndim == 3 and reference_latents.ndim == 3, \
        f"latents / reference_latents must be (N, S, D), got {latents.shape} / {reference_latents.shape}"
    assert latents.shape[1:] == reference_latents.shape[1:], \
        f"shape mismatch: {latents.shape} vs {reference_latents.shape}"

    if normalize:
        latents = model.normalizer.normalize(latents, layer_idx=layer_idx)
        reference_latents = model.normalizer.normalize(reference_latents, layer_idx=layer_idx)

    b, s, d = latents.shape
    d_eff = s * d
    eps = 1e-30

    x_test = latents.detach().float().cpu().reshape(b, d_eff).numpy()
    x_ref = reference_latents.detach().float().cpu().reshape(reference_latents.shape[0], d_eff).numpy()

    k = min(K, x_ref.shape[0])

    # K-NN distances from test → reference
    neigh = NearestNeighbors(n_neighbors=k, metric="minkowski", p=2).fit(x_ref)
    dist_test, _ = neigh.kneighbors(x_test)  # (B, k)
    mean_dist_test = dist_test.mean(axis=-1)  # (B,)

    # LOO K-NN distances within reference (skip self at index 0)
    k_ref = min(K + 1, x_ref.shape[0])
    neigh_ref = NearestNeighbors(n_neighbors=k_ref, metric="minkowski", p=2).fit(x_ref)
    dist_ref, _ = neigh_ref.kneighbors(x_ref)   # (N_ref, k+1)
    mean_dist_ref = dist_ref[:, 1:].mean(axis=-1).mean()  # scalar: skip self (col 0)

    # Classifier score: exp(-(d_test / d_ref)^2), in [0,1]
    p_clean_np = np.exp(-(mean_dist_test / max(mean_dist_ref, eps)) ** 2)
    p_clean = torch.from_numpy(p_clean_np).float()

    # InvGamma posterior on an adaptive sigma² grid
    # InvGamma mode = beta/(a+1) = (d²/2) / (D/2) = d²/D_eff
    # Set grid to [0.01*ref_mode, 10*ref_mode] to cover both in-dist and OOD
    ref_mode = (mean_dist_ref ** 2) / max(d_eff, 1)
    sigma2_min = max(ref_mode * 0.01, eps)
    sigma2_max = ref_mode * 100.0
    sigma2_np = np.linspace(sigma2_min, sigma2_max, num_sigma_bins)

    beta_test = (mean_dist_test ** 2) / 2.0  # (B,)
    a = max(0.5 * d_eff - 1, eps)
    log_post = invgamma.logpdf(
        sigma2_np[None, :],
        a=a,
        loc=0.0,
        scale=np.maximum(beta_test[:, None], eps),
    )
    # normalize across sigma bins (log-sum-exp)
    log_post -= np.log(np.exp(log_post - log_post.max(axis=-1, keepdims=True)).sum(axis=-1, keepdims=True) + eps) + log_post.max(axis=-1, keepdims=True)
    posterior = np.exp(log_post).astype(np.float32)

    posterior_t = torch.from_numpy(posterior)
    sigmas_t = torch.from_numpy(sigma2_np.astype(np.float32))
    ref_mode_sigma2 = max(ref_mode, eps)

    expected_sigma = (posterior_t * sigmas_t[None, :]).sum(dim=-1) / ref_mode_sigma2
    mode_sigma = sigmas_t[posterior_t.argmax(dim=-1)]
    knn_dist = torch.from_numpy(mean_dist_test.astype(np.float32))

    log_prob_val = torch.log(p_clean.clamp_min(eps))

    return SimpleNamespace(
        posterior=posterior_t,
        sigmas=sigmas_t,
        p_clean=p_clean,
        expected_sigma=expected_sigma,
        mode_sigma=mode_sigma,
        knn_dist=knn_dist,
        ref_knn_dist=float(mean_dist_ref),
        log_prob=log_prob_val,
        prob=p_clean,
    )


def log_prob(
    model,
    latents,
    method: str = "hutchinson",
    # hutchinson kwargs
    num_steps: int = 100,
    num_hutchinson_samples: int = 1,
    # dte kwargs
    reference_latents=None,
    K: int = 5,
    num_sigma_bins: int = 100,
    # common
    layer_idx=None,
    generator=None,
    normalize: bool = True,
):
    """
    Compute a per-sample score under the learned flow model.

    method="hutchinson" (default):
        log p(x) via CNF change-of-variables + Hutchinson trace estimator.
        Returns a SimpleNamespace with .log_prob, .prob (=exp(log_prob), may
        underflow to 0 in high dims), .z, .log_p_base, .log_det_flow,
        .log_det_normalize.

    method="dte":
        Non-parametric DTE-InverseGamma posterior p(sigma_t | x) via KNN on
        reference_latents. Returns .log_prob (=log p_clean), .prob (=p_clean
        in [0,1]) plus .posterior, .sigmas, .p_clean, .expected_sigma,
        .mode_sigma, .knn_dist.
    """
    if method == "hutchinson":
        return _log_prob_hutchinson(
            model, latents,
            num_steps=num_steps,
            num_hutchinson_samples=num_hutchinson_samples,
            layer_idx=layer_idx,
            generator=generator,
            normalize=normalize,
        )
    if method == "dte":
        assert reference_latents is not None, \
            "method='dte' requires reference_latents (clean KNN reference set)"
        return dte_posterior(
            model, latents, reference_latents,
            K=K,
            num_sigma_bins=num_sigma_bins,
            layer_idx=layer_idx,
            normalize=normalize,
        )
    raise ValueError(f"unknown log_prob method: {method!r} (expected 'hutchinson' or 'dte')")