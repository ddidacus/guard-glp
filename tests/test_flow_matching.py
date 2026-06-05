"""CPU-only unit tests for flow-matching input preparation (no model/GPU/downloads)."""

import torch

from glp import flow_matching


def test_fm_scheduler_type() -> None:
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )

    assert isinstance(flow_matching.fm_scheduler(), FlowMatchEulerDiscreteScheduler)


def test_fm_prepare_shapes_and_target() -> None:
    """fm_prepare returns matching shapes and the flow-matching velocity target."""
    scheduler = flow_matching.fm_scheduler()
    scheduler.set_timesteps(10)

    model_input = torch.randn(4, 1, 6)
    noise = torch.randn(4, 1, 6)
    u = torch.full((4,), 0.5)

    noisy, target, timesteps, meta = flow_matching.fm_prepare(
        scheduler, model_input, noise, u=u
    )

    assert noisy.shape == model_input.shape
    # the flow-matching target is the velocity: noise - model_input
    assert torch.allclose(target, noise - model_input)
    # timesteps are broadcast to (batch, 1, 1)
    assert timesteps.shape == (4, 1, 1)
    assert set(meta) == {"sigmas", "noise", "u"}


def test_fm_prepare_rejects_non_3d() -> None:
    """A non (batch, seq, dim) input raises a clear error."""
    import pytest

    scheduler = flow_matching.fm_scheduler()
    scheduler.set_timesteps(10)
    with pytest.raises(ValueError, match="batch, seq, dim"):
        flow_matching.fm_prepare(scheduler, torch.randn(4, 6), torch.randn(4, 6))
