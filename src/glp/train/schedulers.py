"""LR-schedule lambdas for the GLP trainer.

Ported verbatim from ``generative_latent_prior/glp_train.py:136-159``. Each
function returns a multiplicative factor applied to the base learning rate by a
``torch.optim.lr_scheduler.LambdaLR``. ``*_with_warmup`` variants ramp linearly
from ``initial_factor`` to ``1.0`` over ``warmup_steps`` then decay to
``final_factor`` by ``max_steps``. The config's ``lr_scheduler.scheduler_cls``
names one of these; the trainer resolves it by name against this module.
"""

import math

__all__ = [
    "linear_scheduler",
    "linear_scheduler_with_warmup",
    "cosine_scheduler",
    "cosine_scheduler_with_warmup",
    "get_scheduler_fn",
]


def linear_scheduler(
    step: float, max_steps: float, initial_factor: float, final_factor: float
) -> float:
    alpha = step / max_steps
    return alpha * final_factor + (1 - alpha) * initial_factor


def linear_scheduler_with_warmup(
    step: int,
    *,
    warmup_steps: float,
    max_steps: float,
    initial_factor: float,
    final_factor: float,
) -> float:
    if step < warmup_steps:
        return linear_scheduler(step, warmup_steps, initial_factor, 1.0)
    if step >= max_steps:
        return final_factor
    return linear_scheduler(
        step - warmup_steps, max_steps - warmup_steps, 1.0, final_factor
    )


def cosine_scheduler(
    step: float, max_steps: float, initial_factor: float, final_factor: float
) -> float:
    alpha = step / max_steps
    cosine_out = 0.5 * (1 + math.cos(math.pi * alpha))
    return final_factor + (initial_factor - final_factor) * cosine_out


def cosine_scheduler_with_warmup(
    step: int,
    *,
    warmup_steps: float,
    max_steps: float,
    initial_factor: float,
    final_factor: float,
) -> float:
    if step < warmup_steps:
        return linear_scheduler(step, warmup_steps, initial_factor, 1.0)
    if step >= max_steps:
        return final_factor
    return cosine_scheduler(
        step - warmup_steps, max_steps - warmup_steps, 1.0, final_factor
    )


_SCHEDULERS = {
    fn.__name__: fn
    for fn in (
        linear_scheduler,
        linear_scheduler_with_warmup,
        cosine_scheduler,
        cosine_scheduler_with_warmup,
    )
}


def get_scheduler_fn(name: str):
    """Resolve a scheduler lambda by name (avoids ``eval`` on config strings)."""
    try:
        return _SCHEDULERS[name]
    except KeyError:
        raise ValueError(
            f"unknown scheduler_cls {name!r}; choose one of {sorted(_SCHEDULERS)}"
        ) from None
