"""GLP training: config-driven flow-matching diffusion trainer.

Ported from ``generative_latent_prior/glp_train.py`` and adapted to reuse
guard-glp's activation consumer and model. See :func:`train`.
"""

from glp.train.trainer import TrainConfig, save_checkpoint, train

__all__ = ["TrainConfig", "train", "save_checkpoint"]
