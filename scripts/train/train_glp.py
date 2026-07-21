"""Train a GLP on built activations.

OmegaConf-CLI entry point (matches the ``${...}`` interpolation in the training
configs and the reference invocation):

    python scripts/train/train_glp.py config=configs/train/glp_llama1b_guardglpbenign.yaml
    # override any field, e.g. a different built dataset / architecture / device:
    python scripts/train/train_glp.py config=<CFG> device=cuda:1 \
        train_dataset=<DIR> glp_kwargs.denoiser_config.n_layers=6

``load_dotenv()`` runs first so a local ``.env`` (HF cache paths etc.) is honored.
"""

import logging
from typing import cast

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    load_dotenv()
    from glp.train import TrainConfig, train

    config_base = OmegaConf.structured(TrainConfig())
    OmegaConf.set_struct(config_base, False)
    config_cli = OmegaConf.from_cli()
    config_path = config_cli.pop("config", None)
    device = config_cli.pop("device", "cuda:0")
    config_file = OmegaConf.load(config_path) if config_path else OmegaConf.create()
    config = cast(DictConfig, OmegaConf.merge(config_base, config_file, config_cli))

    train(config, device=str(device))


if __name__ == "__main__":
    main()
