"""Standalone generation-FD evaluation of a trained GLP checkpoint.

Generates activations from pure noise and computes the Fréchet Distance to a held-out
set of real activations, plus the lower bound FD(real_a, real_b). Paper-faithful defaults
(50k samples, 1000 diffusion steps).

    python scripts/eval/eval_fd.py config=configs/eval/fd_llama1b_guardglpbenign_alltok_layer14.yaml \
        view=useronly checkpoint=final
    # quick: num_timesteps=100 n_samples=5000 ; a mid-run checkpoint: checkpoint=step_45000

Loads local run dirs (runs/<name>/) directly — not via load_glp, whose existence check
only resolves Hub repo ids, not local ``<checkpoint>.safetensors`` files.
"""

import json
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DEFAULTS = {
    "view": "useronly",
    "layer": "14",
    "weights_folder": "",
    "checkpoint": "final",
    "dataset": "",
    "n_samples": 50000,
    "num_timesteps": 1000,
    "layer_idx": None,
    "batch_size": 4096,
    "seed": 0,
    "out": None,
    "wandb_enabled": False,
    "wandb_project": "guard-glp",
    "wandb_run_name": None,
}


def _load_local_glp(weights_folder: str, checkpoint: str, device: str) -> Any:
    """Build a GLP from a local run dir + load ``<checkpoint>.safetensors``."""
    from glp.denoiser import GLP

    run_cfg = OmegaConf.load(f"{weights_folder}/config.yaml")
    glp_kwargs = OmegaConf.to_container(run_cfg.glp_kwargs, resolve=True)
    assert isinstance(glp_kwargs, dict)
    # use the run dir's own normalization stats (saved alongside the weights)
    glp_kwargs["normalizer_config"] = {
        "rep_statistic": f"{weights_folder}/rep_statistics.pt"
    }
    model = GLP(**glp_kwargs)
    model.to(device)
    model.load_pretrained(weights_folder, name=checkpoint)
    return model


def main() -> None:
    load_dotenv()
    cli = OmegaConf.from_cli()
    config_path = cli.pop("config", None)
    device = str(cli.pop("device", "cuda:0"))
    base = OmegaConf.create(_DEFAULTS)
    OmegaConf.set_struct(base, False)
    file_cfg = OmegaConf.load(config_path) if config_path else OmegaConf.create()
    cfg = OmegaConf.merge(base, file_cfg, cli)

    from glp.dataset.act_dataset import load_activation_dataset
    from glp.eval.fd import draw_real_pair, generation_fd

    logger.info(
        "loading GLP from %s (checkpoint=%s)", cfg.weights_folder, cfg.checkpoint
    )
    model = _load_local_glp(str(cfg.weights_folder), str(cfg.checkpoint), device)

    dataset = load_activation_dataset(cfg.dataset)
    real_a, real_b = draw_real_pair(dataset, int(cfg.n_samples), int(cfg.seed))
    report = generation_fd(
        model,
        real_a,
        real_b,
        num_timesteps=int(cfg.num_timesteps),
        batch_size=int(cfg.batch_size),
        seed=int(cfg.seed),
        layer_idx=cfg.layer_idx,
        device=device,
    )
    report.update(
        {
            "weights_folder": str(cfg.weights_folder),
            "checkpoint": str(cfg.checkpoint),
            "dataset": str(cfg.dataset),
        }
    )
    print(json.dumps(report, indent=2))

    out_dir = Path(str(cfg.out) if cfg.out else f"{cfg.weights_folder}/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"fd_{cfg.checkpoint}_t{cfg.num_timesteps}.json"
    out_file.write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", out_file)

    if cfg.wandb_enabled:
        import wandb

        run = wandb.init(
            project=str(cfg.wandb_project),
            name=str(cfg.wandb_run_name or f"fd-{cfg.view}-layer{cfg.layer}"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        run.summary["fd"] = report["fd"]
        run.summary["fd_lower_bound"] = report["lower_bound"]
        run.finish()


if __name__ == "__main__":
    main()
