"""Build trainer-ready activation datasets (per-shard extract + finalize).

Two subcommands following the repo's ``fire`` two-pass convention:

    # pass 1: one process per data shard (one GPU each for hf_baukit)
    python scripts/dataset/build_activations.py run --config=CONFIG --gpu_id=0

    # pass 2: merge shards (single process, no GPU)
    python scripts/dataset/build_activations.py finalize --config=CONFIG

On a SLURM cluster these are submitted for you by
``scripts/dataset/build_activations.sh`` (array pass-1 + dependent CPU finalize).

``glp.dataset.builder`` is imported lazily inside each command, after
``load_dotenv()``, so a local ``.env`` (e.g. ``HF_HOME``/``UV_CACHE_DIR`` on the
shared filesystem) is honored before transformers/datasets resolve cache paths.
"""

import logging

import fire
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run(config: str, gpu_id: int = 0, device: str | None = None) -> None:
    """Extract one data shard of activations from the YAML ``config``.

    ``gpu_id`` is the shard index (strides the corpus); ``device`` overrides CUDA
    device selection (pass ``cuda:0`` under a SLURM array task whose single GPU is
    pinned via ``CUDA_VISIBLE_DEVICES``).
    """
    load_dotenv()
    from glp.dataset.builder import BuildConfig, build_shard

    cfg = BuildConfig.from_yaml(config)
    print(
        f"[shard {gpu_id}/{cfg.num_gpus}] backend={cfg.backend} "
        f"model={cfg.model_name} device={device or 'auto'}"
    )
    print(
        f"[shard {gpu_id}] layers={cfg.extract.layers} gran={cfg.extract.granularity}"
    )
    build_shard(cfg, gpu_id=gpu_id, device=device)


def finalize_cmd(config: str) -> None:
    """Merge all shards produced by ``run`` into final dataset directories."""
    load_dotenv()
    from glp.dataset.builder import BuildConfig, finalize

    cfg = BuildConfig.from_yaml(config)
    print(f"[finalize] merging shards under {cfg.output_dir}")
    finalize(cfg)


if __name__ == "__main__":
    fire.Fire({"run": run, "finalize": finalize_cmd})
