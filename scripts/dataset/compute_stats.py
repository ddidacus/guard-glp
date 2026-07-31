"""Stats pre-pass: stacked per-layer normalization statistics for streaming runs.

Streaming training (``data_mode: streaming``) needs per-layer mean/var *before*
the first step, but its data is never written to disk — so this pre-pass streams
a capped number of activations (``extract.max_tokens``) through the extraction
backend and writes a **stacked** ``rep_statistics.pt`` with ``(n_layers, D)``
mean/var indexed by ABSOLUTE layer id (rows for unextracted layers are NaN so
misuse fails loudly). Consumes a plain dataset-build YAML (``BuildConfig``):

    # single process, one GPU; ~2M tokens is ample for (16, D) mean/var
    python scripts/dataset/compute_stats.py run \
        --config=configs/dataset/stats_guardglpbenign_llama1b_all16.yaml

    # alternative: stack per-layer rep_statistics.pt of already-built static
    # dataset dirs (each named layer_<idx>) into one multi-layer stats file
    python scripts/dataset/compute_stats.py stack-stats \
        --out=data/foo/rep_statistics.pt --n_layers_total=16 \
        data/foo/last/layer_08 data/foo/last/layer_12 data/foo/last/layer_14

The produced file is what a streaming training config points its
``rep_statistic`` / ``glp_kwargs.normalizer_config.rep_statistic`` at.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import fire
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run(
    config: str, device: str | None = None, n_layers_total: int | None = None
) -> None:
    """Stream activations per the YAML ``config`` and write stacked stats.

    ``n_layers_total`` sizes the stacked table; by default it is read from the
    model's HF config (``num_hidden_layers``), so it only needs overriding for
    models whose block count is not discoverable that way.
    """
    load_dotenv()
    import torch

    from glp.dataset.builder import BuildConfig, compute_layer_stats
    from glp.dataset.manifest import git_sha
    from glp.dataset.stats import stacked_normalizer_tensors

    cfg = BuildConfig.from_yaml(config)
    if n_layers_total is None:
        from transformers import AutoConfig

        n_layers_total = int(
            AutoConfig.from_pretrained(cfg.model_name).num_hidden_layers
        )
    print(
        f"[stats] backend={cfg.backend} model={cfg.model_name} "
        f"layers={cfg.extract.layers} granularity={cfg.extract.granularity} "
        f"max_tokens={cfg.extract.max_tokens} n_layers_total={n_layers_total}"
    )

    per_layer, tokens_consumed = compute_layer_stats(cfg, device=device)
    mean, var = stacked_normalizer_tensors(per_layer, n_layers_total)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": mean, "var": var}, out_dir / "rep_statistics.pt")

    manifest = {
        "kind": "stacked_rep_statistics",
        "source_dataset": {
            "path": cfg.dataset.path,
            "name": cfg.dataset.name,
            "split": cfg.dataset.split,
            "revision": cfg.dataset.revision,
            "format": cfg.dataset.format,
            "prompt_view": cfg.dataset.prompt_view,
        },
        "model": cfg.model_name,
        "backend": cfg.backend,
        "layers": sorted(per_layer),
        "n_layers_total": n_layers_total,
        "layer_prefix": cfg.extract.layer_prefix,
        "retain": cfg.extract.retain,
        "granularity": cfg.extract.granularity[0],
        "max_length": cfg.extract.max_length,
        "tokens_consumed": tokens_consumed,
        "samples_per_layer": {
            str(layer): int(stats.count) for layer, stats in per_layer.items()
        },
        "dim": mean.shape[1],
        "git_sha": git_sha(),
        "created_utc": datetime.now(UTC).isoformat(),
    }
    (out_dir / "stats_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"[stats] wrote {out_dir / 'rep_statistics.pt'} "
        f"(({n_layers_total}, {mean.shape[1]}), layers {sorted(per_layer)}, "
        f"{tokens_consumed} tokens)"
    )


def stack_stats(*layer_dirs: str, out: str, n_layers_total: int) -> None:
    """Stack per-layer-dir ``rep_statistics.pt`` files into one stacked file."""
    load_dotenv()
    from glp.dataset.stats import stack_rep_statistics

    stack_rep_statistics([Path(d) for d in layer_dirs], int(n_layers_total), Path(out))
    print(f"[stats] wrote stacked stats to {out}")


if __name__ == "__main__":
    fire.Fire({"run": run, "stack-stats": stack_stats})
