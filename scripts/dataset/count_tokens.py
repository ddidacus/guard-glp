"""Count prompt-token statistics for a build config, before extraction.

Reports, for BOTH prompt views (``user`` and ``full``), how many tokens the
configured tokenizer produces per prompt — so we can size the extraction and see
how many prompts would be truncated at ``extract.max_length``. The prompt strings
are built with the exact same code path as extraction (``load_texts``), so the
counts match what will actually be fed to the model.

    python scripts/dataset/count_tokens.py run --config=CONFIG

``load_dotenv()`` runs first so a local ``.env`` (HF cache paths etc.) is honored
before transformers/datasets resolve caches.
"""

import dataclasses
import logging

import fire
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_VIEWS = ("user", "full")


def _percentile(sorted_vals: list[int], q: float) -> int:
    """Nearest-rank percentile of an already-sorted list (q in [0, 100])."""
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, int(round(q / 100.0 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def _summarize(lengths: list[int], max_length: int) -> dict[str, float]:
    total = sum(lengths)
    n = len(lengths)
    ordered = sorted(lengths)
    return {
        "n_prompts": n,
        "total_tokens": total,
        "mean": (total / n) if n else 0.0,
        "median": _percentile(ordered, 50),
        "min": ordered[0] if ordered else 0,
        "max": ordered[-1] if ordered else 0,
        "p90": _percentile(ordered, 90),
        "p99": _percentile(ordered, 99),
        "n_over_max_length": sum(1 for length in lengths if length > max_length),
    }


def run(config: str) -> None:
    """Print token-count stats for the ``user`` and ``full`` views of ``config``."""
    load_dotenv()
    from transformers import AutoTokenizer

    from glp.dataset.backends import resolve_add_special_tokens
    from glp.dataset.builder import BuildConfig
    from glp.dataset.loader import load_texts

    cfg = BuildConfig.from_yaml(config)
    if cfg.dataset.format != "chat":
        logger.warning(
            "prompt_view only applies to chat datasets; format=%r — counting the "
            "single text view.",
            cfg.dataset.format,
        )
        views: tuple[str, ...] = ("full",)
    else:
        views = _VIEWS

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    add_special_tokens = resolve_add_special_tokens(cfg)
    max_length = cfg.extract.max_length

    print(f"model={cfg.model_name}  dataset={cfg.dataset.path}")
    print(f"add_special_tokens={add_special_tokens}  max_length={max_length}\n")

    for view in views:
        ds_cfg = dataclasses.replace(cfg.dataset, prompt_view=view)
        texts = load_texts(ds_cfg, tokenizer, gpu_id=0, num_gpus=1)
        encoded = tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            padding=False,
            truncation=False,
        )["input_ids"]
        lengths = [len(ids) for ids in encoded]
        stats = _summarize(lengths, max_length)
        print(f"── view={view} ──")
        for key, val in stats.items():
            if isinstance(val, float):
                print(f"  {key:18s}: {val:,.2f}")
            else:
                print(f"  {key:18s}: {val:,}")
        print()


if __name__ == "__main__":
    fire.Fire({"run": run})
