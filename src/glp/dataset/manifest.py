"""Self-documenting provenance manifest for produced activation datasets.

Every finalized ``layer_<NN>/`` directory carries a ``manifest.json`` recording
exactly how it was built (source dataset, model, layer, granularity, sample
count, dtype, filters, git SHA, UTC timestamp). This satisfies the standing
action item to document datasets: how they were built, how many examples, and
in what format. The loader ignores it.
"""

import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from glp.dataset.builder import BuildConfig

logger = logging.getLogger(__name__)


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def build_manifest(
    cfg: "BuildConfig",
    granularity: str,
    layer: int,
    num_samples: int,
    dim: int,
    dtype_str: str,
) -> dict[str, Any]:
    """Assemble the manifest dict for one (granularity, layer) dataset."""
    return {
        "source_dataset": {
            "path": cfg.dataset.path,
            "name": cfg.dataset.name,
            "split": cfg.dataset.split,
            "revision": cfg.dataset.revision,
            "format": cfg.dataset.format,
        },
        "model": cfg.model_name,
        "backend": cfg.backend,
        "layer": layer,
        "layer_prefix": cfg.extract.layer_prefix,
        "retain": cfg.extract.retain,
        "granularity": granularity,
        "num_samples": num_samples,
        "dim": dim,
        "dtype": dtype_str,
        "config_dtype": cfg.extract.dtype,
        "max_length": cfg.extract.max_length,
        "filters": [
            {"column": f.column, "equals": f.equals} for f in cfg.dataset.filters
        ],
        "dedup": cfg.dataset.dedup,
        "max_samples": cfg.dataset.max_samples,
        "git_sha": _git_sha(),
        "created_utc": datetime.now(UTC).isoformat(),
    }


def write_manifest(final_dir: str | Path, manifest: dict[str, Any]) -> None:
    path = Path(final_dir) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("wrote manifest %s", path)
