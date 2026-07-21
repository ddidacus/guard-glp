"""CPU-only test for the GLP training loop (no network, no GPU).

Builds a tiny synthetic activation dataset on disk (via ``MemmapWriter`` +
``rep_statistics.pt``), runs :func:`glp.train.train` for one short epoch on CPU
with a tiny denoiser, and asserts the produced checkpoints reload to the same
weights. This exercises the full port: dataset consumer -> normalizing collator
-> ``GLP.forward`` (flow-matching MSE) -> optimizer/scheduler -> checkpointing.
"""

from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from glp.denoiser import GLP
from glp.train import train
from glp.utils_acts import MemmapWriter

DIM = 16
N = 128


def _write_synthetic_dataset(layer_dir: Path) -> None:
    layer_dir.mkdir(parents=True)
    torch.manual_seed(0)
    mean = torch.randn(DIM) * 3
    std = torch.rand(DIM) * 2 + 0.5
    acts = torch.randn(N, DIM) * std + mean

    writer = MemmapWriter(
        output_dir=layer_dir, file_size=1 << 20, dtype=np.dtype(np.float32)
    )
    for i in range(N):
        writer.write(acts[i].numpy().astype(np.float32))
    writer.flush()
    (layer_dir / "dtype.txt").write_text("float32")
    torch.save({"mean": mean, "var": std**2}, layer_dir / "rep_statistics.pt")


def _make_config(layer_dir: Path, out: Path) -> OmegaConf:
    rep = str(layer_dir / "rep_statistics.pt")
    return OmegaConf.create(
        {
            "output_path": str(out),
            "train_dataset": str(layer_dir),
            "rep_statistic": rep,
            "use_bf16": False,  # CPU
            "num_epochs": 1,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "log_every_n_steps": 2,
            "save_epochs": [1],
            "save_opt_state": True,
            "lr_scheduler": {
                "scheduler_cls": "cosine_scheduler_with_warmup",
                "warmup_ratio": 0.1,
                "initial_factor": 0.01,
                "final_factor": 0.1,
            },
            "glp_kwargs": {
                "normalizer_config": {"rep_statistic": rep},
                "denoiser_config": {
                    "d_input": DIM,
                    "d_model": 2 * DIM,
                    "d_mlp": 4 * DIM,
                    "n_layers": 2,
                    "multi_layer_n_layers": None,
                },
                "tracedict_config": {
                    "layer_prefix": "model.layers",
                    "layers": [8],
                    "retain": "output",
                },
            },
        }
    )


def test_train_round_trip(tmp_path: Path) -> None:
    layer_dir = tmp_path / "data" / "last" / "layer_08"
    _write_synthetic_dataset(layer_dir)
    out = tmp_path / "run"
    config = _make_config(layer_dir, out)

    model = train(config, device="cpu")

    # all expected artifacts written
    assert (out / "final.safetensors").exists()
    assert (out / "rep_statistics.pt").exists()
    assert (out / "config.yaml").exists()
    assert (out / "checkpoints" / "epoch_1.safetensors").exists()
    assert (out / "optimizer_state.pt").exists()

    # checkpoint reloads to identical weights
    reloaded = GLP(**OmegaConf.to_container(config.glp_kwargs, resolve=True))
    reloaded.to("cpu")
    reloaded.load_pretrained(out, name="final")
    for (_, p1), (_, p2) in zip(
        model.named_parameters(), reloaded.named_parameters(), strict=True
    ):
        assert torch.allclose(p1.cpu(), p2.cpu())
