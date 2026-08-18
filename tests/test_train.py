"""CPU-only test for the GLP training loop (no network, no GPU).

Builds a tiny synthetic activation dataset on disk (via ``MemmapWriter`` +
``rep_statistics.pt``), runs :func:`glp.train.train` for one short epoch on CPU
with a tiny denoiser, and asserts the produced checkpoints reload to the same
weights. This exercises the full port: dataset consumer -> normalizing collator
-> ``GLP.forward`` (flow-matching MSE) -> optimizer/scheduler -> checkpointing.
"""

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

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


def _make_config(layer_dir: Path, out: Path) -> DictConfig:
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
    glp_kwargs = cast(
        dict[str, Any], OmegaConf.to_container(config.glp_kwargs, resolve=True)
    )
    reloaded = GLP(**glp_kwargs)
    reloaded.to("cpu")
    reloaded.load_pretrained(out, name="final")
    for (_, p1), (_, p2) in zip(
        model.named_parameters(), reloaded.named_parameters(), strict=True
    ):
        assert torch.allclose(p1.cpu(), p2.cpu())


def test_resume_continues_from_checkpoint(tmp_path: Path) -> None:
    layer_dir = tmp_path / "data" / "last" / "layer_08"
    _write_synthetic_dataset(layer_dir)
    out = tmp_path / "run"

    # first run: 1 epoch (128 samples / batch 32 = 4 gradient steps), save opt state
    cfg1 = _make_config(layer_dir, out)
    cfg1.save_opt_state = True
    train(cfg1, device="cpu")

    state = torch.load(out / "train_state.pt")
    assert state["num_gradient_steps"] == 4
    assert state["checkpoint_name"] == "final"

    glp_kwargs = cast(
        dict[str, Any], OmegaConf.to_container(cfg1.glp_kwargs, resolve=True)
    )
    after_run1 = GLP(**glp_kwargs)
    after_run1.load_pretrained(out, name="final")

    # resume with a 2nd epoch -> continues from step 4 to 8 (does NOT restart at 0)
    cfg2 = _make_config(layer_dir, out)
    cfg2.save_opt_state = True
    cfg2.num_epochs = 2
    cfg2.resume_from = str(out)
    model2 = train(cfg2, device="cpu")

    assert torch.load(out / "train_state.pt")["num_gradient_steps"] == 8
    # training progressed past the resumed weights (4 more optimizer steps)
    changed = any(
        not torch.allclose(p2.cpu(), p1.cpu())
        for (_, p2), (_, p1) in zip(
            model2.named_parameters(), after_run1.named_parameters(), strict=True
        )
    )
    assert changed


def test_load_resume_restores_weights_and_counters(tmp_path: Path) -> None:
    from glp.train.trainer import load_resume

    layer_dir = tmp_path / "data" / "last" / "layer_08"
    _write_synthetic_dataset(layer_dir)
    out = tmp_path / "run"
    cfg = _make_config(layer_dir, out)
    cfg.save_opt_state = True
    trained = train(cfg, device="cpu")

    glp_kwargs = cast(
        dict[str, Any], OmegaConf.to_container(cfg.glp_kwargs, resolve=True)
    )
    fresh = GLP(**glp_kwargs)
    fresh.to("cpu")
    optimizer = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1)

    n_grad, n_train = load_resume(str(out), fresh, optimizer, scheduler, "cpu")
    assert (n_grad, n_train) == (4, 4)
    # load_resume put the trained weights into the fresh model
    for (_, p_fresh), (_, p_trained) in zip(
        fresh.named_parameters(), trained.named_parameters(), strict=True
    ):
        assert torch.allclose(p_fresh.cpu(), p_trained.cpu())
