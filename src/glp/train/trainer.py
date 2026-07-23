"""GLP training loop.

Ported from ``generative_latent_prior/glp_train.py`` (the ``main`` loop,
``TrainConfig`` and ``save_checkpoint``) and adapted to guard-glp: it reuses this
repo's activation consumer (:mod:`glp.dataset.act_dataset`) and model
(:class:`glp.denoiser.GLP`) instead of the reference's in-file copies, resolves
the LR schedule by name (no ``eval``), and makes ``wandb`` optional.

The model, dataset and architecture are entirely config-driven (see
``configs/train/``): ``glp_kwargs.denoiser_config`` sets the architecture,
``train_dataset``/``rep_statistic`` select the data. The flow-matching MSE loss is
computed inside :meth:`glp.denoiser.GLP.forward`.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Subset
from tqdm import tqdm

from glp.dataset.act_dataset import (
    get_activation_dataloader,
    load_activation_dataset,
)
from glp.denoiser import GLP, Normalizer
from glp.train.schedulers import get_scheduler_fn

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    # model
    model_name: str = ""
    glp_kwargs: Any | None = None
    # data
    shuffle: bool = True
    train_dataset: Any = ""  # str | list[str] of built activation directories
    rep_statistic: str = ""
    # validation: hold out the last `val_fraction` of the dataset (a contiguous tail;
    # shards are corpus-strided and training is shuffled, so it is representative and
    # never trained on). val loss is logged every `val_every_n_steps`, evaluated on up
    # to `val_max_batches` batches with a fixed RNG seed so it is comparable across steps.
    val_fraction: float = 0.0
    val_every_n_steps: int | None = None
    val_max_batches: int | None = None
    seed: int = 0
    # dataloader throughput (critical on a network FS): parallel prefetched reads +
    # chunk-shuffling. Defaults reproduce the original single-worker per-sample loader.
    num_workers: int = 0
    prefetch_factor: int | None = None
    pin_memory: bool = False
    persistent_workers: bool = False
    shuffle_chunk_size: int = 0
    # training
    use_bf16: bool = True
    num_epochs: int = 1
    epoch_size: int | None = None
    batch_size: int = 4096
    learning_rate: float = 5e-5
    lr_scheduler: dict[str, Any] | None = None
    gradient_accumulation_steps: int = 1
    gradient_clipping_threshold: float = 1.0
    # logging and saving
    log_every_n_steps: int = 10
    save_every_n_steps: int | None = None
    save_epochs: list[int] | None = None
    save_opt_state: bool = False
    output_path: str | None = None
    # wandb
    wandb_enabled: bool = False
    wandb_entity: str | None = None
    wandb_project: str | None = None
    wandb_run_name: str | None = None


def save_checkpoint(
    model: GLP,
    output_path: Path,
    checkpoint_name: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    save_opt_state: bool = False,
) -> None:
    """Save GLP weights (+ normalizer stats) and, optionally, optimizer/scheduler state."""
    model.save_pretrained(path=output_path, name=checkpoint_name)
    logger.info("Model saved to %s/%s", output_path, checkpoint_name)
    if save_opt_state:
        if optimizer is not None:
            torch.save(optimizer.state_dict(), output_path / "optimizer_state.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), output_path / "scheduler_state.pt")


def _evaluate(
    model: GLP,
    val_loader: Any,
    device: str,
    use_bf16: bool,
    seed: int,
    max_batches: int | None,
) -> float:
    """Mean flow-matching loss over the validation loader.

    A single generator is reseeded to ``seed`` at the start of every call and passed
    to ``GLP.forward`` (which draws both the noise and the flow-matching timesteps from
    it), so each evaluation sees identical (noise, t) draws and the only thing moving
    the val curve is the model weights.
    """
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    total = 0.0
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break
            # the collator only ever emits tensors (latents / layer_idx)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16
            ):
                outputs = model(**batch, generator=gen)
            bsz = batch["latents"].shape[0]
            total += float(outputs.loss.detach()) * bsz
            n += bsz
    model.train()
    return total / max(n, 1)


def train(config: DictConfig, device: str = "cuda:0") -> GLP:
    """Train a GLP from a resolved config. Returns the trained model."""
    # Fill any omitted optional keys from the schema defaults so the loop can rely
    # on them (a direct caller may pass a partial config; the CLI entry point already
    # merges the structured base, in which case this is a harmless no-op).
    base = OmegaConf.structured(TrainConfig())
    OmegaConf.set_struct(base, False)
    config = cast(DictConfig, OmegaConf.merge(base, config))

    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Saving checkpoints to %s", output_path)
    OmegaConf.save(config, output_path / "config.yaml")

    # These datasets are pre-built (static), so the normalization stats must exist.
    rep_statistic = (config.glp_kwargs.get("normalizer_config", {}) or {}).get(
        "rep_statistic"
    )
    if rep_statistic and not Path(rep_statistic).exists():
        raise FileNotFoundError(
            f"rep_statistic not found: {rep_statistic} — run the dataset `finalize` "
            "pass first (it writes rep_statistics.pt)."
        )

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
    logger.info("Config: %s", config)

    wandb_run = None
    if config.wandb_enabled:
        # Optional dependency: imported dynamically so wandb is only required when
        # logging is enabled (and is not a static import the type checker resolves).
        wandb = importlib.import_module("wandb")

        wandb_run = wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=OmegaConf.to_container(config, resolve=True),
        )

    # model (architecture entirely from config.glp_kwargs)
    model = GLP(**config.glp_kwargs)
    model.to(device)
    logger.info("Model param count: %d", sum(p.numel() for p in model.parameters()))

    # data (reuses the in-repo memmap consumer + normalizing collator)
    full_dataset = load_activation_dataset(config.train_dataset)
    per_device_batch = config.batch_size // config.gradient_accumulation_steps
    # The collator runs inside forked DataLoader workers, so it must NOT touch CUDA
    # (model.normalizer lives on the GPU). Normalize with a CPU copy of the stats
    # (identical values, cheap); the collated CPU batch is moved to the GPU in the loop.
    loader_normalizer = Normalizer(
        model.normalizer.mean.detach().cpu().clone(),
        model.normalizer.var.detach().cpu().clone(),
    )
    val_loader = None
    if config.val_fraction and config.val_fraction > 0.0:
        n_total = len(full_dataset)
        n_val = max(1, int(n_total * config.val_fraction))
        # contiguous tail hold-out (range -> O(1) memory even for ~1B samples)
        train_ds: Any = Subset(full_dataset, range(0, n_total - n_val))
        val_ds = Subset(full_dataset, range(n_total - n_val, n_total))
        logger.info("train/val split: %d train, %d val", len(train_ds), len(val_ds))
        val_loader = get_activation_dataloader(
            dataset=val_ds,
            batch_size=per_device_batch,
            normalizer=loader_normalizer,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            persistent_workers=config.persistent_workers,
        )
    else:
        train_ds = full_dataset
    train_dataloader = get_activation_dataloader(
        dataset=train_ds,
        batch_size=per_device_batch,
        normalizer=loader_normalizer,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
        chunk_size=config.shuffle_chunk_size,
        seed=config.seed,
    )

    epoch_size = (
        (config.epoch_size // config.batch_size)
        if config.epoch_size
        else len(train_dataloader)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    if config.lr_scheduler is None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: 1
        )
    else:
        total_num_steps = config.num_epochs * (
            epoch_size // config.gradient_accumulation_steps
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=partial(
                get_scheduler_fn(config.lr_scheduler["scheduler_cls"]),
                warmup_steps=config.lr_scheduler["warmup_ratio"] * total_num_steps,
                max_steps=total_num_steps,
                initial_factor=config.lr_scheduler["initial_factor"],
                final_factor=config.lr_scheduler["final_factor"],
            ),
        )

    train_steps = 0
    num_gradient_steps = 0

    for epoch in range(config.num_epochs):
        model.train()
        gradient_steps_in_epoch = epoch_size // config.gradient_accumulation_steps
        pbar = tqdm(
            total=gradient_steps_in_epoch,
            desc=f"Training Epoch: {epoch + 1}",
            dynamic_ncols=True,
        )
        for step, batch in enumerate(train_dataloader):
            batch = {
                k: (v.to(device) if v is not None else None) for k, v in batch.items()
            }

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=config.use_bf16
            ):
                outputs = model(**batch)
                loss = outputs.loss

            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            train_steps += 1

            if train_steps % config.gradient_accumulation_steps == 0:
                num_gradient_steps += 1

                if config.gradient_clipping_threshold > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clipping_threshold
                    )

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                pbar.update(1)
                pbar.set_description(
                    f"Epoch: {epoch + 1}/{config.num_epochs}, "
                    f"batch {step + 1}/{epoch_size} "
                    f"(loss: {loss.detach().float():.4f})"
                )

                if num_gradient_steps % config.log_every_n_steps == 0:
                    avg_loss = loss.detach().item()
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                # fractional epochs completed (0.0 -> num_epochs), not
                                # the integer loop index (which is flat at 1 epoch)
                                "train/epoch": num_gradient_steps
                                / max(gradient_steps_in_epoch, 1),
                                "train/step": num_gradient_steps,
                                "train/loss": avg_loss,
                                "train/learning_rate": scheduler.get_last_lr()[0],
                            },
                            step=num_gradient_steps,
                        )

                if (
                    val_loader is not None
                    and config.val_every_n_steps
                    and num_gradient_steps % config.val_every_n_steps == 0
                ):
                    val_loss = _evaluate(
                        model,
                        val_loader,
                        device,
                        config.use_bf16,
                        config.seed,
                        config.val_max_batches,
                    )
                    logger.info("step %d: val/loss %.4f", num_gradient_steps, val_loss)
                    if wandb_run is not None:
                        wandb_run.log(
                            {"val/loss": val_loss, "train/step": num_gradient_steps},
                            step=num_gradient_steps,
                        )

                if (
                    config.save_every_n_steps
                    and num_gradient_steps % config.save_every_n_steps == 0
                ):
                    save_checkpoint(
                        model,
                        output_path,
                        f"step_{num_gradient_steps}",
                        optimizer,
                        scheduler,
                        save_opt_state=config.save_opt_state,
                    )

            if step >= gradient_steps_in_epoch * config.gradient_accumulation_steps:
                break

        pbar.close()

        if config.save_epochs and (epoch + 1) in set(config.save_epochs):
            save_checkpoint(model, output_path / "checkpoints", f"epoch_{epoch + 1}")

        # always save the latest checkpoint
        save_checkpoint(
            model,
            output_path,
            "final",
            optimizer,
            scheduler,
            save_opt_state=config.save_opt_state,
        )

    if val_loader is not None:
        final_val = _evaluate(
            model,
            val_loader,
            device,
            config.use_bf16,
            config.seed,
            config.val_max_batches,
        )
        logger.info("final val/loss %.4f (step %d)", final_val, num_gradient_steps)
        if wandb_run is not None:
            wandb_run.log(
                {"val/loss": final_val, "train/step": num_gradient_steps},
                step=num_gradient_steps,
            )

    if wandb_run is not None:
        wandb_run.finish()

    return model
