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
import os
import queue
import threading
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from glp.dataset.act_dataset import (
    ActivationCollator,
    get_activation_dataloader,
    load_activation_dataset,
)
from glp.dataset.streaming import (
    QueueChunkSource,
    StreamingActDataset,
    StreamingConfig,
    start_thread_producer,
)
from glp.denoiser import GLP, Normalizer
from glp.train.schedulers import get_scheduler_fn

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    # model
    model_name: str = ""
    glp_kwargs: Any | None = None
    # data source: "static" (pre-built memmap dirs, the default/original path) or
    # "streaming" (on-the-fly extraction; requires a `streaming:` section and an
    # explicit `epoch_size`, and its rep_statistic comes from the stats pre-pass).
    data_mode: str = "static"
    streaming: Any | None = None  # StreamingConfig schema (see glp.dataset.streaming)
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
    # resume: path to a run dir holding train_state.pt + optimizer_state.pt +
    # scheduler_state.pt + <checkpoint>.safetensors (written by save_checkpoint with
    # save_opt_state=True). Restores weights, optimizer/scheduler state, and the step
    # counter so a run continues from where it stopped (a job that hit the wall clock
    # or crashed resumes instead of restarting). Requires the same config/architecture.
    resume_from: str | None = None
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
    num_gradient_steps: int | None = None,
    train_steps: int | None = None,
) -> None:
    """Save GLP weights (+ normalizer stats) and, optionally, optimizer/scheduler state.

    When ``save_opt_state`` and step counters are given, also writes
    ``train_state.pt`` (the step counters + which checkpoint the optimizer state
    corresponds to) so the run is resumable via ``TrainConfig.resume_from``.
    """
    model.save_pretrained(path=output_path, name=checkpoint_name)
    logger.info("Model saved to %s/%s", output_path, checkpoint_name)
    if save_opt_state:
        if optimizer is not None:
            torch.save(optimizer.state_dict(), output_path / "optimizer_state.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), output_path / "scheduler_state.pt")
        if num_gradient_steps is not None:
            torch.save(
                {
                    "num_gradient_steps": num_gradient_steps,
                    "train_steps": train_steps,
                    "checkpoint_name": checkpoint_name,
                },
                output_path / "train_state.pt",
            )


def load_resume(
    resume_from: str,
    model: GLP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: str,
) -> tuple[int, int]:
    """Restore weights + optimizer/scheduler state + step counters from a run dir.

    Returns ``(num_gradient_steps, train_steps)`` to continue from. All ranks call
    this and load the identical files from the shared FS, so DDP stays in sync.

    A run dir without ``train_state.pt`` is not an error: it is the first launch of
    a config that presets ``resume_from`` (so a requeued job self-resumes), and the
    run simply starts from scratch.
    """
    path = Path(resume_from)
    if not (path / "train_state.pt").is_file():
        logger.info("no checkpoint at %s; starting from scratch", path)
        return 0, 0
    state = torch.load(path / "train_state.pt", map_location="cpu")
    checkpoint_name = state["checkpoint_name"]
    model.denoiser.load_pretrained(path, name=checkpoint_name)
    model.to(device)
    optimizer.load_state_dict(
        torch.load(path / "optimizer_state.pt", map_location=device)
    )
    scheduler.load_state_dict(
        torch.load(path / "scheduler_state.pt", map_location="cpu")
    )
    num_gradient_steps = int(state["num_gradient_steps"])
    train_steps = int(state.get("train_steps") or num_gradient_steps)
    logger.info(
        "resumed from %s at gradient step %d (%s)",
        path,
        num_gradient_steps,
        checkpoint_name,
    )
    return num_gradient_steps, train_steps


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


@dataclass
class _DistContext:
    """Distributed state; the no-op default keeps the single-process path intact."""

    rank: int = 0
    world_size: int = 1
    is_main: bool = True
    enabled: bool = False


def _maybe_init_distributed(device: str) -> _DistContext:
    """Join the process group when launched distributed (``WORLD_SIZE`` env set).

    Rendezvous is env-driven (``MASTER_ADDR``/``MASTER_PORT``/``RANK`` set by
    ``scripts/train/train_glp_stream.py``); without ``WORLD_SIZE`` this is a
    no-op and training runs exactly as before. A ``WORLD_SIZE`` of 1 still
    initializes the (single-member) group so the launcher path is uniform.
    """
    world_size_env = os.environ.get("WORLD_SIZE")
    if world_size_env is None:
        return _DistContext()
    world_size = int(world_size_env)
    rank = int(os.environ["RANK"])
    backend = (
        "nccl" if device.startswith("cuda") and torch.cuda.is_available() else "gloo"
    )
    # Generous collective timeout (default is 10 min): a rank can legitimately be
    # out of the all-reduce for a while — rank 0 writing a multi-GB checkpoint,
    # validation, or a brief producer hiccup while the shuffle buffer drains — and
    # must not trip the NCCL watchdog and abort the whole job. A genuinely dead
    # child is still torn down promptly by the launcher's exit-code watchdog.
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=30),
    )
    logger.info("distributed init: rank %d/%d (%s)", rank, world_size, backend)
    return _DistContext(
        rank=rank, world_size=world_size, is_main=rank == 0, enabled=True
    )


def _check_streaming_stats(model: GLP, layers: list[int]) -> None:
    """Fail fast if the stacked stats do not cover every streamed layer.

    The stats pre-pass writes NaN rows for layers it never saw, so a mismatch
    between the stats config and the training config would otherwise silently
    poison the loss from step one.
    """
    mean, var = model.normalizer.mean, model.normalizer.var
    if mean.ndim < 2 or mean.shape[0] == 1:
        if len(layers) > 1:
            raise ValueError(
                f"streaming {len(layers)} layers requires stacked (n_layers, D) "
                f"rep_statistics (got shape {tuple(mean.shape)}) — run the stats "
                "pre-pass (scripts/dataset/compute_stats.py)"
            )
        return
    if max(layers) >= mean.shape[0]:
        raise ValueError(
            f"rep_statistic covers {mean.shape[0]} layers but layer "
            f"{max(layers)} is being streamed"
        )
    for layer in layers:
        if not (torch.isfinite(mean[layer]).all() and torch.isfinite(var[layer]).all()):
            raise ValueError(
                f"rep_statistic has non-finite stats for layer {layer} — the "
                "stats pre-pass did not cover it (its NaN rows fail loudly here)"
            )


def _build_streaming_data(
    config: DictConfig,
    loader_normalizer: Normalizer,
    per_device_batch: int,
    ctx: _DistContext,
    device: str,
    chunk_queue: Any | None,
) -> tuple[
    DataLoader[Any], DataLoader[Any] | None, StreamingConfig, threading.Event | None
]:
    """Build the streaming train/val loaders from the ``streaming:`` config.

    Under the streaming launcher each rank receives its own producer-fed
    ``chunk_queue``; without one (``num_producers: 0``) an in-process producer
    thread is started on the trainer's device — the 1-GPU/CPU smoke path.
    Returns ``(train_loader, val_loader, streaming_cfg, local_stop_event)``.
    """
    if config.streaming is None:
        raise ValueError("data_mode: streaming requires a `streaming:` config section")
    scfg = StreamingConfig.from_dict(
        cast(dict[str, Any], OmegaConf.to_container(config.streaming, resolve=True)),
        config.model_name,
    )
    if not config.epoch_size:
        raise ValueError(
            "data_mode: streaming requires an explicit `epoch_size` (samples per "
            "rank) — a stream has no length for the LR schedule to derive it from"
        )
    if ctx.world_size > 1 and not scfg.cycle:
        raise ValueError(
            "streaming.cycle=false is single-rank only: with DDP a rank whose "
            "stream ends early would hang the others in NCCL"
        )

    local_stop: threading.Event | None = None
    if chunk_queue is None:
        if scfg.num_producers > 0:
            raise ValueError(
                "streaming.num_producers > 0 requires the streaming launcher "
                "(scripts/train/train_glp_stream.py), which owns the producer "
                "processes; set num_producers: 0 for an in-process thread"
            )
        chunk_queue = queue.Queue(maxsize=scfg.queue_maxsize)
        local_stop = threading.Event()
        start_thread_producer(
            scfg,
            config.model_name,
            chunk_queue,
            local_stop,
            device=device,
            seed=config.seed,
        )

    dataset = StreamingActDataset(
        source=QueueChunkSource(chunk_queue),
        num_producers=max(scfg.num_producers, 1),
        buffer_size=scfg.shuffle_buffer_size,
        min_fill=scfg.shuffle_min_fill,
        seed=config.seed + ctx.rank,
        stall_timeout_s=scfg.stall_timeout_s,
        startup_timeout_s=scfg.startup_timeout_s,
    )
    val_loader: DataLoader[Any] | None = None
    if scfg.val_num_prompts > 0:
        # a list of sample dicts satisfies the map-style dataset protocol
        val_samples = cast("Dataset[Any]", dataset.wait_for_val())
        val_loader = DataLoader(
            val_samples,
            batch_size=per_device_batch,
            shuffle=False,
            collate_fn=ActivationCollator(loader_normalizer),
        )
    train_loader = DataLoader(
        dataset,
        batch_size=per_device_batch,
        drop_last=True,
        collate_fn=ActivationCollator(loader_normalizer),
        num_workers=0,  # the chunk queue must be consumed in-process
    )
    return train_loader, val_loader, scfg, local_stop


def train(
    config: DictConfig,
    device: str = "cuda:0",
    *,
    chunk_queue: Any | None = None,
    producer_stop: Any | None = None,
) -> GLP:
    """Train a GLP from a resolved config. Returns the trained model.

    ``chunk_queue``/``producer_stop`` are provided by the streaming launcher
    (``scripts/train/train_glp_stream.py``): the rank's producer-fed queue and
    the shared event that tells producers to exit once training completes. Both
    are None for static training and for the in-process streaming smoke path.
    """
    # Fill any omitted optional keys from the schema defaults so the loop can rely
    # on them (a direct caller may pass a partial config; the CLI entry point already
    # merges the structured base, in which case this is a harmless no-op).
    base = OmegaConf.structured(TrainConfig())
    OmegaConf.set_struct(base, False)
    config = cast(DictConfig, OmegaConf.merge(base, config))

    # validate BEFORE joining the process group (init blocks on rendezvous)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and config.data_mode != "streaming":
        raise ValueError(
            "distributed training is only supported with data_mode: streaming "
            "(the static path stays single-GPU)"
        )
    ctx = _maybe_init_distributed(device)
    # per-rank seed: noise/timestep draws differ across ranks, runs stay reproducible
    torch.manual_seed(config.seed + ctx.rank)

    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if ctx.is_main:
        logger.info("Saving checkpoints to %s", output_path)
        OmegaConf.save(config, output_path / "config.yaml")

    # Both data modes need pre-computed normalization stats: static datasets get
    # them from `finalize`, streaming runs from the stats pre-pass.
    rep_statistic = (config.glp_kwargs.get("normalizer_config", {}) or {}).get(
        "rep_statistic"
    )
    if rep_statistic and not Path(rep_statistic).exists():
        hint = (
            "run the stats pre-pass first (scripts/dataset/compute_stats.py)"
            if config.data_mode == "streaming"
            else "run the dataset `finalize` pass first (it writes rep_statistics.pt)"
        )
        raise FileNotFoundError(f"rep_statistic not found: {rep_statistic} — {hint}.")

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
    logger.info("Config: %s", config)

    wandb_run = None
    if config.wandb_enabled and ctx.is_main:
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
    if ctx.is_main:
        logger.info("Model param count: %d", sum(p.numel() for p in model.parameters()))
    # DDP wrap for the forward pass only; `model` stays the source of truth for
    # checkpointing/eval. Normalizer buffers are identical constants everywhere,
    # so per-step buffer broadcast is skipped.
    forward_model: Any = model
    if ctx.enabled:
        device_ids = (
            [torch.cuda.current_device()]
            if device.startswith("cuda") and torch.cuda.is_available()
            else None
        )
        forward_model = DistributedDataParallel(
            model, device_ids=device_ids, broadcast_buffers=False
        )

    per_device_batch = config.batch_size // config.gradient_accumulation_steps
    # The collator runs inside forked DataLoader workers, so it must NOT touch CUDA
    # (model.normalizer lives on the GPU). Normalize with a CPU copy of the stats
    # (identical values, cheap); the collated CPU batch is moved to the GPU in the loop.
    loader_normalizer = Normalizer(
        model.normalizer.mean.detach().cpu().clone(),
        model.normalizer.var.detach().cpu().clone(),
    )

    local_stop: threading.Event | None = None
    if config.data_mode == "streaming":
        train_dataloader, val_loader, streaming_cfg, local_stop = _build_streaming_data(
            config, loader_normalizer, per_device_batch, ctx, device, chunk_queue
        )
        _check_streaming_stats(model, list(streaming_cfg.extract.layers))
    elif config.data_mode == "static":
        # data (reuses the in-repo memmap consumer + normalizing collator)
        full_dataset = load_activation_dataset(config.train_dataset)
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
    else:
        raise ValueError(f"unknown data_mode: {config.data_mode!r}")

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

    gradient_steps_in_epoch = epoch_size // config.gradient_accumulation_steps
    if config.resume_from:
        num_gradient_steps, train_steps = load_resume(
            config.resume_from, model, optimizer, scheduler, device
        )
    else:
        train_steps = 0
        num_gradient_steps = 0
    # resume mid-run: skip epochs already completed (streaming is single-epoch, so
    # this is 0; it keeps multi-epoch static resumes correct too).
    start_epoch = num_gradient_steps // max(gradient_steps_in_epoch, 1)

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        pbar = tqdm(
            total=gradient_steps_in_epoch,
            initial=num_gradient_steps - epoch * gradient_steps_in_epoch,
            desc=f"Training Epoch: {epoch + 1}",
            dynamic_ncols=True,
            disable=not ctx.is_main,
        )
        for step, batch in enumerate(train_dataloader):
            batch = {
                k: (v.to(device) if v is not None else None) for k, v in batch.items()
            }

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=config.use_bf16
            ):
                outputs = forward_model(**batch)
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
                    loss_t = loss.detach()
                    if ctx.enabled:
                        # average across ranks so the logged loss is the global one
                        # (SUM/world_size: gloo lacks ReduceOp.AVG)
                        loss_t = loss_t.clone()
                        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
                        loss_t = loss_t / ctx.world_size
                    avg_loss = loss_t.item()
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
                    # all ranks evaluate the same fixed set (keeps them in lockstep;
                    # no collectives inside); only rank 0 logs.
                    val_loss = _evaluate(
                        model,
                        val_loader,
                        device,
                        config.use_bf16,
                        config.seed,
                        config.val_max_batches,
                    )
                    if ctx.is_main:
                        logger.info(
                            "step %d: val/loss %.4f", num_gradient_steps, val_loss
                        )
                    if wandb_run is not None:
                        wandb_run.log(
                            {"val/loss": val_loss, "train/step": num_gradient_steps},
                            step=num_gradient_steps,
                        )

                if (
                    ctx.is_main
                    and config.save_every_n_steps
                    and num_gradient_steps % config.save_every_n_steps == 0
                ):
                    save_checkpoint(
                        model,
                        output_path,
                        f"step_{num_gradient_steps}",
                        optimizer,
                        scheduler,
                        save_opt_state=config.save_opt_state,
                        num_gradient_steps=num_gradient_steps,
                        train_steps=train_steps,
                    )

            # stop at the epoch's absolute gradient-step target (resume-aware: the
            # counter may start > 0, so this is not a per-process batch count). For
            # streaming this also ends the otherwise-infinite (cycling) dataloader.
            if num_gradient_steps >= (epoch + 1) * gradient_steps_in_epoch:
                break

        pbar.close()

        if (
            ctx.is_main
            and config.save_epochs
            and (epoch + 1) in set(config.save_epochs)
        ):
            save_checkpoint(model, output_path / "checkpoints", f"epoch_{epoch + 1}")

        # always save the latest checkpoint
        if ctx.is_main:
            save_checkpoint(
                model,
                output_path,
                "final",
                optimizer,
                scheduler,
                save_opt_state=config.save_opt_state,
                num_gradient_steps=num_gradient_steps,
                train_steps=train_steps,
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
        if ctx.is_main:
            logger.info("final val/loss %.4f (step %d)", final_val, num_gradient_steps)
        if wandb_run is not None:
            wandb_run.log(
                {"val/loss": final_val, "train/step": num_gradient_steps},
                step=num_gradient_steps,
            )

    if wandb_run is not None:
        wandb_run.finish()

    # Wait for every rank to finish BEFORE stopping producers: a rank still a few
    # batches behind must not find drained queues and dead producers.
    if ctx.enabled:
        dist.barrier()
    if producer_stop is not None:
        producer_stop.set()
    if local_stop is not None:
        local_stop.set()
    if ctx.enabled:
        dist.destroy_process_group()

    return model
