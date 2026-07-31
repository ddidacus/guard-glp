"""Streaming GLP training launcher: extraction producers + DDP trainer ranks.

One parent process owns everything (a single SLURM task with all the node's
GPUs): it spawns ``streaming.num_producers`` producer processes (one GPU each,
running the extraction backend) and one DDP trainer rank per remaining GPU,
wired by one bounded ``torch.multiprocessing`` queue per rank (producers
round-robin pooled activation chunks across them). The parent is the watchdog:
if any child dies, everything is torn down.

    # full node (e.g. 2 producer GPUs + 6 DDP ranks on 8xH100):
    python scripts/train/train_glp_stream.py \
        config=configs/train/glp_llama1b_guardglpbenign_stream_all16.yaml

    # 1-GPU / CPU smoke path (in-process producer thread, no DDP):
    python scripts/train/train_glp_stream.py config=<CFG> streaming.num_producers=0

Config: a normal training YAML with ``data_mode: streaming`` and a
``streaming:`` section (see ``glp.dataset.streaming.StreamingConfig``). GPU
partitioning: producers take the first ``num_producers`` visible GPUs, trainer
ranks the rest, each child pinned via ``CUDA_VISIBLE_DEVICES`` before start.
"""

import logging
import os
import socket
import sys
import time
from typing import Any, cast

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _visible_gpus() -> list[str]:
    """Physical GPU ids available to this job, honoring CUDA_VISIBLE_DEVICES."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [g for g in env.split(",") if g != ""]
    import torch

    return [str(i) for i in range(torch.cuda.device_count())]


def _pin_child_env(gpu: str | None) -> None:
    """Pin one GPU (or none) for this child; must run before any CUDA init."""
    if gpu is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # vLLM's workers must be spawned (fork breaks CUDA in the parent producer)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def _producer_entry(
    payload: dict[str, Any],
    producer_id: int,
    gpu: str | None,
    rank_queues: list[Any],
    stop_event: Any,
) -> None:
    _pin_child_env(gpu)
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    import torch.multiprocessing as mp

    mp.set_sharing_strategy("file_system")
    from glp.dataset.streaming import StreamingConfig, producer_loop

    scfg = StreamingConfig.from_dict(payload["streaming"], payload["model_name"])
    device = "cuda:0" if gpu is not None else "cpu"
    producer_loop(
        scfg,
        payload["model_name"],
        producer_id,
        rank_queues,
        stop_event,
        device=device,
        seed=int(payload.get("seed", 0)),
    )


def _trainer_entry(
    payload: dict[str, Any],
    rank: int,
    world_size: int,
    gpu: str | None,
    master_port: int,
    chunk_queue: Any,
    stop_event: Any,
) -> None:
    _pin_child_env(gpu)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = "0"  # each rank sees exactly one (masked) GPU
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    import torch.multiprocessing as mp

    mp.set_sharing_strategy("file_system")
    from glp.train import train

    config = OmegaConf.create(payload)
    device = "cuda:0" if gpu is not None else "cpu"
    train(config, device=device, chunk_queue=chunk_queue, producer_stop=stop_event)


def main() -> None:
    load_dotenv()
    from glp.dataset.streaming import StreamingConfig
    from glp.train import TrainConfig

    config_base = OmegaConf.structured(TrainConfig())
    OmegaConf.set_struct(config_base, False)
    config_cli = OmegaConf.from_cli()
    config_path = config_cli.pop("config", None)
    config_file = OmegaConf.load(config_path) if config_path else OmegaConf.create()
    config = cast(DictConfig, OmegaConf.merge(config_base, config_file, config_cli))
    if config.data_mode != "streaming":
        raise ValueError(
            "train_glp_stream.py requires data_mode: streaming — use "
            "scripts/train/train_glp.py for static datasets"
        )
    payload = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
    scfg = StreamingConfig.from_dict(payload["streaming"], payload["model_name"])

    # Degenerate mode: producer thread inside the (single) trainer process.
    if scfg.num_producers == 0:
        import torch

        from glp.train import train

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info("num_producers=0: in-process producer thread on %s", device)
        train(config, device=device)
        return

    gpus = _visible_gpus()
    if len(gpus) < scfg.num_producers + 1:
        raise RuntimeError(
            f"need at least num_producers+1={scfg.num_producers + 1} GPUs "
            f"(got {len(gpus)}); use streaming.num_producers=0 for the 1-GPU path"
        )
    producer_gpus = gpus[: scfg.num_producers]
    trainer_gpus = gpus[scfg.num_producers :]
    world_size = len(trainer_gpus)
    logger.info(
        "GPU split: producers=%s trainers=%s (world_size=%d)",
        producer_gpus,
        trainer_gpus,
        world_size,
    )

    import torch.multiprocessing as torch_mp

    torch_mp.set_sharing_strategy("file_system")
    ctx = torch_mp.get_context("spawn")
    stop_event = ctx.Event()
    queues = [ctx.Queue(maxsize=scfg.queue_maxsize) for _ in range(world_size)]
    master_port = _free_port()

    # Non-daemon producers: vLLM spawns its own worker subprocesses, and daemonic
    # processes cannot have children.
    producers = [
        ctx.Process(
            target=_producer_entry,
            args=(payload, i, gpu, queues, stop_event),
            name=f"glp-producer-{i}",
        )
        for i, gpu in enumerate(producer_gpus)
    ]
    trainers = [
        ctx.Process(
            target=_trainer_entry,
            args=(
                payload,
                rank,
                world_size,
                gpu,
                master_port,
                queues[rank],
                stop_event,
            ),
            name=f"glp-trainer-{rank}",
        )
        for rank, gpu in enumerate(trainer_gpus)
    ]
    for proc in producers + trainers:
        proc.start()

    # Watchdog: tear everything down if any child fails; finish when trainers do.
    exit_code = 0
    try:
        while True:
            failed = [
                p
                for p in producers + trainers
                if p.exitcode is not None and p.exitcode != 0
            ]
            if failed:
                logger.error(
                    "child failed: %s (exitcode %s); terminating all",
                    failed[0].name,
                    failed[0].exitcode,
                )
                exit_code = 1
                break
            if all(t.exitcode == 0 for t in trainers):
                logger.info("all trainer ranks finished")
                break
            # a producer exiting 0 while trainers still run is only legal after
            # stop_event (or in single-pass mode); otherwise trainers will hit
            # their stall timeout and fail, which the loop above catches.
            time.sleep(5)
    finally:
        stop_event.set()
        deadline = time.time() + 60
        for proc in producers + trainers:
            proc.join(timeout=max(0.0, deadline - time.time()))
        for proc in producers + trainers:
            if proc.is_alive():
                logger.warning("terminating %s", proc.name)
                proc.terminate()
                proc.join(timeout=10)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
