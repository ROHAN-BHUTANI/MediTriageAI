import argparse
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModel

from src.config_manager import TrainingConfig
from src.dataset import MediTriageDataset, load_split_rows
from src.model import MediTriageTransformer
from src.profiler import MemoryProfiler
from src.trainer import EmergentTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="DGX DDP Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--mode", choices=["smoke", "development", "publication"], default="development"
    )
    return parser.parse_args()


def setup_process_group():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        return int(os.environ["LOCAL_RANK"])
    else:
        # Fallback to single process
        return 0


def cleanup_process_group():
    if dist.is_initialized():
        dist.destroy_process_group()


def build_data_loaders(config: TrainingConfig, rank: int, world_size: int, mode: str):
    max_rows = 800 if mode == "smoke" else (10000 if mode == "development" else None)

    # Load raw rows
    dataset_path = Path("meditriage/data/canonical/v1.0.0/dataset.parquet")
    if not dataset_path.exists():
        dataset_path = Path("meditriage/data/processed/dataset.parquet")

    train_rows = load_split_rows(dataset_path, "train", max_rows=max_rows)
    val_rows = load_split_rows(dataset_path, "val", max_rows=max_rows)
    test_rows = load_split_rows(dataset_path, "test", max_rows=max_rows)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        getattr(config, "model_name", "xlm-roberta-base")
    )

    # Create Datasets
    train_ds = MediTriageDataset(train_rows, tokenizer, max_length=128)
    val_ds = MediTriageDataset(val_rows, tokenizer, max_length=128)
    test_ds = MediTriageDataset(test_rows, tokenizer, max_length=128)

    # Distributed Samplers
    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )
    val_sampler = (
        DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    test_sampler = (
        DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )

    # Loaders with performance flags
    num_workers = config.dataloader_workers
    prefetch_factor = config.prefetch_factor if num_workers > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if num_workers > 0 else False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if num_workers > 0 else False,
    )

    return train_loader, val_loader, test_loader, tokenizer, train_sampler


def main():
    args = parse_args()

    try:
        rank = setup_process_group()
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        config = TrainingConfig.from_yaml(args.config)

        # Rank-safe deterministic seeding
        global_seed = config.seed
        local_seed = global_seed + rank
        torch.manual_seed(local_seed)
        torch.cuda.manual_seed_all(local_seed)

        if rank == 0:
            print("==================================================")
            print("DGX DDP Training Initialized")
            print(f"World Size: {world_size}")
            print(f"Mode: {args.mode}")
            print("==================================================")

        train_loader, val_loader, test_loader, tokenizer, train_sampler = (
            build_data_loaders(config, rank, world_size, args.mode)
        )

        # Build Model
        model_name = getattr(config, "model_name", "xlm-roberta-base")
        encoder = AutoModel.from_pretrained(model_name)

        # Apply gradient checkpointing if configured
        if getattr(config, "gradient_checkpointing", False):
            encoder.gradient_checkpointing_enable()

        model = MediTriageTransformer(encoder)

        trainer = EmergentTrainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            tokenizer=tokenizer,
        )

        # Attach sampler for epoch step tracking
        trainer.train_sampler = train_sampler

        profiler = MemoryProfiler(rank=rank)
        profiler.start()

        # Inject profiler hook into the training loop if possible, or just time the whole fit()
        trainer.fit()

        metrics = profiler.stop()
        if rank == 0:
            profiler.export(metrics, Path("performance_report.json"))
            print(
                f"Training completed successfully. Throughput: {metrics.samples_per_sec:.2f} samples/sec."
            )

    except Exception:
        print(f"[Rank {os.environ.get('LOCAL_RANK', '0')}] Critical Failure:")
        traceback.print_exc()
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(1)

    finally:
        cleanup_process_group()


if __name__ == "__main__":
    main()
