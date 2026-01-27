#!/usr/bin/env python
"""Train DeepLabCut model from labeled project.

This script wraps DeepLabCut's training functions to train a pose estimation
model from a labeled DLC project.

Usage:
    python scripts/train_dlc.py CONFIG_PATH [--max-iters N] [--display-iters N]
           [--save-iters N] [--gpu GPU_ID] [--no-evaluate]

Example:
    python scripts/train_dlc.py ~/data/vbn_outputs/1055240613/dlc_project/config.yaml
    python scripts/train_dlc.py config.yaml --max-iters 50000
"""

import argparse
import os
from pathlib import Path

from vbn.utils import setup_logging, print_diagnostic


def train_dlc(
    config_path: Path,
    max_iters: int = 50000,
    display_iters: int = 1000,
    save_iters: int = 10000,
    gpu: int | None = 0,
    evaluate: bool = True
) -> None:
    """Train DeepLabCut model using Python API.
    
    Args:
        config_path: Path to DLC project config.yaml
        max_iters: Maximum training iterations
        display_iters: Print loss every N iterations
        save_iters: Save checkpoint every N iterations
        gpu: GPU device ID (None for CPU)
        evaluate: Run evaluation after training
    """
    logger = setup_logging()
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Set GPU
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    try:
        import deeplabcut
    except ImportError:
        raise ImportError(
            "deeplabcut is required for training. "
            "Install with: pip install deeplabcut"
        )
    
    logger.info(f"Training DLC model from: {config_path}")
    
    # Create training dataset if not exists
    logger.info("Creating training dataset...")
    try:
        deeplabcut.create_training_dataset(str(config_path))
    except Exception as e:
        logger.warning(f"create_training_dataset warning (may be OK): {e}")
    
    # Train network
    logger.info(f"Training for {max_iters} iterations...")
    deeplabcut.train_network(
        str(config_path),
        maxiters=max_iters,
        displayiters=display_iters,
        saveiters=save_iters,
    )
    
    logger.info("Training complete!")
    
    # Evaluate
    if evaluate:
        logger.info("Evaluating model...")
        deeplabcut.evaluate_network(str(config_path), plotting=True)
        logger.info("Evaluation complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Train DeepLabCut model from labeled project"
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to DLC config.yaml"
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=50000,
        help="Maximum training iterations (default: 50000)"
    )
    parser.add_argument(
        "--display-iters",
        type=int,
        default=1000,
        help="Print loss every N iterations (default: 1000)"
    )
    parser.add_argument(
        "--save-iters",
        type=int,
        default=10000,
        help="Save checkpoint every N iterations (default: 10000)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID (default: 0, use -1 for CPU)"
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Skip evaluation after training"
    )
    args = parser.parse_args()
    
    logger = setup_logging()
    
    gpu = args.gpu if args.gpu >= 0 else None
    
    print_diagnostic("DLC Training Configuration", {
        "Config file": args.config_path,
        "Max iterations": args.max_iters,
        "Display iterations": args.display_iters,
        "Save iterations": args.save_iters,
        "GPU": gpu if gpu is not None else "CPU",
        "Evaluate": not args.no_evaluate,
    })
    
    try:
        train_dlc(
            args.config_path,
            args.max_iters,
            args.display_iters,
            args.save_iters,
            gpu,
            evaluate=not args.no_evaluate
        )
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
