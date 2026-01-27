#!/usr/bin/env python
"""Train SLEAP model from labeled project file.

This script wraps the sleap-train CLI to train a pose estimation model
from a labeled SLEAP project.

Usage:
    python scripts/train_sleap.py PROJECT_PATH [--config CONFIG_FILE]
           [--epochs N] [--batch-size N] [--gpu GPU_ID] [--output-dir PATH]

Example:
    python scripts/train_sleap.py ~/data/vbn_outputs/1055240613/labeling/vbn_body.slp
    python scripts/train_sleap.py project.slp --epochs 100 --batch-size 4
"""

import argparse
import subprocess
from pathlib import Path
from datetime import datetime

from vbn.config import get_outputs_dir
from vbn.utils import setup_logging, ensure_dir, print_diagnostic


def train_sleap(
    project_path: Path,
    output_dir: Path | None = None,
    config_file: Path | None = None,
    epochs: int = 100,
    batch_size: int = 4,
    gpu: int | None = 0
) -> Path:
    """Train SLEAP model using sleap-train CLI.
    
    Args:
        project_path: Path to .slp project with labeled frames
        output_dir: Where to save trained model (default: models/sleap/{timestamp})
        config_file: Optional training config JSON
        epochs: Number of training epochs
        batch_size: Batch size for training
        gpu: GPU device ID (None for CPU)
        
    Returns:
        Path to trained model directory
    """
    logger = setup_logging()
    
    if not project_path.exists():
        raise FileNotFoundError(f"Project file not found: {project_path}")
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = get_outputs_dir() / "models" / "sleap" / timestamp
    
    ensure_dir(output_dir)
    
    # Build command
    cmd = [
        "sleap-train",
        str(project_path),
        "--run_name", output_dir.name,
        "--save_dir", str(output_dir.parent),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
    ]
    
    if config_file:
        cmd.extend(["--config", str(config_file)])
    
    if gpu is not None:
        cmd.extend(["--gpu", str(gpu)])
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        logger.info(f"Training complete. Model saved to: {output_dir}")
        return output_dir
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with exit code {e.returncode}")
        raise
    except FileNotFoundError:
        raise RuntimeError(
            "sleap-train command not found. "
            "Install SLEAP with: pip install sleap"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Train SLEAP model from labeled project file"
    )
    parser.add_argument(
        "project_path",
        type=Path,
        help="Path to .slp project file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for model"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Training config JSON file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID (default: 0, use -1 for CPU)"
    )
    args = parser.parse_args()
    
    logger = setup_logging()
    
    gpu = args.gpu if args.gpu >= 0 else None
    
    print_diagnostic("SLEAP Training Configuration", {
        "Project file": args.project_path,
        "Output directory": args.output_dir or "auto",
        "Config file": args.config or "default",
        "Epochs": args.epochs,
        "Batch size": args.batch_size,
        "GPU": gpu if gpu is not None else "CPU",
    })
    
    try:
        model_path = train_sleap(
            args.project_path,
            args.output_dir,
            args.config,
            args.epochs,
            args.batch_size,
            gpu
        )
        logger.info(f"Model saved to: {model_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
