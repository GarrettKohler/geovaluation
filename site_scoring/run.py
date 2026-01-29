#!/usr/bin/env python3
"""
Quick-start script for site scoring model.
Run: python -m site_scoring.run
"""

import argparse
import torch
from pathlib import Path

from .config import Config
from .train import run_training


def main():
    parser = argparse.ArgumentParser(description="Train Site Scoring Model")
    parser.add_argument(
        "--target",
        type=str,
        default="revenue",
        choices=["revenue", "monthly_impressions", "monthly_revenue_per_screen"],
        help="Target variable to predict",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size (default: 4096 for M4 GPU)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: mps, cpu, or cuda (auto-detected if not specified)",
    )
    args = parser.parse_args()

    # Create config
    config = Config(
        target=args.target,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
    )

    if args.device:
        config.device = args.device

    # Print system info
    print("\n" + "=" * 60)
    print("System Information")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"Selected device: {config.device}")

    # Run training
    model, processor, test_metrics = run_training(config)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {config.output_dir / 'best_model.pt'}")
    print(f"Preprocessor saved to: {config.output_dir / 'preprocessor.pkl'}")
    print(f"\nFinal Test Metrics:")
    print(f"  MAE: ${test_metrics['test_mae']:,.2f}")
    print(f"  R²: {test_metrics['test_r2']:.4f}")


if __name__ == "__main__":
    main()
