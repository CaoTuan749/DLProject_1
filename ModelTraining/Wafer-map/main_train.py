#!/usr/bin/env python
"""
Main Training Launcher

This script allows you to choose between:
  - Continual learning training (train_continual.py)
  - Standard full dataset training (train_normal.py)
  
Usage:
    python main_train.py --mode continual --config config.yaml
    or
    python main_train.py --mode standard --config config.yaml
"""

import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Choose Training Pipeline")
    parser.add_argument("--mode", type=str, choices=["continual", "standard"], default="standard",
                        help="Training mode: 'continual' for continual learning, 'standard' for full dataset training.")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to YAML configuration file")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.mode == "continual":
        print("Launching Continual Learning Training Pipeline...")
        subprocess.run(["python", "Wafer-map/train_continual.py", "--config", args.config], check=True)
    else:
        print("Launching Standard Training Pipeline...")
        subprocess.run(["python", "Wafer-map/train_normal.py", "--config", args.config], check=True)

if __name__ == '__main__':
    main()
