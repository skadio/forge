#!/usr/bin/env python3
"""
Single-GPU SAT Pretraining Script

This script is for testing and debugging on a single GPU.
For multi-GPU training, use sat_pretrain.py with torchrun.

Usage:
    python sat_pretrain_single_gpu.py \
        --input_sat_folder ../data/subset_sat_instances/ \
        --epochs 10 \
        --steps_per_instance 10
"""

import argparse
import sys
import os

# PyTorch memory optimization: Enable expandable GPU memory segments BEFORE importing torch
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# Add parent directory to path to use local forge module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from forge.embeddings import Forge
from forge.pipeline import sat_pretrain

if __name__ == "__main__":
    # Parameters
    parser = argparse.ArgumentParser(description='Single-GPU SAT pretraining (testing/debugging)')
    parser.add_argument('--train_config_yaml', type=str, default='../forge/configs/train_config.yaml',
                        help='Path to the training configuration YAML file')
    parser.add_argument('--input_sat_folder', type=str, default='../data/subset_sat_instances/',
                        help='Directory containing input SAT instance files (LP/MPS format)')
    parser.add_argument('--input_sat_instances_file', type=str, default=None,
                        help='File containing list of SAT instances to use from input_sat_folder')
    parser.add_argument('--output_sat_to_satinfo_pkl', type=str, default='../models/subset_sat_to_satinfo.pkl',
                        help='Output path for the sat_to_satinfo pickle')
    parser.add_argument('--output_forge_pretrained_pkl', type=str, default='../models/forge_sat_pretrained.pkl',
                        help='Output path for the pretrained Forge pickle')
    parser.add_argument('--output_log_file', type=str, default='../models/forge_sat_pretrained.log',
                        help='Path to write the pretraining log')
    parser.add_argument('--input_sat_to_satinfo_pkl', type=str, default=None,
                        help='Optional path to an existing sat_to_satinfo pickle to load instead of generating it')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--steps_per_instance', type=int, default=10,
                        help='Number of training steps per SAT instance per epoch')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for the optimizer')
    parser.add_argument('--max_graph_nodes', type=int, default=100000,
                        help='Maximum number of graph nodes when converting SAT instances to bipartite graph')
    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Training on CPU will be very slow.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print(f"SINGLE-GPU TRAINING (No distributed training)")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*80}\n", flush=True)

    # Create Forge with training configuration
    forge = Forge(args.train_config_yaml)
    forge = forge.to(device)

    # Pre-train Forge on SAT instances
    # Note: rank=0, world_size=1 for single-GPU training (no distributed)
    sat_pretrain(forge=forge,
                 input_sat_folder=args.input_sat_folder,
                 input_sat_instances_file=args.input_sat_instances_file,
                 output_sat_to_satinfo_pkl=args.output_sat_to_satinfo_pkl,
                 output_forge_pretrained_pkl=args.output_forge_pretrained_pkl,
                 output_log_file=args.output_log_file,
                 input_sat_to_satinfo_pkl=args.input_sat_to_satinfo_pkl,
                 epochs=args.epochs,
                 steps_per_instance=args.steps_per_instance,
                 learning_rate=args.learning_rate,
                 weight_decay=args.weight_decay,
                 max_graph_nodes=args.max_graph_nodes,
                 rank=0,
                 world_size=1)
    
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Model saved to: {args.output_forge_pretrained_pkl}")
    print(f"{'='*80}\n", flush=True)
