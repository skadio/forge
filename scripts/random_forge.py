import argparse
import sys
import os

# Add parent directory to path to use local forge module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from forge.embeddings import Forge

if __name__ == "__main__":
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config_yaml', type=str, default='../forge/configs/train_config.yaml',
                        help='Path to the training configuration YAML file')
    parser.add_argument('--output_random_forge_pkl', type=str, default='../models/random_forge.pkl',
                        help='Output path for the randomly initialized Forge model state dict')
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"RANDOM FORGE INITIALIZATION")
    print(f"{'='*80}")
    print(f"Config file: {args.train_config_yaml}")
    print(f"Output path: {args.output_random_forge_pkl}")

    # Create randomly initialized Forge model (no special layers like satisfiability layer)
    forge = Forge(args.train_config_yaml)
    
    # Move to CPU (or GPU if available)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    forge = forge.to(device)
    
    print(f"Device: {device}")
    print(f"Model created with random initialization")
    print(f"Total parameters: {sum(p.numel() for p in forge.parameters())}")

    # Save the randomly initialized model's state dict (compatible with torch.load)
    torch.save(forge.state_dict(), args.output_random_forge_pkl)

    print(f"\n{'='*80}")
    print(f"Random Forge model state dict saved to: {args.output_random_forge_pkl}")
    print(f"{'='*80}\n", flush=True)
