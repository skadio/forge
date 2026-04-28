import argparse
import sys
import os

# PyTorch memory optimization: Enable expandable GPU memory segments BEFORE importing torch
# This reduces OOM errors when training models across GPUs
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# NCCL debugging (set these BEFORE importing torch.distributed)
# These help diagnose distributed training issues like timeouts
os.environ['NCCL_DEBUG'] = 'INFO'  # Enable NCCL logging (INFO, WARN, TRACE)
os.environ['NCCL_TIMEOUT'] = '600'  # NCCL collective ops timeout (seconds), will be increased if distributed

# Add parent directory to path to use local forge module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.distributed as dist
from forge.embeddings import Forge
from forge.pipeline import sat_pretrain

if __name__ == "__main__":
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config_yaml', type=str, default='../forge/configs/train_config.yaml',
                        help='Path to the training configuration YAML file')
    parser.add_argument('--input_sat_folder', type=str, default='../data/satcomp2022anni_train_sat_instances/',
                        help='Directory containing input SAT instance files (LP/MPS format)')
    parser.add_argument('--input_sat_instances_file', type=str, default="../data/configs/satcomp2022anni_train.txt",
                        help='File containing list of SAT instances to use from input_sat_folder')
    parser.add_argument('--output_sat_to_satinfo_pkl', type=str, default='../models/sat_to_satinfo.pkl',
                        help='Output path for the sat_to_satinfo pickle')
    parser.add_argument('--output_forge_pretrained_pkl', type=str, default='../models/forge_sat_pretrained.pkl',
                        help='Output path for the pretrained Forge pickle')
    parser.add_argument('--output_log_file', type=str, default='../models/forge_sat_pretrained.log',
                        help='Path to write the pretraining log')
    parser.add_argument('--input_sat_to_satinfo_pkl', type=str, default="../models/satcomp2022anni_sat_to_satinfo.pkl",
                        help='Optional path to an existing sat_to_satinfo pickle to load instead of generating it')
    parser.add_argument('--input_mip_forge_pkl', type=str, default=None,
                        help='Optional path to a pre-trained Forge-MIP model pickle to use as initial weights')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--steps_per_instance', type=int, default=10,
                        help='Number of training steps per SAT instance per epoch')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate for the optimizer (reduced to prevent overfitting)')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay for the optimizer (L2 regularization to prevent overfitting)')
    parser.add_argument('--max_graph_nodes', type=int, default=100000,
                        help='Maximum number of graph nodes when converting SAT instances to bipartite graph')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help='Number of steps to accumulate gradients before optimizer step (higher = more stable updates, better generalization)')
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.8,
                        help='Target GPU memory usage fraction (default: 0.8). Smart fallback to CPU if exceeded.')
    args = parser.parse_args()

    # Detect if running under torchrun (multi-process distributed training)
    # torchrun sets RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    
    if is_distributed:
        # Distributed training with torchrun
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        
        # Increase NCCL timeout for distributed SAT pre-training
        # Default is 10min (600s), set to 60min (3600s) for potentially long-running jobs
        os.environ['NCCL_TIMEOUT'] = '3600'
        
        # Initialize the distributed process group
        dist.init_process_group(backend='nccl')
        
        # Set device to this rank's GPU
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"SAT PRETRAINING (DISTRIBUTED WITH TORCHRUN)")
            print(f"{'='*80}")
            print(f"Distributed training: {world_size} processes")
            print(f"Backend: nccl")
            print(f"Device (rank 0): {device}")
    else:
        # Single-GPU training (standard python script execution)
        rank = 0
        world_size = 1
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*80}")
        print(f"SAT PRETRAINING (SINGLE GPU)")
        print(f"{'='*80}")
        print(f"Device: {device}")
    
    if torch.cuda.is_available():
        if rank == 0:
            print(f"CUDA available: True")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            if is_distributed:
                print(f"Each of {world_size} processes uses 1 GPU")
            else:
                print(f"Using 1 GPU")
            print(f"GPU Memory fraction target: {args.gpu_memory_fraction}")
    
    if rank == 0:
        print(f"{'='*80}\n", flush=True)

    # Create Forge with training configuration
    forge = Forge(args.train_config_yaml)
    forge = forge.to(device)
    
    # Wrap with DistributedDataParallel if in distributed mode
    if is_distributed:
        if rank == 0:
            print(f"Wrapping model with DistributedDataParallel...")
        forge = nn.parallel.DistributedDataParallel(forge, device_ids=[device.index])
    
    if rank == 0:
        print(f"Model ready for training\n", flush=True)

    # Pre-train Forge on SAT instances
    sat_pretrain(forge=forge,
                 input_sat_folder=args.input_sat_folder,
                 input_sat_instances_file=args.input_sat_instances_file,
                 output_sat_to_satinfo_pkl=args.output_sat_to_satinfo_pkl,
                 output_forge_pretrained_pkl=args.output_forge_pretrained_pkl,
                 output_log_file=args.output_log_file,
                 input_sat_to_satinfo_pkl=args.input_sat_to_satinfo_pkl,
                 input_mip_forge_pkl=args.input_mip_forge_pkl,
                 epochs=args.epochs,
                 steps_per_instance=args.steps_per_instance,
                 learning_rate=args.learning_rate,
                 weight_decay=args.weight_decay,
                 max_graph_nodes=args.max_graph_nodes,
                 rank=rank,
                 world_size=world_size,
                 gpu_memory_fraction=args.gpu_memory_fraction)
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Training completed!")
        print(f"Model saved to: {args.output_forge_pretrained_pkl}")
        print(f"{'='*80}\n", flush=True)
    
    # Synchronize all processes before cleanup (critical in distributed training)
    if is_distributed:
        dist.barrier()  # Wait for all ranks to finish
        if rank == 0:
            print("All ranks synchronized. Cleaning up distributed training...", flush=True)
    
    # Clean up distributed training
    if is_distributed:
        dist.destroy_process_group()
