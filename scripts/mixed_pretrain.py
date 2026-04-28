import argparse
import sys
import os
import traceback

# Unbuffered output - see logs in real-time even during crashes
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

print("[STARTUP] Script starting...", file=sys.stderr, flush=True)

# PyTorch memory optimization: Enable expandable GPU memory segments BEFORE importing torch
# This reduces OOM errors when training models across GPUs
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# NCCL debugging (set these BEFORE importing torch.distributed)
# These help diagnose distributed training issues like timeouts
os.environ['NCCL_DEBUG'] = 'INFO'  # Enable NCCL logging (INFO, WARN, TRACE)
os.environ['NCCL_TIMEOUT'] = '600'  # NCCL collective ops timeout (seconds), will be increased if distributed

# Add parent directory to path to use local forge module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("[STARTUP] Importing torch...", file=sys.stderr, flush=True)
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    print("[STARTUP] PyTorch imported successfully", file=sys.stderr, flush=True)
except Exception as e:
    print(f"[ERROR] Failed to import torch: {e}", file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

try:
    print("[STARTUP] Importing Forge modules...", file=sys.stderr, flush=True)
    from forge.embeddings import Forge
    from forge.pipeline import pretrain, sat_pretrain, mixed_pretrain
    print("[STARTUP] Forge modules imported successfully", file=sys.stderr, flush=True)
except Exception as e:
    print(f"[ERROR] Failed to import Forge: {e}", file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config_yaml', type=str, default='../forge/configs/mixed_train_config.yaml',
                        help='Path to the training configuration YAML file')
    
    # MIP-related arguments
    parser.add_argument('--input_mip_folder', type=str, default='../data/instances/',
                        help='Directory containing input MIP instance files')
    parser.add_argument('--input_mip_instances_file', type=str, default='../data/configs/iclr_forge_pretrain.txt',
                        help='File containing list of MIP instances to use from input_mip_folder')
    parser.add_argument('--output_mip_to_mipinfo_pkl', type=str, default='../models/iclr_forge_pretrain_mip_to_mipinfo.pkl',
                        help='Output path for the mip_to_mipinfo pickle')
    parser.add_argument('--input_mip_to_mipinfo_pkl', type=str, default='../models/iclr_forge_pretrain_mip_to_mipinfo.pkl',
                        help='Optional path to an existing mip_to_mipinfo pickle to load instead of generating it')
    parser.add_argument('--mip_relaxation_list', nargs='*', type=float, default=[0.05, 0.01],
                        help='Space-separated list of relaxation values to use during MIP pretraining')
    
    # SAT-related arguments
    parser.add_argument('--input_sat_folder', type=str, default='./data/g4satbench_sat_instances/',
                        help='Directory containing input SAT instance files (LP/MPS format)')
    parser.add_argument('--input_sat_instances_file', type=str, default='../data/configs/g4satbench_train.txt',
                        help='File containing list of SAT instances to use from input_sat_folder')
    parser.add_argument('--output_sat_to_satinfo_pkl', type=str, default='../models/g4satbench_train_sat_to_satinfo.pkl',
                        help='Output path for the sat_to_satinfo pickle')
    parser.add_argument('--input_sat_to_satinfo_pkl', type=str, default='../models/g4satbench_train_sat_to_satinfo.pkl',
                        help='Optional path to an existing sat_to_satinfo pickle to load instead of generating it')
    
    # Combined training arguments
    parser.add_argument('--output_forge_pretrained_pkl', type=str, default='../models/forge_mixed_pretrained.pkl',
                        help='Output path for the final pretrained Forge pickle')
    parser.add_argument('--output_log_file', type=str, default='../models/forge_mixed_pretrained.log',
                        help='Path to write the combined pretraining log')
    
    # Mixed batch training mode
    parser.add_argument('--use_mixed_batch_training', action='store_true',
                        help='Use mixed-batch training with interleaved SAT and MIP instances instead of sequential phases')
    parser.add_argument('--mip_sat_ratio', type=float, default=0.5,
                        help='Ratio of MIP to SAT instances when interleaving (0.0-1.0, default 0.5 for alternating)')
    
    # Training hyperparameters
    parser.add_argument('--mip_epochs', type=int, default=5,
                        help='Number of training epochs for MIP pretraining')
    parser.add_argument('--sat_epochs', type=int, default=10,
                        help='Number of training epochs for SAT pretraining')
    parser.add_argument('--steps_per_instance', type=int, default=10,
                        help='Number of training steps per instance per epoch')
    parser.add_argument('--mip_learning_rate', type=float, default=1e-4,
                        help='Learning rate for MIP pretraining')
    parser.add_argument('--sat_learning_rate', type=float, default=5e-5,
                        help='Learning rate for SAT pretraining (reduced to prevent overfitting)')
    parser.add_argument('--mip_weight_decay', type=float, default=1e-4,
                        help='Weight decay for MIP pretraining')
    parser.add_argument('--sat_weight_decay', type=float, default=1e-3,
                        help='Weight decay for SAT pretraining (higher L2 regularization to prevent overfitting)')
    parser.add_argument('--max_graph_nodes', type=int, default=2100,
                        help='Maximum number of graph nodes when converting instances to bipartite graph')
    parser.add_argument('--sat_max_graph_nodes', type=int, default=100000,
                        help='Maximum number of graph nodes for SAT instances')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help='Number of steps to accumulate gradients before optimizer step (for SAT pretraining)')
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.8,
                        help='Target GPU memory usage fraction (default: 0.8). Smart fallback to CPU if exceeded.')
    parser.add_argument('--skip_mip_pretraining', action='store_true',
                        help='Skip MIP pretraining and only do SAT pretraining')
    parser.add_argument('--skip_sat_pretraining', action='store_true',
                        help='Skip SAT pretraining and only do MIP pretraining')
    
    args = parser.parse_args()
    
    print(f"[STARTUP] Arguments parsed successfully", file=sys.stderr, flush=True)
    print(f"[STARTUP] use_mixed_batch_training={args.use_mixed_batch_training}", file=sys.stderr, flush=True)

    # Detect if running under torchrun (multi-process distributed training)
    # torchrun sets RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    
    print(f"[STARTUP] is_distributed={is_distributed}", file=sys.stderr, flush=True)
    
    if is_distributed:
        # Distributed training with torchrun
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        
        # Increase NCCL timeout for distributed training
        # Default is 10min (600s), set to 60min (3600s) for potentially long-running jobs
        os.environ['NCCL_TIMEOUT'] = '3600'
        
        # Initialize the distributed process group
        dist.init_process_group(backend='nccl')
        
        # Set device to this rank's GPU
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"COMBINED SAT+MIP PRETRAINING (DISTRIBUTED WITH TORCHRUN)")
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
        print(f"COMBINED SAT+MIP PRETRAINING (SINGLE GPU)")
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
    try:
        print(f"[STARTUP] Creating Forge model from config: {args.train_config_yaml}", file=sys.stderr, flush=True)
        forge = Forge(args.train_config_yaml)
        print(f"[STARTUP] Forge model created successfully", file=sys.stderr, flush=True)
        forge = forge.to(device)
        print(f"[STARTUP] Forge model moved to {device}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to create/load Forge model: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    
    # Wrap with DistributedDataParallel if in distributed mode
    if is_distributed:
        if rank == 0:
            print(f"Wrapping model with DistributedDataParallel...")
        try:
            forge = nn.parallel.DistributedDataParallel(forge, device_ids=[device.index])
        except Exception as e:
            print(f"[ERROR] Failed to wrap with DistributedDataParallel: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
    
    if rank == 0:
        print(f"Model ready for training\n", flush=True)

    # =========================================================================
    # TRAINING WITH ERROR HANDLING
    # =========================================================================
    try:
        # MIXED BATCH MODE: Train on interleaved SAT and MIP instances simultaneously
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"MIXED BATCH TRAINING")
            print(f"{'='*80}\n", flush=True)

        mixed_pretrain(forge=forge,
                       input_mip_folder=args.input_mip_folder,
                       input_sat_folder=args.input_sat_folder,
                       input_mip_instances_file=args.input_mip_instances_file,
                       input_sat_instances_file=args.input_sat_instances_file,
                       output_mip_to_mipinfo_pkl=args.output_mip_to_mipinfo_pkl,
                       output_sat_to_satinfo_pkl=args.output_sat_to_satinfo_pkl,
                       output_forge_pretrained_pkl=args.output_forge_pretrained_pkl,
                       output_log_file=args.output_log_file,
                       input_mip_to_mipinfo_pkl=args.input_mip_to_mipinfo_pkl,
                       input_sat_to_satinfo_pkl=args.input_sat_to_satinfo_pkl,
                       relaxation_list=args.mip_relaxation_list,
                       mip_sat_ratio=args.mip_sat_ratio,
                       epochs=args.mip_epochs,  # Use MIP epochs as default
                       steps_per_instance=args.steps_per_instance,
                       mip_learning_rate=args.mip_learning_rate,
                       sat_learning_rate=args.sat_learning_rate,
                       mip_weight_decay=args.mip_weight_decay,
                       sat_weight_decay=args.sat_weight_decay,
                       max_mip_graph_nodes=args.max_graph_nodes,
                       max_sat_graph_nodes=args.sat_max_graph_nodes,
                       gradient_accumulation_steps=args.gradient_accumulation_steps,
                       rank=rank,
                       world_size=world_size,
                       gpu_memory_fraction=args.gpu_memory_fraction)

        if rank == 0:
            print(f"\n{'='*80}")
            print(f"Mixed batch training completed!")
            print(f"Model saved to: {args.output_forge_pretrained_pkl}")
            print(f"{'='*80}\n", flush=True)

        else:
            # SEQUENTIAL MODE: MIP pretraining → SAT pretraining (original behavior)

            # =========================================================================
            # PHASE 1: MIP PRETRAINING
            # =========================================================================
            if not args.skip_mip_pretraining:
                if rank == 0:
                    print(f"\n{'='*80}")
                    print(f"PHASE 1: MIP PRETRAINING")
                    print(f"{'='*80}\n", flush=True)
                
                pretrain(forge=forge,
                         input_mip_folder=args.input_mip_folder,
                         input_mip_instances_file=args.input_mip_instances_file,
                         output_mip_to_mipinfo_pkl=args.output_mip_to_mipinfo_pkl,
                         output_forge_pretrained_pkl=args.output_forge_pretrained_pkl,
                         output_log_file=args.output_log_file if args.skip_sat_pretraining else None,
                         input_mip_to_mipinfo_pkl=args.input_mip_to_mipinfo_pkl,
                         relaxation_list=args.mip_relaxation_list,
                         epochs=args.mip_epochs,
                         steps_per_instance=args.steps_per_instance,
                         learning_rate=args.mip_learning_rate,
                         weight_decay=args.mip_weight_decay,
                         max_graph_nodes=args.max_graph_nodes,
                         rank=rank,
                         world_size=world_size,
                         gpu_memory_fraction=args.gpu_memory_fraction)
                
                if rank == 0:
                    print(f"\n{'='*80}")
                    print(f"MIP pretraining completed!")
                    print(f"Model saved to: {args.output_forge_pretrained_pkl}")
                    if not args.skip_sat_pretraining:
                        print(f"Starting SAT pretraining with MIP-pretrained weights...")
                    print(f"{'='*80}\n", flush=True)

            # =========================================================================
            # PHASE 2: SAT PRETRAINING (using MIP-pretrained weights if available)
            # =========================================================================
            if not args.skip_sat_pretraining:
                if rank == 0:
                    print(f"\n{'='*80}")
                    print(f"PHASE 2: SAT PRETRAINING")
                    print(f"{'='*80}\n", flush=True)
                
                # Determine input MIP forge pickle for SAT pretraining
                input_mip_forge_for_sat = None
                if not args.skip_mip_pretraining:
                    # Use the MIP-pretrained model from Phase 1
                    input_mip_forge_for_sat = args.output_forge_pretrained_pkl
                
                sat_pretrain(forge=forge,
                             input_sat_folder=args.input_sat_folder,
                             input_sat_instances_file=args.input_sat_instances_file,
                             output_sat_to_satinfo_pkl=args.output_sat_to_satinfo_pkl,
                             output_forge_pretrained_pkl=args.output_forge_pretrained_pkl,
                             output_log_file=args.output_log_file,
                             input_sat_to_satinfo_pkl=args.input_sat_to_satinfo_pkl,
                             input_mip_forge_pkl=input_mip_forge_for_sat,
                             epochs=args.sat_epochs,
                             steps_per_instance=args.steps_per_instance,
                             learning_rate=args.sat_learning_rate,
                             weight_decay=args.sat_weight_decay,
                             max_graph_nodes=args.sat_max_graph_nodes,
                             gradient_accumulation_steps=args.gradient_accumulation_steps,
                             rank=rank,
                             world_size=world_size,
                             gpu_memory_fraction=args.gpu_memory_fraction)
                
                if rank == 0:
                    print(f"\n{'='*80}")
                    print(f"SAT pretraining completed!")
                    print(f"Model saved to: {args.output_forge_pretrained_pkl}")
                    print(f"{'='*80}\n", flush=True)
        
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"Pretraining finished!")
            print(f"Final model saved to: {args.output_forge_pretrained_pkl}")
            print(f"{'='*80}\n", flush=True)
    
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}", file=sys.stderr, flush=True)
        print(f"[ERROR] Full traceback:", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        if rank == 0 and torch.cuda.is_available():
            print(f"[DEBUG] GPU Memory at crash: allocated={torch.cuda.memory_allocated() / 1e9:.2f}GB, "
                  f"reserved={torch.cuda.memory_reserved() / 1e9:.2f}GB", file=sys.stderr, flush=True)
        # Continue to cleanup even if training fails
    
    # Synchronize all processes before cleanup (critical in distributed training)
    try:
        if is_distributed:
            dist.barrier()  # Wait for all ranks to finish
            if rank == 0:
                print("All ranks synchronized. Cleaning up distributed training...", flush=True)
    except Exception as e:
        print(f"[WARNING] Barrier failed: {e}", file=sys.stderr, flush=True)
    
    # Clean up distributed training
    try:
        if is_distributed:
            dist.destroy_process_group()
            if rank == 0:
                print("Distributed training cleaned up successfully", flush=True)
    except Exception as e:
        print(f"[WARNING] Cleanup failed: {e}", file=sys.stderr, flush=True)
    
    print("[SHUTDOWN] Script completed", file=sys.stderr, flush=True)
