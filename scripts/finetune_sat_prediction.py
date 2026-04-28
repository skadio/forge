import argparse
import sys
import os
import pickle

# Add parent directory to path to use local forge module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from forge.embeddings import Forge
from forge.pipeline import finetune_sat_prediction
from forge.labeler import SATLabeler
from forge.utils import Constants


def filter_by_criteria(items, problem_type: str = None, difficulty: str = None):
    """Filter items by problem_type and/or difficulty in filename or instance_name.
    
    Parameters
    ----------
    items : dict or list
        Either a dict mapping filenames to satinfo, or list of (satinfo, filename) tuples
    problem_type : str, optional
        Problem type filter (e.g., "3-sat", "k-domset", "sr", "ps")
    difficulty : str, optional
        Difficulty filter (e.g., "easy", "medium", "hard")
    
    Returns
    -------
    dict or list
        Filtered items in the same format as input
    """
    if not problem_type and not difficulty:
        return items
    
    filtered = {}
    is_dict = isinstance(items, dict)
    
    if is_dict:
        items_to_check = items.items()
    else:
        items_to_check = [(None, item) for item in items]
    
    for key, item in items_to_check:
        # Extract filename from various item formats
        filename_to_check = None
        
        if is_dict:
            # Key is filename, value is satinfo
            filename_to_check = key
        elif isinstance(item, tuple):
            # Could be (satinfo, filename) or (filename,) etc.
            for elem in item:
                if isinstance(elem, str):
                    filename_to_check = elem
                    break
                elif hasattr(elem, 'instance_name'):
                    filename_to_check = elem.instance_name
                    break
        elif isinstance(item, str):
            filename_to_check = item
        elif hasattr(item, 'instance_name'):
            filename_to_check = item.instance_name
        
        if filename_to_check is None:
            continue
        
        filename = os.path.basename(filename_to_check).lower()
        
        if problem_type and problem_type.lower() not in filename:
            continue
        if difficulty and difficulty.lower() not in filename:
            continue
        
        if is_dict:
            filtered[key] = item
        else:
            filtered[item] = True
    
    if is_dict:
        return filtered
    else:
        return [item for item in items if item in filtered]

if __name__ == "__main__":

    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config_yaml', type=str, default='../forge/configs/512_train_config_april_26.yaml',
                        help='Path to training config YAML file')
    parser.add_argument('--input_forge_pkl', type=str, default='../models/forge_sat_pretrained.pkl',
                        help='Path to pretrained Forge pickle file (from sat_pretrain.py)')
    parser.add_argument('--input_sat_folder', type=str, default='../data/g4satbench_sat_instances/',
                        help='Path to SAT folder containing SAT instances in LP/MPS format')
    parser.add_argument('--input_sat_instances_file', type=str, default="../data/configs/g4satbench_train.txt",
                        help='File containing list of SAT instances to use from input_sat_folder')
    parser.add_argument('--output_forge_finetuned_pkl', type=str, default=None,
                        help='Path to save fine-tuned Forge pickle file. If None, generates name from config and problem/difficulty filters')
    parser.add_argument('--output_sat_to_satinfo_pkl', type=str, default='../models/finetuned_sat_to_satinfo.pkl',
                        help='Output pickle file to store sat_to_satisfiability_info mapping')
    parser.add_argument('--output_log_file', type=str, default=None,
                        help='Path to write the fine-tuning log. If None, generates name from model filename')
    parser.add_argument('--input_sat_to_satinfo_pkl', type=str, default="../models/g4satbench_train_april19_sat_to_satinfo.pkl",
                        help='Optional path to an existing sat_to_satinfo_pkl to load instead of creating')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--steps_per_instance', type=int, default=10,
                        help='Number of training steps per SAT instance per epoch')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay for the optimizer')
    parser.add_argument('--max_graph_nodes', type=int, default=100000,
                        help='Maximum number of graph nodes (clauses + variables) when converting SAT instances to bipartite graph')
    parser.add_argument('--freeze_level', type=str, default='none', choices=['none', 'partial', 'full'],
                        help='How much to freeze: none (train all), partial (freeze early layers), full (SAT head only)')
    parser.add_argument('--bce_weight', type=float, default=0.5,
                        help='Weight for BCE loss. Use 1.0 for classification-only, 0.95 for combined, 0.0 to try contrastive-only.')
    parser.add_argument('--contrastive_weight', type=float, default=0.5,
                        help='Weight for contrastive loss. Use 0.0 for classification-only, 0.05 for combined, 1.0 to try pure contrastive.')
    parser.add_argument('--num_gpus', type=int, default=0,
                        help='Number of GPUs to use for training (0 = auto-detect, -1 = CPU only)')
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='Comma-separated list of GPU IDs to use (e.g., "0,1,2"). If None, uses all available GPUs.')
    parser.add_argument('--problem_type', type=str, default=None,
                        help='Filter instances by problem type (e.g., "3-sat", "k-domset", "sr", "ps"). If None, uses all instances.')
    parser.add_argument('--difficulty', type=str, default=None,
                        help='Filter instances by difficulty level (e.g., "easy", "medium", "hard"). If None, uses all instances.')
    args = parser.parse_args()

    # Forge model ready for fine-tuning
    forge = Forge(args.train_config_yaml)

    # Print GPU information
    print(f"\n{'='*80}")
    print(f"GPU INFORMATION")
    print(f"{'='*80}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Determine GPU configuration
    if args.num_gpus == -1:
        # CPU only
        device = torch.device("cpu")
        num_gpus = 0
        print(f"Using CPU only (num_gpus=-1)")
    else:
        if torch.cuda.is_available():
            if args.num_gpus == 0:
                # Auto-detect all available GPUs
                num_gpus = torch.cuda.device_count()
            else:
                num_gpus = min(args.num_gpus, torch.cuda.device_count())
            
            # Parse GPU IDs if provided
            if args.gpu_ids is not None:
                gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(',')]
                gpu_ids = [g for g in gpu_ids if g < torch.cuda.device_count()]
                num_gpus = len(gpu_ids)
            else:
                gpu_ids = list(range(num_gpus))
            
            device = torch.device(f"cuda:{gpu_ids[0]}")
            
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Using GPUs: {gpu_ids}")
            for gpu_id in gpu_ids:
                print(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            num_gpus = 0
            device = torch.device("cpu")
            print(f"WARNING: CUDA is not available - using CPU")
    
    print(f"Forge device: {forge.device}")
    print(f"{'='*80}\n", flush=True)

    # Move model to primary device
    forge.to(device)
    
    # Wrap model with DataParallel if using multiple GPUs
    if num_gpus > 1:
        print(f"Wrapping model with DataParallel for {num_gpus} GPUs...\n", flush=True)
        if args.gpu_ids is not None:
            gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(',')]
            gpu_ids = [g for g in gpu_ids if g < torch.cuda.device_count()]
        else:
            gpu_ids = list(range(num_gpus))
        forge = nn.DataParallel(forge, device_ids=gpu_ids)

    # Load and filter SAT to satinfo
    # IMPORTANT: When filters are applied, regenerate from input_sat_instances_file to ensure
    # we only use instances that match BOTH the instances file AND the filter criteria
    input_sat_to_satinfo_pkl_filtered = args.input_sat_to_satinfo_pkl
    if args.problem_type or args.difficulty:
        # When filters are specified, regenerate satinfo from input_sat_folder + input_sat_instances_file
        # This ensures we respect both the instances file AND the filter criteria
        print(f"Filters specified (problem_type={args.problem_type}, difficulty={args.difficulty})")
        print(f"Regenerating SAT info from {args.input_sat_folder} with instances from {args.input_sat_instances_file}...", flush=True)
        
        labeler = SATLabeler()
        sat_to_satinfo = labeler.convert_sat_to_satisfiability_info(
            input_sat_folder=args.input_sat_folder,
            input_sat_instances_file=args.input_sat_instances_file,
            output_sat_to_satinfo_pkl=args.output_sat_to_satinfo_pkl,
            has_return=True
        )
        
        unfiltered_count = len(sat_to_satinfo)
        sat_to_satinfo = filter_by_criteria(sat_to_satinfo, args.problem_type, args.difficulty)
        filtered_count = len(sat_to_satinfo)
        
        print(f"\nFiltered from {unfiltered_count} to {filtered_count} instances")
        print(f"  Criteria: problem_type={args.problem_type}, difficulty={args.difficulty}\n", flush=True)
        
        # Create a temporary filtered pickle file with descriptive name including filter criteria
        basename = os.path.splitext(os.path.basename(args.input_sat_to_satinfo_pkl))[0]
        name_parts = [basename]
        if args.problem_type:
            name_parts.append(args.problem_type)
        if args.difficulty:
            name_parts.append(args.difficulty)
        filtered_filename = '_'.join(name_parts) + '_filtered.pkl'
        filtered_pkl_path = os.path.join(os.path.dirname(args.input_sat_to_satinfo_pkl), filtered_filename)
        
        with open(filtered_pkl_path, 'wb') as f:
            pickle.dump(sat_to_satinfo, f)
        input_sat_to_satinfo_pkl_filtered = filtered_pkl_path
        print(f"Saved filtered satinfo to {filtered_pkl_path}\n", flush=True)

    # Generate output_forge_finetuned_pkl filename based on config and filters if not provided
    if args.output_forge_finetuned_pkl is None:
        # Extract config name without extension
        config_basename = os.path.splitext(os.path.basename(args.train_config_yaml))[0]
        
        # Build name components
        name_parts = ['forge_sat_finetuned', config_basename]
        if args.problem_type:
            name_parts.append(args.problem_type)
        if args.difficulty:
            name_parts.append(args.difficulty)
        
        filename = '_'.join(name_parts) + '.pkl'
        args.output_forge_finetuned_pkl = os.path.join(os.path.dirname(args.output_sat_to_satinfo_pkl), filename)
        print(f"Generated output filename: {args.output_forge_finetuned_pkl}\n", flush=True)
    
    # Generate output_log_file to match the model filename if not provided
    if args.output_log_file is None:
        log_filename = os.path.splitext(os.path.basename(args.output_forge_finetuned_pkl))[0] + '.log'
        args.output_log_file = os.path.join(os.path.dirname(args.output_forge_finetuned_pkl), log_filename)
        print(f"Generated log filename: {args.output_log_file}\n", flush=True)

    # Fine-tune Forge to predict SAT satisfiability
    finetune_sat_prediction(forge=forge,
                           input_forge_pkl=args.input_forge_pkl,
                           model_type=Constants.FORGE_FINE_TUNE_SAT,
                           input_sat_folder=args.input_sat_folder,
                           input_sat_instances_file=args.input_sat_instances_file,
                           output_forge_finetuned_pkl=args.output_forge_finetuned_pkl,
                           output_sat_to_satinfo_pkl=args.output_sat_to_satinfo_pkl,
                           output_log_file=args.output_log_file,
                           input_sat_to_satinfo_pkl=input_sat_to_satinfo_pkl_filtered,
                           epochs=args.epochs,
                           steps_per_instance=args.steps_per_instance,
                           learning_rate=args.learning_rate,
                           weight_decay=args.weight_decay,
                           max_graph_nodes=args.max_graph_nodes,
                           freeze_level=args.freeze_level,
                           bce_weight=args.bce_weight,
                           contrastive_weight=args.contrastive_weight)

