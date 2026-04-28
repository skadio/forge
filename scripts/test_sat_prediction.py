import argparse
import pickle
import sys
import os

# Add parent directory to path to use local forge module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import gurobipy as gp
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

from forge.embeddings import Forge
from forge.processor import SATProcessor, _SATUtils
from forge.utils import Constants

SEPARATOR = "=" * 80


def get_device(num_gpus: int, gpu_ids: str):
    """Determine the device to use for evaluation. Returns (device, gpu_ids_list)."""
    if num_gpus == -1:
        return torch.device("cpu"), []
    
    if not torch.cuda.is_available():
        return torch.device("cpu"), []
    
    if num_gpus == 0:
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = min(num_gpus, torch.cuda.device_count())
    
    if gpu_ids is not None:
        gpu_ids = [int(g.strip()) for g in gpu_ids.split(',')]
        gpu_ids = [g for g in gpu_ids if g < torch.cuda.device_count()]
    else:
        gpu_ids = list(range(num_gpus))
    
    return torch.device(f"cuda:{gpu_ids[0]}"), gpu_ids


def filter_by_criteria(items, problem_type: str = None, difficulty: str = None):
    """Filter items by problem_type and/or difficulty in filename or instance_name."""
    if not problem_type and not difficulty:
        return items
    
    filtered = []
    for item in items:
        # Extract filename from various item formats
        filename_to_check = None
        
        if isinstance(item, tuple):
            # Could be (satinfo, filename) or (filename,) etc.
            for elem in item:
                if isinstance(elem, str):
                    filename_to_check = elem
                    break
                elif hasattr(elem, 'instance_name'):
                    # It's a SATInfo object
                    filename_to_check = elem.instance_name
                    break
        elif isinstance(item, str):
            filename_to_check = item
        elif hasattr(item, 'instance_name'):
            # It's a SATInfo object
            filename_to_check = item.instance_name
        
        if filename_to_check is None:
            continue
        
        filename = os.path.basename(filename_to_check).lower()
        
        if problem_type and problem_type.lower() not in filename:
            continue
        if difficulty and difficulty.lower() not in filename:
            continue
        
        filtered.append(item)
    
    return filtered

SEPARATOR = "=" * 80

if __name__ == "__main__":

    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config_yaml', type=str, default='../forge/configs/1024_train_config.yaml',
                        help='Path to training config YAML file')
    parser.add_argument('--input_forge_finetuned_pkl', type=str, default='../models/forge_sat_finetuned.pkl',
                        help='Path to fine-tuned Forge pickle file')
    parser.add_argument('--input_sat_folder', type=str, default='../data/g4satbench_sat_instances/',
                        help='Path to SAT folder containing SAT instances in LP/MPS format')
    parser.add_argument('--input_sat_instances_file', type=str, default='../data/configs/g4satbench_test.txt',
                        help='File containing list of SAT instances to use from input_sat_folder')
    parser.add_argument('--input_satinfo_pkl', type=str, default='../models/g4satbench_test_sat_to_satinfo.pkl',
                        help='Path to pickle file containing pre-computed SATInfo objects. If provided, loads from pickle instead of SAT files.')
    parser.add_argument('--problem_type', type=str, default=None,
                        help='Filter instances by problem type (e.g., "3-sat", "k-domset", "sr", "ps"). If None, uses all instances.')
    parser.add_argument('--difficulty', type=str, default=None,
                        help='Filter instances by difficulty level (e.g., "easy", "medium", "hard"). If None, uses all instances.')
    parser.add_argument('--max_graph_nodes', type=int, default=100000,
                        help='Maximum number of graph nodes (clauses + variables) when converting SAT instances')
    parser.add_argument('--num_gpus', type=int, default=0,
                        help='Number of GPUs to use for evaluation (0 = auto-detect, -1 = CPU only)')
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='Comma-separated list of GPU IDs to use (e.g., "0,1,2"). If None, uses all available GPUs.')
    parser.add_argument('--output_metrics_file', type=str, default=None,
                        help='Optional file path to save evaluation metrics')
    args = parser.parse_args()

    # Forge model
    forge = Forge(args.train_config_yaml)

    # Determine device
    device, gpu_ids = get_device(args.num_gpus, args.gpu_ids)
    
    # Print GPU information
    print(f"\n{SEPARATOR}")
    print(f"GPU INFORMATION")
    print(f"{SEPARATOR}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if device.type == "cpu":
        print(f"Using CPU only")
    else:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Using GPUs: {gpu_ids}")
        for gpu_id in gpu_ids:
            print(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    
    print(f"{SEPARATOR}\n", flush=True)

    # Load fine-tuned model
    print(f"Loading fine-tuned model from {args.input_forge_finetuned_pkl}...", flush=True)
    forge.to(device)
    forge_base = forge.module if isinstance(forge, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else forge
    forge_base.load_model(input_forge_pkl=args.input_forge_finetuned_pkl, 
                          model_type=Constants.FORGE_FINE_TUNE_SAT)
    forge.eval()  # Set to evaluation mode
    print(f"Model loaded successfully.\n", flush=True)

    # Load SATInfo objects from pickle or SAT files
    satinfo_data = None  # List of (satinfo, filename) tuples
    
    if args.input_satinfo_pkl:
        # Load pre-computed SATInfo objects from pickle
        print(f"Loading pre-computed SATInfo objects from {args.input_satinfo_pkl}...", flush=True)
        with open(args.input_satinfo_pkl, 'rb') as f:
            satinfo_dict = pickle.load(f)
        
        # Convert to list of (satinfo, filename) tuples
        if isinstance(satinfo_dict, dict):
            satinfo_data = [(v, k) for k, v in satinfo_dict.items()]
        elif isinstance(satinfo_dict, list):
            if satinfo_dict and isinstance(satinfo_dict[0], tuple):
                satinfo_data = satinfo_dict
            else:
                # Create tuples: for each SATInfo, use its instance_name if available, else generate name
                satinfo_data = []
                for idx, satinfo in enumerate(satinfo_dict):
                    filename = satinfo.instance_name if (hasattr(satinfo, 'instance_name') and satinfo.instance_name) else f"instance_{idx}"
                    satinfo_data.append((satinfo, filename))
        else:
            print(f"ERROR: Unsupported pickle format. Expected dict or list.")
            sys.exit(1)
        
        print(f"Loaded {len(satinfo_data)} SATInfo objects from pickle.\n", flush=True)
        
        # Filter by criteria
        original_count = len(satinfo_data)
        satinfo_data = filter_by_criteria(satinfo_data, args.problem_type, args.difficulty)
        if original_count != len(satinfo_data):
            print(f"After filtering: {len(satinfo_data)} SATInfo objects\n", flush=True)
        
        gurobi_env = None
    else:
        # Get list of SAT instances from files
        if args.input_sat_instances_file:
            with open(args.input_sat_instances_file, 'r') as f:
                sat_files = [os.path.join(args.input_sat_folder, line.strip()) for line in f.readlines()]
        else:
            sat_files = []
            for root, dirs, files in os.walk(args.input_sat_folder):
                for file in files:
                    if file.endswith(('.lp', '.mps')):
                        sat_files.append(os.path.join(root, file))
        
        # Filter by criteria
        sat_files = filter_by_criteria([(f,) for f in sat_files], args.problem_type, args.difficulty)
        sat_files = [f[0] for f in sat_files]
        
        print(f"Found {len(sat_files)} SAT instances to evaluate.\n", flush=True)
        gurobi_env = _SATUtils.start_gurobi_env()

    # Evaluate model
    predictions = []
    ground_truth = []
    prediction_scores = []
    file_paths = []
    skipped_files = []

    print(f"{SEPARATOR}")
    print(f"EVALUATING SAT PREDICTION MODEL")
    print(f"{SEPARATOR}\n", flush=True)

    with torch.no_grad():
        # Determine what to iterate over
        if satinfo_data is not None:
            iteration_data = satinfo_data
            desc = "Evaluating (from pickle)"
        else:
            iteration_data = [(sat_file,) for sat_file in sat_files]
            desc = "Evaluating"
        
        for item in tqdm(iteration_data, desc=desc):
            try:
                if satinfo_data is not None:
                    satinfo, sat_file = item
                else:
                    sat_file = item[0]
                    sat_model = gp.read(sat_file, env=gurobi_env)
                    satinfo = SATProcessor._sat_model_to_satinfo(sat_model)
                    
                    if satinfo is None:
                        skipped_files.append((sat_file, "Conversion failed"))
                        continue
                
                # Skip if too large
                num_nodes = satinfo.num_clauses + satinfo.num_vars
                if num_nodes > args.max_graph_nodes:
                    skipped_files.append((sat_file, f"Too large ({num_nodes} nodes > {args.max_graph_nodes})"))
                    continue

                # Move tensors to device
                edge_index = satinfo.edge_index.to(device)
                edge_weight = satinfo.edge_weight.to(device)
                feature_tensor = satinfo.feature_tensor.to(device)

                # Forward pass
                h_list, _, _, _, _ = forge(feature_tensor, 
                                            satinfo.num_clauses,
                                            satinfo.num_vars, 
                                            edge_index, 
                                            edge_weight)

                # Predict satisfiability from SAT satisfiability head
                # h_list[-1] is the SAT head output with shape [num_nodes, 1]
                sat_pred_logit = torch.mean(h_list[-1])
                sat_pred_score = torch.sigmoid(sat_pred_logit).item()
                sat_pred = 1 if sat_pred_score >= 0.5 else 0

                # Extract ground truth label from filename
                sat_true = 0 if "_unsat" in sat_file else 1

                # Store results
                predictions.append(sat_pred)
                prediction_scores.append(sat_pred_score)
                ground_truth.append(sat_true)
                file_paths.append(sat_file)

            except Exception as e:
                skipped_files.append((sat_file, str(e)))
                continue

    # Compute metrics
    print(f"\n{SEPARATOR}")
    print(f"EVALUATION RESULTS")
    print(f"{SEPARATOR}\n", flush=True)

    if len(predictions) == 0:
        print("ERROR: No valid predictions made. Check input folder and file format.")
        sys.exit(1)

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    prediction_scores = np.array(prediction_scores)

    # Compute classification metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, zero_division=0)
    recall = recall_score(ground_truth, predictions, zero_division=0)
    f1 = f1_score(ground_truth, predictions, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(ground_truth, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # ROC-AUC
    try:
        auc = roc_auc_score(ground_truth, prediction_scores)
    except:
        auc = None
    
    # Specificity and FPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Print results
    print(f"Total instances evaluated: {len(predictions)}")
    print(f"Skipped instances: {len(skipped_files)}")
    print(f"\nClassification Metrics:")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  F1-Score:    {f1:.4f}")
    if auc is not None:
        print(f"  ROC-AUC:     {auc:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  FPR:         {fpr:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:5d}  |  FP: {fp:5d}")
    print(f"  FN: {fn:5d}  |  TP: {tp:5d}")
    
    # Count by true label
    n_sat = np.sum(ground_truth)
    n_unsat = len(ground_truth) - n_sat
    print(f"\nLabel Distribution:")
    print(f"  SAT instances:   {n_sat}")
    print(f"  UNSAT instances: {n_unsat}")
    
    print(f"\n{SEPARATOR}\n", flush=True)

    # Print skipped files summary
    if skipped_files:
        print(f"Skipped {len(skipped_files)} instances:")
        for file, reason in skipped_files[:10]:
            print(f"  - {os.path.basename(file)}: {reason}")
        if len(skipped_files) > 10:
            print(f"  ... and {len(skipped_files) - 10} more")
        print()

    # Save metrics to file if requested
    if args.output_metrics_file:
        with open(args.output_metrics_file, 'w') as f:
            f.write("SAT Prediction Model Evaluation Results\n")
            f.write(f"{SEPARATOR}\n\n")
            f.write(f"Total instances evaluated: {len(predictions)}\n")
            f.write(f"Skipped instances: {len(skipped_files)}\n\n")
            f.write("Classification Metrics:\n")
            f.write(f"  Accuracy:    {accuracy:.4f}\n")
            f.write(f"  Precision:   {precision:.4f}\n")
            f.write(f"  Recall:      {recall:.4f}\n")
            f.write(f"  F1-Score:    {f1:.4f}\n")
            if auc is not None:
                f.write(f"  ROC-AUC:     {auc:.4f}\n")
            f.write(f"  Specificity: {specificity:.4f}\n")
            f.write(f"  FPR:         {fpr:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(f"  TN: {tn}\n  FP: {fp}\n  FN: {fn}\n  TP: {tp}\n\n")
            f.write("Label Distribution:\n")
            f.write(f"  SAT instances:   {n_sat}\n")
            f.write(f"  UNSAT instances: {n_unsat}\n")
        
        print(f"Metrics saved to {args.output_metrics_file}")
