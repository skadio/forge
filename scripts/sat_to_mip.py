"""
Convert SAT problems in DIMACS CNF format to Mixed Integer Programming (MIP).

This script reads CNF files and converts them to LP format using PuLP,
optionally solving them with the CBC solver.
"""

import argparse
import json
import os
import pickle
from collections import Counter, defaultdict

import numpy as np
import pulp
from pysat.formula import CNF


def is_number(s):
    """Check if a string represents an integer."""
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_expected_satisfiability(filename):
    """Determine if a file should be satisfiable based on filename.
    
    Files starting with 'uu' are unsatisfiable, all others are satisfiable.
    
    Args:
        filename: The name of the CNF file (without path)
        
    Returns:
        bool: True if expected to be satisfiable, False if unsatisfiable
    """
    return not filename.lower().startswith('uu')


def compute_accuracy(results):
    """Compute accuracy metrics from solving results.
    
    Args:
        results: List of dictionaries with keys 'filename', 'expected', 'actual'
        
    Returns:
        dict: Accuracy metrics including overall accuracy, SAT accuracy, UNSAT accuracy
    """
    if not results:
        return {"total": 0, "correct": 0, "accuracy": 0.0}
    
    sat_correct = sum(1 for r in results if r['expected'] and r['actual'])
    sat_total = sum(1 for r in results if r['expected'])
    unsat_correct = sum(1 for r in results if not r['expected'] and not r['actual'])
    unsat_total = sum(1 for r in results if not r['expected'])
    
    total_correct = sat_correct + unsat_correct
    total_problems = len(results)
    accuracy = total_correct / total_problems if total_problems > 0 else 0.0
    
    return {
        "total": total_problems,
        "correct": total_correct,
        "accuracy": accuracy,
        "sat_correct": sat_correct,
        "sat_total": sat_total,
        "sat_accuracy": sat_correct / sat_total if sat_total > 0 else 0.0,
        "unsat_correct": unsat_correct,
        "unsat_total": unsat_total,
        "unsat_accuracy": unsat_correct / unsat_total if unsat_total > 0 else 0.0,
    }


def cnf_to_mip(cnf_file_path, output_path, format="lp", solve=False, verbose=False, track_accuracy=False):
    """
    Convert a CNF file to MIP format (LP or MPS) and optionally solve it.

    Args:
        cnf_file_path: Path to the DIMACS CNF file
        output_path: Path to write the output file
        format: Output format - "lp" or "mps" (default: "lp")
        solve: Whether to solve the problem (default: False)
        verbose: Whether to print detailed output (default: False)
        track_accuracy: Whether to track accuracy against expected result (default: False)
        
    Returns:
        dict or None: If track_accuracy is True and solve is True, returns accuracy result
    """
    if format not in ["lp", "mps"]:
        raise ValueError("Format must be 'lp' or 'mps'")

    # Initialize the MIP Problem (dummy objective for feasibility check)
    prob = pulp.LpProblem("SAT_to_MIP", pulp.LpMinimize)
    prob += 0

    vars_dict = {}
    clauses_count = 0

    # Parse the DIMACS CNF file
    with open(cnf_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue

            parts = line.split()
            if parts[0] == 'p':
                # Line format: p cnf <num_vars> <num_clauses>
                num_vars = int(parts[2])
                # Create binary variables x1, x2, ... xN
                for i in range(1, num_vars + 1):
                    vars_dict[i] = pulp.LpVariable(f"x{i}", cat=pulp.LpBinary)
                continue

            # Parse clause (a sequence of integers ending in 0)
            literals = [int(x) for x in parts if is_number(x) and int(x) != 0]
            if not literals:
                continue

            # Convert Clause to Linear Inequality
            # Rule: Sum(positive_literals) + Sum(1 - negative_literals) >= 1
            constraint_expr = []
            constant_shift = 0

            for lit in literals:
                var_idx = abs(lit)
                if lit > 0:
                    constraint_expr.append(vars_dict[var_idx])
                else:
                    # NOT x_i is represented as (1 - x_i)
                    constraint_expr.append(-vars_dict[var_idx])
                    constant_shift += 1

            # The inequality: sum_expr + constant_shift >= 1
            prob += pulp.lpSum(constraint_expr) >= (1 - constant_shift), f"Clause_{clauses_count}"
            clauses_count += 1

    # Write to specified format
    if format == "lp":
        prob.writeLP(output_path)
    elif format == "mps":
        prob.writeMPS(output_path)

    if verbose:
        print(f"{format.upper()} file saved to {output_path}")

    # Optionally solve the problem
    if solve:
        status = prob.solve(pulp.PULP_CBC_CMD(msg=verbose))
        is_sat = status == pulp.LpStatusOptimal
        
        if verbose:
            print(f"Status: {pulp.LpStatus[status]}")
            if is_sat:
                solution = {name: var.varValue for name, var in vars_dict.items()}
                print("Satisfying Assignment found (showing first 5 variables):")
                for i in list(solution.keys())[:5]:
                    print(f" x{i} = {int(solution[i])}")
            else:
                print("No satisfying assignment exists (UNSAT).")
        
        # Track accuracy if requested
        if track_accuracy:
            filename = os.path.basename(cnf_file_path)
            expected = get_expected_satisfiability(filename)
            return {
                "filename": filename,
                "expected": expected,
                "actual": is_sat,
                "correct": expected == is_sat
            }
    
    return None



def process_directory(cnf_directory, output_directory, format="lp", solve=False, verbose=False, track_accuracy=False):
    """
    Process all CNF files in a directory.

    Args:
        cnf_directory: Directory containing .cnf files
        output_directory: Directory to write output files
        format: Output format - "lp" or "mps" (default: "lp")
        solve: Whether to solve each problem (default: False)
        verbose: Whether to print detailed output (default: False)
        track_accuracy: Whether to track accuracy against expected results (default: False)
        
    Returns:
        dict or None: If track_accuracy is True and solve is True, returns accuracy metrics
    """
    os.makedirs(output_directory, exist_ok=True)

    if not os.path.isdir(cnf_directory):
        print(f"Error: Input directory '{cnf_directory}' does not exist.")
        return

    cnf_files = sorted([f for f in os.listdir(cnf_directory) if f.endswith(".cnf")])

    if not cnf_files:
        print(f"No .cnf files found in {cnf_directory}")
        return

    extension = f".{format}"
    results = [] if track_accuracy and solve else None
    
    for filename in cnf_files:
        cnf_path = os.path.join(cnf_directory, filename)
        output_path = os.path.join(output_directory, filename.replace(".cnf", extension))

        # Skip if output file already exists
        if os.path.exists(output_path):
            if verbose:
                print(f"Skipping {filename} (already exists at {output_path})", flush=True)
            continue

        if verbose:
            print(f"Processing {filename}...")

        try:
            result = cnf_to_mip(cnf_path, output_path, format=format, solve=solve, 
                               verbose=verbose, track_accuracy=track_accuracy)
            if results is not None and result is not None:
                results.append(result)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Compute and return accuracy if tracking
    if results is not None:
        accuracy_metrics = compute_accuracy(results)
        return accuracy_metrics
    
    return None


def get_sinusoidal_pe(index, d_model=10):
    """Calculates 10-dim Positional Encoding as used in G4SATBench."""
    pe = np.zeros(d_model)
    for i in range(0, d_model, 2):
        div_term = np.exp(i * -(np.log(10000.0) / d_model))
        pe[i] = np.sin(index * div_term)
        pe[i + 1] = np.cos(index * div_term)
    return pe.tolist()

def extract_sat_features(cnf_path):
    """Pass 1: Extract all literal and clause features using PySAT."""
    cnf = CNF(from_file=cnf_path)
    num_vars = cnf.nv
    num_clauses = len(cnf.clauses)
    
    lit_counts = Counter()
    lit_horn_counts = Counter()
    
    # Pre-calculate counts for degrees and Horn status
    clause_data = []  # Temporary storage for calculating means
    for c_idx, clause in enumerate(cnf.clauses):
        pos_lits = [l for l in clause if l > 0]
        neg_lits = [l for l in clause if l < 0]
        
        lit_counts.update(clause)
        
        clause_data.append({
            "pos_count": len(pos_lits),
            "neg_count": len(neg_lits),
            "width": len(clause),
            "is_horn": 1 if len(pos_lits) <= 1 else 0
        })
    
    # Calculate means for normalization
    num_clauses_total = len(clause_data)
    mean_pos_count = sum(c["pos_count"] for c in clause_data) / num_clauses_total if num_clauses_total > 0 else 0
    mean_neg_count = sum(c["neg_count"] for c in clause_data) / num_clauses_total if num_clauses_total > 0 else 0
    
    # First pass through clause_data to calculate pos_neg_ratios and mean
    pos_neg_ratios = []
    temp_clause_stats = []
    for c_data in clause_data:
        pos_count = c_data["pos_count"]
        neg_count = c_data["neg_count"]
        pos_neg_ratio = pos_count / (neg_count + 1)
        pos_neg_ratios.append(pos_neg_ratio)
        temp_clause_stats.append((pos_count, neg_count, pos_neg_ratio, c_data["width"], c_data["is_horn"]))
    
    mean_pos_neg_ratio = sum(pos_neg_ratios) / len(pos_neg_ratios) if pos_neg_ratios else 0
    
    # Build final clause_stats with normalized features
    clause_stats = []
    for c_idx, (pos_count, neg_count, pos_neg_ratio, width, is_horn) in enumerate(temp_clause_stats):
        # Normalize by means (avoid division by zero)
        pos_count_normalized = pos_count / mean_pos_count if mean_pos_count > 0 else 0
        neg_count_normalized = neg_count / mean_neg_count if mean_neg_count > 0 else 0
        pos_neg_ratio_normalized = pos_neg_ratio / mean_pos_neg_ratio if mean_pos_neg_ratio > 0 else 0
        
        is_binary = 1 if width == 2 else 0
        is_ternary = 1 if width == 3 else 0
        
        clause_stats.append({
            "width": width,
            "pos_count": pos_count,
            "neg_count": neg_count,
            "pos_neg_ratio": pos_neg_ratio,
            "pos_count_normalized": pos_count_normalized,
            "neg_count_normalized": neg_count_normalized,
            "pos_neg_ratio_normalized": pos_neg_ratio_normalized,
            "is_horn": is_horn,
            "is_binary": is_binary,
            "is_ternary": is_ternary
        })

    # First pass: Calculate base degrees for all variables
    base_features = {}
    for i in range(1, num_vars + 1):
        pos_deg = lit_counts[i] / num_clauses
        neg_deg = lit_counts[-i] / num_clauses
        pos_neg_ratio = lit_counts[i] / (lit_counts[-i] + 1)
        
        base_features[i] = {
            "pos_deg": pos_deg,
            "neg_deg": neg_deg,
            "pos_neg_ratio": pos_neg_ratio
        }
    
    # Calculate means for normalization
    mean_pos_deg = sum(f["pos_deg"] for f in base_features.values()) / num_vars if num_vars > 0 else 0
    mean_neg_deg = sum(f["neg_deg"] for f in base_features.values()) / num_vars if num_vars > 0 else 0
    mean_pos_neg_ratio = sum(f["pos_neg_ratio"] for f in base_features.values()) / num_vars if num_vars > 0 else 0
    
    # Prepare Variable Features (7-dim)
    # Features: degree, pos_degree, neg_degree, pos_neg_ratio, 
    #          pos_degree/mean_pos_degree, neg_degree/mean_neg_degree, 
    #          pos_neg_ratio/mean_pos_neg_ratio
    var_features = {}
    for i in range(1, num_vars + 1):
        pos_deg = base_features[i]["pos_deg"]
        neg_deg = base_features[i]["neg_deg"]
        pos_neg_ratio = base_features[i]["pos_neg_ratio"]
        degree = pos_deg + neg_deg
        
        # Normalize by means (avoid division by zero)
        pos_deg_normalized = pos_deg / mean_pos_deg if mean_pos_deg > 0 else 0
        neg_deg_normalized = neg_deg / mean_neg_deg if mean_neg_deg > 0 else 0
        pos_neg_ratio_normalized = pos_neg_ratio / mean_pos_neg_ratio if mean_pos_neg_ratio > 0 else 0
        
        var_features[i] = [
            degree,
            pos_deg,
            neg_deg,
            pos_neg_ratio,
            pos_deg_normalized,
            neg_deg_normalized,
            pos_neg_ratio_normalized
        ]

    return var_features, clause_stats, cnf

def cnf_to_mip_with_sat_features(cnf_file_path, output_lp_path, output_features_path=None, verbose=False):
    """
    Pass 2: Convert to MIP and attach features.
    
    Args:
        cnf_file_path: Path to CNF file
        output_lp_path: Path to output LP file
        output_features_path: Optional path to save features as NPZ (base name without extension)
        verbose: Whether to print detailed output
        
    Returns:
        tuple: (var_feats dict, mip_constraint_vectors list)
    """
    var_feats, clause_feats, cnf = extract_sat_features(cnf_file_path)
    
    prob = pulp.LpProblem("SAT_to_MIP_FORGE", pulp.LpMinimize)
    prob += 0 # Feasibility objective
    
    vars_dict = {i: pulp.LpVariable(f"x{i}", cat=pulp.LpBinary) for i in range(1, cnf.nv + 1)}
    
    # Store calculated constraint features for the GNN
    mip_constraint_vectors = []

    for idx, clause in enumerate(cnf.clauses):
        stats = clause_feats[idx]
        
        # Algebraic Logic: Sum(pos) + Sum(1 - neg) >= 1  =>  Sum(pos) - Sum(neg) >= 1 - count(neg)
        constraint_expr = []
        for lit in clause:
            if lit > 0:
                constraint_expr.append(vars_dict[lit])
            else:
                constraint_expr.append(-vars_dict[abs(lit)])
        
        rhs = 1 - stats['neg_count']
        prob += pulp.lpSum(constraint_expr) >= rhs, f"C_{idx}"
        
        # Generate the FORGE-style SAT Constraint Feature Vector (10-dim)
        c_vec = [
            stats['width'],                           # width
            stats['pos_count'],                       # pos_degree
            stats['neg_count'],                       # neg_degree
            stats['pos_neg_ratio'],                   # pos_neg_ratio
            stats['pos_count_normalized'],            # pos_degree/mean_pos_degree
            stats['neg_count_normalized'],            # neg_degree/mean_neg_degree
            stats['pos_neg_ratio_normalized'],        # pos_neg_ratio/mean_pos_neg_ratio
            stats['is_horn'],                         # is_horn
            stats['is_binary'],                       # is_binary
            stats['is_ternary'],                      # is_ternary
        ]
        
        mip_constraint_vectors.append(c_vec)

    # Save MIP and Metadata
    prob.writeLP(output_lp_path)
    if verbose:
        print(f"MIP saved to {output_lp_path}. Feature vectors ready for FORGE encoder.")
    
    # Optionally save features as NPZ for efficient loading by mip_to_embeddings
    if output_features_path:
        save_sat_features(var_feats, mip_constraint_vectors, cnf, output_features_path, verbose)
    
    return var_feats, mip_constraint_vectors


def save_sat_features(var_feats, constraint_vecs, cnf, output_base_path, verbose=False):
    """
    Save SAT/MIP constraint and variable features for use in mip_to_embeddings.
    
    This function saves extracted SAT features in three complementary formats:
    
    1. **NPZ (NumPy Compressed)** - Most efficient for FORGE pipeline
       - Loads directly with np.load()
       - Compressed storage, fast I/O
       - Best for large-scale batch processing
       
    2. **Pickle (Python)** - Full feature metadata included
       - Complete feature metadata and dimension info
       - Easy to integrate with FORGE's mip_to_embeddings
       - Preserves all feature descriptions
       
    3. **JSON (Metadata)** - For inspection and documentation
       - Human-readable metadata
       - Useful for debugging and verification
       - Contains feature names and dimensions
    
    Feature Dimensions:
        - Variable features: 7-dim (degree, pos_degree, neg_degree, pos_neg_ratio, 
                                   pos_degree/mean_pos_degree, neg_degree/mean_neg_degree,
                                   pos_neg_ratio/mean_pos_neg_ratio)
        - Constraint features: 10-dim (width, pos_degree, neg_degree, pos_neg_ratio,
                                      pos_degree_normalized, neg_degree_normalized,
                                      pos_neg_ratio_normalized, is_horn, is_binary, is_ternary)
    
    Usage with mip_to_embeddings:
        Load features: feats = load_sat_features('path/to/features', format='npz')
        var_feats = feats['variable_features']      # shape: (num_vars, 7)
        con_feats = feats['constraint_features']    # shape: (num_constraints, 9)
    
    Args:
        var_feats: Dict[int, list] of variable features (7-dim each)
        constraint_vecs: List of constraint feature vectors (9-dim each)
        cnf: CNF object with metadata
        output_base_path: Base path without extension (will add .npz, .pkl, .json as needed)
        verbose: Whether to print detailed output
    """
    os.makedirs(os.path.dirname(output_base_path) or ".", exist_ok=True)
    
    # Convert variable features dict to numpy array (num_vars x 7)
    num_vars = len(var_feats)
    var_features_array = np.array([var_feats[i] for i in range(1, num_vars + 1)], dtype=np.float32)
    
    # Constraint features as numpy array (num_constraints x 17)
    constraint_features_array = np.array(constraint_vecs, dtype=np.float32)
    
    # Save as NPZ (most efficient for mip_to_embeddings integration)
    npz_path = output_base_path + ".npz"
    np.savez_compressed(npz_path,
                        constraint_features=constraint_features_array,
                        variable_features=var_features_array,
                        num_vars=np.array([num_vars]),
                        num_constraints=np.array([len(constraint_vecs)]))
    if verbose:
        print(f"  Features saved (NPZ): {npz_path}")
    
    # Save as pickle for integration with FORGE pipeline (Dict format)
    pkl_path = output_base_path + ".pkl"
    features_dict = {
        "variable_features": var_features_array,      # (num_vars, 7)
        "constraint_features": constraint_features_array,  # (num_constraints, 19)
        "num_vars": num_vars,
        "num_constraints": len(constraint_vecs),
        "feature_metadata": {
            "var_feature_dim": 7,
            "constraint_feature_dim": 9,
            "constraint_feature_names": [
                "width", "pos_degree", "neg_degree", "pos_neg_ratio",
                "pos_degree_normalized", "neg_degree_normalized", "pos_neg_ratio_normalized",
                "is_horn", "is_binary", "is_ternary"
            ],
            "variable_feature_names": [
                "degree", "pos_degree", "neg_degree", "pos_neg_ratio",
                "pos_degree_normalized", "neg_degree_normalized", "pos_neg_ratio_normalized"
            ]
        }
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(features_dict, f)
    if verbose:
        print(f"  Features saved (Pickle): {pkl_path}")
    
    # Save metadata as JSON for inspection
    json_path = output_base_path + ".json"
    metadata = {
        "num_variables": int(num_vars),
        "num_constraints": int(len(constraint_vecs)),
        "variable_feature_dimension": 7,
        "constraint_feature_dimension": 10,
        "variable_feature_names": features_dict["feature_metadata"]["variable_feature_names"],
        "constraint_feature_names": features_dict["feature_metadata"]["constraint_feature_names"],
        "description": "SAT to MIP features extracted using PySAT and FORGE-compatible encodings"
    }
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    if verbose:
        print(f"  Metadata saved (JSON): {json_path}")


def load_sat_features(output_base_path, format="npz"):
    """
    Load SAT features saved by save_sat_features.
    
    Args:
        output_base_path: Base path without extension
        format: "npz" (default) or "pkl"
        
    Returns:
        dict: Contains constraint_features, variable_features, num_vars, num_constraints
    """
    if format == "npz":
        npz_file = output_base_path + ".npz"
        data = np.load(npz_file)
        return {
            "constraint_features": data["constraint_features"],
            "variable_features": data["variable_features"],
            "num_vars": int(data["num_vars"][0]),
            "num_constraints": int(data["num_constraints"][0])
        }
    elif format == "pkl":
        pkl_file = output_base_path + ".pkl"
        with open(pkl_file, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'npz' or 'pkl'")


def process_directory_with_forge_features(cnf_directory, output_directory, verbose=False):
    """
    Process all CNF files in a directory using FORGE features extraction.
    
    Args:
        cnf_directory: Directory containing .cnf files
        output_directory: Directory to write output files
        verbose: Whether to print detailed output (default: False)
        
    Returns:
        dict: Summary of processed files and feature extraction results
    """
    os.makedirs(output_directory, exist_ok=True)
    
    if not os.path.isdir(cnf_directory):
        print(f"Error: Input directory '{cnf_directory}' does not exist.")
        return None
    
    cnf_files = sorted([f for f in os.listdir(cnf_directory) if f.endswith(".cnf")])
    
    if not cnf_files:
        print(f"No .cnf files found in {cnf_directory}")
        return None
    
    processed_files = []
    failed_files = []
    
    for filename in cnf_files:
        cnf_path = os.path.join(cnf_directory, filename)
        output_lp_path = os.path.join(output_directory, filename.replace(".cnf", ".lp"))
        
        if verbose:
            print(f"Processing {filename} with FORGE features...")
        
        try:
            # Create features path by removing file extension
            features_base_path = os.path.join(output_directory, filename.replace(".cnf", ""))
            
            var_feats, constraint_feats = cnf_to_mip_with_sat_features(
                cnf_path, output_lp_path, 
                output_features_path=features_base_path,
                verbose=verbose
            )
            processed_files.append({
                "filename": filename,
                "output": output_lp_path,
                "num_variables": len(var_feats),
                "num_constraints": len(constraint_feats),
                "var_feature_dim": len(next(iter(var_feats.values()))) if var_feats else 0,
                "constraint_feature_dim": len(constraint_feats[0]) if constraint_feats else 0
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            failed_files.append({"filename": filename, "error": str(e)})
    
    return {
        "total": len(cnf_files),
        "processed": len(processed_files),
        "failed": len(failed_files),
        "files": processed_files,
        "errors": failed_files
    }




def load_config_file(config_path, input_directory=None):
    """
    Load a list of file names from a config file and optionally prepend input directory.
    
    Args:
        config_path: Path to the config file (one file name per line)
        input_directory: Optional directory to prepend to file names to create full paths
        
    Returns:
        list: List of file paths (full paths if input_directory provided, otherwise file names)
    """
    if not os.path.isfile(config_path):
        print(f"Error: Config file '{config_path}' does not exist.")
        return []
    
    files = []
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments (lines starting with #)
            if line and not line.startswith('#'):
                if input_directory:
                    files.append(os.path.join(input_directory, line))
                else:
                    files.append(line)
    
    return files


def process_files_from_config(file_list, output_directory, format="lp", solve=False, 
                             verbose=False, track_accuracy=False, forge=False, 
                             save_features=False):
    """
    Process a list of CNF files from a config file.
    
    Args:
        file_list: List of CNF file paths
        output_directory: Directory to write output files
        format: Output format - "lp" or "mps" (default: "lp")
        solve: Whether to solve the problems (default: False)
        verbose: Whether to print detailed output (default: False)
        track_accuracy: Whether to track accuracy (default: False)
        forge: Whether to use FORGE features (default: False)
        save_features: Whether to save features (default: False)
        
    Returns:
        dict: Summary of processing results
    """
    os.makedirs(output_directory, exist_ok=True)
    
    processed_files = []
    failed_files = []
    accuracy_results = []
    
    for file_path in file_list:
        if not os.path.isfile(file_path):
            print(f"Warning: File '{file_path}' not found, skipping...")
            failed_files.append({"filename": file_path, "error": "File not found"})
            continue
        
        if not file_path.endswith(".cnf"):
            print(f"Warning: File '{file_path}' is not a .cnf file, skipping...")
            failed_files.append({"filename": file_path, "error": "Not a .cnf file"})
            continue
        
        filename = os.path.basename(file_path)
        output_lp_path = os.path.join(output_directory, filename.replace(".cnf", ".lp"))
        
        if verbose:
            print(f"Processing {filename}...")
        
        try:
            if forge:
                # FORGE features path
                features_path = None
                if save_features:
                    features_path = os.path.join(output_directory, filename.replace(".cnf", ""))
                
                var_feats, constraint_feats = cnf_to_mip_with_sat_features(
                    file_path, output_lp_path, 
                    output_features_path=features_path,
                    verbose=verbose
                )
                processed_files.append({
                    "filename": filename,
                    "path": file_path,
                    "output": output_lp_path,
                    "num_variables": len(var_feats),
                    "num_constraints": len(constraint_feats),
                    "var_feature_dim": len(next(iter(var_feats.values()))) if var_feats else 0,
                    "constraint_feature_dim": len(constraint_feats[0]) if constraint_feats else 0
                })
            else:
                # Standard path
                result = cnf_to_mip(file_path, output_lp_path, format=format, solve=solve, 
                                   verbose=verbose, track_accuracy=track_accuracy)
                processed_files.append({
                    "filename": filename,
                    "path": file_path,
                    "output": output_lp_path
                })
                if result and track_accuracy:
                    accuracy_results.append(result)
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            failed_files.append({"filename": filename, "error": str(e)})
    
    summary = {
        "total": len(file_list),
        "processed": len(processed_files),
        "failed": len(failed_files),
        "files": processed_files,
        "errors": failed_files
    }
    
    if accuracy_results:
        summary["accuracy"] = compute_accuracy(accuracy_results)
    
    return summary


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Convert SAT problems (CNF format) to Mixed Integer Programming (MIP)."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a single .cnf file or a directory containing .cnf files"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output path for file (required if input is a single file)"
    )

    parser.add_argument(
        "-d", "--output-dir",
        type=str,
        default="../data/sat_instances/",
        help="Output directory for files when processing a directory (default: ../data/sat_instances/)"
    )

    parser.add_argument(
        "-f", "--format",
        type=str,
        choices=["lp", "mps"],
        default="lp",
        help="Output format: 'lp' or 'mps' (default: lp)"
    )

    parser.add_argument(
        "-s", "--solve",
        action="store_true",
        help="Solve the MIP after conversion (default: False)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed output (default: False)"
    )

    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="Track accuracy when solving (files starting with 'uu' are expected UNSAT) (default: False)"
    )

    parser.add_argument(
        "--forge",
        action="store_true",
        help="Use FORGE features extraction path (extract SAT features and compute feature vectors) (default: False)"
    )

    parser.add_argument(
        "--save-features",
        action="store_true",
        help="Save extracted features as NPZ/PKL/JSON files for use in mip_to_embeddings (default: False)"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to a config file containing a list of CNF files to process (one file path per line)"
    )

    args = parser.parse_args()

    # Handle config file if provided
    if args.config:
        if not args.input:
            print("Error: --input directory is required when using --config")
            return
        
        file_list = load_config_file(args.config, input_directory=args.input)
        if not file_list:
            print("Error: No valid files found in config file or config file is empty.")
            return
        
        if args.verbose:
            print(f"Loaded {len(file_list)} files from config: {args.config}")
            print(f"Using input directory: {args.input}")
        
        summary = process_files_from_config(
            file_list, args.output_dir, format=args.format, solve=args.solve,
            verbose=args.verbose, track_accuracy=args.accuracy, forge=args.forge,
            save_features=args.save_features
        )
        
        print(f"\nConfig File Processing Summary:")
        print(f"  Total Files: {summary['total']}")
        print(f"  Processed: {summary['processed']}")
        print(f"  Failed: {summary['failed']}")
        
        if summary['files']:
            print(f"\n  Processed Files:")
            for file_info in summary['files'][:5]:  # Show first 5
                print(f"    - {file_info['filename']}")
            if len(summary['files']) > 5:
                print(f"    ... and {len(summary['files']) - 5} more files")
        
        if summary['errors']:
            print(f"\n  Failed Files:")
            for error_info in summary['errors'][:5]:  # Show first 5 errors
                print(f"    - {error_info['filename']}: {error_info['error']}")
            if len(summary['errors']) > 5:
                print(f"    ... and {len(summary['errors']) - 5} more errors")
        
        if 'accuracy' in summary:
            print(f"\n  Accuracy Metrics:")
            print(f"    Overall: {summary['accuracy']['accuracy']*100:.2f}%")
            print(f"    SAT: {summary['accuracy']['sat_accuracy']*100:.2f}%")
            print(f"    UNSAT: {summary['accuracy']['unsat_accuracy']*100:.2f}%")
        
        return

    if args.forge:
        # FORGE features path
        if os.path.isfile(args.input):
            # Single file with FORGE features
            if not args.output:
                print("Error: --output is required when processing a single file")
                return
            
            if args.verbose:
                print(f"Converting {args.input} with FORGE features...")
            
            # Prepare features output path if saving
            features_path = None
            if args.save_features:
                features_path = args.output.replace(".lp", "").replace(".mps", "")
            
            var_feats, constraint_feats = cnf_to_mip_with_sat_features(
                args.input, args.output, 
                output_features_path=features_path, 
                verbose=args.verbose
            )
            print(f"\nSAT Features Extraction Summary:")
            print(f"  Variables: {len(var_feats)}")
            print(f"  Constraints: {len(constraint_feats)}")
            print(f"  Variable Feature Dimension: {len(next(iter(var_feats.values()))) if var_feats else 0}")
            print(f"  Constraint Feature Dimension: {len(constraint_feats[0]) if constraint_feats else 0}")
            if args.save_features:
                print(f"  Features saved to: {features_path}.{{npz,pkl,json}}")
        
        elif os.path.isdir(args.input):
            # Directory with SAT features
            if args.verbose:
                print(f"Processing all .cnf files in {args.input} with FORGE features...")
            
            summary = process_directory_with_forge_features(args.input, args.output_dir, verbose=args.verbose)
            if summary:
                print(f"\nFORGE Features Processing Summary:")
                print(f"  Total Files: {summary['total']}")
                print(f"  Processed: {summary['processed']}")
                print(f"  Failed: {summary['failed']}")
                if summary['files']:
                    print(f"\n  Processed Files:")
                    for file_info in summary['files'][:5]:  # Show first 5
                        print(f"    - {file_info['filename']}")
                        print(f"      Variables: {file_info['num_variables']}, Constraints: {file_info['num_constraints']}")
                    if len(summary['files']) > 5:
                        print(f"    ... and {len(summary['files']) - 5} more files")
        
        else:
            print(f"Error: '{args.input}' is not a valid file or directory")
    
    else:
        # Standard path (without SAT features)
        if os.path.isfile(args.input):
            # Single file
            if not args.output:
                print("Error: --output is required when processing a single file")
                return

            if args.verbose:
                print(f"Converting {args.input} to {args.format.upper()}...")

            result = cnf_to_mip(args.input, args.output, format=args.format, solve=args.solve, 
                               verbose=args.verbose, track_accuracy=args.accuracy)
            if result:
                print(f"\nAccuracy Result for {result['filename']}:")
                print(f"  Expected: {'SAT' if result['expected'] else 'UNSAT'}")
                print(f"  Actual:   {'SAT' if result['actual'] else 'UNSAT'}")
                print(f"  Correct:  {result['correct']}")

        elif os.path.isdir(args.input):
            # Directory
            if args.verbose:
                print(f"Processing all .cnf files in {args.input} to {args.format.upper()}...")

            metrics = process_directory(args.input, args.output_dir, format=args.format, 
                                       solve=args.solve, verbose=args.verbose, 
                                       track_accuracy=args.accuracy)
            if metrics:
                print(f"\nAccuracy Metrics:")
                print(f"  Total Problems: {metrics['total']}")
                print(f"  Correct: {metrics['correct']}/{metrics['total']}")
                print(f"  Overall Accuracy: {metrics['accuracy']*100:.2f}%")
                print(f"  SAT Accuracy: {metrics['sat_correct']}/{metrics['sat_total']} ({metrics['sat_accuracy']*100:.2f}%)")
                print(f"  UNSAT Accuracy: {metrics['unsat_correct']}/{metrics['unsat_total']} ({metrics['unsat_accuracy']*100:.2f}%)")

        else:
            print(f"Error: '{args.input}' is not a valid file or directory")


if __name__ == "__main__":
    main()