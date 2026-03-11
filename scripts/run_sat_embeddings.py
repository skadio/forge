#!/usr/bin/env python3
"""
Generate FORGE embeddings from pre-extracted SAT features.
Stable version - simple and reliable.
"""

import os
import sys
import gc
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import pacmap
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

# Disable threading
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from forge.embeddings import Forge

import re

def parse_lp_edges_unlimited(lp_file, num_cons, num_vars):
    """Robust, unlimited edge extraction for large SAT-to-MIP LP files."""
    edge_list = []
    # This regex captures the number following 'x' (e.g., extracts '12' from '-3x12')
    var_regex = re.compile(r'x(\d+)')
    
    try:
        with open(lp_file, 'r') as f:
            in_constraints = False
            current_cons_idx = -1
            
            for line in f:
                line = line.strip()
                if not line or line.startswith('\\'): continue

                # Section detection
                if any(x in line for x in ['Subject To', 'Subject to', 'ST:', 'S.T.']):
                    in_constraints = True
                    continue
                if any(x in line for x in ['Bounds', 'End', 'General', 'Binary']):
                    break
                
                if in_constraints:
                    # Detect new constraint or continuation
                    if ':' in line:
                        current_cons_idx += 1
                        # Process only the part after the colon
                        content = line.split(':', 1)[1]
                    else:
                        content = line

                    # Find every variable mentioned in this line
                    for var_match in var_regex.finditer(content):
                        try:
                            # Map variable 'x1' to index 0
                            var_idx = int(var_match.group(1)) - 1
                            
                            if 0 <= var_idx < num_vars:
                                # Standard bipartite edge: [constraint_node, variable_node]
                                edge_list.append([current_cons_idx, var_idx + num_cons])
                        except ValueError:
                            continue
                            
    except Exception as e:
        print(f"\n[Parser Error] {lp_file}: {e}")
        
    return edge_list


def parse_mps_edges(mps_file, num_cons, num_vars, max_edges=5000):
    """Parse MPS file to extract constraint-variable edges."""
    edges = {}  # constraint_id -> set of variable indices
    constraint_id_map = {}
    constraint_counter = 0
    
    try:
        with open(mps_file, 'r') as f:
            in_rows = False
            in_columns = False
            
            for line in f:
                line = line.rstrip()
                
                if line.startswith('ROWS'):
                    in_rows = True
                    in_columns = False
                    continue
                
                if line.startswith('COLUMNS'):
                    in_rows = False
                    in_columns = True
                    continue
                
                if line.startswith('RHS') or line.startswith('BOUNDS') or line.startswith('ENDATA'):
                    in_columns = False
                    break
                
                # Parse ROWS section to map constraint names to IDs
                if in_rows and line and not line.startswith('ROWS'):
                    parts = line.split()
                    if len(parts) >= 2:
                        row_type = parts[0]
                        row_name = parts[1]
                        if row_type != 'N':  # Skip objective
                            constraint_id_map[row_name] = constraint_counter
                            edges[constraint_counter] = set()
                            constraint_counter += 1
                
                # Parse COLUMNS section to extract edges
                if in_columns and line and not line.startswith('COLUMNS'):
                    parts = line.split()
                    if len(parts) >= 2 and 'MARKER' not in line:
                        var_name = parts[0]
                        
                        # Extract variable index from name (e.g., "x1" -> 0, "x2" -> 1)
                        try:
                            var_idx = int(var_name[1:]) - 1
                            if 0 <= var_idx < num_vars:
                                # Process pairs of (constraint, coefficient)
                                for i in range(1, len(parts), 2):
                                    if i + 1 < len(parts):
                                        constraint_name = parts[i]
                                        if constraint_name in constraint_id_map:
                                            cons_idx = constraint_id_map[constraint_name]
                                            edges[cons_idx].add(var_idx)
                        except (ValueError, IndexError):
                            continue
    
    except Exception as e:
        print(f"Error parsing MPS: {e}")
        return []
    
    # Convert to edge list
    edge_list = []
    for cons_idx, var_set in edges.items():
        for var_idx in var_set:
            edge_list.append([cons_idx, var_idx + num_cons])
            if len(edge_list) >= max_edges:
                return edge_list
    
    return edge_list


def parse_lp_edges(lp_file, num_cons, num_vars, max_edges=5000):
    """Parse LP file to extract constraint-variable edges."""
    edges = {}  # constraint_id -> set of variable indices
    constraint_id_map = {}
    constraint_counter = 0
    in_constraints = False
    
    try:
        with open(lp_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('\\'):
                    continue
                
                # Detect constraint section
                if 'Subject To' in line or 'Constraints' in line:
                    in_constraints = True
                    continue
                
                # Stop at bounds or end
                if line.startswith('Bounds') or line.startswith('Binary') or line.startswith('General') or line.startswith('End'):
                    in_constraints = False
                    break
                
                # Parse constraint lines
                if in_constraints and ':' in line:
                    # Format: C_0: - x1 - x2 >= -1
                    parts = line.split(':')
                    if len(parts) >= 2:
                        constraint_name = parts[0].strip()
                        constraint_expr = parts[1]
                        
                        # Map constraint name to ID if not seen
                        if constraint_name not in constraint_id_map:
                            constraint_id_map[constraint_name] = constraint_counter
                            edges[constraint_counter] = set()
                            constraint_counter += 1
                        
                        cons_idx = constraint_id_map[constraint_name]
                        
                        # Extract variables from the expression (e.g., x1, x2, etc.)
                        # Split by whitespace and look for tokens starting with 'x'
                        tokens = constraint_expr.split()
                        for token in tokens:
                            # Remove operators and signs
                            var_name = token.lstrip('+-')
                            
                            # Check if it's a variable name
                            if var_name.startswith('x'):
                                try:
                                    var_idx = int(var_name[1:]) - 1
                                    if 0 <= var_idx < num_vars:
                                        edges[cons_idx].add(var_idx)
                                except (ValueError, IndexError):
                                    continue
    
    except Exception as e:
        print(f"Error parsing LP: {e}")
        return []
    
    # Convert to edge list
    edge_list = []
    for cons_idx, var_set in edges.items():
        for var_idx in var_set:
            edge_list.append([cons_idx, var_idx + num_cons])
            if len(edge_list) >= max_edges:
                return edge_list
    
    return edge_list


def parse_lp_edges_gurobi(lp_file, num_cons, num_vars):
    """Extract edges using Gurobi by loading and analyzing the LP model."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        print("  [Warning] Gurobi not available, falling back to LP parsing")
        return parse_lp_edges_unlimited(lp_file, num_cons, num_vars)
    
    edge_list = []
    
    try:
        # Create empty environment and model
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        
        model = gp.read(lp_file, env=env)
        
        # Extract constraint-variable relationships from the model
        for c_idx, constraint in enumerate(model.getConstrs()):
            # Get the constraint expression
            expr = model.getRow(constraint)
            
            # Iterate through variables in the constraint
            for i in range(expr.size()):
                var = expr.getVar(i)
                var_idx = var.VarName
                
                # Extract variable index from name (e.g., 'x1' -> 0)
                if var_idx.startswith('x'):
                    try:
                        v_idx = int(var_idx[1:]) - 1
                        if 0 <= v_idx < num_vars:
                            edge_list.append([c_idx, v_idx + num_cons])
                    except (ValueError, IndexError):
                        continue
        
        model.dispose()
        env.dispose()
        
        return edge_list
    
    except Exception as e:
        print(f"  [Gurobi Error] {str(e)[:50]}, falling back to LP parsing")
        return parse_lp_edges_unlimited(lp_file, num_cons, num_vars)


def generate_sat_feature_embeddings(config="g4satbench_test", output_file=None, max_instances=None):
    """Generate embeddings from pre-extracted SAT features - STABLE VERSION."""
    
    features_dir = "../data/sat_instances/"
    config_file = f"../data/configs/{config}.txt"
    
    # Auto-generate output filename from config if not provided
    if output_file is None:
        output_file = f"{config}_embeddings.pkl"
    
    # Load FORGE model
    print("Loading FORGE model...")
    forge = Forge(train_config_yaml="../forge/configs/train_config.yaml")
    
    # Auto-detect best device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"  Device: CUDA")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"  Device: MPS")
    else:
        device = torch.device('cpu')
        print(f"  Device: CPU")
    
    forge = forge.to(device)
    print()
    
    # Read instance names
    with open(config_file, 'r') as f:
        instance_names = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(instance_names)} instances...\n")
    
    if max_instances:
        instance_names = instance_names[:max_instances]
        print(f"Limited to first {max_instances} instances\n")
    
    embeddings_dict = {}
    success_count = 0
    
    for idx, instance_name in enumerate(instance_names):
        print(f"[{idx+1}/{len(instance_names)}] {instance_name}...", end=" ", flush=True)
        
        base_name = instance_name.rsplit('.', 1)[0]
        features_file = os.path.join(features_dir, base_name + ".npz")
        
        if not os.path.exists(features_file):
            print("✗ no features")
            continue
        
        try:
            # Load SAT features
            data = np.load(features_file)
            num_cons = int(data['num_constraints'][0])
            num_vars = int(data['num_vars'][0])
            
            # Skip very large instances
            if num_cons > 150000 or num_vars > 100000:
                print(f"✗ too large ({num_cons}c, {num_vars}v)")
                continue
            
            constraint_features = data['constraint_features']
            variable_features = data['variable_features']
            
            # Pad features to 10 dimensions
            cons_feat_matrix = np.hstack([constraint_features[:, :4], np.zeros((num_cons, 6))])
            var_feat_matrix = np.hstack([np.zeros((num_vars, 4)), variable_features[:, :6]])
            feature_matrix_10d = np.vstack([cons_feat_matrix, var_feat_matrix])
            
            # Parse LP or MPS file to extract actual edges
            edge_list = []
            lp_file = os.path.join("../data/sat_instances", base_name + ".lp")
            #mps_file = os.path.join("../data/initial_sat_instances", base_name + ".mps")
            
            if os.path.exists(lp_file):
                # Try Gurobi first, fallback to LP parsing
                edge_list = parse_lp_edges_gurobi(lp_file, num_cons, num_vars)
            # elif os.path.exists(mps_file):
            #     edge_list = parse_mps_edges(mps_file, num_cons, num_vars, max_edges=5000)
            
            print(f"edges={len(edge_list)} ", end="", flush=True)
            
            # Create tensors
            if len(edge_list) > 0:
                edge_index = torch.LongTensor(edge_list).t().contiguous()
                edge_weight = torch.ones(len(edge_list))
            else:
                edge_index = torch.LongTensor(2, 0)
                edge_weight = torch.FloatTensor([])
            
            feature_tensor = torch.FloatTensor(feature_matrix_10d)
            
            # Move to device
            feature_tensor = feature_tensor.to(device)
            edge_index = edge_index.to(device)
            edge_weight = edge_weight.to(device)
            
            # FORGE forward pass
            forge.eval()
            with torch.no_grad():
                num_edges = edge_index.shape[1]
                chunk_size = 500
                print(f"batch({chunk_size}) ", end="", flush=True)
                all_indices = []
                
                for chunk_start in range(0, max(1, num_edges), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, num_edges)
                    chunk_edge_index = edge_index[:, chunk_start:chunk_end]
                    chunk_edge_weight = edge_weight[chunk_start:chunk_end]
                    
                    h_list, logits, loss, chunk_indices, codebook_ = forge.forward(
                        feature_tensor,
                        num_cons,
                        num_vars,
                        chunk_edge_index,
                        chunk_edge_weight
                    )
                    all_indices.extend(chunk_indices.cpu().numpy())
                    del chunk_indices, h_list, logits, loss, codebook_, chunk_edge_index, chunk_edge_weight
                
                indices = torch.LongTensor(all_indices)
                embedding = np.bincount(indices.numpy(), minlength=forge.codebook_size).astype(float)
                del indices
            
            embeddings_dict[instance_name] = embedding
            success_count += 1
            print(f"✓")
            
            # Cleanup
            del data, constraint_features, variable_features, feature_matrix_10d
            del feature_tensor, edge_index, edge_weight, embedding
            
        except Exception as e:
            print(f"✗ {str(e)[:30]}")
            continue
    
    print(f"\n✓ Generated {success_count}/{len(instance_names)} embeddings")
    
    # Save
    if len(embeddings_dict) > 0:
        print(f"Saving to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(embeddings_dict, f)
        print(f"✓ Saved\n")
    
    return embeddings_dict


def visualize_embeddings(embeddings_dict, output_file=None, config=None):
    """Visualize embeddings."""
    
    # Auto-generate output filename from config if not provided
    if output_file is None:
        if config:
            output_file = f"{config}_visualization.png"
        else:
            output_file = "embeddings_visualization.png"
    
    print("Preparing embeddings...")
    color_vec = []
    for key in embeddings_dict.keys():
        filename = key.split('/')[-1]
        if 'G4SATBENCH_' in filename:
            # Extract category from G4SATBENCH_easy_<category>_test_<sat/unsat>-<number>
            parts = filename.replace('.cnf', '').split('_')
            # Format: G4SATBENCH easy <category> test <sat/unsat>-<number>
            if len(parts) >= 5:
                category = parts[2]
                color_vec.append(category)
            else:
                color_vec.append(filename)
        elif 'SATLIB_' in filename:
            parts = filename.replace('.mps', '').split('-')
            category = parts[0].replace('SATLIB_', '')
            color_vec.append(category)
        else:
            try:
                color_vec.append(filename.split('_')[1])
            except:
                color_vec.append(filename)
    
    embed_list = []
    for key in embeddings_dict.keys():
        emb = embeddings_dict[key]
        if isinstance(emb, np.ndarray):
            embed_list.append(emb)
        else:
            embed_list.append(emb.instance_embedding)
    
    embed_mat = np.array(embed_list)
    # L2 normalize instead of column-sum normalization
    embed_mat = embed_mat / (np.linalg.norm(embed_mat, axis=1, keepdims=True) + 1e-10)
    
    print(f"Embedding statistics:")
    print(f"  Shape: {embed_mat.shape}")
    print(f"  Mean: {embed_mat.mean():.6f}, Std: {embed_mat.std():.6f}")
    print(f"  Min: {embed_mat.min():.6f}, Max: {embed_mat.max():.6f}\n")
    pca = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0).fit_transform(embed_mat, init='pca')
    pca = (pca - np.min(pca)) / np.ptp(pca)
    
    # Build index dictionary
    index_dict = {}
    for idx, c in enumerate(color_vec):
        if c not in index_dict:
            index_dict[c] = []
        index_dict[c].append(idx)
    
    # Plot
    print("Plotting...")
    cmap = plt.get_cmap('tab20')
    plt.figure(figsize=(6, 8), dpi=300)
    
    for idx, c in enumerate(sorted(index_dict)):
        plt.scatter(
            pca[index_dict[c], 0] + np.random.rand(len(index_dict[c]))/1000,
            pca[index_dict[c], 1] + np.random.rand(len(index_dict[c]))/1000,
            s=15,
            label=c,
            color=cmap(idx % 20),
            alpha=0.7
        )
    
    plt.title(f"Embeddings ({len(index_dict)} categories)")
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize='small')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_file}\n")
    plt.close()
    
    return embed_mat, color_vec


def cluster_embeddings(embed_mat, color_vec, n_runs=10):
    """Cluster embeddings with multiple algorithms."""
    
    def purity_score(y_true, y_pred):
        from sklearn.metrics.cluster import contingency_matrix
        con_matrix = contingency_matrix(y_true, y_pred)
        return np.sum(np.amax(con_matrix, axis=0)) / np.sum(con_matrix)
    
    num_clusters = len(set(color_vec))
    print(f"Clustering into {num_clusters} categories...\n")
    
    # K-Means
    print("K-Means clustering...")
    res_nmi_km = []
    res_acc_km = []
    
    for run in range(n_runs):
        km = KMeans(n_clusters=num_clusters, random_state=run, n_init=20).fit(embed_mat)
        nmi = normalized_mutual_info_score(color_vec, km.labels_)
        purity = purity_score(np.array(color_vec), km.labels_)
        res_nmi_km.append(nmi)
        res_acc_km.append(purity)
    
    print(f"  NMI: {np.mean(res_nmi_km):.3f} ± {np.std(res_nmi_km):.3f}")
    print(f"  Purity: {np.mean(res_acc_km):.3f} ± {np.std(res_acc_km):.3f}")
    
    # Spectral Clustering
    print("\nSpectral Clustering...")
    try:
        res_nmi_sc = []
        res_acc_sc = []
        for run in range(3):  # Fewer runs due to computational cost
            sc = SpectralClustering(n_clusters=num_clusters, random_state=run, affinity='nearest_neighbors').fit(embed_mat)
            nmi = normalized_mutual_info_score(color_vec, sc.labels_)
            purity = purity_score(np.array(color_vec), sc.labels_)
            res_nmi_sc.append(nmi)
            res_acc_sc.append(purity)
        
        print(f"  NMI: {np.mean(res_nmi_sc):.3f} ± {np.std(res_nmi_sc):.3f}")
        print(f"  Purity: {np.mean(res_acc_sc):.3f} ± {np.std(res_acc_sc):.3f}\n")
    except Exception as e:
        print(f"  Spectral Clustering failed: {str(e)[:100]}\n")


def main():
    """Main function."""
    
    # Check for pickle file argument
    pkl_file = None
    for arg in sys.argv[1:]:
        if arg.endswith('.pkl') or arg.endswith('.pickle'):
            pkl_file = arg
            break
    
    # If pickle file provided, load and visualize
    if pkl_file:
        if not os.path.exists(pkl_file):
            print(f"Error: {pkl_file} not found")
            sys.exit(1)
        
        print("=" * 70)
        print("LOADING AND VISUALIZING EMBEDDINGS")
        print("=" * 70 + "\n")
        
        print(f"Loading {pkl_file}...")
        with open(pkl_file, 'rb') as f:
            embeddings_dict = pickle.load(f)
        print(f"✓ Loaded {len(embeddings_dict)} embeddings\n")
        
        # Visualize
        print("=" * 70)
        print("VISUALIZATION")
        print("=" * 70)
        base_name = os.path.splitext(os.path.basename(pkl_file))[0]
        output_file = f"{base_name}_visualization.png"
        embed_mat, color_vec = visualize_embeddings(embeddings_dict, output_file)
        
        # Cluster
        print("=" * 70)
        print("CLUSTERING")
        print("=" * 70)
        cluster_embeddings(embed_mat, color_vec)
        
        print("=" * 70)
        print("COMPLETE!")
        print("=" * 70)
        return
    
    # Otherwise generate embeddings
    process_all = '--all' in sys.argv
    
    print("=" * 70)
    print("GENERATING EMBEDDINGS FROM SAT FEATURES")
    print("=" * 70 + "\n")
    
    print("STEP 1: Generate embeddings")
    print("-" * 70)
    if not process_all:
        print("(First 50 instances. Use --all for all)\n")
        embeddings_dict = generate_sat_feature_embeddings(max_instances=50)
    else:
        embeddings_dict = generate_sat_feature_embeddings()
    
    if len(embeddings_dict) == 0:
        print("No embeddings generated")
        return
    
    # Visualize
    print("STEP 2: Visualize")
    print("-" * 70)
    embed_mat, color_vec = visualize_embeddings(embeddings_dict, config="g4satbench_test")
    
    # Cluster
    print("STEP 3: Cluster")
    print("-" * 70)
    cluster_embeddings(embed_mat, color_vec)
    
    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
