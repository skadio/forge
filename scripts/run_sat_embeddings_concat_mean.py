#!/usr/bin/env python3
"""
Generate embeddings from SAT features by concatenating all constraint and 
variable features and taking the mean as the instance embedding.

This approach:
1. Extracts all constraint features (up to 4 dims per constraint)
2. Extracts all variable features (up to 6 dims per variable)
3. Concatenates all features from all nodes
4. Takes the mean across all concatenated features as the instance embedding
"""

import os
import sys
import gc
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import pacmap
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

# Disable threading
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import re


def generate_sat_feature_embeddings_concat_mean(config="g4satbench_test", output_file=None, max_instances=None, feature_mode="padded"):
    """
    Generate embeddings by concatenating all constraint and variable features,
    then taking the mean as the instance embedding.
    
    feature_mode: "padded" (default) - use 4,6 padded configuration
                  "full" - use all feature dimensions without padding
    """
    
    features_dir = "../data/sat_instances/"
    config_file = f"../data/configs/{config}.txt"
    
    # Auto-generate output filename from config if not provided
    if output_file is None:
        output_file = f"{config}_embeddings_concat_mean.pkl"
    
    # Read instance names
    with open(config_file, 'r') as f:
        instance_names = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(instance_names)} instances...")
    print(f"Feature mode: {feature_mode}\n")
    
    if max_instances:
        instance_names = instance_names[:max_instances]
        print(f"Limited to first {max_instances} instances\n")
    
    embeddings_dict = {}
    success_count = 0
    embedding_dims = []
    
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
            # if num_cons > 150000 or num_vars > 100000:
            #     print(f"✗ too large ({num_cons}c, {num_vars}v)")
            #     continue
            
            constraint_features = data['constraint_features']  # shape: (num_cons, 4)
            variable_features = data['variable_features']      # shape: (num_vars, 6)
            
            # Get feature dimensions
            c_dims = constraint_features.shape[1]  # typically 4
            v_dims = variable_features.shape[1]    # typically 6
            total_dims = c_dims + v_dims           # typically 10
            
            if feature_mode == "padded":
                # Stack all features: shape (num_cons + num_vars, 10)
                # Pad constraint features (4) to 10, variable features (6) to 10
                all_features_array = np.zeros((num_cons + num_vars, total_dims))
                all_features_array[:num_cons, :c_dims] = constraint_features[:, :c_dims]
                all_features_array[num_cons:, :v_dims] = variable_features[:, :v_dims]
                
                # Take mean across all nodes as the instance embedding
                embedding = all_features_array.mean(axis=0)
            
            elif feature_mode == "full":
                # Same as padded - stack all features with padding to total_dims
                all_features_array = np.zeros((num_cons + num_vars, total_dims))
                all_features_array[:num_cons, :c_dims] = constraint_features[:, :c_dims]
                all_features_array[num_cons:, :v_dims] = variable_features[:, :v_dims]
                
                # Take mean across all nodes as the instance embedding
                embedding = all_features_array.mean(axis=0)
            
            else:
                raise ValueError(f"Unknown feature_mode: {feature_mode}")
            
            embedding_dims.append(len(embedding))
            embeddings_dict[instance_name] = embedding
            success_count += 1
            print(f"✓ (shape={embedding.shape})")
            
            # Cleanup
            del data, constraint_features, variable_features, all_features_array
            
        except Exception as e:
            print(f"✗ {str(e)[:30]}")
            continue
    
    print(f"\n✓ Generated {success_count}/{len(instance_names)} embeddings")
    print(f"  Embedding dimensions: {set(embedding_dims)}")
    
    # Save
    if len(embeddings_dict) > 0:
        print(f"Saving to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(embeddings_dict, f)
        print(f"✓ Saved\n")
    
    return embeddings_dict


def visualize_embeddings(embeddings_dict, output_file=None, config=None):
    """Visualize embeddings using PaCMAP."""
    
    # Create visualizations directory if it doesn't exist
    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Auto-generate output filename from config if not provided
    if output_file is None:
        if config:
            output_file = f"{config}_concat_mean_visualization"
        else:
            output_file = "embeddings_concat_mean_visualization"
        output_file = os.path.join(viz_dir, output_file)
    
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
    # L2 normalize
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
    
    # Save as both PNG and PDF
    png_file = output_file + ".png"
    pdf_file = output_file + ".pdf"
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"✓ Saved to {png_file}")
    print(f"✓ Saved to {pdf_file}\n")
    plt.close()


def cluster_embeddings(embeddings_dict, n_clusters=None, output_file=None):
    """Cluster embeddings and evaluate against ground truth categories."""
    
    # Create visualizations directory if it doesn't exist
    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    print("Clustering embeddings...")
    
    # Prepare embeddings
    embed_list = []
    instance_names = []
    for key in sorted(embeddings_dict.keys()):
        emb = embeddings_dict[key]
        if isinstance(emb, np.ndarray):
            embed_list.append(emb)
        else:
            embed_list.append(emb.instance_embedding)
        instance_names.append(key)
    
    embed_mat = np.array(embed_list)
    # L2 normalize
    embed_mat = embed_mat / (np.linalg.norm(embed_mat, axis=1, keepdims=True) + 1e-10)
    
    # Extract ground truth categories
    categories = []
    for instance_name in instance_names:
        filename = instance_name.split('/')[-1]
        if 'G4SATBENCH_' in filename:
            parts = filename.replace('.cnf', '').split('_')
            if len(parts) >= 5:
                categories.append(parts[2])
            else:
                categories.append(filename)
        elif 'SATLIB_' in filename:
            parts = filename.replace('.mps', '').split('-')
            category = parts[0].replace('SATLIB_', '')
            categories.append(category)
        else:
            try:
                categories.append(filename.split('_')[1])
            except:
                categories.append(filename)
    
    unique_categories = sorted(set(categories))
    category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
    ground_truth = np.array([category_to_idx[cat] for cat in categories])
    
    # Determine number of clusters
    if n_clusters is None:
        n_clusters = len(unique_categories)
    
    print(f"  {len(embed_mat)} instances, {n_clusters} clusters, {embed_mat.shape[1]} dimensions")
    
    # Cluster with single run
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=20)
    cluster_labels = kmeans.fit_predict(embed_mat)
    
    # Evaluate
    nmi = normalized_mutual_info_score(ground_truth, cluster_labels)
    cm = contingency_matrix(ground_truth, cluster_labels)
    purity = np.sum(np.max(cm, axis=1)) / len(ground_truth)
    
    print(f"Results:")
    print(f"  NMI: {nmi:.4f}")
    print(f"  Purity: {purity:.4f}")
    print()
    
    # Prepare PaCMAP projection for visualization
    pca = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0).fit_transform(embed_mat, init='pca')
    pca = (pca - np.min(pca)) / np.ptp(pca)
    
    # Plot clusters with results
    print("Creating cluster visualization...")
    cmap = plt.get_cmap('tab20')
    plt.figure(figsize=(10, 8), dpi=300)
    
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        plt.scatter(
            pca[mask, 0],
            pca[mask, 1],
            s=20,
            label=f"Cluster {cluster_id}",
            color=cmap(cluster_id % 20),
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
    
    # Add results to title
    title_text = f"Clustering Results\nClusters: {n_clusters}, NMI: {nmi:.4f}, Purity: {purity:.4f}"
    plt.title(title_text, fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize='small')
    plt.tight_layout()
    
    # Generate output filename if not provided
    if output_file is None:
        output_file = os.path.join(viz_dir, "clustering_results")
    else:
        output_file = os.path.join(viz_dir, output_file)
    
    # Save as both PNG and PDF
    png_file = output_file + ".png"
    pdf_file = output_file + ".pdf"
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"✓ Saved to {png_file}")
    print(f"✓ Saved to {pdf_file}\n")
    plt.close()
    
    return nmi, purity, n_clusters


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SAT embeddings using concatenation and mean pooling")
    parser.add_argument("--config", type=str, default="g4satbench_test", help="Config name")
    parser.add_argument("--output", type=str, default=None, help="Output pickle file")
    parser.add_argument("--max-instances", type=int, default=None, help="Max instances to process")
    parser.add_argument("--feature-mode", type=str, default="padded", choices=["padded", "full"], help="Feature mode: 'padded' (4,6 config) or 'full' (all features)")
    parser.add_argument("--visualize", action="store_true", help="Create visualization")
    parser.add_argument("--cluster", action="store_true", help="Run clustering evaluation")
    parser.add_argument("--n-clusters", type=int, default=None, help="Number of clusters (default: num categories)")
    
    args = parser.parse_args()
    
    # Generate embeddings
    embeddings = generate_sat_feature_embeddings_concat_mean(
        config=args.config,
        output_file=args.output,
        max_instances=args.max_instances,
        feature_mode=args.feature_mode
    )
    
    # Visualize if requested
    if args.visualize:
        visualize_embeddings(embeddings, config=args.config)
    
    # Cluster if requested
    if args.cluster:
        cluster_embeddings(embeddings, n_clusters=args.n_clusters)
