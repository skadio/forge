#!/usr/bin/env python3
"""
Visualize clusters from pre-computed embeddings pickle file.

Usage:
    python visualize_clusters.py <pickle_file> [--config <config_file>] [--category-mode <mode>] [--no-cluster]
    
    --config <config_file>: Filter embeddings to only include keys listed in config file
    --category-mode <mode>: How to categorize embeddings (default: default)
        - default: Original category extraction from filenames
        - problem_type: Group by problem/domain type only
        - satisfiability: Group by SAT/UNSAT status
        - difficulty: Group by difficulty level (easy/medium/hard)
        - problem_difficulty: Combine problem_type and difficulty
        - all: Combine all attributes (difficulty_problemtype_satisfiability)
    --no-cluster: Skip clustering analysis and just show visualization
"""

import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pacmap
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

# Add parent directory to path to use local forge module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_embeddings(pickle_file):
    """Load embeddings from pickle file."""
    if not os.path.exists(pickle_file):
        raise FileNotFoundError(f"Pickle file not found: {pickle_file}")
    
    print(f"Loading embeddings from {pickle_file}...")
    with open(pickle_file, 'rb') as f:
        embeddings_dict = pickle.load(f)
    
    print(f"✓ Loaded {len(embeddings_dict)} embeddings\n")
    return embeddings_dict


def load_config(config_file):
    """Load list of keys from config file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    keys = []
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                keys.append(line)
    
    print(f"✓ Loaded {len(keys)} keys from config\n")
    return keys


def filter_embeddings_by_config(embeddings_dict, config_keys):
    """Filter embeddings to only include keys from config."""
    filtered_dict = {}
    matched_count = 0
    
    for key in config_keys:
        if key in embeddings_dict:
            filtered_dict[key] = embeddings_dict[key]
            matched_count += 1
        else:
            # Try to find partial matches (for keys that might have path prefixes)
            # Check if key matches end of any embedding key
            for emb_key in embeddings_dict.keys():
                # Get the filename part without extension
                config_base = key.rsplit('.', 1)[0] if '.' in key else key
                emb_base = emb_key.rsplit('.', 1)[0] if '.' in emb_key else emb_key
                emb_filename = emb_key.split('/')[-1]
                emb_filename_base = emb_filename.rsplit('.', 1)[0] if '.' in emb_filename else emb_filename
                
                # Try multiple matching strategies
                if (emb_key.endswith(key) or 
                    emb_key.endswith('/' + key) or
                    emb_base.endswith(config_base) or
                    emb_filename == key or
                    emb_filename_base == config_base):
                    filtered_dict[emb_key] = embeddings_dict[emb_key]
                    matched_count += 1
                    break
    
    print(f"Filtered embeddings: {matched_count}/{len(config_keys)} config keys matched")
    print(f"Using {len(filtered_dict)} embeddings from pickle\n")
    
    return filtered_dict


def extract_categories(embeddings_dict):
    """Extract category labels from embedding keys (default mode)."""
    color_vec = []
    # Sort keys for deterministic ordering
    sorted_keys = sorted(embeddings_dict.keys())
    for key in sorted_keys:
        filename = key.split('/')[-1]
        
        # Extract all attributes for fine-grained categorization
        problem = extract_problem_type(filename)
        difficulty = extract_difficulty(filename)
        satisfiability = extract_satisfiability(filename)
        
        # Combine all attributes
        category = f"{difficulty}_{problem}_{satisfiability}"
        color_vec.append(category)
    
    return color_vec


def extract_problem_type(filename):
    """Extract problem type from filename."""
    if 'G4SATBENCH_' in filename:
        # G4SATBENCH_medium_sr_train_augmented_sat-00495.cnf
        # Format: G4SATBENCH_<difficulty>_<domain>_<split>_<augmentation>_<satisfiability>-<index>
        parts = filename.replace('.cnf', '').split('_')
        # Extract domain (position 2 after G4SATBENCH and difficulty)
        if len(parts) >= 3:
            return parts[2]  # 'sr' or other domain
        else:
            return 'G4SATBENCH'
    elif 'SATLIB_' in filename:
        # SATLIB_PROBLEM_TYPE-rest.mps
        parts = filename.replace('.mps', '').split('-')
        return parts[0].replace('SATLIB_', '')
    elif 'DMIPLIB_' in filename:
        # DMIPLIB_CA-easy_val_instance_9.lp
        parts = filename.replace('.lp', '').split('-')
        return parts[0].replace('DMIPLIB_', '')
    else:
        try:
            return filename.split('_')[0]
        except:
            return filename


def extract_difficulty(filename):
    """Extract difficulty level from filename."""
    filename_lower = filename.lower()
    
    if 'G4SATBENCH_' in filename:
        # G4SATBENCH_<difficulty>_sr_...
        parts = filename.split('_')
        if len(parts) >= 2:
            difficulty = parts[1]  # 'easy', 'medium', 'hard'
            return difficulty
    
    if 'easy' in filename_lower:
        return 'easy'
    elif 'medium' in filename_lower or 'mid' in filename_lower:
        return 'medium'
    elif 'hard' in filename_lower:
        return 'hard'
    else:
        return 'unknown'


def extract_satisfiability(filename):
    """Extract satisfiability status from filename."""
    filename_lower = filename.lower()
    
    if 'unsat' in filename_lower:
        return 'UNSAT'
    elif 'sat' in filename_lower:
        return 'SAT'
    else:
        return 'unknown'


def extract_categories_by_mode(embeddings_dict, mode='default'):
    """Extract category labels based on specified mode."""
    color_vec = []
    
    if mode == 'default':
        return extract_categories(embeddings_dict)
    
    # Sort keys for deterministic ordering
    sorted_keys = sorted(embeddings_dict.keys())
    for key in sorted_keys:
        filename = key.split('/')[-1]
        
        if mode == 'problem_type':
            category = extract_problem_type(filename)
        
        elif mode == 'satisfiability':
            category = extract_satisfiability(filename)
        
        elif mode == 'difficulty':
            category = extract_difficulty(filename)
        
        elif mode == 'problem_difficulty':
            problem = extract_problem_type(filename)
            difficulty = extract_difficulty(filename)
            category = f"{problem}_{difficulty}"
        
        elif mode == 'all':
            problem = extract_problem_type(filename)
            difficulty = extract_difficulty(filename)
            satisfiability = extract_satisfiability(filename)
            category = f"{difficulty}_{problem}_{satisfiability}"
        
        else:
            # Fallback to default
            category = extract_problem_type(filename)
        
        color_vec.append(category)
    
    return color_vec


def prepare_embeddings(embeddings_dict):
    """Convert embeddings to numpy array."""
    embed_list = []
    # Sort keys for deterministic ordering
    sorted_keys = sorted(embeddings_dict.keys())
    for key in sorted_keys:
        emb = embeddings_dict[key]
        if isinstance(emb, np.ndarray):
            embed_list.append(emb)
        else:
            # Handle objects with .instance_embedding attribute
            embed_list.append(emb.instance_embedding)
    
    embed_mat = np.array(embed_list)
    print(f"✓ Prepared embedding matrix with shape {embed_mat.shape}")
    # L2 normalize each embedding - normalize by row (each sample)
    #embed_mat = embed_mat / (np.linalg.norm(embed_mat, axis=1, keepdims=True) + 1e-10)
    # print the shape of the denominator to check for any issues
    #print(f"Denominator shape: {embed_mat.sum(axis=1).shape}")
    embed_mat = embed_mat / (embed_mat.sum(axis = 1, keepdims=True) + 1e-10)
    return embed_mat


def apply_difficulty_modifier(rgb, difficulty):
    """Modify color brightness based on difficulty: lighter for easy, darker for hard."""
    r, g, b = rgb
    if difficulty == 'easy':
        # Lighten (move towards white)
        return (0.6 + r * 0.4, 0.6 + g * 0.4, 0.6 + b * 0.4)
    elif difficulty == 'hard':
        # Darken (move towards black)
        return (r * 0.7, g * 0.7, b * 0.7)
    else:  # medium
        # Keep as is
        return rgb


def apply_satisfiability_modifier(rgb, satisfiability):
    """Modify color saturation based on satisfiability: full for SAT, desaturated for UNSAT."""
    if satisfiability.upper() == 'UNSAT':
        # Desaturate by moving towards grey
        r, g, b = rgb
        grey = (r + g + b) / 3.0
        return (0.5 * r + 0.5 * grey, 0.5 * g + 0.5 * grey, 0.5 * b + 0.5 * grey)
    else:
        # SAT: keep full saturation
        return rgb


def create_color_map(color_vec):
    """Create color mapping: base color per domain, modified by difficulty and satisfiability."""
    
    # Extract unique categories and their attributes
    unique_categories = sorted(set(color_vec))
    color_map = {}
    
    # Define base colors for each domain (distinct hues/colors)
    domain_base_colors = {
        '3-sat': (0.2, 0.4, 0.9),      # Blue
        'ca': (0.1, 0.7, 0.3),         # Green
        'k-clique': (0.9, 0.3, 0.2),   # Red-orange
        'k-domset': (0.9, 0.7, 0.1),   # Gold
        'k-vercov': (0.7, 0.2, 0.8),   # Purple
        'ps': (0.2, 0.8, 0.8),         # Cyan
        'sr': (0.8, 0.2, 0.6),         # Magenta
    }
    
    # Default colors for unknown domains by difficulty
    default_by_difficulty = {
        'easy': (0.3, 0.6, 0.95),      # Light blue
        'medium': (0.3, 0.8, 0.3),     # Light green
        'hard': (0.95, 0.3, 0.3),      # Light red
    }
    
    for category in unique_categories:
        parts = category.split('_')
        color = (0.5, 0.5, 0.5)  # Default grey fallback
        
        # Handle different category formats
        if len(parts) >= 3:
            # Format: difficulty_domain_satisfiability
            difficulty = parts[0]  # easy, medium, hard
            domain = '_'.join(parts[1:-1])  # domain (may have underscores like 'k-clique')
            satisfiability = parts[-1]  # SAT, UNSAT
            
            # Get base color for domain
            if domain in domain_base_colors:
                color = domain_base_colors[domain]
            else:
                color = default_by_difficulty.get(difficulty, (0.5, 0.5, 0.5))
            
            # Apply modifiers
            color = apply_difficulty_modifier(color, difficulty)
            color = apply_satisfiability_modifier(color, satisfiability)
        
        elif len(parts) == 2:
            # Format: domain_difficulty (e.g., sr_easy, 3-sat_hard)
            domain = parts[0]
            difficulty = parts[1]
            
            if difficulty in ['easy', 'medium', 'hard']:
                # Get base color for domain
                if domain in domain_base_colors:
                    color = domain_base_colors[domain]
                else:
                    color = default_by_difficulty.get(difficulty, (0.5, 0.5, 0.5))
                
                # Apply difficulty modifier (no satisfiability in this format)
                color = apply_difficulty_modifier(color, difficulty)
            elif domain in domain_base_colors:
                # Just a domain name, use base color
                color = domain_base_colors[domain]
        
        elif len(parts) == 1:
            # Single category (domain or difficulty)
            category_name = parts[0]
            if category_name in domain_base_colors:
                color = domain_base_colors[category_name]
            elif category_name in default_by_difficulty:
                color = default_by_difficulty[category_name]
        
        else:
            # Single part or unusual format - use basic coloring
            category_lower = category.lower()
            if 'easy' in category_lower:
                color = (0.2, 0.5, 0.95)
            elif 'medium' in category_lower:
                color = (0.2, 0.75, 0.2)
            elif 'hard' in category_lower:
                color = (0.9, 0.1, 0.1)
        
        color_map[category] = color
    
    return color_map


def visualize_embeddings(embeddings_dict, embed_mat, color_vec, output_file="clusters_visualization.png", n_clusters=None, nmi=None, purity=None):
    """Visualize embeddings using PaCMAP. Optionally include clustering results on plot."""
    
    print("Computing PaCMAP projection...")
    pca = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=42).fit_transform(embed_mat, init='pca')
    pca = (pca - np.min(pca)) / np.ptp(pca)
    
    # Build index dictionary
    index_dict = {}
    for idx, c in enumerate(color_vec):
        try:
            index_dict[c].append(idx)
        except:
            index_dict[c] = [idx]
    
    # Create intuitive color map
    color_map = create_color_map(color_vec)
    
    # Plot
    print("Creating visualization...")
    num_unique = len(index_dict)
    
    fig = plt.figure(figsize=(10, 8), dpi=300)
    
    for c in sorted(index_dict):
        color = color_map.get(c, (0.5, 0.5, 0.5))
        plt.scatter(
            pca[index_dict[c], 0],
            pca[index_dict[c], 1],
            s=20,
            label=c,
            color=color,
            alpha=0.8,
            edgecolors='black',
            linewidth=0.3
        )
    
    # Build title with optional clustering info
    title = f"Embedding Clusters ({num_unique} categories)"
    if n_clusters is not None:
        title += f"\nClusters: {n_clusters} | NMI: {nmi:.4f} | Purity: {purity:.4f}"
    
    plt.title(title, fontsize=12, pad=20)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8, ncol=1)
    plt.tight_layout()
    
    # Create visualizations directory if it doesn't exist
    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Save as both PNG and PDF in visualizations directory
    png_file = os.path.join(viz_dir, output_file if output_file.endswith('.png') else output_file.replace('.pdf', '.png'))
    pdf_file = os.path.join(viz_dir, output_file.replace('.png', '.pdf') if output_file.endswith('.png') else output_file)
    
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved PNG to {png_file}")
    plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved PDF to {pdf_file}\n")
    plt.close()
    
    return pca, index_dict


def purity_score(y_true, y_pred):
    """Calculate purity score."""
    con_matrix = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(con_matrix, axis=0)) / np.sum(con_matrix)


def visualize_embedding_histogram(embed_mat, output_file="embedding_values_histogram.png"):
    """Create and save a histogram of all embedding values."""
    print("Creating histogram of embedding values...")
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    # Flatten all embedding values and create histogram
    all_values = embed_mat.flatten()
    ax.hist(all_values, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    
    ax.set_xlabel('Embedding Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Distribution of Embedding Values\n(Total values: {len(all_values):,})', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f'Mean: {np.mean(all_values):.4f}\nStd: {np.std(all_values):.4f}\nMin: {np.min(all_values):.4f}\nMax: {np.max(all_values):.4f}'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Create visualizations directory if it doesn't exist
    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    hist_file = os.path.join(viz_dir, output_file)
    plt.savefig(hist_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved histogram to {hist_file}\n")
    plt.close()


def cluster_embeddings(embed_mat, color_vec, n_runs=10):
    """Cluster embeddings and compute NMI and purity scores."""
    
    print("Running K-means clustering analysis...")
    num_clusters = len(set(color_vec))
    res_nmi = []
    res_acc = []
    
    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}...", end=" ", flush=True)
        km_dist = KMeans(n_clusters=num_clusters, random_state=run, n_init=20).fit(embed_mat)
        
        true = color_vec
        pred = km_dist.labels_
        
        acc = {}
        for c in range(num_clusters):
            indices = np.where(pred == c)[0]
            if len(indices) > 0:
                acc[c] = purity_score(np.array(true)[indices], pred[indices])
            else:
                acc[c] = 0.0
        
        nmi_score = normalized_mutual_info_score(true, pred)
        mean_acc = np.mean(list(acc.values()))
        
        res_nmi.append(nmi_score)
        res_acc.append(mean_acc)
        print(f"NMI={nmi_score:.3f}, ACC={mean_acc:.3f}")
    
    print("\n" + "="*60)
    print("CLUSTERING RESULTS:")
    print("="*60)
    print(f"  Number of clusters: {num_clusters}")
    if len(res_nmi) > 1:
        print(f"  NMI: {np.mean(res_nmi):.3f} ± {np.std(res_nmi):.3f}")
        print(f"  Purity (ACC): {np.mean(res_acc):.3f} ± {np.std(res_acc):.3f}")
    else:
        print(f"  NMI: {res_nmi[0]:.3f}")
        print(f"  Purity (ACC): {res_acc[0]:.3f}")
    print("="*60 + "\n")
    
    return num_clusters, np.mean(res_nmi), np.mean(res_acc)


def main():
    """Main execution function."""
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python visualize_clusters.py <pickle_file> [--config <config_file>] [--category-mode <mode>] [--no-cluster]")
        print("\nCategory modes:")
        print("  default: Original category extraction")
        print("  problem_type: Group by problem/domain type")
        print("  satisfiability: Group by SAT/UNSAT status")
        print("  difficulty: Group by difficulty level (easy/medium/hard)")
        print("  problem_difficulty: Combine problem_type and difficulty")
        print("  all: Combine all attributes (difficulty_problemtype_satisfiability)")
        print("\nExamples:")
        print("  python visualize_clusters.py satlib_mip_to_embeddings.pkl")
        print("  python visualize_clusters.py embeddings.pkl --config data/configs/iclr_test_clusters.txt")
        print("  python visualize_clusters.py embeddings.pkl --category-mode problem_difficulty")
        print("  python visualize_clusters.py embeddings.pkl --config data/configs/iclr_test_clusters.txt --category-mode all --no-cluster")
        sys.exit(1)
    
    # Set seeds for reproducibility
    #np.random.seed(42)
    
    pickle_file = sys.argv[1]
    skip_clustering = '--no-cluster' in sys.argv
    config_file = None
    category_mode = 'default'
    
    # Extract config file if provided
    if '--config' in sys.argv:
        config_idx = sys.argv.index('--config')
        if config_idx + 1 < len(sys.argv):
            config_file = sys.argv[config_idx + 1]
    
    # Extract category mode if provided
    if '--category-mode' in sys.argv:
        mode_idx = sys.argv.index('--category-mode')
        if mode_idx + 1 < len(sys.argv):
            category_mode = sys.argv[mode_idx + 1]
    
    print("=" * 70)
    print("CLUSTER VISUALIZATION")
    print("=" * 70 + "\n")
    
    # Load embeddings
    embeddings_dict = load_embeddings(pickle_file)
    
    # Filter by config if provided
    if config_file:
        print("=" * 70)
        print("FILTERING BY CONFIG")
        print("=" * 70)
        config_keys = load_config(config_file)
        embeddings_dict = filter_embeddings_by_config(embeddings_dict, config_keys)
    
    # Extract categories and embeddings
    print("=" * 70)
    print("CATEGORIZATION")
    print("=" * 70)
    print(f"Category mode: {category_mode}\n")
    
    # Check if we have embeddings after filtering
    if len(embeddings_dict) == 0:
        print("ERROR: No embeddings after filtering!")
        print("Check that config keys match the pickle keys.")
        if config_file:
            print("\nDEBUG: Showing first 5 pickle keys for comparison:")
            # Load again to show examples
            embeddings_dict_all = load_embeddings(pickle_file)
            for i, key in enumerate(list(embeddings_dict_all.keys())[:5]):
                print(f"  {key}")
        sys.exit(1)
    
    color_vec = extract_categories_by_mode(embeddings_dict, category_mode)
    embed_mat = prepare_embeddings(embeddings_dict)
    
    print(f"Embeddings shape: {embed_mat.shape}")
    print(f"Number of unique categories: {len(set(color_vec))}")
    print(f"Categories: {sorted(set(color_vec))}\n")
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(pickle_file))[0]
    if config_file:
        config_base = os.path.splitext(os.path.basename(config_file))[0]
        output_file = f"{base_name}_{config_base}_{category_mode}_visualization.png"
        hist_file = f"{base_name}_{config_base}_histogram.png"
    else:
        output_file = f"{base_name}_{category_mode}_visualization.png"
        hist_file = f"{base_name}_histogram.png"
    
    # Create histogram of embedding values
    print("=" * 70)
    print("EMBEDDING VALUE DISTRIBUTION")
    print("=" * 70)
    visualize_embedding_histogram(embed_mat, hist_file)
    
    # Visualize
    print("=" * 70)
    print("VISUALIZATION")
    print("=" * 70)
    
    # Cluster analysis first to get results for visualization
    if not skip_clustering:
        print("=" * 70)
        print("CLUSTERING ANALYSIS")
        print("=" * 70)
        n_clusters, nmi, purity = cluster_embeddings(embed_mat, color_vec)
        pca, index_dict = visualize_embeddings(embeddings_dict, embed_mat, color_vec, output_file, n_clusters=n_clusters, nmi=nmi, purity=purity)
    else:
        pca, index_dict = visualize_embeddings(embeddings_dict, embed_mat, color_vec, output_file)
    
    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
