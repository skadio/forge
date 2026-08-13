"""
cluster.py — Embed MIP instances with Forge, cluster the embeddings, and visualize.

Three-stage pipeline
--------------------
  Stage 1 (embed):   Generate Forge embeddings for each MIP instance.
                      Requires torch, gurobipy, and forge.  Saves instance_to_embedding.pkl.
  Stage 2 (cluster): Determine optimal k via silhouette analysis, run K-Means.
                      Saves cluster_assignments.pkl.
  Stage 3 (plot):    Project to 2D with UMAP and produce visualizations.

Usage
-----
  # Full pipeline (embed → cluster → plot):
  python cluster.py

  # Only clustering + viz (embeddings already exist):
  python cluster.py --skip_embedding

  # Custom settings:
  python cluster.py --max_k 20 --min_k 2 --output_dir my_results/

  # Force a specific number of clusters:
  python cluster.py --skip_embedding --force_k 5

Outputs (in --output_dir, default: results_cluster/)
-----------------------------------------------------
  instance_to_embedding.pkl      {name: np.ndarray(codebook_size,)} embeddings
  cluster_assignments.pkl        {name: cluster_id, ..., "__meta__": {...}}
  cluster_summary.json           Human-readable cluster report
  umap_clusters.png              2D UMAP scatter coloured by cluster
  silhouette_analysis.png        Silhouette score vs k
  elbow_plot.png                 Inertia (elbow method) vs k
  cluster_sizes.png              Bar chart of cluster sizes
  silhouette_detail.png          Per-cluster silhouette diagram
"""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path
from collections import Counter

import numpy as np

import matplotlib
matplotlib.use("Agg")  # non-interactive backend – safe for servers / CI
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples


# ═══════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent
FORGE_DIR = BASE_DIR
MIPFEAS_DIR = BASE_DIR / "data" / "mipfeas"
FORGE_MODEL_PKL = FORGE_DIR / "models" / "forge_pretrain_trained.pkl"
TRAIN_CONFIG_YAML = FORGE_DIR / "forge" / "configs" / "train_config.yaml"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results_mipfeas_cluster"


# ═══════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════
def _instance_name(path: str) -> str:
    """Strip directory and all extensions (.mps.gz, .lp, etc.) from *path*."""
    base = os.path.basename(str(path))
    for ext in (".mps.gz", ".lp.gz", ".mps", ".lp"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]


# ═══════════════════════════════════════════════════════════════════
# Stage 1 – Embedding generation / loading
# ═══════════════════════════════════════════════════════════════════
def generate_embeddings(instance_dir: str,
                        output_pkl: str,
                        forge_model_pkl: str = None,
                        train_config_yaml: str = None) -> dict:
    """Generate Forge instance-level embeddings for every MIP file in *instance_dir*.

    Requires torch, torch-geometric, gurobipy, vector-quantize-pytorch, and the
    ``forge`` package (which lives under ``git_forge/``).  The function adds
    ``git_forge/`` to ``sys.path`` automatically.

    Returns
    -------
    dict : {instance_name (str): embedding (np.ndarray of shape (codebook_size,))}
    """
    forge_model_pkl = forge_model_pkl or str(FORGE_MODEL_PKL)
    train_config_yaml = train_config_yaml or str(TRAIN_CONFIG_YAML)

    # Make the forge package importable
    forge_path = str(FORGE_DIR)
    if forge_path not in sys.path:
        sys.path.insert(0, forge_path)

    from forge.embeddings import Forge
    from forge.pipeline import mip_to_embeddings
    from forge.utils import Constants

    print(f"[embed] Config:        {train_config_yaml}")
    print(f"[embed] Pretrained:    {forge_model_pkl}")
    print(f"[embed] Instance dir:  {instance_dir}")

    forge = Forge(str(train_config_yaml))

    # Forge's pipeline also persists its own pkl; use a *_raw* sibling.
    forge_raw_pkl = str(output_pkl).replace(".pkl", "_forge_raw.pkl")

    mip_to_emb = mip_to_embeddings(
        forge=forge,
        input_forge_pkl=str(forge_model_pkl),
        model_type=Constants.FORGE_PRE_TRAIN,
        input_mips=str(instance_dir),
        input_mip_instances_file=None,
        output_mip_to_embeddings_pkl=forge_raw_pkl,
        instance_embedding_only=True,
    )

    # Convert to a clean {name → ndarray} dictionary
    instance_to_embedding = {}
    for key, mip_emb in mip_to_emb.items():
        name = _instance_name(key)
        instance_to_embedding[name] = np.asarray(mip_emb.instance_embedding,
                                                  dtype=np.float64)

    os.makedirs(os.path.dirname(output_pkl) or ".", exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(instance_to_embedding, f)
    print(f"[embed] Saved {len(instance_to_embedding)} embeddings → {output_pkl}")
    return instance_to_embedding


def load_embeddings(pkl_path: str) -> dict:
    """Load embeddings from *pkl_path*.

    Transparently handles both the clean ``{name: ndarray}`` format **and** the
    raw Forge format ``{filepath: MIPEmbeddings}``.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    first_val = next(iter(data.values()))
    if isinstance(first_val, np.ndarray):
        print(f"[load] {len(data)} embeddings (clean format) from {pkl_path}")
        return data

    # Forge MIPEmbeddings objects
    clean = {}
    for key, mip_emb in data.items():
        name = _instance_name(key)
        clean[name] = np.asarray(mip_emb.instance_embedding, dtype=np.float64)
    print(f"[load] {len(clean)} embeddings (forge format) from {pkl_path}")
    return clean


# ═══════════════════════════════════════════════════════════════════
# Stage 2 – Clustering
# ═══════════════════════════════════════════════════════════════════
def prepare_matrix(instance_to_embedding: dict, use_pca: bool = True):
    """Build sorted name list and L1-normalised + optionally PCA-reduced embedding matrix.
     The problem with raw counts
     Different MIP instances have wildly different numbers of nodes.
     A small instance like pk1.mps.gz might have 100 nodes total,
     while buildingenergy.mps.gz might have 50,000.
         If you feed these raw vectors into K-Means, the Euclidean distance is dominated by instance size,
         not by structural similarity.
         Two large instances will always be farther apart from each other (in absolute count space)
         than two small instances, even if the large ones have identical proportional code usage patterns.
         K-Means would essentially cluster by instance size, not by structural fingerprint.

    TL;DR — L1 normalization removes the instance-size confound so that
        clustering captures structural similarity in how the GNN maps the MIP graph to discrete codes,
        which is the actual signal you want for identifying hidden MIP families.

    What L1 normalization does
    Dividing each row by its sum (X / row_sums) converts counts to relative frequencies:
    row sum becomes 1.0

    Now every instance lives on the probability simplex — a 5,000-dimensional surface where all vectors sum to 1.
    Distance between instances now measures how differently they distribute their nodes across the codebook,
    regardless of how many nodes they have.

    Why L1 specifically (vs. L2 or other normalization)
    L1 (sum-to-one): Produces a proper probability distribution.
        This is the natural normalization for histograms/count data.
        Distances in this space correspond to divergences between distributions
        (related to total variation distance, and after PCA, to Jensen-Shannon-like dissimilarity).

    L2 (unit-length): Projects onto a hypersphere.
        This is natural for dense real-valued vectors (like word2vec),
        but for sparse count histograms it distorts the geometry — rare codes get inflated relative to common ones.

    StandardScaler (zero-mean, unit-variance per feature): Would treat each code independently,
        destroying the compositional relationship between codes.
        A code that appears in 2% of nodes is not the same "kind of information" as a code appearing in 40% of nodes.

    No normalization: Clusters by instance size, as explained above.

    Returns
    -------
    names  : list[str]          – sorted instance names
    X_norm : np.ndarray (n, d)  – L1-normalised (probability dist over VQ codes)
    X_pca  : np.ndarray (n, p)  – PCA-reduced (p chosen for ≥95 % variance, max 50)
                                   equals X_norm when use_pca=False
    pca    : PCA | None         – fitted PCA object (None when use_pca=False)
    """
    names = sorted(instance_to_embedding.keys())
    X = np.vstack([instance_to_embedding[n] for n in names])

    # L1-normalise → probability distribution over VQ codes
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    X_norm = X / row_sums

    if not use_pca:
        print(f"[prep] {len(names)} instances, "
              f"{X.shape[1]} dims (PCA disabled — using raw L1-normalised vectors)")
        return names, X_norm, X_norm, None

    # PCA to reduce dimensionality while keeping ≥95 % variance (capped at 50)
    n_max = min(50, X_norm.shape[0] - 1, X_norm.shape[1])
    pca = PCA(n_components=n_max, random_state=42)
    X_pca_full = pca.fit_transform(X_norm)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cumvar, 0.95) + 1)
    n_keep = max(n_keep, 2)
    X_pca = X_pca_full[:, :n_keep]

    print(f"[prep] {len(names)} instances, "
          f"{X.shape[1]}→{n_keep} PCA dims "
          f"({cumvar[n_keep - 1] * 100:.1f}% variance explained)")
    return names, X_norm, X_pca, pca


def find_optimal_k(X: np.ndarray, min_k: int = 2, max_k: int = 15):
    """Evaluate K-Means for k ∈ [min_k, max_k] via silhouette score + inertia.

    Returns
    -------
    best_k     : int
    sil_scores : dict {k: float}
    inertias   : dict {k: float}
    """
    max_k = min(max_k, X.shape[0] - 1)  # k < n required
    sil_scores = {}
    inertias = {}

    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=42, max_iter=500)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        sil_scores[k] = sil
        inertias[k] = km.inertia_
        print(f"  k={k:2d}  silhouette={sil:.4f}  inertia={km.inertia_:.1f}")

    best_k = max(sil_scores, key=sil_scores.get)
    print(f"[cluster] Best k by silhouette: {best_k} "
          f"(score={sil_scores[best_k]:.4f})")
    return best_k, sil_scores, inertias


def run_kmeans(X: np.ndarray, k: int):
    """Run K-Means and return (labels, fitted_model)."""
    km = KMeans(n_clusters=k, n_init=30, random_state=42, max_iter=500)
    labels = km.fit_predict(X)
    return labels, km


# ═══════════════════════════════════════════════════════════════════
# Stage 3 – Visualisation
# ═══════════════════════════════════════════════════════════════════
def umap_2d(X: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Project *X* to 2-D with UMAP."""
    try:
        import umap as umap_module
    except ImportError:
        print("[warn] umap-learn not installed.  Install with: pip install umap-learn")
        print("[warn] Falling back to PCA 2-D projection.")
        from sklearn.decomposition import PCA as PCA2
        return PCA2(n_components=2, random_state=random_state).fit_transform(X)

    reducer = umap_module.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                               metric="euclidean", random_state=random_state)
    return reducer.fit_transform(X)


def pacmap_2d(X: np.ndarray) -> np.ndarray:
    """Project *X* to 2-D with PaCMAP (n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)."""
    try:
        import pacmap
    except ImportError:
        print("[warn] pacmap not installed.  Install with: pip install pacmap")
        print("[warn] Falling back to PCA 2-D projection.")
        from sklearn.decomposition import PCA as PCA2
        return PCA2(n_components=2, random_state=42).fit_transform(X)

    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10,
                            MN_ratio=0.5, FP_ratio=2.0)
    return reducer.fit_transform(X, init="pca")


def plot_pacmap(X_2d, labels, names, output_path, k):
    """2-D PaCMAP scatter coloured by cluster, with instance labels for small data sets."""
    fig, ax = plt.subplots(figsize=(14, 10))
    cmap = plt.get_cmap("tab10" if k <= 10 else "tab20")

    for c in range(k):
        mask = labels == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[cmap(c)], label=f"Cluster {c} ({mask.sum()})",
                   s=60, alpha=0.75, edgecolors="w", linewidths=0.5)

    # Annotate individual instances when the dataset is small enough
    if len(names) <= 100:
        for i, name in enumerate(names):
            ax.annotate(name, (X_2d[i, 0], X_2d[i, 1]),
                        fontsize=4.5, alpha=0.55,
                        xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("PaCMAP-1", fontsize=13)
    ax.set_ylabel("PaCMAP-2", fontsize=13)
    ax.set_title(f"MIP Instance Embeddings — {k} Clusters (PaCMAP Projection)",
                 fontsize=15)
    ax.legend(fontsize=10, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {output_path}")


def plot_umap(X_2d, labels, names, output_path, k):
    """2-D scatter coloured by cluster, with instance labels for small data sets."""
    fig, ax = plt.subplots(figsize=(14, 10))
    cmap = plt.get_cmap("tab10" if k <= 10 else "tab20")

    for c in range(k):
        mask = labels == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[cmap(c)], label=f"Cluster {c} ({mask.sum()})",
                   s=60, alpha=0.75, edgecolors="w", linewidths=0.5)

    # Annotate individual instances when the dataset is small enough
    if len(names) <= 100:
        for i, name in enumerate(names):
            ax.annotate(name, (X_2d[i, 0], X_2d[i, 1]),
                        fontsize=4.5, alpha=0.55,
                        xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("UMAP-1", fontsize=13)
    ax.set_ylabel("UMAP-2", fontsize=13)
    ax.set_title(f"MIP Instance Embeddings — {k} Clusters (UMAP Projection)",
                 fontsize=15)
    ax.legend(fontsize=10, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {output_path}")


def plot_silhouette(sil_scores, best_k, output_path):
    """Silhouette score vs. k."""
    ks = sorted(sil_scores.keys())
    scores = [sil_scores[k] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, scores, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.axvline(best_k, color="crimson", ls="--", label=f"Best k = {best_k}")
    ax.set_xlabel("Number of Clusters (k)", fontsize=13)
    ax.set_ylabel("Silhouette Score", fontsize=13)
    ax.set_title("Silhouette Analysis for Optimal k", fontsize=15)
    ax.legend(fontsize=11)
    ax.set_xticks(ks)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {output_path}")


def plot_elbow(inertias, best_k, output_path):
    """Inertia (elbow) vs. k."""
    ks = sorted(inertias.keys())
    vals = [inertias[k] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, vals, "s-", color="darkorange", linewidth=2, markersize=8)
    ax.axvline(best_k, color="crimson", ls="--", label=f"Selected k = {best_k}")
    ax.set_xlabel("Number of Clusters (k)", fontsize=13)
    ax.set_ylabel("Inertia (within-cluster sum of squares)", fontsize=13)
    ax.set_title("Elbow Method", fontsize=15)
    ax.legend(fontsize=11)
    ax.set_xticks(ks)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {output_path}")


def plot_cluster_sizes(labels, k, output_path):
    """Bar chart of cluster sizes."""
    counts = Counter(labels)
    clusters = list(range(k))
    sizes = [counts.get(c, 0) for c in clusters]

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10" if k <= 10 else "tab20")
    bars = ax.bar(clusters, sizes,
                  color=[cmap(c) for c in clusters], edgecolor="w")
    for bar, s in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(s), ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xlabel("Cluster", fontsize=13)
    ax.set_ylabel("Number of Instances", fontsize=13)
    ax.set_title("Cluster Sizes", fontsize=15)
    ax.set_xticks(clusters)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {output_path}")


def plot_silhouette_detail(X, labels, k, output_path):
    """Per-cluster silhouette diagram.
    What is a Silhouette Coefficient?
        For each individual instance i, the silhouette coefficient measures
        how well it fits its assigned cluster vs. the next best cluster:
        s(i) = (b(i) - a(i)) / max(a(i), b(i))
        Where:
        a(i) = mean distance from instance i to all other instances in the same cluster (cohesion — lower is better)
        b(i) = mean distance from instance i to all instances in the nearest other cluster (separation — higher is better)
        The result ranges from -1 to +1:
        Close to +1:     Instance is well inside its cluster, far from others — great fit
        Close to 0:     Instance is on the border between two clusters — ambiguous
        Negative:     Instance is closer to a different cluster than its own — misclassified

    What is the Mean Silhouette Score (red line)?
        It's simply the average of all individual silhouette coefficients across all instances.
        It's the single number used to compare different values of k in the silhouette analysis plot. Higher = better overall clustering.

    How to Read the Plot
        Each horizontal bar for a cluster is built like this:
        Take all instances belonging to cluster C
        Compute each instance's silhouette coefficient
        Sort them from smallest to largest (bottom to top)
        Draw a filled horizontal shape — the width of each thin slice = that instance's silhouette value
        So visually:
        Tall bars = large clusters (many instances)
        Wide bars (extending right) = high silhouette values = tight, well-separated cluster
        Narrow or left-extending bars = weak or bad cluster membership

    What Insights You Can Get
    Good clustering signs:
        All bars extend well to the right of the red line → most instances fit their cluster well
        Bars are roughly uniform in width → no stragglers
        No or very few negative values (bars crossing x=0 into negative territory)
    Warning signs:
        A cluster with many instances having negative silhouette values → that cluster is probably wrong or should be merged
        A cluster bar that is much shorter/thinner than others → weak, poorly-defined cluster
        Large variation in width within one bar → some instances fit well, others don't
        For your silhouette_detail.png specifically:
        Look for clusters where the bar extends past the red mean line — those are your "clean" MIP families
        Any cluster with a thin, ragged, or left-leaning bar likely contains MIP instances that don't naturally belong together — those are structurally mixed instances
        If one cluster has a very tall but narrow bar, you have many similar instances that are nonetheless borderline — consider splitting that cluster
    """
    sample_sil = silhouette_samples(X, labels)
    avg_sil = silhouette_score(X, labels)

    fig, ax = plt.subplots(figsize=(8, max(6, k * 1.2)))
    y_lower = 10
    cmap = plt.get_cmap("tab10" if k <= 10 else "tab20")

    for c in range(k):
        c_sil = np.sort(sample_sil[labels == c])
        size = len(c_sil)
        y_upper = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, c_sil,
                         facecolor=cmap(c), edgecolor=cmap(c), alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size, f"C{c}",
                fontsize=10, fontweight="bold")
        y_lower = y_upper + 10

    ax.axvline(avg_sil, color="red", ls="--",
               label=f"Mean = {avg_sil:.3f}")
    ax.set_xlabel("Silhouette Coefficient", fontsize=13)
    ax.set_ylabel("Instances (sorted within cluster)", fontsize=13)
    ax.set_title("Per-Cluster Silhouette Diagram", fontsize=15)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {output_path}")


# ═══════════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════════
def save_cluster_results(names, labels, k, sil_scores, output_dir):
    """Persist cluster assignments (.pkl) and a human-readable summary (.json)."""

    # ── Pickle ────────────────────────────────────────────────────
    assignments = {name: int(label) for name, label in zip(names, labels)}
    assignments["__meta__"] = {
        "k": k,
        "silhouette": sil_scores.get(k),
        "n_instances": len(names),
    }
    pkl_path = os.path.join(output_dir, "cluster_assignments.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(assignments, f)

    # ── JSON summary ──────────────────────────────────────────────
    cluster_to_instances = {}
    for c in range(k):
        members = sorted(n for n, l in zip(names, labels) if l == c)
        cluster_to_instances[str(c)] = members

    summary = {
        "k": k,
        "silhouette_score": round(sil_scores.get(k, 0), 4),
        "n_instances": len(names),
        "cluster_sizes": {str(c): len(v) for c, v in cluster_to_instances.items()},
        "clusters": cluster_to_instances,
        "silhouette_all_k": {str(kk): round(v, 4)
                             for kk, v in sorted(sil_scores.items())},
    }
    json_path = os.path.join(output_dir, "cluster_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[save] cluster_assignments.pkl  ({len(names)} instances, k={k})")
    print(f"[save] cluster_summary.json")
    return assignments


# ═══════════════════════════════════════════════════════════════════
# Stage 2 + 3 combined runner (one PCA mode)
# ═══════════════════════════════════════════════════════════════════
def run_analysis(instance_to_embedding: dict, args, output_dir: str,
                 use_pca: bool) -> None:
    """Run clustering + visualisation for a single PCA mode.

    Parameters
    ----------
    instance_to_embedding : dict
    args                  : parsed argparse namespace
    output_dir            : directory to write all outputs for this run
    use_pca               : whether to apply PCA before clustering
    """
    label = "with PCA" if use_pca else "no PCA"
    os.makedirs(output_dir, exist_ok=True)

    # ── Stage 2: Clustering ──────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"STAGE 2: Clustering Analysis [{label}]")
    print("=" * 60)

    names, X_norm, X_pca, pca = prepare_matrix(instance_to_embedding,
                                               use_pca=use_pca)

    if args.force_k is not None:
        best_k = args.force_k
        sil_scores = {}
        inertias = {}
        labels, km_model = run_kmeans(X_pca, best_k)
        sil_scores[best_k] = silhouette_score(X_pca, labels)
        inertias[best_k] = km_model.inertia_
        print(f"[cluster] Forced k = {best_k}  "
              f"(silhouette = {sil_scores[best_k]:.4f})")
    else:
        print(f"[cluster] Searching k in [{args.min_k}, {args.max_k}] ...")
        best_k, sil_scores, inertias = find_optimal_k(
            X_pca, args.min_k, args.max_k)
        labels, km_model = run_kmeans(X_pca, best_k)

    save_cluster_results(names, labels, best_k, sil_scores, output_dir)

    # ── Stage 3: Visualisation ───────────────────────────────────
    print("\n" + "=" * 60)
    print(f"STAGE 3: Visualization [{label}]")
    print("=" * 60)

    print("[plot] Computing UMAP 2-D projection ...")
    X_2d_umap = umap_2d(X_pca)
    np.savez(os.path.join(output_dir, "umap_coords.npz"),
             X_2d=X_2d_umap, names=names, labels=labels)

    print("[plot] Computing PaCMAP 2-D projection ...")
    X_2d_pacmap = pacmap_2d(X_pca)
    np.savez(os.path.join(output_dir, "pacmap_coords.npz"),
             X_2d=X_2d_pacmap, names=names, labels=labels)

    plot_umap(X_2d_umap, labels, names,
              os.path.join(output_dir, "umap_clusters.png"), best_k)
    plot_pacmap(X_2d_pacmap, labels, names,
                os.path.join(output_dir, "pacmap_clusters.png"), best_k)

    if len(sil_scores) > 1:
        plot_silhouette(sil_scores, best_k,
                        os.path.join(output_dir, "silhouette_analysis.png"))
        plot_elbow(inertias, best_k,
                   os.path.join(output_dir, "elbow_plot.png"))

    plot_cluster_sizes(labels, best_k,
                       os.path.join(output_dir, "cluster_sizes.png"))
    plot_silhouette_detail(X_pca, labels, best_k,
                           os.path.join(output_dir, "silhouette_detail.png"))

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"CLUSTER SUMMARY [{label}]")
    print("=" * 60)
    for c in range(best_k):
        members = sorted(n for n, lbl in zip(names, labels) if lbl == c)
        print(f"\n  Cluster {c}  ({len(members)} instances):")
        for m in members:
            print(f"    - {m}")

    print(f"\nOutputs saved to: {output_dir}/")
    print("  cluster_assignments.pkl  cluster_summary.json")
    print("  umap_coords.npz          umap_clusters.png")
    print("  pacmap_coords.npz        pacmap_clusters.png")
    if len(sil_scores) > 1:
        print("  silhouette_analysis.png  elbow_plot.png")
    print("  cluster_sizes.png        silhouette_detail.png")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Embed MIP instances with Forge, cluster, and visualize.")
    parser.add_argument("--instance_dir", type=str, default=str(MIPFEAS_DIR),
                        help="Directory containing .mps/.mps.gz/.lp files "
                             "(default: data/mipfeas/)")
    parser.add_argument("--output_dir", type=str,
                        default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory for all outputs (default: results_cluster/)")
    parser.add_argument("--skip_embedding", action="store_true",
                        help="Skip embedding generation; load from existing pkl")
    parser.add_argument("--embedding_pkl", type=str, default="mipfeas_mip_to_embeddings_from_forge_pretrain_trained.pkl",
                        help="Path to an existing instance_to_embedding.pkl "
                             "(implies --skip_embedding)")
    parser.add_argument("--forge_model", type=str,
                        default=str(FORGE_MODEL_PKL),
                        help="Path to the pretrained Forge .pkl")
    parser.add_argument("--min_k", type=int, default=2,
                        help="Minimum k to try (default: 2)")
    parser.add_argument("--max_k", type=int, default=15,
                        help="Maximum k to try (default: 15)")
    parser.add_argument("--force_k", type=int, default=3,
                        help="Force a specific k (skips silhouette search)")
    parser.add_argument("--seed", type=int, default=1283,
                        help="Random seed (default: 1283)")
    parser.add_argument("--pca", choices=["on", "off", "both"], default="both",
                        help="PCA dimensionality reduction before clustering: "
                             "'on'   – always apply PCA (outputs in output_dir/with_pca/ when both); "
                             "'off'  – cluster on raw L1-normalised vectors; "
                             "'both' – run both modes and save to with_pca/ and no_pca/ subdirs "
                             "(default: both)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    emb_pkl = (args.embedding_pkl
               or os.path.join(output_dir, "instance_to_embedding.pkl"))

    # ── Stage 1: Embeddings ──────────────────────────────────────
    print("=" * 60)
    print("STAGE 1: Embeddings")
    print("=" * 60)

    if args.embedding_pkl or args.skip_embedding:
        if not os.path.isfile(emb_pkl):
            print(f"ERROR: Embedding file not found: {emb_pkl}")
            print("Run without --skip_embedding to generate embeddings first.")
            sys.exit(1)
        instance_to_embedding = load_embeddings(emb_pkl)
    else:
        if os.path.isfile(emb_pkl):
            print(f"[embed] Found existing {emb_pkl} — loading "
                  f"(pass --skip_embedding to suppress this message)")
            instance_to_embedding = load_embeddings(emb_pkl)
        else:
            instance_to_embedding = generate_embeddings(
                instance_dir=args.instance_dir,
                output_pkl=emb_pkl,
                forge_model_pkl=args.forge_model,
            )

    if len(instance_to_embedding) < 4:
        print(f"ERROR: Only {len(instance_to_embedding)} instances — need ≥ 4 "
              f"for meaningful clustering.")
        sys.exit(1)

    # ── Stages 2 + 3: Clustering + Visualisation ─────────────────
    # Build (use_pca, run_output_dir) pairs based on --pca flag.
    # When "both", each mode gets its own subdirectory so nothing is overwritten.
    if args.pca == "both":
        modes = [
            (True,  os.path.join(output_dir, "with_pca")),
            (False, os.path.join(output_dir, "no_pca")),
        ]
    elif args.pca == "on":
        modes = [(True, output_dir)]
    else:  # "off"
        modes = [(False, output_dir)]

    for use_pca, run_dir in modes:
        run_analysis(instance_to_embedding, args, run_dir, use_pca)

    print("\nDone!")


if __name__ == "__main__":
    main()

