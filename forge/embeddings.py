import gc
import math
import os
import time
from typing import List, Callable, Optional, Tuple, Union, Dict
from collections import Counter

import gurobipy as gp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import yaml
from vector_quantize_pytorch import VectorQuantize

from forge._wgsage import EdgeWeightedSAGEConv, blockwise_loss
from forge.labeler import GapInfo, SATSatisfiabilityInfo
from forge.processor import (MIPInfo, MIPEmbeddings, MIPProcessor, _MIPUtils,
                             SATInfo, SATEmbeddings, SATProcessor, _SATUtils)
from forge.utils import check_true, Constants, overwrite_if_given, copy_params


class VQDiagnostics:
    """Track VQ codebook health during training.
    
    Monitors code usage frequency, embedding distributions, and perplexity to detect
    codebook collapse early and verify fix effectiveness. Distinguishes between all-time
    code activation (which can be misleading) and current distribution health (perplexity).
    """
    
    def __init__(self, codebook_size: int, embedding_dim: int):
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.code_usage_count = Counter()  # All-time usage for reference
        self.recent_code_usage = Counter()  # Recent window for accurate metrics
        self.embedding_norms = []
        self.embedding_means = []
        self.embedding_sparsities = []  # % of values near zero
        self.commitment_losses = []
        self.batch_perplexities = []
        self.recent_window_size = 100  # Track last 100 batches for rolling metrics
        self.all_indices_recent = []  # Store recent batch indices
        
    def track_batch(self, 
                   indices: torch.Tensor,
                   embeddings: torch.Tensor,
                   commitment_loss: torch.Tensor) -> Dict:
        """
        Track statistics for a single batch.
        
        Parameters
        ----------
        indices : torch.Tensor
            Code assignments, shape (num_nodes,) or (num_nodes, 1)
        embeddings : torch.Tensor
            Pre-quantization embeddings, shape (num_nodes, embedding_dim)
        commitment_loss : torch.Tensor
            Scalar commitment loss value
            
        Returns
        -------
        dict
            Dictionary with batch statistics (current distribution-based)
        """
        # Flatten indices if needed
        if indices.dim() > 1:
            indices = indices.squeeze()
        indices_np = indices.cpu().detach().numpy()
        
        # Update both all-time and recent tracking
        for code_id in indices_np:
            self.code_usage_count[int(code_id)] += 1
            self.recent_code_usage[int(code_id)] += 1
        
        # Keep rolling window of recent indices for accurate metrics
        self.all_indices_recent.extend(indices_np)
        # Trim to recent window
        if len(self.all_indices_recent) > self.recent_window_size * 1000:  # ~1000 nodes per batch
            self.all_indices_recent = self.all_indices_recent[-self.recent_window_size * 1000:]
        
        # Embedding statistics
        emb_np = embeddings.cpu().detach().numpy()
        norms = np.linalg.norm(emb_np, axis=1)
        means = np.mean(emb_np, axis=1)
        # Sparsity: % of embedding values near zero (< 0.01 in absolute value)
        sparsity = 100.0 * np.sum(np.abs(emb_np) < 0.01) / emb_np.size
        
        self.embedding_norms.extend(norms)
        self.embedding_means.extend(means)
        self.embedding_sparsities.append(sparsity)
        self.commitment_losses.append(commitment_loss.item())
        
        # Compute batch perplexity (reflects current batch distribution)
        code_freqs = np.bincount(indices_np, minlength=self.codebook_size)
        # Avoid log(0)
        code_freqs_nonzero = code_freqs[code_freqs > 0] / len(indices_np)
        entropy = -np.sum(code_freqs_nonzero * np.log(code_freqs_nonzero))
        perplexity = np.exp(entropy)
        self.batch_perplexities.append(perplexity)
        
        # Compute CURRENT DISTRIBUTION statistics (more meaningful than all-time)
        recent_codes_used = len(self.recent_code_usage)
        recent_util_pct = 100.0 * recent_codes_used / self.codebook_size
        
        # Code concentration: ratio of top-3 codes to total
        top_3_codes = self.recent_code_usage.most_common(3)
        top_3_count = sum(count for _, count in top_3_codes)
        total_count = sum(self.recent_code_usage.values())
        concentration = 100.0 * top_3_count / total_count if total_count > 0 else 0
        
        return {
            'codes_used_recent': recent_codes_used,
            'utilization_pct_recent': recent_util_pct,  # ACCURATE metric
            'codes_used_all_time': len(self.code_usage_count),  # For reference only
            'utilization_pct_all_time': 100.0 * len(self.code_usage_count) / self.codebook_size,
            'mean_norm': np.mean(norms),
            'std_norm': np.std(norms),
            'mean_value': np.mean(means),
            'perplexity': perplexity,  # Best indicator of concentration
            'top_3_concentration': concentration,  # % of assignments in top 3 codes
            'commitment_loss': commitment_loss.item(),
            'embedding_sparsity': sparsity,  # % of values near zero (poor dimensionality use)
        }
    
    def report(self, prefix: str = "") -> str:
        """Generate summary report."""
        # Recent metrics are more meaningful
        recent_codes_used = len(self.recent_code_usage)
        recent_util_pct = 100.0 * recent_codes_used / self.codebook_size
        
        # All-time for reference
        alltime_codes_used = len(self.code_usage_count)
        alltime_util_pct = 100.0 * alltime_codes_used / self.codebook_size
        
        top_5_codes_recent = self.recent_code_usage.most_common(5)
        top_5_usage = [f"Code {code_id}: {count}" for code_id, count in top_5_codes_recent]
        
        # Calculate concentration in recent window
        top_3_codes = self.recent_code_usage.most_common(3)
        top_3_count = sum(count for _, count in top_3_codes)
        total_count = sum(self.recent_code_usage.values())
        concentration = 100.0 * top_3_count / total_count if total_count > 0 else 0
        
        # Average perplexity indicates actual diversity
        avg_perplexity = np.mean(self.batch_perplexities) if self.batch_perplexities else 0
        avg_sparsity = np.mean(self.embedding_sparsities) if self.embedding_sparsities else 0
        
        report_lines = [
            f"\n{prefix} CODEBOOK DIAGNOSTICS",
            f"  ✓ CODE DIVERSITY (VQ Metrics):",
            f"    Codes Used: {recent_codes_used} / {self.codebook_size} ({recent_util_pct:.1f}%)",
            f"    Top 3 Code Concentration: {concentration:.1f}% (lower is better; 0-1% is excellent)",
            f"    Perplexity: {avg_perplexity:.1f} (higher means more distributed; >50 is good)",
            f"",
            f"  ✓ EMBEDDING SPACE UTILIZATION:",
            f"    Sparsity: {avg_sparsity:.1f}% of values near zero (lower is better; <20% is good)",
            f"    Embedding Norm: mean={np.mean(self.embedding_norms):.4f}, std={np.std(self.embedding_norms):.4f}",
            f"    Avg Commitment Loss: {np.mean(self.commitment_losses):.6f}",
            f"",
            f"  📊 Reference Metrics:",
            f"    All-time Codes: {alltime_codes_used} / {self.codebook_size} ({alltime_util_pct:.1f}%) [may overlap with dead codes]",
            f"    Top 5 Recent Codes: {', '.join(top_5_usage)}",
        ]
        
        # Alert if experiencing issues
        issues = []
        
        # Code diversity check
        if concentration > 80:
            issues.append(f"  ⚠️  CODEBOOK COLLAPSE: {concentration:.1f}% of assignments in top 3 codes!")
        elif concentration > 60:
            issues.append(f"  ⚠️  CODE CONCENTRATION: {concentration:.1f}% in top 3 codes (watch this)")
        
        # Embedding space utilization check
        if avg_sparsity > 90:
            issues.append(f"  ⚠️  EMBEDDING SPACE UNDERUTILIZED: {avg_sparsity:.1f}% of values near zero!")
            issues.append(f"     Codes are clustered in low-dimensional subspace. Suggestions:")
            issues.append(f"     - Reduce orthogonal_reg_weight (may be over-constraining)")
            issues.append(f"     - Reduce lambda_edge (reconstruction pressure may force codes together)")
            issues.append(f"     - Increase codeword_dim (more dimensions for codes to spread)")
            issues.append(f"     - Check if input features are inherently sparse")
        elif avg_sparsity > 50:
            issues.append(f"  ⚠️  EMBEDDING SPARSITY: {avg_sparsity:.1f}% of values near zero (moderate concern)")
        
        if issues:
            report_lines.append(f"")
            for issue in issues:
                report_lines.append(issue)
        elif concentration <= 20 and avg_sparsity <= 50 and avg_perplexity >= 50:
            report_lines.append(f"")
            report_lines.append(f"  ✅ HEALTHY: Good code distribution, well-utilized embedding space")
        
        return "\n".join(report_lines)
    
    def reset(self):
        """Reset counters for next epoch."""
        # Keep all-time count, reset recent window
        self.recent_code_usage.clear()
        self.all_indices_recent.clear()
        self.embedding_norms.clear()
        self.embedding_means.clear()
        self.embedding_sparsities.clear()
        self.commitment_losses.clear()
        self.batch_perplexities.clear()


class Forge(nn.Module):
    """Forge model: GraphSAGE+ encoder with Vector Quantization for MIP graphs.
    It is designed to learn discrete representations of MIP instances represented as bipartite graphs.
    This class constructs a GraphSAGE-based encoder, optional prediction heads, and a vector quantization module.

    """

    def __init__(self,
                 train_config_yaml: Optional[str] = Constants.default_train_config_yaml,
                 input_dim: Optional[int] = None,
                 hidden_dim: Optional[int] = None,
                 codeword_dim: Optional[int] = None,
                 codebook_size: Optional[int] = None,
                 dropout_ratio: Optional[float] = None,
                 # activation: Optional[Callable] = F.relu,
                 activation: Optional[Callable] = F.gelu,
                 norm_type: Optional[str] = None,
                 lambda_edge: Optional[float] = None,
                 lambda_node: Optional[float] = None,
                 orthogonal_reg_weight: Optional[float] = None,
                 is_eval_mode: Optional[bool] = None):
        """Initialize the Forge models.

            This module adapts ideas from VQ-Graph style architectures for Mixed Integer Programming (MIP) instances.
            It builds a graph embedding with GraphSAGE,
            Optionally, it applies
            prediction heads, and performs vector quantization to obtain a discrete representation that can
            be used for reconstruction and downstream heuristics.

            Parameters
            ----------
            train_config_yaml : Optional[str], default=Constants.default_train_config_yaml
                Path to a YAML configuration file that provides default training and model hyperparameters.
                When provided, values from this file are loaded and used unless explicitly overridden via constructor arguments.
                The path is validated by `_validate_args` and must point to an existing readable file;
                passing `None` will raise a ValueError.
                Typical keys expected in the file include `input_dim`, `hidden_dim`, `codebook_dim`, `dropout_ratio`, and
                other parameters documented below.
            input_dim : int, default=10
                Dimensionality of the raw node features provided in `feats` during `forward`.
                Also called feat_dim in some contexts.
                If `input_dim < hidden_dim` the models internally projects to `hidden_dim`
                    (stored as `updated_input_dim`) to allow a wider first hidden representation;
                Otherwise, it keeps the original size.
                This affects the width of all subsequent layers and the quantizer.
            hidden_dim : int, default=1024
                Target hidden embedding size for GraphSAGE layers and subsequent linear layers.
                Acts as the working dimensionality for message passing.
                Larger values increase models capacity and decoder parameter count,
                potentially improving reconstruction at the cost of memory.
            codeword_dim : int, default=1024
                Dimensionality of each code vector in the vector quantization (VQ) codebook(s).
                Can be set lower than `hidden_dim` to encourage compression, or equal for lossless capacity.
                Impacts the expressiveness of discrete embeddings used for mip vector representations.
            codebook_size : int, default=5000
                Number of discrete codes available to the VQ module(s).
                Larger sizes increase capacity for representing structural diversity in MIP graphs.
                Smaller sizes enforce stronger sharing and can improve generalization,
                    but may hurt fine-grained reconstruction.
                Also, determines the length of the distribution vector returned by `mip_to_embeddings`.
            dropout_ratio : float, default=0.4
                Dropout applied after major transformation blocks (GraphSAGE layers and linear layer).
                Higher values regularize more aggressively.
                Lower values risk overfitting large training corpora of MIP instances.
            activation : callable, default=torch.nn.functional.gelu
                Non-linearity used inside SAGEConv layers.
                Provides smooth gradients for all dimensions, improving VQ centering and codebook utilization.
            norm_type : str, default="none"
                Type of optional additional normalization applied via `self.norms` (if populated outside this snippet).
                When set to values other than "none", an auxiliary normalization module is expected at index 0,
                refining stability across instances. Setting "none" skips that step.
            lambda_edge : float, default=1
                Weight scaling the edge reconstruction portion of the unsupervised loss.
                During training, the implementation alternates emphasizing edges vs. nodes,
                    by swapping this with `lamb_node`.
                Increasing `lamb_edge` pushes the quantizer to better reproduce bipartite adjacency patterns.
            lambda_node : float, default=1
                Weight scaling node feature reconstruction loss.
                Larger values bias learning toward accurate feature decoding rather than structural edge patterns.
            orthogonal_reg_weight : float, default=0.0
                Strength of orthogonal regularization passed to the VQ module(s).
                Non-zero values push code vectors toward mutual orthogonality,
                    reducing redundancy and encouraging diverse discrete assignments.
                Typically small (e.g. 0.1–0.5) if used.
            is_eval_mode : bool, default=False
                If True, the forward pass omits reconstruction loss computation,
                    skips adjacency matrix extraction and loss terms for faster inference and hint generation.
                Set to False during training so losses are available.

            Architectural / Algorithmic Interplay
            -------------------------------------
            - (input_dim, hidden_dim) jointly define `updated_input_dim`,
                the base width feeding all decoders and quantizers;
                widening hidden_dim without increasing codebook_size may create under-utilized discrete capacity.
            - codebook_size & codebook_dim trade off discrete resolution vs. memory;
                large codebook_size with small codebook_dim yields many compact codes;
                smaller size with large dim yields fewer but richer codes.
            - has_separate_codebooks switches from a shared latent space, (encouraging
                unified encoding of node/edge information) to specialized spaces (potentially
                better reconstruction when node features and edge structure differ in statistical properties).
            - lamb_edge / lamb_node dynamically steer optimization toward structural and feature faithfulness;
                periodic alternation (implemented in `train_unsupervised`) prevents one modality from dominating.
            - orthogonal_reg_weight influences how distinct discrete codes become;
                higher values help avoid degenerate clustering where many embeddings map to near-identical codes.
            - has_variable_proba_head / has_integral_gap_head gate auxiliary supervisory signals;
                enabling them adds tasks that can regularize embeddings beyond pure reconstruction.
            - is_eval_mode allows deterministic embedding / code assignment without incurring the cost of
                computing reconstruction losses and adjacency transforms, important for downstream MIP heuristics.
            - has_integral_gap_head enables a cut prediction head (`integral_gap_layer`) for,
                    LP gap / cut ratio estimation tasks.
                Used in `mip_to_lp_cut` workflows.
                When active, an additional scalar per variable is produced.
            has_variable_proba_head enables a probability prediction head (`variable_proba_layer`) for,
                    variable membership solution likelihood tasks (BCE loss).
                Activating this adds parameters and changes the forward outputs,
                    appending probability tensors to `h_list`.
                Required for warm-start and triplet training phases.
        """

        super().__init__()

        self._validate_args(train_config_yaml)

        # Store the config file path for later use (e.g., by SATProcessor)
        self.train_config_yaml_path: str = train_config_yaml

        # Read all configs from the given file
        with open(train_config_yaml, 'r') as f:
            config = yaml.safe_load(f)

        # Compute input_dim from feature selection if not explicitly provided
        # and if feature selection is configured
        computed_input_dim = input_dim
        if computed_input_dim is None and config.get('sat_variable_features') is not None:
            # Feature selection is configured, compute input_dim from it
            from forge.utils import get_sat_feature_config
            _, _, computed_input_dim = get_sat_feature_config(config)

        # Store feature selection for SATProcessor to use later
        from forge.utils import get_sat_feature_config
        self.selected_sat_var_features, self.selected_sat_clause_features, _ = get_sat_feature_config(config)

        # Default to config values, but overwrite if a value is given
        self.input_dim: int = overwrite_if_given(config.get('input_dim'), computed_input_dim)
        self.hidden_dim: int = overwrite_if_given(config.get('hidden_dim'), hidden_dim)
        self.codeword_dim: int = overwrite_if_given(config.get('codeword_dim'), codeword_dim)
        self.codebook_size: int = overwrite_if_given(config.get('codebook_size'), codebook_size)
        self.dropout_ratio: float = overwrite_if_given(config.get('dropout_ratio'), dropout_ratio)
        self.activation: Callable = activation
        self.norm_type: str = overwrite_if_given(config.get('norm_type'), norm_type)
        self.lambda_edge: float = overwrite_if_given(config.get('lambda_edge'), lambda_edge)
        self.lambda_node: float = overwrite_if_given(config.get('lambda_node'), lambda_node)
        self.orthogonal_reg_weight: float = float(overwrite_if_given(config.get('orthogonal_reg_weight'),
                                                                      orthogonal_reg_weight))
        self.is_eval_mode: bool = overwrite_if_given(config.get('is_eval_mode'), is_eval_mode)

        # Load additional parameters
        self.graph_sage_aggregation: str = config.get('graph_sage_aggregation')
        self.decoder_edge_dim: int = config.get('decoder_edge_dim')
        self.vq_decay: float = float(config.get('vq_decay'))  # cast to float to ensure it's not a string
        self.vq_commitment_weight: float = float(config.get('vq_commitment_weight'))  # cast to float to ensure it's not a string
        self.vq_is_cosine_sim: bool = config.get('vq_is_cosine_sim')
        self.vq_stochastic_sample_codes: bool = config.get('vq_stochastic_sample_codes', False)  # Enable stochastic sampling to revive dead codes
        self.vq_sample_codebook_temp: float = float(config.get('vq_sample_codebook_temp', 1.0))  # Temperature for sampling

        # Load default training parameters
        self.epochs: int = config.get('epochs')
        self.steps_per_instance: int = config.get('steps_per_instance')
        self.learning_rate: float = float(config.get('learning_rate'))  # cast 1e-4 as float! not scientific str
        self.weight_decay: float = float(config.get('weight_decay'))  # cast 1e-4 as float! not scientific str
        self.max_graph_nodes: int = config.get('max_graph_nodes')
        self.adj_block_size: int = config.get('adj_block_size')  # Block size for adjacency reconstruction loss

        # Load integral gap parameters
        self.integral_gap_safety_eps: float = float(config.get('integral_gap_safety_eps'))  # cast to float to ensure it's not a string

        # Load seed
        self.seed: int = config.get('seed')

        # Load scheduler configuration
        self.use_cosine_warmup_scheduler: bool = config.get('use_cosine_warmup_scheduler', False)
        self.warmup_epochs: int = config.get('warmup_epochs', 1)
        self.min_lr: float = float(config.get('min_lr', 1e-6))

        # Initialize without downstream heads. load_model() can set these later.
        self.has_integral_gap_head: bool = False
        self.has_variable_proba_head: bool = False

        # Update input dim if needed
        self.updated_input_dim: int = max(self.input_dim, self.hidden_dim)

        # Set fields based on input parameters
        self.dropout = nn.Dropout(float(self.dropout_ratio))

        # Forge is initially not trained
        self.is_trained = False

        # Create layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # Populate normalization layers if norm_type is not "none"
        # this is going to be dead code if layernorm thing works later on, 
        # but keeping it here for now for backward compatibility with older models that expect a norm layer at index 0
        if self.norm_type != "none":
            if self.norm_type == "pre":
                # Pre-norm: normalize raw input features before GraphSAGE
                self.norms.append(nn.LayerNorm(self.input_dim))
            elif self.norm_type in ("layer", "post"):
                # Post-norm: normalize after GraphSAGE outputs (hidden_dim)
                self.norms.append(nn.LayerNorm(self.updated_input_dim))
            else:
                # Default fallback
                self.norms.append(nn.LayerNorm(self.updated_input_dim))

        # Set device, if GPU is available use it
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Modified GraphSAGE to accept edge weights
        self.graph_layer_1 = EdgeWeightedSAGEConv(self.input_dim, self.updated_input_dim,
                                                  aggr=self.graph_sage_aggregation)
        self.graph_layer_2 = EdgeWeightedSAGEConv(self.updated_input_dim, self.updated_input_dim,
                                                  aggr=self.graph_sage_aggregation)

        # Linear layers
        self.linear = nn.Linear(self.updated_input_dim, self.updated_input_dim)
        self.integral_gap_layer = nn.Linear(self.updated_input_dim, 1) if self.has_integral_gap_head else None
        self.variable_proba_layer = nn.Linear(self.updated_input_dim, 1) if self.has_variable_proba_head else None

        # REFACTORED: Removed BatchNorm layers to avoid "normalization war" with LayerNorm
        # Mixing LayerNorm + BatchNorm creates gradient chattering conflicts in small GNNs
        # Using LayerNorm consistently instead (batch-size independent, better for variable-sized graphs)
        # OLD CODE (COMMENTED OUT):
        # self.bn1 = nn.BatchNorm1d(self.updated_input_dim)
        # self.bn2 = nn.BatchNorm1d(self.updated_input_dim)
        # self.bn3 = nn.BatchNorm1d(self.updated_input_dim)

        # NEW: Explicit normalization for stable VQ codebook assignment
        # post_gnn_norm: Normalize after graph convolutions (controls variance for next layer)
        # NOTE (April 2026): Removed pre_vq_norm (LayerNorm right before VQ) as it was over-centering
        # For cosine similarity VQ, L2 normalization alone is sufficient and preferred
        self.post_gnn_norm = nn.LayerNorm(self.updated_input_dim)

        # FIX (April 2026): Added pre_vq_norm with elementwise_affine=True to allow learning
        # per-dimension scaling factors. This helps stretch histogram spikes and improves
        # codebook utilization by giving the model control over each dimension's magnitude.
        self.pre_vq_norm = nn.LayerNorm(self.updated_input_dim, elementwise_affine=True)

        # Node decoder
        self.decoder_node = nn.Linear(self.updated_input_dim, self.input_dim)

        # Edge decoders. Edges are decoded as product of two matrices
        self.decoder_edge_1 = nn.Linear(self.updated_input_dim, self.decoder_edge_dim)
        self.decoder_edge_2 = nn.Linear(self.updated_input_dim, self.decoder_edge_dim)

        # Vector quantization module
        # RECENT ADDITION (April 2026): Attempted to add threshold_ema_dead_code and reset_cluster_size
        # for automatic dead code resurrection, but these parameters are not supported in the installed
        # version of vector_quantize_pytorch. Using stochastic sampling instead.
        # UPDATED (April 2026): Added stochastic_sample_codes and sample_codebook_temp to revive dead codes
        # via probabilistic sampling with temperature-controlled randomness.
        # FIX (April 2026): Added layernorm_after_project_in=True to normalize after internal projection
        # Since updated_input_dim (1024) != codeword_dim (256), VQ creates an internal linear projection
        # that was blowing up vector magnitudes. LayerNorm after projection keeps vectors stable.
        self.vq = VectorQuantize(dim=self.updated_input_dim,
                                 codebook_size=self.codebook_size,
                                 codebook_dim=self.codeword_dim,
                                 decay=self.vq_decay,
                                 commitment_weight=self.vq_commitment_weight,
                                 use_cosine_sim=self.vq_is_cosine_sim,
                                 orthogonal_reg_weight=self.orthogonal_reg_weight,
                                 stochastic_sample_codes=self.vq_stochastic_sample_codes,
                                 sample_codebook_temp=self.vq_sample_codebook_temp,
                                 layernorm_after_project_in=True)
        
        # Temperature scaling for SAT prediction head (learned during finetuning)
        self.sat_temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, feature_tensor: torch.Tensor,
                num_cons: int, num_vars: int,
                edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor],
                adj_gpu: Optional[torch.Tensor] = None) \
            -> Tuple[List[torch.Tensor], torch.Tensor, Union[torch.Tensor, int], torch.Tensor, Union[
                torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass of the Forge models.

        Parameters
        ----------
        feature_tensor : torch.Tensor
            Node feature matrix of shape (num_cons + num_vars, feat_dim=10 zero padded features).
            Provided externally by the graph construction utilities.
            This will be transformed through GraphSAGE + linear layer.
        num_cons : int
            Number of constraint nodes (prefix of `feature_tensor`).
            Used to slice embeddings when computing bipartite adjacency reconstruction and separating outputs.
        num_vars : int
            Number of variable nodes (suffix of `feature_tensor`).
            Used for variable_proba and integrality_gap heads in downstream.
        edge_index : torch.LongTensor
            PyG COO connectivity with shape `(2, num_edges)`.
            First row = sources, second row = targets.
            edge_index[0] contains source node indices, edge_index[1] contains target indices.
            Node ordering must match `feature_tensor` (constraints first, then variables).
            Represents the bipartite structure between constraints and variables.
        edge_weight : Optional[torch.FloatTensor]
            1-D tensor of edge weights with length `num_edges`. Can be `None`.
            NOTE: `torch_geometric.nn.SAGEConv` does not accept an `edge_weight` argument.
            If you need weighted message passing, use a conv that supports weights (e.g. `GCNConv`)
            or implement a custom `MessagePassing`.
            In this codebase `edge_weight` is used for adjacency reconstruction / loss, not in `SAGEConv`.
            PyG `SAGEConv` does not consume `edge_weight`; Use a weighted convolution layer if message passing
            should be coefficient-aware.
        adj_gpu : Optional[torch.Tensor], default=None
            Pre-computed adjacency matrix on GPU of shape (num_nodes, num_nodes).
            When provided and not in eval mode, this matrix is used directly for edge reconstruction loss,
            avoiding repeated CPU construction.
            If None and not in eval mode, the adjacency matrix will be built on CPU (slower).

        Returns
        -------
        h_list : List[torch.Tensor]
            Ordered collection of intermediate / final representations.
            Layout (when `separate_codebooks=False` and heads enabled) is:
                0: `h`        - dense embedding after GraphSAGE + linear block (shape: [N, hidden])
                1: `quantized`- quantized latent before decoders (same shape as above)
                2: `quantized_node` - node feature reconstruction logits (shape: [N, 10])
                3: `quantized_edge_1` - edge factor matrix A (shape: [N, 32])
                4: `quantized_edge_2` - edge factor matrix B (shape: [N, 32])
                5: `prob` (optional) - variable membership probabilities (shape: [N, 1])
                6: `cut`  (optional) - variable-level cut score predictions (shape: [N, 1])
        h : torch.Tensor
            Embedding after the linear + activation block (same as h_list[0]).
            Returned separately for convenience in downstream tasks that expect a single dense representation.
        loss : torch.Tensor | int
            Scalar reconstruction + commitment loss (and edge positive emphasis term) if `eval_only=False`;
            set to -1 when `eval_only=True` to signal inference-only mode.
        indices : torch.Tensor
            Code assignments for each node in the graph.
            Shape: [N, 1] where N is the number of nodes.
        codebook : torch.Tensor | Tuple[torch.Tensor, torch.Tensor]
            If `separate_codebooks=False`, a single codebook tensor of shape (codebook_size, codebook_dim).
            Otherwise, a tuple `(codebook_node, codebook_edge)` each with that shape.

        Notes
        -----
        - Adjacency reconstruction uses two low-rank factor matrices (`quantized_edge_1`, `quantized_edge_2`)
            to approximate bipartite edges via (A A^T)(B B^T)^T then min-max rescale.
        - Feature and edge reconstruction losses are scaled by `lamb_node` and `lamb_edge`
            allowing alternating emphasis during pretraining.
        - Commitment and (optionally) orthogonality regularization flow from the VectorQuantize
          modules to encourage discrete, non-redundant code usage.
        - Probability and cut heads operate on the quantized latent, not the pre-quantization `h`.
        """

        # Input
        h = feature_tensor

        # List to hold intermediate layers
        h_list = []

        # REFACTORED FORWARD PASS: LayerNorm-only pipeline for stable VQ assignment
        # Removed: self.norms[0] pre/post norm (kept for backward compat but not used)
        # Removed: all bn1, bn2, bn3 BatchNorm calls (causes gradient chattering)
        # Added: explicit post_gnn_norm and pre_vq_norm for controlled variance
        
        # # OLD CODE (COMMENTED OUT - kept for reference):
        # # Apply pre-norm if enabled (before GraphSAGE)
        # if self.norm_type == "pre":
        #     h = self.norms[0](h)

        # GraphSAGE Layer 1
        h = self.graph_layer_1(h, edge_index, edge_weight=edge_weight)
        h = self.activation(h)  # PyG needs explicit activation
        # # OLD: h = self.bn1(h)  # CAUSES CONFLICT WITH LAYERNORM
        # DISABLED (April 2026): Removed post-GNN normalization to allow natural magnitude growth
        # Post-norm was clipping/standardizing magnitudes, contributing to 95% sparsity
        # # h = self.post_gnn_norm(h)
        # # OLD: if self.norm_type in ("layer", "post"):
        # #     h = self.norms[0](h)
        h = self.dropout(h)

        # GraphSAGE Layer 2
        h = self.graph_layer_2(h, edge_index, edge_weight=edge_weight)
        h = self.activation(h)  # PyG needs explicit activation
        # # OLD: h = self.bn2(h)  # REMOVED
        h = self.dropout(h)

        # Linear Layer
        h = self.linear(h)
        # h = F.relu(h)
        h = F.gelu(h)
        # # OLD: h = self.bn3(h)  # REMOVED
        h = self.dropout(h)
        
        # MOVED (April 2026): Apply pre_vq_norm BEFORE VQ to allow dimension spread
        # This provides a stable learned transformation without forced L2 norm
        h = self.pre_vq_norm(h)  # Learnable LayerNorm for stable scaling

        # Store output at this stage into h_list
        # This is going to be our "embedding" of the input graph
        h_list.append(h)

        # RE-ENABLED (May 2026): L2 normalization before VQ
        # Previous sparsity issue was likely due to over-constrained orthogonal_reg (1e-5 → 1e-2)
        # Combined with aggressive commitment_weight. Now using more conservative params:
        # - vq_decay: 0.99 (slower EMA, more stable)
        # - commitment_weight: 1.0 (force encoder to track codebook)
        # - orthogonal_reg_weight: 1e-2 (light repulsion)
        # L2 normalization ensures unit-norm search space for cosine similarity VQ
        h_normalized_for_vq = F.normalize(h, p=2, dim=-1)
        
        # The same "embedding" is then passed into the vector quantizer below
        quantized, indices, commit_loss = self.vq(h_normalized_for_vq)
        codebook = self.vq.codebook  # or from the forward output
        
        # CRITICAL: Ensure quantized output stays unit norm (in case VQ's internal projection drifts norms)
        # This is the output that will be used by all downstream decoders and heads
        quantized = F.normalize(quantized, p=2, dim=-1)

        quantized_node = self.decoder_node(quantized)
        quantized_edge_1 = self.decoder_edge_1(quantized)
        quantized_edge_2 = self.decoder_edge_2(quantized)

        # The "embedding" is passed into the proba head and gap head below
        variable_proba_head = None
        if self.has_variable_proba_head:
            variable_proba_head = F.sigmoid(self.variable_proba_layer(quantized))

        integral_gap_head = None
        if self.has_integral_gap_head:
            # Use linear layer only instead of sigmoid since gap ratio can be > 1
            integral_gap_head = self.integral_gap_layer(quantized)

        # Training
        feature_rec_loss = None
        edge_rec_loss = None
        if not self.is_eval_mode:

            # Use pre-computed adjacency if provided, otherwise build on CPU (fallback)
            if adj_gpu is not None:
                adj = adj_gpu
                # Reconstruction Loss (other losses are calculated in training code)
                feature_rec_loss = self.lambda_node * F.mse_loss(feature_tensor, quantized_node)

                # Edge reconstruction loss on bipartite block, computed in blocks to allow running larger graphs
                edge_rec_loss = blockwise_loss(quantized_edge_1,
                                               quantized_edge_2,
                                               adj,
                                               num_cons,
                                               lambda_edge=self.lambda_edge,
                                               batch_size=self.adj_block_size)
            else:
                # Convert PyG edge_index to dense adjacency matrix on CPU
                # BUT: Only if graph is small enough to avoid OOM (< 100k nodes)
                num_nodes = num_cons + num_vars
                
                if num_nodes > 100000:
                    # Large graph: skip edge loss (adjacency too big to allocate)
                    # This avoids OOM when allocating NxN matrix
                    feature_rec_loss = self.lambda_node * F.mse_loss(feature_tensor, quantized_node)
                    edge_rec_loss = torch.tensor(0.0, device=quantized_edge_1.device)
                else:
                    # Small enough graph: safe to allocate dense adjacency
                    adj = torch.zeros((num_nodes, num_nodes), device="cpu")

                    ei = edge_index.to("cpu")
                    if edge_weight is not None:
                        ew = edge_weight.to("cpu")
                        adj[ei[0], ei[1]] = ew
                    else:
                        adj[ei[0], ei[1]] = 1.0

                    # Reconstruction Loss (other losses are calculated in training code)
                    feature_rec_loss = self.lambda_node * F.mse_loss(feature_tensor, quantized_node)

                    # Edge reconstruction loss on bipartite block, computed in blocks to allow running larger graphs
                    edge_rec_loss = blockwise_loss(quantized_edge_1,
                                                   quantized_edge_2,
                                                   adj,
                                                   num_cons,
                                                   lambda_edge=self.lambda_edge,
                                                   batch_size=self.adj_block_size)

        h_list.append(quantized)
        h_list.append(quantized_node)
        h_list.append(quantized_edge_1)
        h_list.append(quantized_edge_2)
        if self.has_variable_proba_head:
            h_list.append(variable_proba_head)
        if self.has_integral_gap_head:
            h_list.append(integral_gap_head)
        
        # Add SAT satisfiability prediction head if present (GGNN-style aggregation)
        # ====================================================================
        # IMPROVED: Node-level prediction + logit aggregation (not embedding aggregation)
        # 
        # Instead of aggregating embeddings then predicting:
        #   graph_emb = mean(embeddings) -> logit
        # 
        # We now process each node independently, then aggregate predictions:
        #   node_logits = MLP(embeddings) -> mean(node_logits)
        # 
        # This is more expressive because:
        # - Each node independently contributes to the decision
        # - More capacity to learn discriminative features
        # - Better gradient flow (more paths for gradients to backprop)
        # - Matches modern GNN practice (node classification aggregated to graph-level)
        sat_satisfiability_head = None
        if hasattr(self, 'sat_satisfiability_layer') and self.has_sat_satisfiability_head:
            # Apply MLP to each node's embedding independently
            # OLD: node_logits = self.sat_satisfiability_layer(quantized)
            # NEW: Add residual connection to bypass VQ bottleneck
            # This gives each node access to both the VQ representation AND original features
            node_logits = self.sat_satisfiability_layer(quantized + h)  # [num_nodes, 1]
            # Aggregate node logits via mean pooling
            sat_satisfiability_head = torch.mean(node_logits, dim=0, keepdim=True)  # [1, 1]
            h_list.append(sat_satisfiability_head)

        if not self.is_eval_mode:
            loss = feature_rec_loss + edge_rec_loss + commit_loss
        else:
            loss = -1

        return h_list, h, loss, indices, codebook

    def load_model(self, input_forge_pkl, model_type=Constants.FORGE_PRE_TRAIN, strict=False):
        """Load a fully-trained Forge model (same architecture as checkpoint).
        
        NOTE: This method expects the model architecture to match the checkpoint exactly.
        For transfer learning across different architectures (e.g., different codebook sizes),
        use `load_weights_from_pretrained()` instead.

        Parameters
        ----------
        input_forge_pkl : str
            Path to the pre-trained Forge model pickle file to load.
        model_type : str
            The type of task the model was trained for (default: FORGE_PRE_TRAIN).
            Options: FORGE_PRE_TRAIN, FORGE_FINE_TUNE_INTEGRAL_GAP, FORGE_FINE_TUNE_VARIABLE_PROBA, FORGE_FINE_TUNE_SAT
        strict : bool
            If True, the state_dict keys must match exactly. If False (default), allows loading
            checkpoints with missing or extra keys (for backward compatibility with older models).
        """
        if self.is_trained:
            print("Warning: Forge model is already trained, NOT loading weights, quitting!!")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        if model_type == Constants.FORGE_PRE_TRAIN:
            self.has_integral_gap_head = False
            self.has_variable_proba_head = False
            self.has_sat_satisfiability_head = False
        elif model_type == Constants.FORGE_FINE_TUNE_INTEGRAL_GAP:
            self.has_integral_gap_head = True
            self.has_variable_proba_head = False
            self.has_sat_satisfiability_head = False
        elif model_type == Constants.FORGE_FINE_TUNE_VARIABLE_PROBA:
            self.has_integral_gap_head = False
            self.has_variable_proba_head = True
            self.has_sat_satisfiability_head = False
        elif model_type == Constants.FORGE_FINE_TUNE_SAT:
            self.has_integral_gap_head = False
            self.has_variable_proba_head = False
            self.has_sat_satisfiability_head = True
            # Ensure SAT satisfiability layer exists before loading weights
            if not hasattr(self, 'sat_satisfiability_layer'):
                self.sat_satisfiability_layer = nn.Sequential(
                    nn.LayerNorm(self.updated_input_dim),
                    nn.Linear(self.updated_input_dim, self.updated_input_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.updated_input_dim, self.updated_input_dim // 2),
                    nn.GELU(),
                    nn.Linear(self.updated_input_dim // 2, 1)
                ).to(device)

        self.load_state_dict(torch.load(input_forge_pkl, map_location=device), strict=strict)
        self.is_trained = True

    def load_weights_from_pretrained(self, input_forge_pkl: str) -> None:
        """Load pre-trained weights from a Forge model pickle, handling dimension mismatches gracefully.

        This method automatically detects if the checkpoint has matching dimensions:
        - If dimensions MATCH: Loads ALL weights (full transfer learning)
        - If dimensions MISMATCH: Loads only compatible encoder layers (selective transfer learning)

        This allows initializing encoder weights from a pre-trained MIP model while keeping
        `is_trained=False`, so the model can continue training on a different task
        (e.g., SAT pre-training using MIP-pre-trained weights with different codebook_size).

        Parameters
        ----------
        input_forge_pkl : str
            Path to the pre-trained Forge model pickle file to load weights from.

        Returns
        -------
        None
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        print(f"  Loading state_dict from: {input_forge_pkl}")
        state_dict = torch.load(input_forge_pkl, map_location=device)
        print(f"  Loaded state_dict with {len(state_dict)} keys from checkpoint")
        
        # First pass: detect if checkpoint architecture matches current model
        current_state = self.state_dict()
        has_mismatches = False
        mismatches = []
        
        for key, value in state_dict.items():
            if key in current_state:
                if current_state[key].shape != value.shape:
                    has_mismatches = True
                    mismatches.append((key, value.shape, current_state[key].shape))
        
        if not has_mismatches and len(state_dict) == len(current_state):
            # Perfect match: load everything
            print(f"  Checkpoint architecture matches current model exactly!")
            print(f"  Loading all {len(state_dict)} parameter groups...")
            try:
                self.load_state_dict(state_dict, strict=True)
                print(f"  Successfully loaded ALL weights from checkpoint")
                return
            except RuntimeError as e:
                print(f"  Full load failed ({str(e)[:100]}...), falling back to selective loading")
                has_mismatches = True
        
        if has_mismatches:
            print(f"  Checkpoint architecture differs from current model (found {len(mismatches)} mismatches)")
            print(f"  Using selective transfer learning (encoder layers only)...")
            
            # Load only compatible layers (encoder), skip codebook/decoders with different dimensions
            compatible_state = {}
            skipped_keys = []
            size_mismatch_keys = []
            
            for key, value in state_dict.items():
                # SKIP all VQ-related layers AND buffers (codebook, EMA buffers can have different dimensions between models)
                # This includes codebook, embed_avg, cluster_size, and other internal VQ state that can cause size mismatches
                if 'vq.' in key or 'codebook' in key or 'project_in' in key or 'project_out' in key or 'embed_avg' in key or 'cluster_size' in key:
                    skipped_keys.append(key)
                    continue
                    
                # SKIP task-specific heads (can differ between pretraining and fine-tuning)
                if any(head in key for head in ['integral_gap_layer', 'variable_proba_layer', 'sat_satisfiability_layer']):
                    skipped_keys.append(key)
                    continue
                
                # Load encoder layers (graph, linear, batchnorm, activation, dropout, norms)
                if any(prefix in key for prefix in ['graph_layer_', 'linear', 'bn', 'activation', 'dropout', 'norms']):
                    try:
                        # Check if key exists in current model
                        if key not in current_state:
                            skipped_keys.append(key)
                            continue
                        
                        # Check if size matches current model
                        if current_state[key].shape == value.shape:
                            compatible_state[key] = value
                        else:
                            size_mismatch_keys.append((key, value.shape, current_state[key].shape))
                            skipped_keys.append(key)
                    except Exception as e:
                        skipped_keys.append(key)
                else:
                    skipped_keys.append(key)
            
            if skipped_keys:
                print(f"  Skipped {len(skipped_keys)} incompatible/VQ layers:")
                for key in skipped_keys[:3]:  # Show first 3
                    print(f"    - {key}")
                if len(skipped_keys) > 3:
                    print(f"    ... and {len(skipped_keys) - 3} more")
            
            if size_mismatch_keys:
                print(f"  Found {len(size_mismatch_keys)} layers with size mismatches (skipped):")
                for key, old_shape, new_shape in size_mismatch_keys[:3]:
                    print(f"    - {key}: {old_shape} -> {new_shape}")
                if len(size_mismatch_keys) > 3:
                    print(f"    ... and {len(size_mismatch_keys) - 3} more")
            
            # Load only compatible weights (strict=False to ignore missing keys)
            self.load_state_dict(compatible_state, strict=False)
            print(f"  Successfully loaded {len(compatible_state)} compatible encoder layers")
            print(f"  VQ codebook, task heads, and mismatched layers will be trained from scratch")


    def _pretrain(self,
                  input_mipinfo_list: List[MIPInfo],
                  output_forge_pkl: str,
                  output_log_file: Optional[str],
                  epochs: Optional[int] = None,
                  steps_per_instance: Optional[int] = None,
                  learning_rate: Optional[float] = None,
                  weight_decay: Optional[float] = None,
                  max_graph_nodes: Optional[int] = None,
                  rank: int = 0,
                  world_size: int = 1,
                  gpu_memory_fraction: float = 0.8) -> None:
        """Pretrain the Forge model on provided MIP instances. Sets `is_trained` to True upon completion.

        Parameters
        ----------
        input_mipinfo_list : List[MIPInfo]
            List of precomputed `MIPInfo` objects (returned by `MIPProcessor._mip_model_to_mipinfo`).
        output_forge_pkl : str
            Path to save the model `state_dict` after each epoch.
        output_log_file : Optional[str]
            Optional path to append training logs; if `None`, logs are not written to disk.
        epochs : Optional[int], default=None
            Number of outer training epochs; if `None`, uses the config default.
        steps_per_instance : Optional[int], default=None
            Number of optimization steps to run per instance; if `None`, uses the config default.
        learning_rate : Optional[float], default=None
            Learning rate override for the optimizer; if `None`, uses the config default.
        weight_decay : Optional[float], default=None
            Weight decay for the optimizer; if `None`, uses the config default.
        max_graph_nodes : Optional[int], default=None
            Maximum allowed number of nodes (num_cons + num_vars) to be processed on the device.
            Instances exceeding this value will be skipped during training.
            If `None`, the configuration default loaded from the training YAML is used.
            This can be set based on the available GPU memory to avoid out-of-memory errors.
        rank : int, default=0
            Rank of current process in distributed training (0 for single GPU).
        world_size : int, default=1
            Total number of processes in distributed training (1 for single GPU).
            Dataset will be partitioned so each rank processes a different subset.
        gpu_memory_fraction : float, default=0.8
            Target GPU memory usage fraction (0.0 to 1.0). Smart fallback to CPU if exceeded.

        Returns
        -------
        None
        """

        # BALANCED LOAD DISTRIBUTION FOR MULTI-GPU TRAINING
        # Instead of simple round-robin (which causes load imbalance when instances vary in size),
        # we use a greedy load-balancing strategy:
        # 1. Sort all instances by size (num_cons + num_vars) in descending order
        # 2. Assign each instance to the GPU with the least total work assigned so far
        # This ensures each GPU processes roughly equal computational load, preventing NCCL timeouts
        
        # Sort instances by size (largest first)
        sorted_mipinfo_list = sorted(input_mipinfo_list, 
                                     key=lambda x: x.num_cons + x.num_vars, 
                                     reverse=True)
        
        # Greedy load balancing: assign instances to GPUs with least work
        gpu_workloads = [[] for _ in range(world_size)]  # list of instances per GPU
        gpu_total_nodes = [0] * world_size  # total nodes per GPU
        
        for mipinfo in sorted_mipinfo_list:
            instance_size = mipinfo.num_cons + mipinfo.num_vars
            # Assign to GPU with smallest current workload
            min_gpu = gpu_total_nodes.index(min(gpu_total_nodes))
            gpu_workloads[min_gpu].append(mipinfo)
            gpu_total_nodes[min_gpu] += instance_size
        
        # Get instances assigned to this rank
        partitioned_mipinfo_list = gpu_workloads[rank]
        
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"BALANCED LOAD DISTRIBUTION for {world_size} GPUs:")
            print(f"  Total instances: {len(input_mipinfo_list)}")
            for gpu_id in range(world_size):
                print(f"  GPU {gpu_id}: {len(gpu_workloads[gpu_id])} instances, {gpu_total_nodes[gpu_id]:,} total nodes")
            print(f"  Rank {rank} processes: {len(partitioned_mipinfo_list)} instances, {gpu_total_nodes[rank]:,} nodes")
            print(f"{'='*80}\n", flush=True)

        # Put module into training mode (use super to avoid recursion) and move to device
        # .train() retains dropout and batchnorm behavior
        # whereas .eval() for inference remove dropout and freeze batchnorm
        super().train()
        self.to(self.device)
        
        # Log device information
        print(f"\n{'='*80}")
        print(f"PRETRAINING ON DEVICE: {self.device}")
        print(f"{'='*80}")
        if self.device.type == 'cuda':
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"{'='*80}\n", flush=True)

        # let cuDNN autotune for fixed-size ops
        torch.backends.cudnn.benchmark = True

        # Default to config values, but overwrite if a value is given
        epochs = overwrite_if_given(self.epochs, epochs)
        steps_per_instance = overwrite_if_given(self.steps_per_instance, steps_per_instance)
        learning_rate = overwrite_if_given(self.learning_rate, learning_rate)
        weight_decay = overwrite_if_given(self.weight_decay, weight_decay)
        max_graph_nodes = overwrite_if_given(self.max_graph_nodes, max_graph_nodes)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Setup learning rate scheduler if enabled
        scheduler = None
        if self.use_cosine_warmup_scheduler:
            # Linear warmup for first warmup_epochs, then cosine annealing
            scheduler1 = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,  # Start at very low LR
                end_factor=1.0,     # Ramp to full LR over warmup_epochs
                total_iters=self.warmup_epochs
            )
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs - self.warmup_epochs,  # Cosine for remaining epochs
                eta_min=self.min_lr  # Minimum LR (cools down)
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[scheduler1, scheduler2],
                milestones=[self.warmup_epochs]
            )

        # Initialize diagnostics tracker for codebook health monitoring
        diagnostics = VQDiagnostics(self.codebook_size, self.updated_input_dim)
        
        # Initialize dead code revival tracker
        # Track which codes have been used in a rolling window of batches
        dead_code_window = 20  # Check every 20 batches for dead codes
        code_usage_window = {}  # Dict of {code_id: count} over recent batches
        batch_counter = 0

        t = ""
        main_loss_list = []
        for epoch in range(epochs):
            print("<<< Epoch:", epoch)
            epoch_loss_list = []
            epoch_start = time.time()

            # self.lambda_node = 5.0
            # self.lambda_edge = 100.0

            # VERSION 1: Alternating (gentle: only 1.1x ratio to allow convergence)
            if epoch % 2 == 0:
                self.lambda_node = 3.0
                self.lambda_edge = 7.0
            else:
                self.lambda_node = 5.0
                self.lambda_edge = 5.0
            # # VERSION 2
            # if epoch % 2 == 0:
            #     self.lambda_node = 5.5    # was 7
            #     self.lambda_edge = 4.5    # was 3
            # else:
            #     self.lambda_node = 4.5
            #     self.lambda_edge = 5.5

            # VERSION 2: Curriculum learning - start balanced, gradually shift focus
            # progress = epoch / epochs  # 0→1
            # base_weight = 5.5
            # variation = 2.0 * progress
            
            # if epoch % 2 == 0:
            #     self.lambda_node = base_weight + variation
            #     self.lambda_edge = base_weight - variation
            # else:
            #     self.lambda_node = base_weight - variation
            #     self.lambda_edge = base_weight + variation

            # MIP instances in dataset - process one at a time (correct for variable-sized graphs)
            for idx in range(len(partitioned_mipinfo_list)):

                mipinfo = partitioned_mipinfo_list[idx]

                # Get number of nodes (instances already filtered by max_graph_nodes in pipeline.py)
                # Handle both MIPInfo (num_cons) and SATInfo (num_clauses) objects
                num_cons_or_clauses = getattr(mipinfo, 'num_cons', None) or getattr(mipinfo, 'num_clauses', None)
                num_nodes = num_cons_or_clauses + mipinfo.num_vars

                # Push to device before the for-loop below
                # Check GPU memory before moving data, fallback to CPU if needed
                use_gpu = self.device.type == 'cuda'
                move_device = self.device
                
                if use_gpu:
                    # Get GPU memory stats
                    gpu_max_memory = torch.cuda.get_device_properties(self.device).total_memory
                    gpu_allocated = torch.cuda.memory_allocated(self.device)
                    gpu_used_fraction = gpu_allocated / gpu_max_memory
                    
                    # Estimate memory needed (rough estimate: ~200 bytes per node for tensors)
                    estimated_memory_needed = num_nodes * 200
                    projected_gpu_fraction = (gpu_allocated + estimated_memory_needed) / gpu_max_memory
                    
                    # If projected usage exceeds target, use CPU instead
                    if projected_gpu_fraction > gpu_memory_fraction:
                        move_device = torch.device('cpu')
                        if idx % 50 == 0:
                            print(f"  GPU memory {gpu_used_fraction:.1%} exceeds target {gpu_memory_fraction:.1%}, "
                                  f"moving to CPU for instance: {mipinfo.instance_name}")

                # Push to device before the for-loop below
                # Use non-blocking transfers to overlap compute and memory operations:
                features = mipinfo.feature_tensor.to(move_device, non_blocking=True)
                edge_index = mipinfo.edge_index.to(move_device, non_blocking=True)
                edge_weight = mipinfo.edge_weight.to(move_device, non_blocking=True)

                # Don't pre-compute adjacency on GPU for large graphs (can cause OOM)
                # Let the forward() method handle it intelligently:
                # - For large graphs (>50k nodes): skip edge loss
                # - For small graphs: compute on CPU as needed
                adj_gpu = None

                # Train on this instance for specified steps
                instance_loss_list = []
                for step in range(steps_per_instance):
                    # zero gradients before forward (use set_to_none for speed)
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Compute loss and prediction
                    h_list, logits, loss, indices, codebook_ = self.forward(features,
                                                                            num_cons_or_clauses, mipinfo.num_vars,
                                                                            edge_index, edge_weight, adj_gpu)
                    
                    # Track VQ diagnostics (code usage, embedding health, perplexity)
                    # Use total loss as the commitment loss metric (includes VQ commitment component)
                    with torch.no_grad():
                        loss_scalar = loss.item() if isinstance(loss, torch.Tensor) else loss
                        # FIX: Track the quantized embedding (unit-norm from codebook) not the pre-normalized h
                        stats = diagnostics.track_batch(indices, h_list[1], torch.tensor(loss_scalar))
                        
                        # Keep diagnostics aligned with runtime codebook size in case VQ internals change.
                        runtime_codebook_size = int(self.vq.codebook.data.shape[0])
                        if diagnostics.codebook_size != runtime_codebook_size:
                            diagnostics.codebook_size = runtime_codebook_size

                        # Log utilization at start and end of instance
                        if step == 0 or step == steps_per_instance - 1:
                            if idx % 50 == 0:
                                print(f"    Step {step}/{steps_per_instance}: Codes {stats['codes_used_recent']}/{runtime_codebook_size} "
                                      f"({stats['utilization_pct_recent']:.1f}%), Concentration {stats['top_3_concentration']:.0f}%, "
                                      f"Perplexity {stats['perplexity']:.1f}")
                        
                        # RANDOM RESTARTS: Revive dead codes by re-initializing them
                        # Track code usage in a rolling window, periodically reset unused codes
                        batch_counter += 1
                        
                        # Count which codes were used in this batch
                        indices_np = indices.detach().cpu().numpy().flatten()
                        codes_used_this_batch = set(indices_np)
                        
                        # Update rolling window of code usage
                        for code_id in codes_used_this_batch:
                            if code_id not in code_usage_window:
                                code_usage_window[code_id] = 0
                            code_usage_window[code_id] += 1
                        
                        # Every N batches, check for and revive dead codes
                        if batch_counter % dead_code_window == 0:
                            # SYNCHRONIZE code usage across all GPUs in distributed training
                            # This ensures dead code detection is consistent across all processes
                            if dist.is_available() and dist.is_initialized():
                                # Convert dict to tensor for all_reduce
                                usage_tensor = torch.zeros(self.codebook_size, device=self.device, dtype=torch.float32)
                                for code_id, count in code_usage_window.items():
                                    usage_tensor[code_id] = float(count)
                                
                                # Sum usage counts across all GPUs
                                dist.all_reduce(usage_tensor, op=dist.ReduceOp.SUM)
                                
                                # Convert back to dict
                                code_usage_window.clear()
                                for code_id in range(self.codebook_size):
                                    count = usage_tensor[code_id].item()
                                    if count > 0:
                                        code_usage_window[code_id] = int(count)
                            
                            dead_codes = []
                            for code_id in range(self.codebook_size):
                                if code_usage_window.get(code_id, 0) == 0:
                                    dead_codes.append(code_id)
                            
                            # If we found dead codes, revive them
                            if len(dead_codes) > 0:
                                # Get encoder outputs (pre-VQ embeddings) from current batch
                                h_encoder = h_list[0]  # [num_nodes, hidden_dim], unit-norm L2 normalized
                                
                                # Randomly select num_dead_code encoder outputs to reinitialize dead codes
                                if len(h_encoder) > 0:
                                    num_dead = len(dead_codes)
                                    random_indices = torch.randperm(len(h_encoder), device=h_encoder.device)[:num_dead]
                                    random_encodings = h_encoder[random_indices]  # [num_dead, hidden_dim]
                                    
                                    # Update VQ codebook with revived codes
                                    for i, code_id in enumerate(dead_codes):
                                        self.vq.codebook.data[code_id] = random_encodings[i]
                                    
                                    if idx % 50 == 0:
                                        print(f"    [DEAD CODE REVIVAL] Revived {len(dead_codes)} codes with random encoder outputs @ batch {batch_counter}")
                            
                            # Reset rolling window for next check window
                            code_usage_window.clear()
                    
                    instance_loss_list.append(loss.item())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    # explicitly delete large temporaries to free memory faster
                    del h_list, logits, loss, indices, codebook_

                # Explicitly delete instance tensors after all steps complete
                del features, edge_index, edge_weight, adj_gpu

                # End of instance steps, add average instance loss to epoch loss
                avg_instance_loss = float(np.mean(instance_loss_list)) if len(instance_loss_list) > 0 else 0.0
                epoch_loss_list.append(avg_instance_loss)

                if idx % 50 == 0:
                    mem_allocated = torch.cuda.memory_allocated() / 1e9 if self.device.type == 'cuda' else 0
                    print("Epoch,", epoch,
                          ", Idx,", idx,
                          ", Avg. Instance Loss,", np.round(avg_instance_loss, 3),
                          ", Avg. Epoch Loss,", np.round(np.mean(epoch_loss_list), 3),
                          ", Current Time,", np.round(time.time() - epoch_start, 3),
                          ", GPU Mem (GB):", np.round(mem_allocated, 2),
                          ", ", mipinfo.instance_name, flush=True)

                # Only do garbage collection every 10 instances or for large instances to avoid overhead
                # For small instances, GC overhead dominates compute time
                if idx % 10 == 0 and idx > 0:
                    gc.collect()
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()

            # End of epoch - do final cleanup
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            main_loss_list.append(np.round(np.mean(epoch_loss_list), 3))

            epoch_summary = (">>> DONE! Epoch, " + str(epoch) +
                             " , Avg. Epoch Loss, " + str(np.round(np.mean(epoch_loss_list), 3)) +
                             " , +/-," + str(np.round(np.std(epoch_loss_list), 3)) +
                             " , Epoch Time, " + str(np.round(time.time() - epoch_start, 3)) +
                             " , Avg. Main Loss, " + str(np.round(np.mean(main_loss_list), 3)))
            print(epoch_summary)
            t += epoch_summary + "\n"
            
            # Print codebook diagnostics at epoch end
            if rank == 0:
                # DEBUG: Check actual codebook parameter norms BEFORE diagnostics
                cb_norms = torch.norm(self.vq.codebook.data, p=2, dim=-1)
                cb_mean_norm = cb_norms.mean().item()
                cb_median_norm = cb_norms.median().item()
                cb_std_norm = cb_norms.std().item()
                cb_min_norm = cb_norms.min().item()
                cb_max_norm = cb_norms.max().item()
                print(f"  [ACTUAL CODEBOOK NORMS] mean={cb_mean_norm:.6f}, median={cb_median_norm:.6f}, std={cb_std_norm:.6f}, min={cb_min_norm:.6f}, max={cb_max_norm:.6f}")
                
                diag_report = diagnostics.report(f"EPOCH {epoch} END")
                print(diag_report)
                t += diag_report + "\n"
                diagnostics.reset()  # Reset for next epoch
            
            # Step learning rate scheduler if enabled
            if scheduler is not None:
                current_lr = optimizer.param_groups[0]['lr']
                scheduler.step()
                new_lr = optimizer.param_groups[0]['lr']
                if rank == 0:
                    print(f"  [LR SCHEDULER] LR updated: {current_lr:.2e} → {new_lr:.2e}")


            # Save epoch checkpoint
            dirpath = os.path.dirname(output_forge_pkl)
            base = os.path.basename(output_forge_pkl)
            name, ext = os.path.splitext(base)
            if epoch == epochs - 1:
                # If last epoch, save to the original filename, otherwise add epoch suffix
                save_path = output_forge_pkl
            else:
                save_path = os.path.join(dirpath if dirpath else ".", f"{name}_{epoch}{ext}")

            torch.save(self.state_dict(), save_path)

            if output_log_file is not None:
                with open(output_log_file, 'a') as file:
                    file.write(t)

        # Set Forge as trained
        self.is_trained = True

    def _mixed_pretrain(self,
                        input_mixed_list: List[Tuple[str, Union[MIPInfo, SATInfo]]],
                        output_forge_pkl: str,
                        output_log_file: Optional[str],
                        epochs: Optional[int] = None,
                        steps_per_instance: Optional[int] = None,
                        mip_learning_rate: Optional[float] = None,
                        sat_learning_rate: Optional[float] = None,
                        mip_weight_decay: Optional[float] = None,
                        sat_weight_decay: Optional[float] = None,
                        max_mip_graph_nodes: Optional[int] = None,
                        max_sat_graph_nodes: Optional[int] = None,
                        gradient_accumulation_steps: int = 1,
                        rank: int = 0,
                        world_size: int = 1,
                        gpu_memory_fraction: float = 0.8) -> None:
        """Pretrain the Forge model on mixed SAT and MIP instances with instance-type-specific hyperparameters.
        
        Optimized for SAT training - MIP instances help provide complementary learning signal while
        SAT instances are the primary focus. Each instance type uses its own learning rate and weight decay.

        Parameters
        ----------
        input_mixed_list : List[Tuple[str, Union[MIPInfo, SATInfo]]]
            List of tuples (instance_type, instance_info) where instance_type is 'mip' or 'sat'.
            instance_info is either MIPInfo or SATInfo object.
        output_forge_pkl : str
            Path to save the model state_dict after each epoch.
        output_log_file : Optional[str]
            Optional path to append training logs.
        epochs : Optional[int], default=None
            Number of training epochs.
        steps_per_instance : Optional[int], default=None
            Number of optimization steps per instance.
        mip_learning_rate : Optional[float], default=None
            Learning rate for MIP instances.
        sat_learning_rate : Optional[float], default=None
            Learning rate for SAT instances (primary focus).
        mip_weight_decay : Optional[float], default=None
            Weight decay for MIP instances.
        sat_weight_decay : Optional[float], default=None
            Weight decay for SAT instances (primary focus).
        max_mip_graph_nodes : Optional[int], default=None
            Max nodes for MIP instances.
        max_sat_graph_nodes : Optional[int], default=None
            Max nodes for SAT instances (primary focus).
        gradient_accumulation_steps : int, default=1
            Number of steps to accumulate gradients.
        rank : int, default=0
            Rank in distributed training.
        world_size : int, default=1
            Number of processes in distributed training.
        gpu_memory_fraction : float, default=0.8
            Target GPU memory usage fraction.

        Returns
        -------
        None
        """

        # Balanced load distribution for multi-GPU training
        sorted_mixed_list = sorted(input_mixed_list,
                                   key=lambda x: (getattr(x[1], 'num_cons', None) or getattr(x[1], 'num_clauses', None)) + x[1].num_vars,
                                   reverse=True)

        gpu_workloads = [[] for _ in range(world_size)]
        gpu_total_nodes = [0] * world_size

        for item_type, instance_info in sorted_mixed_list:
            instance_size = (getattr(instance_info, 'num_cons', None) or getattr(instance_info, 'num_clauses', None)) + instance_info.num_vars
            min_gpu = gpu_total_nodes.index(min(gpu_total_nodes))
            gpu_workloads[min_gpu].append((item_type, instance_info))
            gpu_total_nodes[min_gpu] += instance_size

        partitioned_mixed_list = gpu_workloads[rank]

        if rank == 0:
            print(f"\n{'='*80}")
            print(f"MIXED BATCH TRAINING (SAT-Optimized) - BALANCED LOAD DISTRIBUTION for {world_size} GPUs:")
            print(f"  Total instances: {len(input_mixed_list)}")
            for gpu_id in range(world_size):
                mip_count = sum(1 for t, _ in gpu_workloads[gpu_id] if t == 'mip')
                sat_count = sum(1 for t, _ in gpu_workloads[gpu_id] if t == 'sat')
                print(f"  GPU {gpu_id}: {len(gpu_workloads[gpu_id])} instances ({mip_count} MIP, {sat_count} SAT), {gpu_total_nodes[gpu_id]:,} total nodes")
            print(f"  Rank {rank} processes: {len(partitioned_mixed_list)} instances, {gpu_total_nodes[rank]:,} nodes")
            print(f"  NOTE: SAT instances are the primary optimization focus")
            print(f"{'='*80}\n", flush=True)

        # Put module into training mode and move to device
        super().train()
        self.to(self.device)

        if rank == 0:
            print(f"\n{'='*80}")
            print(f"MIXED PRETRAINING ON DEVICE: {self.device}")
            print(f"{'='*80}")
            if self.device.type == 'cuda':
                print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
            print(f"{'='*80}\n", flush=True)

        # Enable cuDNN autotuning
        torch.backends.cudnn.benchmark = True

        # Default to config values, but overwrite if given
        epochs = overwrite_if_given(self.epochs, epochs)
        steps_per_instance = overwrite_if_given(self.steps_per_instance, steps_per_instance)
        mip_learning_rate = overwrite_if_given(self.learning_rate, mip_learning_rate)
        sat_learning_rate = overwrite_if_given(self.learning_rate, sat_learning_rate)
        mip_weight_decay = overwrite_if_given(self.weight_decay, mip_weight_decay)
        sat_weight_decay = overwrite_if_given(self.weight_decay, sat_weight_decay)
        max_mip_graph_nodes = overwrite_if_given(self.max_graph_nodes, max_mip_graph_nodes)
        max_sat_graph_nodes = overwrite_if_given(self.max_graph_nodes, max_sat_graph_nodes)

        if rank == 0:
            print(f"\nHYPERPARAMETERS (Instance-Type Specific):")
            print(f"  MIP  - LR: {mip_learning_rate}, Weight Decay: {mip_weight_decay}, Max Nodes: {max_mip_graph_nodes}")
            print(f"  SAT* - LR: {sat_learning_rate}, Weight Decay: {sat_weight_decay}, Max Nodes: {max_sat_graph_nodes}")
            print(f"  Shared - Epochs: {epochs}, Steps/Instance: {steps_per_instance}")
            print(f"  (* = primary optimization focus)\n", flush=True)

        # Single optimizer shared across both instance types
        # Individual learning rates are applied via per-instance learning rate adjustment
        optimizer = optim.Adam(self.parameters(), lr=sat_learning_rate, weight_decay=sat_weight_decay)

        diagnostics = VQDiagnostics(self.codebook_size, self.updated_input_dim)

        t = ""
        main_loss_list = []

        for epoch in range(epochs):
            print("<<< Epoch:", epoch, flush=True)
            epoch_loss_list = []
            mip_loss_list = []
            sat_loss_list = []
            epoch_start = time.time()

            # Alternating lambda schedule (same as in _pretrain)
            # These control emphasis on node vs edge reconstruction losses
            if epoch % 2 == 0:
                self.lambda_node = 3.0
                self.lambda_edge = 7.0
            else:
                self.lambda_node = 5.0
                self.lambda_edge = 5.0

            # Process instances - interleaved MIP and SAT
            for idx in range(len(partitioned_mixed_list)):
                instance_type, instance_info = partitioned_mixed_list[idx]

                # Get instance-type-specific hyperparameters
                # SAT is the primary optimization target
                if instance_type == 'mip':
                    lr = mip_learning_rate
                    wd = mip_weight_decay
                    max_nodes = max_mip_graph_nodes
                else:  # 'sat' - primary focus
                    lr = sat_learning_rate
                    wd = sat_weight_decay
                    max_nodes = max_sat_graph_nodes

                # Update optimizer learning rate and weight decay for this instance
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    param_group['weight_decay'] = wd

                # Get number of nodes
                # Both MIPInfo (num_cons) and SATInfo (num_clauses) have num_vars
                num_cons_or_clauses = getattr(instance_info, 'num_cons', None) or getattr(instance_info, 'num_clauses', None)
                num_nodes = num_cons_or_clauses + instance_info.num_vars

                # Skip if exceeds max nodes for this instance type
                if max_nodes is not None and num_nodes > max_nodes:
                    if idx % 50 == 0:
                        print(f"  Skipping {instance_info.instance_name}: {num_nodes} nodes > {max_nodes} max", flush=True)
                    continue

                # GPU memory management - same as in _pretrain
                use_gpu = self.device.type == 'cuda'
                move_device = self.device

                if use_gpu:
                    gpu_max_memory = torch.cuda.get_device_properties(self.device).total_memory
                    gpu_allocated = torch.cuda.memory_allocated(self.device)
                    gpu_used_fraction = gpu_allocated / gpu_max_memory
                    # More conservative estimate for SAT instances (can have large edge lists)
                    estimated_memory_needed = num_nodes * 500 if instance_type == 'sat' else num_nodes * 200
                    projected_gpu_fraction = (gpu_allocated + estimated_memory_needed) / gpu_max_memory

                    if projected_gpu_fraction > gpu_memory_fraction:
                        move_device = torch.device('cpu')
                        if idx % 50 == 0 or idx < 5:  # Log first few instances and then every 50
                            print(f"  GPU memory {gpu_used_fraction:.1%} projected to {projected_gpu_fraction:.1%} "
                                  f"(exceeds {gpu_memory_fraction:.1%}), moving to CPU for: {instance_info.instance_name} "
                                  f"({num_nodes} nodes)", flush=True)

                # Move tensors to device
                try:
                    features = instance_info.feature_tensor.to(move_device, non_blocking=True)
                    edge_index = instance_info.edge_index.to(move_device, non_blocking=True)
                    edge_weight = instance_info.edge_weight.to(move_device, non_blocking=True)
                except Exception as e:
                    print(f"ERROR: Failed to move tensors to {move_device} for {instance_info.instance_name}: {e}", flush=True)
                    raise
                
                # CRITICAL: Move model to same device as data to avoid device mismatch
                try:
                    self.to(move_device)
                except Exception as e:
                    print(f"ERROR: Failed to move model to {move_device}: {e}", flush=True)
                    raise
                
                adj_gpu = None

                # Train on this instance
                instance_loss_list = []
                for step in range(steps_per_instance):
                    try:
                        optimizer.zero_grad(set_to_none=True)

                        # Forward pass - model and data are now on same device
                        h_list, logits, loss, indices, codebook_ = self.forward(features,
                                                                                num_cons_or_clauses, instance_info.num_vars,
                                                                                edge_index, edge_weight, adj_gpu)

                        # Track diagnostics
                        with torch.no_grad():
                            loss_scalar = loss.item() if isinstance(loss, torch.Tensor) else loss
                            stats = diagnostics.track_batch(indices, h_list[1], torch.tensor(loss_scalar))
                            if step == 0 or step == steps_per_instance - 1:
                                if idx % 50 == 0 or idx < 5:  # Log first few instances and then every 50
                                    instance_type_str = 'MIP' if instance_type == 'mip' else 'SAT*'
                                    print(f"    [{instance_type_str}] Step {step}/{steps_per_instance}: Codes {stats['codes_used_recent']}/{self.codebook_size} "
                                          f"({stats['utilization_pct_recent']:.1f}%), Loss {loss_scalar:.4f}", flush=True)

                        instance_loss_list.append(loss.item())
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                        optimizer.step()

                        # Cleanup
                        del h_list, logits, loss, indices, codebook_
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower() or 'cuda' in str(e).lower():
                            print(f"ERROR: GPU/CUDA error at instance {idx} ({instance_type}: {instance_info.instance_name}): {e}", flush=True)
                            if self.device.type == 'cuda':
                                print(f"  GPU Memory at crash: allocated={torch.cuda.memory_allocated() / 1e9:.2f}GB, "
                                      f"reserved={torch.cuda.memory_reserved() / 1e9:.2f}GB", flush=True)
                        raise

                # End of instance steps
                del features, edge_index, edge_weight, adj_gpu

                avg_instance_loss = float(np.mean(instance_loss_list)) if len(instance_loss_list) > 0 else 0.0
                epoch_loss_list.append(avg_instance_loss)

                if instance_type == 'mip':
                    mip_loss_list.append(avg_instance_loss)
                else:
                    sat_loss_list.append(avg_instance_loss)

                if idx % 50 == 0 or idx < 5:
                    mem_allocated = torch.cuda.memory_allocated() / 1e9 if self.device.type == 'cuda' else 0
                    instance_type_str = 'MIP' if instance_type == 'mip' else 'SAT*'
                    print("Epoch,", epoch,
                          ", Idx,", idx,
                          f", [{instance_type_str}]",
                          ", Avg. Instance Loss,", np.round(avg_instance_loss, 3),
                          ", Avg. Epoch Loss,", np.round(np.mean(epoch_loss_list), 3),
                          ", Current Time,", np.round(time.time() - epoch_start, 3),
                          ", GPU Mem (GB):", np.round(mem_allocated, 2),
                          ", ", instance_info.instance_name,
                          flush=True)

                if idx % 10 == 0 and idx > 0:
                    gc.collect()
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()

            # End of epoch cleanup and logging
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            main_loss_list.append(np.round(np.mean(epoch_loss_list), 3))

            # Compute separate averages for MIP and SAT
            mip_avg = np.mean(mip_loss_list) if mip_loss_list else 0.0
            sat_avg = np.mean(sat_loss_list) if sat_loss_list else 0.0

            epoch_summary = (f">>> DONE! Epoch {epoch} | "
                            f"Overall Avg Loss: {np.round(np.mean(epoch_loss_list), 3)} +/- {np.round(np.std(epoch_loss_list), 3)} | "
                            f"MIP Avg: {np.round(mip_avg, 3)} | SAT* Avg: {np.round(sat_avg, 3)} | "
                            f"Time: {np.round(time.time() - epoch_start, 3)}s")
            print(epoch_summary, flush=True)
            t += epoch_summary + "\n"

            if rank == 0:
                cb_norms = torch.norm(self.vq.codebook.data, p=2, dim=-1)
                cb_mean_norm = cb_norms.mean().item()
                cb_median_norm = cb_norms.median().item()
                cb_std_norm = cb_norms.std().item()
                print(f"  [CODEBOOK NORMS] mean={cb_mean_norm:.6f}, median={cb_median_norm:.6f}, std={cb_std_norm:.6f}", flush=True)
                diag_report = diagnostics.report(f"EPOCH {epoch} END (Mixed Training)")
                print(diag_report, flush=True)
                t += diag_report + "\n"
                diagnostics.reset()

            # Save checkpoint
            dirpath = os.path.dirname(output_forge_pkl)
            base = os.path.basename(output_forge_pkl)
            name, ext = os.path.splitext(base)
            if epoch == epochs - 1:
                save_path = output_forge_pkl
            else:
                save_path = os.path.join(dirpath if dirpath else ".", f"{name}_{epoch}{ext}")

            try:
                torch.save(self.state_dict(), save_path)
                if rank == 0:
                    print(f"  Checkpoint saved to: {save_path}", flush=True)
            except Exception as e:
                print(f"ERROR: Failed to save checkpoint to {save_path}: {e}", flush=True)
                raise

            if output_log_file is not None:
                try:
                    with open(output_log_file, 'a') as file:
                        file.write(t)
                except Exception as e:
                    print(f"WARNING: Failed to write log to {output_log_file}: {e}", flush=True)

        # Set Forge as trained
        self.is_trained = True

    def _mip_model_to_embeddings(self, mip_model: gp.Model, instance_embedding_only: bool) -> Union[MIPEmbeddings, SATEmbeddings]:
        """
        Convert a Gurobi model into embeddings using the trained Forge encoder.
        
        Supports both MIP and SAT models (SAT as LP/MPS representation of CNF formulas).

        Steps
        -----
        - Convert `mip_model` to `MIPInfo` or `SATInfo` (PyG style).
        - Call `forward()` in eval mode.
        - Build instance code histogram from `indices` assignments and extract per-node quantized embeddings.

        Parameters
        ----------
        mip_model : gurobipy.Model
            An already-loaded Gurobi model object (can represent either MIP or SAT instance).

        Returns
        -------
        Union[MIPEmbeddings, SATEmbeddings]
            Dataclass containing:
            - instance_embedding: 1D numpy array of length `self.codebook_size` with counts of assigned codes
            - embedding_of_constraint/embedding_of_clause: torch.Tensor of shape (num_cons/num_clauses, hidden_dim)
            - embedding_of_variable: torch.Tensor of shape (num_vars, hidden_dim)

        Raises
        ------
        TypeError
            If `mip_model` is not a gurobipy.Model.
        """
        # Validate input type
        if not isinstance(mip_model, gp.Model):
            raise TypeError(f"Error: mip_model must be a gurobipy.model, got {type(mip_model).__name__}")

        # If not trained, warn (keeps previous behavior)
        if not self.is_trained:
            raise ValueError("Error: Forge is not trained and no pre-trained model path is given.")

        # Ensure module is in evaluation mode and on device
        self.eval()

        # Store Forge eval mode (to restore back) and set to eval only to generate embedding
        original_mode = self.is_eval_mode
        self.is_eval_mode = True

        # Detect if this is a SAT or MIP model by checking model properties
        # SAT models typically have fewer variables and specific naming patterns
        is_sat_model = self._detect_sat_model(mip_model)

        # Convert model to appropriate info object
        if is_sat_model:
            # Pass feature selection to ensure correct dimensions
            # For older models without these attributes, None will use defaults
            info = SATProcessor._sat_model_to_satinfo(
                mip_model,
                selected_var_features=getattr(self, 'selected_sat_var_features', None),
                selected_clause_features=getattr(self, 'selected_sat_clause_features', None)
            )
            embedding_class = SATEmbeddings
            print("Detected SAT model", flush=True)
        else:
            info = MIPProcessor._mip_model_to_mipinfo(mip_model)
            embedding_class = MIPEmbeddings
            print("Detected MIP model", flush=True)

        # Validate feature tensor dimensions
        if info.feature_tensor is None:
            raise ValueError(f"Error: feature_tensor is None for model with {info.num_clauses if hasattr(info, 'num_clauses') else info.num_cons} constraints/clauses and {info.num_vars} variables")
        
        expected_feature_dim = self.input_dim
        actual_feature_dim = info.feature_tensor.shape[1] if info.feature_tensor.dim() == 2 else info.feature_tensor.shape[0]
        
        if actual_feature_dim != expected_feature_dim:
            raise ValueError(f"Error: Feature dimension mismatch. Expected {expected_feature_dim}, got {actual_feature_dim}. "
                           f"Feature tensor shape: {info.feature_tensor.shape}, "
                           f"Model has {info.num_clauses if hasattr(info, 'num_clauses') else info.num_cons} clauses and {info.num_vars} variables")

        # Forward pass through trained Forge
        with torch.no_grad():
            h_list, logits, loss, indices, codebook_ = self.forward(info.feature_tensor.to(self.device),
                                                                info.num_cons if hasattr(info, 'num_cons') else info.num_clauses, info.num_vars,
                                                                info.edge_index.to(self.device),
                                                                info.edge_weight.to(self.device))
        # Restore original mode
        self.is_eval_mode = original_mode

        # Compute instance vector, as a frequency distribution of codes assigned to constraints/clauses and variables
        assigned_codes = indices.detach().cpu().numpy()
        instance_embedding = np.bincount(assigned_codes, minlength=self.codebook_size).astype(float)

        if instance_embedding_only:
            if is_sat_model:
                return embedding_class(instance_embedding=instance_embedding,
                                       embedding_of_clause=None,
                                       embedding_of_variable=None)
            else:
                return embedding_class(instance_embedding=instance_embedding,
                                       embedding_of_constraint=None,
                                       embedding_of_variable=None)
        else:
            if is_sat_model:
                embedding_of_clause = h_list[1][:info.num_clauses]
                embedding_of_variable = h_list[1][info.num_clauses:]
                return embedding_class(instance_embedding=instance_embedding,
                                       embedding_of_clause=embedding_of_clause,
                                       embedding_of_variable=embedding_of_variable)
            else:
                embedding_of_constraint = h_list[1][:info.num_cons]
                embedding_of_variable = h_list[1][info.num_cons:]
                return embedding_class(instance_embedding=instance_embedding,
                                       embedding_of_constraint=embedding_of_constraint,
                                       embedding_of_variable=embedding_of_variable)

    @staticmethod
    def _detect_sat_model(mip_model: gp.Model) -> bool:
        """
        Heuristically detect whether a Gurobi model represents a SAT formula or a general MIP.
        
        Parameters
        ----------
        mip_model : gurobipy.Model
            A Gurobi model to analyze.
            
        Returns
        -------
        bool
            True if the model appears to be a SAT formula, False if it appears to be a general MIP.
        """
        # SAT models have a dummy objective with name pattern "OBJ: __dummy"
        try:
            obj = mip_model.getObjective()
            
            # Get the objective expression as a string to check for dummy naming
            obj_str = str(obj)
            if "dummy" in obj_str:
                return True
            
            return False
        except:
            # If we can't determine, default to MIP (more general)
            return False


    def _finetune_integral_gap(self,
                               input_mip_to_gapinfo: Dict[str, GapInfo],
                               output_forge_finetuned_pkl: str,
                               epochs: Optional[int] = None,  # 10,
                               steps_per_instance: Optional[int] = None,  # 10,
                               learning_rate: Optional[float] = None,  # 1e-4,
                               weight_decay: Optional[float] = None,  # 5e-4,
                               max_graph_nodes: Optional[int] = None,  # 30000
                               input_forge_pretrained_pkl: str = ""
                               ) -> None:
        """Fine-tune the Forge model for integral gap prediction.

        Parameters
        ----------
        input_mip_to_gapinfo : Dict[str, GapInfo]
            Dictionary mapping MIP file paths to their GapInfo objects containing LP/MIP solutions.
        output_forge_finetuned_pkl : str
            Path to save the fine-tuned model state_dict.
        epochs : Optional[int], default=None
            Number of fine-tuning epochs; if None, uses config default.
        steps_per_instance : Optional[int], default=None
            Optimization steps per instance; if None, uses config default.
        learning_rate : Optional[float], default=None
            Learning rate for optimizer; if None, uses config default.
        weight_decay : Optional[float], default=None
            Weight decay for optimizer; if None, uses config default.
        max_graph_nodes : Optional[int], default=None
            Maximum graph size to process; larger instances are skipped.

        Returns
        -------
        None
        """

        epochs = overwrite_if_given(self.epochs, epochs)
        steps_per_instance = overwrite_if_given(self.steps_per_instance, steps_per_instance)
        learning_rate = overwrite_if_given(self.learning_rate, learning_rate)
        weight_decay = overwrite_if_given(self.weight_decay, weight_decay)
        max_graph_nodes = overwrite_if_given(self.max_graph_nodes, max_graph_nodes)

        # Set model to training mode and move to device
        self.to(self.device)
        self.train()

        if self.is_trained and not self.has_integral_gap_head and input_forge_pretrained_pkl != "":
            print("Warning: Forge model has been pre-trained but missing integral gap head, adding head.")
            self.has_integral_gap_head = True
            self.integral_gap_layer = nn.Linear(self.updated_input_dim, 1)

            # TODO: See if the stuff below can be simplified. 

            # Create new Forge object
            pre_trained = Forge()

            # Load existing weights into temporary model
            pre_trained.load_model(input_forge_pretrained_pkl, model_type=Constants.FORGE_PRE_TRAIN)

            # Copy parameters from old model to new model with gap head
            copy_params(old_model=pre_trained, new_model=self)

            # Delete temporary model to free memory
            del pre_trained

            # Flush GPU cache to ensure old model is deleted from memory
            torch.cuda.empty_cache()

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Start Gurobi environment
        gurobi_env = _MIPUtils.start_gurobi_env()

        # Mip instances
        mips = list(input_mip_to_gapinfo.keys())
        for epoch in range(epochs):

            epoch_loss = []
            gap_epoch_loss = []

            for idx, mip in enumerate(mips):

                # Read MIP file to a Gurobi model
                mip_model = gp.read(mip, env=gurobi_env)

                # Generate MIPInfo object from Gurobi model, set name, and add to dictionary
                mipinfo = MIPProcessor._mip_model_to_mipinfo(mip_model)

                # Skip if too large to fit in GPU memory
                num_nodes = mipinfo.num_cons + mipinfo.num_vars
                if num_nodes > max_graph_nodes:
                    continue

                # Push to device before the for-loop below
                edge_index = mipinfo.edge_index.to(self.device)
                edge_weight = mipinfo.edge_weight.to(self.device)
                feature_tensor = mipinfo.feature_tensor.to(self.device)

                for step in range(steps_per_instance):

                    optimizer.zero_grad()

                    # Compute loss and prediction
                    h_list, logits, loss, indices, codebook_ = self.forward(feature_tensor, mipinfo.num_cons,
                                                                            mipinfo.num_vars, edge_index, edge_weight)
                    # Predict gap ratio
                    # h_list[-1] is the integral gap head output
                    gap_ratio_pred = torch.mean(h_list[-1][mipinfo.num_cons:, :])
                    gap_ratio_true = input_mip_to_gapinfo[mip].gap_ratio

                    # Make larger gaps >1 appear as small ratios (for both minimization and maximization)
                    if gap_ratio_true > 1:
                        gap_ratio_true = 1 / gap_ratio_true

                    # Comment out - optional variable probability head
                    # Predict variable probabilities
                    # var_proba_pred = h_list[-2][mipinfo.num_cons:, :]
                    # var_proba_truth = torch.Tensor(input_mip_to_gapinfo[mip].mip_sol).to(self.device)

                    try:
                        loss = torch.abs(gap_ratio_pred - gap_ratio_true)
                        loss.backward()
                        optimizer.step()

                        print('', '(', idx, '/', len(mips), ') |', mip,
                              ' | GAP Loss :', loss.item(), end='\r')

                        epoch_loss.append(loss.item())
                        gap_epoch_loss.append(loss.item())
                    except:
                        continue

            print("\nEpoch ", epoch + 1,
                  "| Means | Loss : ", np.mean(epoch_loss),
                  "| Gap Loss : ", np.mean(gap_epoch_loss))
            print()

            torch.save(self.state_dict(), output_forge_finetuned_pkl)

            # Shuffle MIP instances for next epoch
            np.random.shuffle(mips)

        # Close Gurobi environment
        gurobi_env.close()

    def _finetune_sat_prediction(self,
                                 input_sat_to_satinfo: Dict[str, 'SATSatisfiabilityInfo'],
                                 output_forge_finetuned_pkl: str,
                                 output_log_file: Optional[str] = None,
                                 epochs: Optional[int] = None,
                                 steps_per_instance: Optional[int] = None,
                                 learning_rate: Optional[float] = None,
                                 weight_decay: Optional[float] = None,
                                 max_graph_nodes: Optional[int] = None,
                                 freeze_level: str = "none",
                                 bce_weight: float = 1.0,
                                 contrastive_weight: float = 0.0) -> None:
        """Fine-tune the Forge model for SAT satisfiability prediction.

        The model learns to predict satisfiability (SAT or UNSAT) based solely on the SAT instance
        graph structure (clauses and variables). The filename labels (from "_sat" or "_unsat") are
        used only as ground truth for computing the loss during training and should NOT be used
        as input to the model. During evaluation/testing, the model makes predictions from the
        graph alone and accuracy is checked against the filename labels.

        Parameters
        ----------
        input_sat_to_satinfo : Dict[str, SATSatisfiabilityInfo]
            Dictionary mapping SAT file paths to their satisfiability information (is_satisfiable label).
            The labels are extracted from filenames (check for "_sat" or "_unsat") and used only
            as ground truth for computing loss during training.
        output_forge_finetuned_pkl : str
            Path to save the fine-tuned model state_dict.
        output_log_file : Optional[str], default=None
            Optional path to append fine-tuning logs; if None, logs are not written to disk.
        epochs : Optional[int], default=None
            Number of fine-tuning epochs; if None, uses config default.
        steps_per_instance : Optional[int], default=None
            Optimization steps per instance; if None, uses config default.
        learning_rate : Optional[float], default=None
            Learning rate for optimizer; if None, uses config default.
        weight_decay : Optional[float], default=None
            Weight decay for optimizer; if None, uses config default.
        max_graph_nodes : Optional[int], default=None
            Maximum graph size to process; larger instances are skipped.
        freeze_level : str, default="none"
            How much to freeze: 
            - "none": Train all parameters (encoder + SAT head)
            - "partial": Freeze early layers, train graph_layer_2 + linear + SAT head
            - "full": Freeze encoder, train SAT head only
        bce_weight : float, default=1.0
            Weight for BCE (classification) loss in combined loss. Use 1.0 for classification-only, 0.0 for contrastive-only.
            Recommend 0.95 for combined training with light contrastive regularization.
        contrastive_weight : float, default=0.0
            Weight for contrastive (embedding separation) loss in combined loss. Use 1.0 to try pure contrastive learning.
            Recommend 0.05 when combined with BCE. Note: bce_weight + contrastive_weight should sum to 1.0.

        Returns
        -------
        None
        """

        epochs = overwrite_if_given(self.epochs, epochs)
        steps_per_instance = overwrite_if_given(self.steps_per_instance, steps_per_instance)
        learning_rate = overwrite_if_given(self.learning_rate, learning_rate)
        weight_decay = overwrite_if_given(self.weight_decay, weight_decay)
        max_graph_nodes = overwrite_if_given(self.max_graph_nodes, max_graph_nodes)

        # Set model to training mode and move to device
        self.to(self.device)
        self.train()

        # Add SAT satisfiability prediction head if not present
        if not hasattr(self, 'sat_satisfiability_layer'):
            self.has_sat_satisfiability_head = True
            # Node-level prediction head (processes EACH node independently)
            # Then aggregates node-level logits via mean pooling
            # This is more expressive than aggregating embeddings first
            # 
            # Architecture:
            # - Input: [num_nodes, embedding_dim]
            # - Output: [num_nodes, 1] logits
            # - Final: mean(logits) over all nodes
            hidden_dim = self.updated_input_dim
            self.sat_satisfiability_layer = nn.Sequential(
                nn.LayerNorm(self.updated_input_dim),
                nn.Linear(self.updated_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1)  # Per-node binary logit
            ).to(self.device)
            
            # Conservative initialization to prevent activation saturation
            for layer in self.sat_satisfiability_layer:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=1.0)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        # Apply freezing strategy based on freeze_level parameter
        if freeze_level in ["partial", "full"]:
            # Freeze encoder layers for transfer learning
            for param in self.parameters():
                param.requires_grad = False
            
            if freeze_level == "partial":
                # Unfreeze later layers: graph_layer_2, linear layer, and SAT head
                for param in self.graph_layer_2.parameters():
                    param.requires_grad = True
                for param in self.linear.parameters():
                    param.requires_grad = True
            
            # Always unfreeze SAT head for both "partial" and "full"
            for param in self.sat_satisfiability_layer.parameters():
                param.requires_grad = True
        else:
            # For freeze_level="none", ensure all parameters are trainable
            for param in self.parameters():
                param.requires_grad = True
        
        print(f"\n{'='*80}")
        print(f"SAT SATISFIABILITY FINE-TUNING SETUP")
        print(f"{'='*80}")
        print(f"Freeze level: {freeze_level}")
        if freeze_level == "none":
            print(f"Training strategy: Full model (encoder + SAT head)")
        elif freeze_level == "partial":
            print(f"Training strategy: Partial freezing (graph_layer_1 frozen, graph_layer_2 + linear + SAT head trainable)")
        elif freeze_level == "full":
            print(f"Training strategy: SAT head only (encoder frozen)")
        
        # Debug: show which parameters are trainable
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {frozen_params:,}")
        
        # Show encoder status
        encoder_trainable = sum(p.numel() for name, p in self.named_parameters() if 'graph_layer' in name and p.requires_grad)
        encoder_frozen = sum(p.numel() for name, p in self.named_parameters() if 'graph_layer' in name and not p.requires_grad)
        print(f"Graph layers trainable params: {encoder_trainable:,}, frozen: {encoder_frozen:,}")
        
        print(f"Model in training mode: {self.training}")
        print(f"SAT satisfiability layer exists: {hasattr(self, 'sat_satisfiability_layer')}")
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"{'='*80}\n")


        # Get SAT instances before setting up learning rate schedule (needed to compute total_steps)
        # COMMENTED OUT: Learning rate scheduler disabled
        sat_files = list(input_sat_to_satinfo.keys())
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Weighted BCE loss: pos_weight=0.7 reduces emphasis on positive class (SAT)
        # This prevents the model from developing a SAT-heavy bias (always predicting SAT)
        # pos_weight < 1.0 means negatives (UNSAT) are upweighted relative to positives (SAT)
        # Typical range: 0.5-0.75 for class balance recovery; using 0.7 for balanced class performance
        bce_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.7], device=self.device))
        
        # Adaptive learning rate scheduler: ReduceLROnPlateau
        # Only reduces LR when validation metric (accuracy) stops improving
        # This keeps LR high while making progress, drops it when stuck
        # Settings:
        #   factor=0.5: Reduce by half (smoother than 0.1)
        #   patience=5: Drop LR after 5 epochs of no improvement
        #   min_lr=5e-6: Never drop below this floor
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',           # Minimize accuracy
            factor=0.5,           # Reduce LR by half
            patience=3,           # Wait 3 epochs before reducing
            min_lr=5e-6           # Never below this threshold
        )
        
        # Debug: show optimizer param_groups
        print(f"\nOptimizer parameter groups:")
        for i, param_group in enumerate(optimizer.param_groups):
            num_params = sum(p.numel() for p in param_group['params'])
            print(f"  Group {i}: {num_params:,} parameters, lr={param_group['lr']}")
            # Sample first few parameter names
            param_names = []
            for p in param_group['params'][:3]:
                for name, param in self.named_parameters():
                    if param is p:
                        param_names.append(name)
                        break
            if param_names:
                print(f"    Sample params: {param_names}")
        
        # Adaptive learning rate schedule
        print(f"\nLearning rate schedule (ReduceLROnPlateau):")
        print(f"  Initial LR: {learning_rate:.2e}")
        print(f"  Factor: 0.5 (reduce by half)")
        print(f"  Patience: 5 epochs")
        print(f"  Min LR: 5e-06")
        print(f"  Monitored metric: Training accuracy (maximize)")
        print(f"\nLoss configuration:")
        print(f"  BCE weight: {bce_weight}")
        print(f"  Contrastive weight: {contrastive_weight}")

        # Start Gurobi environment
        gurobi_env = _SATUtils.start_gurobi_env()
        
        # Save initial SAT head weights for comparison
        # Access final linear layer [2] of Sequential for weight/bias
        final_layer = self.sat_satisfiability_layer[-1] if isinstance(self.sat_satisfiability_layer, nn.Sequential) else self.sat_satisfiability_layer
        initial_sat_weight = final_layer.weight.data.clone().detach()
        initial_sat_bias = final_layer.bias.data.clone().detach() if final_layer.bias is not None else None
        
        # Contrastive loss: maintain a buffer of recent embeddings for contrastive computation
        embedding_buffer = {'sat': [], 'unsat': []}  # Store (embedding, label) pairs
        buffer_size = 64  # Keep 64 most recent instances of each class for better negatives
        log_content = ""
        for epoch in range(epochs):

            epoch_loss = []
            sat_epoch_loss = []
            contrastive_epoch_loss = []
            epoch_predictions = []  # Track all predictions to monitor learning
            epoch_labels = []  # Track ground truth labels for accuracy computation
            epoch_start = time.time()

            for idx, sat_file in enumerate(sat_files):

                try:
                    # Read SAT file to a Gurobi model
                    sat_model = gp.read(sat_file, env=gurobi_env)

                    # Generate SATInfo object from Gurobi model (graph structure only)
                    # NOTE: Model only sees the graph structure; filename label is NOT passed to the model
                    satinfo = SATProcessor._sat_model_to_satinfo(sat_model)
                    
                    # Skip if too large to fit in GPU memory or if conversion failed
                    if satinfo is None:
                        continue
                    
                    num_nodes = satinfo.num_clauses + satinfo.num_vars
                    if num_nodes > max_graph_nodes:
                        continue

                    # Push to device before the for-loop below
                    edge_index = satinfo.edge_index.to(self.device)
                    edge_weight = satinfo.edge_weight.to(self.device)
                    feature_tensor = satinfo.feature_tensor.to(self.device)

                    for step in range(steps_per_instance):

                        optimizer.zero_grad()

                        # Compute loss and prediction
                        h_list, logits, loss, indices, codebook_ = self.forward(feature_tensor, satinfo.num_clauses,
                                                                                satinfo.num_vars, edge_index, edge_weight)
                        
                        quantized_embeddings = h_list[1]  # The quantized latent representation [num_nodes, embedding_dim]
                        
                        # Verify the shape is correct (GGNN-style: graph-level prediction)
                        if h_list[-1].dim() != 2 or h_list[-1].shape != (1, 1):
                            print(f"\nWARNING: Unexpected h_list[-1] shape: {h_list[-1].shape}. Expected [1, 1] (graph-level).")
                            print(f"h_list length: {len(h_list)}")
                            for i, h in enumerate(h_list):
                                print(f"  h_list[{i}]: shape={h.shape}")
                            continue
                        
                        # Extract logit and apply sigmoid (GGNN approach)
                        sat_pred_logit = h_list[-1].squeeze()  # Scalar
                        sat_pred = torch.sigmoid(sat_pred_logit)
                        
                        # Get instance-level embedding (mean pool across nodes)
                        instance_embedding = torch.mean(quantized_embeddings, dim=0)  # [embedding_dim]
                        
                        # Ground truth label from filename (extracted from "_sat" or "_unsat")
                        # Used only for computing loss; NOT passed to model
                        sat_true = float(0 if "_unsat" in sat_file else 1)
                        sat_class = "sat" if sat_true > 0.5 else "unsat"
                        
                        # Debug: check input to SAT head on first instance of epoch
                        if epoch == 0 and idx == 0 and step == 0:
                            graph_emb = torch.mean(quantized_embeddings, dim=0)  # Aggregate to graph-level
                            input_norm = torch.norm(graph_emb).item()
                            input_mean = torch.mean(graph_emb).item()
                            input_std = torch.std(graph_emb).item()
                            final_layer = self.sat_satisfiability_layer[-1] if isinstance(self.sat_satisfiability_layer, nn.Sequential) else self.sat_satisfiability_layer
                            sat_layer_weight_norm = torch.norm(final_layer.weight).item()
                            sat_layer_bias = final_layer.bias.item() if final_layer.bias is not None else 0.0
                            print(f"\n[SAT Head - Node-level prediction] num_nodes={quantized_embeddings.shape[0]}")
                            print(f"  graph_emb_norm={input_norm:.4f}, mean={input_mean:.4f}, std={input_std:.4f}")
                            print(f"  SAT head final layer - weight_norm={sat_layer_weight_norm:.4f}, bias={sat_layer_bias:.4f}")
                            print(f"  h_list[-1] (aggregated logit): {h_list[-1].item():.4f}, sigmoid(logit)={torch.sigmoid(h_list[-1]).item():.4f}")
                            
                            # Check if encoder is trainable
                            encoder_trainable = any(p.requires_grad for name, p in self.named_parameters() if 'graph_layer' in name)
                            print(f"  Encoder trainable: {encoder_trainable}")

                        try:
                            # ===== BCE Loss: Classification accuracy =====
                            # Use BCEWithLogitsLoss with pos_weight for class balance
                            # This expects raw logits (not sigmoid probabilities)
                            # pos_weight=0.7 reduces emphasis on positive class (SAT) to prevent SAT-heavy bias
                            sat_pred_scaled = torch.sigmoid(sat_pred_logit / self.sat_temperature)  # For predictions/metrics only
                            bce_loss = bce_criterion(sat_pred_logit.unsqueeze(0), 
                                                    torch.tensor(sat_true, device=self.device, dtype=torch.float32).unsqueeze(0))
                            
                            # ===== Contrastive Loss: Embedding separation =====
                            # Use improved formulation: minimize cosine similarity to opposite class
                            # Provides continuous gradients throughout training (no hinge plateau)
                            if contrastive_weight > 0:
                                contrastive_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                                
                                if len(embedding_buffer['sat']) > 0 and len(embedding_buffer['unsat']) > 0:
                                    # Get buffers for OPPOSITE class (negatives)
                                    opposite_class = 'unsat' if sat_class == 'sat' else 'sat'
                                    opposite_embeddings = torch.stack([emb for emb, _ in embedding_buffer[opposite_class]])  # [buffer_size, embedding_dim]
                                    
                                    # Normalize embeddings for cosine similarity
                                    current_norm = F.normalize(instance_embedding, dim=0)  # [embedding_dim]
                                    opposite_norm = F.normalize(opposite_embeddings, dim=1)  # [buffer_size, embedding_dim]
                                    
                                    # Cosine similarity: high = close, low = far
                                    cosine_sims = torch.mm(opposite_norm, current_norm.unsqueeze(1)).squeeze(1)  # [buffer_size]
                                    
                                    # Contrastive loss: minimize MEAN similarity to negatives
                                    # This ensures all negatives contribute gradients (not just hardest)
                                    # Use softmax over similarities to focus on hardest negatives upweighted
                                    sim_weights = F.softmax(cosine_sims / 0.1, dim=0)  # Temperature=0.1 for harder focus
                                    weighted_sim = torch.sum(sim_weights * cosine_sims)
                                    contrastive_loss = weighted_sim  # Minimize this (push away)
                                else:
                                    # Not enough data in both buffers yet - no contrastive signal
                                    contrastive_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
                            else:
                                contrastive_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
                            
                            # ===== Combined Loss =====
                            # Normalize losses and combine with weights
                            if bce_weight > 0 and contrastive_weight > 0:
                                # Combined training: normalize both losses to ~[0,1] range
                                combined_loss = bce_weight * bce_loss + contrastive_weight * contrastive_loss
                            elif bce_weight > 0:
                                # Classification-only
                                combined_loss = bce_loss
                            elif contrastive_weight > 0:
                                # Contrastive-only
                                combined_loss = contrastive_loss
                            else:
                                raise ValueError("At least one of bce_weight or contrastive_weight must be > 0")
                            
                            # Backward pass (no loss scaling)
                            combined_loss.backward()
                            
                            # Gradient clipping to prevent instability
                            grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                            
                            # Debug: Check gradient flow on first few steps of first epoch
                            if epoch == 0 and idx < 1 and step < 2:
                                sat_head_grad_norm = 0.0
                                encoder_grad_norm = 0.0
                                for name, param in self.named_parameters():
                                    if param.grad is not None:
                                        grad_n = torch.norm(param.grad).item()
                                        if 'sat_satisfiability' in name:
                                            sat_head_grad_norm += grad_n ** 2
                                        elif 'graph_layer' in name or 'linear' in name:
                                            encoder_grad_norm += grad_n ** 2
                                sat_head_grad_norm = math.sqrt(sat_head_grad_norm)
                                encoder_grad_norm = math.sqrt(encoder_grad_norm)
                                
                                print(f"\n[Epoch {epoch}, Inst {idx}, Step {step}]")
                                print(f"  Loss: {combined_loss.item():.6f} (BCE: {bce_loss.item():.6f}, Contrastive: {contrastive_loss.item():.6f})")
                                print(f"  Grad norms [grad_norm_before_clip={grad_norm_before_clip:.6f}]:")
                                print(f"    SAT head: {sat_head_grad_norm:.8f}")
                                print(f"    Encoder: {encoder_grad_norm:.8f}")
                                print(f"  Prediction: {sat_pred_scaled.item():.4f} (true: {sat_true})")
                                print(f"  Embedding norm: {torch.norm(instance_embedding).item():.4f}")
                            
                            # # Apply learning rate schedule (commented out)
                            # global_step = epoch * len(sat_files) * steps_per_instance + idx * steps_per_instance + step
                            # lr_scale = get_lr_scale(global_step, total_steps)
                            # for param_group in optimizer.param_groups:
                            #     param_group['lr'] = learning_rate * lr_scale
                            
                            optimizer.step()

                            # CRITICAL FIX (April 2026): Normalize codebook AND sync EMA buffers after each step
                            # This prevents the EMA mechanism from drifting codebook norms away from unit sphere
                            with torch.no_grad():
                                # 1. Normalize the visible codebook parameter
                                norm_vec = F.normalize(self.vq.codebook.data, p=2, dim=-1)
                                self.vq.codebook.copy_(norm_vec)
                                
                                # 2. Sync the internal EMA buffers to maintain consistency
                                # The 'embed_avg' buffer is used by the EMA update mechanism;
                                # keeping it in sync ensures EMA won't drift norms on the next forward pass
                                if hasattr(self.vq, 'embed_avg') and self.vq.embed_avg is not None:
                                    self.vq.embed_avg.copy_(norm_vec * self.vq.cluster_size.unsqueeze(-1))
                                
                                # DEBUG: Verify codebook is actually normalized
                                if step == 0 and idx % 10 == 0:
                                    cb_norms = torch.norm(self.vq.codebook.data, p=2, dim=-1)
                                    cb_mean_norm = cb_norms.mean().item()
                                    cb_max_norm = cb_norms.max().item()
                                    if cb_mean_norm > 1.01 or cb_mean_norm < 0.99:
                                        print(f"    WARNING: Codebook norm drift! mean={cb_mean_norm:.6f}, max={cb_max_norm:.6f}")

                            print('', '(', idx, '/', len(sat_files), ') |', sat_file,
                                  ' | BCE:', f"{bce_loss.item():.4f}", 
                                  ' | Contrastive:', f"{contrastive_loss.item():.4f}",
                                  end='\r')

                            # Track losses (use original unscaled loss for monitoring)
                            epoch_loss.append(combined_loss.item())
                            sat_epoch_loss.append(bce_loss.item())
                            contrastive_epoch_loss.append(contrastive_loss.item())
                            # Track scaled predictions for accuracy monitoring
                            epoch_predictions.append(sat_pred_scaled.item())  # Track scaled prediction
                            epoch_labels.append(sat_true)  # Track ground truth label
                            
                            # Update embedding buffer for next instances (only if using contrastive loss)
                            if contrastive_weight > 0:
                                # Keep most recent instances of each class
                                if len(embedding_buffer[sat_class]) >= buffer_size:
                                    embedding_buffer[sat_class].pop(0)  # Remove oldest
                                embedding_buffer[sat_class].append((instance_embedding.detach(), sat_true))
                            
                        except Exception as e:
                            print(f"\nError in forward/backward pass for {sat_file}: {e}")
                            continue

                except Exception as e:
                    print(f"\nError processing {sat_file}: {e}")
                    continue

            # Compute accuracy for this epoch
            epoch_accuracy = 0.0
            sat_accuracy = 0.0
            unsat_accuracy = 0.0
            if epoch_predictions and epoch_labels:
                predictions_binary = (np.array(epoch_predictions) > 0.5).astype(int)
                epoch_labels_array = np.array(epoch_labels)
                epoch_accuracy = np.mean(predictions_binary == epoch_labels_array) * 100  # Convert to percentage
                
                # Per-class accuracy
                sat_mask = epoch_labels_array == 1
                unsat_mask = epoch_labels_array == 0
                if sat_mask.sum() > 0:
                    sat_accuracy = np.mean(predictions_binary[sat_mask] == epoch_labels_array[sat_mask]) * 100
                if unsat_mask.sum() > 0:
                    unsat_accuracy = np.mean(predictions_binary[unsat_mask] == epoch_labels_array[unsat_mask]) * 100
            
            epoch_summary = (f">>> DONE! Epoch {epoch}, "
                             f"Combined Loss: {np.mean(epoch_loss) if epoch_loss else 0:.3f}, "
                             f"BCE (weighted pos_weight=0.7): {np.mean(sat_epoch_loss) if sat_epoch_loss else 0:.3f}, "
                             f"Contrastive: {np.mean(contrastive_epoch_loss) if contrastive_epoch_loss else 0:.3f}, "
                             f"+/-, {np.std(epoch_loss) if epoch_loss else 0:.3f}, "
                             f"Accuracy: {epoch_accuracy:.2f}% (SAT: {sat_accuracy:.2f}%, UNSAT: {unsat_accuracy:.2f}%), "
                             f"Epoch Time: {np.round(time.time() - epoch_start, 3)}s")
            
            # Add embedding buffer status
            buffer_status = f"[Buffer SAT: {len(embedding_buffer['sat'])}, UNSAT: {len(embedding_buffer['unsat'])}]"
            
            # Add prediction statistics to summary
            if epoch_predictions:
                pred_mean = np.mean(epoch_predictions)
                pred_min = np.min(epoch_predictions)
                pred_max = np.max(epoch_predictions)
                pred_q25 = np.quantile(epoch_predictions, 0.25)
                pred_q75 = np.quantile(epoch_predictions, 0.75)
                epoch_summary += f" | Pred Stats: min={pred_min:.4f}, Q25={pred_q25:.4f}, mean={pred_mean:.4f}, Q75={pred_q75:.4f}, max={pred_max:.4f}"
                
                # Diagnostic: If predictions are always near 0.5, there's no discrimination
                pred_std = np.std(epoch_predictions)
                if pred_std < 0.05:
                    epoch_summary += f" ⚠️  [Low pred variance: {pred_std:.4f} - model not learning to discriminate!]"
            
            # Check encoder status
            if epoch == 0:
                encoder_trainable = any(p.requires_grad for name, p in self.named_parameters() if 'graph_layer' in name)
                if not encoder_trainable:
                    epoch_summary += f" ⚠️  [ENCODER FROZEN - Consider freeze_level='none' for SAT finetuning]"
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            epoch_summary += f" | LR: {current_lr:.2e}"
            
            # Track SAT head weight changes
            final_layer = self.sat_satisfiability_layer[-1] if isinstance(self.sat_satisfiability_layer, nn.Sequential) else self.sat_satisfiability_layer
            current_sat_weight = final_layer.weight.data
            current_sat_bias = final_layer.bias.data if final_layer.bias is not None else None
            weight_change = torch.norm(current_sat_weight - initial_sat_weight).item()
            weight_norm = torch.norm(current_sat_weight).item()
            bias_change = torch.norm(current_sat_bias - initial_sat_bias).item() if initial_sat_bias is not None else 0.0
            
            # Check if any parameters actually changed
            total_param_change = 0.0
            for param in self.parameters():
                if param.requires_grad and param.grad is not None:
                    total_param_change += torch.norm(param.data).item() ** 2
            total_param_change = math.sqrt(total_param_change)
            
            epoch_summary += f" | SAT Weight: norm={weight_norm:.4f}, weight_change={weight_change:.8f}, bias_change={bias_change:.8f}"
            
            # DEBUG: Check actual codebook parameter norms at epoch end
            cb_norms = torch.norm(self.vq.codebook.data, p=2, dim=-1)
            cb_mean_norm = cb_norms.mean().item()
            cb_median_norm = cb_norms.median().item()
            cb_std_norm = cb_norms.std().item()
            cb_min_norm = cb_norms.min().item()
            cb_max_norm = cb_norms.max().item()
            print(f"  [ACTUAL CODEBOOK NORMS] mean={cb_mean_norm:.6f}, median={cb_median_norm:.6f}, std={cb_std_norm:.6f}, min={cb_min_norm:.6f}, max={cb_max_norm:.6f}")
            
            print(f"\n{epoch_summary}")
            
            # Additional diagnostics
            if epoch_labels:
                sat_count = np.sum(np.array(epoch_labels) == 1)
                unsat_count = np.sum(np.array(epoch_labels) == 0)
                print(f"  Class distribution: SAT={sat_count}, UNSAT={unsat_count}")
            
            if not epoch_loss:
                print("⚠️  WARNING: No training loss computed in this epoch! Model may not be learning.")
                print("   Check for errors above or verify that SAT files are being processed correctly.")
            
            print()
            
            log_content += epoch_summary + "\n"

            torch.save(self.state_dict(), output_forge_finetuned_pkl)
            
            # Write to log file after each epoch
            if output_log_file is not None:
                with open(output_log_file, 'a') as file:
                    file.write(epoch_summary + "\n")

            # Update learning rate scheduler based on epoch accuracy
            # ReduceLROnPlateau monitors accuracy and reduces LR when stuck
            scheduler.step(epoch_accuracy)

            # Shuffle SAT instances for next epoch
            np.random.shuffle(sat_files)

        # Close Gurobi environment
        gurobi_env.close()

    def _mip_model_to_gapinfo(self, mip_model: gp.Model, problem_type: str) -> GapInfo:
        """
        Predict integral gap information for a Gurobi model.

        Process
        -------
        - Convert `mip_model` to `MIPInfo`.
        - Run `forward()` in eval mode.
        - Extract `integral_gap` head outputs (variable-level scores) and aggregate to a gap ratio.
        - Solve LP relaxation to obtain `lp_obj` and `lp_sol`, then compute `mip_obj` based on the predicted ratio.

        Parameters
        ----------
        mip_model : gurobipy.Model
            An already-loaded Gurobi model object.
        problem_type : str
            Problem-specific post-processing rules for `mip_obj` computation.

        Returns
        -------
        GapInfo
            Dataclass containing: lp_obj, lp_sol, mip_obj, mip_sol, gap_ratio
            The lp_obj and lp_sol are true values from solving the LP relaxation.n.
            The mip_obj is predicted using the gap ratio prediction from Forge.
            The mip_sol is set to None.

        Raises
        ------
        TypeError
            If `mip_model` is not a gurobipy.Model.
        """
        # Validate input type
        if not isinstance(mip_model, gp.Model):
            raise TypeError(f"Error: mip_model must be a gurobipy.model, got {type(mip_model).__name__}")

        # If not trained, warn (keeps previous behavior)
        if not self.is_trained:
            raise ValueError("Error: Forge is not trained and no pre-trained model path is given.")

        # Ensure module is in evaluation mode and on device
        self.eval()

        # Store Forge eval mode (to restore back) and set to eval only to generate embedding
        original_mode = self.is_eval_mode
        self.is_eval_mode = True

        # Convert MIP model to MIP info with feature tensor and PyG edge_index/edge_weight
        mipinfo = MIPProcessor._mip_model_to_mipinfo(mip_model)

        # Forward pass through trained Forge
        h_list, logits, loss, indices, codebook_ = self.forward(mipinfo.feature_tensor.to(self.device),
                                                                mipinfo.num_cons, mipinfo.num_vars,
                                                                mipinfo.edge_index.to(self.device),
                                                                mipinfo.edge_weight.to(self.device))
        # Restore original mode
        self.is_eval_mode = original_mode

        # Find LP optimal to calculate ratio
        # variable_proba = h_list[-2][mipinfo.num_cons:]
        integral_gap = h_list[-1][mipinfo.num_cons:]

        gap_ratio = torch.mean(integral_gap).item()

        # Read and solve the LP relaxation to generate initial objective value
        lp_model = mip_model.relax()
        lp_model.optimize()
        lp_obj = lp_model.ObjVal
        lp_sol = lp_model.Xn

        mip_obj = lp_obj
        # TODO: Try to replace with GRB.SENSE instead of Constants.MIN_PROBLEMS/MAX_PROBLEMS
        if problem_type in Constants.MIN_PROBLEMS:
            # Minimization: interpret gap_ratio as “how close MIP is to LP”
            gap_ratio += (self.integral_gap_safety_eps * gap_ratio)
            mip_obj = lp_obj + (lp_obj * (1 - gap_ratio))
        elif problem_type in Constants.MAX_PROBLEMS:
            # Maximization: interpret gap_ratio as mip_obj / lp_obj in (0, 1]
            # Small safety margin to the ratio to make sure we are not infeasible
            gap_ratio += (self.integral_gap_safety_eps * gap_ratio)
            mip_obj = lp_obj * gap_ratio
        else:
            raise ValueError(f"Error: Unknown problem type '{problem_type}' for mip_model_to_gapinfo")

        # Create GapInfo with true lp_obj, predicted ratio, and predicted mip_obj but without a mip solution
        gap_info = GapInfo(lp_obj=lp_obj, lp_sol=lp_sol, mip_obj=mip_obj, mip_sol=None, gap_ratio=gap_ratio)

        return gap_info

    @staticmethod
    def _validate_args(train_config_file_path) -> None:
        """Validates arguments for the constructor."""

        # Validate train_config_file_path
        check_true(train_config_file_path is not None,
                   ValueError("Error: train_config_file_path cannot be None"))
        check_true(isinstance(train_config_file_path, str),
                   TypeError(f"Error: train_config_file_path must be a string, "
                             f"got {type(train_config_file_path).__name__}"))
        check_true(train_config_file_path.strip(),
                   ValueError("Error train_config_file_path cannot be empty or whitespace"))

        import os
        check_true(os.path.exists(train_config_file_path),
                   FileNotFoundError(f"Error: Configuration file not found: {train_config_file_path}"))
        check_true(os.path.isfile(train_config_file_path),
                   ValueError(f"Error: train_config_file_path must be a file, "
                              f"not a directory: {train_config_file_path}"))
