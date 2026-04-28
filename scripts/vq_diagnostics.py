"""
Diagnostic utilities for SAT pretraining codebook utilization analysis.

Use these functions to monitor:
- Code usage frequency
- Embedding norm distribution  
- VQ commitment loss
- Codebook perplexity
"""

import numpy as np
import torch
from collections import Counter
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


class VQDiagnostics:
    """Track VQ codebook health during training.
    
    Distinguishes between all-time code activation (misleading) and current 
    distribution health (accurate via perplexity and concentration metrics).
    """
    
    def __init__(self, codebook_size: int, embedding_dim: int):
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.code_usage_count = Counter()  # All-time usage for reference
        self.recent_code_usage = Counter()  # Recent window for accurate metrics
        self.embedding_norms = []
        self.embedding_means = []
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
        
        self.embedding_norms.extend(norms)
        self.embedding_means.extend(means)
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
        
        report_lines = [
            f"\n{prefix} CODEBOOK DIAGNOSTICS",
            f"  ✓ RECENT (Accurate Metric):",
            f"    Codes Used: {recent_codes_used} / {self.codebook_size} ({recent_util_pct:.1f}%)",
            f"    Top 3 Code Concentration: {concentration:.1f}% (lower is better)",
            f"    Perplexity: {avg_perplexity:.1f} (higher means more distributed)",
            f"",
            f"  📊 Reference Metrics:",
            f"    All-time Codes: {alltime_codes_used} / {self.codebook_size} ({alltime_util_pct:.1f}%) [may overlap with dead codes]",
            f"    Embedding Norm: mean={np.mean(self.embedding_norms):.4f}, std={np.std(self.embedding_norms):.4f}",
            f"    Avg Commitment Loss: {np.mean(self.commitment_losses):.6f}",
            f"    Top 5 Recent Codes: {', '.join(top_5_usage)}",
        ]
        
        # Alert if experiencing codebook collapse
        if concentration > 80:
            report_lines.append(f"")
            report_lines.append(f"  ⚠️  CODEBOOK COLLAPSE DETECTED: {concentration:.1f}% of assignments in top 3 codes!")
            report_lines.append(f"     This matches the histogram showing concentration on 1-3 codes.")
            report_lines.append(f"     Suggested fixes:")
            report_lines.append(f"     - Increase vq_commitment_weight (currently pulling embeddings to nearest code)")
            report_lines.append(f"     - Decrease vq_decay (makes EMA updates more aggressive)")
            report_lines.append(f"     - Increase orthogonal_reg_weight (encourages codebook diversity)")
            report_lines.append(f"     - Add code_reset strategy to revive dead codes")
        elif concentration > 60:
            report_lines.append(f"")
            report_lines.append(f"  ⚠️  CODE CONCENTRATION: {concentration:.1f}% in top 3 codes (watch this)")
        elif avg_perplexity >= 50:
            report_lines.append(f"")
            report_lines.append(f"  ✓ Good code distribution (perplexity {avg_perplexity:.1f})")
        
        return "\n".join(report_lines)
    
    def plot_diagnostics(self, output_path: str = None):
        """Visualize diagnostics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Code usage histogram
        codes = list(range(self.codebook_size))
        usage = [self.code_usage_count.get(c, 0) for c in codes]
        axes[0, 0].bar(range(min(100, self.codebook_size)), usage[:100], alpha=0.7)
        axes[0, 0].set_title('Code Usage Frequency (first 100 codes)')
        axes[0, 0].set_ylabel('Uses')
        
        # 2. Embedding norm distribution
        axes[0, 1].hist(self.embedding_norms, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Embedding Norm Distribution')
        axes[0, 1].set_xlabel('Norm')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Embedding mean distribution
        axes[1, 0].hist(self.embedding_means, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Embedding Mean Distribution')
        axes[1, 0].set_xlabel('Mean Value')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Metrics over time
        axes[1, 1].plot(self.batch_perplexities, label='Perplexity', marker='o')
        axes[1, 1].set_title('Perplexity Over Time')
        axes[1, 1].set_xlabel('Batch')
        axes[1, 1].set_ylabel('Perplexity')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved diagnostics plot to {output_path}")
        else:
            plt.show()
    
    def reset(self):
        """Reset counters for next epoch."""
        # Keep all-time count, reset recent window
        self.recent_code_usage.clear()
        self.all_indices_recent.clear()
        self.embedding_norms.clear()
        self.embedding_means.clear()
        self.commitment_losses.clear()
        self.batch_perplexities.clear()


def compute_codebook_coverage(vq_module, 
                             dataloader: List[Tuple],
                             device: torch.device) -> Dict:
    """
    Analyze codebook coverage across entire dataset.
    
    Parameters
    ----------
    vq_module : VectorQuantize
        The VQ module to analyze
    dataloader : List[Tuple]
        List of (embeddings, labels) tuples
    device : torch.device
        Device to run on
        
    Returns
    -------
    dict
        Coverage statistics
    """
    code_usage = Counter()
    codebook = vq_module.codebook  # shape: (codebook_size, codebook_dim)
    codebook_size = codebook.shape[0]
    
    with torch.no_grad():
        for embeddings, _ in dataloader:
            embeddings = embeddings.to(device)
            quantized, indices, _ = vq_module(embeddings)
            
            indices_np = indices.squeeze().cpu().numpy()
            for code_id in indices_np:
                code_usage[int(code_id)] += 1
    
    codes_used = len(code_usage)
    utilization = 100.0 * codes_used / codebook_size
    
    # Codebook norm analysis
    codebook_norms = torch.norm(codebook, dim=1)
    
    return {
        'codes_used': codes_used,
        'codebook_size': codebook_size,
        'utilization_pct': utilization,
        'code_usage_distribution': dict(code_usage),
        'codebook_norm_mean': codebook_norms.mean().item(),
        'codebook_norm_std': codebook_norms.std().item(),
        'codebook_norm_min': codebook_norms.min().item(),
        'codebook_norm_max': codebook_norms.max().item(),
    }
