# Forge Architecture
Forge is a graph neural network designed to learn discrete representations of Mixed Integer Programming (MIP) instances. Here's the architecture breakdown:

* Bipartite graph construction from MIP instances
* Feature tensor creation for constraints and variables
* GraphSAGE-based encoder with edge-weighted message passing
* Vector quantization for discrete embeddings
* Decoder to reconstruct node features and adjacency
* Training objectives including reconstruction losses and commitment loss
* Fine-tuning heads for specific prediction tasks
* Parameter count analysis

## Bipartite Graph Construction
MIP instances are converted to bipartite graphs where:
* Constraint nodes (first num_cons rows): One node per constraint 
* Variable nodes (next num_vars rows): One node per decision variable 
* Edges: Connect variables to constraints they appear in, weighted by constraint coefficients

### Feature Matrix Construction
From _get_feature_tensor_num_cons_num_vars in processor.py:
* Constraint features (4 dims):
  * Is equality constraint (=)
  * Is less-than constraint (<)
  * Is greater-than constraint (>)
  * Right-hand side (RHS) value
* Variable features (6 dims):
  * Is continuous variable
  * Is binary variable
  * Is integer variable
  * Objective coefficient
  * Has lower bound
  * Has upper bound
* Features are min-max normalized per column to [0, 1].

### Stacking:
```
constraint_features = [4 dims] + [6 zeros] → 10 dims
variable_features   = [4 zeros] + [6 dims] → 10 dims
feature_tensor = stack([constraints, variables])  # shape: (num_cons + num_vars, 10)
```

### Edge Index & Weights
From _get_edge_index_weight:
* Uses Gurobi's constraint coefficient matrix (getA())
* Creates symmetric bipartite adjacency (constraints ↔ variables)
* Edge weights are the normalized constraint coefficients

## Input Features to GraphSAGE
The first graph sage layer receives:
* x: feature_tensor (N, 10) where N = num_cons + num_vars
* edge_index: (2, E) connectivity
* edge_weight: (E,) normalized coefficients

## Encoder Architecture
Input (N,10) → GraphSAGE Layers → Linear → Vector Quantization

1. **GraphSAGE Layer 1:** EdgeWeightedSAGEConv(10 → 1024) with mean aggregation
   * Supports edge weights for coefficient-aware message passing
   * Followed by ReLU, BatchNorm, optional normalization, and dropout (0.4)
Input: x(N, 10), edge_index (2, E), edge_weight (E,)
├── lin_neigh(x) → (N, 1024)  # transform for message passing
├──   *  transforms features before message passing (neighbor contribution)
├── propagate: aggregate weighted neighbor messages → (N, 1024)
├── lin_root(x) → (N, 1024)   # transform root features.
├──   * transforms the node's own features (self-loop contribution)
├──   * transforms the original node features (the "root" node being updated) before combining them with aggregated neighbor messages.
└── output = aggr_out + lin_root(x) → (N, 1024)
├──   * The final output combines both: aggr_out + lin_root(x_root)
├──   * Allowing the model to learn different weights for self-features vs. neighbor-aggregated features.
 └── ReLU → BatchNorm → Dropout → (N, 1024)
   
2. **GraphSAGE Layer 2:** EdgeWeightedSAGEConv(1024 → 1024)
   * Same activation/norm/dropout pattern
Input:  (N, 1024)
└── Same pattern → (N, 1024)
   
3. **Linear Block:** Linear(1024 → 1024) + ReLU + BatchNorm + Dropout
Input:  (N, 1024)
└── Linear → ReLU → BatchNorm → Dropout → (N, 1024)

4. **Vector Quantization:** Maps continuous embeddings to discrete codes
Input:  (N, 1024)
├── Find nearest codebook vector for each node
├── Codebook: (5000, 1024)
└── quantized_output → (N, 1024)  # discrete embeddings
   * Codebook size: 5000 codes
   * Codeword dimension: 1024
   * Uses Exponential Moving Average (EMA) decay (0.8) and commitment loss (0.25)
   * EMA decay in vector quantization controls how the codebook embeddings are updated during training.
   * In this model (vq_decay: 0.8), instead of updating codebook vectors via gradient descent, they're updated using an exponential moving average:
   * codebook_new = decay * codebook_old + (1 - decay) * new_embeddings 
   * With decay = 0.8: 80% of the old codebook vector is retained and 20% comes from new input embeddings assigned to that code 
   * Why use EMA? 
   * More stable training than gradient-based updates 
   * Avoids codebook collapse (unused codes)
   * Smoother convergence for discrete representations 
   * Higher decay (closer to 1.0) → slower, more stable updates 
   * Lower decay (closer to 0.0) → faster adaptation, potentially less stable

## Decoder Architecture
* **Node Decoder:** Linear(1024 → 10) — reconstructs node features
  * quantized (N, 1024) → Linear → (N, 10)  # reconstructs feat
* **Edge Decoders:** Two Linear(1024 → 32) layers producing factor matrices 
  * Adjacency reconstructed via (A @ Aᵀ) * (B @ Bᵀ)
  * quantized (N, 1024) → edge_decoder_1 → (N, 32)  # factor A
  * quantized (N, 1024) → edge_decoder_2 → (N, 32)  # factor B
  * Reconstruction: (A @ Aᵀ) * (B @ Bᵀ) → (N, N)  # adjacency matrix

## Training Objective
* Feature reconstruction loss (MSE)
* Edge reconstruction loss (blockwise bipartite adjacency)
* VQ commitment loss 
* λ_node and λ_edge alternate between 1 and 10 each epoch

## Fine-tuning Heads
* Integral gap prediction: Linear(1024 → 1)
* Variable probability: Linear(1024 → 1) with sigmoid

## Edge Weights in the Model
Edge weights enter during message passing in the GraphSAGE layers.
In _wgsage.py, the message() method:
```
def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
    return edge_weight.view(-1, 1) * x_j  # Weight each neighbor contribution
```
* Each neighbor's transformed features are scaled by the edge weight (normalized constraint coefficient) before aggregation. 
* This happens in both GraphSAGE layers.

## Edge Weights in the Loss
Edge weights are not directly used in the loss computation. Instead, the loss reconstructs the adjacency structure:

In faster_blockwise_loss():
```
# Reconstruct adjacency from learned factors
recon = (q1_var @ q1_con.T) * (q2_var @ q2_con.T)

# Compare to target adjacency (binary: edge exists or not)
tgt_block = target_adj_cpu[var_idx_start:var_idx_end, :num_cons]
```
* The target adjacency target_adj_cpu contains the original edge weights from the constraint coefficient matrix. 
* The loss computes MSE between:
  * Reconstructed adjacency (from decoder factors)
  * Target adjacency (normalized coefficients, not just 0/1)
* So edge weights appear in the loss implicitly as the reconstruction target, 
* but the loss function itself doesn't weight edges differently.
* It treats all bipartite entries equally (with extra emphasis on positive edges via edge_scale).

## Model Size

### Encoder
* GraphSAGE Layer 1 (EdgeWeightedSAGEConv(10 → 1024)):
  * lin_neigh: 10 × 1024 = 10,240 (no bias)
  * lin_root: 10 × 1024 + 1024 = 11,264 (with bias)
* GraphSAGE Layer 2 (EdgeWeightedSAGEConv(1024 → 1024)):
  * lin_neigh: 1024 × 1024 = 1,048,576
  * lin_root: 1024 × 1024 + 1024 = 1,049,600
* Linear Block (Linear(1024 → 1024)):
  * 1024 × 1024 + 1024 = 1,049,600
* BatchNorm Layers (3 total, each 1024 dim):
  * 2 params each (γ, β): 3 × 1024 × 2 = 6,144
### Vector Quantization
* Codebook (5000 codes × 1024 dim):
  * 5,120,000 (EMA-updated, may not count as trainable)

### Decoder
* Node Decoder (Linear(1024 → 10)):
  * 1024 × 10 + 10 = 10,250
* Edge Decoder 1 (Linear(1024 → 32)):
  * 1024 × 32 + 32 = 32,800
* Edge Decoder 2 (Linear(1024 → 32)):
  * 1024 × 32 + 32 = 32,800
 
### Total Parameter Count
* Encoder (GraphSAGE + Linear) ~3,169,280
* BatchNorms 6,144
* VQ Codebook 5,120,000
* Decoders 75,850
* Total ~8.37M
* Trainable parameters (excluding codebook): ~3.25M

## Cookbook Parameters
The VQ codebook is excluded from "trainable parameters" because it's not updated via gradient descent.
#### How VQ Codebook Updates Work
From your config (vq_decay: 0.8), the codebook uses Exponential Moving Average (EMA) updates:
#### During forward pass:
```
codebook_new = 0.8 * codebook_old + 0.2 * assigned_embeddings
```
* This happens outside the optimizer's parameter list—no gradients flow through the codebook directly.
  * Gradient descent (backprop) Trainable Counted by model.parameters()
  * EMA (moving average) Not trainable Not counted by model.parameters()

* In PyTorch, EMA-updated tensors are typically registered as buffers (via register_buffer), not parameters. So when you call:
```
sum(p.numel() for p in model.parameters())  # ~3.25M
```
* The codebook's 5.12M entries aren't included.
* The 5.12M still exists and learns
* The codebook is learned—it adapts during training via EMA—but it's not "trainable" in the optimizer sense. 
* That's why the total parameter count separates:
  * Trainable: ~3.25M (optimizer updates these)
  * Total: ~8.37M (includes EMA codebook)

## Model Size Verification
* The pretrained model is 51.5 MiB
* The saved pickle contains model.state_dict(), which includes:
* Total raw parameters: ~8.37M × 4 bytes ≈ 31.9 MiB
The additional ~20 MiB comes from:
* Pickle/torch serialization overhead (metadata, tensor headers, dtype info)
* Optimizer state if accidentally saved (momentum buffers double the size)
* Compression ratio variations

| Component | Parameters | Size \(float32\) |
|---|---|---|
| Trainable \(optimizer-updated\) | \~3.25M | \~12.4 MiB |
| VQ Codebook \(EMA-updated buffer\) | 5,000 × 1,024 = 5.12M | \~19.5 MiB |
| Other buffers \(BatchNorm running stats, etc.\) | \~6K | \~0.02 MiB |

Quick sanity check: 
* If total_elements is around 8.3–8.4M, the file size is correct. 
* If it's significantly higher (~16M+), the optimizer state may have been saved inadvertently.
```
import torch
state = torch.load('forge_pretrained.pkl', map_location='cpu')
total_elements = sum(t.numel() for t in state.values())
print(f"Total elements: {total_elements:,}")  # Should be ~8.37M
print(f"Raw size: {total_elements * 4 / 1024**2:.1f} MiB")
```