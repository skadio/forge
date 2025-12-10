import os
import pickle
from typing import Union, NamedTuple

import numpy as np
import scipy.sparse as sp

import torch
from torch import nn
from torch_geometric.nn import MessagePassing

Num = Union[int, float]
"""Num type is defined as integer or float."""


class Constants(NamedTuple):
    """
    Constant values used by the modules.
    """

    NUM_VARIABLE_FEATURES = 6
    NUM_CONSTRAINT_FEATURES = 4

    # Forge Model Types
    FORGE_PRE_TRAIN = "forge_pretrain"
    FORGE_FINE_TUNE_INTEGRAL_GAP = "forge_fine_tune_integral_gap"
    FORGE_FINE_TUNE_VARIABLE_PROBA = "forge_fine_tune_variable_proba"

    # Names
    _DATA_DIR_NAME = "data"
    _FORGE_DIR_NAME = "forge"
    _CONFIGS_DIR_NAME = "configs"
    _MODELS_DIR_NAME = "models"
    _TEST_DIR_NAME = "tests"
    _TRAIN_CONFIG_NAME = "train_config.yaml"
    _MIPINFO_NAME = "mip_to_mipinfo.pkl"
    _FORGE_PKL_NAME = "forge_pretrained.pkl"
    _FORGE_LOG_NAME = "forge_pretrain.log"

    # Paths
    _CONST_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = _CONST_FILE_DIR + os.sep + ".." + os.sep + _DATA_DIR_NAME
    DATA_TEST_DIR = _CONST_FILE_DIR + os.sep + ".." + os.sep + _TEST_DIR_NAME + os.sep + _DATA_DIR_NAME
    MODELS_DIR = _CONST_FILE_DIR + os.sep + ".." + os.sep + _MODELS_DIR_NAME
    CONFIGS_DIR = _CONST_FILE_DIR + os.sep + ".." + os.sep + _FORGE_DIR_NAME + os.sep + _CONFIGS_DIR_NAME

    default_train_config_yaml = _CONST_FILE_DIR + os.sep + _CONFIGS_DIR_NAME + os.sep + _TRAIN_CONFIG_NAME
    default_mip_to_mipinfo_pkl = _CONST_FILE_DIR + os.sep + ".." + os.sep + _TEST_DIR_NAME + os.sep + _MIPINFO_NAME
    default_forge_pretrained_pkl = _CONST_FILE_DIR + os.sep + ".." + os.sep + _TEST_DIR_NAME + os.sep + _FORGE_PKL_NAME
    default_forge_log_file = _CONST_FILE_DIR + os.sep + ".." + os.sep + _TEST_DIR_NAME + os.sep + _FORGE_LOG_NAME
    """The default train config yaml file."""


def params(torch_model):
    """
    Return number of parameters in a torch model
    """
    return sum(p.numel() for p in torch_model.parameters() if p.requires_grad)


def normalize(mx):
    """
    Row-normalize sparse matrix
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    adj = normalize(adj + sp.eye(adj.shape[0]))
    return adj


def overwrite_if_given(default_val, val):
    return default_val if val is None else val


def check_false(expression: bool, exception: Exception) -> None:
    """
    Checks that given expression is false, otherwise raises the given exception.
    """
    if expression:
        raise exception


def check_true(expression: bool, exception: Exception) -> None:
    """
    Checks that given expression is true, otherwise raises the given exception.
    """
    if not expression:
        raise exception


def load_pickle(pickle_file: str):
    """
    Returns the loaded pickle object.
    """
    with open(pickle_file, 'rb') as infile:
        return pickle.load(infile)


def save_pickle(obj, pickle_file) -> None:
    """
    Save serializable object as pickle file.
    """
    with open(pickle_file, 'wb') as fp:
        pickle.dump(obj, fp)

def copy_params(old_model, new_model):
    small_state_dict = old_model.state_dict()
    large_state_dict = new_model.state_dict()

    for name, param in small_state_dict.items():
        if name in large_state_dict:
            large_state_dict[name].copy_(param)


class EdgeWeightedSAGEConv(MessagePassing):
    """
    GraphSAGE-style convolution that supports scalar edge weights.

    Args
    ----
    in_channels : int
        Input feature dimension.
    out_channels : int
        Output feature dimension.
    aggr : str, default="mean"
        Neighborhood aggregation: "mean", "add", "max", etc.
    """

    def __init__(self, in_channels: int, out_channels: int, aggr: str = "mean"):
        super().__init__(aggr=aggr)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_root = nn.Linear(in_channels, out_channels, bias=True)

    def forward(
        self,
        x: torch.Tensor,                # [N, F_in]
        edge_index: torch.Tensor,       # [2, E]
        edge_weight: torch.Tensor | None = None,  # [E]
    ) -> torch.Tensor:
        
        # x_root is the "center" node embedding (used in update).
        x_root = x

        # Transform neighbor features before message passing
        x = self.lin_neigh(x)

        # Default edge_weight to ones if not provided
        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.size(1))

        # Run message passing:
        #   - message() is called for each edge (i -> j)
        #   - aggregate() (from MessagePassing) combines messages per node
        #   - update() combines aggregated message with root features
        out = self.propagate(
            edge_index=edge_index,
            x=x,
            edge_weight=edge_weight,
            x_root=x_root,
        )

        return out

    def message(
        self,
        x_j: torch.Tensor,              # neighbor features [E, F_out]
        edge_weight: torch.Tensor,      # [E]
    ) -> torch.Tensor:
        # Weight each neighbor contribution
        return edge_weight.view(-1, 1) * x_j

    def update(
        self,
        aggr_out: torch.Tensor,         # aggregated neighbor messages [N, F_out]
        x_root: torch.Tensor,           # root node features [N, F_in]
    ) -> torch.Tensor:
        # Combine neighbor messages with transformed root node features
        return aggr_out + self.lin_root(x_root)
    

def blockwise_loss(self,
                   quantized_one: torch.Tensor,
                   quantized_two: torch.Tensor,
                   target_adj_cpu: torch.Tensor,
                   num_cons: int,
                   lambda_edge: float = 1.0,
                   batch_size: int = 1024) -> torch.Tensor:
    """Blockwise edge reconstruction loss on the bipartite submatrix.

    This computes the loss only on the bipartite portion of the adjacency
    matrix (rows ``num_cons:`` and columns ``:num_cons``), in small row
    blocks to keep GPU memory usage bounded.

    Parameters
    ----------
    self : nn.Module
        Module that owns ``device`` attribute (e.g., ``Forge``).
    quantized_one : torch.Tensor
        First decoded edge factor matrix of shape ``(N, d)``.
    quantized_two : torch.Tensor
        Second decoded edge factor matrix of shape ``(N, d)``.
    target_adj_cpu : torch.Tensor
        Dense adjacency matrix on CPU of shape ``(N, N)``.
    num_cons : int
        Number of constraint nodes; the bipartite block is
        ``rows[num_cons:, :]`` and ``cols[:num_cons]``.
    lambda_edge : float, optional
        Weighting factor for edge reconstruction loss.
    batch_size : int, optional
        Number of variable rows to process per block.

    Returns
    -------
    torch.Tensor
        Scalar tensor with edge reconstruction loss (including positive-edge
        emphasis), suitable for backpropagation.
    """

    device = self.device
    N = quantized_one.size(0)

    # Running sums (kept as tensors for autograd)
    sum_sq = torch.zeros((), device=device)
    sum_sq_pos = torch.zeros((), device=device)
    count_all = 0

    # Process only variable rows: indices [num_cons, N)
    for start in range(num_cons, N, batch_size):
        end = min(start + batch_size, N)

        # Batch × d, move to device
        block_one = quantized_one[start:end, :].to(device)

        # Reconstruct bipartite edges only: variables (rows) × constraints (cols)
        # Shape: [B, num_cons]
        recon_block = block_one @ block_one[:num_cons, :].T

        block_two = quantized_two[start:end, :].to(device)
        recon_block_two = block_two @ block_two[:num_cons, :].T

        recon_block = recon_block @ recon_block_two

        # Min-max rescaling per block to [0, 1]
        block_min = recon_block.min()
        block_max = recon_block.max()
        recon_block = (recon_block - block_min) / (block_max - block_min + 1e-8)

        # Target slice: same bipartite rows/cols, moved to device
        tgt_block = target_adj_cpu[start:end, :num_cons].to(device)

        # Squared error
        diff = tgt_block - recon_block
        sq = diff.pow(2)

        # Mask for positive edges (adjacency > 0)
        edge_scale = (tgt_block > 0).to(recon_block.dtype)
        sq_pos = sq * edge_scale

        sum_sq = sum_sq + sq.sum()
        sum_sq_pos = sum_sq_pos + sq_pos.sum()
        count_all += sq.numel()

    if count_all == 0:
        # No bipartite entries; return zero loss tensor on correct device
        return torch.zeros((), device=device)

    mse = sum_sq / count_all
    pos_mean = sum_sq_pos / count_all

    # Match structure of original loss:
    return lambda_edge * (torch.sqrt(mse) + pos_mean)