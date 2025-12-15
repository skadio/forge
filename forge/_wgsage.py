import torch
from torch import nn
from torch_geometric.nn import MessagePassing


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

    def forward(self,
                x: torch.Tensor,  # [N, F_in]
                edge_index: torch.Tensor,  # [2, E]
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
        out = self.propagate(edge_index=edge_index,
                             x=x,
                             edge_weight=edge_weight,
                             x_root=x_root)

        return out

    def message(self,
                x_j: torch.Tensor,  # neighbor features [E, F_out]
                edge_weight: torch.Tensor,  # [E]
                ) -> torch.Tensor:
        # Weight each neighbor contribution
        return edge_weight.view(-1, 1) * x_j

    def update(self,
               aggr_out: torch.Tensor,  # aggregated neighbor messages [N, F_out]
               x_root: torch.Tensor,  # root node features [N, F_in]
               ) -> torch.Tensor:
        # Combine neighbor messages with transformed root node features
        return aggr_out + self.lin_root(x_root)


def blockwise_loss(quantized_one: torch.Tensor,
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N = quantized_one.size(0)

    # Running sums (kept as tensors for autograd)
    sum_sq = torch.zeros((), device=device)
    sum_sq_pos = torch.zeros((), device=device)
    count_all = 0

    # Potential TODO? 
    # Ideally, only reconstruct either the top right or bottom left 
    # quadrant of the bipartite adjacency block. (Look at _get_edge_index_weight in processor.py)
    # Top right is of shape adj[:num_cons, num_cons:]


    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        # Batch × d, move to device
        block_one = quantized_one[start:end, :].to(device)
        recon_block = block_one @ block_one.T

        block_two = quantized_two[start:end, :].to(device)
        recon_block_two = block_two @ block_two.T

        # Merge both reconstructions
        recon_block = recon_block @ recon_block_two.T

        # Min-max rescaling per block to [0, 1]
        block_min = recon_block.min()
        block_max = recon_block.max()
        recon_block = (recon_block - block_min) / (block_max - block_min + 1e-8)

        tgt_block = target_adj_cpu[start:end, :].to(device)

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
