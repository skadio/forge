import time
from typing import List, Callable, Optional

import dgl
import gurobipy as gp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from dgl.nn import SAGEConv

from forge.utils import check_true, Constants, _overwrite_if_given
# TODO consider changing with the other library to remove the code copy to a fix/static version
from vqgraph.vq import VectorQuantize

try:
    from gurobi_onboarder import init_gurobi
    gurobi_venv, GUROBI_FOUND = init_gurobi.initialize_gurobi()
except:
    gurobi_venv = gp.Env(empty=True)


class Forge(nn.Module):

    def __init__(self,
                 train_config_file_path: Optional[str] = Constants.default_train_config_file,
                 input_dim: Optional[int] = None,
                 hidden_dim: Optional[int] = None,
                 codebook_dim: Optional[int] = None,
                 dropout_ratio: Optional[float] = None,
                 activation: Optional[Callable] = None,
                 norm_type: Optional[str] = None,
                 codebook_size: Optional[int] = None,
                 lambda_edge: Optional[float] = None,
                 lambda_node: Optional[float] = None,
                 has_separate_codebooks: Optional[bool] = None,
                 orthogonal_reg_weight: Optional[float] = None,
                 has_integral_gap_head: Optional[bool] = None,
                 has_variable_proba_head: Optional[bool] = None,
                 is_eval_mode: Optional[bool] = None) -> None:
        """Initialize the Forge models.

            This module adapts ideas from VQ-Graph style architectures for Mixed Integer Programming (MIP) instances.
            It builds a graph embedding with GraphSAGE,
            Optionally, it applies
            prediction heads, and performs vector quantization to obtain a discrete representation that can
            be used for reconstruction and downstream heuristics.

            Parameters
            ----------
            train_config_file_path : Optional[str], default=Constants.default_train_config_file
                Path to a YAML configuration file that provides default training and model
                hyperparameters. When provided, values from this file are loaded and used
                unless explicitly overridden via constructor arguments. The path is validated
                by `_validate_args` and must point to an existing readable file; passing
                `None` will raise a ValueError. Typical keys expected in the file include
                `input_dim`, `hidden_dim`, `codebook_dim`, `dropout_ratio`, and other
                parameters documented below.
            input_dim : int, default=10
                Dimensionality of the raw node features provided in `feats` during `forward`.
                If `input_dim < hidden_dim` the models internally projects to `hidden_dim`
                    (stored as `updated_input_dim`) to allow a wider first hidden representation;
                Otherwise, it keeps the original size.
                This affects the width of all subsequent layers and the quantizer.
            hidden_dim : int, default=1024
                Target hidden embedding size for GraphSAGE layers and subsequent linear layers.
                Acts as the working dimensionality for message passing.
                Larger values increase models capacity and decoder parameter count,
                potentially improving reconstruction at the cost of memory.
            codebook_dim : int, default=1024
                Dimensionality of each code vector in the vector quantization (VQ) codebook(s).
                Can be set lower than `hidden_dim` to encourage compression, or equal for lossless capacity.
                Impacts the expressiveness of discrete embeddings used for mip vector representations.
            dropout_ratio : float, default=0.4
                Dropout applied after major transformation blocks (GraphSAGE layers and linear layer).
                Higher values regularize more aggressively.
                Lower values risk overfitting large training corpora of MIP instances.
            activation : callable, default=torch.nn.functional.relu
                Non-linearity used inside SAGEConv layers.
                Alternatives (e.g. `F.leaky_relu` or `torch.nn.GELU()`) changes gradient flow
                and may alter how sparse / dense the learned embeddings become.
            norm_type : str, default="none"
                Type of optional additional normalization applied via `self.norms` (if populated outside this snippet).
                When set to values other than "none", an auxiliary normalization module is expected at index 0,
                refining stability across instances. Setting "none" skips that step.
            codebook_size : int, default=5000
                Number of discrete codes available to the VQ module(s).
                Larger sizes increase capacity for representing structural diversity in MIP graphs.
                Smaller sizes enforce stronger sharing and can improve generalization,
                    but may hurt fine-grained reconstruction.
                Also, determines the length of the distribution vector returned by `mip_to_vector`.
            lambda_edge : float, default=1
                Weight scaling the edge reconstruction portion of the unsupervised loss.
                During training, the implementation alternates emphasizing edges vs. nodes,
                    by swapping this with `lamb_node`.
                Increasing `lamb_edge` pushes the quantizer to better reproduce bipartite adjacency patterns.
            lambda_node : float, default=1
                Weight scaling node feature reconstruction loss.
                Larger values bias learning toward accurate feature decoding rather than structural edge patterns.
            has_separate_codebooks : bool, default=False
                If True, constructs two independent VQ codebooks (`vq_node`, `vq_edge`) so that,
                    node feature and edge structural embeddings specialize separately.
                If False, a single shared VQ (`vq`) forces joint compression which can encourage,
                    feature–structure entanglement and code reuse.
            orthogonal_reg_weight : float, default=0.0
                Strength of orthogonal regularization passed to the VQ module(s).
                Non-zero values push code vectors toward mutual orthogonality,
                    reducing redundancy and encouraging diverse discrete assignments.
                Typically small (e.g. 0.1–0.5) if used.
            has_integral_gap_head : bool, default=False
                Enables a cut prediction head (`integral_gap_layer`) for,
                    LP gap / cut ratio estimation tasks.
                Used in `mip_to_lp_cut` workflows.
                When active, an additional scalar per variable is produced.
            has_variable_proba_head : bool, default=False
                Enables a probability prediction head (`variable_proba_layer`) for,
                    variable membership solution likelihood tasks (BCE loss).
                Activating this adds parameters and changes the forward outputs,
                    appending probability tensors to `h_list`.
                Required for warm-start and triplet training phases.
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

            Returns
            -------
            None

            Example
            -------
            # TODO test this example (add to tests)
            >> > from forge.embeddings import Forge
            >> > forge = Forge()
            >> > forge.pretrain(model_save_path="forge_pretrained.pth",
            ...                 train_list=my_mip_graph_dataset,
            ...                 log_path="training_log.txt")
        """

        super().__init__()

        self._validate_args(train_config_file_path)

        # Read all configs from the given file
        with open(train_config_file_path, 'r') as f:
            config = yaml.safe_load(f)

        # Default to config values, but overwrite if a value is given
        self.input_dim = _overwrite_if_given(config.get('input_dim'), input_dim)
        self.hidden_dim = _overwrite_if_given(config.get('hidden_dim'), hidden_dim)
        self.codebook_dim = _overwrite_if_given(config.get('codebook_dim'), codebook_dim)
        self.dropout_ratio = _overwrite_if_given(config.get('dropout_ratio'), dropout_ratio)
        self.activation = _overwrite_if_given(config.get('activation'), activation)
        self.norm_type = _overwrite_if_given(config.get('norm_type'), norm_type)
        self.codebook_size = _overwrite_if_given(config.get('codebook_size'), codebook_size)
        self.lambda_edge = _overwrite_if_given(config.get('lambda_edge'), lambda_edge)
        self.lambda_node = _overwrite_if_given(config.get('lambda_node'), lambda_node)
        self.has_separate_codebooks = _overwrite_if_given(config.get('has_separate_codebooks'), has_separate_codebooks)
        self.orthogonal_reg_weight = _overwrite_if_given(config.get('orthogonal_reg_weight'), orthogonal_reg_weight)
        self.has_integral_gap_head = _overwrite_if_given(config.get('has_integral_gap_head'), has_integral_gap_head)
        self.has_variable_proba_head = _overwrite_if_given(config.get('has_variable_prob_head'), has_variable_proba_head)
        self.is_eval_mode = _overwrite_if_given(config.get('is_eval_mode'), is_eval_mode)

        # Load additional parameters from config
        # TODO should these also be in the input param list as above?
        self.graph_sage_aggregation: str = config.get('graph_sage_aggregation')
        self.decoder_edge_dim: int = config.get('decoder_edge_dim')
        self.vq_decay: float = config.get('vq_decay')
        self.vq_commitment_weight: float = config.get('vq_commitment_weight')
        self.vq_is_cosine_sim: bool = config.get('vq_is_cosine_sim')

        # Update input dim if needed
        if input_dim < hidden_dim:
            self.updated_input_dim = hidden_dim
        else:
            self.updated_input_dim = input_dim

        # Set fields based on input parameters
        self.dropout = nn.Dropout(dropout_ratio)

        # Forge is initially not trained
        self.is_trained = False

        # Create layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Graph layers
        self.graph_layer_1 = SAGEConv(input_dim, self.updated_input_dim,
                                      activation=activation, aggregator_type=self.graph_sage_aggregation)
        self.graph_layer_2 = SAGEConv(self.updated_input_dim, self.updated_input_dim,
                                      activation=activation, aggregator_type=self.graph_sage_aggregation)

        # Linear layers
        self.linear = nn.Linear(self.updated_input_dim, self.updated_input_dim)
        self.integral_gap_layer = nn.Linear(self.updated_input_dim, 1) if self.has_integral_gap_head else None
        self.variable_proba_layer = nn.Linear(self.updated_input_dim, 1) if self.has_variable_proba_head else None

        # Batch Norm layers
        self.bn1 = nn.BatchNorm1d(self.updated_input_dim)
        self.bn2 = nn.BatchNorm1d(self.updated_input_dim)
        self.bn3 = nn.BatchNorm1d(self.updated_input_dim)

        # Node Decoder
        self.decoder_node = nn.Linear(self.updated_input_dim, input_dim)

        # Edge Decoders. Edges are decoded as product of two matrices
        self.decoder_edge_1 = nn.Linear(self.updated_input_dim, self.decoder_edge_dim)
        self.decoder_edge_2 = nn.Linear(self.updated_input_dim, self.decoder_edge_dim)

        # TODO: possible to remove/simplify now?
        # Separate codebooks for node feature reconstruction and edge reconstruction?
        if self.has_separate_codebooks:
            self.vq_node = VectorQuantize(dim=self.updated_input_dim,
                                          codebook_size=self.codebook_size,
                                          decay=self.vq_decay,
                                          commitment_weight=self.vq_commitment_weight,
                                          use_cosine_sim=self.vq_is_cosine_sim,
                                          orthogonal_reg_weight=self.orthogonal_reg_weight,
                                          codebook_dim=self.codebook_dim)

            self.vq_edge = VectorQuantize(dim=self.updated_input_dim,
                                          codebook_size=self.codebook_size,
                                          decay=self.vq_decay,
                                          commitment_weight=self.vq_commitment_weight,
                                          use_cosine_sim=self.vq_is_cosine_sim,
                                          orthogonal_reg_weight=self.orthogonal_reg_weight,
                                          codebook_dim=self.codebook_dim)
        else:
            self.vq = VectorQuantize(dim=self.updated_input_dim,
                                     codebook_size=self.codebook_size,
                                     decay=self.vq_decay,
                                     commitment_weight=self.vq_commitment_weight,
                                     use_cosine_sim=self.vq_is_cosine_sim,
                                     orthogonal_reg_weight=self.orthogonal_reg_weight,
                                     codebook_dim=self.codebook_dim)

    def forward(self, g, feats, num_cons, num_vars):
        """Forward pass of the Forge models.

        Parameters
        ----------
        g : dgl.DGLGraph
            Bipartite (constraint + variable) MIP graph in DGL format.
            Must contain edge weights in `g.edata['weight']`.
            The first `num_cons` nodes are interpreted as constraint nodes, and
            The remaining `num_vars` as variable nodes (ordering consistent with graph builder).
        feats : torch.Tensor
            Node feature matrix of shape (num_cons + num_vars, 10 zero padded features).
            Provided externally by the graph construction utilities.
            This will be transformed through GraphSAGE + linear layer.
        num_cons : int
            Number of constraint nodes at the head of `feats` / node index space.
            Used to slice embeddings when computing bipartite adjacency reconstruction and separating outputs.
        num_vars : int
            Number of variable nodes following the constraint nodes.
            Used for variable_proba and integrality_gap heads in downstream.

        Returns
        -------
        h_list : List[torch.Tensor]
            Ordered collection of intermediate / final representations. Layout (when
            `separate_codebooks=False` and heads enabled) is:
                0: `h`        - dense embedding after GraphSAGE + linear block (shape: [N, hidden])
                1: `quantized`- quantized latent before decoders (same shape as above)
                2: `quantized_node` - node feature reconstruction logits (shape: [N, 10])
                3: `quantized_edge_1` - edge factor matrix A (shape: [N, 32])
                4: `quantized_edge_2` - edge factor matrix B (shape: [N, 32])
                5: `prob` (optional) - variable membership probabilities (shape: [N, 1])
                6: `cut`  (optional) - variable-level cut score predictions (shape: [N, 1])
            When `separate_codebooks=True`, the quantized node / edge tensors are produced from
            distinct VQ modules; edge list position differs (no factor split unless decoded similarly).
        h : torch.Tensor
            Embedding after the linear + activation block (same as h_list[0]). Returned separately
            for convenience in downstream tasks that expect a single dense representation.
        loss : torch.Tensor | int
            Scalar reconstruction + commitment loss (and edge positive emphasis term) if
            `eval_only=False`; set to -1 when `eval_only=True` to signal inference-only mode.
        dist : torch.Tensor
            Distance matrix of shape (N, codebook_size) after squeeze, containing distances from
            each node embedding to every code vector. Used for computing code assignments and MIP
            instance distribution vectors.
        codebook : torch.Tensor | Tuple[torch.Tensor, torch.Tensor]
            If `separate_codebooks=False`, a single codebook tensor of shape (codebook_size,
            codebook_dim). Otherwise, a tuple `(codebook_node, codebook_edge)` each with that shape.

        Notes
        -----
        - Adjacency reconstruction uses two low-rank factor matrices (`quantized_edge_1`,
          `quantized_edge_2`) to approximate bipartite edges via (A A^T)(B B^T)^T then min-max rescale.
        - Feature and edge reconstruction losses are scaled by `lamb_node` and `lamb_edge` allowing
          alternating emphasis during training (`train_unsupervised`).
        - Commitment and (optionally) orthogonality regularization flow from the VectorQuantize
          modules to encourage discrete, non-redundant code usage.
        - Probability and cut heads operate on the quantized latent, not the pre-quantization `h`.
        """

        # Input
        h = feats

        # List to hold intermediate layers
        h_list = []

        # Graph SAGE Layer 1
        h = self.graph_layer_1(g, h, edge_weight=g.edata['weight'])
        h = self.bn1(h)
        if self.norm_type != "none":
            h = self.norms[0](h)
        h = self.dropout(h)

        # GraphSAGE Layer 2
        h = self.graph_layer_2(g, h, edge_weight=g.edata['weight'])
        h = self.bn2(h)
        h = self.dropout(h)

        # Linear Layer
        h = self.linear(h)
        h = F.relu(h)
        h = self.bn3(h)
        h = self.dropout(h)

        # Save output at this stage (#TODO: i don't see a "save" here?)
        # This is going to be our "embedding" of the input graph
        h_list.append(h)

        # TODO is this needed? or remove? or comment out?
        # The "embedding" is passed into the prob head and the cut head below
        # if self.prob_head:
        #     prob = F.sigmoid(self.prob_layer(h))

        # if self.cut_head:
        #     cut = F.sigmoid(self.cut_layer(h))

        # The same "embedding" is then passed into the vector quantizer below
        # TODO potential to remove/simplify?
        if self.has_separate_codebooks:
            quantized_edge, _, commit_loss_edge, dist, codebook_edge = self.vq_edge(h)
            quantized_node, _, commit_loss_node, dist, codebook_node = self.vq_node(h)
            quantized_edge = self.decoder_edge(quantized_edge)
            quantized_node = self.decoder_node(quantized_node)
        else:
            quantized, _, commit_loss, dist, codebook = self.vq(h)
            quantized_node = self.decoder_node(quantized)
            quantized_edge_1 = self.decoder_edge_1(quantized)
            quantized_edge_2 = self.decoder_edge_2(quantized)

        # The "embedding" is passed into the prob head and the cut head below
        if self.has_variable_proba_head:
            prob = F.sigmoid(self.variable_proba_layer(quantized))

        if self.has_integral_gap_head:
            cut = F.sigmoid(self.integral_gap_layer(quantized))

        # Training
        if not self.is_eval_mode:

            adj = g.adjacency_matrix().to_dense().to(feats.device)

            # Reconstruction Loss (other losses are calculated in training code)
            feature_rec_loss = self.lamb_node * F.mse_loss(feats, quantized_node)

            adj_quantized_1 = torch.matmul(quantized_edge_1, quantized_edge_1.t())
            adj_quantized_2 = torch.matmul(quantized_edge_2, quantized_edge_2.t())

            adj_quantized = torch.matmul(adj_quantized_1, adj_quantized_2.T)

            # Min Max Rescaling of Adjacency Matrix
            adj_quantized = (adj_quantized - adj_quantized.min()) / (adj_quantized.max() - adj_quantized.min())

            # Look Only at The Bipartite Part of the Graph
            adj = adj[num_cons:, :num_cons]
            adj_quantized = adj_quantized[num_cons:, :num_cons]

            # Higher Penalty for Not Recreating Positive Edges
            edge_scale = adj * 1
            diff = torch.square(adj - adj_quantized)
            diff *= edge_scale

            pos_edge_rec_loss = self.lamb_edge * torch.mean(diff)
            edge_rec_loss = self.lamb_edge * torch.sqrt(F.mse_loss(adj, adj_quantized))
            edge_rec_loss += pos_edge_rec_loss

        # Distance Matrix - Distance From Each Node's Embedding to Each Code in the Codebook
        dist = torch.squeeze(dist)
        h_list.append(quantized)
        h_list.append(quantized_node)
        h_list.append(quantized_edge_1)
        h_list.append(quantized_edge_2)

        if self.has_variable_proba_head:
            h_list.append(prob)

        if self.has_integral_gap_head:
            h_list.append(cut)

        if self.has_separate_codebooks:
            if not self.is_eval_mode:
                loss = feature_rec_loss + edge_rec_loss + commit_loss_edge + commit_loss_node
            else:
                loss = -1
            return h_list, h, loss, dist, (codebook_node, codebook_edge)
        else:
            if not self.is_eval_mode:
                loss = feature_rec_loss + edge_rec_loss + commit_loss
            else:
                loss = -1
            return h_list, h, loss, dist, codebook

    def pretrain(self, model_save_path, train_list, epochs=None, steps_per_instance=None, lr=None, log_path=None):
         # TODO constants
         # pydocs

        # Resolve training hyperparameters: use provided overrides when not None, otherwise config defaults
        epochs = epochs if epochs is not None else self.train_epochs_default
        steps_per_instance = steps_per_instance if steps_per_instance is not None else self.steps_per_instance_default
        lr = lr if lr is not None else self.lr_default

        # Put module into training mode (use super to avoid recursion) and move to device
        super().train()
        self.to(self.device)

        main_loss_list = []
        skip_list = set()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)

        # Loop through data set
        t = ""
        for main_epoch in range(epochs):

            # Alternate between prioritizing node feature reconstruction and edge reconstruction
            if main_epoch % 2 == 0:
                self.lamb_node = 10
                self.lamb_edge = 1
            else:
                self.lamb_node = 1
                self.lamb_edge = 10

            loss_list = []
            epoch_start = time.time()

            # MIP instances in dataset
            for idx in range(10, len(train_list)):

                g, features, num_cons, num_vars = train_list[idx]

                # Some MIP instances are too large to fit in GPU memory
                if g.num_nodes() > 21000:
                    skip_list.add(idx)
                    continue

                for epoch in range(steps_per_instance):
                    # Compute loss and prediction
                    h_list, logits, loss, distances, codebook_ = self.forward(g.to(self.device), features.to(self.device),
                                                                              num_cons, num_vars)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                loss_list.append(loss.item())
                print("\rEpoch: ", main_epoch, "| Loss on Instance", idx, ": ", np.round(loss.item(), 3),
                      "| Mean Loss :", np.round(np.mean(loss_list), 3), end='')
                torch.cuda.empty_cache()

            print("\r")
            print()
            print("------")

            p_string = "Epoch: " + str(main_epoch) + "| Mean Loss: " + str(
                np.round(np.mean(loss_list), 3)) + "+/-" + str(
                np.round(np.std(loss_list), 3)) + " | Time For Epoch : " + str(
                np.round(time.time() - epoch_start, 3)) + "s"
            t += p_string + "\n"
            print(p_string, end='\n')
            print("------")
            print()

            torch.save(self.state_dict(), model_save_path)
            main_loss_list.append(np.round(np.mean(loss_list), 3))
            if log_path is not None:
                with open(log_path, 'a') as file:
                    file.write(t)

            self.is_trained = True

    @staticmethod
    def _validate_args(train_config_file_path) -> None:
        """
        Validates arguments for the constructor.
        """

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
