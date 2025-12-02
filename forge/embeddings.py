import time
from typing import List, Callable, Optional, Tuple, Union, Dict

import dgl
import gurobipy as gp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from dgl.nn import SAGEConv

from forge.labeler import MIPLabeler, GapInfo
from forge.processor import MIPInfo, MIPEmbeddings, MIPProcessor
from forge.utils import check_true, Constants, overwrite_if_given
# TODO consider changing with the other library to remove the code copy to a fix/static version
from vqgraph.vq import VectorQuantize


class Forge(nn.Module):
    """Forge model: GraphSAGE encoder with Vector Quantization for MIP graphs.
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
                 activation: Optional[Callable] = None,
                 norm_type: Optional[str] = None,
                 lambda_edge: Optional[float] = None,
                 lambda_node: Optional[float] = None,
                 orthogonal_reg_weight: Optional[float] = None,
                 is_eval_mode: Optional[bool] = None) -> None:
        """Initialize the Forge models.

            This module adapts ideas from VQ-Graph style architectures for Mixed Integer Programming (MIP) instances.
            It builds a graph embedding with GraphSAGE,
            Optionally, it applies
            prediction heads, and performs vector quantization to obtain a discrete representation that can
            be used for reconstruction and downstream heuristics.

            Parameters
            ----------
            train_config_yaml : Optional[str], default=Constants.default_train_config_yaml
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
            activation : callable, default=torch.nn.functional.relu
                Non-linearity used inside SAGEConv layers.
                Alternatives (e.g. `F.leaky_relu` or `torch.nn.GELU()`) changes gradient flow
                and may alter how sparse / dense the learned embeddings become.
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

            Returns
            -------
            None
        """

        super().__init__()

        self._validate_args(train_config_yaml)

        # Read all configs from the given file
        with open(train_config_yaml, 'r') as f:
            config = yaml.safe_load(f)

        # Default to config values, but overwrite if a value is given
        self.input_dim: int = overwrite_if_given(config.get('input_dim'), input_dim)
        self.hidden_dim: int = overwrite_if_given(config.get('hidden_dim'), hidden_dim)
        self.codeword_dim: int = overwrite_if_given(config.get('codeword_dim'), codeword_dim)
        self.codebook_size: int = overwrite_if_given(config.get('codebook_size'), codebook_size)
        self.dropout_ratio: float = overwrite_if_given(config.get('dropout_ratio'), dropout_ratio)
        self.activation: Callable = overwrite_if_given(config.get('activation'), activation)
        self.norm_type: str = overwrite_if_given(config.get('norm_type'), norm_type)
        self.lambda_edge: float = overwrite_if_given(config.get('lambda_edge'), lambda_edge)
        self.lambda_node: float = overwrite_if_given(config.get('lambda_node'), lambda_node)
        self.orthogonal_reg_weight: float = overwrite_if_given(config.get('orthogonal_reg_weight'),
                                                               orthogonal_reg_weight)
        self.is_eval_mode: bool = overwrite_if_given(config.get('is_eval_mode'), is_eval_mode)

        # Load additional parameters
        # TODO should these also be in __init__ input to overwrite as above? Seems "fixed", no need to overwrite?
        self.graph_sage_aggregation: str = config.get('graph_sage_aggregation')
        self.decoder_edge_dim: int = config.get('decoder_edge_dim')
        self.vq_decay: float = config.get('vq_decay')
        self.vq_commitment_weight: float = config.get('vq_commitment_weight')
        self.vq_is_cosine_sim: bool = config.get('vq_is_cosine_sim')

        # Load default training parameters
        self.epochs: int = config.get('epochs')
        self.steps_per_instance: int = config.get('steps_per_instance')
        self.learning_rate: float = config.get('learning_rate')
        self.max_dgl_nodes: int = config.get('max_dgl_nodes')

        # Load seed
        self.seed: int = config.get('seed')

        # Initialize without downstream heads. load_model() can set these later.
        self.has_integral_gap_head: bool = False
        self.has_variable_proba_head: bool = False

        # Update input dim if needed
        # TODO when/why would this happen? some commentary?
        if self.input_dim < self.hidden_dim:
            self.updated_input_dim = self.hidden_dim
        else:
            self.updated_input_dim = self.input_dim

        # Set fields based on input parameters
        self.dropout = nn.Dropout(dropout_ratio)

        # Forge is initially not trained
        self.is_trained = False

        # Create layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Set device, if GPU is available use it
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Graph layers
        self.graph_layer_1 = SAGEConv(self.input_dim, self.updated_input_dim,
                                      activation=activation, aggregator_type=self.graph_sage_aggregation)
        self.graph_layer_2 = SAGEConv(self.updated_input_dim, self.updated_input_dim,
                                      activation=activation, aggregator_type=self.graph_sage_aggregation)

        # Linear layers
        self.linear = nn.Linear(self.updated_input_dim, self.updated_input_dim)
        self.integral_gap_layer = nn.Linear(self.updated_input_dim, 1) if self.has_integral_gap_head else None
        self.variable_proba_layer = nn.Linear(self.updated_input_dim, 1) if self.has_variable_proba_head else None

        # Batch norm layers
        self.bn1 = nn.BatchNorm1d(self.updated_input_dim)
        self.bn2 = nn.BatchNorm1d(self.updated_input_dim)
        self.bn3 = nn.BatchNorm1d(self.updated_input_dim)

        # Node decoder
        self.decoder_node = nn.Linear(self.updated_input_dim, input_dim)

        # Edge decoders. Edges are decoded as product of two matrices
        self.decoder_edge_1 = nn.Linear(self.updated_input_dim, self.decoder_edge_dim)
        self.decoder_edge_2 = nn.Linear(self.updated_input_dim, self.decoder_edge_dim)

        # Vector quantization module
        self.vq = VectorQuantize(dim=self.updated_input_dim,
                                 codebook_size=self.codebook_size,
                                 decay=self.vq_decay,
                                 commitment_weight=self.vq_commitment_weight,
                                 use_cosine_sim=self.vq_is_cosine_sim,
                                 orthogonal_reg_weight=self.orthogonal_reg_weight,
                                 codebook_dim=self.codeword_dim)

    def load_model(self, input_forge_pkl, model_type=Constants.FORGE_PRE_TRAIN):

        if self.is_trained:
            print("Warning: Forge model is already trained, NOT loading weights, quitting!!")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self = self.to(device)

        if model_type == Constants.FORGE_PRE_TRAIN:
            self.has_integral_gap_head = False
            self.has_variable_proba_head = False
        elif model_type == Constants.FORGE_FINE_TUNE_INTEGRAL_GAP:
            # TODO why are we setting both to true for integral gap??
            self.has_integral_gap_head = True
            self.has_variable_proba_head = True
        elif model_type == Constants.FORGE_FINE_TUNE_VARIABLE_PROBA:
            self.has_integral_gap_head = False
            self.has_variable_proba_head = True

        self.load_state_dict(torch.load(input_forge_pkl, map_location=device))
        self.is_trained = True

    def forward(self, dgl_graph: dgl.DGLGraph, feature_matrix: torch.Tensor, num_cons: int, num_vars: int) \
            -> Tuple[List[torch.Tensor], torch.Tensor, Union[torch.Tensor, int], torch.Tensor, Union[
                torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass of the Forge models.

        Parameters
        ----------
        dgl_graph : dgl.DGLGraph
            Bipartite (constraint + variable) MIP graph in DGL format.
            Must contain edge weights in `g.edata['weight']`.
            The first `num_cons` nodes are interpreted as constraint nodes, and
            The remaining `num_vars` as variable nodes (ordering consistent with graph builder).
        feature_matrix : torch.Tensor
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
        h : torch.Tensor
            Embedding after the linear + activation block (same as h_list[0]).
            Returned separately for convenience in downstream tasks that expect a single dense representation.
        loss : torch.Tensor | int
            Scalar reconstruction + commitment loss (and edge positive emphasis term) if `eval_only=False`;
            set to -1 when `eval_only=True` to signal inference-only mode.
        dist : torch.Tensor
            Distance matrix of shape (N, codebook_size) after squeeze,
            containing distances from each node embedding to every code vector.
            Used for computing code assignments and MIP instance distribution vectors.
        codebook : torch.Tensor | Tuple[torch.Tensor, torch.Tensor]
            If `separate_codebooks=False`, a single codebook tensor of shape (codebook_size, codebook_dim).
            Otherwise, a tuple `(codebook_node, codebook_edge)` each with that shape.

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
        h = feature_matrix

        # List to hold intermediate layers
        h_list = []

        # Graph SAGE Layer 1
        h = self.graph_layer_1(dgl_graph, h, edge_weight=dgl_graph.edata['weight'])
        h = self.bn1(h)
        if self.norm_type != "none":
            h = self.norms[0](h)
        h = self.dropout(h)

        # GraphSAGE Layer 2
        h = self.graph_layer_2(dgl_graph, h, edge_weight=dgl_graph.edata['weight'])
        h = self.bn2(h)
        h = self.dropout(h)

        # Linear Layer
        h = self.linear(h)
        h = F.relu(h)
        h = self.bn3(h)
        h = self.dropout(h)

        # Save output at this stage TODO: i don't see a "save" here?

        # This is going to be our "embedding" of the input graph
        h_list.append(h)

        # The same "embedding" is then passed into the vector quantizer below
        quantized, _, commit_loss, dist, codebook = self.vq(h)
        quantized_node = self.decoder_node(quantized)
        quantized_edge_1 = self.decoder_edge_1(quantized)
        quantized_edge_2 = self.decoder_edge_2(quantized)

        # The "embedding" is passed into the prob head and the cut head below
        variable_proba_head = None
        if self.has_variable_proba_head:
            variable_proba_head = F.sigmoid(self.variable_proba_layer(quantized))

        integral_gap_head = None
        if self.has_integral_gap_head:
            integral_gap_head = F.sigmoid(self.integral_gap_layer(quantized))

        # Training
        feature_rec_loss = None
        edge_rec_loss = None
        if not self.is_eval_mode:
            adj = dgl_graph.adjacency_matrix().to_dense().to(feature_matrix.device)

            # Reconstruction Loss (other losses are calculated in training code)
            feature_rec_loss = self.lambda_node * F.mse_loss(feature_matrix, quantized_node)

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

            pos_edge_rec_loss = self.lambda_edge * torch.mean(diff)
            edge_rec_loss = self.lambda_edge * torch.sqrt(F.mse_loss(adj, adj_quantized))
            edge_rec_loss += pos_edge_rec_loss

        # Distance Matrix - Distance From Each Node's Embedding to Each Code in the Codebook
        dist = torch.squeeze(dist)

        h_list.append(quantized)
        h_list.append(quantized_node)
        h_list.append(quantized_edge_1)
        h_list.append(quantized_edge_2)
        if self.has_variable_proba_head:
            h_list.append(variable_proba_head)
        if self.has_integral_gap_head:
            h_list.append(integral_gap_head)

        if not self.is_eval_mode:
            loss = feature_rec_loss + edge_rec_loss + commit_loss
        else:
            loss = -1

        return h_list, h, loss, dist, codebook

    def _pretrain(self,
                  input_mipinfo_list: List[MIPInfo],
                  output_forge_pkl: str,
                  output_log_file: Optional[str],
                  epochs: Optional[int] = None,
                  steps_per_instance: Optional[int] = None,
                  learning_rate: Optional[float] = None,
                  weight_decay: Optional[float] = None,
                  max_dgl_nodes: Optional[int] = None) -> None:
        """Pretrain the Forge model on provided MIP instances. Sets `is_trained` to True upon completion.

            Parameters
            ----------
            input_mipinfo_list : List[MIPInfo]
                List of training instances.
                Each item is a MIPInfo object containing List[tuple[dgl.DGLGraph, torch.Tensor, int, int]]
                (dgl_graph, feature_matrix, num_cons, num_vars) where:
                - graph: `dgl.DGLGraph` bipartite MIP graph
                - feature_matrix: node feature `torch.Tensor`
                - num_cons: number of constraint nodes (int)
                - num_vars: number of variable nodes (int)
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
            max_dgl_nodes : Optional[int], default=None
                Maximum allowed number of nodes for a DGL graph to be processed on the device.
                Instances with `dgl_graph.num_nodes()` greater than this value will be skipped during training.
                If `None`, the configuration default loaded from the training YAML is used.
                This can be set based on the available GPU memory to avoid out-of-memory errors.

            Returns
            -------
            None
        """

        # Put module into training mode (use super to avoid recursion) and move to device
        super().train()
        self.to(self.device)

        # Default to config values, but overwrite if a value is given
        epochs = overwrite_if_given(self.epochs, epochs)
        steps_per_instance = overwrite_if_given(self.steps_per_instance, steps_per_instance)
        learning_rate = overwrite_if_given(self.learning_rate, learning_rate)
        weight_decay = overwrite_if_given(self.weight_decay, weight_decay)
        max_dgl_nodes = overwrite_if_given(self.max_dgl_nodes, max_dgl_nodes)

        # TODO this main loss list is never used/printed/logged?
        main_loss_list = []
        skip_list = set()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Loop through data set
        t = ""
        for epoch in range(epochs):

            # Alternate between prioritizing node feature reconstruction and edge reconstruction
            if epoch % 2 == 0:
                self.lambda_node = 10
                self.lambda_edge = 1
            else:
                self.lambda_node = 1
                self.lambda_edge = 10

            loss_list = []
            epoch_start = time.time()

            # MIP instances in dataset
            loss = None
            # TODO why is this starting from idx 10?
            for idx in range(10, len(input_mipinfo_list)):

                mip_info = input_mipinfo_list[idx]

                # Some MIP instances are too large to fit in GPU memory
                if mip_info.dgl_graph.num_nodes() > max_dgl_nodes:
                    skip_list.add(idx)
                    continue

                # Push to device before the for-loop below
                dgl_graph = mip_info.dgl_graph.to(self.device)
                features = mip_info.feature_tensor.to(self.device)

                for step in range(steps_per_instance):
                    # Compute loss and prediction
                    h_list, logits, loss, distances, codebook_ = self.forward(dgl_graph,
                                                                              features,
                                                                              mip_info.num_cons,
                                                                              mip_info.num_vars)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                loss_list.append(loss.item())
                print("\rEpoch: ", epoch,
                      "| Loss on Instance", idx, ": ", np.round(loss.item(), 3),
                      "| Mean Loss :", np.round(np.mean(loss_list), 3), end='')
                torch.cuda.empty_cache()

            print("\r")
            print()
            print("------")

            p_string = ("Epoch: " + str(epoch) +
                        "| Mean Loss: " + str(np.round(np.mean(loss_list), 3)) +
                        "+/-" + str(np.round(np.std(loss_list), 3)) +
                        " | Time For Epoch : " + str(np.round(time.time() - epoch_start, 3)) + "s")

            t += p_string + "\n"
            print(p_string, end='\n')
            print("------")
            print()

            torch.save(self.state_dict(), output_forge_pkl)
            main_loss_list.append(np.round(np.mean(loss_list), 3))

            if output_log_file is not None:
                with open(output_log_file, 'a') as file:
                    file.write(t)

        # Set Forge as trained
        self.is_trained = True

    def _mip_model_to_embeddings(self, mip_model: gp.Model) -> MIPEmbeddings:
        """
        Convert a provided Gurobi model into a codebook histogram and
        return per-node embeddings for constraints and variables.

        Parameters
        ----------
        mip_model : gurobipy.Model
            An already-loaded Gurobi model object.

        Returns
        -------
        MIPEmbeddings
            Dataclass containing:
            - mip_embedding: 1D numpy array of length `self.codebook_size` with counts of assigned codes
            - embedding_of_constraint: torch.Tensor of shape (num_cons, hidden_dim)
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

        # Convert MIP model to MIP info with DGL graph, features tensor, num cons/vars
        mip_info = MIPProcessor._mip_model_to_mip_info(mip_model)

        # Forward pass through trained Forge
        h_list, logits, loss, distances, codebook_ = self.forward(mip_info.dgl_graph.to(self.device),
                                                                  mip_info.feature_tensor.to(self.device),
                                                                  mip_info.num_cons,
                                                                  mip_info.num_vars)
        # Restore original mode
        self.is_eval_mode = original_mode

        # Compute mip instance vector, as a frequency distribution of codes assigned to constraints and variables
        assigned_codes = torch.argmin(distances, axis=1).detach().cpu().numpy()
        instance_embedding = np.zeros(self.codebook_size, )
        for c in assigned_codes:
            instance_embedding[c] += 1
        # TODO possible to speed up using:
        #   instance_embedding = np.bincount(assigned_codes, minlength=self.codebook_size).astype(float)

        embedding_of_constraint = h_list[1][:mip_info.num_cons]
        embedding_of_variable = h_list[1][mip_info.num_cons:]

        mip_embeddings = MIPEmbeddings(instance_embedding=instance_embedding,
                                       embedding_of_constraint=embedding_of_constraint,
                                       embedding_of_variable=embedding_of_variable)

        return mip_embeddings

    def _finetune_integral_gap(self,
                               input_mip_to_gapinfo: Dict[str, GapInfo],
                               output_forge_finetuned_pkl : str,
                               epochs: Optional[int] = None, #10
                               steps_per_instance: Optional[int] = None, #10
                               learning_rate: Optional[float] = None, #1e-4
                               weight_decay: Optional[float] = None, #5e-4
                               max_dgl_nodes: Optional[int] = None #30000
                               ) -> None:

        # TODO in pretraining() we have these --needed here?
        # super().train()
        # self.to(self.device)

        # TODO comment on this block for why was this needed? Removing now?
        # copy_params(old_model=pre_trained, new_model=self)
        # del pre_trained

        # TODO comment on this block for why was this needed?
        torch.cuda.empty_cache()

        self.train()

        # TODO: weight decay is set to 5e-4 for fine-tuning vs. 1e-4 in pretraining typo? intentional?
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Start Gurobi environment
        gurobi_env = MIPProcessor._start_gurobi_env()

        # Mip instances
        mips = list(input_mip_to_gapinfo.keys())
        for epoch in range(epochs):

            epoch_loss = []
            bce_epoch_loss = []
            gap_epoch_loss = []

            for idx, mip in enumerate(mips):

                # Read MIP file to a Gurobi model
                mip_model = gp.read(mip, env=gurobi_env)

                # Generate MIPInfo object from Gurobi model, set name, and add to dictionary
                mip_info = self._mip_model_to_mip_info(mip_model)

                # Some instances are too big to fit in the GPU (max_dgl_nodes=30000)
                if mip_info.dgl_graph.num_nodes() <= max_dgl_nodes:

                    # Push to device before the for-loop below
                    dgl_graph = mip_info.dgl_graph.to(self.device)
                    features = mip_info.feature_tensor.to(self.device)

                    for step in range(steps_per_instance):

                        # TODO we did not have this in pretraining() --needed here? comment
                        optimizer.zero_grad()

                        # Compute loss and prediction
                        h_list, logits, loss, distances, codebook_ = self.forward(dgl_graph,
                                                                                  features,
                                                                                  mip_info.num_cons,
                                                                                  mip_info.num_vars)
                        # Predict gap ratio
                        # TODO what's the magic -1? (layer?)
                        gap_ratio_pred = torch.mean(h_list[-1][mip_info.num_cons:, :])
                        gap_ratio_true = input_mip_to_gapinfo[mip].ratio

                        # TODO comment here
                        if gap_ratio_true > 1:
                            gap_ratio_true = 1 / gap_ratio_true

                        # TODO why are we predicting both gap ratio and variable probabilities here?
                        # TODO what's the magic -2? (layer?)
                        # Predict variable probabilities
                        var_proba_pred = h_list[-2][mip_info.num_cons:, :]
                        var_proba_truth = torch.Tensor(input_mip_to_gapinfo[mip].mip_sol).to(self.device)

                        try:
                            gap_loss = torch.abs(gap_ratio_pred - gap_ratio_true)
                            bce_loss = F.binary_cross_entropy(var_proba_pred.flatten(), var_proba_truth.flatten())

                            # TODO comment on weighting here?
                            loss = gap_loss + (0.01 * bce_loss)
                            loss.backward()
                            optimizer.step()

                            print('', '(', idx, '/', len(mips), ') |', mip,
                                  ' | GAP Loss :', gap_loss.item(),
                                  'BCE Loss :', bce_loss.item(), end='\r')

                            epoch_loss.append(loss.item())
                            bce_epoch_loss.append(bce_loss.item())
                            gap_epoch_loss.append(gap_loss.item())
                        except:
                            continue

            print("\nEpoch ", epoch + 1,
                  "| Means | Loss : ", np.mean(epoch_loss),
                  "| Gap Loss : ", np.mean(gap_epoch_loss),
                  "| BCE Loss : ", np.mean(bce_epoch_loss))
            print()

            torch.save(self.state_dict(), output_forge_finetuned_pkl)

            # Shuffle MIP instances for next epoch
            np.random.shuffle(mips)

        # Close Gurobi environment
        gurobi_env.close()

    def _mip_model_to_gap_info(self, mip_model: gp.Model, problem_type:str) -> GapInfo:
        """
        Convert a provided Gurobi model into a codebook histogram and
        return per-node embeddings for constraints and variables.

        Parameters
        ----------
        mip_model : gurobipy.Model
            An already-loaded Gurobi model object.

        Returns
        -------

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
        # TODO this is used in get mip_embeddings but not get primal gap? is it not needed?
        #self.eval()

        # Store Forge eval mode (to restore back) and set to eval only to generate embedding
        original_mode = self.is_eval_mode
        self.is_eval_mode = True

        # Convert MIP model to MIP info with DGL graph, features tensor, num cons/vars
        mip_info = MIPProcessor._mip_model_to_mip_info(mip_model)

        # Forward pass through trained Forge
        h_list, logits, loss, distances, codebook_ = self.forward(mip_info.dgl_graph.to(self.device),
                                                                  mip_info.feature_tensor.to(self.device),
                                                                  mip_info.num_cons,
                                                                  mip_info.num_vars)
        # Restore original mode
        self.is_eval_mode = original_mode

        # Find LP optimal to calculate ratio
        variable_proba = h_list[-2][mip_info.num_cons:]
        integral_gap = h_list[-1][mip_info.num_cons:]

        # TODO what's this exactly? is this the predicted gap ratio?
        gap_ratio = torch.mean(integral_gap).item()

        # Read and solve the LP relaxation to generate initial objective value
        lp_model = mip_model.relax()
        lp_model.optimize()
        lp_obj = lp_model.ObjVal

        # TODO consider generalizing/removing in future
        # Add a buffer to the ratio to make sure we are not infeasible
        mip_obj = lp_obj
        if problem_type in ['SC']:
            gap_ratio += (0.02 * gap_ratio)
            mip_obj = lp_obj + (lp_obj * (1 - gap_ratio))
        elif problem_type in ['MVC']:
            mip_obj = lp_obj + (lp_obj * (1 - gap_ratio))
        elif problem_type in ['GISP']:
            gap_ratio += (0.2 * gap_ratio)
            mip_obj = lp_obj * gap_ratio
        elif problem_type in ['CA']:
            gap_ratio += (0.05 * gap_ratio)
            mip_obj =lp_obj * gap_ratio

        # Create GapInfo with true lp_obj, predicted ratio, and predicted mip_obj but without a mip solution
        gap_info = GapInfo(ratio=gap_ratio, mip_sol=None, mip_obj=mip_obj, lp_obj=lp_obj)

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
