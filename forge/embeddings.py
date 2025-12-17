import time
from typing import List, Callable, Optional, Tuple, Union, Dict

import gurobipy as gp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from vector_quantize_pytorch import VectorQuantize

from forge._wgsage import EdgeWeightedSAGEConv, blockwise_loss
from forge.labeler import GapInfo
from forge.processor import MIPInfo, MIPEmbeddings, MIPProcessor
from forge.utils import check_true, Constants, overwrite_if_given, copy_params


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
                 activation: Optional[Callable] = F.relu,
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
        self.activation: Callable = activation
        self.norm_type: str = overwrite_if_given(config.get('norm_type'), norm_type)
        self.lambda_edge: float = overwrite_if_given(config.get('lambda_edge'), lambda_edge)
        self.lambda_node: float = overwrite_if_given(config.get('lambda_node'), lambda_node)
        self.orthogonal_reg_weight: float = overwrite_if_given(config.get('orthogonal_reg_weight'),
                                                               orthogonal_reg_weight)
        self.is_eval_mode: bool = overwrite_if_given(config.get('is_eval_mode'), is_eval_mode)

        # Load additional parameters
        self.graph_sage_aggregation: str = config.get('graph_sage_aggregation')
        self.decoder_edge_dim: int = config.get('decoder_edge_dim')
        self.vq_decay: float = config.get('vq_decay')
        self.vq_commitment_weight: float = config.get('vq_commitment_weight')
        self.vq_is_cosine_sim: bool = config.get('vq_is_cosine_sim')

        # Load default training parameters
        self.epochs: int = config.get('epochs')
        self.steps_per_instance: int = config.get('steps_per_instance')
        self.learning_rate: float = float(config.get('learning_rate'))  # cast 1e-4 as float! not scientific str
        self.weight_decay: float = float(config.get('weight_decay'))  # cast 1e-4 as float! not scientific str
        self.max_graph_nodes: int = config.get('max_graph_nodes')
        self.adj_block_size: int = config.get('adj_block_size')  # Block size for adjacency reconstruction loss

        # Load integral gap parameters
        self.integral_gap_safety_eps: float = config.get('integral_gap_safety_eps')  # Margin for gap ratio adjustments

        # Load seed
        self.seed: int = config.get('seed')

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

        # Batch norm layers
        self.bn1 = nn.BatchNorm1d(self.updated_input_dim)
        self.bn2 = nn.BatchNorm1d(self.updated_input_dim)
        self.bn3 = nn.BatchNorm1d(self.updated_input_dim)

        # Node decoder
        self.decoder_node = nn.Linear(self.updated_input_dim, self.input_dim)

        # Edge decoders. Edges are decoded as product of two matrices
        self.decoder_edge_1 = nn.Linear(self.updated_input_dim, self.decoder_edge_dim)
        self.decoder_edge_2 = nn.Linear(self.updated_input_dim, self.decoder_edge_dim)

        # Vector quantization module
        self.vq = VectorQuantize(dim=self.updated_input_dim,
                                 codebook_size=self.codebook_size,
                                 codebook_dim=self.codeword_dim,
                                 decay=self.vq_decay,
                                 commitment_weight=self.vq_commitment_weight,
                                 use_cosine_sim=self.vq_is_cosine_sim,
                                 orthogonal_reg_weight=self.orthogonal_reg_weight)

    def forward(self, feature_tensor: torch.Tensor,
                num_cons: int, num_vars: int,
                edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor]) \
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

        # GraphSAGE Layer 1
        h = self.graph_layer_1(h, edge_index, edge_weight=edge_weight)
        h = self.activation(h)  # PyG needs explicit activation
        h = self.bn1(h)
        if self.norm_type != "none":
            h = self.norms[0](h)
        h = self.dropout(h)

        # GraphSAGE Layer 2
        h = self.graph_layer_2(h, edge_index, edge_weight=edge_weight)
        h = self.activation(h)  # PyG needs explicit activation
        h = self.bn2(h)
        h = self.dropout(h)

        # Linear Layer
        h = self.linear(h)
        h = F.relu(h)
        h = self.bn3(h)
        h = self.dropout(h)

        # Store output at this stage into h_list
        # This is going to be our "embedding" of the input graph
        h_list.append(h)

        # The same "embedding" is then passed into the vector quantizer below
        quantized, indices, commit_loss = self.vq(h)
        codebook = self.vq.codebook  # or from the forward output

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

            # Convert PyG edge_index to dense adjacency matrix on CPU
            num_nodes = num_cons + num_vars
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

        if not self.is_eval_mode:
            loss = feature_rec_loss + edge_rec_loss + commit_loss
        else:
            loss = -1

        return h_list, h, loss, indices, codebook

    def load_model(self, input_forge_pkl, model_type=Constants.FORGE_PRE_TRAIN):

        if self.is_trained:
            print("Warning: Forge model is already trained, NOT loading weights, quitting!!")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        if model_type == Constants.FORGE_PRE_TRAIN:
            self.has_integral_gap_head = False
            self.has_variable_proba_head = False
        elif model_type == Constants.FORGE_FINE_TUNE_INTEGRAL_GAP:
            self.has_integral_gap_head = True
            self.has_variable_proba_head = False
        elif model_type == Constants.FORGE_FINE_TUNE_VARIABLE_PROBA:
            self.has_integral_gap_head = False
            self.has_variable_proba_head = True

        self.load_state_dict(torch.load(input_forge_pkl, map_location=device))
        self.is_trained = True

    def _pretrain(self,
                  input_mipinfo_list: List[MIPInfo],
                  output_forge_pkl: str,
                  output_log_file: Optional[str],
                  epochs: Optional[int] = None,
                  steps_per_instance: Optional[int] = None,
                  learning_rate: Optional[float] = None,
                  weight_decay: Optional[float] = None,
                  max_graph_nodes: Optional[int] = None) -> None:
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

        Returns
        -------
        None
        """

        # Put module into training mode (use super to avoid recursion) and move to device
        # .train() retains dropout and batchnorm behavior vs. .eval() for inference remove dropout and freeze batchnorm
        super().train()
        self.to(self.device)

        # Default to config values, but overwrite if a value is given
        epochs = overwrite_if_given(self.epochs, epochs)
        steps_per_instance = overwrite_if_given(self.steps_per_instance, steps_per_instance)
        learning_rate = overwrite_if_given(self.learning_rate, learning_rate)
        weight_decay = overwrite_if_given(self.weight_decay, weight_decay)
        max_graph_nodes = overwrite_if_given(self.max_graph_nodes, max_graph_nodes)

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
            print()
            for idx in range(len(input_mipinfo_list)):

                mipinfo = input_mipinfo_list[idx]

                # Some MIP instances are too large to fit in GPU memory
                num_nodes = mipinfo.num_cons + mipinfo.num_vars
                if num_nodes > max_graph_nodes:
                    skip_list.add(idx)
                    continue

                # Push to device before the for-loop below
                features = mipinfo.feature_tensor.to(self.device)
                edge_index = mipinfo.edge_index.to(self.device)
                edge_weight = mipinfo.edge_weight.to(self.device)

                for step in range(steps_per_instance):
                    # Compute loss and prediction
                    h_list, logits, loss, indices, codebook_ = self.forward(features,
                                                                              mipinfo.num_cons, mipinfo.num_vars,
                                                                              edge_index, edge_weight)

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

            p_string = ("Epoch: " + str(epoch) + " " +
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
        Convert a Gurobi model into `MIPEmbeddings` using the trained Forge encoder.

        Steps
        -----
        - Convert `mip_model` to `MIPInfo` (PyG style).
        - Call `forward()` in eval mode.
        - Build instance code histogram from `indices` assignments and extract per-node quantized embeddings.

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

        # Convert MIP model to MIP info with features tensor, num cons/vars, edge index/weight
        mipinfo = MIPProcessor._mip_model_to_mipinfo(mip_model)

        # Forward pass through trained Forge
        h_list, logits, loss, indices, codebook_ = self.forward(mipinfo.feature_tensor.to(self.device),
                                                                  mipinfo.num_cons, mipinfo.num_vars,
                                                                  mipinfo.edge_index.to(self.device),
                                                                  mipinfo.edge_weight.to(self.device))
        # Restore original mode
        self.is_eval_mode = original_mode

        # Compute mip instance vector, as a frequency distribution of codes assigned to constraints and variables
        assigned_codes = indices.detach().cpu().numpy()
        # instance_embedding = np.zeros(self.codebook_size, )
        # for c in assigned_codes:
        #     instance_embedding[c] += 1
        instance_embedding = np.bincount(assigned_codes, minlength=self.codebook_size).astype(float)

        embedding_of_constraint = h_list[1][:mipinfo.num_cons]
        embedding_of_variable = h_list[1][mipinfo.num_cons:]

        mip_embeddings = MIPEmbeddings(instance_embedding=instance_embedding,
                                       embedding_of_constraint=embedding_of_constraint,
                                       embedding_of_variable=embedding_of_variable)

        return mip_embeddings

    def _finetune_integral_gap(self,
                               input_mip_to_gapinfo: Dict[str, GapInfo],
                               output_forge_finetuned_pkl: str,
                               epochs: Optional[int] = None,  # 10,
                               steps_per_instance: Optional[int] = None,  # 10,
                               learning_rate: Optional[float] = None,  # 1e-4,
                               weight_decay: Optional[float] = None,  # 5e-4,
                               max_graph_nodes: Optional[int] = None  # 30000
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

        if self.is_trained and not self.has_integral_gap_head:
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
        gurobi_env = MIPProcessor._start_gurobi_env()

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
