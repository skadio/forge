"""
processor.py

Utilities to convert Gurobi MIP instances into PyG graph tensors used by the Forge GNN.

This module provides:
- MIPInfo: a lightweight container for instance metadata plus PyG tensors.
- MIPProcessor: read MIP files, optionally relax them,
    convert to PyG edge/feature tensors, and save/load pickled representations.

Notes
-----
- Designed to work with gurobipy, PyTorch Geometric (PyG), PyTorch and SciPy sparse matrices.
- Many helper functions assume Gurobi model APIs such as getVars, getConstrs and getA.
"""
import os
import time
from typing import List, Any, Dict, Optional, Union

# import dgl
import gurobipy as gp
import numpy as np
import scipy.sparse as sp
import torch
import yaml
from tqdm import tqdm

from forge.utils import check_true, Constants, overwrite_if_given, save_pickle, load_pickle


class MIPInfo:
    """
    Container for converted MIP instance data stored.

    Attributes
    ----------
    instance_name : Optional[str]
        Path or unique identifier of the MIP instance.
    feature_tensor : Optional[torch.Tensor]
        Node feature matrix of shape `(num_cons + num_vars, feat_dim)` with constraints stacked first.
    num_cons : Optional[int]
        Number of constraints in the original MIP.
    num_vars : Optional[int]
        Number of variables in the original MIP.
    edge_index : Optional[torch.LongTensor]
        PyG-style edge index tensor of shape (2, num_edges) representing graph connectivity.
        Row 0 = source indices, row 1 = target indices.
    edge_weight : Optional[torch.FloatTensor]
        Edge weights tensor of shape (num_edges,) corresponding to edges in `edge_index`.
    """

    def __init__(self,
                 instance_name: str = None,
                 feature_tensor: Optional[torch.Tensor] = None,
                 num_cons: int = None,
                 num_vars: int = None,
                 edge_index: Optional[torch.Tensor] = None,
                 edge_weight: Optional[torch.Tensor] = None):

        self.instance_name: Optional[str] = instance_name
        # self.dgl_graph: Optional[dgl.DGLGraph] = dgl_graph

        # feature_tensor shape: (num_cons + num_vars, feat_dim=10)
        self.feature_tensor: Optional[torch.Tensor] = feature_tensor
        self.num_cons: Optional[int] = num_cons
        self.num_vars: Optional[int] = num_vars
        # TODO: double-check sizes and index (constraint or var)
        # - `edge_index`: PyG  COO`(2, E)`
        #   connectivity from source (constraint) to target (variable) nodes where there is an Edge
        self.edge_index: Optional[torch.Tensor] = edge_index
        # - `edge_weight`: per-edge normalized coefficient values (FloatTensor of length `E`).
        self.edge_weight: Optional[torch.Tensor] = edge_weight


class MIPEmbeddings:
    """
    Container for learned embeddings related to a MIP instance.

    Attributes
    ----------
    instance_embedding : Optional[np.ndarray]
        Global embedding for the entire MIP of size codebook_size
    embedding_of_constraint : Optional[torch.Tensor]
        Per-constraint embedding vector (shape: (num_constraints, codebook_dim)).
    embedding_of_variable : Optional[torch.Tensor]
        Per-variable embedding vector (shape: (num_variables, codebook_dim)).
    """

    def __init__(self,
                 instance_embedding: Optional[np.ndarray] = None,
                 embedding_of_constraint: Optional[torch.Tensor] = None,
                 embedding_of_variable: Optional[torch.Tensor] = None) -> None:
        """
        Initialize a MIPEmbeddings container.

        Parameters
        ----------
        instance_embedding : Optional[np.ndarray], default: None
            Global MIP embedding vector.
        embedding_of_constraint : Optional[torch.Tensor], default: None
            Embedding vector per constraint.
        embedding_of_variable : Optional[torch.Tensor], default: None
            Embedding vector per variable.
        """
        self.instance_embedding: Optional[np.ndarray]= instance_embedding
        self.embedding_of_constraint: Optional[torch.Tensor]= embedding_of_constraint
        self.embedding_of_variable: Optional[torch.Tensor] = embedding_of_variable


class MIPProcessor:
    """
    Processor that converts MIP instances to graph-based features.

    Usage
    -----
    - Initialize with an optional training config file to set RNG seeds.
    - Call convert_mip_to_mipinfo to read MPS/LP files in a folder and
        produce pickled MIPInfo objects (optionally relaxing constraints).
    - Use load_mipinfo_from_pickles to aggregate multiple pickles into a list.

    Parameters
    ----------
    train_config_file_path : Optional[str]
        Path to YAML train config. Uses Constants.default_train_config_file by default.
    seed : Optional[int]
        Seed to override the config RNG seed.
    """

    def __init__(self,
                 train_config_file_path: Optional[str] = Constants.default_train_config_yaml,
                 seed: Optional[int] = None):

        super().__init__()

        # Read train config
        with open(train_config_file_path, 'r') as f:
            config = yaml.safe_load(f)

        # Set input parameters
        self.seed = overwrite_if_given(config.get('seed'), seed)

        # Set based on input
        self.rng = np.random.default_rng(self.seed)

    def convert_mip_to_mipinfo(self,
                               input_mip_folder: str,
                               output_mip_to_mipinfo_pkl: str,
                               relaxation_list: Optional[List[float]] = None,
                               is_save_relaxed: bool = False,
                               has_return: bool = False) -> Optional[Dict[str, MIPInfo]]:
        """
        Converts MIP instances in a given folder to MIPInfo objects and saves them to a pickle file.

        Parameters
        ----------
        input_mip_folder : str
            Path to the directory containing MIP instance files (`.mps` or `.lp`).
        output_mip_to_mipinfo_pkl : str
            Path where the resulting pickled mapping of instance names to `MIPInfo` objects will be saved.
        relaxation_list : Optional[List[float]], default: None
            If provided, for each instance a set of relaxed instances will be generated by randomly
            removing the specified fraction(s) of constraints (values between 0 and 1).
            Only one relaxed instance is created per ratio in the list.
        is_save_relaxed : bool, default: False
            If True, relaxed MIP instances are written to disk using the original name, plus the relaxation ratio.
        has_return : bool, default: False
            If True the function returns the dictionary mapping instance names to `MIPInfo`,
            otherwise it returns None after saving to `output_file`.

        Returns
        -------
        Optional[Dict[str, List[Any]]]
            The dictionary mapping instance file paths (or relaxed names) to `MIPInfo` when `has_return` is True;
            Otherwise None.
        """

        if relaxation_list is None:
            relaxation_list = []

        # Find and sort MIP instance files by size
        sorted_mip_files = MIPProcessor.get_only_mip_files(input_mip_folder, is_sort_by_size=True)

        # Start Gurobi environment
        gurobi_env = MIPProcessor._start_gurobi_env()

        # Convert each MIP instance to MIPInfo object and store in dictionary
        mip_to_mipinfo = {}
        for idx in tqdm(range(len(sorted_mip_files))):
            
            # Read MIP file to a Gurobi model
            mip_model = gp.read(sorted_mip_files[idx], env=gurobi_env)

            # Generate MIPInfo object from Gurobi model, set name, and add to dictionary
            try:
                mipinfo = self._mip_model_to_mipinfo(mip_model)
            except Exception as e:
                print(f"Warning: Failed to convert MIP {sorted_mip_files[idx]} to MIPInfo due to error: {e}")
                continue
            mipinfo.instance_name = sorted_mip_files[idx]
            mip_to_mipinfo[mipinfo.instance_name] = mipinfo

            if relaxation_list:
                for ratio in relaxation_list:

                    # Create a copy of the original model to remove constraints from
                    mip_model_relaxed = mip_model.copy()
                    cons = mip_model_relaxed.getConstrs()

                    # Choose a random number of constraints within the ratio to remove
                    k = int(len(cons) * ratio)
                    if k <= 0:
                        continue

                    # Choose a random subset of constraints
                    cons_remove_ = self.rng.choice(cons, k, replace=False)

                    # Remove them from the copy model
                    for c in cons_remove_:
                        mip_model_relaxed.remove(c)
                    mip_model_relaxed.update()

                    # Generate MIPInfo object from relaxed model, set name, and add to dictionary
                    mipinfo = self._mip_model_to_mipinfo(mip_model_relaxed)
                    orig_path = sorted_mip_files[idx]
                    base, ext = os.path.splitext(orig_path)
                    mipinfo.instance_name = f"{base}_relaxed_{ratio}{ext}"
                    mip_to_mipinfo[mipinfo.instance_name] = mipinfo

                    # Save the perturbed MIP instance to disk
                    if is_save_relaxed:
                        mip_model_relaxed.write(mipinfo.instance_name)

                    # Release copy model
                    mip_model_relaxed.dispose()

        # Close Gurobi environment
        gurobi_env.close()

        save_pickle(mip_to_mipinfo, output_mip_to_mipinfo_pkl)

        return mip_to_mipinfo if has_return else None

    @staticmethod
    def load_mipinfo_from_pickles(mip_to_mipinfo_files: List[str]) -> List[MIPInfo]:
        """
        Load and aggregate lists of MIPInfo objects from multiple pickled files.

        Parameters
        ----------
        mip_to_mipinfo_files : List[str]
            List of paths to pickled mappings (saved by `convert_mip_to_mipinfo`).

        Returns
        -------
        List[MIPInfo]
            Flattened list of `MIPInfo` objects.
        """
        mipinfo_list = []
        for mip_to_mipinfo_file in mip_to_mipinfo_files:
            mip_to_mipinfo = load_pickle(mip_to_mipinfo_file)
            for mip in mip_to_mipinfo:
                mipinfo_list.append(mip_to_mipinfo[mip])
        return mipinfo_list

    @staticmethod
    def _mip_model_to_mipinfo(mip_model: gp.Model, is_debug: bool = False) -> Union[bool, MIPInfo]:
        """
        Convert a Gurobi model into a MIPInfo.

        The produced MIPInfo contains:
        - `feature_tensor`: node features stacked with constraints first then variables.
            Feature tensor shape: (num_cons + num_vars, feat_dim)
        - `num_cons`, `num_vars`: counts used to interpret the graph layout.
        - `edge_index`: PyG COO `(2, E)`
        - `edge_weight`: per-edge normalized coefficient values (FloatTensor of length `E`).

        Parameters
        ----------
        mip_model : gp.Model
            The Gurobi model to convert. The function will mutate the model (remove zero-columns)
            to ensure a valid bipartite incidence.
        is_debug : bool
            If True, prints timing/debug information.

        Returns
        -------
        Union[bool, MIPInfo]
            Returns False on irrecoverable failure, an empty MIPInfo on soft failure,
            or a populated MIPInfo on success.
        """

        # Remove zero-column variables (vars with no coefficients) to ensure valid bipartite graph
        to_remove = [v for v in mip_model.getVars() if not mip_model.getCol(v).size()]
        mip_model.remove(to_remove)
        mip_model.update()

        s = 0
        if is_debug:
            print("Compute Feature Tensor")
            s = time.time()

        # Get static feature tensor from MIP, number of constraints, and number of variables
        # Feature tensor shape: (num_cons + num_vars, feat_dim=10)
        feature_tensor, num_cons, num_vars = MIPProcessor._get_feature_tensor_num_cons_num_vars(mip_model)

        if is_debug:
            print("Feature Tensor Computed in ", time.time() - s, "seconds")
            print("Creating PyG edge tensors")
            s = time.time()

        # Get edge indexes and weights in Tensor, ready for PyG
        edge_index, edge_weight = MIPProcessor._get_edge_index_weight(mip_model, num_cons, num_vars)
        # edge_index, edge_weight = MIPProcessor._old_get_edge_index_weight(mip_model, num_cons, num_vars, is_debug)

        return MIPInfo(feature_tensor=feature_tensor,
                       num_cons=num_cons, num_vars=num_vars,
                       edge_index=edge_index, edge_weight=edge_weight)

    @staticmethod
    def _get_feature_tensor_num_cons_num_vars(mip_model):
        """
        Extract node-level features from a Gurobi model.

        The function produces a feature tensor where rows correspond to nodes:
            - first `num_cons` rows are constraint features,
            - followed by `num_vars` rows of variable features.
        Features are column-normalized to [0, 1].

        Parameters
        ----------
        mip_model : gp.Model
            Gurobi model to extract features from.

        Returns
        -------
        tuple
            (feature_tensor: torch.FloatTensor, num_cons: int, num_vars: int)
        """

        # Create variable features
        variables = mip_model.getVars()
        num_vars = len(variables)
        features_of_var = np.zeros((num_vars, Constants.NUM_VARIABLE_FEATURES), dtype=float)
        for i, var in enumerate(variables):
            features_of_var[i, 0] = float(var.VType == gp.GRB.CONTINUOUS)
            features_of_var[i, 1] = float(var.VType == gp.GRB.BINARY)
            features_of_var[i, 2] = float(var.VType == gp.GRB.INTEGER)
            features_of_var[i, 3] = float(var.Obj)
            features_of_var[i, 4] = float(var.LB > -gp.GRB.INFINITY)
            features_of_var[i, 5] = float(var.UB <= gp.GRB.INFINITY)

        # Create constraint features
        constraints = mip_model.getConstrs()
        operators = mip_model.Sense
        num_cons = len(constraints)
        features_of_constraint = np.zeros((num_cons, Constants.NUM_CONSTRAINT_FEATURES), dtype=float)
        for i, ct in enumerate(constraints):
            op = operators[i]
            features_of_constraint[i, 0] = float(op == '=')
            features_of_constraint[i, 1] = float(op == '<')
            features_of_constraint[i, 2] = float(op == '>')
            features_of_constraint[i, 3] = float(ct.RHS)

        # Pad with zeros for equal shapes
        cons_feat_matrix = np.hstack([features_of_constraint, np.zeros((num_cons, features_of_var.shape[1]))])
        var_feat_matrix = np.hstack([np.zeros((num_vars, features_of_constraint.shape[1])), features_of_var])

        # Stack up into one feature matrix, constraints come first
        feature_matrix = np.vstack([cons_feat_matrix, var_feat_matrix])

        # Column normalize features
        feature_matrix = (feature_matrix - np.min(feature_matrix, axis=0)) / (
                    np.max(feature_matrix, axis=0) - np.min(feature_matrix, axis=0) + 1e-9)
        feature_matrix[np.isnan(feature_matrix)] = 0

        # Convert features to tensor
        feature_tensor = torch.FloatTensor(np.array(feature_matrix))

        # Return feature tensor, number of constraints, and number of variables
        return feature_tensor, num_cons, num_vars

    @staticmethod
    def _start_gurobi_env():
        """
        Initialize, start and return a Gurobi environment with output disabled.

        Returns
        -------
        gp.Env
            Configured Gurobi environment.
        """
        try:
            from gurobi_onboarder import init_gurobi
            gurobi_env, GUROBI_FOUND = init_gurobi.initialize_gurobi()
        except Exception:
            gurobi_env = gp.Env(empty=True)
        gurobi_env.setParam("OutputFlag", 0)
        gurobi_env.start()
        return gurobi_env

    @staticmethod
    def get_only_mip_files(input_mip_folder: str, is_sort_by_size:bool = False) -> List[str]:
        """
        Find MIP instance files in a directory and return them sorted by file size.

        Parameters
        ----------
        input_mip_folder : str
            Path to the directory containing MIP instance files.
        is_sort_by_size : bool
            If True, sorts the returned file paths by file size in ascending order/smallest first.

        Returns
        -------
        List[str]
            Absolute paths to files with extensions `.mps` or `.lp`, optionally sorted by file size (ascending).
        """

        all_filenames = os.listdir(input_mip_folder)
        all_filepaths = [os.path.join(input_mip_folder, filename) for filename in all_filenames]
        mip_filepaths = [p for p in all_filepaths if p.lower().endswith('.mps') or
                          p.lower().endswith('.lp') or 
                          p.lower().endswith('.mps.gz') or 
                          p.lower().endswith('.lp.gz')]

        # Smallest sized mip first
        if is_sort_by_size:
            mip_filepaths = sorted(mip_filepaths, key=os.path.getsize)

        return mip_filepaths

    @staticmethod
    def get_mip_items(input_mips):
        """
        Normalize input: accept a folder path, a single MIP file path, a list of paths,
        or a gurobipy Model instance (or list/mix of them).
        Returns a list of MIP items (file paths or gp.Model instances).
        """
        inputs = input_mips if isinstance(input_mips, (list, tuple)) else [input_mips]
        mip_items = []
        for item in inputs:
            if isinstance(item, gp.Model):
                mip_items.append(item)
            elif isinstance(item, str) and os.path.isdir(item):
                mip_items.extend(MIPProcessor.get_only_mip_files(item, is_sort_by_size=False))
            elif isinstance(item, str) and os.path.isfile(item):
                mip_items.append(item)
            else:
                raise ValueError(
                    f"Error: Input {item!r} is neither a directory, a file, nor a gurobipy model instance.")
        return mip_items

    @staticmethod
    def _get_edge_index_weight(mip_model, num_cons, num_vars):
        # coefficient_matrix is sparse, getA returns a scipy.sparse.csr_matrix
        coef_sp = mip_model.getA()

        # create empty sparse blocks
        top_left = sp.csr_matrix((num_cons, num_cons))
        bottom_right = sp.csr_matrix((num_vars, num_vars))
        coeff_adj_sp = sp.bmat([[top_left, coef_sp], [coef_sp.transpose(), bottom_right]], format='coo')
        # then the old coeff_adj_coo = coeff_adj_sp  # already COO if format='coo'
        edge_index = torch.tensor(np.vstack([coeff_adj_sp.row, coeff_adj_sp.col]), dtype=torch.long)

        # Coefficient of variable in that constraint
        edge_weights = torch.FloatTensor(coeff_adj_sp.data)

        # TODO push this cast inside _normalize_edge_weights() so we don't have to do it here
        edge_weights_np = coeff_adj_sp.data.astype(float)

        # Normalize edge weights
        edge_weights = MIPProcessor._normalize_edge_weights(edge_weights_np)

        # Validate dimensions
        check_true(edge_index.shape[1] == edge_weights.shape[0],
                   ValueError(f"Error: edge_index has {edge_index.shape[1]} edges but "
                              f"edge_weight has {edge_weights.shape[0]} entries"))

        return edge_index, edge_weights

    @staticmethod
    def _old_get_edge_index_weight(cls, mip_model, num_cons, num_vars):
        # Get coefficient matrix (sparse) and convert to dense (constraint, variable)
        coefficient_matrix = mip_model.getA().todense()

        # Convert to binary adjacency, where any nonzero coefficient becomes `1`
        # Edge presence between a constraint and a variable.
        A = (coefficient_matrix != 0).astype(int)

        # Create full bipartite adjacency matrix
        # top-left and bottom-right are zero blocks (no constraint-constraint or var-var edges)
        # the off-diagonal blocks are `A` and `A.T` (constraint-variable edges).
        # This produces a (num_cons+num_vars) × (num_cons+num_vars) adjacency matrix.
        adj = np.block([[np.zeros((num_cons, num_cons)), A], [A.T, np.zeros((num_vars, num_vars))]])

        # Quit if adjacency is empty
        if adj is None or getattr(adj, "size", 0) == 0:
            return False

        # adj = normalize_adj(adj)

        # Convert the dense adjacency to a CSR sparse matrix and then to COO format.
        # In COO, we can access `.row` and `.col` arrays for graph edge construction (used for `edge_index`).
        # COO format stands for "Coordinate List" format.
        # COO a way to represent sparse matrices by storing only the non-zero entries and their coordinates (row/col)
        # In COO format, a sparse matrix is described by three arrays:
        #   row indices (i)
        #   column indices (j)
        #   values (v)
        # Each entry (i[k], j[k], v[k]) means that the value v[k] is at position (i[k], j[k]) in the matrix.
        adj_sp = sp.csr_matrix(adj)
        adj_coo = adj_sp.tocoo()

        # Create PyG edge_index (shape: 2 x num_edges)
        # COO format so we can access the nonzero coordinates via `adj_coo.row` and `adj_coo.col`.
        # np.array(..) builds a 2×E NumPy array where the first row is source node indices
        #   and the second row is target node indices for every nonzero entry (E = number of edges / nonzeros).
        # torch.tensor(..., dtype=torch.long)` converts that into a PyTorch LongTensor of shape `(2, E)`
        # This is the PyG `edge_index` format (row 0 = sources, row 1 = targets).
        # Long dtype is required because PyG uses these tensors for indexing.
        edge_index = torch.tensor(np.array([adj_coo.row, adj_coo.col]), dtype=torch.long)

        # - np.block(...) constructs a dense block matrix with four blocks:
        #   - top-left: zero matrix for constraint‑to‑constraint (size `num_cons x num_cons`)
        #   - top-right: `coefficient_matrix` (constraint × variable)
        #   - bottom-left: `coefficient_matrix.T` (variable × constraint)
        #   - bottom-right: zero matrix for var‑to‑var (size `num_vars x num_vars`)
        #  The resulting matrix represents the bipartite constraint_variable incidence,
        #  But carrying the actual coefficients instead of 0/1.
        coeff_adj = np.block([[np.zeros((num_cons, num_cons)), coefficient_matrix],
                              [coefficient_matrix.T, np.zeros((num_vars, num_vars))]])
        coeff_adj_sp = sp.csr_matrix(coeff_adj)
        coeff_adj_coo = coeff_adj_sp.tocoo()

        # Gather the nonzero coefficient values from the dense block matrix `coeff_adj`
        # at the coordinate pairs given by the COO sparse representation (`coeff_adj_coo.row`, `coeff_adj_coo.col`).
        # Result is flattened to a 1‑D NumPy array of per‑edge raw coefficients.
        edge_weights = coeff_adj[coeff_adj_coo.row, coeff_adj_coo.col].flatten()

        # Apply min–max normalization to scale the weights into \[0,1\].
        # epsilon is added to the denominator to avoid zero division when all values are (nearly) identical.
        edge_weights = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min() + 1e-6)
        # Shift all weights up slightly so none are exactly zero
        # (often useful if downstream code expects strictly positive weights).
        edge_weights += 1e-4
        # Convert the NumPy array into a PyTorch FloatTensor to be used as per‑edge features in PyG
        # `edge_weight` must align with `edge_index`
        edge_weight = torch.FloatTensor(edge_weights)

        # Decommission DGL
        # dgl_graph.ndata["feat"] = feature_tensor
        # dgl_graph.edata["weight"] = torch.FloatTensor(edge_weights).T
        #
        # check_true(dgl_graph.num_nodes() == dgl_graph.ndata["feat"].shape[0],
        #            ValueError(f"Error: graph has {dgl_graph.num_nodes()} nodes but "
        #                       f"feature matrix has {dgl_graph.ndata['feat'].shape[0]} rows"))

        return edge_index, edge_weight

    @staticmethod
    def _normalize_edge_weights(edge_weights: Optional[np.ndarray],
                               eps: float = 1e-12,
                               small_eps: float = 1e-4) -> torch.FloatTensor:
        """
        Normalize edge weights using the small-eps-only strategy.

        - If `edge_weights` is empty or None, returns an empty FloatTensor.
        - If max - min < eps, sets all weights to `small_eps`.
        - Otherwise performs min-max normalization and adds `small_eps` to avoid exact zeros.

        Returns a `torch.FloatTensor`.
        """
        if edge_weights is None:
            return torch.FloatTensor(np.array([], dtype=float))

        arr = np.asarray(edge_weights, dtype=float)
        if arr.size == 0:
            return torch.FloatTensor(arr)

        minv = arr.min()
        maxv = arr.max()
        if maxv - minv < eps:
            out = np.full_like(arr, small_eps, dtype=float)
        else:
            out = (arr - minv) / (maxv - minv)
            out = out + small_eps

        return torch.FloatTensor(out)
