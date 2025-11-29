"""
processor.py

Utilities to convert Gurobi MIP instances into graph data structures used by a GNN.

This module provides:
- MIPInfo: a lightweight container for instance metadata and graph features.
- MIPProcessor: read MIP files, optionally relaxes them, convert to DGL graphs with node/edge features,
  and save/load pickled representations.

Notes
-----
- Designed to work with gurobipy, DGL, PyTorch and SciPy sparse matrices.
- Many helper functions assume Gurobi model APIs such as getVars, getConstrs and getA.
"""
import os
import time
from typing import List, Any, Dict, Optional, Union

import dgl
import gurobipy as gp
import numpy as np
import scipy.sparse as sp
import torch
import yaml
from tqdm import tqdm

from forge.utils import check_true, Constants, overwrite_if_given, save_pickle, load_pickle


class MIPInfo:
    """
    Container for converted MIP instance data.

    Attributes
    ----------
    instance_name : Optional[str]
        Path or unique identifier of the MIP instance.
    dgl_graph : Optional[dgl.DGLGraph]
        DGL graph representing the bipartite MIP constraint-variable structure.
    feature_tensor : Optional[torch.Tensor]
        Node feature tensor for the DGL graph (constraints first, then variables).
    num_cons : Optional[int]
        Number of constraints in the original MIP.
    num_vars : Optional[int]
        Number of variables in the original MIP.
    """

    def __init__(self,
                 instance_name: str = None,
                 dgl_graph: Optional[dgl.DGLGraph] = None,
                 feature_tensor: Optional[torch.Tensor] = None,
                 num_cons: int = None,
                 num_vars: int = None):

        self.instance_name: Optional[str] = instance_name
        self.dgl_graph: Optional[dgl.DGLGraph] = dgl_graph
        # TODO: what's the row,column of feature tensor? is it num_con + num_var, feat_dim=10?
        self.feature_tensor: Optional[torch.Tensor] = feature_tensor
        self.num_cons: Optional[int] = num_cons
        self.num_vars: Optional[int] = num_vars


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
    - Call convert_mip_to_mipinfo to read MPS/LP files in a folder and produce
      pickled MIPInfo objects (optionally relaxing constraints).
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
            mip_info = self._mip_model_to_mip_info(mip_model)
            mip_info.instance_name = sorted_mip_files[idx]
            mip_to_mipinfo[mip_info.instance_name] = mip_info

            if relaxation_list:
                cons = mip_model.getConstrs()

                for ratio in relaxation_list:

                    # Choose a random subset of constraints
                    cons_remove_ = self.rng.choice(cons, int(len(cons) * ratio), replace=False)

                    # Remove them from the model
                    for c in cons_remove_:
                        mip_model.remove(c)
                    mip_model.update()

                    # Generate MIPInfo object from Gurobi model, set name, and add to dictionary
                    mip_info = self._mip_model_to_mip_info(mip_model)
                    orig_path = sorted_mip_files[idx]
                    base, ext = os.path.splitext(orig_path)
                    mip_info.instance_name = f"{base}_relaxed_{ratio}{ext}"
                    mip_to_mipinfo[mip_info.instance_name] = mip_info

                    # Save the perturbed MIP instance to disk
                    if is_save_relaxed:
                        mip_model.write(mip_info.instance_name)

                    # Re-add removed constraints to restore original model for next iteration
                    for c in cons_remove_:
                        mip_model.addConstrs(c)
                    mip_model.update()

            # Release Gurobi model resources
            mip_model.dispose()

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
    def _mip_model_to_mip_info(mip_model: gp.Model, is_debug: bool = False) -> Union[bool, MIPInfo]:
        """
        Convert a Gurobi model into a MIPInfo.

        The produced MIPInfo contains:
        - `dgl_graph`: a bipartite graph where the first `num_cons` nodes are constraints
          and the next `num_vars` nodes are variables.
        - `feature_tensor`: node features stacked with constraints first then variables.
        - `num_cons`, `num_vars`: counts used to interpret the graph layout.

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

        # TODO: this block needs some commentary explaining what is being done at a high level
        # Is this needed because of relaxed instances?
        to_remove = [v for v in mip_model.getVars() if not mip_model.getCol(v).size()]
        mip_model.remove(to_remove)
        mip_model.update()

        # Get objective equation
        # obj = np.array([x.Obj for x in mip_model.getVars()])

        s = 0
        if is_debug:
            print("Compute Feature Tensor")
            s = time.time()

        # Get feature tensor, number of constraints, and number of variables
        feature_tensor, num_cons, num_vars = MIPProcessor._get_feature_tensor_num_cons_num_vars(mip_model)

        if is_debug:
            print("Feature Tensor Computed in ", time.time() - s, "seconds")

        if is_debug:
            print("Creating DGL Graph")
            s = time.time()

        # Get constraint matrix TODO comment on what's adj and coeff_adj
        coefficient_matrix = mip_model.getA().todense()
        A = (coefficient_matrix != 0).astype(int)

        adj = np.block([[np.zeros((num_cons, num_cons)), A], [A.T, np.zeros((num_vars, num_vars))]])
        if not adj:
            return False
        # adj = normalize_adj(adj)
        adj_sp = sp.csr_matrix(adj)
        adj_coo = adj_sp.tocoo()

        # Create DGL graph
        dgl_graph = dgl.graph((adj_coo.row, adj_coo.col))
        if not dgl_graph:
            if is_debug:
                print("WARNING: DGL returned false, MIPInfo is empty")
            return MIPInfo()
        if is_debug:
            print("Graph Created in ", time.time() - s, "seconds")

        # TODO should this block use coefficient_matrix or A?
        coeff_adj = np.block([[np.zeros((num_cons, num_cons)), coefficient_matrix],
                              [coefficient_matrix.T, np.zeros((num_vars, num_vars))]])
        coeff_adj_sp = sp.csr_matrix(coeff_adj)
        coeff_adj_coo = coeff_adj_sp.tocoo()

        # TODO: commentary explaining the normalization and edge weight computation
        edge_weights = coeff_adj[coeff_adj_coo.row, coeff_adj_coo.col].flatten()
        edge_weights = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min() + 1e-6)
        edge_weights += 1e-4

        # TODO so we are adding custom fields to the dgl graph here? Some comment would be helpful
        # TODO what's row/column of edge tensor? (after transpose)
        dgl_graph.ndata["feat"] = feature_tensor
        dgl_graph.edata["weight"] = torch.FloatTensor(edge_weights).T

        check_true(dgl_graph.num_nodes() == dgl_graph.ndata["feat"].shape[0],
                   ValueError(f"Error: graph has {dgl_graph.num_nodes()} nodes but "
                              f"feature matrix has {dgl_graph.ndata['feat'].shape[0]} rows"))

        # TODO not sure why we are returning dgl_graph and feature_tensor separately since
        # feature_tensor is already stored in dgl_graph.ndata["feat"]
        # and by the same token, why MIPInfo does not have edge_tensor?
        return MIPInfo(dgl_graph=dgl_graph,
                       feature_tensor=dgl_graph.ndata["feat"],
                       num_cons=num_cons,
                       num_vars=num_vars)

    @staticmethod
    def _get_feature_tensor_num_cons_num_vars(mip_model):
        """
        Extract node-level features from a Gurobi model.

        The function produces a feature tensor where rows correspond to nodes:
        first `num_cons` rows are constraint features, followed by `num_vars` rows
        of variable features. Features are column-normalized to [0, 1].

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
        num_variable_features = 6
        features_of_var = np.zeros((num_vars, num_variable_features), dtype=float)
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
        num_constraint_features = 4
        features_of_constraint = np.zeros((num_cons, num_constraint_features), dtype=float)
        for c, o in list(zip(constraints, operators)):
            features_of_constraint[c, 0] = float(o == '=')
            features_of_constraint[c, 1] = float(o == '<')
            features_of_constraint[c, 2] = float(o == '>')
            features_of_constraint[c, 3] = float(c.RHS)

        # Pad with zeros for equal shapes
        var_feat_matrix = np.hstack([np.zeros((num_vars, features_of_constraint.shape[1])), features_of_var])
        cons_feat_matrix = np.hstack([features_of_constraint, np.zeros((num_cons, features_of_var.shape[1]))])

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
    def get_only_mip_files(input_mip_folder: str, is_sort_by_size:bool = False) -> List[str]:
        """
        Find MIP instance files in a directory and return them sorted by file size.

        Parameters
        ----------
        input_mip_folder : str
            Path to the directory containing MIP instance files.
        is_sort_by_size : bool
            If True, sorts the returned file paths by file size in ascending order.

        Returns
        -------
        List[str]
            Absolute paths to files with extensions `.mps` or `.lp`, optionally sorted by file size (ascending).
        """

        all_filenames = os.listdir(input_mip_folder)
        all_filepaths = [os.path.join(input_mip_folder, filename) for filename in all_filenames]
        mip_filepaths = [p for p in all_filepaths if p.lower().endswith('.mps') or p.lower().endswith('.lp')]

        if is_sort_by_size:
            mip_filepaths = sorted(mip_filepaths, key=os.path.getsize)

        return mip_filepaths

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