"""
processor.py

Utilities to convert MIP (Mixed Integer Programming) and SAT (Boolean Satisfiability) 
instances into PyG graph tensors used by the Forge GNN.

This module provides:

MIP Support (Legacy):
- MIPInfo: a lightweight container for MIP instance metadata plus PyG tensors.
- MIPProcessor: read MIP files (MPS/LP format), optionally relax them,
    convert to PyG edge/feature tensors, and save/load pickled representations.

SAT Support (New):
- SATInfo: a lightweight container for SAT instance metadata plus PyG tensors.
- SATEmbeddings: container for learned embeddings from SAT instances.
- SATProcessor: read SAT instances (converted LP/MPS from CNF format),
    extract SAT-specific features (variable and clause features),
    convert to PyG edge/feature tensors, and save/load pickled representations.

Notes
-----
- Designed to work with gurobipy, PyTorch Geometric (PyG), PyTorch and SciPy sparse matrices.
- MIP processor assumes Gurobi model APIs such as getVars, getConstrs and getA.
- SAT processor works with SAT formulas converted to LP/MPS format (via sat_to_mip.py).
- Both processors support single and multi-worker (multiprocessing) conversion.
"""
import os
import warnings
from typing import List, Dict, Optional, Union

import gurobipy as gp
import numpy as np
import scipy.sparse as sp
import torch
import yaml
from tqdm import tqdm

from forge.utils import check_true, Constants, overwrite_if_given, save_pickle, load_pickle
from multiprocessing import get_context
from functools import partial


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

    def __init__(self, instance_name: str = None,
                 num_cons: int = None,
                 num_vars: int = None,
                 feature_tensor: Optional[torch.Tensor] = None,
                 edge_index: Optional[torch.Tensor] = None,
                 edge_weight: Optional[torch.Tensor] = None):
        self.instance_name: Optional[str] = instance_name

        # feature_tensor shape: (num_cons + num_vars, feat_dim=10)
        self.feature_tensor: Optional[torch.Tensor] = feature_tensor
        self.num_cons: Optional[int] = num_cons
        self.num_vars: Optional[int] = num_vars

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
        self.instance_embedding: Optional[np.ndarray] = instance_embedding
        self.embedding_of_constraint: Optional[torch.Tensor] = embedding_of_constraint
        self.embedding_of_variable: Optional[torch.Tensor] = embedding_of_variable


class SATInfo:
    """
    Container for converted SAT (Boolean Satisfiability) instance data.

    Attributes
    ----------
    instance_name : Optional[str]
        Path or unique identifier of the SAT instance.
    feature_tensor : Optional[torch.Tensor]
        Node feature matrix of shape `(num_clauses + num_vars, feat_dim)` with clauses stacked first.
        For SAT: clauses use 4-dim features, variables use 6-dim features (padded to 10-dim).
    num_clauses : Optional[int]
        Number of clauses in the original SAT formula.
    num_vars : Optional[int]
        Number of variables in the original SAT formula.
    edge_index : Optional[torch.LongTensor]
        PyG-style edge index tensor of shape (2, num_edges) representing graph connectivity.
        Row 0 = source indices, row 1 = target indices.
    edge_weight : Optional[torch.FloatTensor]
        Edge weights tensor of shape (num_edges,) corresponding to edges in `edge_index`.
    """

    def __init__(self, instance_name: str = None,
                 num_clauses: int = None,
                 num_vars: int = None,
                 feature_tensor: Optional[torch.Tensor] = None,
                 edge_index: Optional[torch.Tensor] = None,
                 edge_weight: Optional[torch.Tensor] = None):
        self.instance_name: Optional[str] = instance_name

        # feature_tensor shape: (num_clauses + num_vars, feat_dim)
        # Clauses use 10-dim features, variables use 7-dim features (padded to match)
        self.feature_tensor: Optional[torch.Tensor] = feature_tensor
        self.num_clauses: Optional[int] = num_clauses
        self.num_vars: Optional[int] = num_vars

        # - `edge_index`: PyG COO (2, E)
        #   connectivity from source (clause) to target (variable) nodes where there is an edge
        self.edge_index: Optional[torch.Tensor] = edge_index

        # - `edge_weight`: per-edge normalized coefficient values (FloatTensor of length E).
        self.edge_weight: Optional[torch.Tensor] = edge_weight

    @property
    def num_cons(self) -> Optional[int]:
        """Alias for num_clauses to maintain compatibility with MIPInfo interface."""
        return self.num_clauses
    
    @num_cons.setter
    def num_cons(self, value: Optional[int]) -> None:
        """Alias setter for num_clauses to maintain compatibility with MIPInfo interface."""
        self.num_clauses = value


class SATEmbeddings:
    """
    Container for learned embeddings related to a SAT instance.

    Attributes
    ----------
    instance_embedding : Optional[np.ndarray]
        Global embedding for the entire SAT formula of size codebook_size
    embedding_of_clause : Optional[torch.Tensor]
        Per-clause embedding vector (shape: (num_clauses, codebook_dim)).
    embedding_of_variable : Optional[torch.Tensor]
        Per-variable embedding vector (shape: (num_variables, codebook_dim)).
    """

    def __init__(self,
                 instance_embedding: Optional[np.ndarray] = None,
                 embedding_of_clause: Optional[torch.Tensor] = None,
                 embedding_of_variable: Optional[torch.Tensor] = None) -> None:
        """
        Initialize a SATEmbeddings container.

        Parameters
        ----------
        instance_embedding : Optional[np.ndarray], default: None
            Global SAT embedding vector.
        embedding_of_clause : Optional[torch.Tensor], default: None
            Embedding vector per clause.
        embedding_of_variable : Optional[torch.Tensor], default: None
            Embedding vector per variable.
        """
        self.instance_embedding: Optional[np.ndarray] = instance_embedding
        self.embedding_of_clause: Optional[torch.Tensor] = embedding_of_clause
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
                               input_mip_instances_file: Optional[str],
                               output_mip_to_mipinfo_pkl: str,
                               relaxation_list: Optional[List[float]] = None,
                               is_save_relaxed: bool = False,
                               num_parallel_workers: int = 1,
                               has_return: bool = False) -> Optional[Dict[str, MIPInfo]]:
        """
        Converts MIP instances in a given folder to MIPInfo objects and saves them to a pickle file.

        Parameters
        ----------
        input_mip_folder : str
            Path to the directory containing MIP instance files (`.mps` or `.lp`).
        input_mip_instances_file : str
            If provided, only MIP instances listed in this file (one per line) will be processed.
        output_mip_to_mipinfo_pkl : str
            Path where the resulting pickled mapping of instance names to `MIPInfo` objects will be saved.
        relaxation_list : Optional[List[float]], default: None
            If provided, for each instance a set of relaxed instances will be generated by randomly
            removing the specified fraction(s) of constraints (values between 0 and 1).
            Only one relaxed instance is created per ratio in the list.
        is_save_relaxed : bool, default: False
            If True, relaxed MIP instances are written to disk using the original name, plus the relaxation ratio.
        num_parallel_workers : int, default: 1
            Number of parallel worker processes to use for conversion.
        has_return : bool, default: False
            If True the function returns the dictionary mapping instance names to `MIPInfo`,
            otherwise it returns None after saving to `output_file`.

        Returns
        -------
        Optional[Dict[str, List[Any]]]
            Dict mapping instance file paths (or relaxed names) to `MIPInfo` when `has_return` is True;
            Otherwise None.
        """

        if relaxation_list is None:
            relaxation_list = []

        # Normalize worker count
        if not num_parallel_workers or num_parallel_workers < 1:
            num_parallel_workers = 1

        # Find and sort MIP instance files by size
        sorted_mip_files = _MIPUtils.get_only_mip_files(input_mip_folder, input_mip_instances_file,
                                                        is_sort_by_size=True)

        mip_to_mipinfo = {}

        # Sequential path when using a single worker
        if num_parallel_workers == 1:
            # Start Gurobi environment
            gurobi_env = _MIPUtils.start_gurobi_env()

            # Convert each MIP instance to MIPInfo object and store in dictionary
            for idx in tqdm(range(len(sorted_mip_files))):
                
                # Create a local dictionary for this instance (and its relaxations)
                idx_to_mipinfo_dict = MIPProcessor._mip_file_to_mipinfo_dict(sorted_mip_files[idx], gurobi_env,
                                                                             self.rng, relaxation_list, is_save_relaxed)
                # Add mipinfo dict to the global dictionary
                if idx_to_mipinfo_dict:
                    mip_to_mipinfo.update(idx_to_mipinfo_dict)

            # Close Gurobi environment
            gurobi_env.close()
        else:
            # Parallel path using multiprocessing with the requested number of workers
            ctx = get_context("spawn")
            with ctx.Pool(processes=num_parallel_workers) as pool:
                worker = partial(MIPProcessor._run_mipinfo_worker,
                                 is_save_relaxed=is_save_relaxed,
                                 relaxation_list=relaxation_list,
                                 rng=self.rng)

                for result in tqdm(pool.imap_unordered(worker, sorted_mip_files), total=len(sorted_mip_files)):
                    if result:
                        mip_to_mipinfo.update(result)

        save_pickle(mip_to_mipinfo, output_mip_to_mipinfo_pkl)

        return mip_to_mipinfo if has_return else None

    @staticmethod
    def _mip_file_to_mipinfo_dict(mip_file: str, gurobi_env, rng,
                                  relaxation_list: List[float], is_save_relaxed: bool) -> Optional[Dict[str, MIPInfo]]:
        
        print("<< Start: Convert to mipinfo:", mip_file)
        
        mip_to_mipinfo_local: Dict[str, MIPInfo] = {}

        # Read MIP file to a Gurobi model
        mip_model = gp.read(mip_file, env=gurobi_env)

        # Generate MIPInfo object from original model, set name, and add to dictionary
        # Catch numeric issues during conversion in feature normalization
        try:
            # Convert NumPy invalid-value operations into FloatingPointError
            with np.errstate(invalid='raise'):
                # Turn RuntimeWarning into an exception for this call
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=RuntimeWarning)
                    mipinfo = MIPProcessor._mip_model_to_mipinfo(mip_model)
        except (FloatingPointError, RuntimeWarning) as e:
            print(f"Error: Numeric Issue to convert MIP {mip_file} to MIPInfo due to error: {e}")
            return None
        except Exception as e:
            print(f"Error: Failed to convert MIP {mip_file} to MIPInfo due to error: {e}")
            return None

        mipinfo.instance_name = mip_file
        mip_to_mipinfo_local[mipinfo.instance_name] = mipinfo

        if relaxation_list:
            for ratio in relaxation_list:

                # Create a copy of the original model to remove constraints from
                mip_model_relaxed = mip_model.copy()
                cons = mip_model_relaxed.getConstrs()

                # Choose a random number of constraints within the ratio to remove
                k = int(len(cons) * ratio)
                if k <= 0:
                    continue

                # Removing constraints might lead to an exception, repeat until success
                success = False
                while not success:
                    try:
                        # Choose a random subset of constraints
                        cons_remove_ = rng.choice(cons, k, replace=False)

                        # Remove them from the copy model
                        for c in cons_remove_:
                            mip_model_relaxed.remove(c)
                        mip_model_relaxed.update()
                        success = True
                    except Exception as e:
                        print(f"Retrying due to exception: {e}")

                # Generate MIPInfo object from relaxed model, set name, and add to dictionary
                # Catch numeric issues during conversion in feature normalization
                try:
                    # Convert NumPy invalid-value operations into FloatingPointError
                    with np.errstate(invalid='raise'):
                        # Turn RuntimeWarning into an exception for this call
                        with warnings.catch_warnings():
                            warnings.filterwarnings("error", category=RuntimeWarning)
                            mipinfo = MIPProcessor._mip_model_to_mipinfo(mip_model)
                except (FloatingPointError, RuntimeWarning) as e:
                    print(f"Error: Numeric Issue to convert RELAXED MIP {mip_file} to MIPInfo due to error: {e}")
                    continue
                except Exception as e:
                    print(f"Error: Failed to convert RELAXED MIP {mip_file} to MIPInfo due to error: {e}")
                    continue

                # Generate relaxation name. Handle double extensions ".lp.gz" etc.
                base, ext = os.path.splitext(mip_file)
                if ext == ".gz":
                    base2, ext2 = os.path.splitext(base)
                    mipinfo.instance_name = f"{base2}_relaxed_{ratio}{ext2}{ext}"
                else:
                    mipinfo.instance_name = f"{base}_relaxed_{ratio}{ext}"

                # Store mipinfo
                mip_to_mipinfo_local[mipinfo.instance_name] = mipinfo

                # Save the perturbed MIP instance to disk
                if is_save_relaxed:
                    mip_model_relaxed.write(mipinfo.instance_name)

                # Release copy model
                mip_model_relaxed.dispose()

        print(">> Finish: Convert to mipinfo:", mip_file)
        return mip_to_mipinfo_local

    @staticmethod
    def _mip_model_to_mipinfo(mip_model: gp.Model) -> MIPInfo:
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
            The Gurobi model to convert.
            The function will mutate the model (remove zero-columns) to ensure a valid bipartite incidence.

        Returns
        -------
        MIPInfo
            On success, MIPInfo.
            On failure, propagates errors to the caller.
        """

        # Remove zero-column variables (vars with no coefficients) to ensure valid bipartite graph
        # This might error on relaxed instances e.g; MIPLIB_train_supportcase27i.mps.gz
        to_remove = [v for v in mip_model.getVars() if not mip_model.getCol(v).size()]
        mip_model.remove(to_remove)
        mip_model.update()

        # Get static feature tensor from MIP, number of constraints, and number of variables
        # Feature tensor shape: (num_cons + num_vars, feat_dim=10)
        feature_tensor, num_cons, num_vars = MIPProcessor._get_feature_tensor_num_cons_num_vars(mip_model)

        # Get edge indexes and weights in Tensor, ready for PyG
        edge_index, edge_weight = MIPProcessor._get_edge_index_weight(mip_model, num_cons, num_vars)

        return MIPInfo(num_cons=num_cons, num_vars=num_vars, feature_tensor=feature_tensor,
                       edge_index=edge_index, edge_weight=edge_weight)

    @staticmethod
    def _run_mipinfo_worker(mip_file: str,
                            is_save_relaxed: bool,
                            relaxation_list: List[float],
                            rng) -> Optional[Dict[str, MIPInfo]]:

        """Worker function to compute MIPInfo for a single MIP instance.

        This function is designed to be picklable so it can be used with multiprocessing.
        It creates and tears down its own Gurobi environment inside each worker process.

        On success, returns Dict[str, MIPInfo] for the MIP file and its relaxations.
        On fail, returns None.
        """

        try:
            # Start Gurobi environment and set limits for this worker
            gurobi_env = _MIPUtils.start_gurobi_env()

            # Create mipinfo dict for file and its relaxations
            idx_to_mipinfo_dict = MIPProcessor._mip_file_to_mipinfo_dict(mip_file, gurobi_env,
                                                                         rng, relaxation_list, is_save_relaxed)
            gurobi_env.close()

            # Return local mipinfo dict
            return idx_to_mipinfo_dict

        except Exception as exc:
            print(f"\nError while processing {mip_file}: {exc}")
            try:
                # Best-effort cleanup; in some failure modes env may not exist
                gurobi_env.close()
            except Exception:
                pass
            return None

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


class _MIPUtils:

    @staticmethod
    def get_mip_items(input_mips, input_mip_instances_file):
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
                mip_items.extend(_MIPUtils.get_only_mip_files(item, input_mip_instances_file, is_sort_by_size=False))
            elif isinstance(item, str) and os.path.isfile(item):
                mip_items.append(item)
            else:
                raise ValueError(
                    f"Error: Input {item!r} is neither a directory, a file, nor a gurobipy model instance.")
        return mip_items

    @staticmethod
    def get_only_mip_files(input_mip_folder: str, input_mip_instances_file: str,
                           is_sort_by_size: bool = False) -> List[str]:
        """
        Find MIP instance files in a directory and return them sorted by file size.

        Parameters
        ----------
        input_mip_folder : str
            Path to the directory containing MIP instance files.
        input_mip_instances_file: str
            If provided, only include instances from input_mip_folder listed in the file.
        is_sort_by_size : bool
            If True, sorts the returned file paths by file size in ascending order/smallest first.

        Returns
        -------
        List[str]
            Absolute paths to files with extensions `.mps` or `.lp`, optionally sorted by file size (ascending).
        """

        # Get full paths and filter for MIP files
        try:
            all_filenames = os.listdir(input_mip_folder)
        except Exception as e:
            raise ValueError(f"Error: cannot list directory `{'input_mip_folder'}`: {e}")

        all_filepaths = [os.path.join(input_mip_folder, filename) for filename in all_filenames]
        mip_filepaths = [p for p in all_filepaths if p.lower().endswith('.mps') or
                         p.lower().endswith('.lp') or
                         p.lower().endswith('.mps.gz') or
                         p.lower().endswith('.lp.gz')]

        # Filter by input_mip_instance_file, if provided
        if input_mip_instances_file is not None:
            try:
                with open(input_mip_instances_file, 'r') as f:
                    allowed_instances = set(line.strip() for line in f if line.strip())
            except Exception as e:
                raise ValueError(f"Error: cannot read instances file `{input_mip_instances_file}`: {e}")
            mip_filepaths = [p for p in mip_filepaths if os.path.basename(p) in allowed_instances]

        # Smallest sized mip first
        if is_sort_by_size:
            mip_filepaths = sorted(mip_filepaths, key=os.path.getsize)

        return mip_filepaths

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
    def start_gurobi_env():
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


class SATProcessor:
    """
    Processor that converts SAT (Boolean Satisfiability) instances to graph-based features.

    SAT instances are typically provided in DIMACS CNF format or converted LP/MPS format
    (from CNF via sat_to_mip.py). This processor extracts SAT-specific features including:
    - Variable features (6-dim): degree, pos_degree, neg_degree, pos_neg_ratio, pos_deg_norm, neg_deg_norm
    - Clause features (4-dim): width, pos_count, neg_count, pos_neg_ratio

    Usage
    -----
    - Initialize with an optional training config file to set RNG seeds.
    - Call convert_sat_lp_to_satinfo to read converted SAT LP files and produce SATInfo objects.
    - Use load_satinfo_from_pickles to aggregate multiple pickles into a list.

    Parameters
    ----------
    train_config_file_path : Optional[str]
        Path to YAML train config. Uses Constants.default_train_config_yaml by default.
    seed : Optional[int]
        Seed to override the config RNG seed.
    """

    # SAT feature dimensions (10-dim total, matching MIP structure)
    NUM_VARIABLE_FEATURES = 6  # degree, pos_degree, neg_degree, pos_neg_ratio, pos_deg_norm, neg_deg_norm
    NUM_CLAUSE_FEATURES = 4    # width, pos_count, neg_count, pos_neg_ratio
    TOTAL_FEATURE_DIM = 10     # Both clauses and variables padded to 10-dim (matching MIP)

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

    def convert_sat_lp_to_satinfo(self,
                                   input_sat_folder: str,
                                   input_sat_instances_file: Optional[str],
                                   output_sat_to_satinfo_pkl: str,
                                   num_parallel_workers: int = 1,
                                   has_return: bool = False,
                                   max_graph_nodes: Optional[int] = 100000) -> Optional[Dict[str, SATInfo]]:
        """
        Converts SAT instances (in LP/MPS format) to SATInfo objects and saves them to a pickle file.

        Parameters
        ----------
        input_sat_folder : str
            Path to the directory containing SAT instance files (converted to `.lp` or `.mps`).
        input_sat_instances_file : str
            If provided, only SAT instances listed in this file (one per line) will be processed.
        output_sat_to_satinfo_pkl : str
            Path where the resulting pickled mapping of instance names to `SATInfo` objects will be saved.
        num_parallel_workers : int, default: 1
            Number of parallel worker processes to use for conversion.
        has_return : bool, default: False
            If True the function returns the dictionary mapping instance names to `SATInfo`,
            otherwise it returns None after saving to `output_file`.
        max_graph_nodes : Optional[int], default: 100000
            Maximum number of nodes (clauses + variables) in the SAT graph. Instances exceeding
            this limit are skipped since they would be filtered out during pretraining anyway.

        Returns
        -------
        Optional[Dict[str, SATInfo]]
            Dict mapping instance file paths to `SATInfo` when `has_return` is True;
            Otherwise None.
        """

        # Normalize worker count
        if not num_parallel_workers or num_parallel_workers < 1:
            num_parallel_workers = 1

        # Find and sort SAT instance files by size
        sorted_sat_files = _SATUtils.get_only_sat_files(input_sat_folder, input_sat_instances_file,
                                                        is_sort_by_size=True)

        sat_to_satinfo = {}

        # Sequential path when using a single worker
        if num_parallel_workers == 1:
            # Convert each SAT instance to SATInfo object and store in dictionary
            for idx in tqdm(range(len(sorted_sat_files))):
                
                # Create a local dictionary for this instance
                idx_to_satinfo_dict = SATProcessor._sat_file_to_satinfo_dict(sorted_sat_files[idx],
                                                                            self.rng,
                                                                            max_graph_nodes=max_graph_nodes)
                # Add satinfo dict to the global dictionary
                if idx_to_satinfo_dict:
                    sat_to_satinfo.update(idx_to_satinfo_dict)
        else:
            # Parallel path using multiprocessing with the requested number of workers
            ctx = get_context("spawn")
            with ctx.Pool(processes=num_parallel_workers) as pool:
                worker = partial(SATProcessor._run_satinfo_worker)

                for result in tqdm(pool.imap_unordered(worker, sorted_sat_files), total=len(sorted_sat_files)):
                    if result:
                        sat_to_satinfo.update(result)

        save_pickle(sat_to_satinfo, output_sat_to_satinfo_pkl)

        return sat_to_satinfo if has_return else None

    @staticmethod
    def _sat_file_to_satinfo_dict(sat_file: str, rng, max_graph_nodes: Optional[int] = 100000) -> Optional[Dict[str, SATInfo]]:
        
        print(f"<< Start: Convert to satinfo: {sat_file}")
        
        sat_to_satinfo_local: Dict[str, SATInfo] = {}

        # Attempt to load the SAT/MIP model from LP/MPS file
        try:
            # Create a minimal Gurobi environment just to read the model
            gurobi_env = _SATUtils.start_gurobi_env()
            sat_model = gp.read(sat_file, env=gurobi_env)
            gurobi_env.close()

            # Generate SATInfo object from model
            try:
                with np.errstate(invalid='raise'):
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error", category=RuntimeWarning)
                        satinfo = SATProcessor._sat_model_to_satinfo(sat_model)
            except (FloatingPointError, RuntimeWarning) as e:
                print(f"Error: Numeric Issue converting SAT {sat_file} to SATInfo: {e}")
                return None
            except Exception as e:
                print(f"Error: Failed to convert SAT {sat_file} to SATInfo: {e}")
                return None

            # Check if instance size exceeds max_graph_nodes
            num_nodes = satinfo.num_clauses + satinfo.num_vars
            if max_graph_nodes and num_nodes > max_graph_nodes:
                print(f"Skipping {sat_file}: num_nodes={num_nodes} exceeds max_graph_nodes={max_graph_nodes}")
                return None

            satinfo.instance_name = sat_file
            sat_to_satinfo_local[satinfo.instance_name] = satinfo
                
        except Exception as e:
            print(f"Error: Failed to read SAT file {sat_file}: {e}")
            return None

        print(f">> Finish: Convert to satinfo: {sat_file}")
        return sat_to_satinfo_local

    @staticmethod
    def _run_satinfo_worker(sat_file: str) -> Optional[Dict[str, SATInfo]]:

        """Worker function to compute SATInfo for a single SAT instance.

        This function is designed to be picklable so it can be used with multiprocessing.
        It creates and tears down its own Gurobi environment inside each worker process.

        On success, returns Dict[str, SATInfo] for the SAT file.
        On fail, returns None.
        """

        try:
            # Create a temporary RNG for this worker
            temp_rng = np.random.default_rng()
            
            # Create satinfo dict for file
            idx_to_satinfo_dict = SATProcessor._sat_file_to_satinfo_dict(sat_file, temp_rng)

            # Return local satinfo dict
            return idx_to_satinfo_dict

        except Exception as exc:
            print(f"\nError while processing {sat_file}: {exc}")
            return None

    @staticmethod
    def _sat_model_to_satinfo(sat_model: gp.Model) -> SATInfo:
        """
        Convert a Gurobi model (loaded from SAT-converted LP/MPS) into a SATInfo.

        The produced SATInfo contains:
        - `feature_tensor`: node features stacked with clauses (constraints) first then variables.
            Feature tensor shape: (num_clauses + num_vars, feat_dim)
        - `num_clauses`, `num_vars`: counts used to interpret the graph layout.
        - `edge_index`: PyG COO (2, E)
        - `edge_weight`: per-edge normalized coefficient values (FloatTensor of length E).

        Parameters
        ----------
        sat_model : gp.Model
            The Gurobi model to convert (loaded from SAT LP/MPS file).

        Returns
        -------
        SATInfo
            On success, SATInfo.
            On failure, propagates errors to the caller.
        """

        # Get feature tensor from SAT model (constraints = clauses, variables = variables)
        feature_tensor, num_clauses, num_vars = SATProcessor._get_sat_feature_tensor_num_clauses_num_vars(sat_model)

        # Get edge indexes and weights in Tensor, ready for PyG
        edge_index, edge_weight = SATProcessor._get_sat_edge_index_weight(sat_model, num_clauses, num_vars)

        return SATInfo(num_clauses=num_clauses, num_vars=num_vars, feature_tensor=feature_tensor,
                       edge_index=edge_index, edge_weight=edge_weight)

    @staticmethod
    def _get_sat_feature_tensor_num_clauses_num_vars(sat_model):
        """
        Extract node-level features from a SAT model (loaded from LP/MPS).

        For SAT: clauses map to constraints, variables map to variables.
        - Clause features: 4-dim (width, pos_count, neg_count, pos_neg_ratio)
        - Variable features: 6-dim (degree, pos_deg, neg_deg, pos_neg_ratio, pos_deg_norm, neg_deg_norm)

        The function produces a feature tensor where rows correspond to nodes:
            - first `num_clauses` rows are clause features (4-dim, padded to 10-dim),
            - followed by `num_vars` rows of variable features (6-dim, padded to 10-dim).
        Features are column-normalized to [0, 1].

        Parameters
        ----------
        sat_model : gp.Model
            Gurobi model representing a SAT formula (converted from CNF via sat_to_mip.py).

        Returns
        -------
        tuple
            (feature_tensor: torch.FloatTensor, num_clauses: int, num_vars: int)
        """

        # Get variables and count them
        variables = sat_model.getVars()
        num_vars = len(variables)

        # Initialize variable features (6-dim: degree, pos_degree, neg_degree, pos_neg_ratio, pos_deg_norm, neg_deg_norm)
        features_of_var = np.zeros((num_vars, SATProcessor.NUM_VARIABLE_FEATURES), dtype=float)

        # Count positive and negative occurrences of each variable in constraints
        pos_deg_counts = np.zeros(num_vars, dtype=int)
        neg_deg_counts = np.zeros(num_vars, dtype=int)

        # Extract variable occurrences from the constraint matrix
        constraints = sat_model.getConstrs()
        num_clauses = len(constraints)
        
        for c_idx, constraint in enumerate(constraints):
            expr = sat_model.getRow(constraint)
            for i in range(expr.size()):
                var = expr.getVar(i)
                coeff = expr.getCoeff(i)
                var_idx = var.VarName[1:] if var.VarName.startswith('x') else var.VarName
                try:
                    var_idx = int(var_idx) - 1  # Convert x1 -> 0, x2 -> 1, etc.
                    if 0 <= var_idx < num_vars:
                        if coeff > 0:
                            pos_deg_counts[var_idx] += 1
                        else:
                            neg_deg_counts[var_idx] += 1
                except (ValueError, IndexError):
                    pass

        # Calculate statistics for normalization
        pos_deg_values = pos_deg_counts / (num_clauses + 1e-9)
        neg_deg_values = neg_deg_counts / (num_clauses + 1e-9)
        pos_neg_ratio_values = np.divide(pos_deg_counts, neg_deg_counts + 1, 
                                        out=np.zeros_like(pos_deg_counts, dtype=float))

        mean_pos_deg = np.mean(pos_deg_values) if num_vars > 0 else 0
        mean_neg_deg = np.mean(neg_deg_values) if num_vars > 0 else 0
        mean_pos_neg_ratio = np.mean(pos_neg_ratio_values) if num_vars > 0 else 0

        # Fill in variable features
        for i, var in enumerate(variables):
            degree = (pos_deg_counts[i] + neg_deg_counts[i]) / (num_clauses + 1e-9)
            pos_deg = pos_deg_values[i]
            neg_deg = neg_deg_values[i]
            pos_neg_ratio = pos_neg_ratio_values[i]

            # Normalize by means (avoid division by zero)
            pos_deg_normalized = pos_deg / (mean_pos_deg + 1e-9)
            neg_deg_normalized = neg_deg / (mean_neg_deg + 1e-9)
            pos_neg_ratio_normalized = pos_neg_ratio / (mean_pos_neg_ratio + 1e-9)

            features_of_var[i, 0] = degree
            features_of_var[i, 1] = pos_deg
            features_of_var[i, 2] = neg_deg
            features_of_var[i, 3] = pos_neg_ratio
            features_of_var[i, 4] = pos_deg_normalized
            features_of_var[i, 5] = neg_deg_normalized

        # Create clause features (4-dim: width, pos_count, neg_count, pos_neg_ratio)
        features_of_clause = np.zeros((num_clauses, SATProcessor.NUM_CLAUSE_FEATURES), dtype=float)

        # Track clause statistics for normalization
        clause_widths = []
        clause_pos_counts = []
        clause_neg_counts = []
        clause_pos_neg_ratios = []

        for c_idx, constraint in enumerate(constraints):
            expr = sat_model.getRow(constraint)
            width = expr.size()
            pos_count = 0
            neg_count = 0

            for i in range(expr.size()):
                coeff = expr.getCoeff(i)
                if coeff > 0:
                    pos_count += 1
                else:
                    neg_count += 1

            clause_widths.append(width)
            clause_pos_counts.append(pos_count)
            clause_neg_counts.append(neg_count)
            pos_neg_ratio = pos_count / (neg_count + 1)
            clause_pos_neg_ratios.append(pos_neg_ratio)

        # Fill in clause features (4-dim: width, pos_count, neg_count, pos_neg_ratio)
        for c_idx, constraint in enumerate(constraints):
            expr = sat_model.getRow(constraint)
            width = expr.size()
            pos_count = clause_pos_counts[c_idx]
            neg_count = clause_neg_counts[c_idx]
            pos_neg_ratio = clause_pos_neg_ratios[c_idx]

            features_of_clause[c_idx, 0] = width
            features_of_clause[c_idx, 1] = pos_count
            features_of_clause[c_idx, 2] = neg_count
            features_of_clause[c_idx, 3] = pos_neg_ratio

        # Pad with zeros for equal shapes
        clause_feat_matrix = np.hstack([features_of_clause, np.zeros((num_clauses, features_of_var.shape[1]))])
        var_feat_matrix = np.hstack([np.zeros((num_vars, features_of_clause.shape[1])), features_of_var])

        # Stack up into one feature matrix, clauses come first
        feature_matrix = np.vstack([clause_feat_matrix, var_feat_matrix])

        # Column normalize features
        feature_matrix = (feature_matrix - np.min(feature_matrix, axis=0)) / (
                np.max(feature_matrix, axis=0) - np.min(feature_matrix, axis=0) + 1e-9)
        feature_matrix[np.isnan(feature_matrix)] = 0

        # Convert features to tensor
        feature_tensor = torch.FloatTensor(np.array(feature_matrix))

        # Return feature tensor, number of clauses, and number of variables
        return feature_tensor, num_clauses, num_vars

    @staticmethod
    def _get_sat_edge_index_weight(sat_model, num_clauses, num_vars):
        """
        Extract edges and weights from a SAT model.

        Edges represent clause-variable connections in the CNF formula.

        Parameters
        ----------
        sat_model : gp.Model
            Gurobi model representing a SAT formula.
        num_clauses : int
            Number of clauses (constraints) in the formula.
        num_vars : int
            Number of variables.

        Returns
        -------
        tuple
            (edge_index: torch.LongTensor, edge_weight: torch.FloatTensor)
        """

        # Get the coefficient matrix (sparse representation of clause-variable connections)
        coef_sp = sat_model.getA()

        # Create empty sparse blocks for padding
        top_left = sp.csr_matrix((num_clauses, num_clauses))
        bottom_right = sp.csr_matrix((num_vars, num_vars))
        coeff_adj_sp = sp.bmat([[top_left, coef_sp], [coef_sp.transpose(), bottom_right]], format='coo')

        edge_index = torch.tensor(np.vstack([coeff_adj_sp.row, coeff_adj_sp.col]), dtype=torch.long)

        # Convert coefficients to float for normalization
        edge_weights_np = coeff_adj_sp.data.astype(float)

        # Normalize edge weights
        edge_weights = SATProcessor._normalize_sat_edge_weights(edge_weights_np)

        # Validate dimensions
        check_true(edge_index.shape[1] == edge_weights.shape[0],
                   ValueError(f"Error: edge_index has {edge_index.shape[1]} edges but "
                              f"edge_weight has {edge_weights.shape[0]} entries"))

        return edge_index, edge_weights

    @staticmethod
    def _normalize_sat_edge_weights(edge_weights: Optional[np.ndarray],
                                   eps: float = 1e-12,
                                   small_eps: float = 1e-4) -> torch.FloatTensor:
        """
        Normalize SAT edge weights using the small-eps-only strategy.

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


class _SATUtils:

    @staticmethod
    def get_only_sat_files(input_sat_folder: str, input_sat_instances_file: Optional[str],
                           is_sort_by_size: bool = False) -> List[str]:
        """
        Find SAT instance files in a directory and return them sorted by file size.

        Parameters
        ----------
        input_sat_folder : str
            Path to the directory containing SAT instance files (converted LP/MPS).
        input_sat_instances_file : Optional[str]
            If provided, only include instances from input_sat_folder listed in the file.
        is_sort_by_size : bool
            If True, sorts the returned file paths by file size in ascending order (smallest first).

        Returns
        -------
        List[str]
            Absolute paths to files with extensions `.lp`, `.mps` (and .gz variants), 
            optionally sorted by file size (ascending).
        """

        # Get full paths and filter for SAT files
        try:
            all_filenames = os.listdir(input_sat_folder)
        except Exception as e:
            raise ValueError(f"Error: cannot list directory `{input_sat_folder}`: {e}")

        all_filepaths = [os.path.join(input_sat_folder, filename) for filename in all_filenames]
        sat_filepaths = [p for p in all_filepaths if p.lower().endswith('.lp') or
                         p.lower().endswith('.mps') or
                         p.lower().endswith('.lp.gz') or
                         p.lower().endswith('.mps.gz')]

        # Filter by input_sat_instances_file, if provided
        if input_sat_instances_file is not None:
            try:
                with open(input_sat_instances_file, 'r') as f:
                    allowed_instances = set(line.strip() for line in f if line.strip())
            except Exception as e:
                raise ValueError(f"Error: cannot read instances file `{input_sat_instances_file}`: {e}")
            sat_filepaths = [p for p in sat_filepaths if os.path.basename(p) in allowed_instances]

        # Smallest sized sat file first
        if is_sort_by_size:
            sat_filepaths = sorted(sat_filepaths, key=os.path.getsize)

        return sat_filepaths

    @staticmethod
    def load_satinfo_from_pickles(sat_to_satinfo_files: List[str]) -> List[SATInfo]:
        """
        Load and aggregate lists of SATInfo objects from multiple pickled files.

        Parameters
        ----------
        sat_to_satinfo_files : List[str]
            List of paths to pickled mappings (saved by `convert_sat_lp_to_satinfo`).

        Returns
        -------
        List[SATInfo]
            Flattened list of `SATInfo` objects.
        """
        satinfo_list = []
        for sat_to_satinfo_file in sat_to_satinfo_files:
            sat_to_satinfo = load_pickle(sat_to_satinfo_file)
            for sat in sat_to_satinfo:
                satinfo_list.append(sat_to_satinfo[sat])
        return satinfo_list

    @staticmethod
    def start_gurobi_env():
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
