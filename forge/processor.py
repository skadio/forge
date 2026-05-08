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
import warnings
from dataclasses import dataclass
from typing import List, Dict, Optional, Sequence, Union

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


@dataclass
class StaticMIPEmbeddings:
    """Container for static embeddings at instance/constraint/variable levels."""

    instance_embedding: np.ndarray
    embedding_of_constraint: Optional[np.ndarray]
    embedding_of_variable: Optional[np.ndarray]
    instance_feature_names: List[str]
    constraint_feature_names: List[str]
    variable_feature_names: List[str]
    feature_dict: Dict[str, float]


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


MIPItem = Union[str, gp.Model]
MIPInput = Union[MIPItem, Sequence[MIPItem]]


class StaticMIPFeatureEmbedder:
    """Compute selected static C++-style MIP features and return embeddings."""

    def __init__(
        self,
        default_for_missing_value: float = -512.0,
        noone: float = 1234.1234,
        eps: float = 1e-10,
    ) -> None:
        self.default_for_missing_value = float(default_for_missing_value)
        self.noone = float(noone)
        self.eps = float(eps)

    def mip_to_static_embeddings(
        self,
        input_mips: MIPInput,
        input_mip_instances_file: Optional[str],
        output_mip_to_embeddings_pkl: Optional[str] = None,
        has_return: bool = True,
    ) -> Optional[Dict[str, StaticMIPEmbeddings]]:
        """Compute static embeddings for one or more MIP inputs.

        Parameters
        ----------
        input_mips : str | gp.Model | Sequence[str | gp.Model]
            Folder path, file path, Gurobi model, or a list/tuple mixing these types.
        input_mip_instances_file : Optional[str]
            If provided and ``input_mips`` contains folders, include only listed instances.
        output_mip_to_embeddings_pkl : Optional[str]
            If provided, saves the resulting dict as pickle.
        has_return : bool
            If True, returns the mapping; otherwise returns None.
        """
        mip_items = _MIPUtils.get_mip_items(input_mips, input_mip_instances_file)
        gurobi_env = _MIPUtils.start_gurobi_env()

        mip_to_embeddings: Dict[str, StaticMIPEmbeddings] = {}
        for mip_item in tqdm(mip_items):
            if isinstance(mip_item, gp.Model):
                mip_model = mip_item
                key = getattr(mip_model, "ModelName", "gurobi_model")
                should_dispose = False
            else:
                mip_model = gp.read(mip_item, env=gurobi_env)
                key = mip_item
                should_dispose = True

            try:
                mip_to_embeddings[key] = self._mip_model_to_embeddings(mip_model)
            except Exception as exc:
                print(f"Error: failed to compute static embeddings for {key!r}: {exc}")
            finally:
                if should_dispose:
                    mip_model.dispose()

        gurobi_env.close()

        if output_mip_to_embeddings_pkl:
            save_pickle(mip_to_embeddings, output_mip_to_embeddings_pkl)

        return mip_to_embeddings if has_return else None

    def _mip_model_to_embeddings(self, mip_model: gp.Model) -> StaticMIPEmbeddings:
        mip_model.update()

        vars_ = mip_model.getVars()
        constrs = mip_model.getConstrs()

        n_vars = int(mip_model.NumVars)
        n_constr = int(mip_model.NumConstrs)

        A = mip_model.getA().tocsr()
        obj = np.array(mip_model.getAttr("Obj", vars_), dtype=float) if n_vars > 0 else np.array([], dtype=float)
        rhs = np.array(mip_model.getAttr("RHS", constrs), dtype=float) if n_constr > 0 else np.array([], dtype=float)
        senses = np.array(mip_model.getAttr("Sense", constrs), dtype="U1") if n_constr > 0 else np.array([], dtype="U1")
        vtypes = np.array(mip_model.getAttr("VType", vars_), dtype="U1") if n_vars > 0 else np.array([], dtype="U1")
        lbs = np.array(mip_model.getAttr("LB", vars_), dtype=float) if n_vars > 0 else np.array([], dtype=float)
        ubs = np.array(mip_model.getAttr("UB", vars_), dtype=float) if n_vars > 0 else np.array([], dtype=float)

        probtype, nq_vars, nq_constr, nq_nzcnt = self._compute_quadratic_metadata(mip_model)

        feature_dict: Dict[str, float] = {
            "probtype": float(probtype),
            "n_vars": float(n_vars),
            "n_constr": float(n_constr),
            "n_nzcnt": float(A.nnz),
            "nq_vars": float(nq_vars),
            "nq_constr": float(nq_constr),
            "nq_nzcnt": float(nq_nzcnt),
        }

        var_type_features, support_size_per_var, is_unbounded_disc = self._compute_variable_type_features(
            vtypes=vtypes,
            lbs=lbs,
            ubs=ubs,
            n_vars=n_vars,
        )
        feature_dict.update(var_type_features)

        support_stats = self._basic_vect(
            values=support_size_per_var[support_size_per_var > 0],
            notcount=self.noone,
            feat_name="support_size",
            which_statistics="avg-median-varcoef-q90mq10",
        )
        feature_dict.update(support_stats)

        constraint_set_stats = self._compute_rhs_features_by_sense(rhs=rhs, senses=senses)
        feature_dict.update(constraint_set_stats)

        var_set_indices = [
            np.where(vtypes != "C")[0],  # set 0: non-continuous
            np.where(vtypes == "C")[0],  # set 1: continuous
            np.arange(n_vars, dtype=int),  # set 2: all
        ]

        vcg_var_deg_full = np.zeros((3, n_vars), dtype=float)
        vcg_var_weight_full = np.zeros((3, n_vars), dtype=float)
        obj_coef_per_constr_full = np.zeros((3, n_vars), dtype=float)
        obj_coef_per_sqr_constr_full = np.zeros((3, n_vars), dtype=float)

        vcg_constr_deg_full = np.zeros((3, n_constr), dtype=float)
        vcg_constr_weight_full = np.zeros((3, n_constr), dtype=float)
        a_norm_varcoefs_full = np.zeros((3, n_constr), dtype=float)

        for s, active_idx in enumerate(var_set_indices):
            graph_out = self._compute_graph_features_for_set(
                A=A,
                rhs=rhs,
                obj=obj,
                active_idx=active_idx,
                var_set_num=s,
                n_vars=n_vars,
                n_constr=n_constr,
            )
            feature_dict.update(graph_out["instance_features"])

            vcg_var_deg_full[s] = graph_out["vcg_var_degree_full"]
            vcg_var_weight_full[s] = graph_out["vcg_var_sum_full"]
            obj_coef_per_constr_full[s] = graph_out["obj_per_constr_full"]
            obj_coef_per_sqr_constr_full[s] = graph_out["obj_per_sqr_full"]

            vcg_constr_deg_full[s] = graph_out["vcg_constraint_degree"]
            vcg_constr_weight_full[s] = graph_out["vcg_constraint_sum"]
            a_norm_varcoefs_full[s] = graph_out["a_normalized_varcoefs"]

        instance_feature_names = self._instance_feature_name_order()
        instance_embedding = np.array(
            [feature_dict.get(name, self.default_for_missing_value) for name in instance_feature_names],
            dtype=float,
        )

        variable_feature_names, embedding_of_variable = self._build_variable_embeddings(
            vtypes=vtypes,
            obj=obj,
            support_size_per_var=support_size_per_var,
            is_unbounded_disc=is_unbounded_disc,
            vcg_var_deg_full=vcg_var_deg_full,
            vcg_var_weight_full=vcg_var_weight_full,
            obj_coef_per_constr_full=obj_coef_per_constr_full,
            obj_coef_per_sqr_constr_full=obj_coef_per_sqr_constr_full,
        )

        constraint_feature_names, embedding_of_constraint = self._build_constraint_embeddings(
            rhs=rhs,
            senses=senses,
            vcg_constr_deg_full=vcg_constr_deg_full,
            vcg_constr_weight_full=vcg_constr_weight_full,
            a_norm_varcoefs_full=a_norm_varcoefs_full,
        )

        return StaticMIPEmbeddings(
            instance_embedding=instance_embedding,
            embedding_of_constraint=embedding_of_constraint,
            embedding_of_variable=embedding_of_variable,
            instance_feature_names=instance_feature_names,
            constraint_feature_names=constraint_feature_names,
            variable_feature_names=variable_feature_names,
            feature_dict=feature_dict,
        )

    def _compute_variable_type_features(
        self,
        vtypes: np.ndarray,
        lbs: np.ndarray,
        ubs: np.ndarray,
        n_vars: int,
    ) -> tuple[Dict[str, float], np.ndarray, np.ndarray]:
        feature_dict: Dict[str, float] = {}

        num_b = int(np.sum(vtypes == "B"))
        num_i = int(np.sum(vtypes == "I"))
        num_c = int(np.sum(vtypes == "C"))
        num_s = int(np.sum(vtypes == "S"))
        num_n = int(np.sum(vtypes == "N"))

        denom_vars = float(n_vars) if n_vars > 0 else 1.0

        feature_dict.update(
            {
                "num_b_variables": float(num_b),
                "num_i_variables": float(num_i),
                "num_c_variables": float(num_c),
                "num_s_variables": float(num_s),
                "num_n_variables": float(num_n),
                "ratio_b_variables": float(num_b / denom_vars),
                "ratio_i_variables": float(num_i / denom_vars),
                "ratio_c_variables": float(num_c / denom_vars),
                "ratio_s_variables": float(num_s / denom_vars),
                "ratio_n_variables": float(num_n / denom_vars),
            }
        )

        non_cont_mask = vtypes != "C"
        num_i_plus = int(np.sum(non_cont_mask))
        feature_dict["num_i+_variables"] = float(num_i_plus)
        feature_dict["ratio_i+_variables"] = float(num_i_plus / denom_vars)

        is_int_like = np.isin(vtypes, ["I", "N"])
        is_unbounded = (ubs >= 0.5 * gp.GRB.INFINITY) | (lbs <= -0.5 * gp.GRB.INFINITY)
        is_unbounded_disc = is_int_like & is_unbounded

        num_unbounded_disc = int(np.sum(is_unbounded_disc))
        denom_non_cont = float(num_i_plus) if num_i_plus > 0 else 1.0

        feature_dict["num_unbounded_disc"] = float(num_unbounded_disc)
        feature_dict["ratio_unbounded_disc"] = float(num_unbounded_disc / denom_non_cont) if num_i_plus > 0 else 0.0

        support_size_per_var = np.zeros(n_vars, dtype=float)
        for idx in np.where(non_cont_mask)[0]:
            vtype = vtypes[idx]
            if vtype == "B":
                support_size_per_var[idx] = 2.0
            elif vtype in {"I", "N"}:
                if is_unbounded_disc[idx]:
                    continue
                if vtype == "I":
                    support_size_per_var[idx] = (ubs[idx] - lbs[idx] + 1.0)
                else:
                    support_size_per_var[idx] = (ubs[idx] - lbs[idx] + 2.0)
            elif vtype == "S":
                support_size_per_var[idx] = 2.0

        return feature_dict, support_size_per_var, is_unbounded_disc.astype(float)

    def _compute_rhs_features_by_sense(self, rhs: np.ndarray, senses: np.ndarray) -> Dict[str, float]:
        feature_dict: Dict[str, float] = {}
        if rhs.size == 0:
            for c in range(3):
                feature_dict[f"rhs_c_{c}_avg"] = self.default_for_missing_value
                feature_dict[f"rhs_c_{c}_varcoef"] = self.default_for_missing_value
            return feature_dict

        sense_map = {0: "<", 1: "=", 2: ">"}
        for c, sense_char in sense_map.items():
            rhs_vals = rhs[senses == sense_char]
            rhs_stats = self._basic_vect(
                values=rhs_vals,
                notcount=self.noone,
                feat_name=f"rhs_c_{c}",
                which_statistics="avg-varcoef",
            )
            feature_dict.update(rhs_stats)
        return feature_dict

    def _compute_graph_features_for_set(
        self,
        A: sp.csr_matrix,
        rhs: np.ndarray,
        obj: np.ndarray,
        active_idx: np.ndarray,
        var_set_num: int,
        n_vars: int,
        n_constr: int,
    ) -> Dict[str, np.ndarray | Dict[str, float]]:
        t0 = time.perf_counter()

        A_sub = A[:, active_idx] if active_idx.size > 0 else sp.csr_matrix((n_constr, 0), dtype=float)

        vcg_constraint_degree = np.asarray(A_sub.getnnz(axis=1), dtype=float).reshape(-1)
        vcg_constraint_sum = np.asarray(A_sub.sum(axis=1), dtype=float).reshape(-1)

        vcg_var_degree_active = np.asarray(A_sub.getnnz(axis=0), dtype=float).reshape(-1)
        vcg_var_sum_active = np.asarray(A_sub.sum(axis=0), dtype=float).reshape(-1)

        vcg_var_degree_full = np.zeros(n_vars, dtype=float)
        vcg_var_sum_full = np.zeros(n_vars, dtype=float)
        vcg_var_degree_full[active_idx] = vcg_var_degree_active
        vcg_var_sum_full[active_idx] = vcg_var_sum_active

        a_normalized_varcoefs = np.zeros(n_constr, dtype=float)
        A_ij_normalized_vals: List[float] = []

        indptr = A_sub.indptr
        data = A_sub.data

        for i in range(n_constr):
            start, end = indptr[i], indptr[i + 1]
            row_vals = data[start:end]
            if row_vals.size == 0:
                a_normalized_varcoefs[i] = 0.0
                continue

            if abs(rhs[i]) > 1e-6:
                A_ij_normalized_vals.extend((row_vals / rhs[i]).tolist())

            abs_vals = np.abs(row_vals)
            denom = float(np.sum(abs_vals))
            if denom <= self.eps:
                a_normalized_varcoefs[i] = 0.0
            else:
                normed = abs_vals / denom
                a_normalized_varcoefs[i] = self._varcoef(normed)

        A_ij_normalized = np.asarray(A_ij_normalized_vals, dtype=float)

        obj_active = np.abs(obj[active_idx]) if active_idx.size > 0 else np.array([], dtype=float)

        obj_per_constr_active = np.zeros_like(obj_active)
        obj_per_sqr_active = np.zeros_like(obj_active)

        constrained_mask = vcg_var_degree_active > 0
        obj_per_constr_active[constrained_mask] = (
            obj_active[constrained_mask] / vcg_var_degree_active[constrained_mask]
        )
        obj_per_sqr_active[constrained_mask] = (
            obj_active[constrained_mask] / np.sqrt(vcg_var_degree_active[constrained_mask])
        )

        obj_per_constr_full = np.zeros(n_vars, dtype=float)
        obj_per_sqr_full = np.zeros(n_vars, dtype=float)
        obj_per_constr_full[active_idx] = obj_per_constr_active
        obj_per_sqr_full[active_idx] = obj_per_sqr_active

        s = str(var_set_num)
        instance_features: Dict[str, float] = {}

        instance_features.update(
            self._basic_vect(
                values=vcg_constraint_degree,
                notcount=self.noone,
                feat_name=f"vcg_constr_deg{s}",
                which_statistics="avg-median-varcoef-q90mq10",
            )
        )
        instance_features.update(
            self._basic_vect(
                values=vcg_var_degree_active,
                notcount=self.noone,
                feat_name=f"vcg_var_deg{s}",
                which_statistics="avg-median-varcoef-q90mq10",
            )
        )
        instance_features.update(
            self._basic_vect(
                values=vcg_constraint_sum,
                notcount=self.noone,
                feat_name=f"vcg_constr_weight{s}",
                which_statistics="avg-varcoef",
            )
        )
        instance_features.update(
            self._basic_vect(
                values=vcg_var_sum_active,
                notcount=self.noone,
                feat_name=f"vcg_var_weight{s}",
                which_statistics="avg-varcoef",
            )
        )
        instance_features.update(
            self._basic_vect(
                values=A_ij_normalized,
                notcount=self.noone,
                feat_name=f"A_ij_normalized{s}",
                which_statistics="avg-varcoef",
            )
        )
        instance_features.update(
            self._basic_vect(
                values=a_normalized_varcoefs,
                notcount=self.noone,
                feat_name=f"a_normalized_varcoefs{s}",
                which_statistics="avg-varcoef",
            )
        )
        instance_features.update(
            self._basic_vect(
                values=obj_active,
                notcount=self.noone,
                feat_name=f"obj_coefs{s}",
                which_statistics="avg-std",
            )
        )
        instance_features.update(
            self._basic_vect(
                values=obj_per_constr_active[constrained_mask],
                notcount=self.noone,
                feat_name=f"obj_coef_per_constr{s}",
                which_statistics="avg-std",
            )
        )
        instance_features.update(
            self._basic_vect(
                values=obj_per_sqr_active[constrained_mask],
                notcount=self.noone,
                feat_name=f"obj_coef_per_sqr_constr{s}",
                which_statistics="avg-std",
            )
        )

        instance_features[f"time_VCG{s}"] = float(time.perf_counter() - t0)

        return {
            "instance_features": instance_features,
            "vcg_var_degree_full": vcg_var_degree_full,
            "vcg_var_sum_full": vcg_var_sum_full,
            "obj_per_constr_full": obj_per_constr_full,
            "obj_per_sqr_full": obj_per_sqr_full,
            "vcg_constraint_degree": vcg_constraint_degree,
            "vcg_constraint_sum": vcg_constraint_sum,
            "a_normalized_varcoefs": a_normalized_varcoefs,
        }

    def _build_variable_embeddings(
        self,
        vtypes: np.ndarray,
        obj: np.ndarray,
        support_size_per_var: np.ndarray,
        is_unbounded_disc: np.ndarray,
        vcg_var_deg_full: np.ndarray,
        vcg_var_weight_full: np.ndarray,
        obj_coef_per_constr_full: np.ndarray,
        obj_coef_per_sqr_constr_full: np.ndarray,
    ) -> tuple[List[str], Optional[np.ndarray]]:
        variable_feature_names = [
            "is_type_B",
            "is_type_I",
            "is_type_C",
            "is_type_S",
            "is_type_N",
            "is_discrete",
            "is_unbounded_discrete",
            "support_size",
            "obj_abs",
            "vcg_var_deg0",
            "vcg_var_deg1",
            "vcg_var_deg2",
            "vcg_var_weight0",
            "vcg_var_weight1",
            "vcg_var_weight2",
            "obj_coef_per_constr0",
            "obj_coef_per_constr1",
            "obj_coef_per_constr2",
            "obj_coef_per_sqr_constr0",
            "obj_coef_per_sqr_constr1",
            "obj_coef_per_sqr_constr2",
        ]

        n_vars = len(vtypes)
        if n_vars == 0:
            return variable_feature_names, None

        is_type_B = (vtypes == "B").astype(float)
        is_type_I = (vtypes == "I").astype(float)
        is_type_C = (vtypes == "C").astype(float)
        is_type_S = (vtypes == "S").astype(float)
        is_type_N = (vtypes == "N").astype(float)
        is_discrete = (vtypes != "C").astype(float)

        embedding_of_variable = np.column_stack(
            [
                is_type_B,
                is_type_I,
                is_type_C,
                is_type_S,
                is_type_N,
                is_discrete,
                is_unbounded_disc,
                support_size_per_var,
                np.abs(obj),
                vcg_var_deg_full[0],
                vcg_var_deg_full[1],
                vcg_var_deg_full[2],
                vcg_var_weight_full[0],
                vcg_var_weight_full[1],
                vcg_var_weight_full[2],
                obj_coef_per_constr_full[0],
                obj_coef_per_constr_full[1],
                obj_coef_per_constr_full[2],
                obj_coef_per_sqr_constr_full[0],
                obj_coef_per_sqr_constr_full[1],
                obj_coef_per_sqr_constr_full[2],
            ]
        ).astype(float)

        return variable_feature_names, embedding_of_variable

    def _build_constraint_embeddings(
        self,
        rhs: np.ndarray,
        senses: np.ndarray,
        vcg_constr_deg_full: np.ndarray,
        vcg_constr_weight_full: np.ndarray,
        a_norm_varcoefs_full: np.ndarray,
    ) -> tuple[List[str], Optional[np.ndarray]]:
        constraint_feature_names = [
            "rhs",
            "sense_le",
            "sense_eq",
            "sense_ge",
            "vcg_constr_deg0",
            "vcg_constr_deg1",
            "vcg_constr_deg2",
            "vcg_constr_weight0",
            "vcg_constr_weight1",
            "vcg_constr_weight2",
            "a_normalized_varcoefs0",
            "a_normalized_varcoefs1",
            "a_normalized_varcoefs2",
        ]

        n_constr = len(rhs)
        if n_constr == 0:
            return constraint_feature_names, None

        sense_le = (senses == "<").astype(float)
        sense_eq = (senses == "=").astype(float)
        sense_ge = (senses == ">").astype(float)

        embedding_of_constraint = np.column_stack(
            [
                rhs,
                sense_le,
                sense_eq,
                sense_ge,
                vcg_constr_deg_full[0],
                vcg_constr_deg_full[1],
                vcg_constr_deg_full[2],
                vcg_constr_weight_full[0],
                vcg_constr_weight_full[1],
                vcg_constr_weight_full[2],
                a_norm_varcoefs_full[0],
                a_norm_varcoefs_full[1],
                a_norm_varcoefs_full[2],
            ]
        ).astype(float)

        return constraint_feature_names, embedding_of_constraint

    def _instance_feature_name_order(self) -> List[str]:
        names: List[str] = [
            "probtype",
            "n_vars",
            "n_constr",
            "n_nzcnt",
            "nq_vars",
            "nq_constr",
            "nq_nzcnt",
            "num_b_variables",
            "num_i_variables",
            "num_c_variables",
            "num_s_variables",
            "num_n_variables",
            "ratio_b_variables",
            "ratio_i_variables",
            "ratio_c_variables",
            "ratio_s_variables",
            "ratio_n_variables",
            "num_i+_variables",
            "ratio_i+_variables",
            "num_unbounded_disc",
            "ratio_unbounded_disc",
            "support_size_avg",
            "support_size_median",
            "support_size_varcoef",
            "support_size_q90mq10",
            "rhs_c_0_avg",
            "rhs_c_0_varcoef",
            "rhs_c_1_avg",
            "rhs_c_1_varcoef",
            "rhs_c_2_avg",
            "rhs_c_2_varcoef",
        ]

        for s in range(3):
            names.extend(
                [
                    f"vcg_constr_deg{s}_avg",
                    f"vcg_constr_deg{s}_median",
                    f"vcg_constr_deg{s}_varcoef",
                    f"vcg_constr_deg{s}_q90mq10",
                    f"vcg_var_deg{s}_avg",
                    f"vcg_var_deg{s}_median",
                    f"vcg_var_deg{s}_varcoef",
                    f"vcg_var_deg{s}_q90mq10",
                    f"vcg_constr_weight{s}_avg",
                    f"vcg_constr_weight{s}_varcoef",
                    f"vcg_var_weight{s}_avg",
                    f"vcg_var_weight{s}_varcoef",
                    f"A_ij_normalized{s}_avg",
                    f"A_ij_normalized{s}_varcoef",
                    f"a_normalized_varcoefs{s}_avg",
                    f"a_normalized_varcoefs{s}_varcoef",
                    f"obj_coefs{s}_avg",
                    f"obj_coefs{s}_std",
                    f"obj_coef_per_constr{s}_avg",
                    f"obj_coef_per_constr{s}_std",
                    f"obj_coef_per_sqr_constr{s}_avg",
                    f"obj_coef_per_sqr_constr{s}_std",
                    f"time_VCG{s}",
                ]
            )

        return names

    def _compute_quadratic_metadata(self, mip_model: gp.Model) -> tuple[int, int, int, int]:
        is_mip = bool(int(mip_model.IsMIP))

        q_obj_nnz = 0
        q_obj_var_count = 0

        try:
            q_obj = mip_model.getQ()
            if q_obj is not None:
                q_obj_coo = q_obj.tocoo()
                if q_obj_coo.nnz > 0:
                    nonzero_mask = np.abs(q_obj_coo.data) > self.eps
                    q_obj_nnz = int(np.count_nonzero(nonzero_mask))
                    if q_obj_nnz > 0:
                        touched = np.concatenate([q_obj_coo.row[nonzero_mask], q_obj_coo.col[nonzero_mask]])
                        q_obj_var_count = int(np.unique(touched).size)
        except Exception:
            q_obj_nnz = 0
            q_obj_var_count = 0

        nq_constr = int(mip_model.NumQConstrs)
        q_constr_nnz = 0

        if nq_constr > 0:
            for qc in mip_model.getQConstrs():
                try:
                    q_constr_nnz += self._count_quadratic_terms(mip_model.getQCRow(qc))
                except Exception:
                    continue

        if nq_constr > 0:
            probtype = 5 if is_mip else 4  # MIQCP or QCP
        elif q_obj_nnz > 0:
            probtype = 3 if is_mip else 2  # MIQP or QP
        else:
            probtype = 1 if is_mip else 0  # MILP or LP

        nq_vars = q_obj_var_count
        nq_nzcnt = int(q_obj_nnz + q_constr_nnz)
        return probtype, nq_vars, nq_constr, nq_nzcnt

    def _count_quadratic_terms(self, qc_row_obj) -> int:
        if qc_row_obj is None:
            return 0

        stack = [qc_row_obj]
        total = 0

        while stack:
            cur = stack.pop()
            if cur is None:
                continue

            if isinstance(cur, (tuple, list)):
                stack.extend(cur)
                continue

            cls_name = cur.__class__.__name__.lower()
            if "quadexpr" in cls_name:
                size_fn = getattr(cur, "size", None)
                if callable(size_fn):
                    try:
                        total += int(size_fn())
                    except Exception:
                        pass

        return total

    def _basic_vect(
        self,
        values: np.ndarray,
        notcount: float,
        feat_name: str,
        which_statistics: str,
    ) -> Dict[str, float]:
        arr = np.asarray(values, dtype=float).reshape(-1)

        if abs(notcount - self.noone) > 1e-10:
            filtered = arr[np.abs(arr - notcount) > 1e-10]
        else:
            filtered = arr

        mysum = 0.0
        mymean = self.default_for_missing_value
        mystd = self.default_for_missing_value
        mymin = self.default_for_missing_value
        mymax = self.default_for_missing_value
        myq10 = self.default_for_missing_value
        myq25 = self.default_for_missing_value
        mymedian = self.default_for_missing_value
        myq75 = self.default_for_missing_value
        myq90 = self.default_for_missing_value
        myvarcoef = self.default_for_missing_value
        myinvvarcoef = self.default_for_missing_value

        n = int(filtered.size)
        if n > 0:
            mysum = float(np.sum(filtered))
            mymean = float(mysum / n)
            mystd = float(np.sqrt(np.mean((filtered - mymean) ** 2)))

            sorted_vals = np.sort(filtered)
            mymin = float(sorted_vals[0])
            myq10 = float(sorted_vals[n // 10])
            myq25 = float(sorted_vals[n // 4])
            mymedian = float(sorted_vals[n // 2])
            myq75 = float(sorted_vals[(3 * n) // 4])
            myq90 = float(sorted_vals[(9 * n) // 10])
            mymax = float(sorted_vals[-1])

            myvarcoef = self._varcoef(filtered)
            std_for_inv = 1e-10 if abs(mystd) < 1e-10 else mystd
            myinvvarcoef = float(mymean / std_for_inv)

        output: Dict[str, float] = {}
        for token in which_statistics.split("-"):
            if token == "avg":
                output[f"{feat_name}_{token}"] = mymean
            elif token == "sum":
                output[f"{feat_name}_{token}"] = mysum
            elif token == "std":
                output[f"{feat_name}_{token}"] = mystd
            elif token == "min":
                output[f"{feat_name}_{token}"] = mymin
            elif token == "max":
                output[f"{feat_name}_{token}"] = mymax
            elif token == "median":
                output[f"{feat_name}_{token}"] = mymedian
            elif token == "q10":
                output[f"{feat_name}_{token}"] = myq10
            elif token == "q25":
                output[f"{feat_name}_{token}"] = myq25
            elif token == "q75":
                output[f"{feat_name}_{token}"] = myq75
            elif token == "q90":
                output[f"{feat_name}_{token}"] = myq90
            elif token == "varcoef":
                output[f"{feat_name}_{token}"] = myvarcoef
            elif token == "invvarcoef":
                output[f"{feat_name}_{token}"] = myinvvarcoef
            elif token == "q75dq25":
                if self._is_missing(myq75) or self._is_missing(myq25):
                    value = self.default_for_missing_value
                elif myq75 < 1e-6:
                    value = 0.0
                elif myq25 < 1e-6:
                    value = self.default_for_missing_value
                else:
                    value = float(myq75 / myq25)
                output[f"{feat_name}_{token}"] = value
            elif token == "q75mq25":
                if self._is_missing(myq75) or self._is_missing(myq25):
                    value = self.default_for_missing_value
                else:
                    value = float(myq75 - myq25)
                output[f"{feat_name}_{token}"] = value
            elif token == "q90mq10":
                if self._is_missing(myq90) or self._is_missing(myq10):
                    value = self.default_for_missing_value
                else:
                    value = float(myq90 - myq10)
                output[f"{feat_name}_{token}"] = value
            elif token == "maxmmin":
                if self._is_missing(mymax) or self._is_missing(mymin):
                    value = self.default_for_missing_value
                else:
                    value = float(mymax - mymin)
                output[f"{feat_name}_{token}"] = value

        return output

    def _varcoef(self, values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return self.default_for_missing_value

        mean = float(np.mean(values))
        std = float(np.sqrt(np.mean((values - mean) ** 2)))
        mean_for_varcoef = 1e-10 if abs(mean) < 1e-10 else mean
        return float(std / mean_for_varcoef)

    def _is_missing(self, value: float) -> bool:
        return abs(float(value) - self.default_for_missing_value) < 1e-10


def mip_to_static_embeddings(
    input_mips: MIPInput,
    input_mip_instances_file: Optional[str],
    output_mip_to_embeddings_pkl: Optional[str] = None,
    has_return: bool = True,
) -> Optional[Dict[str, StaticMIPEmbeddings]]:
    """Convenience wrapper around StaticMIPFeatureEmbedder."""
    embedder = StaticMIPFeatureEmbedder()
    return embedder.mip_to_static_embeddings(
        input_mips=input_mips,
        input_mip_instances_file=input_mip_instances_file,
        output_mip_to_embeddings_pkl=output_mip_to_embeddings_pkl,
        has_return=has_return,
    )
