import urllib.parse
import os
import pickle
from typing import Union, NamedTuple
import h5py
import torch

import numpy as np
import scipy.sparse as sp

Num = Union[int, float]
"""Num type is defined as integer or float."""


class Constants(NamedTuple):
    """
    Constant values used by the modules.
    """

    MIN_PROBLEMS = {"SC", "MVC"}
    MAX_PROBLEMS = {"GISP", "CA"}
    NUM_VARIABLE_FEATURES = 6
    NUM_CONSTRAINT_FEATURES = 4

    # Forge Model Types
    FORGE_PRE_TRAIN = "pretrain"
    FORGE_FINE_TUNE_INTEGRAL_GAP = "fine_tune_integral_gap"
    FORGE_FINE_TUNE_VARIABLE_PROBA = "fine_tune_variable_proba"

    # Names
    _DATA_DIR_NAME = "data"
    _INSTANCES_DIR_NAME = "instances"
    _FORGE_DIR_NAME = "forge"
    _CONFIGS_DIR_NAME = "configs"
    _MODELS_DIR_NAME = "models"
    _TESTS_DIR_NAME = "tests"
    _TRAIN_CONFIG_NAME = "train_config.yaml"
    _MIPINFO_NAME = "mip_to_mipinfo.pkl"
    _EMBEDDINGS_NAME = "mip_to_embeddings.pkl"
    _GAPINFO_NAME = "mip_to_gapinfo.pkl"
    _FORGE_PKL_NAME = "forge_pretrained.pkl"
    _FORGE_LOG_NAME = "forge_pretrain.log"
    _UNIT_TEST_INSTANCES_NAME = "instances_unit_test.txt"

    # Folders
    _CONST_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_INSTANCE_DIR = _CONST_FILE_DIR + os.sep + ".." + os.sep + _DATA_DIR_NAME + os.sep + _INSTANCES_DIR_NAME
    DATA_TEST_INSTANCE_DIR = _CONST_FILE_DIR + os.sep + ".." + os.sep + _TESTS_DIR_NAME + os.sep + _DATA_DIR_NAME + os.sep + _INSTANCES_DIR_NAME
    MODELS_DIR = _CONST_FILE_DIR + os.sep + ".." + os.sep + _MODELS_DIR_NAME
    CONFIGS_DIR = _CONST_FILE_DIR + os.sep + ".." + os.sep + _FORGE_DIR_NAME + os.sep + _CONFIGS_DIR_NAME

    # File paths
    default_train_config_yaml = _CONST_FILE_DIR + os.sep + _CONFIGS_DIR_NAME + os.sep + _TRAIN_CONFIG_NAME
    default_mip_to_embeddings_pkl = _CONST_FILE_DIR + os.sep + ".." + os.sep + _TESTS_DIR_NAME + os.sep + _MIPINFO_NAME
    default_mip_to_mipinfo_pkl = _CONST_FILE_DIR + os.sep + ".." + os.sep + _TESTS_DIR_NAME + os.sep + _MIPINFO_NAME
    default_mip_to_gapinfo_pkl = _CONST_FILE_DIR + os.sep + ".." + os.sep + _TESTS_DIR_NAME + os.sep + _GAPINFO_NAME
    default_forge_pretrained_pkl = _CONST_FILE_DIR + os.sep + ".." + os.sep + _TESTS_DIR_NAME + os.sep + _FORGE_PKL_NAME
    default_forge_log_file = _CONST_FILE_DIR + os.sep + ".." + os.sep + _TESTS_DIR_NAME + os.sep + _FORGE_LOG_NAME
    default_instances_unit_test_txt = _CONST_FILE_DIR + os.sep + ".." + os.sep + _TESTS_DIR_NAME + os.sep + _DATA_DIR_NAME + os.sep + _UNIT_TEST_INSTANCES_NAME


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


def _to_numpy(x, dtype=None):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if dtype is not None:
        return arr.astype(dtype)
    return arr


def _safe_group_name(key: str) -> str:
    """
    Encode arbitrary string to a name safe for HDF5 groups (no '/').
    Uses percent-encoding so it is reversible.
    """
    return urllib.parse.quote(str(key), safe="")


def save_mip_embeddings_hdf5(mip_to_embeddings: dict, output_h5_path: str,
                             dtype=np.float32, compression="gzip", compression_opts=4):
    os.makedirs(os.path.dirname(output_h5_path) or ".", exist_ok=True)
    with h5py.File(output_h5_path, "w") as hf:
        for key, emb in mip_to_embeddings.items():
            grp_name = _safe_group_name(str(key))
            g = hf.create_group(grp_name)
            # store original key so we can restore exact path when loading
            g.attrs['orig_key'] = str(key)
            for attr in ("instance_embedding", "embedding_of_constraint", "embedding_of_variable"):
                val = getattr(emb, attr, None)
                arr = _to_numpy(val, dtype=dtype if val is not None else None)
                if arr is None:
                    # create an empty dataset to preserve the field and mark it as None
                    ds = g.create_dataset(attr, shape=(0,), dtype=dtype)
                    ds.attrs['is_none'] = True
                else:
                    ds = g.create_dataset(attr, data=arr, compression=compression, compression_opts=compression_opts)
                    ds.attrs['is_none'] = False

def load_mip_embeddings_hdf5(input_h5_path: str, reconstruct_fn=None, as_namespace: bool = True):
    import types
    out = {}
    with h5py.File(input_h5_path, "r") as hf:
        for grp_name in hf:
            grp = hf[grp_name]
            loaded = {}
            for ds_name, ds in grp.items():
                if ds.attrs.get('is_none', False):
                    loaded[ds_name] = None
                else:
                    loaded[ds_name] = ds[()]
            # prefer stored original key; fallback to group name
            original_key = grp.attrs.get('orig_key', grp_name)
            if reconstruct_fn:
                out[original_key] = reconstruct_fn(original_key, loaded)
            elif as_namespace:
                out[original_key] = types.SimpleNamespace(**loaded)
            else:
                out[original_key] = loaded
    return out


def convert_hdf5_to_pickle(input_h5_path: str, output_pickle_path: str,
                           reconstruct_fn=None, as_namespace: bool = True) -> None:
    """
    Convert an HDF5 embeddings file to a pickle file.

    Parameters
    - `input_h5_path`: path to the HDF5 file (e.g. `mip_to_embeddings.hdf5`)
    - `output_pickle_path`: path to write the pickle (e.g. `mip_to_embeddings.pkl`)
    - `reconstruct_fn`: optional callable(key, loaded_dict) -> object to reconstruct original objects
    - `as_namespace`: if True and no reconstruct_fn, convert records to SimpleNamespace
    """
    import os
    import pickle

    # use the existing loader in this module
    data = load_mip_embeddings_hdf5(input_h5_path, reconstruct_fn=reconstruct_fn, as_namespace=as_namespace)

    os.makedirs(os.path.dirname(output_pickle_path) or ".", exist_ok=True)
    with open(output_pickle_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)