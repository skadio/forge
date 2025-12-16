import os
import pickle
from typing import Union, NamedTuple

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
    _TESTING_DIR_NAME = "all_mip_data"  # All MIP data now lives in the same folder
    _TRAINING_DIR_NAME = "all_mip_data" # All MIP data now lives in the same folder
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
    _DATA_SPLIT_MASK = "data_splits_to_mip_instances_unittest.txt" # This file contains the data split masks for various experiments
    
    # Folders
    _CONST_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = _CONST_FILE_DIR + os.sep + ".." + os.sep + _DATA_DIR_NAME
    DATA_TESTING_DIR = DATA_DIR + os.sep + _TESTING_DIR_NAME
    DATA_TRAINING_DIR = DATA_DIR + os.sep + _TRAINING_DIR_NAME
    DATA_TESTS_DIR = _CONST_FILE_DIR + os.sep + ".." + os.sep + _TESTS_DIR_NAME + os.sep + _DATA_DIR_NAME
    MODELS_DIR = _CONST_FILE_DIR + os.sep + ".." + os.sep + _MODELS_DIR_NAME
    CONFIGS_DIR = _CONST_FILE_DIR + os.sep + ".." + os.sep + _FORGE_DIR_NAME + os.sep + _CONFIGS_DIR_NAME

    # File paths
    default_train_config_yaml = _CONST_FILE_DIR + os.sep + _CONFIGS_DIR_NAME + os.sep + _TRAIN_CONFIG_NAME
    default_mip_to_embeddings_pkl = _CONST_FILE_DIR + os.sep + ".." + os.sep + _TESTS_DIR_NAME + os.sep + _MIPINFO_NAME
    default_mip_to_mipinfo_pkl = _CONST_FILE_DIR + os.sep + ".." + os.sep + _TESTS_DIR_NAME + os.sep + _MIPINFO_NAME
    default_mip_to_gapinfo_pkl = _CONST_FILE_DIR + os.sep + ".." + os.sep + _TESTS_DIR_NAME + os.sep + _GAPINFO_NAME
    default_forge_pretrained_pkl = _CONST_FILE_DIR + os.sep + ".." + os.sep + _TESTS_DIR_NAME + os.sep + _FORGE_PKL_NAME
    default_forge_log_file = _CONST_FILE_DIR + os.sep + ".." + os.sep + _TESTS_DIR_NAME + os.sep + _FORGE_LOG_NAME


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

def write_data_splits_to_textfile(data_splits, out_file):

    """Write a dictionary of data splits to a plain text file.
    The file is written in a simple line-based format:
    - Each split name is written as a line starting with "SPLIT:".
    - Each instance name belonging to that split is written as a line
      starting with "INSTANCE:".

    Parameters
    ----------
    data_splits : dict[str, list[str]]
        Mapping from split name (e.g. 'pretrain', 'test_integral_gap')
        to a list of instance file names.
    out_file : str
        Path to the output text file that will be created/overwritten.
    """



    line = ""
    for split in data_splits:
        line += f"SPLIT:{split}\n"
        for instance in data_splits[split]:
            line += f"INSTANCE:{instance}\n"

    # Write data_splits into a text file 
    with open(out_file, 'w') as f:
        f.write(line)

def read_data_splits_from_textfile(in_file):

    """Read data splits from a text file created by write_data_splits_to_textfile.
    The function parses lines starting with "SPLIT:" and "INSTANCE:" and
    reconstructs the original data_splits dictionary.
    Parameters
    ----------
    in_file : str
        Path to the input text file previously written by
        write_data_splits_to_textfile.
    Returns
    -------
    dict[str, list[str]]
        Reconstructed mapping from split name to list of instance names.
    """

    data_splits = {}
    current_split = None

    with open(in_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("SPLIT:"):
                current_split = line.split("SPLIT:")[1]
                data_splits[current_split] = []
            elif line.startswith("INSTANCE:") and current_split is not None:
                instance = line.split("INSTANCE:")[1]
                data_splits[current_split].append(instance)

    return data_splits