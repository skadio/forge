import os
from typing import Union, List, Sequence, Dict, Any

import gurobipy as gp

from forge.embeddings import Forge
from forge.processor import MIPProcessor
from forge.utils import check_true, save_pickle


def pretrain(forge: Forge,
             input_mip_folder: str,
             relaxation_list: List[float],
             output_mip_to_mipinfo_pkl: str,
             output_forge_pkl: str,
             output_log_file: str) -> None:
    """Pre-train a Forge model from a folder of MIP files.

    Parameters
    ----------
    forge : Forge
        Forge instance to train. Must be a `Forge` object.
    input_mip_folder : str
        Path to a directory containing MIP files to convert to MIPInfo.
    relaxation_list : List[str]
        Sequence of relaxation to apply to MIP instance to generate relaxed instances.
    output_mip_to_mipinfo_pkl : str
        Filepath where the generated mip_to_mipinfo mapping will be saved (pickle).
    output_forge_pkl : str
        Filepath where the trained Forge object will be saved (pickle).
    output_log_file : str
        Filepath for storing pre_training logs.

    Raises
    ------
    TypeError
        If `forge` is not a `Forge` instance.
    ValueError
        If required inputs are missing or invalid during processing.
    """
    _validate_pretrain_args(forge)

    # MIP Processor
    mip_processor = MIPProcessor(seed=forge.seed)

    # Create MIP to MIPInfo dictionary
    mip_processor.convert_mip_to_mipinfo(input_mip_folder=input_mip_folder,
                                         output_mip_to_mipinfo_pkl=output_mip_to_mipinfo_pkl,
                                         relaxation_list=relaxation_list,
                                         has_return=False)

    # List of MIPInfo objects for training
    mipinfo_list = mip_processor.load_mipinfo_from_pickles([output_mip_to_mipinfo_pkl])

    # Pre-train the Forge model
    forge._pretrain(input_mipinfo_list=mipinfo_list,
                    output_forge_pkl=output_forge_pkl,
                    output_log_file=output_log_file)


def mip_to_embeddings(forge: Forge, input_mips: Union[str, gp.Model, Sequence[Union[str, gp.Model]]],
                      output_mip_to_embeddings_pkl: str) -> Dict[str, Any]:
    """
    Generate embeddings for one or more MIP inputs using a trained Forge instance.

    Parameters
    ----------
    forge : Forge
        A trained Forge instance.
    input_mips : str | gp.Model | Sequence[str | gp.Model]
        Path to a directory containing MIP files,
        Path to a single MIP file,
        A single gurobipy model instance,
        Or a list/tuple mixing these types.
    output_mip_to_embeddings_pkl : str
        Filepath where the resulting mapping from MIP identifiers to embeddings
        will be saved (pickle).

    Returns
    -------
    Dict[str, Any]
        Mapping from MIP identifier (file path or model name) to embeddings object.

    Raises
    ------
    ValueError
        If any element of `input_mips` is not a supported type.
    """
    _validate_pretrain_args(forge, check_trained=True)

    # Normalize input: accept a folder path, a single MIP file path, a list of paths,
    # or a gurobipy Model instance (or list/mix of them).
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
            raise ValueError(f"Error: Input {item!r} is neither a directory, a file, nor a gurobipy model instance.")

    # Setup Gurobi environment
    try:
        from gurobi_onboarder import init_gurobi
        gurobi_venv, GUROBI_FOUND = init_gurobi.initialize_gurobi()
    except:
        gurobi_venv = gp.Env(empty=True)
    gurobi_venv.setParam("OutputFlag", 0)
    gurobi_venv.start()

    mip_to_embeddings = {}
    for mip_item in mip_items:

        # Read MIP file to a Gurobi model (or use the provided model)
        if isinstance(mip_item, gp.Model):
            mip_model = mip_item
            key = getattr(mip_model, "ModelName", "gurobi_model")
        else:
            mip_model = gp.read(mip_item, env=gurobi_venv)
            key = mip_item

        # Convert MIP to vector representation
        mip_embeddings = forge._mip_model_to_embeddings(mip_model)
        mip_to_embeddings[key] = mip_embeddings

    save_pickle(mip_to_embeddings, output_mip_to_embeddings_pkl)

    return mip_to_embeddings


def _validate_pretrain_args(forge, check_trained=False):
    check_true(isinstance(forge, Forge), TypeError("Error: Forge input should be a Forge instance."))

    if check_trained:
        check_true(forge.is_trained, ValueError("Error: Forge has not been trained."))

    pass
    # # Train/test data
    # check_true(data is not None, ValueError("Data input cannot be none."))
    # check_true(isinstance(data, (str, pd.DataFrame)),
    #            TypeError("Data should be string of filepath or data frame."))
    #
    # if save_file is not None:
    #     check_true(isinstance(save_file, (bool, str)),
    #                TypeError("Save file should be boolean or a string filepath."))
