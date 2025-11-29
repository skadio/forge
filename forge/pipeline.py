import os
from typing import Union, List, Sequence, Dict, Any, Optional

import gurobipy as gp

from forge.embeddings import Forge
from forge.processor import MIPProcessor
from forge.utils import check_true, save_pickle, load_pickle
from forge.labeler import MIPLabeler


def finetune_integral_gap(forge: Forge,
                          input_mip_folder: str,
                          output_forge_finetuned_pkl: str,
                          output_mip_to_gapinfo_pkl: str,
                          input_mip_to_gapinfo_pkl: Optional[str] = None,
                          epochs: Optional[int] = None,
                          steps_per_instance: Optional[int] = None,
                          learning_rate: Optional[float] = None,
                          max_dgl_nodes: Optional[int] = None) -> None:
    """Fine-tune a pre-trained Forge model for integrality gap prediction from a folder of MIP files.

    Parameters
    ----------
    forge : Forge
        The Forge model to be fine-tuned.
    input_mip_folder : str
        Path to the folder containing MIP files for fine-tuning.
    output_forge_finetuned_pkl : str
        Path to save the fine-tuned Forge model as a pickle file.
    output_mip_to_gapinfo_pkl : str
        Path to save the mapping from MIP files to gap information as a pickle file.
    input_mip_to_gapinfo_pkl : Optional[str], default=None
        Path to an existing pickle file containing MIP to integral gap information.
        If not provided, it will be generated from the mip isntances in the `input_mip_folder`.
    epochs : Optional[int], default=None
        Number of epochs for fine-tuning.
    steps_per_instance : Optional[int], default=None
        Number of steps per instance for fine-tuning.
    learning_rate : Optional[float], default=None
        Learning rate for fine-tuning.
    max_dgl_nodes : Optional[int], default=None
        Maximum number of DGL nodes allowed.

    Returns
    -------
    None
        This function does not return anything.

    Raises
    ------
    ValueError
        If the provided input paths are invalid or required files/folders do not exist.
    """
    _validate_forge(forge, check_trained=True)

    # Use existing mip_to_integral_gap if given, or generate it
    if input_mip_to_gapinfo_pkl:
        check_true(os.path.isfile(input_mip_to_gapinfo_pkl),
                   ValueError(f"Error: {input_mip_to_gapinfo_pkl!r} does not exist."))
        mip_to_gapinfo = load_pickle(input_mip_to_gapinfo_pkl)
    else:
        check_true(input_mip_to_gapinfo_pkl is None and os.path.isdir(input_mip_folder),
                   ValueError("Error: Either `input_mip_to_integral_gap_pkl` must be provided, "
                              "or a valid `input_mip_folder` must be specified to generate it."))

        # Get MIP to integral gap ratio labels for fine-tuning
        labeler = MIPLabeler()
        mip_to_gapinfo = labeler.get_mip_to_integral_gap(input_mip_folder=input_mip_folder,
                                                         output_mip_to_gapinfo_pkl=output_mip_to_gapinfo_pkl,
                                                         time_limit=120,
                                                         has_return=True)

    # Fine-tune the Forge model
    forge._finetune_integral_gap(input_mip_to_gapinfo=mip_to_gapinfo,
                                 output_forge_finetuned_pkl=output_forge_finetuned_pkl,
                                 epochs=epochs,
                                 steps_per_instance=steps_per_instance,
                                 learning_rate=learning_rate,
                                 max_dgl_nodes=max_dgl_nodes)


def pretrain(forge: Forge,
             input_mip_folder: Optional[str],
             output_mip_to_mipinfo_pkl: str,
             output_forge_pretrained_pkl: str,
             output_log_file: str,
             input_mip_to_mipinfo_pkl: Optional[str] = None,
             relaxation_list: Optional[List[float]] = None,
             epochs: Optional[int] = None,
             steps_per_instance: Optional[int] = None,
             learning_rate: Optional[float] = None,
             max_dgl_nodes: Optional[int] = None) -> None:
    """Pre-train a Forge model.

    You can either:
        - provide a folder of MIP files via `input_mip_folder` to generate `output_mip_to_mipinfo_pkl`, or
        - provide an existing MIPInfo pickle via `input_mip_to_mipinfo_pkl` to skip conversion
         and load the prepared data directly.

    Parameters
    ----------
    forge : `Forge`
        Forge instance to train. Must be a `Forge` object.
    input_mip_folder : str or None
        Path to a directory containing MIP files to convert to MIPInfo.
        Provide `None` if using `input_mip_to_mipinfo_pkl`.
    output_mip_to_mipinfo_pkl : str
        Filepath where the generated mip_to_mipinfo mapping will be saved (pickle).
    output_forge_pretrained_pkl : str
        Filepath where the trained Forge object will be saved (pickle).
    output_log_file : str
        Filepath for storing pre_training logs.
    input_mip_to_mipinfo_pkl : str or None
        Optional path to an existing mip_to_mipinfo pickle to load instead of generating it.
    relaxation_list : List[float]
        Sequence of relaxation values to apply to MIP instance to generate relaxed instances.
    epochs : Optional[int], optional
        Number of training epochs. If `None`, a default value configured in `forge` will be used.
    steps_per_instance : Optional[int], optional
        Number of training steps to perform per instance per epoch. If `None`, a sensible default
        from `forge` or the training pipeline will be used.
    learning_rate : Optional[float], optional
        Learning rate for the optimizer. If `None`, the learning rate defined in `forge` will be used.
    max_dgl_nodes : Optional[int], optional
        Maximum number of graph nodes when converting MIP instances to DGL graphs. If `None`, no
        additional node cap is applied beyond defaults in the conversion utilities.

    Raises
    ------
    TypeError
        If `forge` is not a `Forge` instance.
    ValueError
        If neither `input_mip_folder` nor `input_mip_to_mipinfo_pkl` is provided, or if a provided
        path is invalid.
    """
    _validate_forge(forge)

    # MIP processor
    mip_processor = MIPProcessor(seed=forge.seed)

    # Use existing mip_to_integral_gap if given, or generate it
    if input_mip_to_mipinfo_pkl:
        check_true(os.path.isfile(input_mip_to_mipinfo_pkl),
                   ValueError(f"Error: `input_mip_to_mipinfo_pkl` {input_mip_to_mipinfo_pkl!r} does not exist."))
        pkl_to_load = input_mip_to_mipinfo_pkl
    else:
        check_true(input_mip_to_mipinfo_pkl is None and os.path.isdir(input_mip_folder),
                   ValueError("Error: Either `input_mip_to_mipinfo_pkl` must be provided, "
                              "or a valid `input_mip_folder` must be specified to generate it."))

        # Convert MIP files to MIPInfo objects and save to pickle
        mip_processor.convert_mip_to_mipinfo(input_mip_folder=input_mip_folder,
                                             output_mip_to_mipinfo_pkl=output_mip_to_mipinfo_pkl,
                                             relaxation_list=relaxation_list,
                                             has_return=False)
        pkl_to_load = output_mip_to_mipinfo_pkl

    # Load MIPInfo objects for training
    mipinfo_list = mip_processor.load_mipinfo_from_pickles([pkl_to_load])

    # Pre-train the Forge model
    forge._pretrain(input_mipinfo_list=mipinfo_list,
                    output_forge_pkl=output_forge_pretrained_pkl,
                    output_log_file=output_log_file,
                    epochs=epochs,
                    steps_per_instance=steps_per_instance,
                    learning_rate=learning_rate,
                    max_dgl_nodes=max_dgl_nodes)


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
    _validate_forge(forge, check_trained=True)

    # Normalize input: accept a folder path, a single MIP file path, a list of paths,
    # or a gurobipy Model instance (or list/mix of them).
    inputs = input_mips if isinstance(input_mips, (list, tuple)) else [input_mips]

    # MIP items can be a file, folder, or gurobipy model
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

    # Start Gurobi environment
    gurobi_env = MIPProcessor._start_gurobi_env()

    mip_to_embeddings = {}
    for mip_item in mip_items:

        # Read MIP file to a Gurobi model (or use the provided model)
        if isinstance(mip_item, gp.Model):
            mip_model = mip_item
            key = getattr(mip_model, "ModelName", "gurobi_model")
        else:
            mip_model = gp.read(mip_item, env=gurobi_env)
            key = mip_item

        # Convert MIP to vector representation
        mip_embeddings = forge._mip_model_to_embeddings(mip_model)
        mip_to_embeddings[key] = mip_embeddings

    # Close Gurobi environment
    gurobi_env.close()

    save_pickle(mip_to_embeddings, output_mip_to_embeddings_pkl)

    return mip_to_embeddings


def _validate_forge(forge, check_trained=False):
    check_true(isinstance(forge, Forge), TypeError("Error: Forge input should be a Forge instance."))

    if check_trained:
        check_true(forge.is_trained, ValueError("Error: Forge has not been trained."))

    # # Train/test data
    # check_true(data is not None, ValueError("Data input cannot be none."))
    # check_true(isinstance(data, (str, pd.DataFrame)),
    #            TypeError("Data should be string of filepath or data frame."))
    #
    # if save_file is not None:
    #     check_true(isinstance(save_file, (bool, str)),
    #                TypeError("Save file should be boolean or a string filepath."))
