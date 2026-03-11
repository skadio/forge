import gc
import os
from typing import Union, List, Sequence, Dict, Optional, Iterator

import gurobipy as gp
from tqdm import tqdm
import torch
import numpy as np

from forge.embeddings import Forge
from forge.labeler import MIPLabeler, GapInfo, SATLabeler, SATSatisfiabilityInfo
from forge.processor import (MIPProcessor, _MIPUtils, MIPEmbeddings, MIPInfo,
                             SATProcessor, _SATUtils, SATEmbeddings, SATInfo)
from forge.utils import check_true, save_pickle, load_pickle


def finetune_integral_gap(forge: Forge,
                          input_forge_pkl: str,
                          model_type: str,
                          input_mip_folder: str,
                          input_mip_instances_file: Optional[str],
                          output_forge_finetuned_pkl: str,
                          output_mip_to_gapinfo_pkl: str,
                          input_mip_to_gapinfo_pkl: Optional[str] = None,
                          epochs: Optional[int] = None,
                          steps_per_instance: Optional[int] = None,
                          learning_rate: Optional[float] = None,
                          weight_decay: Optional[float] = None,
                          max_graph_nodes: Optional[int] = None,
                          gapinfo_time_limit: int = 120,
                          num_parallel_workers: int = 1) -> None:
    """Fine-tune a pre-trained Forge model for integrality gap prediction from a folder of MIP files.

    Parameters
    ----------
    forge : Forge
        The Forge model to be fine-tuned.
    input_forge_pkl : str
        Path to the pre-trained Forge model pickle file.
    model_type : str
        The type of the model to use (e.g., "fine-tune").
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
    weight_decay : Optional[float], default=None
        Weight decay for fine-tuning.
    max_graph_nodes : Optional[int], default=None
        Maximum number of graph nodes allowed.
    gapinfo_time_limit : int, default=120
        Time limit (in seconds) for computing gap information for each MIP instance.
    num_parallel_workers : int, default=5
        Number of parallel workers to use for processing MIP instances.

    Returns
    -------
    None
        This function does not return anything.

    Raises
    ------
    ValueError
        If the provided input paths are invalid or required files/folders do not exist.
    """

    # Load pre-trained Forge model ready for fine-tuning
    forge.load_model(input_forge_pkl=input_forge_pkl, model_type=model_type)

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
        mip_to_gapinfo = labeler.convert_mip_to_gapinfo(input_mip_folder=input_mip_folder,
                                                        input_mip_instances_file=input_mip_instances_file,
                                                        output_mip_to_gapinfo_pkl=output_mip_to_gapinfo_pkl,
                                                        gapinfo_time_limit=gapinfo_time_limit,
                                                        gurobi_num_threads=1,
                                                        num_parallel_workers=num_parallel_workers,
                                                        has_return=True)

    # Fine-tune the Forge model
    forge._finetune_integral_gap(input_mip_to_gapinfo=mip_to_gapinfo,
                                 output_forge_finetuned_pkl=output_forge_finetuned_pkl,
                                 epochs=epochs,
                                 steps_per_instance=steps_per_instance,
                                 learning_rate=learning_rate,
                                 weight_decay=weight_decay,
                                 max_graph_nodes=max_graph_nodes)


def finetune_sat_prediction(forge: Forge,
                           input_forge_pkl: str,
                           model_type: str,
                           input_sat_folder: str,
                           input_sat_instances_file: Optional[str],
                           output_forge_finetuned_pkl: str,
                           output_sat_to_satinfo_pkl: str,
                           input_sat_to_satinfo_pkl: Optional[str] = None,
                           epochs: Optional[int] = None,
                           steps_per_instance: Optional[int] = None,
                           learning_rate: Optional[float] = None,
                           weight_decay: Optional[float] = None,
                           max_graph_nodes: Optional[int] = None) -> None:
    """Fine-tune a pre-trained Forge model for SAT satisfiability prediction from a folder of SAT files.

    Parameters
    ----------
    forge : Forge
        The Forge model to be fine-tuned.
    input_forge_pkl : str
        Path to the pre-trained Forge model pickle file.
    model_type : str
        The type of the model to use (e.g., "fine-tune").
    input_sat_folder : str
        Path to the folder containing SAT files for fine-tuning (in LP/MPS format).
    input_sat_instances_file : Optional[str]
        Optional file containing list of SAT instances to use from input_sat_folder.
    output_forge_finetuned_pkl : str
        Path to save the fine-tuned Forge model as a pickle file.
    output_sat_to_satinfo_pkl : str
        Path to save the mapping from SAT files to satisfiability information as a pickle file.
    input_sat_to_satinfo_pkl : Optional[str], default=None
        Path to an existing pickle file containing SAT to satisfiability information.
        If not provided, it will be generated from the SAT instances in the `input_sat_folder`
        by extracting satisfiability labels from filenames (must contain "_sat" or "_unsat").
    epochs : Optional[int], default=None
        Number of epochs for fine-tuning.
    steps_per_instance : Optional[int], default=None
        Number of steps per instance for fine-tuning.
    learning_rate : Optional[float], default=None
        Learning rate for fine-tuning.
    weight_decay : Optional[float], default=None
        Weight decay for fine-tuning.
    max_graph_nodes : Optional[int], default=None
        Maximum number of graph nodes allowed.

    Returns
    -------
    None
        This function does not return anything.

    Raises
    ------
    ValueError
        If the provided input paths are invalid or required files/folders do not exist.
    """

    # Load pre-trained Forge model ready for fine-tuning
    forge.load_model(input_forge_pkl=input_forge_pkl, model_type=model_type)

    _validate_forge(forge, check_trained=True)

    # Use existing sat_to_satinfo if given, or generate it
    if input_sat_to_satinfo_pkl:
        check_true(os.path.isfile(input_sat_to_satinfo_pkl),
                   ValueError(f"Error: {input_sat_to_satinfo_pkl!r} does not exist."))
        sat_to_satinfo = load_pickle(input_sat_to_satinfo_pkl)
    else:
        check_true(input_sat_to_satinfo_pkl is None and os.path.isdir(input_sat_folder),
                   ValueError("Error: Either `input_sat_to_satinfo_pkl` must be provided, "
                              "or a valid `input_sat_folder` must be specified to generate it."))

        # Get SAT satisfiability labels from filenames (extract from "_sat" or "_unsat" in filename)
        labeler = SATLabeler()
        sat_to_satinfo = labeler.convert_sat_to_satisfiability_info(input_sat_folder=input_sat_folder,
                                                                    input_sat_instances_file=input_sat_instances_file,
                                                                    output_sat_to_satinfo_pkl=output_sat_to_satinfo_pkl,
                                                                    has_return=True)

    # Fine-tune the Forge model
    forge._finetune_sat_prediction(input_sat_to_satinfo=sat_to_satinfo,
                                   output_forge_finetuned_pkl=output_forge_finetuned_pkl,
                                   epochs=epochs,
                                   steps_per_instance=steps_per_instance,
                                   learning_rate=learning_rate,
                                   weight_decay=weight_decay,
                                   max_graph_nodes=max_graph_nodes)


def pretrain(forge: Forge,
             input_mip_folder: Optional[str],
             input_mip_instances_file: Optional[str],
             output_mip_to_mipinfo_pkl: str,
             output_forge_pretrained_pkl: str,
             output_log_file: str,
             input_mip_to_mipinfo_pkl: Optional[str] = None,
             relaxation_list: Optional[List[float]] = None,
             epochs: Optional[int] = None,
             steps_per_instance: Optional[int] = None,
             learning_rate: Optional[float] = None,
             weight_decay: Optional[float] = None,
             max_graph_nodes: Optional[int] = None) -> None:
    """Pre-train a Forge model.

    You can provide either:
        1) a folder of MIP files via `input_mip_folder` and `input_mip_instances_file` to consider
        2) Existing MIPInfo pickle via `input_mip_to_mipinfo_pkl` to skip conversion and
        load the prepared data directly.
        If input_mip_to_mipinfo_pkl is given, skips input_mip_folder and input_mip_instances_file.

    Parameters
    ----------
    forge : `Forge`
        Forge instance to train. Must be a `Forge` object.
    input_mip_folder : str or None
        Path to a directory containing MIP files to convert to MIPInfo.
        Provide `None` if using `input_mip_to_mipinfo_pkl`.
    input_mip_instances_file : str
        If provided, only include instances from input_mip_folder listed in the file.
    output_mip_to_mipinfo_pkl : str
        Filepath where the generated mip_to_mipinfo mapping will be saved (pickle).
    output_forge_pretrained_pkl : str
        Filepath where the trained Forge object will be saved (pickle).
    output_log_file : str
        Filepath for storing pre_training logs.
    input_mip_to_mipinfo_pkl : str or None
        Optional path to an existing mip_to_mipinfo pickle to load instead of generating it.
        If given, skips input_mip_folder and input_mip_instances_file.
    relaxation_list : List[float]
        Sequence of relaxation values to apply to MIP instance to generate relaxed instances.
    epochs : Optional[int], optional
        Number of training epochs. If `None`, a default value configured in `forge` will be used.
    steps_per_instance : Optional[int], optional
        Number of training steps to perform per instance per epoch. If `None`, a sensible default
        from `forge` or the training pipeline will be used.
    learning_rate : Optional[float], optional
        Learning rate for the optimizer. If `None`, the learning rate defined in `forge` will be used.
    weight_decay : Optional[float], optional
        Weight decay for the optimizer. If `None`, the weight decay defined in `forge`
    max_graph_nodes : Optional[int], optional
        Maximum number of graph nodes when converting MIP instances to bipartite graph.
        If `None`, no additional node cap is applied beyond defaults in the conversion utilities.

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
                              "or a valid `input_mip_folder` must be specified to generate it." +
                              f" input_mip_folder={input_mip_folder!r}."))

        # Convert MIP files to MIPInfo objects and save to pickle
        mip_processor.convert_mip_to_mipinfo(input_mip_folder=input_mip_folder,
                                             input_mip_instances_file=input_mip_instances_file,
                                             output_mip_to_mipinfo_pkl=output_mip_to_mipinfo_pkl,
                                             relaxation_list=relaxation_list, has_return=False)
        pkl_to_load = output_mip_to_mipinfo_pkl

    # Load MIPInfo objects for training
    mipinfo_list = _MIPUtils.load_mipinfo_from_pickles([pkl_to_load])

    # Pre-train the Forge model
    forge._pretrain(input_mipinfo_list=mipinfo_list,
                    output_forge_pkl=output_forge_pretrained_pkl,
                    output_log_file=output_log_file,
                    epochs=epochs,
                    steps_per_instance=steps_per_instance,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    max_graph_nodes=max_graph_nodes)


def mip_to_embeddings(forge: Forge,
                      input_forge_pkl: str,
                      model_type: str,
                      input_mips: Union[str, gp.Model, Sequence[Union[str, gp.Model]]],
                      input_mip_instances_file: Optional[str],
                      output_mip_to_embeddings_pkl: str,
                      instance_embedding_only: bool) -> Dict[str, MIPEmbeddings]:
    """
    Generate embeddings for one or more MIP inputs using a trained Forge instance.

    Parameters
    ----------
    forge : Forge
        A trained Forge instance.
    input_forge_pkl : str
        Path to the input Forge pickle file.
    model_type : str
        The type of the model to use (e.g., "fine-tune").
    input_mips : str | gp.Model | Sequence[str | gp.Model]
        Path to a directory containing MIP files,
        Path to a single MIP file,
        A single gurobipy model instance,
        Or a list/tuple mixing these types.
    input_mip_instances_file: Optional[str]
        The file containing the list of MIP instances to process from input_mips.
    output_mip_to_embeddings_pkl : str
        Filepath where the resulting mapping from MIP identifiers to embeddings
        will be saved (pickle).
    instance_embedding_only: bool
        If true, only generate and save the instance-level embedding. Takes less space/memory.
        Skip variable and constraint-level embeddings. Requires considerable memory to store all.

    Returns
    -------
    Dict[str, MIPEmbeddings]
        Mapping from MIP identifier (file path or model name) to embeddings object.

    Raises
    ------
    ValueError
        If any element of `input_mips` is not a supported type.
    """

    # Load pre-trained Forge model
    forge.load_model(input_forge_pkl=input_forge_pkl, model_type=model_type)
    _validate_forge(forge, check_trained=True)

    # Normalize input: accept a folder path, a single MIP file path, a list of paths,
    mip_items = _MIPUtils.get_mip_items(input_mips, input_mip_instances_file)

    # Start Gurobi environment
    gurobi_env = _MIPUtils.start_gurobi_env()

    def _move_to_cpu_and_detach(obj):
        """Recursively move torch.Tensors to CPU and detach; preserve containers."""
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        if isinstance(obj, (list, tuple)):
            converted = [_move_to_cpu_and_detach(x) for x in obj]
            return type(obj)(converted)
        if isinstance(obj, dict):
            return {k: _move_to_cpu_and_detach(v) for k, v in obj.items()}
        return obj

    # For each MIP item, create MIP model, and generate embedding
    mip_to_embeddings = {}
    for idx, mip_item in enumerate(tqdm(mip_items)):
        print("\n", mip_item)

        # Read MIP file to a Gurobi model (or use the provided model)
        if isinstance(mip_item, gp.Model):
            mip_model = mip_item
            key = getattr(mip_model, "ModelName", "gurobi_model")
        else:
            mip_model = gp.read(mip_item, env=gurobi_env)
            key = mip_item

        # Inference without building grads
        # Convert MIP to vector representation
        mip_embeddings = forge._mip_model_to_embeddings(mip_model, instance_embedding_only)

        # Move all tensors in the returned embeddings to CPU and detach
        for name, val in vars(mip_embeddings).items():
            try:
                setattr(mip_embeddings, name, _move_to_cpu_and_detach(val))
            except Exception:
                # If attribute can't be processed, leave it (safe fallback)
                pass

        mip_to_embeddings[key] = mip_embeddings

        # Cleanup large refs
        del mip_embeddings
        if not isinstance(mip_item, gp.Model): # don't delete a user-provided model
            del mip_model

         # Periodic cleanup to avoid fragmentation
        if idx % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Close Gurobi environment
    gurobi_env.close()

    save_pickle(mip_to_embeddings, output_mip_to_embeddings_pkl)
    # save_mip_embeddings_hdf5(mip_to_embeddings, output_mip_to_embeddings_pkl, dtype=np.float16)

    return mip_to_embeddings


def mip_to_gapinfo(forge: Forge,
                   input_forge_pkl: str,
                   model_type: str,
                   input_mips: Union[str, gp.Model, Sequence[Union[str, gp.Model]]],
                   input_mip_instances_file: Optional[str],
                   output_mip_to_gapinfo_pkl: str,
                   problem_type: str) -> Dict[str, GapInfo]:
    """
    Generate gap information for one or more MIP inputs using a trained Forge instance.

    Parameters
    ----------
    forge : Forge
        A Forge instance.
    input_forge_pkl : str
        Path to the input Forge pickle file.
    model_type : str
        The type of the model to use (e.g., "fine-tune").
    input_mips : str | gp.Model | Sequence[str | gp.Model]
        Path to a directory containing MIP files,
        Path to a single MIP file,
        A single gurobipy model instance,
        Or a list/tuple mixing these types.
    input_mip_instances_file : Optional[str]
        If provided, only include instances from input_mips listed in the file.
    output_mip_to_gapinfo_pkl : str
        Filepath where the resulting mapping from MIP identifiers to gap information will be saved (pickle).
    num_parallel_workers : int
        The number of parallel workers to use for processing MIP instances.
    problem_type : str
        The type of problem for which gap information is to be computed.

    Returns
    -------
    Dict[str, GapInfo]
        Mapping from MIP identifier (file path or model name) to gap information object.
        GapInfo.ratio is the predicted ratio without solving the MIP
        GapInfo.mip_sol is None with no solution computed
        GapInfo.mip_obj is the predicted objective without solving the MIP
        GapInfo.lp_obj is the true lp objective
    Raises
    ------
    ValueError
        If any element of `input_mips` is not a supported type.
    """

    # Load pre-trained Forge model
    forge.load_model(input_forge_pkl=input_forge_pkl, model_type=model_type)

    _validate_forge(forge, check_trained=True)

    # Normalize input: accept a folder path, a single MIP file path, a list of paths,
    mip_items = MIPProcessor.get_mip_items(input_mips, input_mip_instances_file)

    # Start Gurobi environment
    gurobi_env = _MIPUtils.start_gurobi_env()

    # For each MIP item, create MIP model, and generate embedding
    mip_to_gap_info = {}
    for mip_item in mip_items:

        print("<< Start: Create GapInfo", mip_item)

        # Read MIP file to a Gurobi model (or use the provided model)
        if isinstance(mip_item, gp.Model):
            mip_model = mip_item
            # Using id() in case multiple unnamed models are provided
            key = getattr(mip_model, "ModelName", f"gurobi_{id(mip_model)}")
        else:
            mip_model = gp.read(mip_item, env=gurobi_env)
            key = mip_item

        # Convert MIP to vector representation
        gap_info = forge._mip_model_to_gapinfo(mip_model, problem_type)
        mip_to_gap_info[key] = gap_info

        print("<< Finish: Create GapInfo", mip_item)

    # Close Gurobi environment
    gurobi_env.close()

    save_pickle(mip_to_gap_info, output_mip_to_gapinfo_pkl)

    return mip_to_gap_info


def mip_to_mipinfo(forge: Forge,
                   input_mip_folder: Optional[str],
                   input_mip_instances_file: Optional[str],
                   output_mip_to_mipinfo_pkl: str,
                   relaxation_list: Optional[List[float]] = None,
                   num_parallel_workers: int = 1,
                   has_return: bool = True) -> Dict[str, MIPInfo]:
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
    input_mip_instances_file : str
        If provided, only include instances from input_mip_folder listed in the file.
    output_mip_to_mipinfo_pkl : str
        Filepath where the generated mip_to_mipinfo mapping will be saved (pickle).
    relaxation_list : List[float]
        Sequence of relaxation values to apply to MIP instance to generate relaxed instances.
    num_parallel_workers: int
        The number of parallel worker processes to use for conversion.
    has_return: bool
        Whether to return the generated mip_to_mipinfo mapping.

    Raises
    ------
    TypeError
        If `forge` is not a `Forge` instance.
    ValueError
        If `input_mip_folder` is not provided or invalid.

    Returns
    -------
    Dict[str, MIPInfo]
        Mapping from MIP identifier (file path or model name) to mip information object.
        MIPInfo object fields:
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
    _validate_forge(forge)

    # MIP processor
    mip_processor = MIPProcessor(seed=forge.seed)

    check_true(os.path.isdir(input_mip_folder),
               ValueError("Error: invalid `input_mip_folder` input_mip_folder={input_mip_folder!r}."))

    # Convert MIP files to MIPInfo objects and save to pickle
    mip_to_mipinfo = mip_processor.convert_mip_to_mipinfo(input_mip_folder=input_mip_folder,
                                                          input_mip_instances_file=input_mip_instances_file,
                                                          output_mip_to_mipinfo_pkl=output_mip_to_mipinfo_pkl,
                                                          relaxation_list=relaxation_list,
                                                          is_save_relaxed=True,
                                                          num_parallel_workers=num_parallel_workers,
                                                          has_return=has_return)
    return mip_to_mipinfo


def sat_pretrain(forge: Forge,
                 input_sat_folder: Optional[str],
                 input_sat_instances_file: Optional[str],
                 output_sat_to_satinfo_pkl: str,
                 output_forge_pretrained_pkl: str,
                 output_log_file: str,
                 input_sat_to_satinfo_pkl: Optional[str] = None,
                 input_mip_forge_pkl: Optional[str] = None,
                 epochs: Optional[int] = None,
                 steps_per_instance: Optional[int] = None,
                 learning_rate: Optional[float] = None,
                 weight_decay: Optional[float] = None,
                 max_graph_nodes: Optional[int] = None,
                 gradient_accumulation_steps: int = 1,
                 rank: int = 0,
                 world_size: int = 1,
                 gpu_memory_fraction: float = 0.8) -> None:
    """Pre-train a Forge model on SAT instances.

    You can provide either:
        1) a folder of SAT files via `input_sat_folder` and `input_sat_instances_file` to consider
        2) Existing SATInfo pickle via `input_sat_to_satinfo_pkl` to skip conversion and
        load the prepared data directly.
        If input_sat_to_satinfo_pkl is given, skips input_sat_folder and input_sat_instances_file.

    Parameters
    ----------
    forge : `Forge`
        Forge instance to train. Must be a `Forge` object.
    input_sat_folder : str or None
        Path to a directory containing SAT files (LP/MPS format) to convert to SATInfo.
        Provide `None` if using `input_sat_to_satinfo_pkl`.
    input_sat_instances_file : str
        If provided, only include instances from input_sat_folder listed in the file.
    output_sat_to_satinfo_pkl : str
        Filepath where the generated sat_to_satinfo mapping will be saved (pickle).
    output_forge_pretrained_pkl : str
        Filepath where the trained Forge object will be saved (pickle).
    output_log_file : str
        Filepath for storing pre_training logs.
    input_sat_to_satinfo_pkl : str or None
        Optional path to an existing sat_to_satinfo pickle to load instead of generating it.
        If given, skips input_sat_folder and input_sat_instances_file.
    input_mip_forge_pkl : str or None
        Optional path to a pre-trained Forge-MIP model pickle to use as initial weights.
        If provided, the weights from the pre-trained MIP model are loaded before training on SAT instances.
        This allows transfer learning from MIP pre-training to SAT pre-training.
    epochs : Optional[int], optional
        Number of training epochs. If `None`, a default value configured in `forge` will be used.
    steps_per_instance : Optional[int], optional
        Number of training steps to perform per instance per epoch. If `None`, a sensible default
        from `forge` or the training pipeline will be used.
    learning_rate : Optional[float], optional
        Learning rate for the optimizer. If `None`, the learning rate defined in `forge` will be used.
    weight_decay : Optional[float], optional
        Weight decay for the optimizer. If `None`, the weight decay defined in `forge`
    max_graph_nodes : Optional[int], optional
        Maximum number of graph nodes when converting SAT instances to bipartite graph.
        If `None`, no additional node cap is applied beyond defaults in the conversion utilities.
    gradient_accumulation_steps : int, default=1
        Number of steps to accumulate gradients before optimizer step.
    rank : int, default=0
        Rank of current process in distributed training (0 for single GPU).
    world_size : int, default=1
        Total number of processes in distributed training (1 for single GPU).
        Dataset will be partitioned so each rank processes a different subset.
    gpu_memory_fraction : float, default=0.8
        Target GPU memory usage fraction (0.0 to 1.0). Smart fallback to CPU if exceeded.

    Raises
    ------
    TypeError
        If `forge` is not a `Forge` instance.
    ValueError
        If neither `input_sat_folder` nor `input_sat_to_satinfo_pkl` is provided, or if a provided
        path is invalid.
    """
    _validate_forge(forge)

    # Extract underlying module if wrapped with DataParallel or DistributedDataParallel
    import torch.nn as nn
    if isinstance(forge, nn.DataParallel):
        forge_module = forge.module
    elif isinstance(forge, nn.parallel.DistributedDataParallel):
        forge_module = forge.module
    else:
        forge_module = forge

    # SAT processor
    sat_processor = SATProcessor(seed=forge_module.seed)

    # Use existing sat_to_satinfo if given, or generate it
    if input_sat_to_satinfo_pkl:
        check_true(os.path.isfile(input_sat_to_satinfo_pkl),
                   ValueError(f"Error: `input_sat_to_satinfo_pkl` {input_sat_to_satinfo_pkl!r} does not exist."))
        pkl_to_load = input_sat_to_satinfo_pkl
    else:
        check_true(input_sat_to_satinfo_pkl is None and os.path.isdir(input_sat_folder),
                   ValueError("Error: Either `input_sat_to_satinfo_pkl` must be provided, "
                              "or a valid `input_sat_folder` must be specified to generate it." +
                              f" input_sat_folder={input_sat_folder!r}."))

        # Convert SAT files to SATInfo objects and save to pickle
        sat_processor.convert_sat_lp_to_satinfo(input_sat_folder=input_sat_folder,
                                               input_sat_instances_file=input_sat_instances_file,
                                               output_sat_to_satinfo_pkl=output_sat_to_satinfo_pkl,
                                               num_parallel_workers=1,
                                               has_return=False,
                                               max_graph_nodes=max_graph_nodes)
        pkl_to_load = output_sat_to_satinfo_pkl

    # Load SATInfo objects for training
    satinfo_list = _SATUtils.load_satinfo_from_pickles([pkl_to_load])
    
    # CRITICAL FOR TORCHRUN: Partition instances by rank to avoid all processes loading all data
    # Without this, each process loads the full dataset → memory multiplies by num_processes
    if world_size > 1:
        original_count = len(satinfo_list)
        
        # Shuffle first to ensure each rank gets a representative sample (not just indices i, i+world_size, i+2*world_size...)
        # This is critical for fair gradient averaging: if unshuffled, each rank specializes on a subset of instance types
        import random
        random.seed(42)  # Fixed seed for reproducibility across all ranks
        random.shuffle(satinfo_list)
        
        # Now do round-robin partitioning on shuffled data
        # Rank 0 gets indices 0, world_size, 2*world_size, ...
        partitioned_list = [satinfo_list[i] for i in range(rank, len(satinfo_list), world_size)]
        
        # AGGRESSIVE MEMORY CLEANUP: Delete full list immediately, keep only partition
        del satinfo_list
        satinfo_list = partitioned_list
        del partitioned_list
        
        if rank == 0:
            print(f"\nShuffled then partitioned {original_count} instances across {world_size} ranks")
            print(f"(Data shuffled with seed=42 to ensure representative distribution per rank)")
            print(f"Rank 0 will process: {len(satinfo_list)} instances")
            print(f"(Each rank processes ~{original_count // world_size} instances)\n")
    
    # Very aggressive garbage collection
    import gc as gc_module
    gc_module.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Load pre-trained MIP weights if provided
    if input_mip_forge_pkl:
        check_true(os.path.isfile(input_mip_forge_pkl),
                   ValueError(f"Error: `input_mip_forge_pkl` {input_mip_forge_pkl!r} does not exist."))
        print(f"\nLoading pre-trained MIP weights from: {input_mip_forge_pkl}")
        forge_module.load_weights_from_pretrained(input_mip_forge_pkl)
        print(f"Pre-trained MIP weights loaded successfully!\n")

    # Pre-train the Forge model (use underlying module if wrapped with DataParallel or DistributedDataParallel)
    # Since we already partitioned instances by rank above, pass world_size=1 to avoid double partitioning
    forge_module = forge.module if isinstance(forge, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else forge
    forge_module._pretrain(input_mipinfo_list=satinfo_list,
                           output_forge_pkl=output_forge_pretrained_pkl,
                           output_log_file=output_log_file,
                           epochs=epochs,
                           steps_per_instance=steps_per_instance,
                           learning_rate=learning_rate,
                           weight_decay=weight_decay,
                           max_graph_nodes=max_graph_nodes,
                           rank=0,  # After partition, each process is rank 0 for its subset
                           world_size=1,  # After partition, each process has its own data
                           gpu_memory_fraction=gpu_memory_fraction)


def _validate_forge(forge, check_trained=False):
    # Validate Forge instance (handle DataParallel/DistributedDataParallel wrapper)
    import torch.nn as nn
    
    # If wrapped with DataParallel or DistributedDataParallel, extract the underlying module
    if isinstance(forge, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        forge_module = forge.module
    else:
        forge_module = forge
    
    check_true(isinstance(forge_module, Forge), TypeError("Error: Forge input should be a Forge instance."))

    if check_trained:
        check_true(forge_module.is_trained, ValueError("Error: Forge has not been trained."))


def sat_to_satinfo(forge: Forge,
                   input_sat_folder: Optional[str],
                   input_sat_instances_file: Optional[str],
                   output_sat_to_satinfo_pkl: str,
                   num_parallel_workers: int = 1,
                   has_return: bool = True,
                   max_graph_nodes: Optional[int] = 100000) -> Dict[str, SATInfo]:
    """Convert SAT LP/MPS files to SATInfo objects.

    SAT instances should be provided as LP or MPS files (typically converted from DIMACS CNF
    format via sat_to_mip.py).

    Parameters
    ----------
    forge : `Forge`
        Forge instance. Must be a `Forge` object.
    input_sat_folder : str or None
        Path to a directory containing SAT files (LP/MPS format) to convert to SATInfo.
    input_sat_instances_file : str
        If provided, only include instances from input_sat_folder listed in the file.
    output_sat_to_satinfo_pkl : str
        Filepath where the generated sat_to_satinfo mapping will be saved (pickle).
    num_parallel_workers: int
        The number of parallel worker processes to use for conversion.
    has_return: bool
        Whether to return the generated sat_to_satinfo mapping.
    max_graph_nodes : Optional[int], default: 100000
        Maximum number of nodes (clauses + variables) in the SAT graph. Instances exceeding
        this limit are skipped since they would be filtered out during pretraining anyway.

    Raises
    ------
    TypeError
        If `forge` is not a `Forge` instance.
    ValueError
        If `input_sat_folder` is not provided or invalid.

    Returns
    -------
    Dict[str, SATInfo]
        Mapping from SAT identifier (file path) to SAT information object.
        SATInfo object fields:
            instance_name : Optional[str]
                Path or unique identifier of the SAT instance.
            feature_tensor : Optional[torch.Tensor]
                Node feature matrix of shape `(num_clauses + num_vars, feat_dim=10)` with clauses stacked first.
            num_clauses : Optional[int]
                Number of clauses in the original SAT formula.
            num_vars : Optional[int]
                Number of variables in the original SAT formula.
            edge_index : Optional[torch.LongTensor]
                PyG-style edge index tensor of shape (2, num_edges) representing graph connectivity.
            edge_weight : Optional[torch.FloatTensor]
                Edge weights tensor of shape (num_edges,) corresponding to edges in `edge_index`.
    """
    _validate_forge(forge)

    # Extract underlying module if wrapped with DataParallel
    import torch.nn as nn
    forge_module = forge.module if isinstance(forge, nn.DataParallel) else forge

    # SAT processor
    sat_processor = SATProcessor(seed=forge_module.seed)

    check_true(os.path.isdir(input_sat_folder),
               ValueError("Error: invalid `input_sat_folder` input_sat_folder={input_sat_folder!r}."))

    # Convert SAT files to SATInfo objects and save to pickle
    sat_to_satinfo = sat_processor.convert_sat_lp_to_satinfo(input_sat_folder=input_sat_folder,
                                                             input_sat_instances_file=input_sat_instances_file,
                                                             output_sat_to_satinfo_pkl=output_sat_to_satinfo_pkl,
                                                             num_parallel_workers=num_parallel_workers,
                                                             has_return=has_return,
                                                             max_graph_nodes=max_graph_nodes)
    return sat_to_satinfo


def sat_to_embeddings(forge: Forge,
                      input_forge_pkl: str,
                      model_type: str,
                      input_sat_folder: str,
                      input_sat_instances_file: Optional[str],
                      output_sat_to_embeddings_pkl: str,
                      instance_embedding_only: bool,
                      max_graph_nodes: Optional[int] = 100000) -> Dict[str, SATEmbeddings]:
    """
    Generate embeddings for SAT instances using a trained Forge model.

    Parameters
    ----------
    forge : Forge
        A trained Forge instance.
    input_forge_pkl : str
        Path to the input Forge pickle file.
    model_type : str
        The type of the model to use (e.g., "fine-tune").
    input_sat_folder : str
        Path to a directory containing SAT LP/MPS files.
    input_sat_instances_file : Optional[str]
        The file containing the list of SAT instances to process from input_sat_folder.
    output_sat_to_embeddings_pkl : str
        Filepath where the resulting mapping from SAT identifiers to embeddings
        will be saved (pickle).
    instance_embedding_only: bool
        If true, only generate and save the instance-level embedding.
        Skip clause and variable-level embeddings to save memory.
    max_graph_nodes : Optional[int], default: 100000
        Maximum number of nodes (clauses + variables) in the SAT graph. Instances exceeding
        this limit are skipped.

    Returns
    -------
    Dict[str, SATEmbeddings]
        Mapping from SAT identifier (file path) to embeddings object.

    Raises
    ------
    ValueError
        If input paths are invalid.
    """

    # Load pre-trained Forge model
    forge.load_model(input_forge_pkl=input_forge_pkl, model_type=model_type)
    _validate_forge(forge, check_trained=True)

    # Extract underlying module if wrapped with DataParallel
    import torch.nn as nn
    forge_module = forge.module if isinstance(forge, nn.DataParallel) else forge

    # SAT processor
    sat_processor = SATProcessor(seed=forge_module.seed)

    # Get list of SAT files
    sat_files = _SATUtils.get_only_sat_files(input_sat_folder, input_sat_instances_file, is_sort_by_size=False)

    def _move_to_cpu_and_detach(obj):
        """Recursively move torch.Tensors to CPU and detach; preserve containers."""
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        if isinstance(obj, (list, tuple)):
            converted = [_move_to_cpu_and_detach(x) for x in obj]
            return type(obj)(converted)
        if isinstance(obj, dict):
            return {k: _move_to_cpu_and_detach(v) for k, v in obj.items()}
        return obj

    # For each SAT file, generate embedding
    sat_to_embeddings = {}
    for idx, sat_file in enumerate(tqdm(sat_files)):
        print("\n", sat_file)

        try:
            # Read SAT file to a Gurobi model
            gurobi_env = _SATUtils.start_gurobi_env()
            sat_model = gp.read(sat_file, env=gurobi_env)
            gurobi_env.close()
            
            # Check if instance size exceeds max_graph_nodes
            num_clauses = sat_model.getAttr("NumConstrs")
            num_vars = sat_model.getAttr("NumVars")
            num_nodes = num_clauses + num_vars
            if max_graph_nodes and num_nodes > max_graph_nodes:
                print(f"Skipping {sat_file}: num_nodes={num_nodes} exceeds max_graph_nodes={max_graph_nodes}")
                continue

            # Inference without building grads
            # Convert SAT to vector representation
            sat_embeddings = forge._mip_model_to_embeddings(sat_model, instance_embedding_only)

            # Move all tensors in the returned embeddings to CPU and detach
            for name, val in vars(sat_embeddings).items():
                try:
                    setattr(sat_embeddings, name, _move_to_cpu_and_detach(val))
                except Exception:
                    # If attribute can't be processed, leave it (safe fallback)
                    pass

            sat_to_embeddings[sat_file] = sat_embeddings

            # Cleanup large refs
            del sat_embeddings
            del sat_model

        except Exception as e:
            print(f"Error processing {sat_file}: {e}")
            continue

        # Periodic cleanup to avoid fragmentation
        if idx % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    save_pickle(sat_to_embeddings, output_sat_to_embeddings_pkl)

    return sat_to_embeddings
