import os
from typing import Dict, List, Tuple, Union
import pandas as pd
import numpy as np
from forge.embeddings import Forge
from forge.utils import Constants, save_pickle, load_pickle, check_true, check_false

def pretrain(forge: Forge, save_file) -> None:
    """
    Trains Forge

    Parameters
    ----------
    forge : Forge
        The forge models to be trained.
        The object is updated in-place.

    save_file: str

    Returns
    -------
    Returns nothing.
    """
    _validate_forge(forge)
    _validate_args()
    _validate_save(save_file)

    # Import data
    train_data_df = load_data(data=data)

    # Save file
    if save_file is not None:
        if isinstance(save_file, str):
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            save_pickle(forge, save_file)
        elif save_file:
            save_pickle(forge, "recommender.pkl")

def _validate_args():

    # Train/test data
    check_true(data is not None, ValueError("Data input cannot be none."))
    check_true(isinstance(data, (str, pd.DataFrame)),
               TypeError("Data should be string of filepath or data frame."))


def _validate_forge(forge, check_trained=False):
    check_true(isinstance(forge, Forge), TypeError("Forge input should be a Forge instance."))
    if check_trained:
        check_true(forge.is_trained, ValueError("Forge has not been trained."))


def _validate_save(save_file):
    if save_file is not None:
        check_true(isinstance(save_file, (bool, str)),
                   TypeError("Save file should be boolean or a string filepath."))


