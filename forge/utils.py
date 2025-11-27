# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

"""
:Author:

This module provides a number of constants and helper functions.
"""
import pickle

from typing import Dict, Union, NamedTuple, NewType, NoReturn

Num = Union[int, float]
"""Num type is defined as integer or float."""


class Constants(NamedTuple):
    """
    Constant values used by the modules.
    """

    default_train_config_file = "train_config.yaml"
    """The default train config file."""

    default_config_version = "default"
    """The default config version to use from train_config.yaml."""


def _overwrite_if_given(default_val, val):
    return val if val is not None else default_val


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
