"""Central location which contains functions for loading the datasets."""
import math
from typing import Literal, Optional, Tuple, Union, overload

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import uces.tabular
from uces.tabular import TabularDataset

SupportedDataset = Union[TabularDataset, MNIST]


@overload
def load_dataset(
    dataset: Literal["mnist"], data_dir: str, results_dir: str
) -> Tuple[MNIST, None, MNIST, int]:
    ...


@overload
def load_dataset(
    dataset: Literal["admissions", "breastcancer"], data_dir: str, results_dir: str
) -> Tuple[TabularDataset, Optional[TabularDataset], TabularDataset, int]:
    ...


@overload
def load_dataset(
    dataset: str, data_dir: str, results_dir: str
) -> Tuple[SupportedDataset, Optional[SupportedDataset], SupportedDataset, int]:
    ...


def load_dataset(
    dataset, data_dir, results_dir
) -> Tuple[SupportedDataset, Optional[SupportedDataset], SupportedDataset, int]:
    """Loads the given classification dataset.

    :returns: (train dataset, val dataset (or None if not available), test dataset, num classes)
    """
    if dataset == "mnist":
        train_data = MNIST(data_dir, train=True, download=True, transform=ToTensor())
        test_data = MNIST(data_dir, train=False, download=True, transform=ToTensor())
        n_classes = 10
        return train_data, None, test_data, n_classes

    elif dataset == "admissions":
        return uces.tabular.load_admissions()

    elif dataset == "breastcancer":
        return uces.tabular.load_breast_cancer()

    elif dataset == "bostonhousing":
        return uces.tabular.load_bostonhousing()

    else:
        raise ValueError(f"Unknown dataset {dataset}")
