"""Evaluation of CEs based on the IM1 and IM2 metrics.

IM1 and IM2 were introduced in
"Interpretable Counterfactual Explanations Guided by Prototypes"; Van Looveren, Klaise
https://arxiv.org/abs/1907.02584

This module defines autoencoders for evaluating IM1 and IM2 for MNIST, the Boston Housing dataset,
and the Wisconsin Breast Cancer dataset. The configurations match those in the paper above.
"""
import itertools
import os
import uuid
from abc import ABC, abstractmethod
from glob import glob
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset

import uces.datasets
from uces.datasets import SupportedDataset
from uces.utils import AssertShape

_CHECKPOINT_SUB_DIR = "im_eval"
_CHECKPOINT_FILE_NAME_PREFIX = "checkpoint"
_EPS = 1e-7


class _AutoEncoder(Module, ABC):
    @property
    @abstractmethod
    def input_shape(self) -> Tuple:
        pass


class _MnistAllClassesAE(_AutoEncoder):
    def __init__(self) -> None:
        super().__init__()
        self._encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            AssertShape((16, 28, 28)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            AssertShape((16, 14, 14)),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        self._decoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(self.input_shape)
        return self._decoder(self._encoder(x))

    @property
    def input_shape(self) -> Tuple:
        return (-1, 1, 28, 28)


class _MnistClassSpecificAE(_AutoEncoder):
    def __init__(self) -> None:
        super().__init__()
        self._encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            AssertShape((16, 14, 14)),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            AssertShape((8, 7, 7)),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
        )
        self._decoder = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(self.input_shape)
        return self._decoder(self._encoder(x))

    @property
    def input_shape(self) -> Tuple:
        return (-1, 1, 28, 28)


class _BreastCancerAE(_AutoEncoder):
    def __init__(self) -> None:
        super().__init__()
        self._encoder = nn.Sequential(
            nn.Linear(30, 20), nn.ReLU(), nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 6),
        )
        self._decoder = nn.Sequential(
            nn.Linear(6, 10), nn.ReLU(), nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 30),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(self.input_shape)
        return self._decoder(self._encoder(x))

    @property
    def input_shape(self) -> Tuple:
        return (-1, 30)


class _BostonHousingAE(_AutoEncoder):
    def __init__(self) -> None:
        super().__init__()
        self._encoder = nn.Sequential(
            nn.Linear(13, 20), nn.ReLU(), nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 6),
        )
        self._decoder = nn.Sequential(
            nn.Linear(6, 10), nn.ReLU(), nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 13),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(self.input_shape)
        return self._decoder(self._encoder(x))

    @property
    def input_shape(self) -> Tuple:
        return (-1, 13)


def evaluate(
    dataset: str,
    data_dir: str,
    results_dir: str,
    counterfactuals: Tensor,
    original_classes: Tensor,
    counterfactual_classes: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Returns the scores of the given CEs under (IM1, IM2)."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    vaes = _load_vaes(results_dir, device, dataset)
    if vaes is None:
        vaes = _train_and_save_vaes(dataset, data_dir, results_dir, device)

    counterfactuals = counterfactuals.to(device)
    original_classes = original_classes.to(device)
    counterfactual_classes = counterfactual_classes.to(device)

    im1s = []
    im2s = []
    for ce, original_class, counterfactual_class in zip(
        counterfactuals, original_classes, counterfactual_classes
    ):
        im1, im2 = _evaluate_counterfactual(
            vaes, ce, original_class.item(), counterfactual_class.item()
        )
        im1s.append(im1)
        im2s.append(im2)
    return torch.cat(im1s), torch.cat(im2s)


def _evaluate_counterfactual(
    models: Dict[Union[int, str], _AutoEncoder],
    counterfactual: Tensor,
    original_class: int,
    counterfactual_class: int,
) -> Tuple[Tensor, Tensor]:
    counterfactual = counterfactual.view(models[counterfactual_class].input_shape)
    recon_cf_class = models[counterfactual_class](counterfactual)
    recon_original_class = models[original_class](counterfactual)
    im1_numerator = F.mse_loss(counterfactual, recon_cf_class)
    im1_denominator = F.mse_loss(counterfactual, recon_original_class)
    im1 = im1_numerator / (im1_denominator + _EPS)

    recon_all_class = models["all"](counterfactual)
    im2_numerator = F.mse_loss(recon_cf_class, recon_all_class)
    im2_denominator = torch.mean(torch.abs(counterfactual))
    im2 = im2_numerator / (im2_denominator + _EPS)

    return im1.view(1), im2.view(1)


def _load_vaes(
    results_dir: str, device: torch.device, dataset: str
) -> Optional[Dict[Union[int, str], _AutoEncoder]]:
    file_paths = glob(
        os.path.join(
            results_dir, _CHECKPOINT_SUB_DIR, f"{_CHECKPOINT_FILE_NAME_PREFIX}_{dataset}_*.pt"
        )
    )

    if len(file_paths) == 0:
        return None

    file_path = file_paths[0]
    print(f"Loaded IM AE for evaluation from {os.path.basename(file_path)}")

    state_dicts = torch.load(file_path, map_location=device)
    models = {}
    for class_id, state_dict in state_dicts.items():
        models[class_id] = _create_autoencoder(class_id, dataset).to(device)
        models[class_id].load_state_dict(state_dict)
    return models


def _train_and_save_vaes(
    dataset: str, data_dir: str, results_dir: str, device: torch.device
) -> Dict[Union[int, str], _AutoEncoder]:
    print("Training autoencoders for evaluation...")
    train_dataset, _, test_dataset, n_classes = uces.datasets.load_dataset(
        dataset, data_dir, results_dir
    )

    models = {}

    for class_id in itertools.chain(range(0, n_classes), ["all"]):
        if class_id != "all":
            assert isinstance(class_id, int)
            train_subset = _subset_dataset(train_dataset, class_id=class_id)
            test_subset = _subset_dataset(test_dataset, class_id=class_id)
        else:
            train_subset = train_dataset
            test_subset = test_dataset

        print(f"Training model for class {class_id}")
        model = _train_model(dataset, train_subset, class_id, device)
        test_loss = _test_model(test_subset, device, model)
        print(f"class {class_id}, test loss = {test_loss:.2f}")

        models[class_id] = model

    _save_models(models, results_dir, dataset)

    return models


def _save_models(
    models: Dict[Union[int, str], _AutoEncoder], results_dir: str, dataset: str
) -> None:
    state_dicts = {k: model.state_dict() for k, model in models.items()}
    dir_path = os.path.join(results_dir, _CHECKPOINT_SUB_DIR)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(
        dir_path, f"{_CHECKPOINT_FILE_NAME_PREFIX}_{dataset}_{uuid.uuid1()}.pt"
    )
    torch.save(state_dicts, file_path, _use_new_zipfile_serialization=False)


def _train_model(
    dataset_name: str, dataset: Dataset, class_id: Union[str, int], device: torch.device
) -> _AutoEncoder:
    train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    autoencoder = _create_autoencoder(class_id, dataset_name).to(device)
    autoencoder.train()

    optimizer = Adam(autoencoder.parameters())
    losses = []

    epochs = _get_epochs(class_id, dataset_name)
    for epoch in range(epochs):
        for inputs, _ in train_dataloader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            recon = autoencoder(inputs)
            loss = F.mse_loss(inputs, recon.view(inputs.size()))
            loss.backward()
            losses.append(loss.detach())
            optimizer.step()
        loss = torch.stack(losses).mean().item()
        if epoch == epochs - 1 or epoch % 10 == 0:
            print(f"Epoch {epoch}: train loss={loss:.5f}")

    return autoencoder


def _create_autoencoder(class_id: Union[str, int], dataset_name: str) -> _AutoEncoder:
    if dataset_name == "mnist" or dataset_name.startswith("simulatedmnist"):
        if class_id == "all":
            return _MnistAllClassesAE()
        else:
            return _MnistClassSpecificAE()
    elif dataset_name == "breastcancer" or dataset_name.startswith("simulatedbc"):
        return _BreastCancerAE()
    elif dataset_name == "bostonhousing":
        return _BostonHousingAE()
    else:
        raise NotImplementedError


def _get_epochs(class_id: Union[str, int], dataset_name: str) -> int:
    if dataset_name == "mnist" or dataset_name.startswith("simulatedmnist"):
        if class_id == "all":
            return 4
        else:
            return 30
    elif dataset_name == "breastcancer" or dataset_name.startswith("simulatedbc"):
        return 500
    elif dataset_name == "bostonhousing":
        return 500
    else:
        raise NotImplementedError


def _test_model(dataset: Dataset, device: torch.device, model: _AutoEncoder) -> float:
    test_dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
    model.eval()
    with torch.no_grad():
        losses = []
        for inputs, _ in test_dataloader:
            inputs = inputs.to(device)
            recon = model(inputs)
            losses.append(F.mse_loss(inputs, recon.view(inputs.size())).detach())
        return torch.stack(losses).mean().item()


def _subset_dataset(dataset: SupportedDataset, class_id: int) -> Dataset:
    class_indices = torch.arange(len(dataset))[dataset.targets == class_id]
    return Subset(dataset, class_indices)
