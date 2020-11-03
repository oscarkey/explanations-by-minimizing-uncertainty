"""Defines methods for generating CEs as evaluted by exp_*_eval.py.

Additional methods are implemented in the method_* modules.
"""
import os
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from glob import glob
from typing import Dict, List, NamedTuple, Optional

import torch
import uces.datasets
from torch import Tensor
from uces.datasets import SupportedDataset
from uces.generators import GreedyGenerator
from uces.models import Ensemble

_COUNTERFACTUAL_SAVE_FILE_PREFIX = "counterfactuals"


class SavedCounterfactuals(NamedTuple):
    counterfactuals: Tensor
    ce_indices: List[int]

    def save(self, results_dir: str, method_name: str, dataset: str) -> None:
        dir_path = os.path.join(results_dir, method_name)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        file_name = f"{_COUNTERFACTUAL_SAVE_FILE_PREFIX}_{dataset}_n{len(self.ce_indices)}_{uuid.uuid1()}.pt"
        file_path = os.path.join(results_dir, method_name, file_name)
        torch.save(self, file_path, _use_new_zipfile_serialization=False)

    @staticmethod
    def load(
        results_dir: str,
        method_name: str,
        dataset: str,
        ce_indices: List[int],
        device: torch.device,
    ) -> "Optional[SavedCounterfactuals]":
        file_pattern = f"{_COUNTERFACTUAL_SAVE_FILE_PREFIX}_{dataset}_n{len(ce_indices)}_*.pt"
        file_paths = glob(os.path.join(results_dir, method_name, file_pattern))
        if len(file_paths) == 0:
            return None

        file_path = file_paths[0]
        print(f"Loading cached counterfactuals from {file_path}")

        saved = torch.load(file_path, map_location=device)
        assert isinstance(saved, SavedCounterfactuals)
        if saved.ce_indices != ce_indices:
            raise ValueError("Loaded CEs were for wrong indices in dataset")

        return saved


class CEMethod(ABC):
    """Base class for defining CE generation methods."""

    def get_counterfactuals(
        self,
        results_dir: str,
        data_dir: str,
        ensemble_id: int,
        dataset: str,
        originals: Tensor,
        target_labels: Tensor,
        ce_indices: List[int],
        load_from_cache: bool = True,
    ) -> Tensor:
        """Generate CEs for the given originals and target classes.

        This method will cache generated CEs to disk, and attempt to load these on subsequent calls
        for the same method. This makes it easier to debug evaluation code, as often generating CEs
        can take a long time. Set `load_from_cache`=False to disable loading (though generated CEs)
        will still be saved.
        """

        if load_from_cache:
            saved_counterfactuals = SavedCounterfactuals.load(
                results_dir, self.name, dataset, ce_indices, originals.device
            )
        else:
            saved_counterfactuals = None

        if saved_counterfactuals is None:
            train_dataset, _, _, n_classes = uces.datasets.load_dataset(
                dataset, data_dir, results_dir
            )
            counterfactuals = self._generate_counterfactuals(
                results_dir,
                ensemble_id,
                dataset,
                train_dataset,
                n_classes,
                originals,
                target_labels,
            )
            saved_counterfactuals = SavedCounterfactuals(counterfactuals, ce_indices)
            saved_counterfactuals.save(results_dir, self.name, dataset)

        return saved_counterfactuals.counterfactuals

    @abstractmethod
    def _generate_counterfactuals(
        self,
        results_dir: str,
        ensemble_id: int,
        dataset: str,
        train_dataset: SupportedDataset,
        n_classes: int,
        originals: Tensor,
        targets: Tensor,
    ) -> Tensor:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def use_adversarial_training(self) -> bool:
        """Whether _generate_counterfactuals() should be passed an ensemble trained using adversarial training."""
        pass


class TrainingDataMethod(CEMethod):
    """Dummy method which just picks a random instance from the training set in the relevant class."""

    def _generate_counterfactuals(
        self,
        results_dir: str,
        ensemble_id: int,
        dataset: str,
        train_dataset: SupportedDataset,
        n_classes: int,
        originals: Tensor,
        targets: Tensor,
    ) -> Tensor:
        instances_by_class = self._get_instances_by_class(train_dataset)
        counterfactuals = [instances_by_class[target.item()].pop() for target in targets]
        return torch.stack(counterfactuals, axis=0)

    @staticmethod
    def _get_instances_by_class(dataset: SupportedDataset) -> Dict[int, List[Tensor]]:
        instances = defaultdict(lambda: [])
        for x, label in dataset:
            if isinstance(label, Tensor):
                label = label.item()
            instances[label].append(x)
        return instances

    @property
    def name(self) -> str:
        return "training_data"

    @property
    def use_adversarial_training(self) -> bool:
        return False


class OurMethod(CEMethod):
    def __init__(self, adv_training: bool, ensemble_size: int) -> None:
        super().__init__()
        self._adv_training = adv_training
        self._ensemble_size = ensemble_size

    def _generate_counterfactuals(
        self,
        results_dir: str,
        ensemble_id: int,
        dataset: str,
        train_dataset: SupportedDataset,
        n_classes: int,
        originals: Tensor,
        targets: Tensor,
    ) -> Tensor:
        input_flat_size = originals.view(originals.size(0), -1).size(1)
        ensemble = Ensemble(
            results_dir, ensemble_id, input_flat_size, n_classes, self._ensemble_size
        )
        ensemble.load(device=originals.device)
        ensemble.eval()

        if dataset == "mnist":
            n_changes = 5
            max_iters = 5000
        elif dataset.startswith("simulatedmnist"):
            n_changes = 5
            max_iters = 500
        elif dataset == "breastcancer":
            n_changes = 10
            max_iters = 200
        elif dataset.startswith("simulatedbc"):
            n_changes = 5
            max_iters = 500
        elif dataset == "bostonhousing":
            n_changes = 10
            max_iters = 500
        else:
            raise ValueError

        delta = torch.full((input_flat_size,), 1.0 / n_changes)
        generator = GreedyGenerator(
            ensemble,
            confidence_threshold=0.99,
            n_changes=n_changes,
            max_iters=max_iters,
            perturbations=delta,
            num_classes=n_classes,
        )
        counterfactuals, _, _ = generator.generate_targeted(originals, targets)

        return counterfactuals

    @property
    def name(self) -> str:
        return f"adv_training={self._adv_training},ensemble_size={self._ensemble_size}"

    @property
    def use_adversarial_training(self) -> bool:
        return self._adv_training
