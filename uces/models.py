import csv
import os
from argparse import Namespace
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Module, ModuleList
from torch.nn.modules.flatten import Flatten

_CONFIG_FILE_NAME = "config.txt"
_ENSEMBLE_CHECKPOINTS_DIR = "ensembles"


class MLP(Module):
    """A single multi-layer perceptron for classification."""

    def __init__(self, n_hidden: int, input_flat_size: int, n_classes: int):
        super().__init__()
        self._net = nn.Sequential(  #
            Flatten(),  #
            nn.Linear(in_features=input_flat_size, out_features=n_hidden),  #
            nn.ReLU(),  #
            nn.BatchNorm1d(num_features=n_hidden),  #
            nn.Linear(in_features=n_hidden, out_features=n_hidden),  #
            nn.ReLU(),  #
            nn.BatchNorm1d(num_features=n_hidden),  #
            nn.Linear(in_features=n_hidden, out_features=n_classes),
        )

    def forward(self, x: Tensor):
        x = self._net(x)
        probs = F.softmax(x, dim=-1)
        return {"probs": probs, "logits": x}


def save_config(experiment_path: str, args: Namespace) -> None:
    """Saves the configuration to a CSV file."""
    with open(os.path.join(experiment_path, _CONFIG_FILE_NAME), "w") as file:
        writer = csv.writer(file)
        for key, value in args.__dict__.items():
            writer.writerow((key, value))


def load_config(experiment_path: str) -> Namespace:
    """Loads the model configuration from a CSV file."""
    config = {}
    with open(os.path.join(experiment_path, _CONFIG_FILE_NAME)) as file:
        lines = [line.replace("\n", "") for line in file if line != "\n"]
        reader = csv.reader(lines, delimiter=",")
        for key, value in reader:
            # Only restore keys we care about for the model.
            if key in ["ensemble_size", "n_hidden"]:
                value = int(value.strip())
            elif key in ["dataset"]:
                value = value.strip()
            elif key in ["adv_training"]:
                if value == "True":
                    value = True
                elif value == "False":
                    value = False
                else:
                    raise ValueError
            else:
                continue
            config[key] = value
    return Namespace(**config)


def get_config_for_checkpoint(results_root: str, checkpoint_id: int) -> Namespace:
    return load_config(os.path.join(results_root, _ENSEMBLE_CHECKPOINTS_DIR, str(checkpoint_id)))


class Ensemble(Module):
    """An ensemble of MLPs.

    Supports loading the MLPs from disk checkpoints.
    """

    def __init__(
        self,
        results_root: str,
        checkpoint_id: int,
        input_flat_size: int,
        n_classes: int,
        n_components: Optional[int] = None,
    ):
        """Creates a new instance

        :param n_components: number of models to load, or None to load all available models at the
                             path
        """
        super().__init__()
        self._experiment_path = os.path.join(
            results_root, _ENSEMBLE_CHECKPOINTS_DIR, str(checkpoint_id)
        )
        self._n_components = n_components
        self._models = None
        self._input_flat_size = input_flat_size
        self._n_classes = n_classes

    def load(self, device: str) -> None:
        """Restores the models in the ensemble from disk checkpoints."""
        if self._models is not None:
            raise ValueError("Already loaded")

        args = load_config(self._experiment_path)
        print(f"Loaded ensemble with config: {vars(args)}")

        if self._n_components == None:
            self._n_components = args.ensemble_size

        self._models = ModuleList([])
        for ensemble_i in range(self._n_components):
            model = MLP(
                n_hidden=args.n_hidden,
                input_flat_size=self._input_flat_size,
                n_classes=self._n_classes,
            )
            model = model.to(device)
            path = os.path.join(
                self._experiment_path, _get_ensemble_component_file_name(ensemble_i)
            )
            model.load_state_dict(torch.load(path, map_location=device))
            self._models.append(model)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        if self._models is None:
            raise ValueError("Must load first")

        probs = torch.stack([model(x)["probs"] for model in self._models], dim=0).mean(0)
        logits = torch.stack([model(x)["logits"] for model in self._models], dim=0).mean(0)
        return {"probs": probs, "logits": logits}

    def get_models(self) -> List[MLP]:
        """Returns a list of the models which make up the ensemble."""
        if self._models is None:
            raise ValueError("Must load first")
        return self._models


def save_ensemble_model(path: str, model: MLP, ensemble_i: int) -> None:
    checkpoint_path = os.path.join(path, _get_ensemble_component_file_name(ensemble_i))
    torch.save(model.state_dict(), checkpoint_path)


def get_fresh_checkpoint_path(results_root: str) -> str:
    """Creates a new directory to hold the ensemble checkpoints.

    The directories are named numerically, with the new directory being the next unused number in
    the sequence.
    """
    ensemble_checkpoints_path = os.path.join(results_root, _ENSEMBLE_CHECKPOINTS_DIR)
    if not os.path.isdir(ensemble_checkpoints_path):
        os.makedirs(ensemble_checkpoints_path)

    last_id = -1
    for sub_name in os.listdir(ensemble_checkpoints_path):
        sub_path = os.path.join(ensemble_checkpoints_path, sub_name)
        if not os.path.isdir(os.path.join(sub_path)):
            continue
        exp_id = _try_parse_int(sub_name)
        if exp_id is None:
            continue
        last_id = max(exp_id, last_id)

    new_id = last_id + 1
    results_path = os.path.join(ensemble_checkpoints_path, f"{new_id}")
    os.mkdir(results_path)
    print(f"Saving results to {results_path}")
    return results_path


def _try_parse_int(string: str) -> Optional[int]:
    try:
        return int(string)
    except ValueError:
        return None


def _get_ensemble_component_file_name(component_index: int) -> str:
    """Returns the file name for the checkpoint of the ith model in the ensemble."""
    return f"ensemble_{component_index}"
