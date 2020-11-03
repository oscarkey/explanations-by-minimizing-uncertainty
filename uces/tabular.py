"""Contains tools to load tabular datasets.

This is quite complicated, we should refactor it.
"""
import csv
from typing import Any, Callable, Dict, List, Literal, NamedTuple, Tuple, Union

import numpy as np
import torch
from numpy.random.mtrand import RandomState
from torch import Tensor
from torch.utils.data import Dataset

Split = Literal["train", "val", "test"]
Transform = Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]
NoTransform = lambda x, y: (x, y)


class _TabularDataConfig(NamedTuple):
    perturbations_unnormalized: Dict[str, float]
    value_range: Dict[str, Tuple[float, float]]
    ignore_columns: List[str]
    output_columns: List[str]


def _parse_tabular_data_config(file_path: str) -> _TabularDataConfig:
    config: Dict[str, Any] = {"value_range": {}, "perturbations_unnormalized": {}}
    with open(file_path) as file:
        reader = csv.reader(file, delimiter=":")
        for key, value in reader:
            key = key.strip()
            value = value.strip()
            if key.startswith("perturbation "):
                field = " ".join(key.split(" ")[1:])
                config["perturbations_unnormalized"][field] = float(value)
            elif key.startswith("range "):
                field = " ".join(key.split(" ")[1:])
                min_max = [float(v) for v in value.split(",")]
                config["value_range"][field] = (min_max[0], min_max[1])
            elif key == "ignore_columns":
                config["ignore_columns"] = value.split(",")
            elif key == "output_columns":
                config["output_columns"] = value.split(",")
            else:
                raise ValueError(f'Unknown tabular data config key "{key}"')
    return _TabularDataConfig(**config)


class TabularDataset(Dataset):
    """Loads CSV data, keeps the entire dataset in memory."""

    def __init__(
        self,
        data_file_path: str,
        config_file_path: str,
        split: Split,
        val_fraction: float = 0.1,
        test_fraction: float = 0.2,
        split_index: int = 0,
        transform: Transform = NoTransform,
    ):
        """Creates a new instance.

        :param data_file_path: path of the CSV file to load data from
        :param config_file_path: path of the config file for the data (see class doc)
        :param split: whether to load the training, validation, or test portion of the data
        :param val_fraction: fraction of the total rows to reserve as validation data, the rest
                             will be training and test data
        :param test_fraction: fraction of the total rows to reserve as test data, the rest will be
                              training and validation data
        :param split_index: seed for the RNG which splits the dataset into training, valdition, and
                            test rows
        :param transform: function to apply to each normalised row(s) before returning it,
                          (input row(s), target row(s)) -> (input row(s), target row(s))
        """
        super().__init__()
        self._config = _parse_tabular_data_config(config_file_path)
        self._dataset_columns, all_rows = _load_tabular_data(
            data_file_path, self._config.ignore_columns
        )
        self._row_min, self._row_range = self._get_row_stats(self._config, self._dataset_columns)

        self._dataset_values = self._get_split_rows(
            all_rows, split, val_fraction, test_fraction, split_index
        )

        x_idx, y_idx = self._get_input_output_column_indices(self._config, self._dataset_columns)
        self._input_column_indices, self._output_column_indices = x_idx, y_idx

        self._transform = transform

    def __getitem__(self, index: Union[int, Tensor, List[int]]) -> Tuple[Tensor, Tensor]:
        if isinstance(index, Tensor):
            index = index.tolist()
        row = self._dataset_values[index].clone()
        row -= self._row_min
        row /= self._row_range
        return self._transform(row[self._input_column_indices], row[self._output_column_indices])

    def get_unnormalized(self, index: Union[int, Tensor, List[int]]) -> Tuple[Tensor, Tensor]:
        """Returns (inputs, targets), where the inputs have not been normalised.

        No transform is applied to these rows.
        """
        if isinstance(index, Tensor):
            index = index.tolist()
        row = self._dataset_values[index].clone()
        return (row[self._input_column_indices], row[self._output_column_indices])

    @property
    def targets(self) -> Tensor:
        """Returns the targets normalized and transformed."""
        data_normalized = self._dataset_values.clone()
        data_normalized -= self._row_min
        data_normalized /= self._row_range
        _, outputs_normalized_transformed = self._transform(
            data_normalized[:, self._input_column_indices],
            data_normalized[:, self._output_column_indices],
        )
        assert outputs_normalized_transformed.ndim == 1 or (
            outputs_normalized_transformed.ndim == 2
            and outputs_normalized_transformed.size(1) == 1
        ), ".targets requires exactly one output column (to match TorchVision datasets)."
        return outputs_normalized_transformed.view(-1).clone()

    @property
    def column_names(self) -> Tuple[List[str], List[str]]:
        """Returns (input columns, output columns)."""
        return (
            np.array(self._dataset_columns)[self._input_column_indices].tolist(),
            np.array(self._dataset_columns)[self._output_column_indices].tolist(),
        )

    @property
    def perturbations(self) -> Tensor:
        """Returns the allowed perturbations in each input dimension.

        :returns: tensor with the same shape as the input tensor returned by __getitem__
        """
        input_columns, output_columns = self.column_names
        perturbations_unnormalized = []
        for column in self._dataset_columns:
            if column in input_columns:
                perturbations_unnormalized.append(self._config.perturbations_unnormalized[column])
            else:
                perturbations_unnormalized.append(0.0)
        return (torch.tensor(perturbations_unnormalized) / self._row_range)[
            self._input_column_indices
        ]

    def __len__(self) -> int:
        return self._dataset_values.size(0)

    def unnormalize(self, inputs: Tensor) -> Tensor:
        """Returns the given inputs with normalization undone."""
        # The normalizing constants are applied to the entire row, so add dummy outputs to the
        # inputs to make a row.
        row = torch.zeros_like(self._row_range)
        row[self._input_column_indices] = inputs
        row *= self._row_range
        row += self._row_min
        return row[self._input_column_indices]

    @staticmethod
    def _get_row_stats(
        config: _TabularDataConfig, column_names: List[str]
    ) -> Tuple[Tensor, Tensor]:
        mins = torch.empty(len(column_names))
        ranges = torch.empty(len(column_names))
        for i, column in enumerate(column_names):
            if column not in config.value_range:
                raise ValueError(f'Range for column "{column}" missing from dataset config.')

            column_min, column_max = config.value_range[column]
            mins[i] = column_min
            ranges[i] = column_max - column_min
        return mins, ranges

    @staticmethod
    def _get_input_output_column_indices(
        config: _TabularDataConfig, column_names: List[str]
    ) -> Tuple[List[int], List[int]]:
        input_indices = []
        output_indices = []
        for i, column in enumerate(column_names):
            if column in config.output_columns:
                output_indices.append(i)
            else:
                input_indices.append(i)
        return input_indices, output_indices

    @staticmethod
    def _get_split_rows(
        all_rows: Tensor, split: Split, val_fraction: float, test_fraction: float, split_index: int
    ) -> Tensor:
        assert 0 <= val_fraction and 0 <= test_fraction and (val_fraction + test_fraction) < 1
        if split in ("train", "val"):
            train_val_rows = _split_rows(
                all_rows, return_b=False, b_fraction=test_fraction, split_index=split_index
            )
            val_fraction_of_train_and_val = val_fraction / (1 - test_fraction)
            return _split_rows(
                train_val_rows,
                return_b=(split == "val"),
                b_fraction=val_fraction_of_train_and_val,
                split_index=split_index,
            )
        else:
            return _split_rows(
                all_rows, return_b=True, b_fraction=test_fraction, split_index=split_index
            )


def _split_rows(all_rows: Tensor, return_b: bool, b_fraction: float, split_index: int) -> Tensor:
    n_b_points = int(all_rows.size(0) * b_fraction)
    b_indices = RandomState(split_index).choice(
        np.arange(all_rows.size(0)), n_b_points, replace=False
    )
    if return_b:
        to_select = torch.full((all_rows.size(0),), False, dtype=torch.bool)
        to_select[b_indices] = True
    else:
        to_select = torch.full((all_rows.size(0),), True, dtype=torch.bool)
        to_select[b_indices] = False

    return all_rows[to_select]


def _load_tabular_data(file_path: str, ignore_columns: List[str]) -> Tuple[List[str], Tensor]:
    with open(file_path) as file:
        reader = csv.reader(file)
        all_columns = [c.strip() for c in next(reader)]

        filtered_rows = []
        for row in reader:
            filtered_rows.append([])
            for c, v in zip(all_columns, row):
                if c in ignore_columns:
                    continue

                # 'M' and 'B' are specific to the breastcancer dataset. In future we should find a
                # better way to do this.
                if v == "M":
                    filtered_rows[-1].append(1)
                elif v == "B":
                    filtered_rows[-1].append(0)
                else:
                    filtered_rows[-1].append(float(v))

        filtered_values = torch.tensor(filtered_rows)
        filtered_columns = [c for c in all_columns if c not in ignore_columns]
        return filtered_columns, filtered_values


def load_admissions() -> Tuple[TabularDataset, TabularDataset, TabularDataset, int]:
    def transform(inputs: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        # Classify applications as accepted if the admission probability is > 0.7.
        classes = torch.where(
            targets > 0.75, torch.tensor(1, dtype=torch.long), torch.tensor(0, dtype=torch.long),
        )
        classes = classes.view(-1)
        return inputs, classes

    dataset_file = "demo_datasets/graduate_admissions_dataset.csv"
    config_file = "demo_datasets/graduate_admissions_config.txt"
    train_data = TabularDataset(dataset_file, config_file, split="train", transform=transform)
    val_data = TabularDataset(dataset_file, config_file, split="val", transform=transform)
    test_data = TabularDataset(dataset_file, config_file, split="test", transform=transform)
    # TODO: should load this from the dataset config file.
    n_classes = 2

    return train_data, val_data, test_data, n_classes


def load_breast_cancer() -> Tuple[TabularDataset, TabularDataset, TabularDataset, int]:
    def transform(inputs: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        # Convert the target index into a long tensor, rather than float.
        classes = targets.long().view(-1)
        return inputs, classes

    dataset_file = "demo_datasets/breast_cancer_data.csv"
    config_file = "demo_datasets/breast_cancer_config.txt"
    train_data = TabularDataset(dataset_file, config_file, split="train", transform=transform)
    val_data = TabularDataset(dataset_file, config_file, split="val", transform=transform)
    test_data = TabularDataset(dataset_file, config_file, split="test", transform=transform)
    # TODO: should load this from the dataset config file.
    n_classes = 2

    return train_data, val_data, test_data, n_classes


def load_bostonhousing() -> Tuple[TabularDataset, TabularDataset, TabularDataset, int]:
    def transform(inputs: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        # Classify as "1" if price above median value (21.2) normalized.
        classes = torch.where(
            targets > 0.36, torch.tensor(1, dtype=torch.long), torch.tensor(0, dtype=torch.long),
        )
        classes = classes.view(-1)
        return inputs, classes

    dataset_file = "demo_datasets/boston_housing.csv"
    config_file = "demo_datasets/boston_housing_config.txt"
    train_data = TabularDataset(dataset_file, config_file, split="train", transform=transform)
    val_data = TabularDataset(dataset_file, config_file, split="val", transform=transform)
    test_data = TabularDataset(dataset_file, config_file, split="test", transform=transform)
    # TODO: should load this from the dataset config file.
    n_classes = 2

    return train_data, val_data, test_data, n_classes
