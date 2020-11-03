"""Generates a set of CEs, computes various evaluation metrics, and displays the CEs.

This is not the full quantitative evaluation because only one set of CEs is computed.
See exp_quantitative_eval.py for a full quantitative evaluation.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from typing import Iterable, List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
import uces.datasets
import uces.im_evaluation as im_evaluation
import uces.models
import uces.utils
from toposort import CircularDependencyError, toposort
from torch import Tensor
from torch.utils.data import Dataset
from uces.datasets import SupportedDataset
from uces.tabular import TabularDataset

from experiments.method_prototypes import PrototypesMethod
from experiments.methods import CEMethod, OurMethod, TrainingDataMethod


def main(results_dir: str, data_dir: str, no_adv_training_id: str, adv_training_id: str) -> None:
    dataset_name = get_dataset(results_dir, no_adv_training_id, adv_training_id)
    print(f"Evaluating on dataset {dataset_name}")
    _, _, test_dataset, n_classes = uces.datasets.load_dataset(dataset_name, data_dir, results_dir)

    seed = 0
    originals, original_labels, target_labels, ce_indices = get_originals_targets(
        dataset_name, test_dataset, seed
    )

    methods: List[CEMethod] = [
        TrainingDataMethod(),
        PrototypesMethod(),
        OurMethod(adv_training=False, ensemble_size=1),
        OurMethod(adv_training=True, ensemble_size=1),
        OurMethod(adv_training=False, ensemble_size=30),
        OurMethod(adv_training=True, ensemble_size=30),
    ]

    counterfactuals = []
    for method in methods:
        ensemble_id = int(
            adv_training_id if method.use_adversarial_training else no_adv_training_id
        )
        counterfactuals.append(
            method.get_counterfactuals(
                results_dir,
                data_dir,
                ensemble_id,
                dataset_name,
                originals,
                target_labels,
                ce_indices,
            )
        )

    if dataset_name == "mnist" or dataset_name.startswith("simulatedmnist"):
        _plot_mnist(methods, originals, counterfactuals, target_labels)
    elif isinstance(test_dataset, TabularDataset):
        # _plot_tabular_distribution(
        #     test_dataset, methods, originals, counterfactuals, target_labels
        # )
        _print_tabular_examples(test_dataset, methods, originals, counterfactuals, target_labels)
    else:
        raise ValueError

    im_scores = _compute_im_scores(
        dataset_name, data_dir, results_dir, counterfactuals, original_labels, target_labels
    )
    l1_rank_modes = _compute_l1_ranks(originals, counterfactuals)

    for i, method in enumerate(methods):
        print(
            f"{method.name.ljust(40)}: "
            f"im1={im_scores[i][0].mean():.3f} ({torch.std(im_scores[i][0]):.3f}), "
            f"im2={im_scores[i][1].mean():.3f} ({torch.std(im_scores[i][1]):.3f}), "
            f"mode L1 ranking={l1_rank_modes[i]} "
        )

    _compare_logps(
        methods, counterfactuals, target_labels, test_dataset, dataset_name, data_dir, results_dir
    )


def _compute_im_scores(
    dataset: str,
    data_dir: str,
    results_dir: str,
    counterfactuals: List[Tensor],
    original_classes: Tensor,
    counterfactual_classes: Tensor,
) -> List[Tuple[Tensor, Tensor]]:
    im_scores = []
    for method_counterfactuals in counterfactuals:
        im_scores.append(
            im_evaluation.evaluate(
                dataset,
                data_dir,
                results_dir,
                method_counterfactuals,
                original_classes,
                counterfactual_classes,
            )
        )
    return im_scores


def _compute_l1_ranks(originals: Tensor, counterfactuals: List[Tensor]) -> Tensor:
    l1_distances = []
    for method_counterfactuals in counterfactuals:
        l1_distances.append(
            (originals - method_counterfactuals).abs().view(originals.size(0), -1).sum(1)
        )
    l1_rankings = torch.argsort(torch.stack(l1_distances, axis=1), axis=1, descending=False)
    return torch.mode(l1_rankings, axis=0)[0]


def get_dataset(results_dir: str, no_adv_training_id: str, adv_training_id: str) -> str:
    """Returns the name of the dataset associated with the ensemble ids.

    Asserts that the ids match.
    """
    no_adv_config = uces.models.get_config_for_checkpoint(results_dir, no_adv_training_id)
    adv_config = uces.models.get_config_for_checkpoint(results_dir, adv_training_id)
    if no_adv_config.adv_training:
        raise ValueError("No adv training checkpoint had adv training enabled")
    if not adv_config.adv_training:
        raise ValueError("Adv training checkpoint had adv training disabled")
    if no_adv_config.dataset != adv_config.dataset:
        raise ValueError("Adv and no adv checkpoints are for different datasets")
    return adv_config.dataset


def get_originals_targets(
    dataset_name: str, test_data: Dataset, seed: int, n_counterfactuals: int = 100
) -> Tuple[Tensor, Tensor, Tensor, List[int]]:
    """Chooses which test inputs to generate CEs for, and the target classes of those CEs.

    :returns: (originals, original labels, target labels, indicies), where indicies indicates which
              originals in the dataset were chosen
    """
    originals, original_labels, indicies = _get_originals(test_data, seed, n_counterfactuals)
    target_labels = _choose_targets(dataset_name, original_labels, seed)
    return originals, original_labels, target_labels, indicies


def _get_originals(
    test_data: Dataset, seed: int, n_counterfactuals: int
) -> Tuple[Tensor, Tensor, List[int]]:
    if n_counterfactuals > len(test_data):
        raise ValueError("Requested more CEs than length of dataset.")

    indicies = np.random.default_rng(seed=seed + 1).choice(
        len(test_data), n_counterfactuals, replace=False
    )
    originals = torch.stack([test_data[i][0] for i in indicies])
    labels = torch.tensor([test_data[i][1] for i in indicies], device=originals.device)
    return originals, labels, indicies.tolist()


def _choose_targets(dataset: str, original_labels: Tensor, seed: int) -> Tensor:
    if dataset in ("breastcancer", "bostonhousing") or dataset.startswith("simulatedbc"):
        return torch.ones_like(original_labels) - original_labels

    elif dataset == "mnist" or dataset.startswith("simulatedmnist"):
        possible_targets = {
            0: [3, 6, 8],
            1: [4, 7, 9],
            2: [3, 7],
            3: [0, 2, 8],
            4: [1, 9],
            5: [6, 8],
            6: [0, 5, 8],
            7: [1, 2, 9],
            8: [0, 3, 6],
            9: [1, 4, 7],
        }
        generator = np.random.default_rng(seed=seed + 2)
        targets = [
            generator.choice(possible_targets[original.item()], 1) for original in original_labels
        ]
        return torch.tensor(np.concatenate(targets), device=original_labels.device)

    else:
        raise ValueError


def _plot_mnist(
    methods: Sequence[CEMethod],
    originals: Tensor,
    counterfactuals: List[Tensor],
    target_labels: Tensor,
) -> None:
    num_examples = min(5, len(originals))
    fig, axes = plt.subplots(num_examples, 1 + len(methods), squeeze=False)
    for row in range(num_examples):
        axes[row, 0].set_ylabel(f"target {target_labels[row]}")
        axes[row, 0].imshow(originals[row].view(28, 28).detach().cpu().numpy())

    for method_i, (method, method_counterfactuals) in enumerate(zip(methods, counterfactuals)):
        for row in range(num_examples):
            col = method_i + 1
            if row == 2:
                axes[row, col].set_ylabel(method.name)
            axes[row, col].imshow(method_counterfactuals[row].view(28, 28).detach().cpu().numpy())

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig("mnist_counterfactuals.png")
    plt.close()


def _plot_tabular_distribution(
    dataset: Union[TabularDataset],
    methods: Sequence[CEMethod],
    originals: Tensor,
    counterfactuals: List[Tensor],
    target_labels: Tensor,
) -> None:
    print("Plotting...")
    num_examples = min(8, len(originals))
    n_features = originals.size(1)
    xs, ys = uces.utils.get_inputs_and_targets(dataset, device=originals.device)
    fig, axes = plt.subplots(n_features, len(methods), figsize=(20, 20))
    for method_i, (method, method_counterfactuals) in enumerate(zip(methods, counterfactuals)):
        print("method", method.name)
        for feature_i in range(n_features):
            for c in [0, 1]:
                xs_in_class = xs[ys.view(-1) == c]
                color = f"C{c}"
                axes[feature_i, method_i].hist(
                    xs_in_class[:, feature_i].detach().cpu().numpy(),
                    alpha=0.5,
                    color=color,
                    bins=100,
                )

            if feature_i == 3:
                axes[feature_i, method_i].set_ylabel(method.name)

            for ce, target in zip(
                method_counterfactuals[:num_examples], target_labels[:num_examples]
            ):
                color = f"C{target}"
                axes[feature_i, method_i].axvline(
                    ce[feature_i].detach().cpu().numpy(), color=color
                )

    plt.savefig("tabular_counterfactuals.png")
    plt.close()


def _print_tabular_examples(
    dataset: TabularDataset,
    methods: Sequence[CEMethod],
    originals: Tensor,
    counterfactuals: List[Tensor],
    target_labels: Tensor,
) -> None:
    for i in range(4):
        print(f"target: {target_labels[i]}")
        _print_justified(dataset.column_names[0])

        original_unnormalized = dataset.unnormalize(originals[i])
        original_str = [f"{feature:.2f}" for feature in original_unnormalized]
        print("original")
        _print_justified(original_str)

        for method, ces in zip(methods, counterfactuals):
            delta = dataset.unnormalize(ces[i]) - original_unnormalized
            delta_str = [f"{feature:+.2f}" for feature in delta]
            print(method.name)
            _print_justified(delta_str)

        print("")


def _print_justified(values: Iterable[str]) -> None:
    justified = [x.rjust(8) for x in values]
    print(" ".join(justified))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--no_adv_training_id", type=str, required=True)
    parser.add_argument("--adv_training_id", type=str, required=True)
    args = parser.parse_args()

    main(args.results_dir, args.data_dir, args.no_adv_training_id, args.adv_training_id)
