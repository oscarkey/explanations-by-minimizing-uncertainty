from __future__ import annotations

from argparse import ArgumentParser
from typing import List, Tuple

import numpy as np
import torch
import uces.datasets
import uces.im_evaluation as im_evaluation
import uces.models
import uces.utils
from torch import Tensor

from experiments.exp_qualitative_eval import get_dataset, get_originals_targets
from experiments.method_prototypes import PrototypesMethod
from experiments.method_wachter import AlibiWachterMethod
from experiments.methods import CEMethod, OurMethod, TrainingDataMethod


def main(results_dir: str, data_dir: str, no_adv_training_id: str, adv_training_id: str) -> None:
    dataset_name = get_dataset(results_dir, no_adv_training_id, adv_training_id)
    print(f"Evaluating on dataset {dataset_name}")
    _, _, test_dataset, n_classes = uces.datasets.load_dataset(dataset_name, data_dir, results_dir)

    methods: List[CEMethod] = [
        TrainingDataMethod(),
        AlibiWachterMethod(),
        PrototypesMethod(),
        OurMethod(adv_training=False, ensemble_size=1),
        OurMethod(adv_training=True, ensemble_size=1),
        OurMethod(adv_training=False, ensemble_size=30),
        OurMethod(adv_training=True, ensemble_size=30),
    ]

    n_repeats = 10
    im1 = np.zeros((n_repeats, len(methods)))
    l1_dist = np.zeros((n_repeats, len(methods)))
    for repeat in range(n_repeats):
        print(f"Starting repeat {repeat + 1} / {n_repeats}")

        seed = 12345 + repeat
        originals, original_labels, target_labels, ce_indices = get_originals_targets(
            dataset_name, test_dataset, seed
        )

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
                    load_from_cache=False,
                )
            )
        im_scores = _compute_im_scores(
            dataset_name, data_dir, results_dir, counterfactuals, original_labels, target_labels,
        )
        l1_dist_p = _compute_l1_distances(originals, counterfactuals)
        l1_dist[repeat] = torch.stack(l1_dist_p, axis=1).mean(axis=0).detach().numpy().squeeze()
        for method_i in range(len(methods)):
            im1[repeat, method_i] = im_scores[method_i][0].mean().detach().numpy()

    for i, method in enumerate(methods):
        print(
            f"{method.name}: "
            f"im1={np.mean(im1, axis=0)[i]:.4f} ({np.std(im1, axis=0)[i]:.4f}) "
            f"l1={np.mean(l1_dist, axis=0)[i]:.4f} ({np.std(l1_dist, axis=0)[i]:.4f})"
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


def _compute_l1_distances(originals: Tensor, counterfactuals: List[Tensor]) -> List[Tensor]:
    l1_distances = []
    for method_counterfactuals in counterfactuals:
        l1_distances.append(
            (originals - method_counterfactuals).abs().view(originals.size(0), -1).sum(1)
        )
    return l1_distances


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--no_adv_training_id", type=str, required=True)
    parser.add_argument("--adv_training_id", type=str, required=True)
    args = parser.parse_args()

    main(args.results_dir, args.data_dir, args.no_adv_training_id, args.adv_training_id)
