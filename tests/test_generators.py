from typing import cast
from unittest.mock import Mock

import torch
import torch.nn.functional as F
from uces.generators import GreedyGenerator
from torch import Tensor


class TestGreedyGenerator:
    def test__generate_targeted__only_changes_feature_max_times(self):
        ensemble = Mock()

        def ensemble_call(*args, **kwargs):
            xs = cast(Tensor, args[0])
            # We have two inputs and two classes. We connect the logit for class "0" to xs[0], and
            # the logit for class "1" to the xs[1]. We then add 1 to the logit for class "1". Thus,
            # the original ([0., 0.]) will be classified as "1", and the algorithm will try to
            # increment xs[0] as much as possible to change the classification to "0".
            logits = xs + torch.tensor([0.0, 1.0])
            probs = F.softmax(logits, dim=1)
            return {"logits": logits, "probs": probs}

        ensemble.side_effect = ensemble_call

        perturbations = torch.tensor([0.1, 0.1])
        originals = torch.tensor([0.0, 0.0]).reshape(1, -1)
        targets = torch.tensor([0])

        explanation_with_7_changes, _, _ = GreedyGenerator(
            ensemble,
            confidence_threshold=0.999,
            n_changes=7,
            max_iters=20,
            num_classes=2,
            perturbations=perturbations,
        ).generate_targeted(originals, targets)

        explanation_with_2_changes, _, _ = GreedyGenerator(
            ensemble,
            confidence_threshold=0.999,
            n_changes=2,
            max_iters=20,
            num_classes=2,
            perturbations=perturbations,
        ).generate_targeted(originals, targets)

        # By checking both 2 and 10 max changes, we ensure that one of them isn't just the max by
        # chance.
        # Note that that the values are clamped to 1.0, so we can't have more than 10 changes at
        # perturbation size 0.1.
        assert torch.allclose(explanation_with_7_changes, torch.tensor([0.7, 0.0]))
        assert torch.allclose(explanation_with_2_changes, torch.tensor([0.2, 0.0]))

    def test__generate_targeted__once_feature_changed_max_times_changes_other_feature(self):
        ensemble = Mock()

        def ensemble_call(*args, **kwargs):
            xs = cast(Tensor, args[0])
            # We have two inputs and two classes. The matrix multiplication connects both inputs to
            # the logit for class "0". However, xs[0] has 2x the influence of xs[1], thus the
            # algorithm will increment xs[0] first and then switch to xs[1] once xs[0] has been
            # incremented the maximum number of times.
            logits = torch.matmul(xs, torch.tensor([[2.0, 0.0,], [1.0, 0.0]])) + torch.tensor(
                [0.0, 1.0]
            )
            probs = F.softmax(logits, dim=1)
            return {"logits": logits, "probs": probs}

        ensemble.side_effect = ensemble_call

        perturbations = torch.tensor([0.2, 0.1])
        originals = torch.tensor([0.0, 0.0]).reshape(1, -1)
        targets = torch.tensor([0])

        explanation, _, _ = GreedyGenerator(
            ensemble,
            confidence_threshold=1.0,
            n_changes=2,
            max_iters=20,
            num_classes=2,
            perturbations=perturbations,
        ).generate_targeted(originals, targets)

        assert torch.allclose(explanation, torch.tensor([0.4, 0.2]))

    def test__generate_targeted__features_remain_leq_1(self):
        ensemble = Mock()

        def ensemble_call(*args, **kwargs):
            xs = cast(Tensor, args[0])
            xs = cast(Tensor, args[0])
            # We have two inputs and two classes. We connect the logit for class "0" to xs[0], and
            # the logit for class "1" to the xs[1]. We then add 1 to the logit for class "1". Thus,
            # the original ([0., 0.]) will be classified as "1", and the algorithm will try to
            # increment xs[0] as much as possible to change the classification to "0".
            logits = xs + torch.tensor([0.0, 1.0])
            probs = F.softmax(logits, dim=1)
            return {"logits": logits, "probs": probs}

        ensemble.side_effect = ensemble_call

        perturbations = torch.tensor([0.45, 0.1])
        originals = torch.tensor([0.0, 0.0]).reshape(1, -1)
        targets = torch.tensor([0])

        explanation, _, _ = GreedyGenerator(
            ensemble,
            confidence_threshold=0.999,
            n_changes=30,
            max_iters=30,
            num_classes=2,
            perturbations=perturbations,
        ).generate_targeted(originals, targets)

        assert torch.allclose(explanation, torch.tensor([1.0, 0.0]))

    def test__generate_targeted__features_remain_geq_0(self):
        ensemble = Mock()

        def ensemble_call(*args, **kwargs):
            xs = cast(Tensor, args[0])
            xs = cast(Tensor, args[0])
            logits = torch.matmul(xs, torch.tensor([[-1.0, 0.0], [0.0, 0.0]]))
            probs = F.softmax(logits, dim=1)
            return {"logits": logits, "probs": probs}

        ensemble.side_effect = ensemble_call

        perturbations = torch.tensor([0.45, 0.1])
        originals = torch.tensor([1.0, 0.1]).reshape(1, -1)
        targets = torch.tensor([0])

        explanation, _, _ = GreedyGenerator(
            ensemble,
            confidence_threshold=0.999,
            n_changes=30,
            max_iters=30,
            num_classes=2,
            perturbations=perturbations,
        ).generate_targeted(originals, targets)

        assert torch.allclose(explanation, torch.tensor([0.0, 0.1]))
