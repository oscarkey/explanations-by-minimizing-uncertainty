"""Contains generators of counterfactual examples."""
import time
from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from uces.models import Ensemble
import copy
from uces.utils import assert_shape


class CounterfactualGenerator(ABC):
    """Base class for counterfactual generators."""

    @abstractmethod
    def generate_untargeted(self, originals: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Generates counterfactuals for a batch of inputs.

        The counterfactuals are untargeted, so will end up in any class other than the original
        class.

        :param originals: [batch_size, input_size], where input_size is defined by the implementing
                          class, examples for which to generate counterfactuals.
        :returns: (counterfactuals, original predicted labels, counterfactual predicted labels)
        """
        pass

    @abstractmethod
    def generate_targeted(
        self, originals: Tensor, targets: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Generates counterfactuals for a batch of inputs.

        :param originals: [batch_size, input_size], where input_size is defined by the implementing
                          class, examples for which to generate counterfactuals.
        :param targets: [batch_size,], tensor of longs specifying the target class for each c
                        ounterfactual in the batch
        :returns: (counterfactuals, original predicted labels, counterfactual predicted labels)
        """
        pass


class LossFunction(ABC):
    @abstractmethod
    def loss(self, logits: Tensor, original_labels: Tensor) -> Tensor:
        pass


class UntargetedLossFunction(LossFunction):
    """A loss function for generating CEs in any class but the original class.

    It just pushes the generator away from the original class.
    """

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self._n_classes = n_classes

    def loss(self, logits: Tensor, original_labels: Tensor) -> Tensor:
        if self._n_classes == 2:
            # If there are only two classes then it is more likely that the network's predictions
            # will be very accurate, resulting in a loss of ~zero and so a zero gradient. Thus, in
            # this case compute the gradient targeted towards the other class, which is equivalent
            # to an untargeted gradient away from the original class.
            return -F.cross_entropy(logits, self._invert_classes(original_labels))
        else:
            return F.cross_entropy(logits, original_labels)

    @staticmethod
    def _invert_classes(xs: Tensor) -> Tensor:
        return torch.where(xs == torch.ones_like(xs), torch.tensor(0), torch.tensor(1))


class TargetedLossFunction(LossFunction):
    """Loss function for generating CEs in a particular target class."""

    def __init__(self, n_classes: int, targets: Tensor):
        super().__init__()
        assert targets.dtype == torch.long
        assert targets.min() >= 0
        assert targets.max() < n_classes
        self._targets = targets

    def loss(self, logits: Tensor, original_labels: Tensor) -> Tensor:
        batch_size = logits.size(0)
        assert self._targets.size() == (batch_size,)
        return -F.cross_entropy(logits, self._targets)


class GreedyGenerator(CounterfactualGenerator):
    """Generator which uses greedy updates to generate counterfactual examples.

    This is the generator we use to achieve the results in the paper.
    """

    def __init__(
        self,
        ensemble: Ensemble,
        confidence_threshold: float,
        n_changes: int,
        max_iters: int,
        num_classes: int,
        perturbations: Tensor,
    ):
        super().__init__()
        self._ensemble = copy.deepcopy(ensemble)
        self._n_classes = num_classes
        self._confidence_threshold = confidence_threshold
        self._max_iters = max_iters
        self._n_changes = n_changes
        self._perturbations = perturbations

    def generate_untargeted(self, originals: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return self._generate(originals, UntargetedLossFunction(self._n_classes))

    def generate_targeted(
        self, originals: Tensor, targets: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        assert originals.device == targets.device
        return self._generate(originals, TargetedLossFunction(self._n_classes, targets))

    def _generate(
        self, originals: Tensor, loss_function: LossFunction
    ) -> Tuple[Tensor, Tensor, Tensor]:
        start_time = time.time()
        batch_size = originals.size(0)

        examples = originals.clone().detach().view(batch_size, -1)

        self._ensemble.to(examples.device)
        self._ensemble.eval()

        original_labels = self._ensemble(originals)["probs"].argmax(dim=1).detach()
        assert_shape(original_labels, (batch_size,))

        batch_perturbations = (
            self._perturbations.to(examples.device).unsqueeze(0).repeat(batch_size, 1)
        )

        altered_pixels = torch.zeros(size=examples.shape, device=examples.device, dtype=torch.int)
        i = 0
        for i in range(self._max_iters):
            prediction, confidence, grad = self._get_prediction_and_grad(
                self._ensemble, loss_function, examples, original_labels
            )

            have_changed_class = torch.argmax(prediction, -1) != original_labels
            if torch.sum(have_changed_class) == batch_size:
                break

            # If we have already changed a pixel n_changes times, set the gradient to zero so we
            # don't change it again.
            grad[altered_pixels >= self._n_changes] = 0.0

            # We want to change the pixel with the largest gradient, which is the most sensitive.
            max_mask = grad.abs() == grad.abs().max(dim=1, keepdim=True)[0]
            have_changed_class_mask = have_changed_class.view(batch_size, 1).repeat(
                1, max_mask.size(1)
            )
            confidence_mask = (
                (confidence < self._confidence_threshold)
                .view(batch_size, 1)
                .repeat(1, max_mask.size(1))
            )
            # Change the pixel with the largest gradient, if it is part of an example which either
            # hasn't changed class, or the class prediction is not >=95%.
            to_change_mask = (
                max_mask & (~have_changed_class_mask | confidence_mask) & (grad != 0.0)
            )
            grad_sign = grad[to_change_mask].sign()
            perturbation_size = batch_perturbations[to_change_mask]
            examples[to_change_mask] += perturbation_size * grad_sign
            altered_pixels[to_change_mask] += 1

            examples = torch.clamp(examples, 0.0, 1.0)

        end_time = time.time()
        print(
            f"Generated {batch_size} counterfactuals in {end_time - start_time:.2f}s "
            f"over {i + 1} iterations"
        )

        counterfactual_labels = self._ensemble(examples)["logits"].argmax(1)

        return examples.view(originals.size()), original_labels, counterfactual_labels

    @staticmethod
    def _get_prediction_and_grad(
        ensemble: Ensemble, loss_function: LossFunction, examples: Tensor, original_labels: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        examples = examples.clone().detach()
        assert examples.grad is None
        examples.requires_grad = True
        output = ensemble(examples)
        loss = loss_function.loss(output["logits"], original_labels)
        loss.backward()
        confidence, _ = torch.max(output["probs"], dim=1)
        return output["probs"], confidence, examples.grad.clone()
