from typing import Tuple, Union

import torch
from numpy import ndarray
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset


def assert_shape(x: Union[Tensor, ndarray], shape):
    """Asserts that the given Tensor or ndarray has the given shape."""
    assert x.shape == shape, f"Wanted shape {shape} but had {x.shape}"


class AssertShape(Module):
    def __init__(self, shape_without_batch: Tuple):
        super().__init__()
        self._shape_without_batch = shape_without_batch

    def forward(self, x: Tensor) -> Tensor:
        assert (
            x.shape[1:] == self._shape_without_batch
        ), f"Wanted shape {self._shape_without_batch}, got {x.shape[1:]}."
        return x


def get_inputs_and_targets(
    dataset: Dataset, device: torch.device = torch.device("cuda")
) -> Tuple[Tensor, Tensor]:
    """Iterates over the data set and returns a pair of all the inputs and all the targets."""
    xs, cs = zip(*[(x, c) for x, c in dataset])
    if isinstance(cs[0], Tensor):
        cs_tensor = torch.stack(cs, axis=0)
    else:
        cs_tensor = torch.tensor(cs).view(-1, 1)
    return torch.stack(xs, axis=0).to(device), cs_tensor.to(device)
