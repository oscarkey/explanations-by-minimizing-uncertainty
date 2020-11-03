"""Contains tests for uces.tabular.

Run using pytest.
"""
from tempfile import NamedTemporaryFile
from typing import Tuple

import torch
from torch import Tensor
from uces.tabular import TabularDataset

_TEST_CONFIG = """ignore_columns:input b
range input a:0,5
range input c:0,400
range output:0,1
perturbation input a:1
perturbation input c:10
output_columns:output
"""

_TEST_CSV = """input a,input b,input c,output
1,11,100,0
2,12,200,1
3,13,300,0
3.9,14,399,1
"""


class TestTabularDataset:
    def setup_method(self, method):
        self._config_file = NamedTemporaryFile(mode="w")
        self._config_file.write(_TEST_CONFIG)
        self._config_file.flush()
        self._csv_file = NamedTemporaryFile(mode="w")
        self._csv_file.write(_TEST_CSV)
        self._csv_file.flush()

    def teardown_method(self, method):
        self._config_file.close()
        self._csv_file.close()

    def test__constructor__5050_train_test_split__reserves_two_rows_for_train_and_test(self):
        train_dataset = TabularDataset(
            self._csv_file.name,
            self._config_file.name,
            split="train",
            val_fraction=0.0,
            test_fraction=0.5,
        )
        test_dataset = TabularDataset(
            self._csv_file.name,
            self._config_file.name,
            split="test",
            val_fraction=0.0,
            test_fraction=0.5,
        )

        assert len(train_dataset) == 2
        assert len(test_dataset) == 2

        assert not torch.allclose(train_dataset[0][0], test_dataset[0][0])

    def test__constructor__502525_train_val_test_split__reserves_two_one_one_rows(self):
        train_dataset = TabularDataset(
            self._csv_file.name,
            self._config_file.name,
            split="train",
            val_fraction=0.25,
            test_fraction=0.25,
        )
        val_dataset = TabularDataset(
            self._csv_file.name,
            self._config_file.name,
            split="val",
            val_fraction=0.25,
            test_fraction=0.25,
        )
        test_dataset = TabularDataset(
            self._csv_file.name,
            self._config_file.name,
            split="test",
            val_fraction=0.25,
            test_fraction=0.25,
        )

        assert len(train_dataset) == 2
        assert len(val_dataset) == 1
        assert len(test_dataset) == 1

        assert not torch.allclose(train_dataset[0][0], val_dataset[0][0])
        assert not torch.allclose(train_dataset[0][0], test_dataset[0][0])
        assert not torch.allclose(val_dataset[0][0], test_dataset[0][0])

    def test__constructor__constant_split_index__reserves_same_rows_for_each_split(self):
        # Repeat this several times to reduce the probability we randomly see the same rows.
        for split in ["train", "val", "test"]:
            original_inputs, _ = TabularDataset(
                self._csv_file.name,
                self._config_file.name,
                split=split,
                val_fraction=0.25,
                test_fraction=0.25,
                split_index=1,
            )[0]
            for _ in range(20):
                test_inputs, _ = TabularDataset(
                    self._csv_file.name,
                    self._config_file.name,
                    split=split,
                    val_fraction=0.25,
                    test_fraction=0.25,
                    split_index=1,
                )[0]
                assert torch.allclose(original_inputs, test_inputs)

    def test__constructor__different_split_indices__sometimes_reserves_different_rows_for_testing(
        self,
    ):
        # Repeat this several times because for some splits we will randomly get the same rows.
        # This test is not flakey because all randomness is determined by the split value, which is
        # deterministic.
        for split in ["train", "val", "test"]:
            original_inputs, _ = TabularDataset(
                self._csv_file.name,
                self._config_file.name,
                split=split,
                val_fraction=0.25,
                test_fraction=0.25,
                split_index=0,
            )[0]
            found_different = False
            for split_index in range(1, 20):
                test_inputs, _ = TabularDataset(
                    self._csv_file.name,
                    self._config_file.name,
                    split=split,
                    val_fraction=0.25,
                    test_fraction=0.25,
                    split_index=split_index,
                )[0]
                found_different = found_different or not torch.allclose(
                    original_inputs, test_inputs
                )
            assert found_different

    def test__len__returns_length(self):
        dataset = TabularDataset(
            self._csv_file.name,
            self._config_file.name,
            split="train",
            val_fraction=0.0,
            test_fraction=0.0,
        )
        assert len(dataset) == 4

    def test__index__returns_normalised_rows(self):
        dataset = TabularDataset(
            self._csv_file.name,
            self._config_file.name,
            split="train",
            val_fraction=0.0,
            test_fraction=0.0,
        )

        inputs0, labels0 = dataset[0]
        assert torch.allclose(inputs0, torch.tensor([0.2, 0.25]))
        assert torch.allclose(labels0, torch.tensor([0.0]))

        inputs1, labels1 = dataset[1]
        assert torch.allclose(inputs1, torch.tensor([0.4, 0.50]))
        assert torch.allclose(labels1, torch.tensor([1.0]))

    def test__index_twice__returns_same_row_each_time(self):
        dataset = TabularDataset(
            self._csv_file.name,
            self._config_file.name,
            split="train",
            val_fraction=0.0,
            test_fraction=0.0,
        )

        inputs0, labels0 = dataset[0]
        inputs1, labels1 = dataset[0]

        assert torch.allclose(inputs0, torch.tensor([0.2, 0.25]))
        assert torch.allclose(labels0, torch.tensor([0.0]))
        assert torch.allclose(inputs1, torch.tensor([0.2, 0.25]))
        assert torch.allclose(labels1, torch.tensor([0.0]))

    def test__index__modify_row__does_not_modify_underlying_data(self):
        dataset = TabularDataset(
            self._csv_file.name,
            self._config_file.name,
            split="train",
            val_fraction=0.0,
            test_fraction=0.0,
        )

        inputs0, labels0 = dataset[0]
        inputs0 /= 2.0
        labels0 /= 2.0
        inputs1, labels1 = dataset[0]

        assert torch.allclose(inputs1, torch.tensor([0.2, 0.25]))
        assert torch.allclose(labels1, torch.tensor([0.0]))

    def test__index__has_transform__applies_transform(self):
        def transform(inputs: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
            return inputs * 2, targets * 3

        dataset = TabularDataset(
            self._csv_file.name,
            self._config_file.name,
            split="train",
            val_fraction=0.0,
            test_fraction=0.0,
            transform=transform,
        )

        inputs0, labels0 = dataset[0]
        assert torch.allclose(inputs0, torch.tensor([0.4, 0.50]))
        assert torch.allclose(labels0, torch.tensor([0.0]))

        inputs1, labels1 = dataset[1]
        assert torch.allclose(inputs1, torch.tensor([0.8, 1.0]))
        assert torch.allclose(labels1, torch.tensor([3.0]))

    def test__get_unnormalized__returns_unnormalized_rows(self):
        dataset = TabularDataset(
            self._csv_file.name,
            self._config_file.name,
            split="train",
            val_fraction=0.0,
            test_fraction=0.0,
        )

        inputs0, labels0 = dataset.get_unnormalized(0)
        assert torch.allclose(inputs0, torch.tensor([1.0, 100.0]))
        assert torch.allclose(labels0, torch.tensor([0.0]))

        inputs1, labels1 = dataset.get_unnormalized(1)
        assert torch.allclose(inputs1, torch.tensor([2.0, 200.0]))
        assert torch.allclose(labels1, torch.tensor([1.0]))

    def test__get_unnormalized__modify_row__does_not_modify_underlying_data(self):
        dataset = TabularDataset(
            self._csv_file.name,
            self._config_file.name,
            split="train",
            val_fraction=0.0,
            test_fraction=0.0,
        )

        inputs0, labels0 = dataset.get_unnormalized(0)
        inputs0 /= 2.0
        labels0 /= 2.0
        inputs1, labels1 = dataset.get_unnormalized(0)

        assert torch.allclose(inputs1, torch.tensor([1.0, 100.0]))
        assert torch.allclose(labels1, torch.tensor([0.0]))

    def test__index__has_transform__does_not_apply_transform(self):
        def transform(inputs: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
            return inputs * 2, targets * 3

        dataset = TabularDataset(
            self._csv_file.name,
            self._config_file.name,
            split="train",
            val_fraction=0.0,
            test_fraction=0.0,
            transform=transform,
        )

        inputs0, labels0 = dataset.get_unnormalized(0)
        assert torch.allclose(inputs0, torch.tensor([1.0, 100.0]))
        assert torch.allclose(labels0, torch.tensor([0.0]))

        inputs1, labels1 = dataset.get_unnormalized(1)
        assert torch.allclose(inputs1, torch.tensor([2.0, 200.0]))
        assert torch.allclose(labels1, torch.tensor([1.0]))

    def test__column_names__returns_input_and_output_columns(self):
        dataset = TabularDataset(
            self._csv_file.name,
            self._config_file.name,
            split="train",
            val_fraction=0.0,
            test_fraction=0.0,
        )
        input_columns, output_columns = dataset.column_names
        assert input_columns == ["input a", "input c"]
        assert output_columns == ["output"]

    def test__perturbations__returns_perturbations_for_input_columns_normalized(self):
        dataset = TabularDataset(
            self._csv_file.name,
            self._config_file.name,
            split="train",
            val_fraction=0.0,
            test_fraction=0.0,
        )
        assert torch.allclose(dataset.perturbations, torch.tensor([0.2, 0.025]))

    def test__unnormalize__returns_unnormalized_values(self):
        dataset = TabularDataset(
            self._csv_file.name,
            self._config_file.name,
            split="train",
            val_fraction=0.0,
            test_fraction=0.0,
        )
        inputs, outputs = dataset[0]
        unnormalized_inputs = dataset.unnormalize(inputs)
        assert torch.allclose(unnormalized_inputs, torch.tensor([1.0, 100.0]))

    def test__targets__returns_normalized_and_transformed_tagets(self):
        def transform(inputs: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
            return inputs * 2, targets * 3

        dataset = TabularDataset(
            self._csv_file.name,
            self._config_file.name,
            split="train",
            val_fraction=0.0,
            test_fraction=0.0,
            transform=transform,
        )

        targets = dataset.targets
        assert torch.allclose(targets, torch.tensor([0.0, 3.0, 0.0, 3.0]))
