"""Implementation of 'Interpretable Counterfactual Explanations Guided by Prototypes'

Based on the original paper authored by Arnaud Van Looveren and Janis Klaise, and available at
https://arxiv.org/abs/1907.02584

We have used the implementation of the method available in the library ALIBI
(https://github.com/SeldonIO/alibi), with the configuration matching that in the original paper as
far as possible. See the appendix of our paper for details.
"""

import os
from typing import Optional

import numpy as np
import tensorflow as tf
import torch
import uces.utils
from alibi.explainers.cfproto import CounterFactualProto
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    UpSampling2D,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from torch import Tensor
from uces.datasets import SupportedDataset

from .methods import CEMethod

# ALIBI is not compatible with v2 behaviour, thus disable it.
tf.compat.v1.disable_v2_behavior()

print("TF version: ", tf.__version__)
print("Eager execution enabled: ", tf.executing_eagerly())  # False


class PrototypesMethod(CEMethod):
    def __init__(self, k: Optional[int] = None, theta: Optional[int] = None) -> None:
        """Constructs a new instance.

        If k and theta are None, will use the default values for the dataset.
        """
        super().__init__()
        self.k = k
        self.theta = theta

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
        if dataset == "mnist" or dataset.startswith("simulatedmnist"):
            dataset_type = "mnist"
        elif dataset == "breastcancer" or dataset.startswith("simulatedbc"):
            dataset_type = "breastcancer"
        elif dataset == "bostonhousing":
            dataset_type = "bostonhousing"
        else:
            raise ValueError

        x_train, y_train = uces.utils.get_inputs_and_targets(
            train_dataset, device=originals.device
        )
        x_train = x_train.detach().cpu().numpy()
        y_train = y_train.detach().cpu().numpy()
        x_test = originals.detach().cpu().numpy()

        if dataset_type == "mnist":
            x_train = np.reshape(x_train, (-1, 28, 28, 1))
            x_test = np.reshape(x_test, (-1, 28, 28, 1))
        y_train = to_categorical(y_train)

        tabular_train_mean = np.mean(x_train, axis=0)
        tabular_train_std = np.std(x_train, axis=0)
        if dataset_type == "mnist":
            x_train = ((x_train - x_train.min()) / (x_train.max() - x_train.min())) - 0.5
            x_test = ((x_test - x_test.min()) / (x_test.max() - x_test.min())) - 0.5
        else:
            x_train = (x_train - tabular_train_mean) / tabular_train_std
            x_test = (x_test - tabular_train_mean) / tabular_train_std

        base_path = os.path.join(results_dir, self.name)
        if not os.path.exists(base_path):
            os.mkdir(base_path)

        classifier_path = os.path.join(results_dir, self.name, f"{dataset}_classifier.h5")
        if os.path.exists(classifier_path):
            classifier = load_model(classifier_path)
        else:
            if dataset_type == "mnist":
                classifier = _mnist_cnn_model()
                classifier.fit(x_train, y_train, batch_size=64, epochs=3, verbose=1)
            elif dataset_type == "breastcancer":
                classifier = _bc_classifier_model()
                classifier.fit(x_train, y_train, batch_size=128, epochs=500, verbose=1)
            elif dataset_type == "bostonhousing":
                classifier = _bh_classifier_model()
                classifier.fit(x_train, y_train, batch_size=128, epochs=500, verbose=1)
            else:
                raise ValueError
            classifier.save(classifier_path, save_format="h5")

        if dataset_type == "mnist":
            ae_path = os.path.join(results_dir, self.name, f"{dataset}_ae.h5")
            enc_path = os.path.join(results_dir, self.name, f"{dataset}_enc.h5")
            if os.path.exists(ae_path) and os.path.exists(enc_path):
                ae = load_model(ae_path)
                enc = load_model(enc_path, compile=False)
            else:
                ae, enc, dec = _mnist_ae_model()
                ae.fit(
                    x_train,
                    x_train,
                    batch_size=128,
                    epochs=4,
                    validation_data=(x_test, x_test),
                    verbose=1,
                )
                ae.save(ae_path, save_format="h5")
                enc.save(enc_path, save_format="h5")

            cf = CounterFactualProto(
                classifier,
                shape=(1,) + x_train.shape[1:],
                gamma=100.0,
                theta=100.0 if self.theta is None else self.theta,
                ae_model=ae,
                enc_model=enc,
                max_iterations=2000,
                feature_range=(x_train.min(), x_train.max()),
                c_init=1.0,
                c_steps=1,
            )
            cf.fit(x_train)
        else:
            # For breastcancer the hyperparameters are taken from the prototypes paper.
            # For bostonhousing, we used the same hyperaparameters as for breastcancer, except for
            # k and theta which we found by grid search (k is set below).
            cf = CounterFactualProto(
                classifier,
                use_kdtree=True,
                shape=(1,) + x_train.shape[1:],
                theta=100.0 if self.theta is None else self.theta,
                max_iterations=2000,
                feature_range=(x_train.min(axis=0), x_train.max(axis=0)),
                c_init=1.0,
                c_steps=1,
            )
            cf.fit(x_train)

        if self.k is None:
            if dataset == "bostonhousing" and self.k is None:
                k: Optional[int] = 10
            else:
                # Setting k=None uses the default value appropriate for the encoder/kd-tree.
                k = None
        else:
            k = self.k

        counterfactuals_list = []
        for j in range(targets.size(0)):
            X = x_test[j].reshape((1,) + x_test[0].shape)
            explanation = cf.explain(X, target_class=[targets[0].item()], k=k, verbose=True,)
            if explanation.cf is None:
                print(f"Failed to find CE for original {j}")
                counterfactuals_list.append(torch.tensor(cf.sess.run(cf.adv)))
            else:
                print("Counterfactual prediction: {}".format(explanation.cf["class"]))
                print("Closest prototype class: {}".format(explanation.id_proto))
                counterfactuals_list.append(torch.tensor(explanation.cf["X"]))

        # Prototypes uses different normalization to our code, so we need to undo this.
        if dataset_type == "mnist":
            # We generate CEs normalized between -0.5 and 0.5, but need to return them between 0 and 1.
            counterfactuals_normalized = torch.cat(counterfactuals_list)
            counterfactuals_renormalized = counterfactuals_normalized + 0.5
        else:
            # We generate CEs standardized to mean 0 std dev 1. We need to undo this.
            counterfactuals_normalized = torch.cat(counterfactuals_list)
            counterfactuals_renormalized = (
                counterfactuals_normalized * tabular_train_std + tabular_train_mean
            )

        return counterfactuals_renormalized.view(originals.size())

    @property
    def name(self) -> str:
        return f"alibi_prototypes2_k{self.k}_theta{self.theta}"

    @property
    def use_adversarial_training(self) -> bool:
        return False


def _mnist_cnn_model():
    x_in = Input(shape=(28, 28, 1))
    x = Conv2D(filters=32, kernel_size=2, padding="same", activation="relu")(x_in)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(filters=64, kernel_size=2, padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x_out = Dense(10, activation="softmax")(x)

    cnn = Model(inputs=x_in, outputs=x_out)
    cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return cnn


def _mnist_ae_model():
    x_in = Input(shape=(28, 28, 1))
    x = Conv2D(16, (3, 3), activation="relu", padding="same")(x_in)
    x = Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    encoded = Conv2D(1, (3, 3), activation=None, padding="same")(x)
    encoder = Model(x_in, encoded)

    dec_in = Input(shape=(14, 14, 1))
    x = Conv2D(16, (3, 3), activation="relu", padding="same")(dec_in)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    decoded = Conv2D(1, (3, 3), activation=None, padding="same")(x)
    decoder = Model(dec_in, decoded)

    x_out = decoder(encoder(x_in))
    autoencoder = Model(x_in, x_out)
    autoencoder.compile(loss="mse", optimizer="adam")

    return autoencoder, encoder, decoder


def _bc_classifier_model():
    x_in = Input(shape=(30,))
    x = Dense(40, activation="relu")(x_in)
    x = Dense(40, activation="relu")(x)
    x_out = Dense(2, activation="softmax")(x)
    classifier = Model(x_in, x_out)
    classifier.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

    return classifier


def _bh_classifier_model():
    x_in = Input(shape=(13,))
    x = Dense(40, activation="relu")(x_in)
    x = Dense(40, activation="relu")(x)
    x_out = Dense(2, activation="softmax")(x)
    classifier = Model(x_in, x_out)
    classifier.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

    return classifier
