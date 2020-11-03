"""Trains an ensemble of models in parallel, and saves them to disk."""
import time
from argparse import ArgumentParser, Namespace

import ray
import uces.datasets
import torch
import torch.nn.functional as F
from uces import models
from uces.models import MLP
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader


def main(
    config_to_save: Namespace,
    results_root: str,
    data_dir: str,
    ensemble_size: int,
    batch_size: int,
    n_hidden: int,
    adversarial_training: bool,
    random_perturbation_training: bool,
    n_epochs: int,
    dataset: str,
) -> None:
    ray.init(include_dashboard=False)
    experiment_path = models.get_fresh_checkpoint_path(results_root)
    models.save_config(experiment_path, config_to_save)

    train_data, val_data, _, n_classes = uces.datasets.load_dataset(
        dataset, data_dir, results_root
    )
    input_size_flat = train_data[0][0].view(-1).size(0)

    if val_data is None:
        raise ValueError("This dataset needs to be updated to support a validation split.")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True)

    if dataset in ("admissions", "breastcancer", "bostonhousing") or dataset.startswith(
        "simulatedbc"
    ):
        # Use 1/2 max perturbation to avoid changing the class.
        perturbations = train_data.perturbations * 0.5
    elif dataset == "mnist" or dataset.startswith("simulatedmnist"):
        perturbations = torch.full((input_size_flat,), 0.15)
    else:
        raise ValueError

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_func = _train_model.options(num_gpus=1.0 if device == "cuda" else 0)
    model_ids = []
    for component_i in range(ensemble_size):
        model_ids.append(
            train_func.remote(
                component_i,
                device,
                experiment_path,
                train_loader,
                adversarial_training,
                random_perturbation_training,
                input_size_flat,
                n_classes,
                n_hidden,
                n_epochs,
                perturbations,
            )
        )
    ray.wait(model_ids)
    for model_id in model_ids:
        print("Validation accuracy: ", _test_model(ray.get(model_id), device, val_loader))


@ray.remote
def _train_model(
    ensemble_i: int,
    device: str,
    results_path: str,
    train_loader: DataLoader,
    adversarial_training: bool,
    random_perturbation_training: bool,
    input_size_flat: int,
    n_classes: int,
    n_hidden: int,
    n_epochs: int,
    perturbations: Tensor,
) -> MLP:
    torch.manual_seed(ensemble_i)

    model = MLP(n_hidden, input_size_flat, n_classes).to(device)
    optimizer = Adam(model.parameters())

    print(f"Training model {ensemble_i}...")
    for epoch in range(n_epochs):
        start_time = time.time()
        loss = _run_epoch(
            model,
            device,
            train_loader,
            optimizer,
            adversarial_training,
            random_perturbation_training,
            perturbations,
        )
        end_time = time.time()
        print(f"Epoch loss {loss:.2f}; Training at {1 / (end_time - start_time):.2f} epoch/s")
    print(f"Training done for model {ensemble_i}")

    models.save_ensemble_model(results_path, model, ensemble_i)
    return model


def _run_epoch(
    model: MLP,
    device: str,
    train_loader: DataLoader,
    optimizer,
    adversarial_training: bool,
    random_perturbation_training: bool,
    perturbations: Tensor,
) -> float:
    model.train()
    perturbations = perturbations.to(device)

    batch_losses = []
    for batch_i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        targets = targets.view(inputs.size(0))

        batch_size = inputs.size(0)
        batch_perturbations = perturbations.unsqueeze(0).repeat(batch_size, 1).view(inputs.size())

        inputs.requires_grad = True

        output = model(inputs)
        loss = F.cross_entropy(output["logits"], targets)
        optimizer.zero_grad()
        loss.backward()

        batch_losses.append(torch.mean(loss).detach())

        if adversarial_training or random_perturbation_training:
            new_images = None
            if adversarial_training:
                new_images = inputs + batch_perturbations * inputs.grad.sign()

            elif random_perturbation_training:
                rand_direction = 2 * torch.randint(0, 1, inputs.shape, device=device) - 1
                direction = batch_perturbations * rand_direction
                new_images = inputs + direction

            new_images = torch.clamp(new_images, 0.0, 1.0)
            new_inputs = torch.cat((inputs, new_images), 0)
            y_new = torch.cat((targets, targets), 0)

            optimizer.zero_grad()
            output_new = model(new_inputs)
            aux_loss = F.cross_entropy(output_new["logits"], y_new)
            aux_loss.backward()

        optimizer.step()

    return torch.stack(batch_losses).mean().item()


def _test_model(model: MLP, device, test_loader: DataLoader) -> float:
    """Returns the accuracy of the model on the test dataset."""
    model.eval()

    n_correct = 0
    n_total = 0

    for batch_i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.view(images.size(0))

        preds_onehot = model(images)["probs"]
        assert preds_onehot.ndim == 2
        preds_labels = torch.argmax(preds_onehot, dim=1)

        n_correct += int(torch.sum(preds_labels == labels))
        n_total += int(labels.size(0))

    return float(n_correct) / n_total


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--adv_training", action="store_true")
    parser.add_argument("--rand_per_training", action="store_true")
    parser.add_argument("--epochs", type=int, nargs="?", default=50)
    parser.add_argument(
        "--ensemble_size",
        type=int,
        nargs="?",
        default=20,
        help="The number of models in the ensemble",
    )
    parser.add_argument("--n_hidden", type=int, nargs="?", default=80)
    parser.add_argument("--batch_size", type=int, nargs="?", default=128)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "admissions", "breastcancer", "bostonhousing",],
        default="admissions",
    )
    args = parser.parse_args()

    main(
        args,
        args.results_dir,
        args.data_dir,
        args.ensemble_size,
        args.batch_size,
        args.n_hidden,
        args.adv_training,
        args.rand_per_training,
        args.epochs,
        args.dataset,
    )
