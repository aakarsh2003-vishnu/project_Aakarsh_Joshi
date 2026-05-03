# train.py
# Main training file for the vehicle classifier.
# The train_model signature is kept simple because the grader imports it.

import json
import os

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import config


def train_model(model, num_epochs, train_loader, loss_fn, optimizer):
    """Train the vehicle classifier and save the final weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    history = {"loss": [], "accuracy": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{num_epochs}]", leave=True)
        for batch_imgs, batch_labels in loop:
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_imgs)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)
            loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.3f}")

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_acc)
        print(f"Epoch {epoch:02d}: loss={epoch_loss:.4f}, accuracy={epoch_acc:.4f} - train.py:50")

        torch.save(model.state_dict(), config.WEIGHTS_PATH)

    print(f"Training complete. Weights saved to {config.WEIGHTS_PATH} - train.py:54")
    return history


def evaluate_model(model, data_loader, loss_fn):
    """Evaluate classification accuracy and confusion matrix."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    total_loss = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros(config.num_classes, config.num_classes, dtype=torch.int64)

    with torch.no_grad():
        for batch_imgs, batch_labels in data_loader:
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_imgs)
            loss = loss_fn(logits, batch_labels)

            total_loss += loss.item() * batch_imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)

            for true_label, pred_label in zip(batch_labels.cpu(), preds.cpu()):
                confusion[true_label, pred_label] += 1

    per_class = {}
    for idx, name in enumerate(config.class_names):
        class_total = int(confusion[idx].sum().item())
        class_correct = int(confusion[idx, idx].item())
        per_class[name] = class_correct / class_total if class_total else 0.0

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "total": total,
        "per_class_accuracy": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def make_stratified_split(dataset):
    """Make an 80/20 split while keeping each class balanced."""
    train_indices = []
    test_indices = []

    for class_idx in range(config.num_classes):
        indices = [i for i, sample in enumerate(dataset.samples) if sample[1] == class_idx]
        generator = torch.Generator().manual_seed(config.random_seed + class_idx)
        order = torch.randperm(len(indices), generator=generator).tolist()
        shuffled = [indices[i] for i in order]
        split_at = int(len(shuffled) * config.train_split)
        train_indices.extend(shuffled[:split_at])
        test_indices.extend(shuffled[split_at:])

    return train_indices, test_indices


if __name__ == "__main__":
    import torch.nn as nn
    from torch.optim import Adam

    from dataset import VehicleDataset
    from model import VehicleClassifier

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    full_dataset = VehicleDataset(root_dir=config.DATA_DIR, augment=True)
    train_indices, test_indices = make_stratified_split(full_dataset)

    train_loader = DataLoader(
        Subset(full_dataset, train_indices),
        batch_size=config.batchsize,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_dataset = VehicleDataset(root_dir=config.DATA_DIR, augment=False)
    test_loader = DataLoader(
        Subset(test_dataset, test_indices),
        batch_size=config.batchsize,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = VehicleClassifier()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.lr)

    history = train_model(model, config.epochs, train_loader, loss_fn, optimizer)
    test_metrics = evaluate_model(model, test_loader, loss_fn)

    output = {
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "epochs": config.epochs,
        "batch_size": config.batchsize,
        "learning_rate": config.lr,
        "history": history,
        "test_metrics": test_metrics,
    }

    with open(config.TRAIN_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Test accuracy: {test_metrics['accuracy']:.4f} - train.py:164")
    print(f"Training history saved to {config.TRAIN_HISTORY_PATH} - train.py:165")
