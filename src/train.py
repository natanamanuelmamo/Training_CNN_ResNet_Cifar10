"""
train.py
========
A single shared training loop used for BOTH models, so that optimizer,
learning-rate schedule, number of epochs, batch size, and loss function are
guaranteed to be identical between the two runs -- the "fair comparison"
the assignment asks for.
"""

import time
import torch
import torch.nn as nn


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """Runs one epoch. If train=True, updates weights; otherwise eval-only."""
    model.train() if train else model.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, device,
                 epochs=30, lr=0.1, momentum=0.9, weight_decay=5e-4,
                 lr_milestones=(15, 25), lr_gamma=0.1, log_fn=print):
    """
    Trains `model` and returns a history dict plus total training time.

    Optimizer: SGD with momentum + weight decay (standard choice for
    CIFAR-style ResNets, as used in the original paper).
    LR schedule: step decay at `lr_milestones`.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                 momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(lr_milestones), gamma=lr_gamma)

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion,
                                           optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion,
                                       optimizer, device, train=False)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        log_fn(f"Epoch {epoch:3d}/{epochs} | "
               f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
               f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    total_time = time.time() - start_time
    return history, total_time


@torch.no_grad()
def evaluate_test(model, test_loader, device):
    """Final test-set evaluation. Returns accuracy and (y_true, y_pred) for
    a confusion matrix."""
    model.eval()
    all_preds, all_labels = [], []
    correct, total = 0, 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    accuracy = correct / total
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    return accuracy, y_true, y_pred
