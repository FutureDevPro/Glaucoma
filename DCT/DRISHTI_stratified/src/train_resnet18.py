import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.models import ResNet18_Weights, resnet18

from dataset import train_loader, validation_loader, test_loader

# ============================================================
# Configuration
# ============================================================
NUM_CLASSES = 2
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
BATCH_SIZE = 8
N_SPLITS = 5
RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

OUTPUT_ROOT = Path(r"C:\Users\udit2\Desktop\Glaucoma\DCT\DRISHTI_stratified\src")
MODELS_DIR = OUTPUT_ROOT / "model"
RESULTS_DIR = OUTPUT_ROOT / "result"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def labels_from_dataset(dataset):
    """Read labels from ImageFolder-like datasets, including nested subsets."""
    if isinstance(dataset, Subset):
        parent_labels = labels_from_dataset(dataset.dataset)
        return [parent_labels[i] for i in dataset.indices]

    if isinstance(dataset, ConcatDataset):
        combined = []
        for child in dataset.datasets:
            combined.extend(labels_from_dataset(child))
        return combined

    if hasattr(dataset, "targets"):
        return [int(label) for label in dataset.targets]

    if hasattr(dataset, "samples"):
        return [int(label) for _, label in dataset.samples]

    raise TypeError(
        f"Cannot infer labels from dataset type {type(dataset).__name__}. "
        "The dataset must expose .targets or .samples, or be a Subset/ConcatDataset."
    )


def make_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels, all_predictions, all_probabilities = [], [], []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        probabilities = torch.softmax(outputs, dim=1)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_predictions.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
        all_probabilities.extend(probabilities.cpu().numpy().tolist())

    total = len(all_labels)
    if total == 0:
        raise RuntimeError("Evaluation loader is empty.")

    return {
        "loss": running_loss / total,
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision_score(all_labels, all_predictions, zero_division=0),
        "recall": recall_score(all_labels, all_predictions, zero_division=0),
        "f1": f1_score(all_labels, all_predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_predictions, labels=[0, 1]),
        "labels": all_labels,
        "predictions": all_predictions,
        "probabilities": all_probabilities,
    }


def image_paths_from_dataset(dataset):
    """Return image paths in dataset order, including nested Subset/ConcatDataset."""
    if isinstance(dataset, Subset):
        parent_paths = image_paths_from_dataset(dataset.dataset)
        return [parent_paths[i] for i in dataset.indices]

    if isinstance(dataset, ConcatDataset):
        paths = []
        for child in dataset.datasets:
            paths.extend(image_paths_from_dataset(child))
        return paths

    if hasattr(dataset, "samples"):
        return [str(path) for path, _ in dataset.samples]

    # Some custom datasets expose paths through a common attribute.
    for attr in ("image_paths", "paths", "files"):
        if hasattr(dataset, attr):
            return [str(path) for path in getattr(dataset, attr)]

    # Fallback: dataset may expose paths via .imgs (ImageFolder-compatible).
    if hasattr(dataset, "imgs"):
        return [str(path) for path, _ in dataset.imgs]

    raise TypeError(
        f"Cannot infer image paths from dataset type {type(dataset).__name__}. "
        "Expose .samples, .imgs, .image_paths, .paths, or .files."
    )


def save_prediction_csv(labels, predictions, probabilities, image_paths, filename):
    """Save image-wise ground truth, predicted class, and class confidence probabilities."""
    with (RESULTS_DIR / filename).open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "image_path", "true_label", "true_class", "predicted_label",
            "predicted_class", "normal_probability", "glaucoma_probability",
            "confidence_probability", "correct"
        ])
        for path, true_label, prediction, probs in zip(
            image_paths, labels, predictions, probabilities
        ):
            writer.writerow([
                str(path),
                int(true_label),
                "Glaucoma" if int(true_label) == 1 else "Normal",
                int(prediction),
                "Glaucoma" if int(prediction) == 1 else "Normal",
                f"{float(probs[0]):.8f}",
                f"{float(probs[1]):.8f}",
                f"{float(max(probs)):.8f}",
                int(true_label) == int(prediction),
            ])


def save_confusion_matrix_csv(labels, predictions, filename):
    """Write the confusion matrix to a CSV file with named rows and columns."""
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    with (RESULTS_DIR / filename).open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["true_label", "predicted_normal", "predicted_glaucoma"])
        writer.writerow(["Normal", int(cm[0, 0]), int(cm[0, 1])])
        writer.writerow(["Glaucoma", int(cm[1, 0]), int(cm[1, 1])])


def save_confusion_matrix_plot(labels, predictions, filename, title):
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    from sklearn.metrics import ConfusionMatrixDisplay
    display = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Glaucoma"])
    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=200)
    plt.close(fig)


def save_fold_plots(history, fold):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history["train_loss"], label="Training loss")
    ax.plot(epochs, history["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"DCT ResNet18 Fold {fold}: Loss")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"fold_{fold}_loss.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history["train_accuracy"], label="Training accuracy")
    ax.plot(epochs, history["val_accuracy"], label="Validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"DCT ResNet18 Fold {fold}: Accuracy")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"fold_{fold}_accuracy.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history["val_f1"], label="Validation F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 score")
    ax.set_title(f"DCT ResNet18 Fold {fold}: Validation F1")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"fold_{fold}_f1.png", dpi=200)
    plt.close(fig)


def main():
    # Pool the original training and validation splits, then create new folds.
    cv_dataset = ConcatDataset([train_loader.dataset, validation_loader.dataset])
    labels = np.asarray(labels_from_dataset(cv_dataset), dtype=np.int64)
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)

    print("=" * 72)
    print("DCT RESNET18: STRATIFIED K-FOLD CROSS-VALIDATION")
    print("Training split images:", len(train_loader.dataset))
    print("Validation split images:", len(validation_loader.dataset))
    print("Total images used in CV:", len(cv_dataset))
    print("Class counts (label 0, label 1):", class_counts.tolist())
    print(f"Folds: {N_SPLITS} | Epochs per fold: {NUM_EPOCHS}")
    print("=" * 72)

    if np.any(class_counts < N_SPLITS):
        raise ValueError(
            f"Each class needs at least {N_SPLITS} images for stratified {N_SPLITS}-fold CV. "
            f"Current class counts (0, 1): {class_counts.tolist()}"
        )

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []
    all_oof_labels, all_oof_predictions, all_oof_probabilities = [], [], []
    all_oof_image_paths = []
    cv_image_paths = image_paths_from_dataset(cv_dataset)

    for fold, (train_indices, val_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels), start=1
    ):
        print(f"\n{'=' * 24} FOLD {fold}/{N_SPLITS} {'=' * 24}")
        set_seed(RANDOM_STATE + fold)

        fold_train_loader = DataLoader(
            Subset(cv_dataset, train_indices.tolist()),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )
        fold_val_loader = DataLoader(
            Subset(cv_dataset, val_indices.tolist()),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        model = make_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        best_f1 = -1.0
        best_epoch = 0
        best_path = MODELS_DIR / f"resnet18_dct_fold_{fold}_best.pth"
        history = {
            "train_loss": [], "train_accuracy": [],
            "val_loss": [], "val_accuracy": [], "val_f1": [],
        }

        for epoch in range(NUM_EPOCHS):
            train_loss, train_accuracy = train_one_epoch(
                model, fold_train_loader, criterion, optimizer
            )
            val = evaluate(model, fold_val_loader, criterion)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            history["val_loss"].append(val["loss"])
            history["val_accuracy"].append(val["accuracy"])
            history["val_f1"].append(val["f1"])

            print(
                f"Fold {fold} | Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] "
                f"| Train loss: {train_loss:.4f} | Train acc: {train_accuracy:.4f} "
                f"| Val loss: {val['loss']:.4f} | Val acc: {val['accuracy']:.4f} "
                f"| Val F1: {val['f1']:.4f}"
            )

            if val["f1"] > best_f1:
                best_f1 = val["f1"]
                best_epoch = epoch + 1
                torch.save(
                    {
                        "fold": fold,
                        "epoch": best_epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_f1": best_f1,
                        "representation": "per-channel signed-log DCT",
                        "class_to_idx": {"normal": 0, "glaucoma": 1},
                        "train_indices": train_indices.tolist(),
                        "validation_indices": val_indices.tolist(),
                    },
                    best_path,
                )

        # Load the best checkpoint for this fold and compute held-out fold metrics.
        checkpoint = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_val = evaluate(model, fold_val_loader, criterion)
        all_oof_labels.extend(best_val["labels"])
        all_oof_predictions.extend(best_val["predictions"])
        all_oof_probabilities.extend(best_val["probabilities"])

        fold_val_paths = [cv_image_paths[i] for i in val_indices]
        all_oof_image_paths.extend(fold_val_paths)
        save_prediction_csv(
            best_val["labels"], best_val["predictions"], best_val["probabilities"],
            fold_val_paths, f"fold_{fold}_validation_image_probabilities.csv"
        )
        save_confusion_matrix_csv(
            best_val["labels"], best_val["predictions"],
            f"fold_{fold}_validation_confusion_matrix.csv"
        )
        save_confusion_matrix_plot(
            best_val["labels"], best_val["predictions"],
            f"fold_{fold}_validation_confusion_matrix.png",
            f"DCT ResNet18 Fold {fold} Validation"
        )

        fold_results.append({
            "fold": fold,
            "best_epoch": best_epoch,
            "train_images": len(train_indices),
            "validation_images": len(val_indices),
            "loss": best_val["loss"],
            "accuracy": best_val["accuracy"],
            "precision": best_val["precision"],
            "recall": best_val["recall"],
            "f1": best_val["f1"],
            "checkpoint": str(best_path),
        })

        print(f"\nBest checkpoint for fold {fold}: {best_path}")
        print("Fold confusion matrix (rows=true [normal, glaucoma], columns=predicted):")
        print(best_val["confusion_matrix"])
        save_fold_plots(history, fold)

    # Save one row per fold.
    metrics_csv = RESULTS_DIR / "resnet18_dct_kfold_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fold_results[0].keys()))
        writer.writeheader()
        writer.writerows(fold_results)

    print("\n" + "=" * 72)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 72)
    for metric in ("accuracy", "precision", "recall", "f1", "loss"):
        values = np.asarray([row[metric] for row in fold_results], dtype=float)
        print(f"{metric.capitalize():<10}: {values.mean():.4f} +/- {values.std(ddof=1):.4f}")

    pooled_cm = confusion_matrix(all_oof_labels, all_oof_predictions, labels=[0, 1])
    save_prediction_csv(
        all_oof_labels, all_oof_predictions, all_oof_probabilities,
        all_oof_image_paths, "all_folds_out_of_fold_image_probabilities.csv"
    )
    save_confusion_matrix_csv(
        all_oof_labels, all_oof_predictions, "all_folds_pooled_confusion_matrix.csv"
    )
    save_confusion_matrix_plot(
        all_oof_labels, all_oof_predictions, "all_folds_pooled_confusion_matrix.png",
        "DCT ResNet18: Pooled Out-of-Fold Predictions"
    )
    print("\nPooled out-of-fold confusion matrix:")
    print(pooled_cm)

    # Debugging: each image in the CV pool should appear in exactly one held-out fold.
    print("\nCross-validation debugging")
    print("-" * 50)
    print("Images available for CV:", len(cv_dataset))
    print("Out-of-fold labels:", len(all_oof_labels))
    print("Out-of-fold predictions:", len(all_oof_predictions))
    print("Pooled confusion matrix total:", int(pooled_cm.sum()))
    print("Sum of validation images across folds:", sum(r["validation_images"] for r in fold_results))
    if len(cv_dataset) == len(all_oof_labels) == len(all_oof_predictions) == int(pooled_cm.sum()):
        print("PASS: every CV image was evaluated once in its held-out fold.")
    else:
        print("WARNING: counts differ; investigate dataset splitting or evaluation.")

    print(f"\nFold metrics CSV: {metrics_csv}")
    print(f"Models saved in: {MODELS_DIR}")
    print(f"Results, probability CSVs, confusion matrices, and plots saved in: {RESULTS_DIR}")
    print("The independent test_loader was not used for selecting the best fold models.")
    print("To keep the test set untouched, test evaluation is intentionally not run in this script.")


if __name__ == "__main__":
    main()
