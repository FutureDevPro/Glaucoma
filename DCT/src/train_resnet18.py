"""
ResNet18 training script using DCT-transformed fundus images.

Run from the folder containing dataset_dct.py:
    python train_resnet18_dct.py

This keeps the training/evaluation structure of the original script.
The test loader combines ACRIMA and DRISHTI, matching the original
dataset.py design; it is NOT a pure cross-dataset evaluation.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from dataset import train_loader, validation_loader, test_loader


# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 2
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

models_dir = Path(r"C:\Users\udit2\Desktop\Glaucoma\DCT\models_dct")
models_dir.mkdir(parents=True, exist_ok=True)
best_model_path = models_dir / "resnet18_dct_best.pth"

results_dir = Path(r"C:\Users\udit2\Desktop\Glaucoma\DCT\results_dct")
results_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# Model
# ============================================================

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)


# ============================================================
# Train and validation functions
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())

    total = len(all_labels)
    if total == 0:
        raise RuntimeError("Evaluation loader is empty.")

    metrics = {
        "loss": running_loss / total,
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision_score(all_labels, all_predictions, zero_division=0),
        "recall": recall_score(all_labels, all_predictions, zero_division=0),
        "f1": f1_score(all_labels, all_predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(
            all_labels, all_predictions, labels=[0, 1]
        ),
        "labels": all_labels,
        "predictions": all_predictions,
    }
    return metrics


# ============================================================
# Training loop
# ============================================================

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
    "val_f1": [],
}
best_val_f1 = -1.0

print("\n" + "=" * 70)
print("Starting ResNet18 DCT training")
print("=" * 70)

for epoch in range(NUM_EPOCHS):
    train_loss, train_accuracy = train_one_epoch(
        model, train_loader, criterion, optimizer
    )
    val = evaluate(model, validation_loader, criterion)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_loss"].append(val["loss"])
    history["val_accuracy"].append(val["accuracy"])
    history["val_f1"].append(val["f1"])

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val['loss']:.4f} | "
        f"Val Acc: {val['accuracy']:.4f} | "
        f"Val Precision: {val['precision']:.4f} | "
        f"Val Recall: {val['recall']:.4f} | "
        f"Val F1: {val['f1']:.4f} | LR: {current_lr:.6f}"
    )

    if val["f1"] > best_val_f1:
        best_val_f1 = val["f1"]
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val["f1"],
                "representation": "per-channel signed-log DCT",
            },
            best_model_path,
        )
        print(f"  -> Saved best DCT model (validation F1={val['f1']:.4f})")


# ============================================================
# Save history CSV and plots
# ============================================================

csv_path = results_dir / "resnet18_dct_history.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "epoch", "train_loss", "train_accuracy",
        "val_loss", "val_accuracy", "val_f1"
    ])
    for i in range(NUM_EPOCHS):
        writer.writerow([
            i + 1,
            history["train_loss"][i],
            history["train_accuracy"][i],
            history["val_loss"][i],
            history["val_accuracy"][i],
            history["val_f1"][i],
        ])

epochs = range(1, NUM_EPOCHS + 1)

def save_plot(y1, y2, label1, label2, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, y1, label=label1)
    plt.plot(epochs, y2, label=label2)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / filename, dpi=300)
    plt.show()

save_plot(
    history["train_loss"], history["val_loss"],
    "Training Loss", "Validation Loss", "Loss",
    "ResNet18 DCT Training and Validation Loss", "resnet18_dct_loss.png"
)
save_plot(
    history["train_accuracy"], history["val_accuracy"],
    "Training Accuracy", "Validation Accuracy", "Accuracy",
    "ResNet18 DCT Training and Validation Accuracy", "resnet18_dct_accuracy.png"
)

plt.figure(figsize=(10, 6))
plt.plot(epochs, history["val_f1"], label="Validation F1")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("ResNet18 DCT Validation F1")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(results_dir / "resnet18_dct_validation_f1.png", dpi=300)
plt.show()


# ============================================================
# Final test evaluation
# ============================================================

checkpoint = torch.load(best_model_path, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
print(f"\nBest epoch: {checkpoint['epoch']}")
print(f"Best validation F1: {checkpoint['val_f1']:.4f}")

test = evaluate(model, test_loader, criterion)

print("\n" + "=" * 70)
print("FINAL RESNET18 DCT TEST RESULTS")
print("=" * 70)
print(f"Test Accuracy : {test['accuracy']:.4f}")
print(f"Test Precision: {test['precision']:.4f}")
print(f"Test Recall   : {test['recall']:.4f}")
print(f"Test F1       : {test['f1']:.4f}")
print("\nConfusion Matrix (rows=true [normal, glaucoma], columns=predicted):")
print(test["confusion_matrix"])
print(f"\nCheckpoint: {best_model_path}")
print(f"History CSV: {csv_path}")
print("=" * 70)
