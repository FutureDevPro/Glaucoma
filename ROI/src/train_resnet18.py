import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

import matplotlib.pyplot as plt

import csv

from pathlib import Path

from torch.optim import Adam

from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from dataset import (
    train_loader,
    validation_loader,
    test_loader
)


# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 2

LEARNING_RATE = 5e-4

WEIGHT_DECAY = 1e-4

NUM_EPOCHS = 50

BATCH_SIZE = 16


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("Device:", DEVICE)


# ============================================================
# Model save directory
# ============================================================

models_dir = Path(
    r"C:\Users\udit2\Desktop\Glaucoma\ROI\model"
)

models_dir.mkdir(
    parents=True,
    exist_ok=True
)


best_model_path = (
    models_dir / "resnet18_oracle_roi_best.pth"
)


# ============================================================
# Model
# ============================================================

weights = ResNet18_Weights.DEFAULT

model = resnet18(
    weights=weights
)


# Replace ImageNet classifier
model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)


model = model.to(DEVICE)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# Optimizer
# ============================================================

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# Learning-rate scheduler
# ============================================================

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS
)


# ============================================================
# Training function
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        # Clear gradients
        optimizer.zero_grad()

        # Forward
        outputs = model(images)

        # Loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# Validation function
# ============================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    all_labels = []

    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    total = len(all_labels)

    epoch_loss = (
        running_loss / total
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    return (
        epoch_loss,
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# Training history
# ============================================================

history = {

    "train_loss": [],

    "train_accuracy": [],

    "val_loss": [],

    "val_accuracy": [],

    "val_f1": []
}


best_val_f1 = -1.0


print()
print("=" * 70)
print("Starting ResNet18 Oracle-ROI training")
print("=" * 70)


# ============================================================
# Training loop
# ============================================================

for epoch in range(NUM_EPOCHS):

    train_loss, train_accuracy = (
        train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )
    )


    (
        val_loss,
        val_accuracy,
        val_precision,
        val_recall,
        val_f1
    ) = validate(
        model,
        validation_loader,
        criterion
    )


    # Save history

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_accuracy
    )

    history["val_f1"].append(
        val_f1
    )


    # Scheduler

    scheduler.step()

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    print(
        f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f} | "
        f"Val F1: {val_f1:.4f} | "
        f"LR: {current_lr:.6f}"
    )


    # --------------------------------------------------------
    # Save best model according to validation F1
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        torch.save(
            {
                "epoch": epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_f1": val_f1
            },
            best_model_path
        )

        print(
            f"  -> Best Oracle-ROI model saved "
            f"(Validation F1 = {val_f1:.4f})"
        )


# ============================================================
# Save training history
# ============================================================

results_dir = Path(
    r"C:\Users\udit2\Desktop\Glaucoma\ROI\result2"
)

results_dir.mkdir(
    parents=True,
    exist_ok=True
)


csv_path = (
    results_dir /
    "resnet18_oracle_roi_history.csv"
)


with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "val_f1"
    ])

    for epoch in range(NUM_EPOCHS):

        writer.writerow([
            epoch + 1,

            history["train_loss"][epoch],

            history["train_accuracy"][epoch],

            history["val_loss"][epoch],

            history["val_accuracy"][epoch],

            history["val_f1"][epoch]
        ])


print(
    f"\nTraining history saved to: "
    f"{csv_path}"
)


# ============================================================
# Plot 1: Loss
# ============================================================

epochs = range(
    1,
    NUM_EPOCHS + 1
)


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    history["train_loss"],
    label="Training Loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "ResNet18 Oracle-ROI "
    "Training and Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    results_dir /
    "resnet18_oracle_roi_loss.png",
    dpi=300
)

plt.show()


# ============================================================
# Plot 2: Accuracy
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    history["train_accuracy"],
    label="Training Accuracy"
)

plt.plot(
    epochs,
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "ResNet18 Oracle-ROI "
    "Training and Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    results_dir /
    "resnet18_oracle_roi_accuracy.png",
    dpi=300
)

plt.show()


# ============================================================
# Plot 3: Validation F1
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    history["val_f1"],
    label="Validation F1"
)

plt.xlabel("Epoch")

plt.ylabel("F1 Score")

plt.title(
    "ResNet18 Oracle-ROI "
    "Validation F1 Score"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    results_dir /
    "resnet18_oracle_roi_validation_f1.png",
    dpi=300
)

plt.show()


# ============================================================
# Load best model
# ============================================================

print()
print("=" * 70)
print("Loading best Oracle-ROI ResNet18 checkpoint")
print("=" * 70)


checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


print(
    f"Best epoch: "
    f"{checkpoint['epoch']}"
)

print(
    f"Best validation F1: "
    f"{checkpoint['val_f1']:.4f}"
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

model.eval()

all_labels = []

all_predictions = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# Test metrics
# ============================================================

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)


test_precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)


test_recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)


test_f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)


cm = confusion_matrix(
    all_labels,
    all_predictions
)


# ============================================================
# Results
# ============================================================

print()
print("=" * 70)
print("FINAL RESNET18 ORACLE-ROI TEST RESULTS")
print("=" * 70)

print(
    f"Test Accuracy : {test_accuracy:.4f}"
)

print(
    f"Test Precision: {test_precision:.4f}"
)

print(
    f"Test Recall   : {test_recall:.4f}"
)

print(
    f"Test F1       : {test_f1:.4f}"
)

print()

print("Confusion Matrix")

print(cm)

print("=" * 70)

print()
print("IMPORTANT:")
print(
    "This is an ORACLE-ROI diagnostic experiment."
)

print(
    "Ground-truth optic-disc centers were used "
    "to generate test ROIs."
)

print(
    "Do NOT present this as a deployable "
    "localization pipeline."
)

print("=" * 70)