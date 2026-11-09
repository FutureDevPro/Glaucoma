import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
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
BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# Model save directory
# ============================================================

models_dir = Path(r"C:\Users\udit2\Desktop\Glaucoma\Genralization (DRISHTI - DRISHTI)\models")
models_dir.mkdir(parents=True, exist_ok=True)

best_model_path = models_dir / "efficientnet_b0_best.pth"


# ============================================================
# Model
# ============================================================

weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(weights=weights)

# Replace the original ImageNet classifier.
# Original output = 1000 classes
# Our output = 2 classes
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)


# ============================================================
# Loss function
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

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# Validation function
# ============================================================

def validate(model, loader, criterion):

    model.eval()

    running_loss = 0.0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    total = len(all_labels)

    epoch_loss = running_loss / total

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
print("Starting EfficientNet-B0 training")
print("=" * 70)

for epoch in range(NUM_EPOCHS):

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
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

    # Save metrics for plotting
    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)

    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)
    history["val_f1"].append(val_f1)

    # Update learning rate
    scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f} | "
        f"Val F1: {val_f1:.4f} | "
        f"LR: {current_lr:.6f}"
    )

    # Save best model according to validation F1
    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
            },
            best_model_path
        )

        print(
            f"  -> Best model saved "
            f"(Validation F1 = {val_f1:.4f})"
        )

# ============================================================
# Save CSV
# ============================================================

results_dir = Path(r"C:\Users\udit2\Desktop\Glaucoma\Genralization (DRISHTI - DRISHTI)\results")
results_dir.mkdir(parents=True, exist_ok=True)

csv_path = results_dir / "efficientnet_b0_history.csv"

with open(csv_path, "w", newline="") as f:
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

print(f"\nTraining history saved to: {csv_path}")



# ============================================================
# Plot training history
# ============================================================


epochs = range(1, NUM_EPOCHS + 1)


# ------------------------------------------------------------
# Graph 1: Loss
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

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
plt.title("EfficientNet-B0 Training and Validation Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    results_dir / "efficientnet_b0_loss.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# Graph 2: Accuracy
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

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
plt.title("EfficientNet-B0 Training and Validation Accuracy")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    results_dir / "efficientnet_b0_accuracy.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# Graph 3: Validation F1
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    history["val_f1"],
    label="Validation F1"
)

plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("EfficientNet-B0 Validation F1 Score")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    results_dir / "efficientnet_b0_validation_f1.png",
    dpi=300
)

plt.show()


# ============================================================
# Load best model
# ============================================================

print()
print("=" * 70)
print("Loading best EfficientNet-B0 checkpoint")
print("=" * 70)

checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    f"Best epoch: {checkpoint['epoch']}"
)

print(
    f"Best validation F1: "
    f"{checkpoint['val_f1']:.4f}"
)


# ============================================================
# Final test evaluation
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
print("FINAL EfficientNet-B0 TEST RESULTS")
print("=" * 70)

print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall   : {test_recall:.4f}")
print(f"Test F1       : {test_f1:.4f}")

print()
print("Confusion Matrix")
print(cm)

print("=" * 70)