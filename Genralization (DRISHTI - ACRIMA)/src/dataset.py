from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# Paths
# ============================================================

DATASET_DIR = Path(r"C:\Users\udit2\Desktop\Glaucoma\dataset")


# ============================================================
# Image transformations
# ============================================================

# Training augmentation is used ONLY for ACRIMA training.
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# Validation and cross-dataset testing use deterministic transforms.
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# Custom ImageFolder with explicit label mapping
# ============================================================

class GlaucomaImageFolder(datasets.ImageFolder):

    def find_classes(self, directory):
        classes = ["normal", "glaucoma"]

        class_to_idx = {
            "normal": 0,
            "glaucoma": 1
        }

        return classes, class_to_idx


# ============================================================
# ACRIMA
# ============================================================
# ACRIMA is the SOURCE dataset.
#
# Training:
#   DRISHTI training
#
# Validation:
#   DRISHTI validation
#
# ACRIMA is NOT used during training or validation.
# ============================================================

acrima_train = GlaucomaImageFolder(
    DATASET_DIR / "DRISHTI" / "training",
    transform=train_transform
)

acrima_validation = GlaucomaImageFolder(
    DATASET_DIR / "DRISHTI" / "validation",
    transform=eval_transform
)


# ============================================================
# DRISHTI
# ============================================================
# DRISHTI is the UNSEEN TARGET dataset.
#
# Only the TESTING split is used.
#
# IMPORTANT:
# Do not use DRISHTI training/validation images for training,
# model selection, threshold tuning, or hyperparameter tuning.
# ============================================================

drishti_test = GlaucomaImageFolder(
    DATASET_DIR / "ACRIMA" / "testing",
    transform=eval_transform
)


# ============================================================
# DataLoaders
# ============================================================

BATCH_SIZE = 16

train_loader = DataLoader(
    acrima_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    acrima_validation,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    drishti_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# Dataset information
# ============================================================

print("========================================")
print("DRISHTI → ACRIMA")
print("Cross-Dataset Generalization Experiment")
print("========================================")

print("Classes:", acrima_train.classes)
print("Class mapping:", acrima_train.class_to_idx)
print()

print("SOURCE DATASET")
print("------------------------------")
print("DRISHTI training   :", len(acrima_train))
print("DRISHTI validation :", len(acrima_validation))
print()

print("TARGET DATASET")
print("------------------------------")
print("ACRIMA testing   :", len(drishti_test))
print()

print("DataLoader sizes")
print("------------------------------")
print("Training batches   :", len(train_loader))
print("Validation batches :", len(validation_loader))
print("Test batches       :", len(test_loader))
print()

print("Experiment:")
print("Train      : DRISHTI training")
print("Validation : DRISHTI validation")
print("Test       : ACRIMA testing")
print("========================================")
