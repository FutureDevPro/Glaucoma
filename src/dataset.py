from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms


# ============================================================
# Paths
# ============================================================

# This file is inside Glaucoma-CNN/src/
# Therefore ../dataset points to Glaucoma-CNN/dataset/
DATASET_DIR = Path(r"C:\Users\udit2\Desktop\Glaucoma\dataset")


# ============================================================
# Image transformations
# ============================================================

# Applied only to training images.
# These transformations are allowed to modify the training images
# so that the model sees slightly different versions during training.
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


# Applied to validation and testing images.
# No random augmentation here.
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
# ACRIMA datasets
# ============================================================

acrima_train = GlaucomaImageFolder(
    DATASET_DIR / "ACRIMA" / "training",
    transform=train_transform
)

acrima_validation = GlaucomaImageFolder(
    DATASET_DIR / "ACRIMA" / "validation",
    transform=eval_transform
)

acrima_test = GlaucomaImageFolder(
    DATASET_DIR / "ACRIMA" / "testing",
    transform=eval_transform
)


# ============================================================
# DRISHTI datasets
# ============================================================

drishti_train = GlaucomaImageFolder(
    DATASET_DIR / "DRISHTI" / "training",
    transform=train_transform
)

drishti_validation = GlaucomaImageFolder(
    DATASET_DIR / "DRISHTI" / "validation",
    transform=eval_transform
)

drishti_test = GlaucomaImageFolder(
    DATASET_DIR / "DRISHTI" / "testing",
    transform=eval_transform
)


# ============================================================
# Combine ACRIMA + DRISHTI
# ============================================================

train_dataset = ConcatDataset([
    acrima_train,
    drishti_train
])

validation_dataset = ConcatDataset([
    acrima_validation,
    drishti_validation
])

test_dataset = ConcatDataset([
    acrima_test,
    drishti_test
])


# ============================================================
# DataLoaders
# ============================================================

BATCH_SIZE = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# Dataset information
# ============================================================

print("========================================")
print("Dataset information")
print("========================================")

print("Classes:", acrima_train.classes)
print("Class mapping:", acrima_train.class_to_idx)
print()

print("ACRIMA training   :", len(acrima_train))
print("DRISHTI training  :", len(drishti_train))
print("Combined training :", len(train_dataset))

print()

print("ACRIMA validation   :", len(acrima_validation))
print("DRISHTI validation  :", len(drishti_validation))
print("Combined validation :", len(validation_dataset))

print()

print("ACRIMA testing   :", len(acrima_test))
print("DRISHTI testing  :", len(drishti_test))
print("Combined testing :", len(test_dataset))

print("========================================")   