"""

DCT is applied independently to each RGB channel after resizing and
training-time augmentation. The signed-log coefficients are standardized
per channel per image so their numerical range is usable by a CNN.
This is an experimental representation, not a guaranteed improvement.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms


# ============================================================
# Paths
# ============================================================

DATASET_DIR = Path(r"C:\Users\udit2\Desktop\Glaucoma\dataset")


# ============================================================
# DCT transform
# ============================================================

class RGBDCTTransform:
    """
    Input:  torch Tensor [3, H, W], values in [0, 1]
    Output: torch Tensor [3, H, W], signed-log DCT coefficients,
            standardized independently for each channel.

    Note: standardizing each image changes absolute coefficient scale.
    Treat this as one experimental DCT representation and compare it
    against the original-image baseline using the same data splits.
    """

    def __call__(self, image_tensor):
        image_np = image_tensor.detach().cpu().numpy().astype(np.float32)
        output = np.empty_like(image_np, dtype=np.float32)

        for channel_idx in range(image_np.shape[0]):
            channel = np.ascontiguousarray(image_np[channel_idx])
            coefficients = cv2.dct(channel)

            # Preserve the sign while compressing the coefficient range.
            signed_log = np.sign(coefficients) * np.log1p(np.abs(coefficients))

            # Per-image/channel standardization; prevents a few large
            # coefficients from dominating the numeric range.
            mean = float(signed_log.mean())
            std = float(signed_log.std())
            if std > 1e-6:
                signed_log = (signed_log - mean) / std
            else:
                signed_log = signed_log - mean

            # Guard against extreme values without converting to uint8.
            output[channel_idx] = np.clip(signed_log, -5.0, 5.0)

        return torch.from_numpy(output)


# Apply geometric augmentation before DCT. Do not apply ImageNet RGB
# normalization to DCT coefficients; this is a different representation.
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    RGBDCTTransform(),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    RGBDCTTransform(),
])


# ============================================================
# Custom ImageFolder with explicit label mapping
# ============================================================

class GlaucomaImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        classes = ["normal", "glaucoma"]
        class_to_idx = {"normal": 0, "glaucoma": 1}
        return classes, class_to_idx


# ============================================================
# ACRIMA datasets
# ============================================================

acrima_train = GlaucomaImageFolder(
    DATASET_DIR / "ACRIMA" / "training",
    transform=train_transform,
)

acrima_validation = GlaucomaImageFolder(
    DATASET_DIR / "ACRIMA" / "validation",
    transform=eval_transform,
)

acrima_test = GlaucomaImageFolder(
    DATASET_DIR / "ACRIMA" / "testing",
    transform=eval_transform,
)


# ============================================================
# DRISHTI datasets
# ============================================================

drishti_train = GlaucomaImageFolder(
    DATASET_DIR / "DRISHTI_cropped_600" / "training",
    transform=train_transform,
)

drishti_validation = GlaucomaImageFolder(
    DATASET_DIR / "DRISHTI_cropped_600" / "validation",
    transform=eval_transform,
)

drishti_test = GlaucomaImageFolder(
    DATASET_DIR / "DRISHTI_cropped_600" / "testing",
    transform=eval_transform,
)


# ============================================================
# Combine ACRIMA + DRISHTI (same split design as your original file)
# ============================================================

train_dataset = ConcatDataset([acrima_train, drishti_train])
validation_dataset = ConcatDataset([acrima_validation, drishti_validation])
test_dataset = ConcatDataset([acrima_test, drishti_test])


# ============================================================
# DataLoaders
# ============================================================

BATCH_SIZE = 16

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
validation_loader = DataLoader(
    validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)


# ============================================================
# Dataset information
# ============================================================

print("=" * 50)
print("DCT Dataset information")
print("=" * 50)
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
print("=" * 50)
