from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


# ============================================================
# 1. Dataset paths and configuration
# ============================================================

DATASET_DIR = Path(
    r"C:\Users\udit2\Desktop\Glaucoma\dataset\DRISHTI_cropped"
)

TRAIN_DIR = DATASET_DIR / "training"
TEST_DIR = DATASET_DIR / "testing"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
VALIDATION_SIZE = 0.20
RANDOM_STATE = 42

CLASS_TO_IDX = {
    "normal": 0,
    "glaucoma": 1,
}


# ============================================================
# 2. RGB DCT transformation
# ============================================================

class RGBDCTTransform:
    """
    Input:  Tensor [3, H, W] with values in [0, 1].
    Output: Tensor [3, H, W] containing signed-log, standardized,
            and clipped DCT coefficients.
    """

    def __call__(self, image_tensor):
        image_np = (
            image_tensor.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        output = np.empty_like(image_np, dtype=np.float32)

        # Apply DCT independently to the R, G, and B channels.
        for channel_idx in range(image_np.shape[0]):
            channel = np.ascontiguousarray(image_np[channel_idx])
            coefficients = cv2.dct(channel)

            # Compress coefficient magnitudes while preserving signs.
            signed_log = (
                np.sign(coefficients)
                * np.log1p(np.abs(coefficients))
            )

            # Standardize each channel of each image independently.
            mean = float(signed_log.mean())
            std = float(signed_log.std())

            if std > 1e-6:
                signed_log = (signed_log - mean) / std
            else:
                signed_log = signed_log - mean

            # Limit extreme standardized values.
            output[channel_idx] = np.clip(
                signed_log, -5.0, 5.0
            )

        return torch.from_numpy(output)


# Augmentation is applied only to the training subset.
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    RGBDCTTransform(),
])

# Validation and testing have no random augmentation.
eval_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    RGBDCTTransform(),
])


# ============================================================
# 3. ImageFolder with explicit class mapping
# ============================================================

class GlaucomaImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        classes = ["normal", "glaucoma"]

        for class_name in classes:
            class_path = Path(directory) / class_name
            if not class_path.is_dir():
                raise FileNotFoundError(
                    f"Missing class folder: {class_path}"
                )

        return classes, CLASS_TO_IDX.copy()


# ============================================================
# 4. Dataset that applies a transform to selected image paths
# ============================================================

class GlaucomaImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = list(samples)
        self.transform = transform

        if not self.samples:
            raise ValueError("The dataset split contains no images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, int(label)


# ============================================================
# 5. Load training-folder samples and create stratified split
# ============================================================

if not TRAIN_DIR.is_dir():
    raise FileNotFoundError(f"Training folder does not exist: {TRAIN_DIR}")

if not TEST_DIR.is_dir():
    raise FileNotFoundError(f"Testing folder does not exist: {TEST_DIR}")

# Load paths and labels only; transforms are applied later by each split.
training_source = GlaucomaImageFolder(
    str(TRAIN_DIR),
    transform=None,
)

all_train_samples = training_source.samples
all_train_paths = [sample[0] for sample in all_train_samples]
all_train_labels = [sample[1] for sample in all_train_samples]

# Preserve class proportions in both training and validation subsets.
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_train_paths,
    all_train_labels,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=all_train_labels,
)

train_samples = list(zip(train_paths, train_labels))
validation_samples = list(zip(val_paths, val_labels))

train_dataset = GlaucomaImageDataset(
    train_samples,
    transform=train_transform,
)

validation_dataset = GlaucomaImageDataset(
    validation_samples,
    transform=eval_transform,
)


# ============================================================
# 6. Load the independent test folder
# ============================================================

test_source = GlaucomaImageFolder(
    str(TEST_DIR),
    transform=None,
)

test_dataset = GlaucomaImageDataset(
    test_source.samples,
    transform=eval_transform,
)


# ============================================================
# 7. DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# 8. Print total counts and per-class distributions
# ============================================================

def print_split_info(name, samples):
    normal_count = sum(label == CLASS_TO_IDX["normal"] for _, label in samples)
    glaucoma_count = sum(
        label == CLASS_TO_IDX["glaucoma"] for _, label in samples
    )

    print(f"\n{name}")
    print("-" * 40)
    print("Total images :", len(samples))
    print("Normal       :", normal_count)
    print("Glaucoma     :", glaucoma_count)


print("\n" + "=" * 50)
print("DRISHTI DCT DATASET INFORMATION")
print("=" * 50)
print("Class mapping:", CLASS_TO_IDX)
print(f"Validation fraction of training folder: {VALIDATION_SIZE:.0%}")
print("Random state:", RANDOM_STATE)

print_split_info("Training subset", train_samples)
print_split_info("Validation subset", validation_samples)
print_split_info("Independent testing set", test_source.samples)

print("\nDataLoader batch size:", BATCH_SIZE)
print("DCT representation: enabled for training, validation, and testing")
print("Testing images are not used to create the train/validation split.")
print("=" * 50)
