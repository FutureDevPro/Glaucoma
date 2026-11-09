from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ============================================================
# Paths
# ============================================================

DATASET_DIR = Path(
    r"C:\Users\udit2\Desktop\Glaucoma\dataset\DRISHTI"
)


# ============================================================
# ROI configuration
# ============================================================

# Ground-truth optic-disc center is used.
# This is intentionally an ORACLE-ROI experiment.

ROI_SIZE = 600

MODEL_INPUT_SIZE = 224


# ============================================================
# SAVE CROPPED ROI OUTPUTS
# ============================================================

SAVE_ROI_IMAGES = True

ROI_OUTPUT_DIR = Path(
    r"C:\Users\udit2\Desktop\Glaucoma\ROI\cropped_images_600"
)


# ============================================================
# Image transformations
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize(
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


eval_transform = transforms.Compose([
    transforms.Resize(
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Oracle ROI Dataset
# ============================================================

class OracleROIDataset(Dataset):

    def __init__(
        self,
        image_root,
        center_root,
        split_name,
        transform=None,
        roi_size=ROI_SIZE,
        save_roi_images=SAVE_ROI_IMAGES
    ):

        self.image_root = Path(image_root)

        self.center_root = Path(center_root)

        self.split_name = split_name

        self.transform = transform

        self.roi_size = roi_size

        self.save_roi_images = save_roi_images

        self.samples = []


        # ----------------------------------------------------
        # Class mapping
        # ----------------------------------------------------

        class_names = [
            "normal",
            "glaucoma"
        ]

        self.class_to_idx = {
            "normal": 0,
            "glaucoma": 1
        }

        self.classes = class_names


        # ----------------------------------------------------
        # Find images
        # ----------------------------------------------------

        for class_name in class_names:

            class_dir = (
                self.image_root / class_name
            )

            if not class_dir.exists():

                raise FileNotFoundError(
                    f"Class directory not found:\n"
                    f"{class_dir}"
                )


            for image_path in sorted(
                class_dir.glob("*.png")
            ):

                image_name = image_path.stem


                # Example:
                #
                # drishtiGS_002.png
                #
                # ->
                #
                # drishtiGS_002_diskCenter.txt

                center_filename = (
                    f"{image_name}_diskCenter.txt"
                )


                center_path = (
                    self.center_root /
                    center_filename
                )


                if not center_path.exists():

                    raise FileNotFoundError(
                        "\nMissing optic-disc center file:\n"
                        f"Image : {image_path}\n"
                        f"Center: {center_path}"
                    )


                label = (
                    self.class_to_idx[class_name]
                )


                self.samples.append(
                    (
                        image_path,
                        center_path,
                        label
                    )
                )


        # ----------------------------------------------------
        # Output directory
        # ----------------------------------------------------

        if self.save_roi_images:

            self.output_root = (
                ROI_OUTPUT_DIR /
                self.split_name
            )

            self.output_root.mkdir(
                parents=True,
                exist_ok=True
            )


        # ----------------------------------------------------
        # Dataset information
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("Oracle ROI Dataset")
        print("=" * 60)

        print(
            "Split           :",
            self.split_name
        )

        print(
            "Image directory:",
            self.image_root
        )

        print(
            "Center directory:",
            self.center_root
        )

        print(
            "ROI size        :",
            self.roi_size
        )

        print(
            "Number samples  :",
            len(self.samples)
        )

        print(
            "Classes         :",
            self.classes
        )

        print(
            "Class mapping   :",
            self.class_to_idx
        )

        if self.save_roi_images:

            print(
                "ROI output      :",
                self.output_root
            )

        print("=" * 60)


        # ----------------------------------------------------
        # Generate and save ROI images
        #
        # This happens ONCE during dataset construction.
        #
        # It does NOT happen inside __getitem__.
        #
        # Therefore training augmentation does not repeatedly
        # overwrite the saved ROI images.
        # ----------------------------------------------------

        if self.save_roi_images:

            self.generate_roi_outputs()


    # ========================================================
    # Read optic-disc center
    # ========================================================

    def read_center(
        self,
        center_path
    ):

        with open(
            center_path,
            "r"
        ) as f:

            values = (
                f.read()
                .strip()
                .split()
            )


        if len(values) < 2:

            raise ValueError(
                f"Invalid center file:\n"
                f"{center_path}"
            )


        # Drishti:
        #
        # x = horizontal coordinate
        # y = vertical coordinate

        cx = int(
            float(values[1])
        )

        cy = int(
            float(values[0])
        )


        return cx, cy


    # ========================================================
    # Crop ROI
    # ========================================================

    def crop_roi(
        self,
        image,
        cx,
        cy
    ):

        width, height = image.size

        half = self.roi_size // 2


        # Desired crop boundaries

        left = cx - half

        top = cy - half

        right = cx + half

        bottom = cy + half


        # ----------------------------------------------------
        # Create black canvas.
        #
        # This handles cases where the requested crop extends
        # beyond the original image.
        # ----------------------------------------------------

        roi = Image.new(
            "RGB",
            (
                self.roi_size,
                self.roi_size
            ),
            (0, 0, 0)
        )


        # ----------------------------------------------------
        # Intersection with original image
        # ----------------------------------------------------

        src_left = max(
            0,
            left
        )

        src_top = max(
            0,
            top
        )

        src_right = min(
            width,
            right
        )

        src_bottom = min(
            height,
            bottom
        )


        # No overlap

        if (
            src_left >= src_right
            or
            src_top >= src_bottom
        ):

            return roi


        # Crop valid region

        cropped = image.crop(
            (
                src_left,
                src_top,
                src_right,
                src_bottom
            )
        )


        # Destination position

        dst_left = (
            src_left - left
        )

        dst_top = (
            src_top - top
        )


        roi.paste(
            cropped,
            (
                dst_left,
                dst_top
            )
        )


        return roi


    # ========================================================
    # Save ROI image
    # ========================================================

    def save_roi(
        self,
        roi,
        image_path,
        label
    ):

        class_name = (
            "glaucoma"
            if label == 1
            else "normal"
        )


        class_output_dir = (
            self.output_root /
            class_name
        )


        class_output_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        output_filename = (
            f"{image_path.stem}_ROI.png"
        )


        output_path = (
            class_output_dir /
            output_filename
        )


        roi.save(
            output_path
        )


    # ========================================================
    # Generate all ROI outputs
    # ========================================================

    def generate_roi_outputs(self):

        print()

        print(
            f"Generating {self.split_name} "
            f"ROI images..."
        )


        saved_count = 0


        for (
            image_path,
            center_path,
            label
        ) in self.samples:


            # Load original image

            image = Image.open(
                image_path
            ).convert("RGB")


            # Read ground-truth center

            cx, cy = self.read_center(
                center_path
            )


            # Crop ROI

            roi = self.crop_roi(
                image,
                cx,
                cy
            )


            # Save

            self.save_roi(
                roi,
                image_path,
                label
            )


            saved_count += 1


        print(
            f"Saved {saved_count} ROI images."
        )

        print()


    # ========================================================
    # Dataset item
    # ========================================================

    def __getitem__(
        self,
        index
    ):

        image_path, center_path, label = (
            self.samples[index]
        )


        # Load original image

        image = Image.open(
            image_path
        ).convert("RGB")


        # Ground-truth center

        cx, cy = self.read_center(
            center_path
        )


        # Crop around TRUE optic-disc center

        roi = self.crop_roi(
            image,
            cx,
            cy
        )


        # Apply training/evaluation transform

        if self.transform is not None:

            roi = self.transform(
                roi
            )


        return roi, label


    # ========================================================
    # Length
    # ========================================================

    def __len__(self):

        return len(self.samples)


# ============================================================
# TRAINING DATASET
# ============================================================

train_dataset = OracleROIDataset(

    image_root=(
        DATASET_DIR /
        "training"
    ),

    center_root=(
        DATASET_DIR /
        "training_validation_center"
    ),

    split_name="training",

    transform=train_transform,

    roi_size=ROI_SIZE,

    save_roi_images=True
)


# ============================================================
# VALIDATION DATASET
# ============================================================

validation_dataset = OracleROIDataset(

    image_root=(
        DATASET_DIR /
        "validation"
    ),

    center_root=(
        DATASET_DIR /
        "training_validation_center"
    ),

    split_name="validation",

    transform=eval_transform,

    roi_size=ROI_SIZE,

    save_roi_images=True
)


# ============================================================
# TEST DATASET
#
# Oracle experiment:
# ground-truth testing centers are intentionally used
# to create the diagnostic ROI.
# ============================================================

test_dataset = OracleROIDataset(

    image_root=(
        DATASET_DIR /
        "testing"
    ),

    center_root=(
        DATASET_DIR /
        "testing_center"
    ),

    split_name="testing",

    transform=eval_transform,

    roi_size=ROI_SIZE,

    save_roi_images=True
)


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

print()
print("=" * 60)
print("DRISHTI ORACLE-ROI EXPERIMENT")
print("=" * 60)

print(
    "Training samples   :",
    len(train_dataset)
)

print(
    "Validation samples :",
    len(validation_dataset)
)

print(
    "Testing samples    :",
    len(test_dataset)
)

print(
    "Training batches   :",
    len(train_loader)
)

print(
    "Validation batches :",
    len(validation_loader)
)

print(
    "Testing batches    :",
    len(test_loader)
)

print()

print(
    "ROI size           :",
    ROI_SIZE
)

print(
    "Model input        :",
    MODEL_INPUT_SIZE
)

print()

print(
    "Saved ROI directory:"
)

print(
    ROI_OUTPUT_DIR
)

print()

print(
    "IMPORTANT:"
)

print(
    "This is an ORACLE-ROI diagnostic experiment."
)

print(
    "Ground-truth optic-disc centers are used "
    "to generate the ROIs."
)

print(
    "The saved ROI images are intended for "
    "visual quality inspection."
)

print("=" * 60)