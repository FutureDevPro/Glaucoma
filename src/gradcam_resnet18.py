import csv
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"C:\Users\udit2\Desktop\Glaucoma")

MODEL_PATH = PROJECT_DIR / "models" / "resnet18_best.pth"

# Your existing dataset structure
TEST_DIRS = {
    "ACRIMA": PROJECT_DIR / "dataset" / "ACRIMA" / "testing",
    "DRISHTI": PROJECT_DIR / "dataset" / "DRISHTI" / "testing",
}

OUTPUT_DIR = PROJECT_DIR / "results" / "gradcam"

IMAGE_SIZE = 224

CLASS_NAMES = {
    0: "normal",
    1: "glaucoma"
}

# ImageNet normalization used by your trained model
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

(OUTPUT_DIR / "correct" / "normal").mkdir(
    parents=True, exist_ok=True
)

(OUTPUT_DIR / "correct" / "glaucoma").mkdir(
    parents=True, exist_ok=True
)

(OUTPUT_DIR / "incorrect" / "false_positive").mkdir(
    parents=True, exist_ok=True
)

(OUTPUT_DIR / "incorrect" / "false_negative").mkdir(
    parents=True, exist_ok=True
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])


# ============================================================
# MODEL
# ============================================================

print("=" * 70)
print("Loading ResNet18")
print("=" * 70)

model = resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print("Checkpoint:", MODEL_PATH)
print("Best epoch:", checkpoint["epoch"])
print("Validation F1:", checkpoint["val_f1"])
print("Device:", DEVICE)


# ============================================================
# GRAD-CAM TARGET LAYER
# ============================================================

# Last convolutional block of ResNet18
target_layers = [
    model.layer4[-1]
]


# ============================================================
# FIND IMAGES
# ============================================================

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG"
}


def get_images(dataset_dir):

    images = []

    for class_name, class_index in [
        ("normal", 0),
        ("glaucoma", 1)
    ]:

        class_dir = dataset_dir / class_name

        if not class_dir.exists():
            print(
                f"WARNING: Directory not found: {class_dir}"
            )
            continue

        for image_path in class_dir.rglob("*"):

            if image_path.suffix in VALID_EXTENSIONS:

                images.append({
                    "path": image_path,
                    "true_label": class_index,
                    "true_class": class_name
                })

    return images


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    true_label,
    dataset_name,
    writer
):

    # --------------------------------------------------------
    # Read original image
    # --------------------------------------------------------

    original = Image.open(image_path).convert("RGB")

    # Resize for model / CAM
    resized = original.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # RGB image in [0,1] for visualization
    rgb_image = np.asarray(
        resized
    ).astype(np.float32) / 255.0

    # --------------------------------------------------------
    # Prepare tensor
    # --------------------------------------------------------

    input_tensor = transform(
        original
    ).unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            input_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

    predicted_name = CLASS_NAMES[
        predicted_class
    ]

    true_name = CLASS_NAMES[
        true_label
    ]

    correct = (
        predicted_class == true_label
    )

    # --------------------------------------------------------
    # Determine error type
    # --------------------------------------------------------

    if correct:

        category = "correct"

        output_subdir = (
            OUTPUT_DIR
            / "correct"
            / true_name
        )

    else:

        if (
            true_label == 0
            and predicted_class == 1
        ):

            category = "false_positive"

            output_subdir = (
                OUTPUT_DIR
                / "incorrect"
                / "false_positive"
            )

        elif (
            true_label == 1
            and predicted_class == 0
        ):

            category = "false_negative"

            output_subdir = (
                OUTPUT_DIR
                / "incorrect"
                / "false_negative"
            )

        else:

            category = "incorrect"

            output_subdir = (
                OUTPUT_DIR
                / "incorrect"
            )

            output_subdir.mkdir(
                parents=True,
                exist_ok=True
            )

    output_subdir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    # Visualize the predicted class
    targets = [
        ClassifierOutputTarget(
            predicted_class
        )
    ]

    with GradCAM(
        model=model,
        target_layers=target_layers
    ) as cam:

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]

    # --------------------------------------------------------
    # Create heatmap overlay
    # --------------------------------------------------------

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    base_name = image_path.stem

    overlay_path = (
        output_subdir
        / f"{base_name}_gradcam.jpg"
    )

    heatmap_path = (
        output_subdir
        / f"{base_name}_heatmap.jpg"
    )

    original_path = (
        output_subdir
        / f"{base_name}_original.jpg"
    )

    # Save overlay
    Image.fromarray(
        visualization
    ).save(
        overlay_path,
        quality=95
    )

    # Save original resized image
    Image.fromarray(
        (
            rgb_image * 255
        ).astype(np.uint8)
    ).save(
        original_path,
        quality=95
    )

    # Save raw heatmap
    plt.imsave(
        heatmap_path,
        grayscale_cam,
        cmap="jet"
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    writer.writerow([
        dataset_name,
        image_path.name,
        str(image_path),
        true_name,
        predicted_name,
        f"{confidence:.6f}",
        "correct" if correct else "incorrect",
        category,
        str(overlay_path)
    ])

    return correct, predicted_class, confidence


# ============================================================
# MAIN
# ============================================================

def main():

    csv_path = (
        OUTPUT_DIR
        / "gradcam_results.csv"
    )

    total = 0
    correct = 0

    print()
    print("=" * 70)
    print("Starting Grad-CAM analysis")
    print("=" * 70)

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        writer.writerow([
            "dataset",
            "image",
            "image_path",
            "true_class",
            "predicted_class",
            "confidence",
            "result",
            "category",
            "gradcam_path"
        ])

        # ----------------------------------------------------
        # Process ACRIMA + DRISHTI
        # ----------------------------------------------------

        for dataset_name, dataset_dir in TEST_DIRS.items():

            print()
            print("-" * 70)
            print(
                f"Processing {dataset_name}"
            )
            print(
                f"Directory: {dataset_dir}"
            )
            print("-" * 70)

            images = get_images(
                dataset_dir
            )

            print(
                f"Images found: {len(images)}"
            )

            for index, item in enumerate(
                images,
                start=1
            ):

                try:

                    is_correct, _, _ = process_image(
                        image_path=item["path"],
                        true_label=item["true_label"],
                        dataset_name=dataset_name,
                        writer=writer
                    )

                    total += 1

                    if is_correct:
                        correct += 1

                    print(
                        f"[{index:03d}/{len(images):03d}] "
                        f"{item['path'].name}"
                    )

                except Exception as e:

                    print(
                        f"ERROR processing "
                        f"{item['path']}: {e}"
                    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    accuracy = (
        correct / total
        if total > 0
        else 0
    )

    print()
    print("=" * 70)
    print("GRAD-CAM ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Total images : {total}"
    )

    print(
        f"Correct      : {correct}"
    )

    print(
        f"Accuracy     : {accuracy:.4f}"
    )

    print()
    print(
        f"Results saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        f"CSV saved to:"
    )

    print(
        csv_path
    )

    print("=" * 70)


if __name__ == "__main__":
    main()