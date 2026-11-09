import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
from pathlib import Path
import csv


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Path to your trained .pth file
MODEL_PATH = Path(
    r"C:\Users\udit2\Desktop\Glaucoma\Genralization (ACRIMA - DRISHTI)\models\resnet18_best.pth"
)

# Path to DRISHTI testing folder
TEST_DIR = Path(
    r"C:\Users\udit2\Desktop\Glaucoma\dataset\DRISHTI\testing"
)

# Where to save confidence results
OUTPUT_CSV = Path(
    r"C:\Users\udit2\Desktop\Glaucoma\Genralization (ACRIMA - DRISHTI)\results\drishti_confidence_scores.csv"
)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)


# ============================================================
# 2. CLASS MAPPING
# ============================================================

# Same mapping used in your dataset.py
CLASS_NAMES = {
    0: "normal",
    1: "glaucoma"
}


import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import csv


# ============================================================
#                  CONFIGURATION
#        CHANGE ONLY THIS SECTION FOR A NEW MODEL
# ============================================================

# ------------------------------------------------------------
# 1. MODEL SETTINGS
# ------------------------------------------------------------

# Path to the trained .pth / .pt checkpoint
MODEL_PATH = Path(
    r"C:\Users\udit2\Desktop\Glaucoma\Genralization (ACRIMA - DRISHTI)\models\resnet18_best.pth"
)

# Model architecture
#
# Currently supported:
# "resnet18"
# "resnet50"
# "resnet101"
# "efficientnet_b0"
# "efficientnet_b1"
# "mobilenet_v2"
#
MODEL_ARCHITECTURE = "resnet18"

# Number of output classes
NUM_CLASSES = 2


# ------------------------------------------------------------
# 2. DATASET SETTINGS
# ------------------------------------------------------------

# Root folder containing class folders
#
# Expected:
#
# testing/
#     normal/
#     glaucoma/
#
TEST_DIR = Path(
    r"C:\Users\udit2\Desktop\Glaucoma\dataset\DRISHTI\testing"
)


# ------------------------------------------------------------
# 3. CLASS MAPPING
# ------------------------------------------------------------

# IMPORTANT:
# This MUST match the class mapping used when training
# the model.
#
# Your current dataset.py:
# 0 = normal
# 1 = glaucoma

CLASS_NAMES = {
    0: "normal",
    1: "glaucoma"
}


# ------------------------------------------------------------
# 4. IMAGE PREPROCESSING
# ------------------------------------------------------------

# Image size used by the model
IMAGE_SIZE = (224, 224)

# ImageNet normalization
NORMALIZE_MEAN = [
    0.485,
    0.456,
    0.406
]

NORMALIZE_STD = [
    0.229,
    0.224,
    0.225
]


# ------------------------------------------------------------
# 5. CHECKPOINT SETTINGS
# ------------------------------------------------------------

# Your checkpoint contains:
#
# checkpoint["model_state_dict"]
#
# Set this to True if the .pth contains a checkpoint
# dictionary like your current training code.
#
# Set False if the .pth contains only model.state_dict().
#
CHECKPOINT_HAS_MODEL_STATE_DICT = True


# ------------------------------------------------------------
# 6. OUTPUT SETTINGS
# ------------------------------------------------------------

OUTPUT_CSV = Path(
    r"C:\Users\udit2\Desktop\Glaucoma\Genralization (ACRIMA - DRISHTI)\results\confidence_scores.csv"
)


# ------------------------------------------------------------
# 7. DEVICE
# ------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
#             END OF CONFIGURATION SECTION
# ============================================================


print("=" * 70)
print("MODEL CONFIDENCE SCORE ANALYSIS")
print("=" * 70)

print("Device       :", DEVICE)
print("Model        :", MODEL_ARCHITECTURE)
print("Model path   :", MODEL_PATH)
print("Test folder  :", TEST_DIR)
print("Number class :", NUM_CLASSES)

print("=" * 70)


# ============================================================
# 8. VALIDATE CONFIGURATION
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nModel file not found:\n{MODEL_PATH}"
    )


if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"\nTest directory not found:\n{TEST_DIR}"
    )


# ============================================================
# 9. PREPROCESSING
# ============================================================

test_transform = transforms.Compose([

    transforms.Resize(IMAGE_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=NORMALIZE_MEAN,
        std=NORMALIZE_STD
    )
])


# ============================================================
# 10. CREATE MODEL
# ============================================================

def create_model(
    architecture,
    num_classes
):

    architecture = architecture.lower()

    print("\nCreating model:", architecture)

    # --------------------------------------------------------
    # ResNet18
    # --------------------------------------------------------

    if architecture == "resnet18":

        model = models.resnet18(
            weights=None
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes
        )


    # --------------------------------------------------------
    # ResNet50
    # --------------------------------------------------------

    elif architecture == "resnet50":

        model = models.resnet50(
            weights=None
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes
        )


    # --------------------------------------------------------
    # ResNet101
    # --------------------------------------------------------

    elif architecture == "resnet101":

        model = models.resnet101(
            weights=None
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes
        )


    # --------------------------------------------------------
    # EfficientNet-B0
    # --------------------------------------------------------

    elif architecture == "efficientnet_b0":

        model = models.efficientnet_b0(
            weights=None
        )

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )


    # --------------------------------------------------------
    # EfficientNet-B1
    # --------------------------------------------------------

    elif architecture == "efficientnet_b1":

        model = models.efficientnet_b1(
            weights=None
        )

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )


    # --------------------------------------------------------
    # MobileNet-V2
    # --------------------------------------------------------

    elif architecture == "mobilenet_v2":

        model = models.mobilenet_v2(
            weights=None
        )

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )


    # --------------------------------------------------------
    # Unsupported architecture
    # --------------------------------------------------------

    else:

        raise ValueError(
            f"\nUnsupported architecture: {architecture}\n\n"
            "Currently supported:\n"
            "resnet18\n"
            "resnet50\n"
            "resnet101\n"
            "efficientnet_b0\n"
            "efficientnet_b1\n"
            "mobilenet_v2"
        )


    return model


model = create_model(
    MODEL_ARCHITECTURE,
    NUM_CLASSES
)


# ============================================================
# 11. LOAD CHECKPOINT
# ============================================================

print("\nLoading checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# ============================================================
# 12. LOAD MODEL WEIGHTS
# ============================================================

if CHECKPOINT_HAS_MODEL_STATE_DICT:

    # Example:
    #
    # {
    #     "epoch": ...,
    #     "model_state_dict": ...,
    #     "optimizer_state_dict": ...,
    #     "val_f1": ...
    # }

    print("\nCheckpoint keys:")

    if isinstance(checkpoint, dict):
        print(
            checkpoint.keys()
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    # Example:
    #
    # torch.save(model.state_dict(), "model.pth")

    model.load_state_dict(
        checkpoint
    )


# Move model to GPU/CPU
model = model.to(DEVICE)

# Evaluation mode
model.eval()


print("\nModel loaded successfully.")


# ============================================================
# 13. PRINT CHECKPOINT INFORMATION
# ============================================================

if CHECKPOINT_HAS_MODEL_STATE_DICT:

    if isinstance(checkpoint, dict):

        if "epoch" in checkpoint:

            print(
                "Checkpoint epoch:",
                checkpoint["epoch"]
            )

        if "val_f1" in checkpoint:

            print(
                "Validation F1:",
                checkpoint["val_f1"]
            )


# ============================================================
# 14. FIND TEST IMAGES
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


samples = []


for class_id, class_name in CLASS_NAMES.items():

    class_folder = TEST_DIR / class_name

    if not class_folder.exists():

        print(
            f"\nWARNING: Class folder not found:"
            f"\n{class_folder}"
        )

        continue


    for image_path in sorted(
        class_folder.iterdir()
    ):

        if image_path.is_file():

            if image_path.suffix.lower() in image_extensions:

                samples.append(
                    (
                        image_path,
                        class_id
                    )
                )


# ============================================================
# 15. DATASET SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(
    "Total images:",
    len(samples)
)


for class_id, class_name in CLASS_NAMES.items():

    count = sum(
        1
        for _, label in samples
        if label == class_id
    )

    print(
        f"{class_name.capitalize()} images:",
        count
    )


# ============================================================
# 16. CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 17. RUN INFERENCE
# ============================================================

results = []


print("\n" + "=" * 70)
print("CALCULATING CONFIDENCE SCORES")
print("=" * 70)


with torch.no_grad():

    for index, (image_path, true_label) in enumerate(samples):

        try:

            # ------------------------------------------------
            # Load image
            # ------------------------------------------------

            image = Image.open(
                image_path
            ).convert("RGB")


            # ------------------------------------------------
            # Preprocess
            # ------------------------------------------------

            image_tensor = test_transform(
                image
            )


            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)


            # Move to device
            image_tensor = image_tensor.to(
                DEVICE
            )


            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            outputs = model(
                image_tensor
            )


            # ------------------------------------------------
            # Softmax probabilities
            # ------------------------------------------------

            probabilities = torch.softmax(
                outputs,
                dim=1
            )


            # ------------------------------------------------
            # Predicted class
            # ------------------------------------------------

            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()


            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidence = probabilities[
                0,
                predicted_class
            ].item()


            # ------------------------------------------------
            # Correct / incorrect
            # ------------------------------------------------

            correct = (
                predicted_class == true_label
            )


            # ------------------------------------------------
            # Build result
            # ------------------------------------------------

            result = {

                "filename":
                    image_path.name,

                "true_class":
                    CLASS_NAMES[true_label],

                "predicted_class":
                    CLASS_NAMES[predicted_class],

                "confidence":
                    confidence,

                "correct":
                    correct
            }


            # ------------------------------------------------
            # Add probability for every class
            # ------------------------------------------------

            for class_id, class_name in CLASS_NAMES.items():

                result[
                    f"{class_name}_probability"
                ] = probabilities[
                    0,
                    class_id
                ].item()


            results.append(
                result
            )


            # ------------------------------------------------
            # Print result
            # ------------------------------------------------

            probability_text = " | ".join(

                f"{class_name}: "
                f"{probabilities[0, class_id].item():.4f}"

                for class_id, class_name
                in CLASS_NAMES.items()
            )


            print(

                f"{index + 1:03d} | "

                f"{image_path.name:30s} | "

                f"True: "
                f"{CLASS_NAMES[true_label]:10s} | "

                f"Pred: "
                f"{CLASS_NAMES[predicted_class]:10s} | "

                f"Confidence: "
                f"{confidence:.4f} | "

                f"{probability_text} | "

                f"{'CORRECT' if correct else 'WRONG'}"
            )


        except Exception as e:

            print(
                f"\nERROR processing "
                f"{image_path.name}: {e}"
            )


# ============================================================
# 18. SAVE CSV
# ============================================================

if len(results) > 0:

    # Get fields from first result
    fieldnames = list(
        results[0].keys()
    )


    with open(
        OUTPUT_CSV,
        mode="w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )


# ============================================================
# 19. SUMMARY
# ============================================================

total = len(results)


correct_count = sum(

    1
    for result in results
    if result["correct"]

)


wrong_count = (
    total - correct_count
)


accuracy = (

    correct_count / total
    if total > 0
    else 0

)


print("\n" + "=" * 70)
print("CONFIDENCE SCORE ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"Total images        : {total}"
)

print(
    f"Correct predictions : {correct_count}"
)

print(
    f"Wrong predictions   : {wrong_count}"
)

print(
    f"Accuracy            : {accuracy:.4f}"
)

print(
    f"Accuracy (%)        : {accuracy * 100:.2f}%"
)

print("\nResults saved to:")

print(
    OUTPUT_CSV
)

print("=" * 70)

# ============================================================
# 4. CREATE RESNET18 ARCHITECTURE
# ============================================================

print("\nCreating ResNet18 model...")

model = models.resnet18(weights=None)

# Your training code replaced the original FC layer
# with a 2-class classifier.
model.fc = nn.Linear(
    model.fc.in_features,
    2
)


# ============================================================
# 5. LOAD .PTH CHECKPOINT
# ============================================================

print("Loading checkpoint:")
print(MODEL_PATH)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

print("\nCheckpoint contents:")
print(checkpoint.keys())


# ============================================================
# 6. LOAD TRAINED WEIGHTS
# ============================================================

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()

print("\nModel loaded successfully.")

# Optional information stored in your checkpoint
if "epoch" in checkpoint:
    print("Best epoch:", checkpoint["epoch"])

if "val_f1" in checkpoint:
    print("Validation F1:", checkpoint["val_f1"])


# ============================================================
# 7. FIND TEST IMAGES
# ============================================================

# Expected structure:
#
# DRISHTI/
#     testing/
#         normal/
#             image1.jpg
#             image2.jpg
#
#         glaucoma/
#             image3.jpg
#             image4.jpg

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


samples = []


for class_id, class_name in CLASS_NAMES.items():

    class_folder = TEST_DIR / class_name

    if not class_folder.exists():
        print(
            f"WARNING: Folder not found: {class_folder}"
        )
        continue

    for image_path in class_folder.iterdir():

        if image_path.suffix.lower() in image_extensions:

            samples.append(
                (
                    image_path,
                    class_id
                )
            )


print("\nTotal test images:", len(samples))

print("Normal images:",
      sum(1 for _, label in samples if label == 0))

print("Glaucoma images:",
      sum(1 for _, label in samples if label == 1))


# ============================================================
# 8. CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 9. RUN INFERENCE
# ============================================================

results = []


print("\nCalculating confidence scores...\n")


with torch.no_grad():

    for index, (image_path, true_label) in enumerate(samples):

        try:

            # ------------------------------------------------
            # Load image
            # ------------------------------------------------

            image = Image.open(image_path).convert("RGB")


            # ------------------------------------------------
            # Apply preprocessing
            # ------------------------------------------------

            image_tensor = test_transform(image)


            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)


            # Move to GPU/CPU
            image_tensor = image_tensor.to(DEVICE)


            # ------------------------------------------------
            # Model prediction
            # ------------------------------------------------

            outputs = model(image_tensor)


            # ------------------------------------------------
            # Convert logits to probabilities
            # ------------------------------------------------

            probabilities = torch.softmax(
                outputs,
                dim=1
            )


            # ------------------------------------------------
            # Get prediction
            # ------------------------------------------------

            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()


            # ------------------------------------------------
            # Get probabilities
            # ------------------------------------------------

            normal_probability = probabilities[
                0, 0
            ].item()

            glaucoma_probability = probabilities[
                0, 1
            ].item()


            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------
            #
            # Confidence = probability of the predicted class
            #

            confidence = probabilities[
                0,
                predicted_class
            ].item()


            # ------------------------------------------------
            # Correct / Incorrect
            # ------------------------------------------------

            correct = (
                predicted_class == true_label
            )


            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            results.append({

                "filename": image_path.name,

                "true_class":
                    CLASS_NAMES[true_label],

                "predicted_class":
                    CLASS_NAMES[predicted_class],

                "confidence":
                    confidence,

                "normal_probability":
                    normal_probability,

                "glaucoma_probability":
                    glaucoma_probability,

                "correct":
                    correct
            })


            # ------------------------------------------------
            # Print result
            # ------------------------------------------------

            print(
                f"{index + 1:03d} | "
                f"{image_path.name:30s} | "
                f"True: {CLASS_NAMES[true_label]:8s} | "
                f"Pred: {CLASS_NAMES[predicted_class]:8s} | "
                f"Confidence: {confidence:.4f} | "
                f"Normal: {normal_probability:.4f} | "
                f"Glaucoma: {glaucoma_probability:.4f} | "
                f"{'CORRECT' if correct else 'WRONG'}"
            )


        except Exception as e:

            print(
                f"ERROR processing {image_path.name}: {e}"
            )


# ============================================================
# 10. SAVE RESULTS TO CSV
# ============================================================

with open(
    OUTPUT_CSV,
    mode="w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "filename",
            "true_class",
            "predicted_class",
            "confidence",
            "normal_probability",
            "glaucoma_probability",
            "correct"
        ]
    )

    writer.writeheader()

    writer.writerows(results)


# ============================================================
# 11. SUMMARY
# ============================================================

total = len(results)

correct_count = sum(
    1
    for result in results
    if result["correct"]
)

accuracy = (
    correct_count / total
    if total > 0
    else 0
)


print("\n" + "=" * 70)
print("CONFIDENCE SCORE ANALYSIS COMPLETE")
print("=" * 70)

print(f"Total images       : {total}")
print(f"Correct predictions: {correct_count}")
print(f"Wrong predictions  : {total - correct_count}")
print(f"Accuracy           : {accuracy:.4f}")

print("\nResults saved to:")
print(OUTPUT_CSV)

print("=" * 70)