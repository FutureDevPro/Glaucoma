import torch
import torch.nn as nn

from torchvision import models, transforms

from PIL import Image
from pathlib import Path

import csv


# ============================================================
#                    CONFIGURATION
#        CHANGE ONLY THIS SECTION
# ============================================================


# ------------------------------------------------------------
# 1. MODEL CONFIGURATION
# ------------------------------------------------------------
#
# Put the path of each trained .pth file here.
#
# IMPORTANT:
# The architecture must match the architecture used
# to train the corresponding .pth file.
#
# Supported:
#   "resnet18"
#   "efficientnet_b0"
#   "mobilenet_v3_small"
#


MODELS = [

    {
        "name": "ResNet18",

        "architecture": "resnet18",

        "path": Path(
            r"C:\Users\udit2\Desktop\Glaucoma\models\resnet18_best.pth"
        ),

        # True if checkpoint contains:
        # checkpoint["model_state_dict"]
        "checkpoint_has_model_state_dict": True
    },


    {
        "name": "EfficientNet-B0",

        "architecture": "efficientnet_b0",

        "path": Path(
            r"C:\Users\udit2\Desktop\Glaucoma\models\efficientnet_b0_best.pth"
        ),

        "checkpoint_has_model_state_dict": True
    },


    {
        "name": "MobileNetV3-Small",

        "architecture": "mobilenet_v3_small",

        "path": Path(
            r"C:\Users\udit2\Desktop\Glaucoma\models\mobilenet_v3_small_best.pth"
        ),

        "checkpoint_has_model_state_dict": True
    }

]


# ------------------------------------------------------------
# 2. DATASET CONFIGURATION
# ------------------------------------------------------------
#
# The same three models will be tested on both datasets.
#
# Expected structure:
#
# ACRIMA/
#     testing/
#         normal/
#         glaucoma/
#
# DRISHTI/
#     testing/
#         normal/
#         glaucoma/
#


DATASETS = {

    "ACRIMA": Path(
        r"C:\Users\udit2\Desktop\Glaucoma\dataset\ACRIMA\testing"
    ),

    "DRISHTI": Path(
        r"C:\Users\udit2\Desktop\Glaucoma\dataset\DRISHTI\testing"
    )

}


# ------------------------------------------------------------
# 3. CLASS MAPPING
# ------------------------------------------------------------
#
# This MUST match the class mapping used during training.
#
# Your current dataset.py:
#
# 0 = normal
# 1 = glaucoma
#

CLASS_NAMES = {

    0: "normal",

    1: "glaucoma"

}


# ------------------------------------------------------------
# 4. NUMBER OF CLASSES
# ------------------------------------------------------------

NUM_CLASSES = 2


# ------------------------------------------------------------
# 5. IMAGE PREPROCESSING
# ------------------------------------------------------------
#
# This matches your current dataset.py evaluation transform:
#
# Resize -> ToTensor -> ImageNet Normalize
#
# Your current evaluation preprocessing uses 224x224.
#

IMAGE_SIZE = (224, 224)


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
# 6. OUTPUT DIRECTORY
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    r"C:\Users\udit2\Desktop\Glaucoma\results\model_comparison"
)


# ------------------------------------------------------------
# 7. DEVICE
# ------------------------------------------------------------

DEVICE = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"

)


# ============================================================
#              END OF CONFIGURATION
# ============================================================


print("=" * 80)
print("MULTI-MODEL CONFIDENCE SCORE ANALYSIS")
print("=" * 80)

print("Device:", DEVICE)

print("\nModels:")

for model_config in MODELS:

    print(
        f"  {model_config['name']} "
        f"({model_config['architecture']})"
    )

print("\nDatasets:")

for dataset_name in DATASETS:

    print(
        f"  {dataset_name}"
    )

print("=" * 80)


# ============================================================
# 8. PREPROCESSING
# ============================================================


test_transform = transforms.Compose([

    transforms.Resize(
        IMAGE_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=NORMALIZE_MEAN,

        std=NORMALIZE_STD

    )

])


# ============================================================
# 9. CREATE MODEL
# ============================================================


def create_model(
    architecture,
    num_classes
):

    architecture = architecture.lower()


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
    # MobileNetV3-Small
    # --------------------------------------------------------

    elif architecture == "mobilenet_v3_small":

        model = models.mobilenet_v3_small(
            weights=None
        )

        model.classifier[3] = nn.Linear(

            model.classifier[3].in_features,

            num_classes

        )


    # --------------------------------------------------------
    # Unknown architecture
    # --------------------------------------------------------

    else:

        raise ValueError(

            f"\nUnsupported architecture: "
            f"{architecture}\n\n"

            "Supported architectures:\n"

            "resnet18\n"

            "efficientnet_b0\n"

            "mobilenet_v3_small"

        )


    return model


# ============================================================
# 10. LOAD MODEL
# ============================================================


def load_model(
    model_config
):

    model_name = model_config["name"]

    architecture = model_config["architecture"]

    model_path = model_config["path"]

    checkpoint_has_model_state_dict = (
        model_config[
            "checkpoint_has_model_state_dict"
        ]
    )


    print("\n" + "-" * 80)

    print(
        f"Loading: {model_name}"
    )

    print(
        f"Architecture: {architecture}"
    )

    print(
        f"Path: {model_path}"
    )


    # Check model file

    if not model_path.exists():

        raise FileNotFoundError(

            f"\nModel file not found:\n"
            f"{model_path}"

        )


    # Create architecture

    model = create_model(

        architecture,

        NUM_CLASSES

    )


    # Load checkpoint

    checkpoint = torch.load(

        model_path,

        map_location=DEVICE

    )


    # --------------------------------------------------------
    # Load state dictionary
    # --------------------------------------------------------

    if checkpoint_has_model_state_dict:

        if not isinstance(
            checkpoint,
            dict
        ):

            raise ValueError(

                f"{model_name}: "
                "Expected checkpoint dictionary."

            )


        if "model_state_dict" not in checkpoint:

            raise KeyError(

                f"{model_name}: "
                "'model_state_dict' not found "
                "inside checkpoint."

            )


        state_dict = checkpoint[
            "model_state_dict"
        ]


    else:

        state_dict = checkpoint


    # Load weights

    model.load_state_dict(
        state_dict
    )


    # Device

    model = model.to(
        DEVICE
    )


    # Evaluation mode

    model.eval()


    print(
        f"{model_name} loaded successfully."
    )


    # Print checkpoint information

    if isinstance(
        checkpoint,
        dict
    ):

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


    return model


# ============================================================
# 11. FIND DATASET IMAGES
# ============================================================


IMAGE_EXTENSIONS = {

    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"

}


def get_samples(
    dataset_path
):

    samples = []


    for class_id, class_name in CLASS_NAMES.items():

        class_folder = (
            dataset_path /
            class_name
        )


        if not class_folder.exists():

            print(
                f"WARNING: "
                f"Class folder not found:\n"
                f"{class_folder}"
            )

            continue


        for image_path in sorted(
            class_folder.iterdir()
        ):

            if (

                image_path.is_file()

                and

                image_path.suffix.lower()
                in IMAGE_EXTENSIONS

            ):

                samples.append(

                    (
                        image_path,
                        class_id
                    )

                )


    return samples


# ============================================================
# 12. RUN MODEL ON DATASET
# ============================================================


def evaluate_model(
    model,
    model_name,
    dataset_name,
    samples
):

    results = []


    print("\n" + "=" * 80)

    print(
        f"{model_name} -> {dataset_name}"
    )

    print("=" * 80)


    with torch.no_grad():

        for index, (
            image_path,
            true_label
        ) in enumerate(samples):


            try:

                # ------------------------------------------------
                # Load image
                # ------------------------------------------------

                image = Image.open(
                    image_path
                ).convert("RGB")


                # ------------------------------------------------
                # Preprocessing
                # ------------------------------------------------

                image_tensor = (
                    test_transform(image)
                )


                # Add batch dimension

                image_tensor = (
                    image_tensor.unsqueeze(0)
                )


                # Move to GPU / CPU

                image_tensor = (
                    image_tensor.to(DEVICE)
                )


                # ------------------------------------------------
                # Forward pass
                # ------------------------------------------------

                outputs = model(
                    image_tensor
                )


                # ------------------------------------------------
                # Softmax
                # ------------------------------------------------

                probabilities = torch.softmax(

                    outputs,

                    dim=1

                )


                # ------------------------------------------------
                # Prediction
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

                    predicted_class
                    ==
                    true_label

                )


                # ------------------------------------------------
                # Result
                # ------------------------------------------------

                result = {

                    "dataset":
                        dataset_name,

                    "model":
                        model_name,

                    "filename":
                        image_path.name,

                    "true_class":
                        CLASS_NAMES[
                            true_label
                        ],

                    "predicted_class":
                        CLASS_NAMES[
                            predicted_class
                        ],

                    "confidence":
                        confidence,

                    "correct":
                        correct

                }


                # ------------------------------------------------
                # Store probability of every class
                # ------------------------------------------------

                for (

                    class_id,

                    class_name

                ) in CLASS_NAMES.items():

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
                # Print
                # ------------------------------------------------

                probability_text = " | ".join(

                    f"{class_name}: "
                    f"{probabilities[0, class_id].item():.4f}"

                    for (

                        class_id,

                        class_name

                    ) in CLASS_NAMES.items()

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

                    f"ERROR processing "
                    f"{image_path.name}: {e}"

                )


    return results


# ============================================================
# 13. SUMMARY CALCULATION
# ============================================================


def calculate_summary(
    results
):

    total = len(results)


    if total == 0:

        return {

            "total": 0,

            "correct": 0,

            "wrong": 0,

            "accuracy": 0,

            "average_confidence": 0,

            "correct_confidence": 0,

            "wrong_confidence": 0

        }


    correct_results = [

        result

        for result in results

        if result["correct"]

    ]


    wrong_results = [

        result

        for result in results

        if not result["correct"]

    ]


    average_confidence = (

        sum(

            result["confidence"]

            for result in results

        )

        / total

    )


    correct_confidence = (

        sum(

            result["confidence"]

            for result in correct_results

        )

        / len(correct_results)

        if len(correct_results) > 0

        else 0

    )


    wrong_confidence = (

        sum(

            result["confidence"]

            for result in wrong_results

        )

        / len(wrong_results)

        if len(wrong_results) > 0

        else 0

    )


    correct_count = len(
        correct_results
    )


    wrong_count = len(
        wrong_results
    )


    accuracy = (

        correct_count
        /
        total

    )


    return {

        "total":
            total,

        "correct":
            correct_count,

        "wrong":
            wrong_count,

        "accuracy":
            accuracy,

        "average_confidence":
            average_confidence,

        "correct_confidence":
            correct_confidence,

        "wrong_confidence":
            wrong_confidence

    }


# ============================================================
# 14. MAIN
# ============================================================


OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True

)


all_results = []

summary_results = []


# ------------------------------------------------------------
# Load all models once
# ------------------------------------------------------------

loaded_models = []


for model_config in MODELS:

    model = load_model(
        model_config
    )


    loaded_models.append({

        "name":
            model_config["name"],

        "model":
            model

    })


# ------------------------------------------------------------
# Process each dataset
# ------------------------------------------------------------

for dataset_name, dataset_path in DATASETS.items():


    print("\n\n" + "#" * 80)

    print(
        f"DATASET: {dataset_name}"
    )

    print(
        f"PATH: {dataset_path}"
    )

    print("#" * 80)


    if not dataset_path.exists():

        print(
            f"WARNING: Dataset not found:\n"
            f"{dataset_path}"
        )

        continue


    # Find images

    samples = get_samples(
        dataset_path
    )


    print(
        f"\nTotal images: {len(samples)}"
    )


    for class_id, class_name in CLASS_NAMES.items():

        count = sum(

            1

            for _, label in samples

            if label == class_id

        )


        print(

            f"{class_name.capitalize()} images: "
            f"{count}"

        )


    # --------------------------------------------------------
    # Run every model
    # --------------------------------------------------------

    for loaded_model in loaded_models:

        model_name = (
            loaded_model["name"]
        )

        model = (
            loaded_model["model"]
        )


        results = evaluate_model(

            model,

            model_name,

            dataset_name,

            samples

        )


        # Add to master results

        all_results.extend(
            results
        )


        # Summary

        summary = calculate_summary(
            results
        )


        summary[

            "dataset"

        ] = dataset_name


        summary[

            "model"

        ] = model_name


        summary_results.append(
            summary
        )


# ============================================================
# 15. SAVE IMAGE-LEVEL RESULTS
# ============================================================


if len(all_results) > 0:

    image_results_path = (

        OUTPUT_DIR /

        "all_model_confidence_scores.csv"

    )


    fieldnames = list(
        all_results[0].keys()
    )


    with open(

        image_results_path,

        "w",

        newline="",

        encoding="utf-8"

    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=fieldnames

        )


        writer.writeheader()


        writer.writerows(
            all_results
        )


    print(
        "\nImage-level results saved:"
    )

    print(
        image_results_path
    )


# ============================================================
# 16. SAVE SUMMARY
# ============================================================


if len(summary_results) > 0:

    summary_path = (

        OUTPUT_DIR /

        "model_confidence_summary.csv"

    )


    fieldnames = [

        "dataset",

        "model",

        "total",

        "correct",

        "wrong",

        "accuracy",

        "average_confidence",

        "correct_confidence",

        "wrong_confidence"

    ]


    with open(

        summary_path,

        "w",

        newline="",

        encoding="utf-8"

    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=fieldnames

        )


        writer.writeheader()


        writer.writerows(
            summary_results
        )


    print(
        "Summary saved:"
    )

    print(
        summary_path
    )


# ============================================================
# 17. PRINT FINAL COMPARISON
# ============================================================


print("\n\n")

print("=" * 100)

print("FINAL MODEL COMPARISON")

print("=" * 100)


print(

    f"{'Dataset':12s} | "

    f"{'Model':20s} | "

    f"{'Accuracy':10s} | "

    f"{'Avg Conf.':10s} | "

    f"{'Correct Conf.':14s} | "

    f"{'Wrong Conf.':12s}"

)


print("-" * 100)


for summary in summary_results:

    print(

        f"{summary['dataset']:12s} | "

        f"{summary['model']:20s} | "

        f"{summary['accuracy']:.4f}     | "

        f"{summary['average_confidence']:.4f}     | "

        f"{summary['correct_confidence']:.4f}         | "

        f"{summary['wrong_confidence']:.4f}"

    )


print("=" * 100)

print("\nAnalysis complete.")