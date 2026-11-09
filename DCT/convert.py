from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Image path
# ============================================================

IMAGE_PATH = Path(
    r"C:\Users\udit2\Desktop\Glaucoma\DCT\drishtiGS_069_ROI.png"
)

# Change the path above if the image is stored elsewhere.


# ============================================================
# Load and resize image
# ============================================================

image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Match the resizing operation in your dataset.py
image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

# Convert to float32 and normalize to [0, 1]
image = image.astype(np.float32) / 255.0


# ============================================================
# Apply DCT independently to each RGB channel
# ============================================================

dct_output = np.empty_like(image, dtype=np.float32)

for channel_idx in range(3):

    # Extract one RGB channel
    channel = np.ascontiguousarray(image[:, :, channel_idx])

    # Apply 2D DCT
    coefficients = cv2.dct(channel)

    # Signed-log compression
    signed_log = (
        np.sign(coefficients)
        * np.log1p(np.abs(coefficients))
    )

    # Standardize independently for each channel
    mean = float(signed_log.mean())
    std = float(signed_log.std())

    if std > 1e-6:
        signed_log = (signed_log - mean) / std
    else:
        signed_log = signed_log - mean

    # Clip extreme values, matching your dataset.py
    dct_output[:, :, channel_idx] = np.clip(
        signed_log, -5.0, 5.0
    )


# ============================================================
# Visualize original image and DCT representation
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original image
axes[0, 0].imshow(image)
axes[0, 0].set_title("Original DRISHTI ROI (224 × 224)")
axes[0, 0].axis("off")

# Combined RGB DCT representation
axes[0, 1].imshow(dct_output)
axes[0, 1].set_title("RGB DCT: Signed-Log + Standardization")
axes[0, 1].axis("off")

# Individual channel visualization
channel_names = ["Red DCT", "Green DCT", "Blue DCT"]

for idx, ax in enumerate([axes[1, 0], axes[1, 1]]):
    channel_idx = idx
    im = ax.imshow(
        dct_output[:, :, channel_idx],
        cmap="gray",
        vmin=-5,
        vmax=5
    )
    ax.set_title(channel_names[channel_idx])
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()


# ============================================================
# Save the DCT representation
# ============================================================

output_path = IMAGE_PATH.with_name(
    IMAGE_PATH.stem + "_DCT.png"
)

# For visualization, map [-5, 5] to [0, 255].
# This saved PNG is for viewing, not for model input.
visualization = np.clip(
    (dct_output + 5.0) / 10.0 * 255.0,
    0,
    255
).astype(np.uint8)

cv2.imwrite(
    str(output_path),
    cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
)

print("Original image shape:", image.shape)
print("DCT output shape:", dct_output.shape)
print("DCT output range:", dct_output.min(), "to", dct_output.max())
print("Saved visualization:", output_path)
