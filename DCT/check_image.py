import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = r"C:\Users\udit2\Desktop\Glaucoma\DCT\drishtiGS_069_ROI.png"

# Load image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Cannot load image: {image_path}")

# Resize and normalize
image = cv2.resize(image, (224, 224))
image = image.astype(np.float32) / 255.0

# Calculate 2D DCT
dct_image = cv2.dct(image)

# Separate DC coefficient from AC coefficients
ac_coefficients = dct_image.copy()
ac_coefficients[0, 0] = 0

# Log magnitude of AC coefficients
ac_log = np.log1p(np.abs(ac_coefficients))

# Normalize AC coefficients for visualization
ac_visual = cv2.normalize(
    ac_log, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

# Print numerical diagnostics
print("DCT minimum:", dct_image.min())
print("DCT maximum:", dct_image.max())
print("DC coefficient:", dct_image[0, 0])
print("AC absolute maximum:", np.abs(ac_coefficients).max())
print("AC absolute mean:", np.abs(ac_coefficients).mean())

# Display
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(ac_visual, cmap="gray", vmin=0, vmax=255)
plt.title("DCT: AC Coefficients Only")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(ac_log, cmap="inferno")
plt.title("Log Magnitude of AC Coefficients")
plt.colorbar()
plt.axis("off")

plt.tight_layout()
plt.show()

ac_coefficients = dct_image.copy()
ac_coefficients[0, 0] = 0

print("DC coefficient:", dct_image[0, 0])
print("Maximum absolute AC:", np.abs(ac_coefficients).max())
print("Mean absolute AC:", np.abs(ac_coefficients).mean())
print("AC standard deviation:", ac_coefficients.std())
print("99th percentile:", np.percentile(np.abs(ac_coefficients), 99))

import numpy as np
import matplotlib.pyplot as plt

# Use your existing DCT output
ac_coefficients = dct_image.copy()
ac_coefficients[0, 0] = 0

# Log magnitude
ac_log = np.log1p(np.abs(ac_coefficients))

# Use the 99th percentile for display scaling
vmax = np.percentile(ac_log, 99)

plt.figure(figsize=(8, 7))

plt.imshow(
    ac_log,
    cmap="gray",
    vmin=0,
    vmax=vmax
)

plt.title("DCT AC Coefficients — Percentile Scaling")
plt.colorbar(label="Log magnitude")
plt.axis("off")

plt.tight_layout()
plt.show()