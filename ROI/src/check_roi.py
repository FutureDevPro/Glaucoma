from pathlib import Path
from PIL import Image, ImageDraw


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = Path(
    r"C:\Users\udit2\Desktop\Glaucoma\dataset\DRISHTI\training\glaucoma\drishtiGS_069.png"
)

CENTER_PATH = Path(
    r"C:\Users\udit2\Desktop\Glaucoma\ROI\check_crop\drishtiGS_069_diskCenter_new.txt"
)

ROI_SIZE = 512

OUTPUT_PATH = Path(
    r"C:\Users\udit2\Desktop\Glaucoma\ROI\check_crop\diagnostic_drishitGS_007.png"
)


# ============================================================
# 1. CHECK FILES
# ============================================================

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )

if not CENTER_PATH.exists():
    raise FileNotFoundError(
        f"Center file not found:\n{CENTER_PATH}"
    )


# ============================================================
# 2. LOAD IMAGE
# ============================================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")

width, height = image.size


print("=" * 60)
print("OPTIC DISC CENTER DIAGNOSTIC")
print("=" * 60)

print()
print("Image:")
print(IMAGE_PATH)

print()
print("Image dimensions:")
print(f"Width  = {width}")
print(f"Height = {height}")


# ============================================================
# 3. READ DISK CENTER
# ============================================================

with open(
    CENTER_PATH,
    "r"
) as f:

    values = (
        f.read()
        .strip()
        .split()
    )


if len(values) < 2:
    raise ValueError(
        f"Center file does not contain two coordinates:\n"
        f"{CENTER_PATH}"
    )


cx = float(values[0])
cy = float(values[1])


print()
print("Center file:")
print(CENTER_PATH)

print()
print("Ground-truth center:")
print(f"cx = {cx}")
print(f"cy = {cy}")


# ============================================================
# 4. CHECK WHETHER CENTER IS INSIDE IMAGE
# ============================================================

inside = (
    0 <= cx < width
    and
    0 <= cy < height
)


print()
print("Center inside image:")

if inside:
    print("YES")
else:
    print("NO")

    print()
    print(
        "WARNING: The center coordinates do not match "
        "the image dimensions."
    )


# ============================================================
# 5. PROPOSE 512 × 512 ROI
# ============================================================

half = ROI_SIZE / 2

left = int(round(cx - half))
top = int(round(cy - half))

right = int(round(cx + half))
bottom = int(round(cy + half))


print()
print("Proposed ROI:")
print(
    f"left   = {left}"
)

print(
    f"top    = {top}"
)

print(
    f"right  = {right}"
)

print(
    f"bottom = {bottom}"
)


# ============================================================
# 6. CHECK WHETHER ROI FITS INSIDE IMAGE
# ============================================================

roi_inside = (
    left >= 0
    and
    top >= 0
    and
    right <= width
    and
    bottom <= height
)


print()
print("Entire ROI inside image:")

if roi_inside:
    print("YES")
else:
    print("NO")

    print()
    print(
        "WARNING: The requested ROI extends "
        "outside the image boundary."
    )


# ============================================================
# 7. DRAW CENTER + ROI ON ORIGINAL IMAGE
# ============================================================

diagnostic = image.copy()

draw = ImageDraw.Draw(
    diagnostic
)


# ------------------------------------------------------------
# Draw ROI rectangle
# ------------------------------------------------------------

draw.rectangle(
    [
        left,
        top,
        right,
        bottom
    ],
    outline=(255, 0, 0),
    width=5
)


# ------------------------------------------------------------
# Draw center point
# ------------------------------------------------------------

point_radius = 12

draw.ellipse(
    [
        cx - point_radius,
        cy - point_radius,
        cx + point_radius,
        cy + point_radius
    ],
    fill=(0, 255, 0),
    outline=(255, 255, 255),
    width=3
)


# ------------------------------------------------------------
# Draw crosshair
# ------------------------------------------------------------

draw.line(
    [
        cx - 30,
        cy,
        cx + 30,
        cy
    ],
    fill=(0, 255, 0),
    width=4
)

draw.line(
    [
        cx,
        cy - 30,
        cx,
        cy + 30
    ],
    fill=(0, 255, 0),
    width=4
)


# ============================================================
# 8. SAVE DIAGNOSTIC IMAGE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

diagnostic.save(
    OUTPUT_PATH
)


print()
print("Diagnostic image saved to:")
print(OUTPUT_PATH)

print()
print("=" * 60)
print("DONE")
print("=" * 60)