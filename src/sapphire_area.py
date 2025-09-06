import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------
# Settings
# -------------------------
IMG_PATH = "../images/sapphire.jpg"
OUT_DIR = "../outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Load image
# -------------------------
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Could not load image: {IMG_PATH}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -------------------------
# (a) Segmentation: Use HSV to detect blue sapphires
# -------------------------
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

# Wider blue range
lower_blue = np.array([90, 40, 40])    # allow darker/less saturated blues
upper_blue = np.array([140, 255, 255]) # allow brighter/more saturated blues

mask = cv2.inRange(hsv, lower_blue, upper_blue)

# -------------------------
# (b) Morphological operations
# -------------------------
kernel = np.ones((7,7), np.uint8)

# First clean noise
mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Then fill holes
mask_filled = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

# -------------------------
# (c) Connected components
# -------------------------
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_filled, connectivity=8)

# Get areas (excluding background label 0)
areas = stats[1:, cv2.CC_STAT_AREA]
print(f"Detected components: {num_labels-1}")
print(f"Areas in pixels: {areas}")

# -------------------------
# (d) Compute actual areas
# -------------------------
f = 8.0  # mm (example focal length)
d = 480.0  # mm (example distance)
scale_factor = (f / d) ** 2  # mm² per pixel

actual_areas = areas * scale_factor
print(f"Actual areas (mm²): {actual_areas}")

# -------------------------
# Display results
# -------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0,0].imshow(img_rgb)
axes[0,0].set_title("Original")
axes[0,0].axis('off')

axes[0,1].imshow(mask, cmap='gray')
axes[0,1].set_title("Binary Mask (raw)")
axes[0,1].axis('off')

axes[1,0].imshow(mask_filled, cmap='gray')
axes[1,0].set_title("Mask (cleaned + holes filled)")
axes[1,0].axis('off')

axes[1,1].imshow(labels, cmap='tab20')
axes[1,1].set_title("Connected Components")
axes[1,1].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "q10_results_fixed.png"), dpi=120)
plt.show()

# -------------------------
# Save outputs
# -------------------------
cv2.imwrite(os.path.join(OUT_DIR, "q10_mask.png"), mask)
cv2.imwrite(os.path.join(OUT_DIR, "q10_mask_filled.png"), mask_filled)

print("✅ Sapphire segmentation completed (fixed).")
