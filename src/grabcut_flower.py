import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------
# Settings
# -------------------------
IMG_PATH = "../images/daisy.jpg"
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
# (a) GrabCut Segmentation
# -------------------------
# Define rectangle around flower (adjust coordinates if needed)
rect = (50, 50, 400, 400)  # x, y, w, h

# Create mask and background/foreground models
mask = np.zeros(img.shape[:2], dtype=np.uint8)
bgd_model = np.zeros((1, 65), dtype=np.float64)
fgd_model = np.zeros((1, 65), dtype=np.float64)

# Apply grabCut
cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Final mask: 0=background, 1=foreground
mask_final = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)

# Extract foreground and background
fg = cv2.bitwise_and(img, img, mask=mask_final)
bg = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask_final))

# -------------------------
# (b) Enhanced Image with Blurred Background
# -------------------------
# Blur background
bg_blur = cv2.GaussianBlur(bg, (25, 25), 0)

# Combine foreground with blurred background
enhanced = cv2.add(fg, bg_blur)

# -------------------------
# (c) Why is the background just beyond the edge of the flower quite dark?
# -------------------------
# Answer:
# The background just beyond the edge of the flower appears dark because:
# - GrabCut assigns uncertain pixels (near edges) to "probable background"
# - These pixels are treated as background and blurred
# - When combined with foreground, they become darker due to blending
# - This is especially noticeable when the original background has high contrast

# -------------------------
# Display results
# -------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0,0].imshow(img_rgb)
axes[0,0].set_title("Original")
axes[0,0].axis('off')

axes[0,1].imshow(mask_final, cmap='gray')
axes[0,1].set_title("Segmentation Mask")
axes[0,1].axis('off')

axes[0,2].imshow(fg)
axes[0,2].set_title("Foreground")
axes[0,2].axis('off')

axes[1,0].imshow(bg)
axes[1,0].set_title("Background")
axes[1,0].axis('off')

axes[1,1].imshow(bg_blur)
axes[1,1].set_title("Blurred Background")
axes[1,1].axis('off')

axes[1,2].imshow(enhanced)
axes[1,2].set_title("Enhanced Image")
axes[1,2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "q8_results.png"), dpi=120)
plt.show()

# -------------------------
# Save outputs
# -------------------------
cv2.imwrite(os.path.join(OUT_DIR, "q8_mask.png"), mask_final)
cv2.imwrite(os.path.join(OUT_DIR, "q8_foreground.png"), fg)
cv2.imwrite(os.path.join(OUT_DIR, "q8_background.png"), bg)
cv2.imwrite(os.path.join(OUT_DIR, "q8_enhanced.png"), enhanced)

print("✅ Question 8 completed.")