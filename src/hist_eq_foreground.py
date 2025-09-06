import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------
# Settings
# -------------------------
IMG_PATH = "../images/daisy.jpg"  # Figure 5 = daisy
OUT_DIR = "../outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Load image
# -------------------------
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Could not load image: {IMG_PATH}")

# Convert to RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to HSV
hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

# -------------------------
# Step (a): Display H, S, V planes
# -------------------------
fig1, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(h, cmap='gray')
axes[0].set_title("Hue Plane")
axes[0].axis('off')

axes[1].imshow(s, cmap='gray')
axes[1].set_title("Saturation Plane")
axes[1].axis('off')

axes[2].imshow(v, cmap='gray')
axes[2].set_title("Value Plane")
axes[2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "q5_hsv_planes.png"), dpi=120)
plt.show()

# -------------------------
# Step (b): Threshold to extract foreground mask
# -------------------------
# Use Value (V) plane: flower is brighter than background
_, mask = cv2.threshold(v, 100, 255, cv2.THRESH_BINARY)

# Optional: Clean up mask
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close holes
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise

# -------------------------
# Step (c): Extract foreground & compute histogram
# -------------------------
fg = cv2.bitwise_and(v, v, mask=mask)  # Use Value channel for equalization

# Compute histogram of foreground
hist_fg = cv2.calcHist([fg], [0], mask, [256], [0,256]).flatten()

# -------------------------
# Step (d): Cumulative sum of histogram
# -------------------------
cdf = np.cumsum(hist_fg)
cdf_normalized = cdf * 255 / (cdf[-1] + 1e-8)  # Normalize
lut = np.clip(cdf_normalized, 0, 255).astype(np.uint8)

# -------------------------
# Step (e): Apply histogram equalization to foreground
# -------------------------
v_eq = cv2.LUT(v, lut)  # Equalize full Value channel using foreground CDF

# But apply only to foreground
v_enhanced = np.copy(v)
v_enhanced[mask > 0] = v_eq[mask > 0]  # Only update foreground pixels

# Recombine
hsv_enhanced = cv2.merge([h, s, v_enhanced])
img_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)

# -------------------------
# Step (f): Extract background and combine
# -------------------------
# Background remains original
v_bg = v.copy()
v_bg[mask > 0] = 0  # Zero out foreground
bg_hsv = cv2.merge([h, s, v_bg])
bg_rgb = cv2.cvtColor(bg_hsv, cv2.COLOR_HSV2RGB)

# Final = enhanced foreground + original background
final = np.where(mask[..., None] > 0, img_enhanced, img_rgb)

# -------------------------
# Save and Display Results
# -------------------------
fig2, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0,0].imshow(img_rgb)
axes[0,0].set_title("Original Image")
axes[0,0].axis('off')

axes[0,1].imshow(h, cmap='gray')
axes[0,1].set_title("Hue Plane")
axes[0,1].axis('off')

axes[0,2].imshow(s, cmap='gray')
axes[0,2].set_title("Saturation Plane")
axes[0,2].axis('off')

axes[1,0].imshow(v, cmap='gray')
axes[1,0].set_title("Value Plane")
axes[1,0].axis('off')

axes[1,1].imshow(mask, cmap='gray')
axes[1,1].set_title("Foreground Mask")
axes[1,1].axis('off')

axes[1,2].imshow(final)
axes[1,2].set_title("Result (Equalized Foreground)")
axes[1,2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "q5_result.png"), dpi=120)
plt.show()

# -------------------------
# Histogram Comparison (Optional)
# -------------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(hist_fg, color='blue', label='Foreground (Original)')
plt.title("Foreground Histogram (Before)")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)

# New histogram after equalization
hist_eq = cv2.calcHist([v_enhanced], [0], mask, [256], [0,256]).flatten()
plt.subplot(1, 2, 2)
plt.plot(hist_eq, color='red', label='Foreground (Equalized)')
plt.title("Foreground Histogram (After)")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "q5_histograms.png"), dpi=120)
plt.show()

# -------------------------
# Save output
# -------------------------
cv2.imwrite(os.path.join(OUT_DIR, "q5_mask.png"), mask)
cv2.imwrite(os.path.join(OUT_DIR, "q5_enhanced_foreground.png"), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))

print("✅ Question 5 completed.")
print("📁 Outputs saved to:", OUT_DIR)