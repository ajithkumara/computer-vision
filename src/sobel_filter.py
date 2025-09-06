import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------
# Settings
# -------------------------
IMG_PATH = "../images/einstein.png"
OUT_DIR = "../outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Load image
# -------------------------
img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not load image: {IMG_PATH}")

print(f"Image loaded: shape={img.shape}, dtype={img.dtype}")

# -------------------------
# Sobel kernels
# -------------------------
Kx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])

Ky = np.array([[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]])

# -------------------------
# (a) Using cv2.filter2D
# -------------------------
grad_x_filter2d = cv2.filter2D(img, -1, Kx)
grad_y_filter2d = cv2.filter2D(img, -1, Ky)

mag_filter2d = np.sqrt(grad_x_filter2d ** 2 + grad_y_filter2d ** 2)
mag_filter2d = np.clip(mag_filter2d, 0, 255).astype(np.uint8)


# -------------------------
# (b) Manual convolution
# -------------------------
def sobel_manual(img, Kx, Ky):
    h, w = img.shape
    pad = 1
    img_padded = np.pad(img, pad, mode='constant', constant_values=0)

    grad_x = np.zeros((h, w), dtype=np.float32)
    grad_y = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            patch = img_padded[i:i + 3, j:j + 3]
            grad_x[i, j] = np.sum(patch * Kx)
            grad_y[i, j] = np.sum(patch * Ky)

    mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    mag = np.clip(mag, 0, 255).astype(np.uint8)

    return grad_x.astype(np.float32), grad_y.astype(np.float32), mag


grad_x_manual, grad_y_manual, mag_manual = sobel_manual(img, Kx, Ky)

# -------------------------
# (c) Using separability
# -------------------------
# Vertical kernel: [1; 2; 1]
vertical_kernel = np.array([[1], [2], [1]], dtype=np.float32)
# Horizontal kernel: [1 0 -1]
horizontal_kernel = np.array([[1, 0, -1]], dtype=np.float32)

# Step 1: Apply vertical then horizontal for X gradient
temp = cv2.filter2D(img, -1, vertical_kernel)
grad_x_separable = cv2.filter2D(temp, -1, horizontal_kernel)

# For Y gradient: apply horizontal first, then vertical
temp_y = cv2.filter2D(img, -1, horizontal_kernel)
grad_y_separable = cv2.filter2D(temp_y, -1, vertical_kernel)

# Magnitude
mag_separable = np.sqrt(grad_x_separable ** 2 + grad_y_separable ** 2)
mag_separable = np.clip(mag_separable, 0, 255).astype(np.uint8)

# -------------------------
# Display results
# -------------------------
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Original
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title("Original")
axes[0, 0].axis('off')

# Filter2D result
axes[0, 1].imshow(mag_filter2d, cmap='gray')
axes[0, 1].set_title("Sobel (filter2D)")
axes[0, 1].axis('off')

# Manual result
axes[0, 2].imshow(mag_manual, cmap='gray')
axes[0, 2].set_title("Sobel (Manual)")
axes[0, 2].axis('off')

# Separable result
axes[0, 3].imshow(mag_separable, cmap='gray')
axes[0, 3].set_title("Sobel (Separable)")
axes[0, 3].axis('off')

# X-gradient (filter2D)
axes[1, 0].imshow(grad_x_filter2d, cmap='gray')
axes[1, 0].set_title("X-gradient (filter2D)")
axes[1, 0].axis('off')

# Y-gradient (filter2D)
axes[1, 1].imshow(grad_y_filter2d, cmap='gray')
axes[1, 1].set_title("Y-gradient (filter2D)")
axes[1, 1].axis('off')

# X-gradient (Manual)
axes[1, 2].imshow(grad_x_manual, cmap='gray')
axes[1, 2].set_title("X-gradient (Manual)")
axes[1, 2].axis('off')

# Y-gradient (Manual)
axes[1, 3].imshow(grad_y_manual, cmap='gray')
axes[1, 3].set_title("Y-gradient (Manual)")
axes[1, 3].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "q6_comparison.png"), dpi=120)
plt.show()

# -------------------------
# Save outputs
# -------------------------
cv2.imwrite(os.path.join(OUT_DIR, "q6_filter2d.png"), mag_filter2d)
cv2.imwrite(os.path.join(OUT_DIR, "q6_manual.png"), mag_manual)
cv2.imwrite(os.path.join(OUT_DIR, "q6_separable.png"), mag_separable)

print("✅ Question 6 completed.")