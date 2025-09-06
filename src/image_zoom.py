import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------
# Settings
# -------------------------
IMG_PATH_LARGE = "../images/im01.png"  # Large original
IMG_PATH_SMALL = "../images/im01small.png"  # Small version
OUT_DIR = "../outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Load images
# -------------------------
img_large = cv2.imread(IMG_PATH_LARGE)
img_small = cv2.imread(IMG_PATH_SMALL)

if img_large is None or img_small is None:
    raise FileNotFoundError("Could not load images. Check paths.")

# Convert to RGB
img_large_rgb = cv2.cvtColor(img_large, cv2.COLOR_BGR2RGB)
img_small_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

# -------------------------
# Zoom function (nearest-neighbor)
# -------------------------
def zoom_nearest_neighbor(img, scale):
    h, w = img.shape[:2]
    h_new, w_new = int(h * scale), int(w * scale)
    zoomed = np.zeros((h_new, w_new, 3), dtype=np.uint8)

    for i in range(h_new):
        for j in range(w_new):
            src_i = int(i / scale)
            src_j = int(j / scale)
            zoomed[i, j] = img[src_i, src_j]
    return zoomed

# -------------------------
# Zoom function (bilinear interpolation)
# -------------------------
def zoom_bilinear(img, scale):
    h, w = img.shape[:2]
    h_new, w_new = int(h * scale), int(w * scale)
    zoomed = np.zeros((h_new, w_new, 3), dtype=np.float32)

    for i in range(h_new):
        for j in range(w_new):
            src_i = i / scale
            src_j = j / scale

            # Find four surrounding pixels
            i0 = int(src_i)
            i1 = min(i0 + 1, h - 1)
            j0 = int(src_j)
            j1 = min(j0 + 1, w - 1)

            # Weights
            wi = src_i - i0
            wj = src_j - j0
            w00 = (1 - wi) * (1 - wj)
            w01 = (1 - wi) * wj
            w10 = wi * (1 - wj)
            w11 = wi * wj

            # Interpolate each channel
            for c in range(3):
                val = (
                    w00 * img[i0, j0, c] +
                    w01 * img[i0, j1, c] +
                    w10 * img[i1, j0, c] +
                    w11 * img[i1, j1, c]
                )
                zoomed[i, j, c] = val

    return zoomed.astype(np.uint8)

# -------------------------
# Apply zoom (scale=4)
# -------------------------
scale = 4
nn_zoomed = zoom_nearest_neighbor(img_small_rgb, scale)
bi_zoomed = zoom_bilinear(img_small_rgb, scale)

# -------------------------
# Compute normalized SSD
# -------------------------
def compute_normalized_ssd(orig, zoomed):
    diff = orig.astype(np.float32) - zoomed.astype(np.float32)
    ssd = np.sum(diff**2)
    norm = np.sum(orig.astype(np.float32)**2)
    return ssd / norm if norm != 0 else 0

ssd_nn = compute_normalized_ssd(img_large_rgb, nn_zoomed)
ssd_bi = compute_normalized_ssd(img_large_rgb, bi_zoomed)

print(f"Nearest-Neighbor SSD: {ssd_nn:.6f}")
print(f"Bilinear Interpolation SSD: {ssd_bi:.6f}")

# -------------------------
# Display results
# -------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0,0].imshow(img_large_rgb)
axes[0,0].set_title("Original")
axes[0,0].axis('off')

axes[0,1].imshow(nn_zoomed)
axes[0,1].set_title("Nearest-Neighbor (SSD: {:.6f})".format(ssd_nn))
axes[0,1].axis('off')

axes[0,2].imshow(bi_zoomed)
axes[0,2].set_title("Bilinear (SSD: {:.6f})".format(ssd_bi))
axes[0,2].axis('off')

# Repeat for im02
img_large_2 = cv2.imread("../images/im02.png")
img_small_2 = cv2.imread("../images/im02small.png")

img_large_2_rgb = cv2.cvtColor(img_large_2, cv2.COLOR_BGR2RGB)
img_small_2_rgb = cv2.cvtColor(img_small_2, cv2.COLOR_BGR2RGB)

nn_zoomed_2 = zoom_nearest_neighbor(img_small_2_rgb, scale)
bi_zoomed_2 = zoom_bilinear(img_small_2_rgb, scale)

ssd_nn_2 = compute_normalized_ssd(img_large_2_rgb, nn_zoomed_2)
ssd_bi_2 = compute_normalized_ssd(img_large_2_rgb, bi_zoomed_2)

print(f"im02 - Nearest-Neighbor SSD: {ssd_nn_2:.6f}")
print(f"im02 - Bilinear Interpolation SSD: {ssd_bi_2:.6f}")

axes[1,0].imshow(img_large_2_rgb)
axes[1,0].set_title("Original (im02)")
axes[1,0].axis('off')

axes[1,1].imshow(nn_zoomed_2)
axes[1,1].set_title("Nearest-Neighbor (SSD: {:.6f})".format(ssd_nn_2))
axes[1,1].axis('off')

axes[1,2].imshow(bi_zoomed_2)
axes[1,2].set_title("Bilinear (SSD: {:.6f})".format(ssd_bi_2))
axes[1,2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "q7_comparison.png"), dpi=120)
plt.show()

# -------------------------
# Save outputs
# -------------------------
cv2.imwrite(os.path.join(OUT_DIR, "q7_im01_nn.png"), cv2.cvtColor(nn_zoomed, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(OUT_DIR, "q7_im01_bi.png"), cv2.cvtColor(bi_zoomed, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(OUT_DIR, "q7_im02_nn.png"), cv2.cvtColor(nn_zoomed_2, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(OUT_DIR, "q7_im02_bi.png"), cv2.cvtColor(bi_zoomed_2, cv2.COLOR_RGB2BGR))

print("✅ Question 7 completed.")