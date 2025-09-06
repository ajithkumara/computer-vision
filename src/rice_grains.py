import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------
# Settings
# -------------------------
IMG_PATH = "../images/rice.png"
OUT_DIR = "../outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Load clean rice image
# -------------------------
img_clean = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
if img_clean is None:
    raise FileNotFoundError(f"Could not load image: {IMG_PATH}")

print(f"Image loaded: shape={img_clean.shape}, dtype={img_clean.dtype}")

# -------------------------
# (a) Add Gaussian noise and remove it
# -------------------------
# Add Gaussian noise
mean = 0
std = 25
noise_gaussian = np.random.normal(mean, std, img_clean.shape).astype(np.float32)
img_gaussian_noisy = img_clean + noise_gaussian
img_gaussian_noisy = np.clip(img_gaussian_noisy, 0, 255).astype(np.uint8)

# Remove Gaussian noise with Gaussian blur
img_gaussian_filtered = cv2.GaussianBlur(img_gaussian_noisy, (5, 5), sigmaX=1.0)

# -------------------------
# (b) Add salt-and-pepper noise and remove it
# -------------------------
# Generate salt-and-pepper noise: 0 or 255
prob = 0.02
noise_salt_pepper = np.random.choice([0, 255], size=img_clean.shape, p=[1-prob, prob])

# ✅ FIX: Convert noise to uint8 before adding
noise_salt_pepper = noise_salt_pepper.astype(np.uint8)

# Add noise to clean image
img_salt_pepper_noisy = cv2.add(img_clean, noise_salt_pepper)

# Remove salt-and-pepper noise with median filter
img_salt_pepper_filtered = cv2.medianBlur(img_salt_pepper_noisy, ksize=3)

# -------------------------
# (c) Otsu's thresholding
# -------------------------
_, thresh_gaussian = cv2.threshold(img_gaussian_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, thresh_salt_pepper = cv2.threshold(img_salt_pepper_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# -------------------------
# (d) Morphological operations
# -------------------------
kernel = np.ones((3,3), np.uint8)

# Remove small objects and fill holes
morph_gaussian = cv2.morphologyEx(thresh_gaussian, cv2.MORPH_OPEN, kernel, iterations=2)
morph_gaussian = cv2.morphologyEx(morph_gaussian, cv2.MORPH_CLOSE, kernel, iterations=2)

morph_salt_pepper = cv2.morphologyEx(thresh_salt_pepper, cv2.MORPH_OPEN, kernel, iterations=2)
morph_salt_pepper = cv2.morphologyEx(morph_salt_pepper, cv2.MORPH_CLOSE, kernel, iterations=2)

# -------------------------
# (e) Connected components to count grains
# -------------------------
num_labels_gaussian, _, _, _ = cv2.connectedComponentsWithStats(morph_gaussian, connectivity=8)
num_labels_salt_pepper, _, _, _ = cv2.connectedComponentsWithStats(morph_salt_pepper, connectivity=8)

count_gaussian = num_labels_gaussian - 1
count_salt_pepper = num_labels_salt_pepper - 1

print(f"Rice grains counted (Gaussian): {count_gaussian}")
print(f"Rice grains counted (Salt-and-Pepper): {count_salt_pepper}")

# -------------------------
# Display results
# -------------------------
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Row 1: Gaussian noise
axes[0,0].imshow(img_gaussian_noisy, cmap='gray')
axes[0,0].set_title("Noisy (Gaussian)")
axes[0,0].axis('off')

axes[0,1].imshow(img_gaussian_filtered, cmap='gray')
axes[0,1].set_title("Denoised")
axes[0,1].axis('off')

axes[0,2].imshow(thresh_gaussian, cmap='gray')
axes[0,2].set_title("Otsu Threshold")
axes[0,2].axis('off')

axes[0,3].imshow(morph_gaussian, cmap='gray')
axes[0,3].set_title(f"Final (Count: {count_gaussian})")
axes[0,3].axis('off')

# Row 2: Salt-and-pepper noise
axes[1,0].imshow(img_salt_pepper_noisy, cmap='gray')
axes[1,0].set_title("Noisy (Salt-and-Pepper)")
axes[1,0].axis('off')

axes[1,1].imshow(img_salt_pepper_filtered, cmap='gray')
axes[1,1].set_title("Denoised")
axes[1,1].axis('off')

axes[1,2].imshow(thresh_salt_pepper, cmap='gray')
axes[1,2].set_title("Otsu Threshold")
axes[1,2].axis('off')

axes[1,3].imshow(morph_salt_pepper, cmap='gray')
axes[1,3].set_title(f"Final (Count: {count_salt_pepper})")
axes[1,3].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "q9_results.png"), dpi=120)
plt.show()

# -------------------------
# Save outputs
# -------------------------
cv2.imwrite(os.path.join(OUT_DIR, "q9_morph_gaussian.png"), morph_gaussian)
cv2.imwrite(os.path.join(OUT_DIR, "q9_morph_salt_pepper.png"), morph_salt_pepper)

print("✅ Question 9 completed.")