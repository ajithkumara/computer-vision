import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Settings
IMG_PATH = "../images/highlights_and_shadows.jpg"
OUT_DIR = "../outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Load color image
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Could not load image: {IMG_PATH}")

# Convert BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to L*a*b*
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
L, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]

# Apply gamma correction to L channel
gamma = 0.6  # Adjust this value (0.4–0.8) for best visual result
L_corrected = np.power(L / 255.0, gamma) * 255.0
L_corrected = np.clip(L_corrected, 0, 255).astype(np.uint8)

# Reconstruct image
lab_corrected = cv2.merge([L_corrected, a, b])
img_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2RGB)

# Save and display comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(img)
ax1.set_title("Original")
ax1.axis("off")

ax2.imshow(img_corrected)
ax2.set_title(f"Gamma Corrected (γ = {gamma})")
ax2.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "q3_comparison.png"), dpi=120)
plt.show()

# Histograms
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(L.ravel(), bins=256, range=(0,255), color='gray', alpha=0.7)
plt.title("Original L* Channel Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(L_corrected.ravel(), bins=256, range=(0,255), color='red', alpha=0.7)
plt.title("Corrected L* Channel Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "q3_histograms.png"), dpi=120)
plt.show()

print(f"Gamma value used: γ = {gamma}")