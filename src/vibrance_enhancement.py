import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Settings
IMG_PATH = "../images/spider.png"
OUT_DIR = "../outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Load color image
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Could not load image: {IMG_PATH}")

# Convert BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to HSV
hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

# Define parameters
sigma = 70
a = 0.6  # Tune this between 0 and 1 for best visual result

# Apply vibrance transformation to saturation channel
def vibrance_transform(x, a, sigma):
    exp_term = np.exp(-((x - 128)**2) / (2 * sigma**2))
    return np.clip(x + a * 128 * exp_term, 0, 255)

s_enhanced = vibrance_transform(s, a, sigma).astype(np.uint8)

# Recombine HSV
hsv_enhanced = cv2.merge([h, s_enhanced, v])
img_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)

# Plot transformation curve
x = np.arange(256)
y = vibrance_transform(x, a, sigma)
plt.figure(figsize=(8, 5))
plt.plot(x, y, color='black', linewidth=2)
plt.title("Vibrance Enhancement Transformation")
plt.xlabel("Input Intensity")
plt.ylabel("Output Intensity")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUT_DIR, "task4_curve.png"), dpi=120)
plt.show()

# Display results
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img_rgb)
axes[0].set_title("Original")
axes[0].axis('off')

axes[1].imshow(img_enhanced)
axes[1].set_title(f"Vibrance Enhanced (a = {a:.2f})")
axes[1].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "task4_comparison.png"), dpi=120)
plt.show()

# Save output
cv2.imwrite(os.path.join(OUT_DIR, "task4_vibrance_enhanced.png"), cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2BGR))

print(f"Task 4 completed. a = {a}")