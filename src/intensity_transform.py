import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Step 1: Load the image
# ------------------------
img_path = '../images/im01.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"❌ Error: Could not load image from {img_path}")
    exit()

print(f"✅ Image loaded successfully. Shape: {img.shape}")

# ------------------------
# Step 2: Build Look-Up Table (LUT)
# Transformation matches Fig. 1a:
#   0–50   → 0–100   (dark pixels stretched)
#   50–150 → 100–255 (midtones expanded)
#   150–255 → 150–255 (brights preserved)
# ------------------------
lut = np.zeros(256, dtype=np.uint8)

for i in range(256):
    if i <= 50:
        # 0 → 0, 50 → 100 (slope = 2)
        lut[i] = i * 2
    elif i <= 150:
        # 50 → 100, 150 → 255 (slope ≈ 1.55)
        lut[i] = 100 + int((i - 50) * 1.55)
    else:
        # 150 → 150, 255 → 255 (slope = 1)
        lut[i] = i

# ------------------------
# Step 3: Apply transformation
# ------------------------
transformed = cv2.LUT(img, lut)

# ------------------------
# Step 4: Plot the transformation curve
# ------------------------
plt.figure(figsize=(8, 6))
plt.plot(lut, color='blue', linewidth=2, label='Intensity Mapping')
plt.title("Intensity Transformation Function (Fig. 1a)")
plt.xlabel("Input Pixel Value")
plt.ylabel("Output Pixel Value")
plt.grid(True, alpha=0.4)
plt.legend()
plt.xlim(0, 255)
plt.ylim(0, 255)
plt.savefig('../outputs/task1_curve.png', dpi=120, bbox_inches='tight')
plt.show()

# ------------------------
# Step 5: Save and compare results
# ------------------------
cv2.imwrite('../outputs/task1_result.png', transformed)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(img, cmap='gray')
ax1.set_title("Original")
ax1.axis('off')

ax2.imshow(transformed, cmap='gray')
ax2.set_title("After Transformation")
ax2.axis('off')

plt.tight_layout()
plt.savefig('../outputs/task1_comparison.png', dpi=120, bbox_inches='tight')
plt.show()

print("🎉 Task 1 completed!")
print("Saved outputs in ../outputs/:")
print("  - task1_result.png (transformed image)")
print("  - task1_curve.png (mapping curve)")
print("  - task1_comparison.png (side-by-side view)")
