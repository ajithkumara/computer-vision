import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Load the brain proton density image ---
img_path = "../images/brain_proton_density_slice.png"  # correct filename
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"❌ Could not load image: {img_path}")

print(f"✅ Brain image loaded, shape = {img.shape}")

# --- Function to create LUTs for selective enhancement ---
def create_lut(accent="white"):
    lut = np.zeros(256, dtype=np.uint8)

    for i in range(256):
        if accent == "white":
            # White matter = high intensity → enhance brighter values
            if i < 100:
                lut[i] = int(i * 0.4)  # suppress very dark
            elif i < 180:
                lut[i] = int(80 + (i - 100) * 2)  # expand mid-high
            else:
                lut[i] = 255  # saturate very bright
        elif accent == "gray":
            # Gray matter = mid intensity → enhance mid-range
            if i < 60:
                lut[i] = int(i * 0.4)  # suppress dark background
            elif i < 150:
                lut[i] = int((i - 60) * 2)  # stretch mid-tones
            else:
                lut[i] = 200  # cap very high intensities
    return lut

# --- Create LUTs ---
lut_white = create_lut("white")
lut_gray = create_lut("gray")

# --- Apply transformations ---
img_white = cv2.LUT(img, lut_white)
img_gray = cv2.LUT(img, lut_gray)

# --- Plot transformation curves ---
plt.figure(figsize=(10,5))
plt.plot(lut_white, label="White matter enhancement", color="blue")
plt.plot(lut_gray, label="Gray matter enhancement", color="red")
plt.title("Intensity Transformation Functions")
plt.xlabel("Input Pixel Value")
plt.ylabel("Output Pixel Value")
plt.legend()
plt.grid(alpha=0.4)
plt.savefig("../outputs/task2_curves.png", dpi=100, bbox_inches="tight")
plt.show()

# --- Show original vs enhanced images ---
fig, axes = plt.subplots(1, 3, figsize=(15,6))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(img_white, cmap="gray")
axes[1].set_title("White Matter Accentuated")
axes[1].axis("off")

axes[2].imshow(img_gray, cmap="gray")
axes[2].set_title("Gray Matter Accentuated")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("../outputs/task2_comparison.png", dpi=100, bbox_inches="tight")
plt.show()

# --- Save results ---
cv2.imwrite("../outputs/task2_white.png", img_white)
cv2.imwrite("../outputs/task2_gray.png", img_gray)

print("🎯 Task 2 completed! Results saved in ../outputs/")
