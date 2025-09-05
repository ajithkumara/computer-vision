import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Paths
img_path = '../images/brain_proton_density_slice.png'
out_dir = '../outputs'
os.makedirs(out_dir, exist_ok=True)

# Load image
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise SystemExit(f"Cannot load {img_path}")


def build_lut_for_tissue(mode='gray'):
    """
    Build LUT tailored for enhancing either gray or white matter.

    mode: 'gray' or 'white'
        - 'gray': enhance mid-range intensities (~80–120)
        - 'white': enhance higher intensities (~120–200), but avoid clipping
    """
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if mode == 'gray':
            # Boost mid-level intensities (gray matter)
            if i < 70:
                val = i * 0.9
            elif i < 110:
                val = i * 1.4
            else:
                val = i * 1.0
        else:  # mode == 'white'
            # Enhance white matter
            if i < 100:
                val = i * 0.8
            elif i < 160:
                val = i * 1.3
            else:
                val = i * 0.9
        lut[i] = np.uint8(np.clip(round(val), 0, 255))
    return lut


def plot_diagram_and_lut(lut, filename, title):
    """Plots the LUT mapping."""
    plt.figure(figsize=(8,6))
    xs = np.arange(256)
    plt.plot(xs, lut[xs], color='blue', linewidth=1.5, label='LUT mapping')
    plt.legend()
    plt.xlim(0,255)
    plt.ylim(0,255)
    plt.xlabel("Input Intensity")
    plt.ylabel("Output Intensity")
    plt.title(title)
    plt.grid(alpha=0.4)
    plt.savefig(os.path.join(out_dir, filename), dpi=120, bbox_inches='tight')
    plt.close()


# --- Part (a): Enhance Gray Matter ---
print("Enhancing gray matter...")
lut_gray = build_lut_for_tissue('gray')
plot_diagram_and_lut(lut_gray, "q2_gray_matter_curve.png",
                     "Intensity Transformation for Gray Matter")
transformed_gray = cv2.LUT(img, lut_gray)
cv2.imwrite(os.path.join(out_dir, "q2_gray_matter_enhanced.png"), transformed_gray)

# --- Part (b): Enhance White Matter ---
print("Enhancing white matter...")
lut_white = build_lut_for_tissue('white')
plot_diagram_and_lut(lut_white, "q2_white_matter_curve.png",
                     "Intensity Transformation for White Matter")
transformed_white = cv2.LUT(img, lut_white)
cv2.imwrite(os.path.join(out_dir, "q2_white_matter_enhanced.png"), transformed_white)

# --- Side-by-Side Comparison ---
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
axes[0].imshow(img, cmap='gray')
axes[0].set_title("Original")
axes[0].axis('off')

axes[1].imshow(transformed_gray, cmap='gray')
axes[1].set_title("Gray Matter Enhanced")
axes[1].axis('off')

axes[2].imshow(transformed_white, cmap='gray')
axes[2].set_title("White Matter Enhanced")
axes[2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "q2_comparison.png"), dpi=120, bbox_inches='tight')
plt.show()

print("✅ All outputs saved to", out_dir)
