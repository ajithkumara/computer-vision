import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

img_path = '../images/emma.jpg'   # keep your path
out_dir  = '../outputs'
os.makedirs(out_dir, exist_ok=True)

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise SystemExit(f"Cannot load {img_path}")

def build_lut(mode='lower'):
    """
    mode: 'lower' or 'upper'
      'lower' -> choose lut[50] = 50 and lut[150] = 150
      'upper' -> choose lut[50] = 100 and lut[150] = 255
    """
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if i < 50:
            # segment from (0,0) -> (50,50) OR (0,0)->(50,100) depending on mode
            if mode == 'upper':
                val = 2 * i            # 0->0, 50->100 (upper)
            else:
                val = i                # 0->0, 50->50  (lower)
        elif i == 50:
            val = 100 if mode == 'upper' else 50
        elif 50 < i < 150:
            # (50,100) -> (150,255) slope = 155/100 = 1.55
            val = 100.0 + (i - 50) * 1.55
        elif i == 150:
            val = 255 if mode == 'upper' else 150
        else:
            # (150,150) -> (255,255) slope = 1
            val = i
        # clip & cast
        lut[i] = np.uint8(np.clip(int(round(val)), 0, 255))
    return lut

def plot_diagram_and_lut(lut, filename):
    # plot the black diagram exactly (piecewise with vertical jumps)
    plt.figure(figsize=(8,6))
    # black diagram segments:
    plt.plot([0, 50], [0, 50], color='black', linewidth=2)        # (0,0)->(50,50)
    plt.plot([50, 50], [50, 100], color='black', linewidth=2)     # vertical up at x=50
    plt.plot([50, 150], [100, 255], color='black', linewidth=2)   # (50,100)->(150,255)
    plt.plot([150, 150], [255, 150], color='black', linewidth=2)  # vertical down at x=150
    plt.plot([150, 255], [150, 255], color='black', linewidth=2)  # (150,150)->(255,255)

    # overlay the actual LUT mapping (blue)
    xs = np.arange(256)
    plt.plot(xs, lut[xs], color='black', linewidth=1, label='LUT mapping (applied)')
    # mark the special points
    plt.scatter([50,50,150,150], [50,100,255,150], c='black', s=30)
    plt.legend()
    plt.xlim(0,255)
    plt.ylim(0,255)
    plt.xlabel("Input intensity")
    plt.ylabel("Output intensity")
    plt.title("Intensity Transformation (diagram in black + applied LUT in blue)")
    plt.grid(alpha=0.4)
    plt.savefig(os.path.join(out_dir, filename), dpi=120, bbox_inches='tight')
    plt.show()

# Choose mode: 'lower' to make segment go to (50,50) THEN (50,100) in the diagram plotting
mode = 'lower'   # change to 'upper' if you want LUT to map 50->100, 150->255
lut = build_lut(mode=mode)
plot_diagram_and_lut(lut, f'task1_curve_mode_{mode}.png')

# apply LUT and save
transformed = cv2.LUT(img, lut)
cv2.imwrite(os.path.join(out_dir, f'task1_result_mode_{mode}.png'), transformed)

# side-by-side comparison
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
ax1.imshow(img, cmap='gray'); ax1.set_title('Original'); ax1.axis('off')
ax2.imshow(transformed, cmap='gray'); ax2.set_title(f'Transformed (mode={mode})'); ax2.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f'task1_comparison_mode_{mode}.png'), dpi=120, bbox_inches='tight')
plt.show()
print("Saved outputs to", out_dir)
