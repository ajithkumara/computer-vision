import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load Fig.4 image
img = cv2.imread("../images/highlights_and_shadows.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# Parameters
a = 0.8
sigma = 70.0

x = np.arange(256)
transformation = np.minimum(x + a * 128 * np.exp(-((x-128)**2)/(2*sigma**2)), 255)

# Apply transformation on S plane
s_new = transformation[s]

# Recombine
hsv_new = cv2.merge([h, np.uint8(s_new), v])
result = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

# Show
plt.plot(x, transformation); plt.title("Vibrance Transformation")
plt.show()

cv2.imshow("Original", img)
cv2.imshow("Enhanced Vibrance", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
