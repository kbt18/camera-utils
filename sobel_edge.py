import cv2
import numpy as np

# Load image in grayscale
image = cv2.imread(r'C:\Users\kbtod\Pictures\Camera Roll\WIN_20250616_15_50_15_Pro.jpg', cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("Image 'input.jpg' not found.")

# Apply Sobel filter in X and Y direction
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges

# Compute the gradient magnitude
sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)

# Convert to 8-bit image (0–255) for display
sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude)

# Show result
cv2.imshow('Sobel Edge Detection', sobel_magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save output
cv2.imwrite('sobel_output.jpg', sobel_magnitude)
