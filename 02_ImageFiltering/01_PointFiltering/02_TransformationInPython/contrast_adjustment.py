import cv2
import numpy as np

def adjust_contrast(image, alpha):
    return np.clip(alpha * image, 0, 255).astype(np.uint8)

# Load an image in grayscale
image = cv2.imread('../01_Images/SyntheticImage_Base.png', cv2.IMREAD_GRAYSCALE)
contrast_adjusted = adjust_contrast(image, 1.5)

cv2.imwrite('contrast_adjustment.png', contrast_adjusted)
