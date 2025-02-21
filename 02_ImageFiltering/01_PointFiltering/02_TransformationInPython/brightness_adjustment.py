import cv2
import numpy as np

def adjust_brightness(image, value):
    return np.clip(image + value, 0, 255).astype(np.uint8)

# Load an image in grayscale
image = cv2.imread('../01_Images/SyntheticImage_Base.png', cv2.IMREAD_GRAYSCALE)
brightness_adjusted = adjust_brightness(image, 50)

cv2.imwrite('brightness_adjustment.png', brightness_adjusted)
