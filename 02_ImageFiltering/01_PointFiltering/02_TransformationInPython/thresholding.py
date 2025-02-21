import cv2
import numpy as np

def threshold_image(image, thresh_value=128):
    _, binary = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY)
    return binary

# Load an image in grayscale
image = cv2.imread('../01_Images/SyntheticImage_Base.png', cv2.IMREAD_GRAYSCALE)
thresholded = threshold_image(image, 128)

cv2.imwrite('thresholding.png', thresholded)
