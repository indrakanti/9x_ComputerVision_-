import cv2
import numpy as np

def invert_image(image):
    return 255 - image

# Load an image in grayscale
image = cv2.imread('../01_Images/SyntheticImage_Base.png', cv2.IMREAD_GRAYSCALE)
inverted = invert_image(image)

cv2.imwrite('image_inversion.png', inverted)
