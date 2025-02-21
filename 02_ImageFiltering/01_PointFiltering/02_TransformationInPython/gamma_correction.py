import cv2
import numpy as np

def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Load an image in grayscale
image = cv2.imread('../01_Images/SyntheticImage_Base.png', cv2.IMREAD_GRAYSCALE)
gamma_corrected = gamma_correction(image, 0.5)

cv2.imwrite('gamma_correction.png', gamma_corrected)
