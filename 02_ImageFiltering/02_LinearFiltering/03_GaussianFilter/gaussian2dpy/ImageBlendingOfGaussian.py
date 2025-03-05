import cv2
import numpy as np

def apply_gaussian_filter(image, kernel_size, sigma):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def blend_images(original, filtered, alpha):
    return cv2.addWeighted(original, alpha, filtered, 1.0 - alpha, 0.0)

if __name__ == "__main__":
    image = cv2.imread("../01_Images/original_image.png", cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found!")
        exit(1)
    
    gaussian_filtered = apply_gaussian_filter(image, 5, 1.4)
    cv2.imwrite("../01_Images/gaussian_filtered.png", gaussian_filtered)
    
    blended_image = blend_images(image, gaussian_filtered, 0.6)
    cv2.imwrite("../01_Images/blended_image.png", blended_image)
    
    print("Gaussian filtering and blending applied successfully!")