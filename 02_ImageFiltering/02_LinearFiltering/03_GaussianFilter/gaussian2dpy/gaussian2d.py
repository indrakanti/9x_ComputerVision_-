import cv2
import numpy as np

def apply_gaussian_filter(image, kernel_size, sigma):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

if __name__ == "__main__":
    image = cv2.imread("../../01_Images/Checker_original.png", cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found!")
        exit(1)
    
    gaussian_filter_3x3 = apply_gaussian_filter(image, 3, 1.0)
    cv2.imwrite("../../01_Images/gaussian_filter_3x3.png", gaussian_filter_3x3)
    
    gaussian_filter_5x5 = apply_gaussian_filter(image, 5, 1.4)
    cv2.imwrite("../../01_Images/gaussian_filter_5x5.png", gaussian_filter_5x5)
    
    gaussian_filter_7x7 = apply_gaussian_filter(image, 7, 2.0)
    cv2.imwrite("../../01_Images/gaussian_filter_7x7.png", gaussian_filter_7x7)
    
    print("Gaussian filtering applied successfully!")
