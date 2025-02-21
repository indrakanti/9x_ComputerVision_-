import cv2
import numpy as np

def apply_box_filter(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))

if __name__ == "__main__":
    image = cv2.imread("../../01_Images/original_image.png", cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found!")
        exit(1)
    
    box_filter_3x3 = apply_box_filter(image, 3)
    cv2.imwrite("../01_Images/box_filter_3x3.png", box_filter_3x3)
    
    box_filter_5x5 = apply_box_filter(image, 5)
    cv2.imwrite("../01_Images/box_filter_5x5.png", box_filter_5x5)
    
    box_filter_7x7 = apply_box_filter(image, 7)
    cv2.imwrite("../01_Images/box_filter_7x7.png", box_filter_7x7)
    
    print("Box filtering applied successfully!")
