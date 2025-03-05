import cv2
import numpy as np

def apply_gaussian_filter(image, kernel_size, sigma):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def downsample_image(image):
    return cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

def create_gaussian_pyramid(image, levels):
    pyramid = [image]
    for _ in range(1, levels):
        blurred = apply_gaussian_filter(pyramid[-1], 5, 1.0)
        downsampled = downsample_image(blurred)
        pyramid.append(downsampled)
    return pyramid

if __name__ == "__main__":
    image = cv2.imread("../01_Images/original_image.png", cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found!")
        exit(1)
    
    gaussian_pyramid = create_gaussian_pyramid(image, 3)
    
    cv2.imwrite("../01_Images/gaussian_pyramid_level1.png", gaussian_pyramid[1])
    cv2.imwrite("../01_Images/gaussian_pyramid_level2.png", gaussian_pyramid[2])
    
    print("Gaussian pyramid generated successfully!")
