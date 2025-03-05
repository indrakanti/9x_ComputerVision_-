import cv2
import numpy as np

def apply_gaussian_filter(image, kernel_size, sigma):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def downsample_image(image):
    return cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

def upsample_image(image, size):
    return cv2.resize(image, size)

def create_laplacian_pyramid(image, levels):
    gaussian_pyramid = [image]
    laplacian_pyramid = []
    
    for _ in range(1, levels):
        blurred = apply_gaussian_filter(gaussian_pyramid[-1], 5, 1.0)
        downsampled = downsample_image(blurred)
        gaussian_pyramid.append(downsampled)
    
    for i in range(levels - 1):
        upsampled = upsample_image(gaussian_pyramid[i + 1], (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
        laplacian = cv2.subtract(gaussian_pyramid[i], upsampled)
        laplacian_pyramid.append(laplacian)
    
    laplacian_pyramid.append(gaussian_pyramid[-1])  # Smallest level remains unchanged
    return laplacian_pyramid

if __name__ == "__main__":
    image = cv2.imread("../../01_Images/Checker_original.png", cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found!")
        exit(1)
    
    laplacian_pyramid = create_laplacian_pyramid(image, 3)
    
    cv2.imwrite("../../01_Images/laplacian_pyramid_level1.png", laplacian_pyramid[0])
    cv2.imwrite("../../01_Images/laplacian_pyramid_level2.png", laplacian_pyramid[1])
    
    print("Laplacian pyramid generated successfully!")
