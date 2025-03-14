#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat applyGaussianSmoothing(const cv::Mat& image, int kernelSize, double sigma) {
    cv::Mat smoothed;
    cv::GaussianBlur(image, smoothed, cv::Size(kernelSize, kernelSize), sigma);
    return smoothed;
}

cv::Mat applySobelFilter(const cv::Mat& image, cv::Mat& gradientMagnitude) {
    cv::Mat gradX, gradY;
    cv::Sobel(image, gradX, CV_64F, 1, 0, 3);
    cv::Sobel(image, gradY, CV_64F, 0, 1, 3);
    
    cv::magnitude(gradX, gradY, gradientMagnitude);
    
    cv::Mat absGradX, absGradY;
    cv::convertScaleAbs(gradX, absGradX);
    cv::convertScaleAbs(gradY, absGradY);
    
    cv::Mat edgeDetected;
    cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, edgeDetected);
    return edgeDetected;
}

cv::Mat applyLaplacianFilter(const cv::Mat& image) {
    cv::Mat laplacian, absLaplacian;
    cv::Laplacian(image, laplacian, CV_64F, 3);
    cv::convertScaleAbs(laplacian, absLaplacian);
    return absLaplacian;
}

int main() {
    cv::Mat image = cv::imread("../01_Images/original_image.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Image not found!" << std::endl;
        return -1;
    }
    
    // Step 1: Apply Gaussian Smoothing
    cv::Mat smoothedImage = applyGaussianSmoothing(image, 3, 1.0);
    cv::imwrite("../01_Images/smoothed_image.png", smoothedImage);
    
    // Step 2: Apply Sobel Filter
    cv::Mat gradientMagnitude;
    cv::Mat sobelEdges = applySobelFilter(smoothedImage, gradientMagnitude);
    cv::imwrite("../01_Images/sobel_edges.png", sobelEdges);
    cv::imwrite("../01_Images/gradient_magnitude.png", gradientMagnitude);
    
    // Step 3: Apply Laplacian Filter
    cv::Mat laplacianEdges = applyLaplacianFilter(smoothedImage);
    cv::imwrite("../01_Images/laplacian_edges.png", laplacianEdges);
    
    return 0;
}
