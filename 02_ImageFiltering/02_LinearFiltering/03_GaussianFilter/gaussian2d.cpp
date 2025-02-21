#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm> // For std::clamp

cv::Mat applyGaussianFilter(const cv::Mat& image, int kernelSize, double sigma) {
    cv::Mat result;
    cv::GaussianBlur(image, result, cv::Size(kernelSize, kernelSize), sigma);
    return result;
}

int main() {
    cv::Mat image = cv::imread("../01_Images/Checker_original.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Image not found!" << std::endl;
        return -1;
    }
    
    cv::Mat gaussianFilter3x3 = applyGaussianFilter(image, 3, 1.0);
    cv::imwrite("../01_Images/gaussian_filter_3x3.png", gaussianFilter3x3);
    
    cv::Mat gaussianFilter5x5 = applyGaussianFilter(image, 5, 1.4);
    cv::imwrite("../01_Images/gaussian_filter_5x5.png", gaussianFilter5x5);
    
    cv::Mat gaussianFilter7x7 = applyGaussianFilter(image, 7, 2.0);
    cv::imwrite("../01_Images/gaussian_filter_7x7.png", gaussianFilter7x7);
    
    return 0;
}
