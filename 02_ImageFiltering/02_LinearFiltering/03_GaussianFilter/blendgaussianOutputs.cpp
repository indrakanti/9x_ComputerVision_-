#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm> // For std::clamp

cv::Mat applyGaussianFilter(const cv::Mat& image, int kernelSize, double sigma) {
    cv::Mat result;
    cv::GaussianBlur(image, result, cv::Size(kernelSize, kernelSize), sigma);
    return result;
}

cv::Mat blendImages(const cv::Mat& original, const cv::Mat& filtered, double alpha) {
    cv::Mat blended;
    cv::addWeighted(original, alpha, filtered, 1.0 - alpha, 0.0, blended);
    return blended;
}

int main() {
    cv::Mat image = cv::imread("../../01_Images/Checker_original.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Image not found!" << std::endl;
        return -1;
    }
    
    cv::Mat gaussianFiltered = applyGaussianFilter(image, 5, 1.4);
    cv::imwrite("../../01_Images/gaussian_filter_5x5_new.png", gaussianFiltered);
    
    cv::Mat blendedImage = blendImages(image, gaussianFiltered, 0.6);
    cv::imwrite("../../01_Images/blended_image.png", blendedImage);
    
    return 0;
}
