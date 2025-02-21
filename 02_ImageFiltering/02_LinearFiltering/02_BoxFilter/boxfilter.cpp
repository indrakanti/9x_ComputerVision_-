#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm> // For std::clamp

cv::Mat applyBoxFilter(const cv::Mat& image, int kernelSize) {
    cv::Mat result;
    cv::blur(image, result, cv::Size(kernelSize, kernelSize));
    return result;
}

int main() {
    cv::Mat image = cv::imread("../01_Images/original_image.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Image not found!" << std::endl;
        return -1;
    }
    
    cv::Mat boxFilter3x3 = applyBoxFilter(image, 3);
    cv::imwrite("../01_Images/box_filter_3x3.png", boxFilter3x3);
    
    cv::Mat boxFilter5x5 = applyBoxFilter(image, 5);
    cv::imwrite("../01_Images/box_filter_5x5.png", boxFilter5x5);
    
    cv::Mat boxFilter7x7 = applyBoxFilter(image, 7);
    cv::imwrite("../01_Images/box_filter_7x7.png", boxFilter7x7);
    
    return 0;
}
