#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat applyGaussianFilter(const cv::Mat& image, int kernelSize, double sigma) {
    cv::Mat result;
    cv::GaussianBlur(image, result, cv::Size(kernelSize, kernelSize), sigma);
    return result;
}

cv::Mat downsampleImage(const cv::Mat& image) {
    cv::Mat downsampled;
    cv::resize(image, downsampled, cv::Size(image.cols / 2, image.rows / 2));
    return downsampled;
}

std::vector<cv::Mat> createGaussianPyramid(const cv::Mat& image, int levels) {
    std::vector<cv::Mat> pyramid;
    pyramid.push_back(image);
    for (int i = 1; i < levels; ++i) {
        cv::Mat blurred = applyGaussianFilter(pyramid.back(), 5, 1.0);
        cv::Mat downsampled = downsampleImage(blurred);
        pyramid.push_back(downsampled);
    }
    return pyramid;
}

int main() {
    cv::Mat image = cv::imread("../../01_Images/Checker_original.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Image not found!" << std::endl;
        return -1;
    }
    
    std::vector<cv::Mat> gaussianPyramid = createGaussianPyramid(image, 3);
    
    cv::imwrite("../../01_Images/gaussian_pyramid_level1.png", gaussianPyramid[1]);
    cv::imwrite("../../01_Images/gaussian_pyramid_level2.png", gaussianPyramid[2]);
    
    return 0;
}
