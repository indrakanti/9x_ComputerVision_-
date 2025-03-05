#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

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

cv::Mat upsampleImage(const cv::Mat& image, cv::Size size) {
    cv::Mat upsampled;
    cv::resize(image, upsampled, size);
    return upsampled;
}

std::vector<cv::Mat> createLaplacianPyramid(const cv::Mat& image, int levels) {
    std::vector<cv::Mat> gaussianPyramid;
    std::vector<cv::Mat> laplacianPyramid;
    
    // Construct Gaussian Pyramid
    gaussianPyramid.push_back(image);
    for (int i = 1; i < levels; ++i) {
        cv::Mat blurred = applyGaussianFilter(gaussianPyramid.back(), 5, 1.0);
        cv::Mat downsampled = downsampleImage(blurred);
        gaussianPyramid.push_back(downsampled);
    }
    
    // Construct Laplacian Pyramid
    for (int i = 0; i < levels - 1; ++i) {
        cv::Mat upsampled = upsampleImage(gaussianPyramid[i + 1], gaussianPyramid[i].size());
        cv::Mat laplacian = gaussianPyramid[i] - upsampled;
        laplacianPyramid.push_back(laplacian);
    }
    laplacianPyramid.push_back(gaussianPyramid.back()); // Smallest level remains unchanged
    
    return laplacianPyramid;
}

int main() {
    cv::Mat image = cv::imread("../../01_Images/Checker_original.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Image not found!" << std::endl;
        return -1;
    }
    
    std::vector<cv::Mat> laplacianPyramid = createLaplacianPyramid(image, 3);
    
    cv::imwrite("../../01_Images/laplacian_pyramid_level1.png", laplacianPyramid[0]);
    cv::imwrite("../../01_Images/laplacian_pyramid_level2.png", laplacianPyramid[1]);
    
    return 0;
}
