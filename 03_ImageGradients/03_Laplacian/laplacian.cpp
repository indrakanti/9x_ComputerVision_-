#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat applyLaplacianFilter(const cv::Mat& image) {
    cv::Mat laplacian, absLaplacian;
    cv::Laplacian(image, laplacian, CV_64F, 3);
    cv::convertScaleAbs(laplacian, absLaplacian);
    return absLaplacian;
}

int main() {
    cv::Mat image = cv::imread("../01_Images/original.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Image not found!" << std::endl;
        return -1;
    }
    
    cv::Mat laplacianEdges = applyLaplacianFilter(image);
    cv::imwrite("../01_Images/laplacian_edges.png", laplacianEdges);
    
    return 0;
}