#include <opencv2/opencv.hpp>
#include <iostream>

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

int main() {
    cv::Mat image = cv::imread("../01_Images/original.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Image not found!" << std::endl;
        return -1;
    }
    
    cv::Mat gradientMagnitude;
    cv::Mat sobelEdges = applySobelFilter(image, gradientMagnitude);
    
    cv::imwrite("../01_Images/sobel_edges.png", sobelEdges);
    cv::imwrite("../01_Images/gradient_magnitude.png", gradientMagnitude);
    
    return 0;
}
