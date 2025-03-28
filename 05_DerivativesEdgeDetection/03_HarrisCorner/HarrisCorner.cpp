#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Load the grayscale image
    cv::Mat image = cv::imread("../01_Images/original_image.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // Parameters for Harris Corner Detection
    int blockSize = 2;     // neighborhood size
    int apertureSize = 3;  // Sobel operator size
    double k = 0.04;       // Harris detector parameter

    // Harris corner detection
    cv::Mat harris_response;
    cv::cornerHarris(image, harris_response, blockSize, apertureSize, k);

    // Normalize and convert to 8-bit image
    cv::Mat harris_normalized;
    cv::normalize(harris_response, harris_normalized, 0, 255, cv::NORM_MINMAX);
    cv::convertScaleAbs(harris_normalized, harris_normalized);

    // Convert grayscale to BGR for visualization
    cv::Mat color_image;
    cv::cvtColor(image, color_image, cv::COLOR_GRAY2BGR);

    // Mark corners
    for (int y = 0; y < harris_normalized.rows; y++) {
        for (int x = 0; x < harris_normalized.cols; x++) {
            if ((int)harris_normalized.at<uchar>(y, x) > 125) {
                cv::circle(color_image, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), 1);
            }
        }
    }

    // Save the result
    cv::imwrite("../01_Images/harris_corners_visual.png", color_image);
    std::cout << "Harris corners detected and saved as harris_corners_visual.png" << std::endl;

    return 0;
}