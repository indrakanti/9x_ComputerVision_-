#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Load grayscale image
    cv::Mat image = cv::imread("../01_Images/original_image.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // Apply Gaussian blur to reduce noise
    cv::Mat blurred;
    cv::GaussianBlur(image, blurred, cv::Size(3, 3), 1.0);

    // Compute partial derivatives using Sobel
    cv::Mat grad_x, grad_y;
    cv::Sobel(blurred, grad_x, CV_64F, 1, 0, 3);
    cv::Sobel(blurred, grad_y, CV_64F, 0, 1, 3);

    // Compute gradient magnitude
    cv::Mat magnitude;
    cv::magnitude(grad_x, grad_y, magnitude);

    // Normalize and convert for display
    cv::Mat edge_map;
    cv::normalize(magnitude, edge_map, 0, 255, cv::NORM_MINMAX);
    edge_map.convertTo(edge_map, CV_8U);

    // Save edge detected image
    cv::imwrite("../01_Images/edge_detected.png", edge_map);

    std::cout << "Edge detection completed and saved as edge_detected.png" << std::endl;
    return 0;
}
