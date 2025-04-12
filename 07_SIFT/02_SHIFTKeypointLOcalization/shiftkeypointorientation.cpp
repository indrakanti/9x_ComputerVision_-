#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

int main() {
    // Load grayscale image
    cv::Mat img = cv::imread("../01_Images/img1.png", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Could not load image.\n";
        return -1;
    }

    // Create SIFT detector with default parameters (includes localization + orientation)
    auto sift = cv::SIFT::create();

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    // Detect keypoints and compute descriptors
    sift->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

    // Draw keypoints with orientation (rich drawing mode shows orientation + scale)
    cv::Mat output;
    cv::drawKeypoints(img, keypoints, output, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Save result
    cv::imwrite("../01_Images/sift_keypoints_oriented.png", output);
    std::cout << "Saved oriented keypoints visualization.\n";

    return 0;
}
