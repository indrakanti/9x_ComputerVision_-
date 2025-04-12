#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

int main() {
    cv::Mat img1 = cv::imread("../01_Images/img1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("../01_Images/img2.png", cv::IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty()) {
        std::cerr << "Could not load input images.\n";
        return -1;
    }

    // ORB: Fast and scale/rotation invariant
    auto orb = cv::ORB::create(500);

    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;

    orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    orb->detectAndCompute(img2, cv::noArray(), kp2, desc2);

    // Match descriptors
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(desc1, desc2, matches);

    // Draw good matches
    cv::Mat output;
    cv::drawMatches(img1, kp1, img2, kp2, matches, output);
    cv::imwrite("../01_Images/matched_output.png", output);

    std::cout << "Matched features drawn and saved to matched_output.png\n";
    return 0;
}
