#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

int main() {
    // Load grayscale images
    cv::Mat img1 = cv::imread("../01_Images/img1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("../01_Images/img2.png", cv::IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty()) {
        std::cerr << "Could not load images.\n";
        return -1;
    }

    // Create SIFT detector and descriptor extractor
    auto sift = cv::SIFT::create();

    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;

    // Detect keypoints and compute descriptors
    sift->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    sift->detectAndCompute(img2, cv::noArray(), kp2, desc2);

    // Match using BFMatcher with Euclidean distance
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(desc1, desc2, knn_matches, 2);

    // Apply Lowe's ratio test
    std::vector<cv::DMatch> good_matches;
    const float ratio_thresh = 0.75f;
    for (const auto& m : knn_matches) {
        if (m[0].distance < ratio_thresh * m[1].distance) {
            good_matches.push_back(m[0]);
        }
    }

    // Draw good matches
    cv::Mat output;
    cv::drawMatches(img1, kp1, img2, kp2, good_matches, output,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Save and report
    cv::imwrite("../01_Images/sift_descriptor_matches.png", output);
    std::cout << "Saved descriptor matches to sift_descriptor_matches.png" << std::endl;
    std::cout << "Good matches found: " << good_matches.size() << std::endl;

    return 0;
}
