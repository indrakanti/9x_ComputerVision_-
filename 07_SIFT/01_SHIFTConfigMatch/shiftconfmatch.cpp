#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

int main() {
    // Load configuration
    cv::FileStorage fs("../config/sift_config.yml", cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open config file." << std::endl;
        return -1;
    }

    int num_octaves, num_scales;
    float contrast_thresh, edge_thresh, sigma, ratio_thresh;

    fs["sift"]["num_octaves"] >> num_octaves;
    fs["sift"]["num_scales_per_octave"] >> num_scales;
    fs["sift"]["contrast_threshold"] >> contrast_thresh;
    fs["sift"]["edge_threshold"] >> edge_thresh;
    fs["sift"]["sigma"] >> sigma;
    fs["matcher"]["ratio_threshold"] >> ratio_thresh;

    // Load images
    cv::Mat img1 = cv::imread("../01_Images/img1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("../01_Images/img2.png", cv::IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty()) {
        std::cerr << "Failed to load images." << std::endl;
        return -1;
    }

    // Create SIFT detector with loaded parameters
    auto sift = cv::SIFT::create(0, num_octaves, contrast_thresh, edge_thresh, sigma);

    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;

    sift->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    sift->detectAndCompute(img2, cv::noArray(), kp2, desc2);

    // Match descriptors using BFMatcher + Lowe’s ratio test
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(desc1, desc2, knn_matches, 2);

    std::vector<cv::DMatch> good_matches;
    for (const auto& m : knn_matches) {
        if (m[0].distance < ratio_thresh * m[1].distance) {
            good_matches.push_back(m[0]);
        }
    }

    // Draw matches
    cv::Mat output;
    cv::drawMatches(img1, kp1, img2, kp2, good_matches, output);
    cv::imwrite("../01_Images/sift_matched_output.png", output);

    std::cout << "Matching complete. Output saved as sift_matched_output.png" << std::endl;
    return 0;
}
