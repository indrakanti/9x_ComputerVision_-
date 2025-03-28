#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./taylor_series_plane [original|realistic|gradient]" << std::endl;
        return -1;
    }

    std::string choice = argv[1];
    std::string image_path;

    if (choice == "original") {
        image_path = "../01_Images/original_image.png";
    } else if (choice == "realistic") {
        image_path = "../01_Images/realistic_patch_image.png";
    } else if (choice == "gradient") {
        image_path = "../01_Images/gradient_patch_image.png";
    } else {
        std::cerr << "Invalid option. Use one of: original, realistic, gradient" << std::endl;
        return -1;
    }

    // Load grayscale image
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // Define a small patch at the center
    int cx = image.cols / 2;
    int cy = image.rows / 2;
    int patch_size = 11;
    cv::Rect patch_roi(cx - patch_size / 2, cy - patch_size / 2, patch_size, patch_size);
    cv::Mat patch = image(patch_roi);

    // Compute image gradients in the full image
    cv::Mat Ix, Iy;
    cv::Sobel(image, Ix, CV_64F, 1, 0, 3);
    cv::Sobel(image, Iy, CV_64F, 0, 1, 3);

    // Extract gradients from the patch
    cv::Mat gx = Ix(patch_roi);
    cv::Mat gy = Iy(patch_roi);

    // Estimate intensity at shifted positions using Taylor approximation
    double u = 1.0, v = 1.0;  // Small shift
    cv::Mat taylor_estimate = patch + gx * u + gy * v;

    // Normalize and save for visualization
    cv::Mat norm_patch, norm_estimate;
    cv::normalize(patch, norm_patch, 0, 255, cv::NORM_MINMAX);
    cv::normalize(taylor_estimate, norm_estimate, 0, 255, cv::NORM_MINMAX);
    norm_patch.convertTo(norm_patch, CV_8U);
    norm_estimate.convertTo(norm_estimate, CV_8U);

    cv::imwrite("../01_Images/patch_original.png", norm_patch);
    cv::imwrite("../01_Images/patch_taylor_approximation.png", norm_estimate);

    std::cout << "Taylor approximation of image patch shift saved." << std::endl;
    return 0;
}
