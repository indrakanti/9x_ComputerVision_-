#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./eigenvalue_patch_analysis [original|realistic|gradient]" << std::endl;
        return -1;
    }

    std::string input = argv[1];
    std::string path;
    if (input == "original") path = "../01_Images/original_image.png";
    else if (input == "realistic") path = "../01_Images/realistic_patch_image.png";
    else if (input == "gradient") path = "../01_Images/gradient_patch_image.png";
    else {
        std::cerr << "Invalid input. Choose original, realistic, or gradient." << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    int cx = image.cols / 2, cy = image.rows / 2;
    int patch_size = 11;
    cv::Rect roi(cx - patch_size / 2, cy - patch_size / 2, patch_size, patch_size);
    cv::Mat patch = image(roi);

    // Compute gradients
    cv::Mat Ix, Iy;
    cv::Sobel(image, Ix, CV_64F, 1, 0, 3);
    cv::Sobel(image, Iy, CV_64F, 0, 1, 3);
    cv::Mat gx = Ix(roi);
    cv::Mat gy = Iy(roi);

    // Compute structure tensor components
    cv::Mat Ix2 = gx.mul(gx);
    cv::Mat Iy2 = gy.mul(gy);
    cv::Mat Ixy = gx.mul(gy);

    double sum_Ix2 = cv::sum(Ix2)[0];
    double sum_Iy2 = cv::sum(Iy2)[0];
    double sum_Ixy = cv::sum(Ixy)[0];

    // Build 2x2 matrix M
    cv::Mat M = (cv::Mat_<double>(2, 2) << sum_Ix2, sum_Ixy,
                                           sum_Ixy, sum_Iy2);

    // Compute eigenvalues
    cv::Mat eigenvalues;
    cv::eigen(M, eigenvalues);

    std::cout << "Eigenvalues of the patch covariance matrix:" << std::endl;
    std::cout << "Lambda1: " << eigenvalues.at<double>(0) << std::endl;
    std::cout << "Lambda2: " << eigenvalues.at<double>(1) << std::endl;

    // Interpretation
    double l1 = eigenvalues.at<double>(0);
    double l2 = eigenvalues.at<double>(1);
    if (l1 < 1e-3 && l2 < 1e-3)
        std::cout << "Region Type: Flat" << std::endl;
    else if ((l1 > 10 * l2) || (l2 > 10 * l1))
        std::cout << "Region Type: Edge (Taco-like)" << std::endl;
    else
        std::cout << "Region Type: Corner (Bowl-like)" << std::endl;

    return 0;
}
