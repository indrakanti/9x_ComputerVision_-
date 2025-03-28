#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Load grayscale image
    cv::Mat image = cv::imread("../01_Images/original_image.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Image not found!" << std::endl;
        return -1;
    }

    // Compute image gradients
    cv::Mat Ix, Iy;
    cv::Sobel(image, Ix, CV_64F, 1, 0, 3);
    cv::Sobel(image, Iy, CV_64F, 0, 1, 3);

    // Compute products of derivatives
    cv::Mat Ix2 = Ix.mul(Ix);
    cv::Mat Iy2 = Iy.mul(Iy);
    cv::Mat Ixy = Ix.mul(Iy);

    // Apply Gaussian filter to each product for smoothing
    cv::Mat Sx2, Sy2, Sxy;
    cv::GaussianBlur(Ix2, Sx2, cv::Size(7, 7), 2.0);
    cv::GaussianBlur(Iy2, Sy2, cv::Size(7, 7), 2.0);
    cv::GaussianBlur(Ixy, Sxy, cv::Size(7, 7), 2.0);

    // Visualize each component
    cv::Mat vis_x2, vis_y2, vis_xy;
    cv::normalize(Sx2, vis_x2, 0, 255, cv::NORM_MINMAX);
    cv::normalize(Sy2, vis_y2, 0, 255, cv::NORM_MINMAX);
    cv::normalize(Sxy, vis_xy, 0, 255, cv::NORM_MINMAX);

    vis_x2.convertTo(vis_x2, CV_8U);
    vis_y2.convertTo(vis_y2, CV_8U);
    vis_xy.convertTo(vis_xy, CV_8U);

    cv::imwrite("../01_Images/cov_Ix2.png", vis_x2);
    cv::imwrite("../01_Images/cov_Iy2.png", vis_y2);
    cv::imwrite("../01_Images/cov_Ixy.png", vis_xy);

    std::cout << "Covariance matrix components saved as images." << std::endl;
    return 0;
}
