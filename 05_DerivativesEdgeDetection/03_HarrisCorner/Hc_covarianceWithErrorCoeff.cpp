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

    // Compute Harris response (error function)
    double k = 0.04;
    cv::Mat detM = Sx2.mul(Sy2) - Sxy.mul(Sxy);
    cv::Mat traceM = Sx2 + Sy2;
    cv::Mat harris_response = detM - k * traceM.mul(traceM);

    // Normalize and save Harris response image
    cv::Mat harris_vis;
    cv::normalize(harris_response, harris_vis, 0, 255, cv::NORM_MINMAX);
    harris_vis.convertTo(harris_vis, CV_8U);
    cv::imwrite("../01_Images/harris_response.png", harris_vis);

    std::cout << "Covariance matrix components and Harris response saved as images." << std::endl;
    return 0;
}
