#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./harris_response [image_path]" << std::endl;
        return -1;
    }

    std::string image_path = argv[1];
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    // Convert to float
    cv::Mat img_float;
    image.convertTo(img_float, CV_32F);

    // Compute gradients
    cv::Mat Ix, Iy;
    cv::Sobel(img_float, Ix, CV_32F, 1, 0, 3);
    cv::Sobel(img_float, Iy, CV_32F, 0, 1, 3);

    // Compute products of derivatives
    cv::Mat Ix2 = Ix.mul(Ix);
    cv::Mat Iy2 = Iy.mul(Iy);
    cv::Mat Ixy = Ix.mul(Iy);

    // Apply Gaussian filtering
    cv::GaussianBlur(Ix2, Ix2, cv::Size(5, 5), 1.0);
    cv::GaussianBlur(Iy2, Iy2, cv::Size(5, 5), 1.0);
    cv::GaussianBlur(Ixy, Ixy, cv::Size(5, 5), 1.0);

    // Compute Harris response
    double k = 0.04;
    cv::Mat detM = Ix2.mul(Iy2) - Ixy.mul(Ixy);
    cv::Mat traceM = Ix2 + Iy2;
    cv::Mat R = detM - k * traceM.mul(traceM);

    // Normalize and threshold
    cv::Mat R_norm, R_norm_scaled;
    cv::normalize(R, R_norm, 0, 255, cv::NORM_MINMAX, CV_32F);
    R_norm.convertTo(R_norm_scaled, CV_8U);

    // Draw corners
    cv::Mat result;
    cv::cvtColor(image, result, cv::COLOR_GRAY2BGR);
    for (int y = 1; y < R.rows - 1; ++y) {
        for (int x = 1; x < R.cols - 1; ++x) {
            float val = R_norm.at<float>(y, x);
            if (val > 125 &&
                val > R_norm.at<float>(y - 1, x) && val > R_norm.at<float>(y + 1, x) &&
                val > R_norm.at<float>(y, x - 1) && val > R_norm.at<float>(y, x + 1)) {
                cv::circle(result, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
            }
        }
    }

    cv::imwrite("../01_Images/harris_response_result.png", result);
    std::cout << "Harris response result saved as 'harris_response_result.png'." << std::endl;
    return 0;
}
