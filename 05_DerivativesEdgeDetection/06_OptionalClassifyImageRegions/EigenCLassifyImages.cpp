 #include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./eigenvalue_region_classification [image_path]" << std::endl;
        return -1;
    }

    std::string image_path = argv[1];
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    // Compute image gradients
    cv::Mat Ix, Iy;
    cv::Sobel(image, Ix, CV_64F, 1, 0, 3);
    cv::Sobel(image, Iy, CV_64F, 0, 1, 3);

    // Compute components of the structure tensor
    cv::Mat Ix2 = Ix.mul(Ix);
    cv::Mat Iy2 = Iy.mul(Iy);
    cv::Mat Ixy = Ix.mul(Iy);

    // Smooth the components
    cv::GaussianBlur(Ix2, Ix2, cv::Size(5, 5), 1.0);
    cv::GaussianBlur(Iy2, Iy2, cv::Size(5, 5), 1.0);
    cv::GaussianBlur(Ixy, Ixy, cv::Size(5, 5), 1.0);

    cv::Mat output = image.clone();
    cv::cvtColor(output, output, cv::COLOR_GRAY2BGR);

    for (int y = 5; y < image.rows - 5; y += 10) {
        for (int x = 5; x < image.cols - 5; x += 10) {
            double a = Ix2.at<double>(y, x);
            double b = Ixy.at<double>(y, x);
            double c = Iy2.at<double>(y, x);

            cv::Mat M = (cv::Mat_<double>(2, 2) << a, b, b, c);
            cv::Mat eigenvalues;
            cv::eigen(M, eigenvalues);

            double lambda1 = eigenvalues.at<double>(0);
            double lambda2 = eigenvalues.at<double>(1);
            double eps = 1e-3;

            if (lambda1 < eps && lambda2 < eps) {
                // Flat - no marking
            } else if (lambda1 > 10 * lambda2 || lambda2 > 10 * lambda1) {
                // Edge
                cv::circle(output, cv::Point(x, y), 2, cv::Scalar(0, 255, 255), -1); // Yellow
            } else {
                // Corner
                cv::circle(output, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1); // Red
            }
        }
    }

    cv::imwrite("../01_Images/eigenvalue_classification_result.png", output);
    std::cout << "Region classification based on eigenvalues saved as 'eigenvalue_classification_result.png'." << std::endl;

    return 0;
}
