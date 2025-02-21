#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm> // For std::clamp

cv::Mat adjustBrightness(const cv::Mat& image, int value) {
    cv::Mat result;
    image.convertTo(result, -1, 1, value);
    return result;
}

cv::Mat adjustContrast(const cv::Mat& image, double alpha) {
    cv::Mat result;
    image.convertTo(result, -1, alpha, 0);
    return result;
}

cv::Mat gammaCorrection(const cv::Mat& image, double gamma) {
    cv::Mat result;
    cv::Mat lookup(1, 256, CV_8U);
    uchar* p = lookup.ptr();
    for (int i = 0; i < 256; ++i)
        p[i] = static_cast<uchar>(std::clamp(std::pow(i / 255.0, gamma) * 255.0, 0.0, 255.0));
    cv::LUT(image, lookup, result);
    return result;
}

cv::Mat invertImage(const cv::Mat& image) {
    return 255 - image;
}

cv::Mat thresholdImage(const cv::Mat& image, int threshValue = 128) {
    cv::Mat binary;
    cv::threshold(image, binary, threshValue, 255, cv::THRESH_BINARY);
    return binary;
}

int main() {
    cv::Mat image = cv::imread("../01_Images/original_image.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Image not found!" << std::endl;
        return -1;
    }
    
    cv::Mat brightnessAdjusted = adjustBrightness(image, 50);
    cv::imwrite("../01_Images/brightness_adjustment.png", brightnessAdjusted);
    
    cv::Mat darkened = adjustBrightness(image, -50);
    cv::imwrite("../01_Images/darkening.png", darkened);
    
    cv::Mat contrastAdjusted = adjustContrast(image, 1.5);
    cv::imwrite("../01_Images/contrast_adjustment.png", contrastAdjusted);
    
    cv::Mat gammaCorrected = gammaCorrection(image, 0.5);
    cv::imwrite("../01_Images/gamma_correction.png", gammaCorrected);
    
    cv::Mat inverted = invertImage(image);
    cv::imwrite("../01_Images/image_inversion.png", inverted);
    
    cv::Mat thresholded = thresholdImage(image, 128);
    cv::imwrite("../01_Images/thresholding.png", thresholded);
    
    return 0;
}
