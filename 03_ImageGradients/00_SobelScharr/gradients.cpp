#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string input_path;
    std::string output_dir;
    std::string method = "both";
    std::string border = "reflect101";
    bool self_test = false;
    bool help = false;
};

struct GradientResult {
    cv::Mat gx;
    cv::Mat gy;
    cv::Mat magnitude;
    cv::Mat direction_deg;
};

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program
        << " --input <image> --output-dir <directory> [--method sobel|scharr|both]"
           " [--border reflect101|replicate]\n"
        << "  " << program << " --self-test\n\n"
        << "Options:\n"
        << "  --input <path>        Input image path\n"
        << "  --output-dir <path>   Directory for output images\n"
        << "  --method <name>       sobel, scharr, or both (default: both)\n"
        << "  --border <name>       reflect101 or replicate (default: reflect101)\n"
        << "  --self-test           Run deterministic in-memory checks\n"
        << "  --help                Show this help text\n";
}

bool parseArguments(int argc, char** argv, Options& options) {
    auto requireValue = [argc, argv](int& index, const std::string& option) -> std::string {
        if (index + 1 >= argc) {
            throw std::runtime_error("Missing value for " + option);
        }
        ++index;
        return argv[index];
    };

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--input") {
                options.input_path = requireValue(i, arg);
            } else if (arg == "--output-dir") {
                options.output_dir = requireValue(i, arg);
            } else if (arg == "--method") {
                options.method = requireValue(i, arg);
            } else if (arg == "--border") {
                options.border = requireValue(i, arg);
            } else if (arg == "--self-test") {
                options.self_test = true;
            } else if (arg == "--help" || arg == "-h") {
                options.help = true;
            } else {
                throw std::runtime_error("Unknown option: " + arg);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "Argument error: " << error.what() << '\n';
        return false;
    }
    return true;
}

bool validateOptions(const Options& options) {
    if (options.self_test || options.help) {
        return true;
    }
    if (options.input_path.empty() || options.output_dir.empty()) {
        std::cerr << "Error: --input and --output-dir are required.\n";
        return false;
    }
    if (options.method != "sobel" && options.method != "scharr" && options.method != "both") {
        std::cerr << "Error: --method must be sobel, scharr, or both.\n";
        return false;
    }
    if (options.border != "reflect101" && options.border != "replicate") {
        std::cerr << "Error: --border must be reflect101 or replicate.\n";
        return false;
    }
    return true;
}

int borderType(const std::string& border) {
    return border == "replicate" ? cv::BORDER_REPLICATE : cv::BORDER_REFLECT_101;
}

cv::Mat toFloatGray(const cv::Mat& input) {
    cv::Mat gray;
    if (input.channels() == 1) {
        gray = input;
    } else {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat result;
    gray.convertTo(result, CV_32F);
    return result;
}

GradientResult finishGradientResult(cv::Mat gx, cv::Mat gy) {
    GradientResult result;
    result.gx = std::move(gx);
    result.gy = std::move(gy);
    cv::magnitude(result.gx, result.gy, result.magnitude);
    cv::phase(result.gx, result.gy, result.direction_deg, true);
    return result;
}

GradientResult computeSobel(const cv::Mat& image, int border_type) {
    cv::Mat gx;
    cv::Mat gy;
    cv::Sobel(image, gx, CV_32F, 1, 0, 3, 1.0, 0.0, border_type);
    cv::Sobel(image, gy, CV_32F, 0, 1, 3, 1.0, 0.0, border_type);
    return finishGradientResult(gx, gy);
}

GradientResult computeScharr(const cv::Mat& image, int border_type) {
    cv::Mat gx;
    cv::Mat gy;
    cv::Scharr(image, gx, CV_32F, 1, 0, 1.0, 0.0, border_type);
    cv::Scharr(image, gy, CV_32F, 0, 1, 1.0, 0.0, border_type);
    return finishGradientResult(gx, gy);
}

std::pair<cv::Mat, cv::Mat> sobelKernels() {
    const cv::Mat kx = (cv::Mat_<float>(3, 3) <<
        -1.0f, 0.0f, 1.0f,
        -2.0f, 0.0f, 2.0f,
        -1.0f, 0.0f, 1.0f);
    return {kx, kx.t()};
}

std::pair<cv::Mat, cv::Mat> scharrKernels() {
    const cv::Mat kx = (cv::Mat_<float>(3, 3) <<
        -3.0f, 0.0f, 3.0f,
        -10.0f, 0.0f, 10.0f,
        -3.0f, 0.0f, 3.0f);
    return {kx, kx.t()};
}

cv::Mat visualizeSigned(const cv::Mat& input) {
    double min_value = 0.0;
    double max_value = 0.0;
    cv::minMaxLoc(input, &min_value, &max_value);
    const double max_abs = std::max(std::abs(min_value), std::abs(max_value));
    if (max_abs < 1e-12) {
        return cv::Mat(input.size(), CV_8U, cv::Scalar(128));
    }

    cv::Mat display = input * static_cast<float>(127.0 / max_abs) + 128.0f;
    cv::Mat result;
    display.convertTo(result, CV_8U);
    return result;
}

cv::Mat visualizeMagnitude(const cv::Mat& magnitude) {
    double max_value = 0.0;
    cv::minMaxLoc(magnitude, nullptr, &max_value);
    if (max_value < 1e-12) {
        return cv::Mat(magnitude.size(), CV_8U, cv::Scalar(0));
    }

    cv::Mat result;
    magnitude.convertTo(result, CV_8U, 255.0 / max_value);
    return result;
}

cv::Mat visualizeDirection(const cv::Mat& direction_deg) {
    cv::Mat result;
    direction_deg.convertTo(result, CV_8U, 255.0 / 360.0);
    return result;
}

bool writeImage(const fs::path& path, const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image)) {
        std::cerr << "Error: failed to write " << path << '\n';
        return false;
    }
    return true;
}

bool writeGradientSet(const fs::path& output_dir,
                      const std::string& prefix,
                      const GradientResult& gradient) {
    bool ok = true;
    ok &= writeImage(output_dir / (prefix + "_gx_signed.png"), visualizeSigned(gradient.gx));
    ok &= writeImage(output_dir / (prefix + "_gy_signed.png"), visualizeSigned(gradient.gy));
    ok &= writeImage(output_dir / (prefix + "_magnitude.png"),
                     visualizeMagnitude(gradient.magnitude));
    ok &= writeImage(output_dir / (prefix + "_direction.png"),
                     visualizeDirection(gradient.direction_deg));
    return ok;
}

double maxAbsDifference(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat difference;
    cv::absdiff(a, b, difference);
    double max_value = 0.0;
    cv::minMaxLoc(difference, nullptr, &max_value);
    return max_value;
}

double interiorMean(const cv::Mat& image) {
    if (image.rows < 3 || image.cols < 3) {
        return cv::mean(image)[0];
    }
    return cv::mean(image(cv::Rect(1, 1, image.cols - 2, image.rows - 2)))[0];
}

double angleDistanceDegrees(double a, double b) {
    double difference = std::fmod(std::abs(a - b), 360.0);
    if (difference > 180.0) {
        difference = 360.0 - difference;
    }
    return difference;
}

bool runSelfTest() {
    bool passed = true;
    auto check = [&passed](bool condition, const std::string& name) {
        if (!condition) {
            std::cerr << "Self-test failed: " << name << '\n';
            passed = false;
        }
    };

    const int border = cv::BORDER_REFLECT_101;

    cv::Mat constant(16, 20, CV_32F, cv::Scalar(42.0f));
    const auto constant_sobel = computeSobel(constant, border);
    const auto constant_scharr = computeScharr(constant, border);
    check(maxAbsDifference(constant_sobel.magnitude,
                           cv::Mat::zeros(constant.size(), CV_32F)) <= 1e-5,
          "Sobel constant image has zero gradient");
    check(maxAbsDifference(constant_scharr.magnitude,
                           cv::Mat::zeros(constant.size(), CV_32F)) <= 1e-5,
          "Scharr constant image has zero gradient");

    cv::Mat ramp_x(16, 20, CV_32F);
    cv::Mat ramp_y(16, 20, CV_32F);
    cv::Mat ramp_xy(16, 20, CV_32F);
    for (int row = 0; row < ramp_x.rows; ++row) {
        for (int col = 0; col < ramp_x.cols; ++col) {
            ramp_x.at<float>(row, col) = static_cast<float>(col);
            ramp_y.at<float>(row, col) = static_cast<float>(row);
            ramp_xy.at<float>(row, col) = static_cast<float>(row + col);
        }
    }

    const auto sobel_x = computeSobel(ramp_x, border);
    check(interiorMean(sobel_x.gx) > 0.0, "Sobel x-ramp has positive Gx");
    check(std::abs(interiorMean(sobel_x.gy)) <= 1e-5, "Sobel x-ramp has zero Gy");
    check(angleDistanceDegrees(interiorMean(sobel_x.direction_deg), 0.0) <= 0.5,
          "Sobel x-ramp direction is 0 degrees");

    const auto sobel_y = computeSobel(ramp_y, border);
    check(std::abs(interiorMean(sobel_y.gx)) <= 1e-5, "Sobel y-ramp has zero Gx");
    check(interiorMean(sobel_y.gy) > 0.0, "Sobel y-ramp has positive Gy");
    check(angleDistanceDegrees(interiorMean(sobel_y.direction_deg), 90.0) <= 0.5,
          "Sobel y-ramp direction is 90 degrees");

    const auto sobel_xy = computeSobel(ramp_xy, border);
    check(angleDistanceDegrees(interiorMean(sobel_xy.direction_deg), 45.0) <= 0.5,
          "Sobel diagonal ramp direction is 45 degrees");

    const auto scharr_xy = computeScharr(ramp_xy, border);
    check(angleDistanceDegrees(interiorMean(scharr_xy.direction_deg), 45.0) <= 0.5,
          "Scharr diagonal ramp direction is 45 degrees");

    cv::Mat pattern(23, 31, CV_32F);
    for (int row = 0; row < pattern.rows; ++row) {
        for (int col = 0; col < pattern.cols; ++col) {
            pattern.at<float>(row, col) =
                static_cast<float>((row * 11 + col * 17 + (row * col) % 19) % 251);
        }
    }

    const auto [sobel_kx, sobel_ky] = sobelKernels();
    cv::Mat sobel_manual_x;
    cv::Mat sobel_manual_y;
    cv::filter2D(pattern, sobel_manual_x, CV_32F, sobel_kx,
                 cv::Point(-1, -1), 0.0, border);
    cv::filter2D(pattern, sobel_manual_y, CV_32F, sobel_ky,
                 cv::Point(-1, -1), 0.0, border);
    const auto sobel_reference = computeSobel(pattern, border);
    check(maxAbsDifference(sobel_manual_x, sobel_reference.gx) <= 1e-4,
          "Explicit Sobel Gx kernel matches cv::Sobel");
    check(maxAbsDifference(sobel_manual_y, sobel_reference.gy) <= 1e-4,
          "Explicit Sobel Gy kernel matches cv::Sobel");

    const auto [scharr_kx, scharr_ky] = scharrKernels();
    cv::Mat scharr_manual_x;
    cv::Mat scharr_manual_y;
    cv::filter2D(pattern, scharr_manual_x, CV_32F, scharr_kx,
                 cv::Point(-1, -1), 0.0, border);
    cv::filter2D(pattern, scharr_manual_y, CV_32F, scharr_ky,
                 cv::Point(-1, -1), 0.0, border);
    const auto scharr_reference = computeScharr(pattern, border);
    check(maxAbsDifference(scharr_manual_x, scharr_reference.gx) <= 1e-4,
          "Explicit Scharr Gx kernel matches cv::Scharr");
    check(maxAbsDifference(scharr_manual_y, scharr_reference.gy) <= 1e-4,
          "Explicit Scharr Gy kernel matches cv::Scharr");

    cv::Mat signed_pattern = cv::Mat::zeros(11, 17, CV_32F);
    signed_pattern.colRange(4, 8).setTo(200.0f);
    const auto signed_gradient = computeSobel(signed_pattern, border);
    double min_gx = 0.0;
    double max_gx = 0.0;
    cv::minMaxLoc(signed_gradient.gx, &min_gx, &max_gx);
    check(min_gx < 0.0 && max_gx > 0.0,
          "Signed derivative retains opposite edge polarities");

    if (passed) {
        std::cout << "Sobel/Scharr gradient self-test passed.\n";
    }
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parseArguments(argc, argv, options)) {
        printUsage(argv[0]);
        return 2;
    }

    if (options.help) {
        printUsage(argv[0]);
        return 0;
    }

    if (!validateOptions(options)) {
        printUsage(argv[0]);
        return 2;
    }

    if (options.self_test) {
        return runSelfTest() ? 0 : 1;
    }

    const cv::Mat input_u8 = cv::imread(options.input_path, cv::IMREAD_GRAYSCALE);
    if (input_u8.empty()) {
        std::cerr << "Error: could not read input image: " << options.input_path << '\n';
        return 1;
    }
    const cv::Mat input = toFloatGray(input_u8);
    const int border = borderType(options.border);

    const fs::path output_dir(options.output_dir);
    std::error_code error;
    fs::create_directories(output_dir, error);
    if (error) {
        std::cerr << "Error: could not create output directory " << output_dir
                  << ": " << error.message() << '\n';
        return 1;
    }

    bool ok = true;
    if (options.method == "sobel" || options.method == "both") {
        const auto sobel = computeSobel(input, border);
        ok &= writeGradientSet(output_dir, "sobel", sobel);

        double min_x = 0.0;
        double max_x = 0.0;
        cv::minMaxLoc(sobel.gx, &min_x, &max_x);
        std::cout << "Sobel Gx range: [" << min_x << ", " << max_x << "]\n";
    }

    if (options.method == "scharr" || options.method == "both") {
        const auto scharr = computeScharr(input, border);
        ok &= writeGradientSet(output_dir, "scharr", scharr);

        double min_x = 0.0;
        double max_x = 0.0;
        cv::minMaxLoc(scharr.gx, &min_x, &max_x);
        std::cout << "Scharr Gx range: [" << min_x << ", " << max_x << "]\n";
    }

    if (!ok) {
        return 1;
    }

    std::cout << "Input: " << options.input_path << '\n'
              << "Size: " << input.cols << "x" << input.rows << '\n'
              << "Method: " << options.method << '\n'
              << "Border: " << options.border << '\n'
              << "Outputs written to: " << output_dir << '\n';

    return 0;
}
