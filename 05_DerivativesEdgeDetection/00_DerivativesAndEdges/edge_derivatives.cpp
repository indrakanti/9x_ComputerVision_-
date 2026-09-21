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
    double sigma = 1.0;
    double gradient_threshold = 100.0;
    double zero_crossing_threshold = 15.0;
    std::string border = "reflect101";
    bool self_test = false;
    bool help = false;
};

struct DerivativeResult {
    cv::Mat smoothed;
    cv::Mat gx;
    cv::Mat gy;
    cv::Mat magnitude;
    cv::Mat gradient_edges;
    cv::Mat laplacian;
    cv::Mat zero_crossings;
};

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program
        << " --input <image> --output-dir <directory> [options]\n"
        << "  " << program << " --self-test\n\n"
        << "Options:\n"
        << "  --input <path>                  Input image\n"
        << "  --output-dir <path>             Output directory\n"
        << "  --sigma <value>                 Gaussian sigma (default: 1.0)\n"
        << "  --gradient-threshold <value>    Sobel magnitude threshold (default: 100)\n"
        << "  --zero-crossing-threshold <v>   Minimum Laplacian sign-change contrast (default: 15)\n"
        << "  --border reflect101|replicate   Border policy (default: reflect101)\n"
        << "  --self-test                     Run deterministic tests\n"
        << "  --help                          Show this help\n";
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
            } else if (arg == "--sigma") {
                options.sigma = std::stod(requireValue(i, arg));
            } else if (arg == "--gradient-threshold") {
                options.gradient_threshold = std::stod(requireValue(i, arg));
            } else if (arg == "--zero-crossing-threshold") {
                options.zero_crossing_threshold = std::stod(requireValue(i, arg));
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
    if (options.sigma < 0.0) {
        std::cerr << "Error: --sigma must be >= 0.\n";
        return false;
    }
    if (options.gradient_threshold < 0.0 || options.zero_crossing_threshold < 0.0) {
        std::cerr << "Error: thresholds must be >= 0.\n";
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

cv::Mat smoothImage(const cv::Mat& image, double sigma, int border_type) {
    if (sigma <= 0.0) {
        return image.clone();
    }
    cv::Mat smoothed;
    cv::GaussianBlur(image, smoothed, cv::Size(0, 0), sigma, sigma, border_type);
    return smoothed;
}

cv::Mat thresholdMagnitude(const cv::Mat& magnitude, double threshold) {
    cv::Mat edges;
    cv::compare(magnitude, threshold, edges, cv::CMP_GE);
    return edges;
}

cv::Mat detectZeroCrossings(const cv::Mat& laplacian, double min_contrast) {
    CV_Assert(laplacian.type() == CV_32F);

    cv::Mat edges = cv::Mat::zeros(laplacian.size(), CV_8U);
    for (int row = 1; row < laplacian.rows - 1; ++row) {
        for (int col = 1; col < laplacian.cols - 1; ++col) {
            const float center = laplacian.at<float>(row, col);
            bool crossing = false;

            for (int dy = -1; dy <= 1 && !crossing; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }
                    const float neighbor = laplacian.at<float>(row + dy, col + dx);
                    const bool opposite_sign =
                        (center > 0.0f && neighbor < 0.0f) ||
                        (center < 0.0f && neighbor > 0.0f);
                    const double contrast = std::abs(
                        static_cast<double>(center) - static_cast<double>(neighbor));
                    if (opposite_sign && contrast >= min_contrast) {
                        crossing = true;
                        break;
                    }
                }
            }

            if (crossing) {
                edges.at<unsigned char>(row, col) = 255;
            }
        }
    }
    return edges;
}

DerivativeResult computeDerivatives(const cv::Mat& image,
                                    double sigma,
                                    double gradient_threshold,
                                    double zero_crossing_threshold,
                                    int border_type) {
    DerivativeResult result;
    result.smoothed = smoothImage(image, sigma, border_type);

    cv::Sobel(result.smoothed, result.gx, CV_32F, 1, 0, 3, 1.0, 0.0, border_type);
    cv::Sobel(result.smoothed, result.gy, CV_32F, 0, 1, 3, 1.0, 0.0, border_type);
    cv::magnitude(result.gx, result.gy, result.magnitude);
    result.gradient_edges = thresholdMagnitude(result.magnitude, gradient_threshold);

    cv::Laplacian(result.smoothed, result.laplacian, CV_32F, 1, 1.0, 0.0, border_type);
    result.zero_crossings = detectZeroCrossings(result.laplacian, zero_crossing_threshold);
    return result;
}

cv::Mat visualizeSigned(const cv::Mat& input) {
    double min_value = 0.0;
    double max_value = 0.0;
    cv::minMaxLoc(input, &min_value, &max_value);
    const double max_abs = std::max(std::abs(min_value), std::abs(max_value));
    if (max_abs < 1e-12) {
        return cv::Mat(input.size(), CV_8U, cv::Scalar(128));
    }

    cv::Mat normalized = input * static_cast<float>(127.0 / max_abs) + 128.0f;
    cv::Mat output;
    normalized.convertTo(output, CV_8U);
    return output;
}

cv::Mat visualizeNonNegative(const cv::Mat& input) {
    double max_value = 0.0;
    cv::minMaxLoc(input, nullptr, &max_value);
    if (max_value < 1e-12) {
        return cv::Mat(input.size(), CV_8U, cv::Scalar(0));
    }

    cv::Mat output;
    input.convertTo(output, CV_8U, 255.0 / max_value);
    return output;
}

cv::Mat toByteImage(const cv::Mat& input) {
    cv::Mat output;
    input.convertTo(output, CV_8U);
    return output;
}

bool writeImage(const fs::path& path, const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image)) {
        std::cerr << "Error: failed to write " << path << '\n';
        return false;
    }
    return true;
}

double maxAbsDifference(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat difference;
    cv::absdiff(a, b, difference);
    double max_value = 0.0;
    cv::minMaxLoc(difference, nullptr, &max_value);
    return max_value;
}

double meanAbsoluteValue(const cv::Mat& input) {
    cv::Mat absolute;
    cv::absdiff(input, cv::Scalar(0), absolute);
    return cv::mean(absolute)[0];
}

int countEdges(const cv::Mat& edge_map) {
    return cv::countNonZero(edge_map);
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

    cv::Mat constant(32, 40, CV_32F, cv::Scalar(50.0f));
    const auto constant_result = computeDerivatives(constant, 0.0, 1.0, 1.0, border);
    check(maxAbsDifference(constant_result.magnitude,
                           cv::Mat::zeros(constant.size(), CV_32F)) <= 1e-5,
          "constant image has zero first derivative");
    check(maxAbsDifference(constant_result.laplacian,
                           cv::Mat::zeros(constant.size(), CV_32F)) <= 1e-5,
          "constant image has zero second derivative");
    check(countEdges(constant_result.zero_crossings) == 0,
          "constant image has no zero crossings");

    cv::Mat ramp(32, 40, CV_32F);
    for (int row = 0; row < ramp.rows; ++row) {
        for (int col = 0; col < ramp.cols; ++col) {
            ramp.at<float>(row, col) = static_cast<float>(3 * col + 2 * row);
        }
    }
    const auto ramp_result = computeDerivatives(ramp, 0.0, 1.0, 1.0, border);
    const cv::Rect interior(2, 2, ramp.cols - 4, ramp.rows - 4);
    check(cv::mean(ramp_result.magnitude(interior))[0] > 0.0,
          "linear ramp has non-zero first derivative");
    check(meanAbsoluteValue(ramp_result.laplacian(interior)) <= 1e-5,
          "linear ramp has zero Laplacian in the interior");

    cv::Mat step(48, 64, CV_32F, cv::Scalar(0.0f));
    step.colRange(32, step.cols).setTo(200.0f);
    const auto step_result = computeDerivatives(step, 1.0, 100.0, 5.0, border);
    check(countEdges(step_result.gradient_edges) > 0,
          "step edge produces first-derivative edge pixels");
    check(countEdges(step_result.zero_crossings) > 0,
          "step edge produces Laplacian zero crossings");

    double lap_min = 0.0;
    double lap_max = 0.0;
    cv::minMaxLoc(step_result.laplacian, &lap_min, &lap_max);
    check(lap_min < 0.0 && lap_max > 0.0,
          "step edge produces signed Laplacian lobes");

    const int low_count = countEdges(thresholdMagnitude(step_result.magnitude, 50.0));
    const int high_count = countEdges(thresholdMagnitude(step_result.magnitude, 400.0));
    check(high_count <= low_count,
          "higher gradient threshold cannot create more edge pixels");

    cv::Mat explicit_laplacian;
    const cv::Mat laplacian_kernel = (cv::Mat_<float>(3, 3) <<
         0.0f, 1.0f, 0.0f,
         1.0f,-4.0f, 1.0f,
         0.0f, 1.0f, 0.0f);
    cv::filter2D(ramp, explicit_laplacian, CV_32F, laplacian_kernel,
                 cv::Point(-1, -1), 0.0, border);
    cv::Mat opencv_laplacian;
    cv::Laplacian(ramp, opencv_laplacian, CV_32F, 1, 1.0, 0.0, border);
    check(maxAbsDifference(explicit_laplacian, opencv_laplacian) <= 1e-5,
          "explicit 4-neighbor Laplacian matches cv::Laplacian ksize=1");

    cv::Mat noisy(64, 64, CV_32F);
    for (int row = 0; row < noisy.rows; ++row) {
        for (int col = 0; col < noisy.cols; ++col) {
            const int deterministic_noise =
                ((row * 37 + col * 53 + row * col * 7) % 61) - 30;
            noisy.at<float>(row, col) =
                120.0f + static_cast<float>(deterministic_noise);
        }
    }

    cv::Mat raw_laplacian;
    cv::Laplacian(noisy, raw_laplacian, CV_32F, 1, 1.0, 0.0, border);
    const cv::Mat smoothed_noise = smoothImage(noisy, 1.2, border);
    cv::Mat smooth_laplacian;
    cv::Laplacian(smoothed_noise, smooth_laplacian, CV_32F, 1, 1.0, 0.0, border);
    check(meanAbsoluteValue(smooth_laplacian) < meanAbsoluteValue(raw_laplacian),
          "Gaussian smoothing reduces Laplacian response to deterministic noise");

    if (passed) {
        std::cout << "Derivative/edge-detection self-test passed.\n";
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
    const auto result = computeDerivatives(
        input,
        options.sigma,
        options.gradient_threshold,
        options.zero_crossing_threshold,
        borderType(options.border));

    const fs::path output_dir(options.output_dir);
    std::error_code error;
    fs::create_directories(output_dir, error);
    if (error) {
        std::cerr << "Error: could not create output directory " << output_dir
                  << ": " << error.message() << '\n';
        return 1;
    }

    bool ok = true;
    ok &= writeImage(output_dir / "smoothed.png", toByteImage(result.smoothed));
    ok &= writeImage(output_dir / "gradient_magnitude.png",
                     visualizeNonNegative(result.magnitude));
    ok &= writeImage(output_dir / "gradient_edges.png", result.gradient_edges);
    ok &= writeImage(output_dir / "laplacian_signed.png",
                     visualizeSigned(result.laplacian));
    ok &= writeImage(output_dir / "laplacian_absolute.png",
                     visualizeNonNegative(cv::abs(result.laplacian)));
    ok &= writeImage(output_dir / "laplacian_zero_crossings.png",
                     result.zero_crossings);

    if (!ok) {
        return 1;
    }

    double lap_min = 0.0;
    double lap_max = 0.0;
    cv::minMaxLoc(result.laplacian, &lap_min, &lap_max);

    std::cout << "Input: " << options.input_path << '\n'
              << "Size: " << input.cols << "x" << input.rows << '\n'
              << "Gaussian sigma: " << options.sigma << '\n'
              << "Gradient threshold: " << options.gradient_threshold << '\n'
              << "Zero-crossing threshold: " << options.zero_crossing_threshold << '\n'
              << "Gradient edge pixels: " << countEdges(result.gradient_edges) << '\n'
              << "Zero-crossing pixels: " << countEdges(result.zero_crossings) << '\n'
              << "Laplacian range: [" << lap_min << ", " << lap_max << "]\n"
              << "Outputs written to: " << output_dir << '\n';

    return 0;
}
