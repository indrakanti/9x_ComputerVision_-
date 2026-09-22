#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

constexpr unsigned char kWeak = 75;
constexpr unsigned char kStrong = 255;

struct Options {
    std::string input_path;
    std::string output_dir;
    double sigma = 1.0;
    double low_threshold = 50.0;
    double high_threshold = 100.0;
    std::string border = "reflect101";
    bool self_test = false;
    bool help = false;
};

struct CannyStages {
    cv::Mat smoothed;
    cv::Mat gx;
    cv::Mat gy;
    cv::Mat magnitude;
    cv::Mat direction_deg;
    cv::Mat nms;
    cv::Mat threshold_labels;
    cv::Mat edges;
};

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program
        << " --input <image> --output-dir <directory> [options]\n"
        << "  " << program << " --self-test\n\n"
        << "Options:\n"
        << "  --input <path>                Input image\n"
        << "  --output-dir <path>           Output directory\n"
        << "  --sigma <value>               Gaussian sigma (default: 1.0)\n"
        << "  --low-threshold <value>       Canny low threshold (default: 50)\n"
        << "  --high-threshold <value>      Canny high threshold (default: 100)\n"
        << "  --border reflect101|replicate Border policy (default: reflect101)\n"
        << "  --self-test                   Run deterministic tests\n"
        << "  --help                        Show this help\n";
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
            } else if (arg == "--low-threshold") {
                options.low_threshold = std::stod(requireValue(i, arg));
            } else if (arg == "--high-threshold") {
                options.high_threshold = std::stod(requireValue(i, arg));
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
    if (options.low_threshold < 0.0 ||
        options.high_threshold <= 0.0 ||
        options.low_threshold > options.high_threshold) {
        std::cerr << "Error: thresholds must satisfy 0 <= low <= high and high > 0.\n";
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

    cv::Mat output;
    gray.convertTo(output, CV_32F);
    return output;
}

cv::Mat smoothImage(const cv::Mat& image, double sigma, int border_type) {
    if (sigma <= 0.0) {
        return image.clone();
    }

    cv::Mat smoothed;
    cv::GaussianBlur(image, smoothed, cv::Size(0, 0), sigma, sigma, border_type);
    return smoothed;
}

int quantizeDirection(float angle_deg) {
    double angle = std::fmod(static_cast<double>(angle_deg), 180.0);
    if (angle < 0.0) {
        angle += 180.0;
    }

    if (angle < 22.5 || angle >= 157.5) {
        return 0;
    }
    if (angle < 67.5) {
        return 45;
    }
    if (angle < 112.5) {
        return 90;
    }
    return 135;
}

cv::Mat nonMaximumSuppression(const cv::Mat& magnitude,
                              const cv::Mat& direction_deg) {
    CV_Assert(magnitude.type() == CV_32F);
    CV_Assert(direction_deg.type() == CV_32F);
    CV_Assert(magnitude.size() == direction_deg.size());

    cv::Mat output = cv::Mat::zeros(magnitude.size(), CV_32F);

    for (int row = 1; row < magnitude.rows - 1; ++row) {
        for (int col = 1; col < magnitude.cols - 1; ++col) {
            const float center = magnitude.at<float>(row, col);
            const int direction = quantizeDirection(direction_deg.at<float>(row, col));

            float before = 0.0f;
            float after = 0.0f;

            switch (direction) {
                case 0:
                    before = magnitude.at<float>(row, col - 1);
                    after = magnitude.at<float>(row, col + 1);
                    break;
                case 45:
                    before = magnitude.at<float>(row + 1, col - 1);
                    after = magnitude.at<float>(row - 1, col + 1);
                    break;
                case 90:
                    before = magnitude.at<float>(row - 1, col);
                    after = magnitude.at<float>(row + 1, col);
                    break;
                case 135:
                    before = magnitude.at<float>(row - 1, col - 1);
                    after = magnitude.at<float>(row + 1, col + 1);
                    break;
                default:
                    break;
            }

            if (center >= before && center >= after) {
                output.at<float>(row, col) = center;
            }
        }
    }

    return output;
}

cv::Mat doubleThreshold(const cv::Mat& nms,
                        double low_threshold,
                        double high_threshold) {
    CV_Assert(nms.type() == CV_32F);

    cv::Mat labels = cv::Mat::zeros(nms.size(), CV_8U);
    for (int row = 0; row < nms.rows; ++row) {
        for (int col = 0; col < nms.cols; ++col) {
            const float value = nms.at<float>(row, col);
            if (value >= high_threshold) {
                labels.at<unsigned char>(row, col) = kStrong;
            } else if (value >= low_threshold) {
                labels.at<unsigned char>(row, col) = kWeak;
            }
        }
    }
    return labels;
}

cv::Mat hysteresis(const cv::Mat& labels) {
    CV_Assert(labels.type() == CV_8U);

    cv::Mat edges = cv::Mat::zeros(labels.size(), CV_8U);
    std::queue<cv::Point> frontier;

    for (int row = 0; row < labels.rows; ++row) {
        for (int col = 0; col < labels.cols; ++col) {
            if (labels.at<unsigned char>(row, col) == kStrong) {
                edges.at<unsigned char>(row, col) = 255;
                frontier.emplace(col, row);
            }
        }
    }

    while (!frontier.empty()) {
        const cv::Point point = frontier.front();
        frontier.pop();

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) {
                    continue;
                }

                const int x = point.x + dx;
                const int y = point.y + dy;
                if (x < 0 || x >= labels.cols || y < 0 || y >= labels.rows) {
                    continue;
                }

                if (labels.at<unsigned char>(y, x) == kWeak &&
                    edges.at<unsigned char>(y, x) == 0) {
                    edges.at<unsigned char>(y, x) = 255;
                    frontier.emplace(x, y);
                }
            }
        }
    }

    return edges;
}

CannyStages runCannyFromScratch(const cv::Mat& image,
                                double sigma,
                                double low_threshold,
                                double high_threshold,
                                int border_type) {
    CannyStages stages;
    stages.smoothed = smoothImage(image, sigma, border_type);

    cv::Sobel(stages.smoothed, stages.gx, CV_32F, 1, 0, 3, 1.0, 0.0, border_type);
    cv::Sobel(stages.smoothed, stages.gy, CV_32F, 0, 1, 3, 1.0, 0.0, border_type);
    cv::magnitude(stages.gx, stages.gy, stages.magnitude);
    cv::phase(stages.gx, stages.gy, stages.direction_deg, true);

    stages.nms = nonMaximumSuppression(stages.magnitude, stages.direction_deg);
    stages.threshold_labels =
        doubleThreshold(stages.nms, low_threshold, high_threshold);
    stages.edges = hysteresis(stages.threshold_labels);

    return stages;
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

cv::Mat visualizeDirection(const cv::Mat& direction_deg) {
    cv::Mat output;
    direction_deg.convertTo(output, CV_8U, 255.0 / 360.0);
    return output;
}

cv::Mat visualizeThresholdLabels(const cv::Mat& labels) {
    cv::Mat output = cv::Mat::zeros(labels.size(), CV_8U);
    for (int row = 0; row < labels.rows; ++row) {
        for (int col = 0; col < labels.cols; ++col) {
            const unsigned char value = labels.at<unsigned char>(row, col);
            if (value == kStrong) {
                output.at<unsigned char>(row, col) = 255;
            } else if (value == kWeak) {
                output.at<unsigned char>(row, col) = 128;
            }
        }
    }
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

int countValue(const cv::Mat& input, unsigned char value) {
    cv::Mat mask;
    cv::compare(input, value, mask, cv::CMP_EQ);
    return cv::countNonZero(mask);
}

double binaryIoU(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat a_mask;
    cv::Mat b_mask;
    cv::compare(a, 0, a_mask, cv::CMP_GT);
    cv::compare(b, 0, b_mask, cv::CMP_GT);

    cv::Mat intersection;
    cv::Mat union_mask;
    cv::bitwise_and(a_mask, b_mask, intersection);
    cv::bitwise_or(a_mask, b_mask, union_mask);

    const int union_count = cv::countNonZero(union_mask);
    if (union_count == 0) {
        return 1.0;
    }
    return static_cast<double>(cv::countNonZero(intersection)) /
           static_cast<double>(union_count);
}

bool runSelfTest() {
    bool passed = true;
    auto check = [&passed](bool condition, const std::string& name) {
        if (!condition) {
            std::cerr << "Self-test failed: " << name << '\n';
            passed = false;
        }
    };

    check(quantizeDirection(0.0f) == 0, "0 degrees quantizes to 0");
    check(quantizeDirection(20.0f) == 0, "20 degrees quantizes to 0");
    check(quantizeDirection(30.0f) == 45, "30 degrees quantizes to 45");
    check(quantizeDirection(80.0f) == 90, "80 degrees quantizes to 90");
    check(quantizeDirection(140.0f) == 135, "140 degrees quantizes to 135");
    check(quantizeDirection(179.0f) == 0, "179 degrees quantizes to 0");
    check(quantizeDirection(225.0f) == 45, "225 degrees wraps to 45");

    cv::Mat nms_magnitude = cv::Mat::zeros(5, 5, CV_32F);
    nms_magnitude.at<float>(2, 1) = 2.0f;
    nms_magnitude.at<float>(2, 2) = 5.0f;
    nms_magnitude.at<float>(2, 3) = 2.0f;
    cv::Mat nms_direction = cv::Mat::zeros(5, 5, CV_32F);
    const cv::Mat nms_result =
        nonMaximumSuppression(nms_magnitude, nms_direction);
    check(std::abs(nms_result.at<float>(2, 2) - 5.0f) <= 1e-6f,
          "NMS keeps local maximum");
    check(cv::countNonZero(nms_result) == 1,
          "NMS suppresses non-maxima along gradient direction");

    cv::Mat threshold_input = (cv::Mat_<float>(1, 4) <<
        0.0f, 30.0f, 60.0f, 120.0f);
    const cv::Mat labels = doubleThreshold(threshold_input, 50.0, 100.0);
    check(labels.at<unsigned char>(0, 0) == 0, "below-low threshold rejected");
    check(labels.at<unsigned char>(0, 1) == 0, "second below-low threshold rejected");
    check(labels.at<unsigned char>(0, 2) == kWeak, "middle response marked weak");
    check(labels.at<unsigned char>(0, 3) == kStrong, "high response marked strong");

    cv::Mat hysteresis_labels = cv::Mat::zeros(6, 8, CV_8U);
    hysteresis_labels.at<unsigned char>(2, 1) = kStrong;
    hysteresis_labels.at<unsigned char>(2, 2) = kWeak;
    hysteresis_labels.at<unsigned char>(2, 3) = kWeak;
    hysteresis_labels.at<unsigned char>(1, 4) = kWeak;
    hysteresis_labels.at<unsigned char>(4, 6) = kWeak;
    const cv::Mat hysteresis_result = hysteresis(hysteresis_labels);
    check(cv::countNonZero(hysteresis_result) == 4,
          "hysteresis keeps strong edge and connected weak chain");
    check(hysteresis_result.at<unsigned char>(4, 6) == 0,
          "hysteresis rejects isolated weak edge");

    cv::Mat constant(40, 48, CV_32F, cv::Scalar(80.0f));
    const auto constant_result =
        runCannyFromScratch(constant, 1.0, 20.0, 40.0, cv::BORDER_REFLECT_101);
    check(cv::countNonZero(constant_result.edges) == 0,
          "constant image produces no Canny edges");

    cv::Mat step(64, 80, CV_32F, cv::Scalar(0.0f));
    step.colRange(40, step.cols).setTo(200.0f);
    const auto step_result =
        runCannyFromScratch(step, 1.0, 40.0, 80.0, cv::BORDER_REFLECT_101);
    check(cv::countNonZero(step_result.edges) > 0,
          "vertical step produces final Canny edges");
    check(countValue(step_result.threshold_labels, kStrong) > 0,
          "vertical step produces strong threshold labels");

    cv::Mat zero_direction = cv::Mat::zeros(step_result.magnitude.size(), CV_32F);
    cv::phase(step_result.gx, step_result.gy, zero_direction, true);
    check(maxAbsDifference(zero_direction, step_result.direction_deg) <= 1e-5,
          "stored direction matches OpenCV phase calculation");

    if (passed) {
        std::cout << "Canny-from-scratch self-test passed.\n";
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
    const int border_type = borderType(options.border);
    const auto stages = runCannyFromScratch(
        input,
        options.sigma,
        options.low_threshold,
        options.high_threshold,
        border_type);

    cv::Mat opencv_smoothed_u8;
    if (options.sigma <= 0.0) {
        opencv_smoothed_u8 = input_u8.clone();
    } else {
        cv::GaussianBlur(input_u8,
                         opencv_smoothed_u8,
                         cv::Size(0, 0),
                         options.sigma,
                         options.sigma,
                         border_type);
    }

    cv::Mat opencv_edges;
    cv::Canny(opencv_smoothed_u8,
              opencv_edges,
              options.low_threshold,
              options.high_threshold,
              3,
              true);

    cv::Mat difference;
    cv::bitwise_xor(stages.edges, opencv_edges, difference);

    const fs::path output_dir(options.output_dir);
    std::error_code error;
    fs::create_directories(output_dir, error);
    if (error) {
        std::cerr << "Error: could not create output directory " << output_dir
                  << ": " << error.message() << '\n';
        return 1;
    }

    bool ok = true;
    ok &= writeImage(output_dir / "01_smoothed.png", toByteImage(stages.smoothed));
    ok &= writeImage(output_dir / "02_gradient_magnitude.png",
                     visualizeNonNegative(stages.magnitude));
    ok &= writeImage(output_dir / "03_gradient_direction.png",
                     visualizeDirection(stages.direction_deg));
    ok &= writeImage(output_dir / "04_non_maximum_suppression.png",
                     visualizeNonNegative(stages.nms));
    ok &= writeImage(output_dir / "05_double_threshold.png",
                     visualizeThresholdLabels(stages.threshold_labels));
    ok &= writeImage(output_dir / "06_hysteresis_edges.png", stages.edges);
    ok &= writeImage(output_dir / "07_opencv_canny_reference.png", opencv_edges);
    ok &= writeImage(output_dir / "08_reference_difference.png", difference);

    if (!ok) {
        return 1;
    }

    std::cout << "Input: " << options.input_path << '\n'
              << "Size: " << input.cols << "x" << input.rows << '\n'
              << "Gaussian sigma: " << options.sigma << '\n'
              << "Low threshold: " << options.low_threshold << '\n'
              << "High threshold: " << options.high_threshold << '\n'
              << "Strong labels: "
              << countValue(stages.threshold_labels, kStrong) << '\n'
              << "Weak labels: "
              << countValue(stages.threshold_labels, kWeak) << '\n'
              << "Final edge pixels: " << cv::countNonZero(stages.edges) << '\n'
              << "OpenCV reference edge pixels: "
              << cv::countNonZero(opencv_edges) << '\n'
              << "Binary IoU with OpenCV reference: "
              << binaryIoU(stages.edges, opencv_edges) << '\n'
              << "Outputs written to: " << output_dir << '\n';

    return 0;
}
