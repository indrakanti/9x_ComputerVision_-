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
    int levels = 4;
    bool self_test = false;
    bool help = false;
};

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " --input <image> --output-dir <directory> [--levels N]\n"
        << "  " << program << " --self-test\n\n"
        << "Options:\n"
        << "  --input <path>        Input image path\n"
        << "  --output-dir <path>   Directory for pyramid outputs\n"
        << "  --levels <2-8>        Number of Gaussian/Laplacian levels (default: 4)\n"
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
            } else if (arg == "--levels") {
                options.levels = std::stoi(requireValue(i, arg));
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
    if (options.input_path.empty()) {
        std::cerr << "Error: --input is required.\n";
        return false;
    }
    if (options.output_dir.empty()) {
        std::cerr << "Error: --output-dir is required.\n";
        return false;
    }
    if (options.levels < 2 || options.levels > 8) {
        std::cerr << "Error: --levels must be in the range 2..8.\n";
        return false;
    }
    return true;
}

cv::Mat toFloatGray(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image;
    } else {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat result;
    gray.convertTo(result, CV_32F);
    return result;
}

std::vector<cv::Mat> buildGaussianPyramid(const cv::Mat& input, int levels) {
    std::vector<cv::Mat> gaussian;
    gaussian.reserve(static_cast<std::size_t>(levels));
    gaussian.push_back(input.clone());

    for (int level = 1; level < levels; ++level) {
        if (gaussian.back().cols < 2 || gaussian.back().rows < 2) {
            throw std::runtime_error("Image became too small for the requested pyramid depth");
        }

        cv::Mat reduced;
        cv::pyrDown(gaussian.back(), reduced);
        gaussian.push_back(reduced);
    }

    return gaussian;
}

std::vector<cv::Mat> buildLaplacianPyramid(const std::vector<cv::Mat>& gaussian) {
    if (gaussian.size() < 2) {
        throw std::runtime_error("Gaussian pyramid must contain at least two levels");
    }

    std::vector<cv::Mat> laplacian;
    laplacian.reserve(gaussian.size());

    for (std::size_t level = 0; level + 1 < gaussian.size(); ++level) {
        cv::Mat expanded;
        cv::pyrUp(gaussian[level + 1], expanded, gaussian[level].size());
        laplacian.push_back(gaussian[level] - expanded);
    }

    laplacian.push_back(gaussian.back().clone());
    return laplacian;
}

cv::Mat reconstructFromLaplacian(const std::vector<cv::Mat>& laplacian) {
    if (laplacian.empty()) {
        throw std::runtime_error("Laplacian pyramid must not be empty");
    }

    cv::Mat reconstructed = laplacian.back().clone();
    for (int level = static_cast<int>(laplacian.size()) - 2; level >= 0; --level) {
        cv::Mat expanded;
        cv::pyrUp(reconstructed, expanded, laplacian[static_cast<std::size_t>(level)].size());
        reconstructed = expanded + laplacian[static_cast<std::size_t>(level)];
    }

    return reconstructed;
}

cv::Mat toByteImage(const cv::Mat& image) {
    cv::Mat result;
    image.convertTo(result, CV_8U);
    return result;
}

cv::Mat visualizeSigned(const cv::Mat& image) {
    double min_value = 0.0;
    double max_value = 0.0;
    cv::minMaxLoc(image, &min_value, &max_value);
    const double max_abs = std::max(std::abs(min_value), std::abs(max_value));

    if (max_abs < 1e-12) {
        return cv::Mat(image.size(), CV_8U, cv::Scalar(128));
    }

    cv::Mat normalized = image * static_cast<float>(127.0 / max_abs) + 128.0f;
    cv::Mat result;
    normalized.convertTo(result, CV_8U);
    return result;
}

cv::Mat visualizeDifference(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat difference;
    cv::absdiff(a, b, difference);

    double max_value = 0.0;
    cv::minMaxLoc(difference, nullptr, &max_value);
    if (max_value < 1e-12) {
        return cv::Mat(difference.size(), CV_8U, cv::Scalar(0));
    }

    cv::Mat normalized = difference * static_cast<float>(255.0 / max_value);
    cv::Mat result;
    normalized.convertTo(result, CV_8U);
    return result;
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

double rootMeanSquareError(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat difference = a - b;
    cv::Mat squared = difference.mul(difference);
    return std::sqrt(cv::mean(squared)[0]);
}

std::size_t totalPixels(const std::vector<cv::Mat>& pyramid) {
    std::size_t total = 0;
    for (const auto& level : pyramid) {
        total += static_cast<std::size_t>(level.total());
    }
    return total;
}

bool runSelfTest() {
    bool passed = true;
    auto check = [&passed](bool condition, const std::string& name) {
        if (!condition) {
            std::cerr << "Self-test failed: " << name << '\n';
            passed = false;
        }
    };

    cv::Mat input(48, 64, CV_32F);
    for (int row = 0; row < input.rows; ++row) {
        for (int col = 0; col < input.cols; ++col) {
            input.at<float>(row, col) =
                static_cast<float>((row * 7 + col * 13 + (row * col) % 31) % 256);
        }
    }

    const auto gaussian = buildGaussianPyramid(input, 4);
    check(gaussian.size() == 4, "Gaussian level count");
    check(gaussian[0].size() == cv::Size(64, 48), "Gaussian level 0 size");
    check(gaussian[1].size() == cv::Size(32, 24), "Gaussian level 1 size");
    check(gaussian[2].size() == cv::Size(16, 12), "Gaussian level 2 size");
    check(gaussian[3].size() == cv::Size(8, 6), "Gaussian level 3 size");

    const auto laplacian = buildLaplacianPyramid(gaussian);
    check(laplacian.size() == gaussian.size(), "Laplacian level count");

    double residual_min = 0.0;
    double residual_max = 0.0;
    cv::minMaxLoc(laplacian.front(), &residual_min, &residual_max);
    check(residual_min < 0.0 && residual_max > 0.0,
          "Laplacian residual retains signed detail");

    const cv::Mat reconstructed = reconstructFromLaplacian(laplacian);
    const double max_error = maxAbsDifference(input, reconstructed);
    check(max_error <= 1e-4, "Laplacian reconstruction max error");

    cv::Mat constant(48, 64, CV_32F, cv::Scalar(42.0f));
    const auto constant_gaussian = buildGaussianPyramid(constant, 4);
    for (std::size_t level = 0; level < constant_gaussian.size(); ++level) {
        cv::Mat expected(constant_gaussian[level].size(), CV_32F, cv::Scalar(42.0f));
        check(maxAbsDifference(constant_gaussian[level], expected) <= 1e-4,
              "Constant image preserved at Gaussian level " + std::to_string(level));
    }

    if (passed) {
        std::cout << "Image-pyramid self-test passed. Reconstruction max error: "
                  << max_error << '\n';
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

    std::vector<cv::Mat> gaussian;
    std::vector<cv::Mat> laplacian;
    cv::Mat reconstructed;
    try {
        gaussian = buildGaussianPyramid(input, options.levels);
        laplacian = buildLaplacianPyramid(gaussian);
        reconstructed = reconstructFromLaplacian(laplacian);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    const fs::path output_dir(options.output_dir);
    std::error_code error;
    fs::create_directories(output_dir, error);
    if (error) {
        std::cerr << "Error: could not create output directory " << output_dir
                  << ": " << error.message() << '\n';
        return 1;
    }

    bool ok = true;
    for (std::size_t level = 0; level < gaussian.size(); ++level) {
        ok &= writeImage(output_dir / ("gaussian_level_" + std::to_string(level) + ".png"),
                         toByteImage(gaussian[level]));
    }

    for (std::size_t level = 0; level + 1 < laplacian.size(); ++level) {
        ok &= writeImage(output_dir / ("laplacian_level_" + std::to_string(level) + ".png"),
                         visualizeSigned(laplacian[level]));
    }
    ok &= writeImage(output_dir / "laplacian_base.png", toByteImage(laplacian.back()));

    if (gaussian.size() >= 2) {
        cv::Mat naive_downsample;
        cv::resize(input, naive_downsample, gaussian[1].size(), 0.0, 0.0, cv::INTER_NEAREST);
        ok &= writeImage(output_dir / "naive_downsample_level_1.png",
                         toByteImage(naive_downsample));
        ok &= writeImage(output_dir / "gaussian_downsample_level_1.png",
                         toByteImage(gaussian[1]));
    }

    ok &= writeImage(output_dir / "reconstructed.png", toByteImage(reconstructed));
    ok &= writeImage(output_dir / "reconstruction_difference.png",
                     visualizeDifference(input, reconstructed));

    if (!ok) {
        return 1;
    }

    const double max_error = maxAbsDifference(input, reconstructed);
    const double rmse = rootMeanSquareError(input, reconstructed);
    const double storage_ratio = static_cast<double>(totalPixels(gaussian)) /
                                 static_cast<double>(input.total());

    std::cout << "Input: " << options.input_path << '\n'
              << "Base size: " << input.cols << "x" << input.rows << '\n'
              << "Levels: " << gaussian.size() << '\n';

    for (std::size_t level = 0; level < gaussian.size(); ++level) {
        std::cout << "  G" << level << ": " << gaussian[level].cols << "x"
                  << gaussian[level].rows << '\n';
    }

    std::cout << "Gaussian pyramid pixel ratio vs base: " << storage_ratio << '\n'
              << "Reconstruction max abs error: " << max_error << '\n'
              << "Reconstruction RMSE: " << rmse << '\n'
              << "Outputs written to: " << output_dir << '\n';

    return 0;
}
