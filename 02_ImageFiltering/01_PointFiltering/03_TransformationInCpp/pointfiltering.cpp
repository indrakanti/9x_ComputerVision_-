#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string input_path;
    std::string output_dir;
    int brightness = 50;
    int darken = 50;
    double contrast = 1.5;
    double gamma = 0.5;
    int threshold = 128;
    bool self_test = false;
    bool help = false;
};

cv::Mat adjustBrightness(const cv::Mat& image, int value) {
    cv::Mat result;
    image.convertTo(result, -1, 1.0, value);
    return result;
}

cv::Mat adjustContrast(const cv::Mat& image, double alpha) {
    cv::Mat result;
    image.convertTo(result, -1, alpha, 0.0);
    return result;
}

cv::Mat gammaCorrection(const cv::Mat& image, double gamma) {
    cv::Mat result;
    cv::Mat lookup(1, 256, CV_8U);
    auto* values = lookup.ptr<unsigned char>();

    for (int i = 0; i < 256; ++i) {
        const double normalized = static_cast<double>(i) / 255.0;
        const double corrected = std::pow(normalized, gamma) * 255.0;
        values[i] = cv::saturate_cast<unsigned char>(std::round(corrected));
    }

    cv::LUT(image, lookup, result);
    return result;
}

cv::Mat invertImage(const cv::Mat& image) {
    return cv::Scalar::all(255) - image;
}

cv::Mat thresholdImage(const cv::Mat& image, int threshold_value) {
    cv::Mat binary;
    cv::threshold(image, binary, threshold_value, 255, cv::THRESH_BINARY);
    return binary;
}

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program
        << " --input <image> --output-dir <directory> [options]\n"
        << "  " << program << " --self-test\n\n"
        << "Options:\n"
        << "  --brightness <int>   Brightness offset (default: 50)\n"
        << "  --darken <int>       Darkening magnitude (default: 50)\n"
        << "  --contrast <double>  Contrast multiplier (default: 1.5)\n"
        << "  --gamma <double>     Gamma exponent, must be > 0 (default: 0.5)\n"
        << "  --threshold <0-255>  Binary threshold (default: 128)\n"
        << "  --self-test          Run deterministic in-memory checks\n"
        << "  --help               Show this help text\n";
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
            } else if (arg == "--brightness") {
                options.brightness = std::stoi(requireValue(i, arg));
            } else if (arg == "--darken") {
                options.darken = std::stoi(requireValue(i, arg));
            } else if (arg == "--contrast") {
                options.contrast = std::stod(requireValue(i, arg));
            } else if (arg == "--gamma") {
                options.gamma = std::stod(requireValue(i, arg));
            } else if (arg == "--threshold") {
                options.threshold = std::stoi(requireValue(i, arg));
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
    if (options.contrast < 0.0) {
        std::cerr << "Error: --contrast must be >= 0.\n";
        return false;
    }
    if (options.gamma <= 0.0) {
        std::cerr << "Error: --gamma must be > 0.\n";
        return false;
    }
    if (options.threshold < 0 || options.threshold > 255) {
        std::cerr << "Error: --threshold must be in the range 0..255.\n";
        return false;
    }
    if (options.darken < 0) {
        std::cerr << "Error: --darken must be >= 0.\n";
        return false;
    }

    return true;
}

bool equalMat(const cv::Mat& actual, const cv::Mat& expected) {
    return actual.size() == expected.size() && actual.type() == expected.type() &&
           cv::norm(actual, expected, cv::NORM_INF) == 0.0;
}

bool runSelfTest() {
    const cv::Mat input = (cv::Mat_<unsigned char>(1, 5) << 0, 64, 128, 192, 255);

    const cv::Mat expected_brightness =
        (cv::Mat_<unsigned char>(1, 5) << 10, 74, 138, 202, 255);
    const cv::Mat expected_contrast =
        (cv::Mat_<unsigned char>(1, 5) << 0, 96, 192, 255, 255);
    const cv::Mat expected_gamma =
        (cv::Mat_<unsigned char>(1, 5) << 0, 16, 64, 145, 255);
    const cv::Mat expected_inversion =
        (cv::Mat_<unsigned char>(1, 5) << 255, 191, 127, 63, 0);
    const cv::Mat expected_threshold =
        (cv::Mat_<unsigned char>(1, 5) << 0, 0, 0, 255, 255);

    struct Check {
        const char* name;
        cv::Mat actual;
        cv::Mat expected;
    };

    const Check checks[] = {
        {"brightness", adjustBrightness(input, 10), expected_brightness},
        {"contrast", adjustContrast(input, 1.5), expected_contrast},
        {"gamma", gammaCorrection(input, 2.0), expected_gamma},
        {"inversion", invertImage(input), expected_inversion},
        {"threshold", thresholdImage(input, 128), expected_threshold},
    };

    bool passed = true;
    for (const auto& check : checks) {
        if (!equalMat(check.actual, check.expected)) {
            std::cerr << "Self-test failed: " << check.name << '\n';
            passed = false;
        }
    }

    if (passed) {
        std::cout << "Point-filter self-test passed.\n";
    }
    return passed;
}

bool writeImage(const fs::path& path, const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image)) {
        std::cerr << "Error: failed to write " << path << '\n';
        return false;
    }
    return true;
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

    const cv::Mat image = cv::imread(options.input_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: could not read input image: " << options.input_path << '\n';
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
    ok &= writeImage(output_dir / "brightness_adjustment.png",
                     adjustBrightness(image, options.brightness));
    ok &= writeImage(output_dir / "darkening.png",
                     adjustBrightness(image, -options.darken));
    ok &= writeImage(output_dir / "contrast_adjustment.png",
                     adjustContrast(image, options.contrast));
    ok &= writeImage(output_dir / "gamma_correction.png",
                     gammaCorrection(image, options.gamma));
    ok &= writeImage(output_dir / "image_inversion.png", invertImage(image));
    ok &= writeImage(output_dir / "thresholding.png",
                     thresholdImage(image, options.threshold));

    if (!ok) {
        return 1;
    }

    std::cout << "Input: " << options.input_path << '\n'
              << "Image: " << image.cols << "x" << image.rows << " pixels, "
              << image.channels() << " channel(s)\n"
              << "Outputs written to: " << output_dir << '\n';

    return 0;
}
