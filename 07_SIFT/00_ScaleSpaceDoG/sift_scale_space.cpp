#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string input_path;
    std::string output_dir;
    int octaves = 4;
    int intervals = 3;
    double sigma0 = 1.6;
    double contrast_threshold = 5.0;
    bool self_test = false;
    bool help = false;
};

struct ScaleExtremum {
    int octave = 0;
    int dog_layer = 0;
    cv::Point point;
    float response = 0.0f;
};

using Octave = std::vector<cv::Mat>;
using Pyramid = std::vector<Octave>;

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program
        << " --input <image> --output-dir <directory> [options]\n"
        << "  " << program << " --self-test\n\n"
        << "Options:\n"
        << "  --input <path>                 Input image\n"
        << "  --output-dir <path>            Output directory\n"
        << "  --octaves <N>                  Number of octaves (default: 4)\n"
        << "  --intervals <N>                Scale intervals per octave (default: 3)\n"
        << "  --sigma <value>                Base sigma for each octave (default: 1.6)\n"
        << "  --contrast-threshold <value>   Raw DoG extrema threshold (default: 5.0)\n"
        << "  --self-test                    Run deterministic tests\n"
        << "  --help                         Show this help\n";
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
            } else if (arg == "--octaves") {
                options.octaves = std::stoi(requireValue(i, arg));
            } else if (arg == "--intervals") {
                options.intervals = std::stoi(requireValue(i, arg));
            } else if (arg == "--sigma") {
                options.sigma0 = std::stod(requireValue(i, arg));
            } else if (arg == "--contrast-threshold") {
                options.contrast_threshold = std::stod(requireValue(i, arg));
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
    if (options.octaves <= 0 || options.intervals <= 0) {
        std::cerr << "Error: octaves and intervals must be > 0.\n";
        return false;
    }
    if (options.sigma0 <= 0.0) {
        std::cerr << "Error: --sigma must be > 0.\n";
        return false;
    }
    if (options.contrast_threshold < 0.0) {
        std::cerr << "Error: --contrast-threshold must be >= 0.\n";
        return false;
    }
    return true;
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

double scaleStep(int intervals) {
    return std::pow(2.0, 1.0 / static_cast<double>(intervals));
}

std::vector<double> absoluteSigmas(int intervals, double sigma0) {
    const int gaussian_layers = intervals + 3;
    const double k = scaleStep(intervals);

    std::vector<double> sigmas;
    sigmas.reserve(static_cast<std::size_t>(gaussian_layers));
    for (int layer = 0; layer < gaussian_layers; ++layer) {
        sigmas.push_back(sigma0 * std::pow(k, layer));
    }
    return sigmas;
}

std::vector<double> incrementalSigmas(int intervals, double sigma0) {
    const std::vector<double> absolute = absoluteSigmas(intervals, sigma0);
    std::vector<double> incremental(absolute.size(), 0.0);
    incremental[0] = sigma0;

    for (std::size_t i = 1; i < absolute.size(); ++i) {
        const double previous_sq = absolute[i - 1] * absolute[i - 1];
        const double target_sq = absolute[i] * absolute[i];
        incremental[i] = std::sqrt(std::max(0.0, target_sq - previous_sq));
    }
    return incremental;
}

cv::Mat downsampleBy2(const cv::Mat& input) {
    const int rows = std::max(1, input.rows / 2);
    const int cols = std::max(1, input.cols / 2);
    cv::Mat output(rows, cols, input.type());

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            output.at<float>(row, col) =
                input.at<float>(std::min(2 * row, input.rows - 1),
                                std::min(2 * col, input.cols - 1));
        }
    }
    return output;
}

Pyramid buildGaussianPyramid(const cv::Mat& input,
                             int octaves,
                             int intervals,
                             double sigma0) {
    const std::vector<double> incremental =
        incrementalSigmas(intervals, sigma0);

    Pyramid pyramid;
    pyramid.reserve(static_cast<std::size_t>(octaves));

    cv::Mat octave_base = input.clone();

    for (int octave_index = 0; octave_index < octaves; ++octave_index) {
        if (octave_base.rows < 8 || octave_base.cols < 8) {
            break;
        }

        Octave octave;
        octave.reserve(incremental.size());

        cv::Mat first;
        cv::GaussianBlur(octave_base,
                         first,
                         cv::Size(0, 0),
                         incremental[0],
                         incremental[0],
                         cv::BORDER_REFLECT_101);
        octave.push_back(first);

        for (std::size_t layer = 1; layer < incremental.size(); ++layer) {
            cv::Mat next;
            cv::GaussianBlur(octave.back(),
                             next,
                             cv::Size(0, 0),
                             incremental[layer],
                             incremental[layer],
                             cv::BORDER_REFLECT_101);
            octave.push_back(next);
        }

        pyramid.push_back(octave);

        if (octave_index + 1 < octaves) {
            octave_base = downsampleBy2(
                pyramid.back().at(static_cast<std::size_t>(intervals)));
        }
    }

    return pyramid;
}

Pyramid buildDoGPyramid(const Pyramid& gaussian) {
    Pyramid dog;
    dog.reserve(gaussian.size());

    for (const auto& octave : gaussian) {
        Octave dog_octave;
        if (octave.size() >= 2) {
            dog_octave.reserve(octave.size() - 1);
        }

        for (std::size_t layer = 0; layer + 1 < octave.size(); ++layer) {
            cv::Mat difference;
            cv::subtract(octave[layer + 1], octave[layer], difference,
                         cv::noArray(), CV_32F);
            dog_octave.push_back(difference);
        }
        dog.push_back(std::move(dog_octave));
    }

    return dog;
}

bool isScaleSpaceExtremum(const Octave& dog_octave,
                          int layer,
                          int row,
                          int col) {
    const float center = dog_octave.at(static_cast<std::size_t>(layer))
                             .at<float>(row, col);
    bool greater_than_all = true;
    bool less_than_all = true;

    for (int dl = -1; dl <= 1; ++dl) {
        const cv::Mat& image =
            dog_octave.at(static_cast<std::size_t>(layer + dl));

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dl == 0 && dy == 0 && dx == 0) {
                    continue;
                }

                const float neighbor = image.at<float>(row + dy, col + dx);
                if (center <= neighbor) {
                    greater_than_all = false;
                }
                if (center >= neighbor) {
                    less_than_all = false;
                }

                if (!greater_than_all && !less_than_all) {
                    return false;
                }
            }
        }
    }

    return greater_than_all || less_than_all;
}

std::vector<ScaleExtremum> detectScaleSpaceExtrema(
    const Pyramid& dog,
    double contrast_threshold) {
    std::vector<ScaleExtremum> extrema;

    for (std::size_t octave_index = 0; octave_index < dog.size(); ++octave_index) {
        const Octave& dog_octave = dog[octave_index];
        if (dog_octave.size() < 3) {
            continue;
        }

        for (std::size_t layer = 1; layer + 1 < dog_octave.size(); ++layer) {
            const cv::Mat& image = dog_octave[layer];

            for (int row = 1; row < image.rows - 1; ++row) {
                for (int col = 1; col < image.cols - 1; ++col) {
                    const float response = image.at<float>(row, col);
                    if (std::abs(static_cast<double>(response)) <
                        contrast_threshold) {
                        continue;
                    }

                    if (isScaleSpaceExtremum(
                            dog_octave,
                            static_cast<int>(layer),
                            row,
                            col)) {
                        extrema.push_back({
                            static_cast<int>(octave_index),
                            static_cast<int>(layer),
                            cv::Point(col, row),
                            response
                        });
                    }
                }
            }
        }
    }

    return extrema;
}

cv::Mat visualizeSigned(const cv::Mat& input) {
    double min_value = 0.0;
    double max_value = 0.0;
    cv::minMaxLoc(input, &min_value, &max_value);
    const double max_abs = std::max(std::abs(min_value), std::abs(max_value));

    if (max_abs < 1e-12) {
        return cv::Mat(input.size(), CV_8U, cv::Scalar(128));
    }

    cv::Mat normalized =
        input * static_cast<float>(127.0 / max_abs) + 128.0f;
    cv::Mat output;
    normalized.convertTo(output, CV_8U);
    return output;
}

cv::Mat toByteImage(const cv::Mat& input) {
    cv::Mat output;
    input.convertTo(output, CV_8U);
    return output;
}

std::string indexedName(const std::string& prefix,
                        int octave,
                        int layer) {
    std::ostringstream stream;
    stream << prefix
           << "_o" << std::setw(2) << std::setfill('0') << octave
           << "_l" << std::setw(2) << std::setfill('0') << layer
           << ".png";
    return stream.str();
}

bool writeImage(const fs::path& path, const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image)) {
        std::cerr << "Error: failed to write " << path << '\n';
        return false;
    }
    return true;
}

cv::Mat drawExtremaOnBase(const cv::Mat& input,
                          const std::vector<ScaleExtremum>& extrema) {
    cv::Mat base_u8 = toByteImage(input);
    cv::Mat output;
    cv::cvtColor(base_u8, output, cv::COLOR_GRAY2BGR);

    for (const auto& extremum : extrema) {
        const int scale = 1 << extremum.octave;
        const cv::Point original_point(
            extremum.point.x * scale,
            extremum.point.y * scale);
        const int radius = std::max(2, scale + extremum.dog_layer);
        cv::circle(output,
                   original_point,
                   radius,
                   cv::Scalar(0, 0, 255),
                   1,
                   cv::LINE_AA);
    }

    return output;
}

double maxAbsValue(const cv::Mat& input) {
    double min_value = 0.0;
    double max_value = 0.0;
    cv::minMaxLoc(input, &min_value, &max_value);
    return std::max(std::abs(min_value), std::abs(max_value));
}

bool hasExtremumNear(const std::vector<ScaleExtremum>& extrema,
                     int octave,
                     int layer,
                     cv::Point point,
                     double radius) {
    const double radius_sq = radius * radius;
    for (const auto& extremum : extrema) {
        if (extremum.octave != octave || extremum.dog_layer != layer) {
            continue;
        }
        const double dx = extremum.point.x - point.x;
        const double dy = extremum.point.y - point.y;
        if (dx * dx + dy * dy <= radius_sq) {
            return true;
        }
    }
    return false;
}

bool runSelfTest() {
    bool passed = true;
    auto check = [&passed](bool condition, const std::string& name) {
        if (!condition) {
            std::cerr << "Self-test failed: " << name << '\n';
            passed = false;
        }
    };

    const int intervals = 3;
    const double sigma0 = 1.6;
    const double k = scaleStep(intervals);
    check(std::abs(std::pow(k, intervals) - 2.0) <= 1e-12,
          "k^intervals equals 2");

    const auto absolute = absoluteSigmas(intervals, sigma0);
    const auto incremental = incrementalSigmas(intervals, sigma0);
    check(absolute.size() == static_cast<std::size_t>(intervals + 3),
          "Gaussian octave has intervals + 3 layers");
    check(incremental.size() == absolute.size(),
          "incremental sigma table matches Gaussian layer count");

    for (std::size_t i = 1; i < absolute.size(); ++i) {
        const double reconstructed =
            std::sqrt(absolute[i - 1] * absolute[i - 1] +
                      incremental[i] * incremental[i]);
        check(std::abs(reconstructed - absolute[i]) <= 1e-10,
              "incremental sigma reaches target absolute sigma");
    }

    cv::Mat constant(64, 64, CV_32F, cv::Scalar(90.0f));
    const Pyramid constant_gaussian =
        buildGaussianPyramid(constant, 3, intervals, sigma0);
    const Pyramid constant_dog = buildDoGPyramid(constant_gaussian);

    check(!constant_gaussian.empty(), "constant Gaussian pyramid is created");
    check(constant_gaussian.front().size() ==
              static_cast<std::size_t>(intervals + 3),
          "Gaussian layer count is correct");
    check(constant_dog.front().size() ==
              static_cast<std::size_t>(intervals + 2),
          "DoG layer count is Gaussian count minus one");

    bool dog_is_zero = true;
    for (const auto& octave : constant_dog) {
        for (const auto& layer : octave) {
            if (maxAbsValue(layer) > 1e-4) {
                dog_is_zero = false;
            }
        }
    }
    check(dog_is_zero, "constant image produces near-zero DoG response");
    check(detectScaleSpaceExtrema(constant_dog, 0.1).empty(),
          "constant image produces no scale-space extrema");

    if (constant_gaussian.size() >= 2) {
        check(constant_gaussian[1][0].cols ==
                  std::max(1, constant_gaussian[0][0].cols / 2) &&
              constant_gaussian[1][0].rows ==
                  std::max(1, constant_gaussian[0][0].rows / 2),
              "next octave is downsampled by two");
    }

    cv::Mat impulse = cv::Mat::zeros(65, 65, CV_32F);
    impulse.at<float>(32, 32) = 255.0f;
    const Pyramid impulse_gaussian =
        buildGaussianPyramid(impulse, 1, intervals, sigma0);
    check(impulse_gaussian.front().front().at<float>(32, 32) >
              impulse_gaussian.front().back().at<float>(32, 32),
          "Gaussian peak decreases as scale increases");

    const Pyramid impulse_dog = buildDoGPyramid(impulse_gaussian);
    cv::Mat explicit_difference;
    cv::subtract(impulse_gaussian[0][1],
                 impulse_gaussian[0][0],
                 explicit_difference,
                 cv::noArray(),
                 CV_32F);
    cv::Mat dog_difference;
    cv::absdiff(explicit_difference, impulse_dog[0][0], dog_difference);
    check(maxAbsValue(dog_difference) <= 1e-6,
          "DoG layer equals adjacent Gaussian subtraction");

    Pyramid synthetic_dog(1);
    for (int layer = 0; layer < 3; ++layer) {
        synthetic_dog[0].push_back(
            cv::Mat::zeros(7, 7, CV_32F));
    }
    synthetic_dog[0][1].at<float>(3, 3) = 10.0f;
    synthetic_dog[0][1].at<float>(2, 2) = -12.0f;

    const auto synthetic_extrema =
        detectScaleSpaceExtrema(synthetic_dog, 1.0);
    check(hasExtremumNear(synthetic_extrema, 0, 1, cv::Point(3, 3), 0.1),
          "26-neighbor test detects a positive scale-space extremum");
    check(hasExtremumNear(synthetic_extrema, 0, 1, cv::Point(2, 2), 0.1),
          "26-neighbor test detects a negative scale-space extremum");

    if (passed) {
        std::cout << "SIFT scale-space self-test passed.\n";
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

    const cv::Mat input_u8 =
        cv::imread(options.input_path, cv::IMREAD_GRAYSCALE);
    if (input_u8.empty()) {
        std::cerr << "Error: could not read input image: "
                  << options.input_path << '\n';
        return 1;
    }

    const cv::Mat input = toFloatGray(input_u8);
    const Pyramid gaussian = buildGaussianPyramid(
        input, options.octaves, options.intervals, options.sigma0);
    const Pyramid dog = buildDoGPyramid(gaussian);
    const auto extrema =
        detectScaleSpaceExtrema(dog, options.contrast_threshold);

    const fs::path output_dir(options.output_dir);
    const fs::path gaussian_dir = output_dir / "gaussian";
    const fs::path dog_dir = output_dir / "dog";

    std::error_code error;
    fs::create_directories(gaussian_dir, error);
    if (error) {
        std::cerr << "Error: could not create " << gaussian_dir
                  << ": " << error.message() << '\n';
        return 1;
    }
    fs::create_directories(dog_dir, error);
    if (error) {
        std::cerr << "Error: could not create " << dog_dir
                  << ": " << error.message() << '\n';
        return 1;
    }

    bool ok = true;
    for (std::size_t octave = 0; octave < gaussian.size(); ++octave) {
        for (std::size_t layer = 0; layer < gaussian[octave].size(); ++layer) {
            ok &= writeImage(
                gaussian_dir /
                    indexedName("gaussian",
                                static_cast<int>(octave),
                                static_cast<int>(layer)),
                toByteImage(gaussian[octave][layer]));
        }
    }

    for (std::size_t octave = 0; octave < dog.size(); ++octave) {
        for (std::size_t layer = 0; layer < dog[octave].size(); ++layer) {
            ok &= writeImage(
                dog_dir /
                    indexedName("dog",
                                static_cast<int>(octave),
                                static_cast<int>(layer)),
                visualizeSigned(dog[octave][layer]));
        }
    }

    ok &= writeImage(output_dir / "scale_space_extrema.png",
                     drawExtremaOnBase(input, extrema));

    if (!ok) {
        return 1;
    }

    const double k = scaleStep(options.intervals);
    std::cout << "Input: " << options.input_path << '\n'
              << "Base image size: " << input.cols << "x" << input.rows << '\n'
              << "Built octaves: " << gaussian.size() << '\n'
              << "Intervals per octave: " << options.intervals << '\n'
              << "Gaussian layers per octave: " << options.intervals + 3 << '\n'
              << "DoG layers per octave: " << options.intervals + 2 << '\n'
              << "Base sigma: " << options.sigma0 << '\n'
              << "Scale multiplier k: " << k << '\n'
              << "Raw extrema above threshold: " << extrema.size() << '\n'
              << "Outputs written to: " << output_dir << '\n';

    return 0;
}
