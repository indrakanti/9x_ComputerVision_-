#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string input_path;
    std::string output_dir;
    double harris_k = 0.04;
    double harris_quality = 0.01;
    double shi_quality = 0.01;
    int max_corners = 250;
    int min_distance = 8;
    int fast_threshold = 20;
    bool self_test = false;
    bool help = false;
};

struct TensorFields {
    cv::Mat ix2;
    cv::Mat iy2;
    cv::Mat ixy;
};

struct CornerResults {
    cv::Mat harris_response;
    cv::Mat shi_response;
    std::vector<cv::Point> harris_points;
    std::vector<cv::Point> shi_points;
    std::vector<cv::KeyPoint> fast_keypoints;
};

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program
        << " --input <image> --output-dir <directory> [options]\n"
        << "  " << program << " --self-test\n\n"
        << "Options:\n"
        << "  --input <path>              Input image\n"
        << "  --output-dir <path>         Output directory\n"
        << "  --harris-k <value>          Harris k (default: 0.04)\n"
        << "  --harris-quality <0..1>     Fraction of max Harris response (default: 0.01)\n"
        << "  --shi-quality <0..1>        Fraction of max Shi response (default: 0.01)\n"
        << "  --max-corners <N>           Max Harris/Shi corners (default: 250)\n"
        << "  --min-distance <pixels>     Minimum point spacing (default: 8)\n"
        << "  --fast-threshold <value>    FAST intensity threshold (default: 20)\n"
        << "  --self-test                 Run deterministic tests\n"
        << "  --help                      Show this help\n";
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
            } else if (arg == "--harris-k") {
                options.harris_k = std::stod(requireValue(i, arg));
            } else if (arg == "--harris-quality") {
                options.harris_quality = std::stod(requireValue(i, arg));
            } else if (arg == "--shi-quality") {
                options.shi_quality = std::stod(requireValue(i, arg));
            } else if (arg == "--max-corners") {
                options.max_corners = std::stoi(requireValue(i, arg));
            } else if (arg == "--min-distance") {
                options.min_distance = std::stoi(requireValue(i, arg));
            } else if (arg == "--fast-threshold") {
                options.fast_threshold = std::stoi(requireValue(i, arg));
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
    if (options.harris_k <= 0.0 || options.harris_k >= 0.25) {
        std::cerr << "Error: --harris-k must be in (0, 0.25).\n";
        return false;
    }
    if (options.harris_quality <= 0.0 || options.harris_quality > 1.0 ||
        options.shi_quality <= 0.0 || options.shi_quality > 1.0) {
        std::cerr << "Error: quality values must be in (0, 1].\n";
        return false;
    }
    if (options.max_corners <= 0 || options.min_distance < 0 ||
        options.fast_threshold < 0) {
        std::cerr << "Error: max-corners must be > 0; distances/thresholds must be >= 0.\n";
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

TensorFields computeStructureTensor(const cv::Mat& image, int block_size) {
    cv::Mat gx;
    cv::Mat gy;
    cv::Sobel(image, gx, CV_32F, 1, 0, 3);
    cv::Sobel(image, gy, CV_32F, 0, 1, 3);

    cv::Mat gx2 = gx.mul(gx);
    cv::Mat gy2 = gy.mul(gy);
    cv::Mat gxy = gx.mul(gy);

    TensorFields tensor;
    cv::boxFilter(gx2, tensor.ix2, CV_32F, cv::Size(block_size, block_size),
                  cv::Point(-1, -1), false, cv::BORDER_REFLECT_101);
    cv::boxFilter(gy2, tensor.iy2, CV_32F, cv::Size(block_size, block_size),
                  cv::Point(-1, -1), false, cv::BORDER_REFLECT_101);
    cv::boxFilter(gxy, tensor.ixy, CV_32F, cv::Size(block_size, block_size),
                  cv::Point(-1, -1), false, cv::BORDER_REFLECT_101);
    return tensor;
}

cv::Mat harrisResponse(const TensorFields& tensor, double k) {
    cv::Mat response(tensor.ix2.size(), CV_32F);
    for (int row = 0; row < response.rows; ++row) {
        for (int col = 0; col < response.cols; ++col) {
            const float a = tensor.ix2.at<float>(row, col);
            const float b = tensor.ixy.at<float>(row, col);
            const float c = tensor.iy2.at<float>(row, col);
            const double det = static_cast<double>(a) * c -
                               static_cast<double>(b) * b;
            const double trace = static_cast<double>(a) + c;
            response.at<float>(row, col) =
                static_cast<float>(det - k * trace * trace);
        }
    }
    return response;
}

cv::Mat shiTomasiResponse(const TensorFields& tensor) {
    cv::Mat response(tensor.ix2.size(), CV_32F);
    for (int row = 0; row < response.rows; ++row) {
        for (int col = 0; col < response.cols; ++col) {
            const float a = tensor.ix2.at<float>(row, col);
            const float b = tensor.ixy.at<float>(row, col);
            const float c = tensor.iy2.at<float>(row, col);
            const double trace = static_cast<double>(a) + c;
            const double discriminant =
                std::sqrt(std::max(0.0,
                    static_cast<double>(a - c) * (a - c) +
                    4.0 * static_cast<double>(b) * b));
            response.at<float>(row, col) =
                static_cast<float>(0.5 * (trace - discriminant));
        }
    }
    return response;
}

std::vector<cv::Point> selectLocalMaxima(const cv::Mat& response,
                                         double quality,
                                         int max_corners,
                                         int min_distance) {
    double min_value = 0.0;
    double max_value = 0.0;
    cv::minMaxLoc(response, &min_value, &max_value);
    if (max_value <= 0.0) {
        return {};
    }

    const float threshold = static_cast<float>(quality * max_value);

    cv::Mat dilated;
    cv::dilate(response, dilated, cv::Mat());

    struct Candidate {
        float response;
        cv::Point point;
    };
    std::vector<Candidate> candidates;

    for (int row = 1; row < response.rows - 1; ++row) {
        for (int col = 1; col < response.cols - 1; ++col) {
            const float value = response.at<float>(row, col);
            if (value >= threshold &&
                value >= dilated.at<float>(row, col)) {
                candidates.push_back({value, cv::Point(col, row)});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& lhs, const Candidate& rhs) {
                  return lhs.response > rhs.response;
              });

    std::vector<cv::Point> selected;
    const double min_distance_sq =
        static_cast<double>(min_distance) * min_distance;

    for (const auto& candidate : candidates) {
        bool far_enough = true;
        for (const auto& existing : selected) {
            const double dx = candidate.point.x - existing.x;
            const double dy = candidate.point.y - existing.y;
            if (dx * dx + dy * dy < min_distance_sq) {
                far_enough = false;
                break;
            }
        }
        if (far_enough) {
            selected.push_back(candidate.point);
            if (static_cast<int>(selected.size()) >= max_corners) {
                break;
            }
        }
    }

    return selected;
}

CornerResults detectCorners(const cv::Mat& image,
                            double harris_k,
                            double harris_quality,
                            double shi_quality,
                            int max_corners,
                            int min_distance,
                            int fast_threshold) {
    const TensorFields tensor = computeStructureTensor(image, 3);

    CornerResults results;
    results.harris_response = harrisResponse(tensor, harris_k);
    results.shi_response = shiTomasiResponse(tensor);

    results.harris_points = selectLocalMaxima(
        results.harris_response, harris_quality, max_corners, min_distance);
    results.shi_points = selectLocalMaxima(
        results.shi_response, shi_quality, max_corners, min_distance);

    cv::Mat image_u8;
    image.convertTo(image_u8, CV_8U);
    cv::FAST(image_u8,
             results.fast_keypoints,
             fast_threshold,
             true,
             cv::FastFeatureDetector::TYPE_9_16);

    if (static_cast<int>(results.fast_keypoints.size()) > max_corners) {
        std::sort(results.fast_keypoints.begin(), results.fast_keypoints.end(),
                  [](const cv::KeyPoint& lhs, const cv::KeyPoint& rhs) {
                      return lhs.response > rhs.response;
                  });
        results.fast_keypoints.resize(static_cast<std::size_t>(max_corners));
    }

    return results;
}

cv::Mat visualizeResponse(const cv::Mat& response) {
    double min_value = 0.0;
    double max_value = 0.0;
    cv::minMaxLoc(response, &min_value, &max_value);

    cv::Mat output;
    if (max_value <= min_value) {
        return cv::Mat(response.size(), CV_8U, cv::Scalar(0));
    }

    cv::normalize(response, output, 0, 255, cv::NORM_MINMAX, CV_8U);
    return output;
}

cv::Mat drawPoints(const cv::Mat& image,
                   const std::vector<cv::Point>& points) {
    cv::Mat gray_u8;
    image.convertTo(gray_u8, CV_8U);
    cv::Mat output;
    cv::cvtColor(gray_u8, output, cv::COLOR_GRAY2BGR);

    for (const auto& point : points) {
        cv::circle(output, point, 3, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }
    return output;
}

cv::Mat drawKeypoints(const cv::Mat& image,
                      const std::vector<cv::KeyPoint>& keypoints) {
    cv::Mat gray_u8;
    image.convertTo(gray_u8, CV_8U);
    cv::Mat output;
    cv::drawKeypoints(gray_u8,
                      keypoints,
                      output,
                      cv::Scalar(0, 0, 255),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    return output;
}

bool writeImage(const fs::path& path, const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image)) {
        std::cerr << "Error: failed to write " << path << '\n';
        return false;
    }
    return true;
}

std::pair<double, double> eigenvalues2x2(double a, double b, double c) {
    const double trace = a + c;
    const double discriminant =
        std::sqrt(std::max(0.0, (a - c) * (a - c) + 4.0 * b * b));
    return {0.5 * (trace + discriminant),
            0.5 * (trace - discriminant)};
}

bool hasPointNear(const std::vector<cv::Point>& points,
                  cv::Point expected,
                  double radius) {
    const double radius_sq = radius * radius;
    for (const auto& point : points) {
        const double dx = point.x - expected.x;
        const double dy = point.y - expected.y;
        if (dx * dx + dy * dy <= radius_sq) {
            return true;
        }
    }
    return false;
}

bool hasKeypointNear(const std::vector<cv::KeyPoint>& points,
                     cv::Point2f expected,
                     double radius) {
    const double radius_sq = radius * radius;
    for (const auto& keypoint : points) {
        const double dx = keypoint.pt.x - expected.x;
        const double dy = keypoint.pt.y - expected.y;
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

    const auto flat_eigen = eigenvalues2x2(0.0, 0.0, 0.0);
    check(std::abs(flat_eigen.first) <= 1e-12 &&
          std::abs(flat_eigen.second) <= 1e-12,
          "flat tensor has two near-zero eigenvalues");

    const auto edge_eigen = eigenvalues2x2(100.0, 0.0, 0.0);
    check(edge_eigen.first > 0.0 &&
          std::abs(edge_eigen.second) <= 1e-12,
          "edge tensor has one large and one small eigenvalue");

    const auto corner_eigen = eigenvalues2x2(100.0, 0.0, 100.0);
    check(corner_eigen.first > 0.0 &&
          corner_eigen.second > 0.0 &&
          std::abs(corner_eigen.first - corner_eigen.second) <= 1e-12,
          "corner tensor has two strong eigenvalues");

    const double k = 0.04;
    const double flat_harris = 0.0;
    const double edge_harris = 0.0 - k * 100.0 * 100.0;
    const double corner_harris = 100.0 * 100.0 -
                                 k * 200.0 * 200.0;
    check(flat_harris == 0.0, "flat Harris response is zero");
    check(edge_harris < 0.0, "edge Harris response is negative");
    check(corner_harris > 0.0, "corner Harris response is positive");

    cv::Mat constant(64, 64, CV_32F, cv::Scalar(100.0f));
    const auto constant_result =
        detectCorners(constant, 0.04, 0.01, 0.01, 100, 5, 10);
    check(constant_result.harris_points.empty(),
          "constant image has no Harris corners");
    check(constant_result.shi_points.empty(),
          "constant image has no Shi-Tomasi corners");
    check(constant_result.fast_keypoints.empty(),
          "constant image has no FAST corners");

    cv::Mat square = cv::Mat::zeros(96, 96, CV_32F);
    cv::rectangle(square, cv::Point(24, 24), cv::Point(72, 72),
                  cv::Scalar(255.0f), cv::FILLED);

    const auto square_result =
        detectCorners(square, 0.04, 0.01, 0.01, 100, 8, 10);

    const std::vector<cv::Point> expected = {
        {24, 24}, {72, 24}, {24, 72}, {72, 72}
    };

    for (const auto& point : expected) {
        check(hasPointNear(square_result.harris_points, point, 5.0),
              "Harris detects synthetic square corner near expected location");
        check(hasPointNear(square_result.shi_points, point, 5.0),
              "Shi-Tomasi detects synthetic square corner near expected location");
    }

    check(!square_result.fast_keypoints.empty(),
          "FAST detects corners on synthetic square");

    bool fast_near_any_corner = false;
    for (const auto& point : expected) {
        if (hasKeypointNear(square_result.fast_keypoints,
                            cv::Point2f(static_cast<float>(point.x),
                                        static_cast<float>(point.y)),
                            6.0)) {
            fast_near_any_corner = true;
            break;
        }
    }
    check(fast_near_any_corner,
          "FAST produces a keypoint near a synthetic square corner");

    cv::Mat edge = cv::Mat::zeros(96, 96, CV_32F);
    edge.colRange(48, edge.cols).setTo(255.0f);
    const TensorFields edge_tensor = computeStructureTensor(edge, 3);
    const cv::Mat edge_harris_response = harrisResponse(edge_tensor, 0.04);
    const cv::Mat edge_shi_response = shiTomasiResponse(edge_tensor);

    double edge_harris_max = 0.0;
    double edge_shi_max = 0.0;
    cv::minMaxLoc(edge_harris_response, nullptr, &edge_harris_max);
    cv::minMaxLoc(edge_shi_response, nullptr, &edge_shi_max);
    check(edge_harris_max <= 1e-3,
          "pure straight edge does not produce positive Harris corner response");
    check(edge_shi_max <= 1e-3,
          "pure straight edge has near-zero Shi-Tomasi minimum eigenvalue");

    if (passed) {
        std::cout << "Corner-detection self-test passed.\n";
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
        std::cerr << "Error: could not read input image: "
                  << options.input_path << '\n';
        return 1;
    }

    const cv::Mat input = toFloatGray(input_u8);
    const CornerResults results =
        detectCorners(input,
                      options.harris_k,
                      options.harris_quality,
                      options.shi_quality,
                      options.max_corners,
                      options.min_distance,
                      options.fast_threshold);

    const fs::path output_dir(options.output_dir);
    std::error_code error;
    fs::create_directories(output_dir, error);
    if (error) {
        std::cerr << "Error: could not create output directory "
                  << output_dir << ": " << error.message() << '\n';
        return 1;
    }

    bool ok = true;
    ok &= writeImage(output_dir / "01_harris_response.png",
                     visualizeResponse(results.harris_response));
    ok &= writeImage(output_dir / "02_harris_corners.png",
                     drawPoints(input, results.harris_points));
    ok &= writeImage(output_dir / "03_shi_tomasi_response.png",
                     visualizeResponse(results.shi_response));
    ok &= writeImage(output_dir / "04_shi_tomasi_corners.png",
                     drawPoints(input, results.shi_points));
    ok &= writeImage(output_dir / "05_fast_keypoints.png",
                     drawKeypoints(input, results.fast_keypoints));

    if (!ok) {
        return 1;
    }

    std::cout << "Input: " << options.input_path << '\n'
              << "Size: " << input.cols << "x" << input.rows << '\n'
              << "Harris k: " << options.harris_k << '\n'
              << "Harris corners: " << results.harris_points.size() << '\n'
              << "Shi-Tomasi corners: " << results.shi_points.size() << '\n'
              << "FAST keypoints: " << results.fast_keypoints.size() << '\n'
              << "Outputs written to: " << output_dir << '\n';

    return 0;
}
