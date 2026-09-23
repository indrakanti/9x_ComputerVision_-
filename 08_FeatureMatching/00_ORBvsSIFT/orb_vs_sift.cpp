#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

struct Options {
    std::string image1_path;
    std::string image2_path;
    std::string output_dir;
    int max_features = 500;
    double ratio_threshold = 0.75;
    int benchmark_iterations = 20;
    bool self_test = false;
    bool help = false;
};

struct FeatureSet {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    double detect_describe_ms = 0.0;
};

struct Match {
    int query_index = -1;
    int train_index = -1;
    float best_distance = 0.0f;
    float second_distance = 0.0f;
};

struct MethodReport {
    std::string name;
    FeatureSet image1;
    FeatureSet image2;
    std::vector<Match> matches;
    double match_ms = 0.0;
    std::size_t bytes_per_descriptor = 0;
    std::size_t total_descriptor_bytes = 0;
};

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program
        << " --image1 <path> --image2 <path> --output-dir <dir> [options]\n"
        << "  " << program << " --self-test\n\n"
        << "Options:\n"
        << "  --image1 <path>           First grayscale image\n"
        << "  --image2 <path>           Second grayscale image\n"
        << "  --output-dir <path>       Output directory\n"
        << "  --max-features <N>        Feature budget for both methods (default: 500)\n"
        << "  --ratio <0..1>            Lowe-style ratio threshold (default: 0.75)\n"
        << "  --iterations <N>          Benchmark repetitions (default: 20)\n"
        << "  --self-test               Run deterministic tests\n"
        << "  --help                    Show this help\n";
}

bool parseArguments(int argc, char** argv, Options& options) {
    auto value = [argc, argv](int& i, const std::string& name) {
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value for " + name);
        }
        ++i;
        return std::string(argv[i]);
    };

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--image1") {
                options.image1_path = value(i, arg);
            } else if (arg == "--image2") {
                options.image2_path = value(i, arg);
            } else if (arg == "--output-dir") {
                options.output_dir = value(i, arg);
            } else if (arg == "--max-features") {
                options.max_features = std::stoi(value(i, arg));
            } else if (arg == "--ratio") {
                options.ratio_threshold = std::stod(value(i, arg));
            } else if (arg == "--iterations") {
                options.benchmark_iterations = std::stoi(value(i, arg));
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
    if (options.help || options.self_test) {
        return true;
    }
    if (options.image1_path.empty() ||
        options.image2_path.empty() ||
        options.output_dir.empty()) {
        std::cerr << "Error: --image1, --image2 and --output-dir are required.\n";
        return false;
    }
    if (options.max_features <= 0 || options.benchmark_iterations <= 0) {
        std::cerr << "Error: feature count and iterations must be > 0.\n";
        return false;
    }
    if (options.ratio_threshold <= 0.0 || options.ratio_threshold >= 1.0) {
        std::cerr << "Error: --ratio must be in (0,1).\n";
        return false;
    }
    return true;
}

float l2Distance(const cv::Mat& a, const cv::Mat& b) {
    CV_Assert(a.type() == CV_32F && b.type() == CV_32F);
    CV_Assert(a.rows == 1 && b.rows == 1 && a.cols == b.cols);

    double sum = 0.0;
    for (int col = 0; col < a.cols; ++col) {
        const double delta =
            static_cast<double>(a.at<float>(0, col)) -
            static_cast<double>(b.at<float>(0, col));
        sum += delta * delta;
    }
    return static_cast<float>(std::sqrt(sum));
}

int popcountByte(unsigned char value) {
    int count = 0;
    while (value != 0U) {
        value &= static_cast<unsigned char>(value - 1U);
        ++count;
    }
    return count;
}

float hammingDistance(const cv::Mat& a, const cv::Mat& b) {
    CV_Assert(a.type() == CV_8U && b.type() == CV_8U);
    CV_Assert(a.rows == 1 && b.rows == 1 && a.cols == b.cols);

    int distance = 0;
    for (int col = 0; col < a.cols; ++col) {
        const unsigned char x =
            static_cast<unsigned char>(
                a.at<unsigned char>(0, col) ^
                b.at<unsigned char>(0, col));
        distance += popcountByte(x);
    }
    return static_cast<float>(distance);
}

template <typename Distance>
std::vector<Match> ratioMatch(const cv::Mat& query,
                              const cv::Mat& train,
                              double ratio_threshold,
                              Distance distance) {
    std::vector<Match> matches;
    if (query.empty() || train.rows < 2) {
        return matches;
    }

    for (int query_index = 0; query_index < query.rows; ++query_index) {
        float best = std::numeric_limits<float>::infinity();
        float second = std::numeric_limits<float>::infinity();
        int best_index = -1;

        const cv::Mat query_row = query.row(query_index);
        for (int train_index = 0; train_index < train.rows; ++train_index) {
            const float d = distance(query_row, train.row(train_index));
            if (d < best) {
                second = best;
                best = d;
                best_index = train_index;
            } else if (d < second) {
                second = d;
            }
        }

        if (best_index >= 0 &&
            std::isfinite(second) &&
            best < static_cast<float>(ratio_threshold) * second) {
            matches.push_back({
                query_index,
                best_index,
                best,
                second
            });
        }
    }
    return matches;
}

FeatureSet runDetector(const cv::Mat& image,
                       const cv::Ptr<cv::Feature2D>& detector) {
    FeatureSet result;
    const auto start = Clock::now();
    detector->detectAndCompute(
        image,
        cv::noArray(),
        result.keypoints,
        result.descriptors);
    const auto end = Clock::now();

    result.detect_describe_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}

double benchmarkDetector(const cv::Mat& image,
                         const std::function<cv::Ptr<cv::Feature2D>()>& factory,
                         int iterations) {
    double total_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        auto detector = factory();
        const auto start = Clock::now();
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        detector->detectAndCompute(
            image,
            cv::noArray(),
            keypoints,
            descriptors);
        const auto end = Clock::now();
        total_ms +=
            std::chrono::duration<double, std::milli>(end - start).count();
    }
    return total_ms / iterations;
}

template <typename Distance>
double benchmarkMatching(const cv::Mat& query,
                         const cv::Mat& train,
                         double ratio,
                         int iterations,
                         Distance distance) {
    if (query.empty() || train.rows < 2) {
        return 0.0;
    }

    double total_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        const auto start = Clock::now();
        const auto matches =
            ratioMatch(query, train, ratio, distance);
        const auto end = Clock::now();
        total_ms +=
            std::chrono::duration<double, std::milli>(end - start).count();
        if (matches.size() > static_cast<std::size_t>(
                query.rows + train.rows)) {
            throw std::runtime_error("Unreachable match-count guard.");
        }
    }
    return total_ms / iterations;
}

std::size_t descriptorBytesPerRow(const cv::Mat& descriptors) {
    if (descriptors.empty()) {
        return 0;
    }
    return static_cast<std::size_t>(descriptors.cols) *
           descriptors.elemSize();
}

std::size_t descriptorBytesTotal(const cv::Mat& descriptors) {
    if (descriptors.empty()) {
        return 0;
    }
    return static_cast<std::size_t>(descriptors.rows) *
           descriptorBytesPerRow(descriptors);
}

MethodReport runSift(const cv::Mat& image1,
                     const cv::Mat& image2,
                     int max_features,
                     double ratio,
                     int iterations) {
    const auto factory = [max_features]() {
        return cv::SIFT::create(max_features);
    };

    MethodReport report;
    report.name = "SIFT";
    report.image1 = runDetector(image1, factory());
    report.image2 = runDetector(image2, factory());

    const auto start = Clock::now();
    report.matches =
        ratioMatch(
            report.image1.descriptors,
            report.image2.descriptors,
            ratio,
            l2Distance);
    const auto end = Clock::now();
    report.match_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    report.image1.detect_describe_ms =
        benchmarkDetector(image1, factory, iterations);
    report.image2.detect_describe_ms =
        benchmarkDetector(image2, factory, iterations);
    report.match_ms =
        benchmarkMatching(
            report.image1.descriptors,
            report.image2.descriptors,
            ratio,
            iterations,
            l2Distance);

    report.bytes_per_descriptor =
        descriptorBytesPerRow(report.image1.descriptors);
    report.total_descriptor_bytes =
        descriptorBytesTotal(report.image1.descriptors) +
        descriptorBytesTotal(report.image2.descriptors);

    return report;
}

MethodReport runOrb(const cv::Mat& image1,
                    const cv::Mat& image2,
                    int max_features,
                    double ratio,
                    int iterations) {
    const auto factory = [max_features]() {
        return cv::ORB::create(
            max_features,
            1.2f,
            8,
            31,
            0,
            2,
            cv::ORB::HARRIS_SCORE,
            31,
            20);
    };

    MethodReport report;
    report.name = "ORB";
    report.image1 = runDetector(image1, factory());
    report.image2 = runDetector(image2, factory());

    const auto start = Clock::now();
    report.matches =
        ratioMatch(
            report.image1.descriptors,
            report.image2.descriptors,
            ratio,
            hammingDistance);
    const auto end = Clock::now();
    report.match_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    report.image1.detect_describe_ms =
        benchmarkDetector(image1, factory, iterations);
    report.image2.detect_describe_ms =
        benchmarkDetector(image2, factory, iterations);
    report.match_ms =
        benchmarkMatching(
            report.image1.descriptors,
            report.image2.descriptors,
            ratio,
            iterations,
            hammingDistance);

    report.bytes_per_descriptor =
        descriptorBytesPerRow(report.image1.descriptors);
    report.total_descriptor_bytes =
        descriptorBytesTotal(report.image1.descriptors) +
        descriptorBytesTotal(report.image2.descriptors);

    return report;
}

cv::Mat drawMatchesManual(
    const cv::Mat& image1,
    const cv::Mat& image2,
    const MethodReport& report) {
    cv::Mat left;
    cv::Mat right;
    cv::cvtColor(image1, left, cv::COLOR_GRAY2BGR);
    cv::cvtColor(image2, right, cv::COLOR_GRAY2BGR);

    const int rows = std::max(left.rows, right.rows);
    cv::Mat output(
        rows,
        left.cols + right.cols,
        CV_8UC3,
        cv::Scalar(0, 0, 0));

    left.copyTo(output(cv::Rect(0, 0, left.cols, left.rows)));
    right.copyTo(output(cv::Rect(
        left.cols, 0, right.cols, right.rows)));

    for (const auto& match : report.matches) {
        const cv::Point p1 =
            report.image1.keypoints[
                static_cast<std::size_t>(match.query_index)].pt;
        cv::Point p2 =
            report.image2.keypoints[
                static_cast<std::size_t>(match.train_index)].pt;
        p2.x += left.cols;

        cv::circle(output, p1, 3, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        cv::circle(output, p2, 3, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        cv::line(output, p1, p2, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
    return output;
}

bool writeImage(const fs::path& path, const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image)) {
        std::cerr << "Error: failed to write " << path << '\n';
        return false;
    }
    return true;
}

void printReport(const MethodReport& report) {
    std::cout
        << "\n" << report.name << "\n"
        << "  image1 keypoints: "
        << report.image1.keypoints.size() << '\n'
        << "  image2 keypoints: "
        << report.image2.keypoints.size() << '\n'
        << "  descriptor type: "
        << (report.image1.descriptors.type() == CV_32F
                ? "CV_32F"
                : report.image1.descriptors.type() == CV_8U
                    ? "CV_8U"
                    : "other") << '\n'
        << "  descriptor columns: "
        << report.image1.descriptors.cols << '\n'
        << "  bytes / descriptor: "
        << report.bytes_per_descriptor << '\n'
        << "  total descriptor bytes (both images): "
        << report.total_descriptor_bytes << '\n'
        << "  accepted ratio-test matches: "
        << report.matches.size() << '\n'
        << std::fixed << std::setprecision(3)
        << "  avg detect+describe image1: "
        << report.image1.detect_describe_ms << " ms\n"
        << "  avg detect+describe image2: "
        << report.image2.detect_describe_ms << " ms\n"
        << "  avg manual match time: "
        << report.match_ms << " ms\n";
}

cv::Mat createSyntheticTexture() {
    cv::Mat image(192, 192, CV_8U, cv::Scalar(30));

    const int cell = 16;
    for (int row = 0; row < image.rows; row += cell) {
        for (int col = 0; col < image.cols; col += cell) {
            if (((row / cell) + (col / cell)) % 2 == 0) {
                cv::rectangle(
                    image,
                    cv::Rect(col, row,
                             std::min(cell, image.cols - col),
                             std::min(cell, image.rows - row)),
                    cv::Scalar(210),
                    cv::FILLED);
            }
        }
    }

    cv::circle(image, cv::Point(48, 48), 18, cv::Scalar(90), 2);
    cv::circle(image, cv::Point(145, 55), 13, cv::Scalar(250), 2);
    cv::line(image, cv::Point(20, 160), cv::Point(170, 115),
             cv::Scalar(120), 3);
    cv::putText(image, "9X", cv::Point(65, 120),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(160), 2);
    return image;
}

bool runSelfTest() {
    bool passed = true;
    auto check = [&passed](bool condition, const std::string& name) {
        if (!condition) {
            std::cerr << "Self-test failed: " << name << '\n';
            passed = false;
        }
    };

    cv::Mat binary_a =
        (cv::Mat_<unsigned char>(1, 4) << 0x00, 0xFF, 0x0F, 0x55);
    cv::Mat binary_b =
        (cv::Mat_<unsigned char>(1, 4) << 0x00, 0x0F, 0xF0, 0xAA);

    const float manual_hamming =
        hammingDistance(binary_a, binary_b);
    const float opencv_hamming =
        static_cast<float>(
            cv::norm(binary_a, binary_b, cv::NORM_HAMMING));
    check(std::abs(manual_hamming - opencv_hamming) <= 1e-6f,
          "manual Hamming distance matches OpenCV");

    cv::Mat float_a =
        (cv::Mat_<float>(1, 4) << 0.0f, 1.0f, 2.0f, 3.0f);
    cv::Mat float_b =
        (cv::Mat_<float>(1, 4) << 0.5f, 1.5f, 1.0f, 4.0f);

    const float manual_l2 = l2Distance(float_a, float_b);
    const float opencv_l2 =
        static_cast<float>(
            cv::norm(float_a, float_b, cv::NORM_L2));
    check(std::abs(manual_l2 - opencv_l2) <= 1e-6f,
          "manual L2 distance matches OpenCV");

    cv::Mat binary_query =
        (cv::Mat_<unsigned char>(1, 2) << 0x00, 0x00);
    cv::Mat binary_train =
        (cv::Mat_<unsigned char>(3, 2) <<
            0x00, 0x01,
            0x0F, 0x0F,
            0xFF, 0xFF);
    const auto binary_matches =
        ratioMatch(
            binary_query,
            binary_train,
            0.75,
            hammingDistance);
    check(binary_matches.size() == 1,
          "Hamming ratio test accepts a distinctive binary neighbor");

    const cv::Mat synthetic = createSyntheticTexture();

    auto sift = cv::SIFT::create(300);
    std::vector<cv::KeyPoint> sift_keypoints;
    cv::Mat sift_descriptors;
    sift->detectAndCompute(
        synthetic,
        cv::noArray(),
        sift_keypoints,
        sift_descriptors);

    check(!sift_descriptors.empty(),
          "SIFT produces descriptors on deterministic texture");
    if (!sift_descriptors.empty()) {
        check(sift_descriptors.type() == CV_32F,
              "SIFT descriptor storage is CV_32F");
        check(sift_descriptors.cols == 128,
              "SIFT descriptor has 128 scalar elements");
        check(descriptorBytesPerRow(sift_descriptors) == 512,
              "SIFT descriptor occupies 512 bytes as 128 floats");
    }

    auto orb = cv::ORB::create(300);
    std::vector<cv::KeyPoint> orb_keypoints;
    cv::Mat orb_descriptors;
    orb->detectAndCompute(
        synthetic,
        cv::noArray(),
        orb_keypoints,
        orb_descriptors);

    check(!orb_descriptors.empty(),
          "ORB produces descriptors on deterministic texture");
    if (!orb_descriptors.empty()) {
        check(orb_descriptors.type() == CV_8U,
              "ORB descriptor storage is CV_8U");
        check(orb_descriptors.cols == 32,
              "ORB descriptor has 32 bytes");
        check(descriptorBytesPerRow(orb_descriptors) == 32,
              "ORB descriptor occupies 32 bytes / 256 bits");
    }

    if (!sift_descriptors.empty() && !orb_descriptors.empty()) {
        check(descriptorBytesPerRow(sift_descriptors) >
                  descriptorBytesPerRow(orb_descriptors),
              "SIFT descriptor storage per feature is larger than ORB");
    }

    if (passed) {
        std::cout << "ORB-vs-SIFT self-test passed.\n";
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

    const cv::Mat image1 =
        cv::imread(options.image1_path, cv::IMREAD_GRAYSCALE);
    const cv::Mat image2 =
        cv::imread(options.image2_path, cv::IMREAD_GRAYSCALE);
    if (image1.empty() || image2.empty()) {
        std::cerr << "Error: could not read both images.\n";
        return 1;
    }

    const MethodReport sift =
        runSift(
            image1,
            image2,
            options.max_features,
            options.ratio_threshold,
            options.benchmark_iterations);
    const MethodReport orb =
        runOrb(
            image1,
            image2,
            options.max_features,
            options.ratio_threshold,
            options.benchmark_iterations);

    const fs::path output_dir(options.output_dir);
    std::error_code error;
    fs::create_directories(output_dir, error);
    if (error) {
        std::cerr << "Error: could not create output directory "
                  << output_dir << ": " << error.message() << '\n';
        return 1;
    }

    bool ok = true;
    ok &= writeImage(
        output_dir / "sift_matches.png",
        drawMatchesManual(image1, image2, sift));
    ok &= writeImage(
        output_dir / "orb_matches.png",
        drawMatchesManual(image1, image2, orb));

    if (!ok) {
        return 1;
    }

    std::cout
        << "Comparison uses the same feature budget (max "
        << options.max_features << ") and ratio threshold ("
        << options.ratio_threshold << ").\n"
        << "Timing is illustrative for this machine/run; CI does not assert performance.\n";

    printReport(sift);
    printReport(orb);

    std::cout
        << "\nDistance metrics\n"
        << "  SIFT: Euclidean/L2 over float descriptors\n"
        << "  ORB:  Hamming over binary descriptors\n"
        << "\nOutputs\n"
        << "  " << (output_dir / "sift_matches.png") << '\n'
        << "  " << (output_dir / "orb_matches.png") << '\n';

    return 0;
}
