#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string image1_path;
    std::string image2_path;
    std::string output_dir;
    int max_features = 800;
    double ratio_threshold = 0.75;
    int ransac_iterations = 1000;
    double reprojection_threshold = 3.0;
    unsigned int seed = 42;
    bool self_test = false;
    bool help = false;
};

struct Match {
    int query_index = -1;
    int train_index = -1;
    float best_distance = 0.0f;
    float second_distance = 0.0f;
};

struct Correspondence {
    cv::Point2f source;
    cv::Point2f destination;
    int match_index = -1;
};

struct RansacResult {
    cv::Mat homography;
    std::vector<unsigned char> inlier_mask;
    int inlier_count = 0;
    double inlier_rmse = std::numeric_limits<double>::infinity();
};

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program
        << " --image1 <path> --image2 <path> --output-dir <dir> [options]\n"
        << "  " << program << " --self-test\n\n"
        << "Options:\n"
        << "  --image1 <path>               First grayscale image\n"
        << "  --image2 <path>               Second grayscale image\n"
        << "  --output-dir <path>           Output directory\n"
        << "  --max-features <N>            SIFT feature budget (default: 800)\n"
        << "  --ratio <0..1>                Lowe ratio threshold (default: 0.75)\n"
        << "  --ransac-iterations <N>       RANSAC hypotheses (default: 1000)\n"
        << "  --reprojection-threshold <px> Inlier threshold in pixels (default: 3.0)\n"
        << "  --seed <N>                    Deterministic RANSAC seed (default: 42)\n"
        << "  --self-test                   Run deterministic tests\n"
        << "  --help                        Show this help\n";
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
            } else if (arg == "--ransac-iterations") {
                options.ransac_iterations = std::stoi(value(i, arg));
            } else if (arg == "--reprojection-threshold") {
                options.reprojection_threshold = std::stod(value(i, arg));
            } else if (arg == "--seed") {
                options.seed = static_cast<unsigned int>(std::stoul(value(i, arg)));
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
    if (options.max_features <= 0 ||
        options.ransac_iterations <= 0 ||
        options.reprojection_threshold <= 0.0) {
        std::cerr << "Error: feature count, RANSAC iterations, and reprojection threshold must be > 0.\n";
        return false;
    }
    if (options.ratio_threshold <= 0.0 ||
        options.ratio_threshold >= 1.0) {
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

std::vector<Match> ratioMatch(const cv::Mat& query,
                              const cv::Mat& train,
                              double ratio_threshold) {
    std::vector<Match> matches;
    if (query.empty() || train.rows < 2) {
        return matches;
    }

    for (int query_index = 0; query_index < query.rows; ++query_index) {
        float best = std::numeric_limits<float>::infinity();
        float second = std::numeric_limits<float>::infinity();
        int best_index = -1;

        for (int train_index = 0; train_index < train.rows; ++train_index) {
            const float distance =
                l2Distance(query.row(query_index), train.row(train_index));

            if (distance < best) {
                second = best;
                best = distance;
                best_index = train_index;
            } else if (distance < second) {
                second = distance;
            }
        }

        if (best_index >= 0 &&
            std::isfinite(second) &&
            best < static_cast<float>(ratio_threshold) * second) {
            matches.push_back({query_index, best_index, best, second});
        }
    }

    return matches;
}

std::vector<Correspondence> toCorrespondences(
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<Match>& matches) {
    std::vector<Correspondence> correspondences;
    correspondences.reserve(matches.size());

    for (std::size_t index = 0; index < matches.size(); ++index) {
        const Match& match = matches[index];

        if (match.query_index < 0 ||
            match.query_index >= static_cast<int>(keypoints1.size()) ||
            match.train_index < 0 ||
            match.train_index >= static_cast<int>(keypoints2.size())) {
            continue;
        }

        correspondences.push_back({
            keypoints1[static_cast<std::size_t>(match.query_index)].pt,
            keypoints2[static_cast<std::size_t>(match.train_index)].pt,
            static_cast<int>(index)
        });
    }

    return correspondences;
}

bool projectPoint(const cv::Mat& homography,
                  const cv::Point2f& point,
                  cv::Point2f& projected) {
    if (homography.empty() || homography.rows != 3 || homography.cols != 3) {
        return false;
    }

    cv::Mat h64;
    homography.convertTo(h64, CV_64F);

    const double x = point.x;
    const double y = point.y;
    const double denominator =
        h64.at<double>(2, 0) * x +
        h64.at<double>(2, 1) * y +
        h64.at<double>(2, 2);

    if (std::abs(denominator) < 1e-12) {
        return false;
    }

    projected.x = static_cast<float>(
        (h64.at<double>(0, 0) * x +
         h64.at<double>(0, 1) * y +
         h64.at<double>(0, 2)) / denominator);
    projected.y = static_cast<float>(
        (h64.at<double>(1, 0) * x +
         h64.at<double>(1, 1) * y +
         h64.at<double>(1, 2)) / denominator);

    return std::isfinite(projected.x) && std::isfinite(projected.y);
}

double reprojectionError(const cv::Mat& homography,
                         const Correspondence& correspondence) {
    cv::Point2f projected;
    if (!projectPoint(homography, correspondence.source, projected)) {
        return std::numeric_limits<double>::infinity();
    }

    const double dx = static_cast<double>(projected.x) - correspondence.destination.x;
    const double dy = static_cast<double>(projected.y) - correspondence.destination.y;
    return std::sqrt(dx * dx + dy * dy);
}

std::array<int, 4> sampleFourUnique(int count, std::mt19937& rng) {
    std::uniform_int_distribution<int> distribution(0, count - 1);
    std::array<int, 4> indices{};
    int filled = 0;

    while (filled < 4) {
        const int candidate = distribution(rng);
        bool duplicate = false;
        for (int i = 0; i < filled; ++i) {
            if (indices[static_cast<std::size_t>(i)] == candidate) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            indices[static_cast<std::size_t>(filled)] = candidate;
            ++filled;
        }
    }

    return indices;
}

bool finiteHomography(const cv::Mat& homography) {
    if (homography.empty() || homography.rows != 3 || homography.cols != 3) {
        return false;
    }

    cv::Mat h64;
    homography.convertTo(h64, CV_64F);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            if (!std::isfinite(h64.at<double>(row, col))) {
                return false;
            }
        }
    }
    return true;
}

std::vector<unsigned char> classifyInliers(
    const cv::Mat& homography,
    const std::vector<Correspondence>& correspondences,
    double threshold,
    int& inlier_count,
    double& squared_error_sum) {
    std::vector<unsigned char> mask(correspondences.size(), 0);
    inlier_count = 0;
    squared_error_sum = 0.0;

    for (std::size_t i = 0; i < correspondences.size(); ++i) {
        const double error = reprojectionError(homography, correspondences[i]);
        if (error <= threshold) {
            mask[i] = 1;
            ++inlier_count;
            squared_error_sum += error * error;
        }
    }
    return mask;
}

cv::Mat estimateFromFour(const std::vector<Correspondence>& correspondences,
                         const std::array<int, 4>& indices) {
    std::array<cv::Point2f, 4> source{};
    std::array<cv::Point2f, 4> destination{};

    for (int i = 0; i < 4; ++i) {
        const Correspondence& correspondence =
            correspondences[static_cast<std::size_t>(
                indices[static_cast<std::size_t>(i)])];
        source[static_cast<std::size_t>(i)] = correspondence.source;
        destination[static_cast<std::size_t>(i)] = correspondence.destination;
    }

    return cv::getPerspectiveTransform(source.data(), destination.data());
}

RansacResult runHomographyRansac(
    const std::vector<Correspondence>& correspondences,
    int iterations,
    double reprojection_threshold,
    unsigned int seed) {
    RansacResult best;
    if (correspondences.size() < 4) {
        return best;
    }

    std::mt19937 rng(seed);
    int best_inlier_count = 0;
    double best_error_sum = std::numeric_limits<double>::infinity();

    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto indices =
            sampleFourUnique(static_cast<int>(correspondences.size()), rng);

        cv::Mat candidate;
        try {
            candidate = estimateFromFour(correspondences, indices);
        } catch (const cv::Exception&) {
            continue;
        }

        if (!finiteHomography(candidate)) {
            continue;
        }

        int inlier_count = 0;
        double squared_error_sum = 0.0;
        const auto mask =
            classifyInliers(candidate,
                            correspondences,
                            reprojection_threshold,
                            inlier_count,
                            squared_error_sum);

        const bool better_count = inlier_count > best_inlier_count;
        const bool same_count_lower_error =
            inlier_count == best_inlier_count &&
            squared_error_sum < best_error_sum;

        if (better_count || same_count_lower_error) {
            best_inlier_count = inlier_count;
            best_error_sum = squared_error_sum;
            best.homography = candidate.clone();
            best.inlier_mask = mask;
        }
    }

    if (best_inlier_count < 4 || best.homography.empty()) {
        return RansacResult{};
    }

    std::vector<cv::Point2f> inlier_source;
    std::vector<cv::Point2f> inlier_destination;
    for (std::size_t i = 0; i < correspondences.size(); ++i) {
        if (best.inlier_mask[i] != 0U) {
            inlier_source.push_back(correspondences[i].source);
            inlier_destination.push_back(correspondences[i].destination);
        }
    }

    const cv::Mat refined =
        cv::findHomography(inlier_source, inlier_destination, 0);
    if (finiteHomography(refined)) {
        best.homography = refined;
    }

    int final_inlier_count = 0;
    double final_squared_error_sum = 0.0;
    best.inlier_mask =
        classifyInliers(best.homography,
                        correspondences,
                        reprojection_threshold,
                        final_inlier_count,
                        final_squared_error_sum);

    best.inlier_count = final_inlier_count;
    best.inlier_rmse =
        final_inlier_count > 0
            ? std::sqrt(final_squared_error_sum / final_inlier_count)
            : std::numeric_limits<double>::infinity();

    return best;
}

cv::Mat drawMatches(const cv::Mat& image1,
                    const cv::Mat& image2,
                    const std::vector<cv::KeyPoint>& keypoints1,
                    const std::vector<cv::KeyPoint>& keypoints2,
                    const std::vector<Match>& matches,
                    const std::vector<unsigned char>* inlier_mask) {
    cv::Mat left;
    cv::Mat right;
    cv::cvtColor(image1, left, cv::COLOR_GRAY2BGR);
    cv::cvtColor(image2, right, cv::COLOR_GRAY2BGR);

    const int rows = std::max(left.rows, right.rows);
    cv::Mat output(rows,
                   left.cols + right.cols,
                   CV_8UC3,
                   cv::Scalar(0, 0, 0));

    left.copyTo(output(cv::Rect(0, 0, left.cols, left.rows)));
    right.copyTo(output(cv::Rect(left.cols, 0, right.cols, right.rows)));

    for (std::size_t index = 0; index < matches.size(); ++index) {
        if (inlier_mask != nullptr &&
            (index >= inlier_mask->size() || (*inlier_mask)[index] == 0U)) {
            continue;
        }

        const Match& match = matches[index];
        const cv::Point2f p1f =
            keypoints1[static_cast<std::size_t>(match.query_index)].pt;
        const cv::Point2f p2f =
            keypoints2[static_cast<std::size_t>(match.train_index)].pt;

        const cv::Point p1(cvRound(p1f.x), cvRound(p1f.y));
        const cv::Point p2(left.cols + cvRound(p2f.x), cvRound(p2f.y));

        cv::circle(output, p1, 3, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        cv::circle(output, p2, 3, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        cv::line(output, p1, p2, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    return output;
}

cv::Mat drawProjectedBounds(const cv::Mat& image2,
                            const cv::Mat& homography,
                            cv::Size source_size) {
    cv::Mat output;
    cv::cvtColor(image2, output, cv::COLOR_GRAY2BGR);

    const std::vector<cv::Point2f> source_corners = {
        {0.0f, 0.0f},
        {static_cast<float>(source_size.width - 1), 0.0f},
        {static_cast<float>(source_size.width - 1),
         static_cast<float>(source_size.height - 1)},
        {0.0f, static_cast<float>(source_size.height - 1)}
    };

    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(source_corners, projected, homography);

    for (std::size_t i = 0; i < projected.size(); ++i) {
        const cv::Point a(cvRound(projected[i].x), cvRound(projected[i].y));
        const cv::Point2f& next = projected[(i + 1) % projected.size()];
        const cv::Point b(cvRound(next.x), cvRound(next.y));
        cv::line(output, a, b, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
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

cv::Point2f applyKnownHomography(const cv::Mat& homography,
                                 const cv::Point2f& point) {
    cv::Point2f projected;
    if (!projectPoint(homography, point, projected)) {
        throw std::runtime_error("Known homography projection failed.");
    }
    return projected;
}

double pointDistance(const cv::Point2f& a, const cv::Point2f& b) {
    const double dx = static_cast<double>(a.x) - b.x;
    const double dy = static_cast<double>(a.y) - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

bool runSelfTest() {
    bool passed = true;
    auto check = [&passed](bool condition, const std::string& name) {
        if (!condition) {
            std::cerr << "Self-test failed: " << name << '\n';
            passed = false;
        }
    };

    const cv::Mat known_h =
        (cv::Mat_<double>(3, 3) <<
            1.08, 0.04, 18.0,
           -0.03, 1.05, 12.0,
            0.0004, -0.0002, 1.0);

    std::vector<Correspondence> correspondences;
    int index = 0;

    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 5; ++col) {
            const cv::Point2f source(
                20.0f + 30.0f * col,
                25.0f + 28.0f * row);

            cv::Point2f destination =
                applyKnownHomography(known_h, source);

            const float noise_x =
                static_cast<float>(((index * 7) % 5) - 2) * 0.08f;
            const float noise_y =
                static_cast<float>(((index * 11) % 5) - 2) * 0.08f;

            destination.x += noise_x;
            destination.y += noise_y;

            correspondences.push_back({source, destination, index});
            ++index;
        }
    }

    correspondences.push_back({
        cv::Point2f(15.0f, 15.0f),
        cv::Point2f(210.0f, 30.0f),
        index++});
    correspondences.push_back({
        cv::Point2f(150.0f, 20.0f),
        cv::Point2f(20.0f, 220.0f),
        index++});
    correspondences.push_back({
        cv::Point2f(25.0f, 150.0f),
        cv::Point2f(220.0f, 210.0f),
        index++});
    correspondences.push_back({
        cv::Point2f(160.0f, 140.0f),
        cv::Point2f(40.0f, 40.0f),
        index++});

    const RansacResult result =
        runHomographyRansac(correspondences, 2000, 1.0, 12345);

    check(!result.homography.empty(),
          "RANSAC recovers a homography");
    check(result.inlier_count >= 19,
          "RANSAC recovers almost all synthetic inliers");
    check(result.inlier_count <= 20,
          "RANSAC rejects synthetic outliers");
    check(result.inlier_rmse < 0.5,
          "RANSAC inlier reprojection RMSE is low");

    if (!result.homography.empty()) {
        const std::vector<cv::Point2f> probes = {
            {0.0f, 0.0f},
            {180.0f, 0.0f},
            {180.0f, 160.0f},
            {0.0f, 160.0f},
            {90.0f, 80.0f}
        };

        for (const auto& probe : probes) {
            const cv::Point2f expected =
                applyKnownHomography(known_h, probe);
            const cv::Point2f actual =
                applyKnownHomography(result.homography, probe);

            check(pointDistance(expected, actual) < 1.0,
                  "recovered homography projects probe near ground truth");
        }
    }

    const std::vector<Correspondence> too_few(
        correspondences.begin(),
        correspondences.begin() + 3);

    const RansacResult insufficient =
        runHomographyRansac(too_few, 100, 2.0, 1);

    check(insufficient.homography.empty(),
          "fewer than four correspondences cannot estimate homography");

    const cv::Mat query =
        (cv::Mat_<float>(2, 3) <<
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f);

    const cv::Mat train =
        (cv::Mat_<float>(4, 3) <<
            0.99f, 0.01f, 0.0f,
            0.0f, 0.98f, 0.02f,
            0.70f, 0.70f, 0.0f,
            0.0f, 0.70f, 0.70f);

    const auto ratio_matches = ratioMatch(query, train, 0.75);
    check(ratio_matches.size() == 2,
          "ratio matcher accepts two distinctive synthetic neighbors");

    if (passed) {
        std::cout << "Feature matching/RANSAC self-test passed.\n";
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

    auto sift = cv::SIFT::create(options.max_features);

    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    cv::Mat descriptors1;
    cv::Mat descriptors2;

    sift->detectAndCompute(
        image1, cv::noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(
        image2, cv::noArray(), keypoints2, descriptors2);

    const auto ratio_matches =
        ratioMatch(descriptors1, descriptors2, options.ratio_threshold);

    const auto correspondences =
        toCorrespondences(keypoints1, keypoints2, ratio_matches);

    if (correspondences.size() < 4) {
        std::cerr
            << "Error: fewer than four ratio-test matches; "
            << "cannot estimate a homography.\n";
        return 1;
    }

    const RansacResult ransac =
        runHomographyRansac(correspondences,
                            options.ransac_iterations,
                            options.reprojection_threshold,
                            options.seed);

    if (ransac.homography.empty() || ransac.inlier_count < 4) {
        std::cerr
            << "Error: RANSAC could not find a valid homography consensus.\n";
        return 1;
    }

    std::vector<unsigned char> match_inlier_mask(
        ratio_matches.size(), 0);

    for (std::size_t i = 0; i < correspondences.size(); ++i) {
        if (i < ransac.inlier_mask.size() &&
            ransac.inlier_mask[i] != 0U) {
            const int match_index = correspondences[i].match_index;
            if (match_index >= 0 &&
                match_index < static_cast<int>(match_inlier_mask.size())) {
                match_inlier_mask[
                    static_cast<std::size_t>(match_index)] = 1;
            }
        }
    }

    const fs::path output_dir(options.output_dir);
    std::error_code error;
    fs::create_directories(output_dir, error);

    if (error) {
        std::cerr
            << "Error: could not create output directory "
            << output_dir << ": " << error.message() << '\n';
        return 1;
    }

    bool ok = true;
    ok &= writeImage(
        output_dir / "01_ratio_test_matches.png",
        drawMatches(image1,
                    image2,
                    keypoints1,
                    keypoints2,
                    ratio_matches,
                    nullptr));

    ok &= writeImage(
        output_dir / "02_ransac_inliers.png",
        drawMatches(image1,
                    image2,
                    keypoints1,
                    keypoints2,
                    ratio_matches,
                    &match_inlier_mask));

    ok &= writeImage(
        output_dir / "03_projected_image_bounds.png",
        drawProjectedBounds(image2,
                            ransac.homography,
                            image1.size()));

    if (!ok) {
        return 1;
    }

    const int outliers =
        static_cast<int>(ratio_matches.size()) -
        ransac.inlier_count;

    const double inlier_ratio =
        ratio_matches.empty()
            ? 0.0
            : static_cast<double>(ransac.inlier_count) /
              ratio_matches.size();

    std::cout
        << "Image 1 SIFT keypoints: " << keypoints1.size() << '\n'
        << "Image 2 SIFT keypoints: " << keypoints2.size() << '\n'
        << "Ratio-test matches: " << ratio_matches.size() << '\n'
        << "RANSAC inliers: " << ransac.inlier_count << '\n'
        << "RANSAC outliers: " << outliers << '\n'
        << "Inlier ratio: " << inlier_ratio << '\n'
        << "Inlier reprojection RMSE: "
        << ransac.inlier_rmse << " px\n"
        << "RANSAC iterations: " << options.ransac_iterations << '\n'
        << "Reprojection threshold: "
        << options.reprojection_threshold << " px\n"
        << "Seed: " << options.seed << '\n'
        << "Outputs written to: " << output_dir << '\n';

    return 0;
}
