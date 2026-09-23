#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kSpatialCells = 4;
constexpr int kOrientationBins = 8;
constexpr int kDescriptorLength =
    kSpatialCells * kSpatialCells * kOrientationBins;

struct Options {
    std::string image1_path;
    std::string image2_path;
    std::string output_dir;
    int max_features = 400;
    double ratio_threshold = 0.75;
    double descriptor_clip = 0.2;
    bool self_test = false;
    bool help = false;
};

struct DescriptorRecord {
    cv::KeyPoint keypoint;
    std::vector<float> descriptor;
};

struct Match {
    int query_index = -1;
    int train_index = -1;
    float best_distance = 0.0f;
    float second_distance = 0.0f;
};

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program
        << " --image1 <path> --image2 <path> --output-dir <dir> [options]\n"
        << "  " << program << " --self-test\n\n"
        << "Options:\n"
        << "  --image1 <path>          First grayscale image\n"
        << "  --image2 <path>          Second grayscale image\n"
        << "  --output-dir <path>      Output directory\n"
        << "  --max-features <N>       Max oriented keypoints per image (default: 400)\n"
        << "  --ratio <0..1>           Lowe ratio threshold (default: 0.75)\n"
        << "  --clip <value>           Descriptor component clip (default: 0.2)\n"
        << "  --self-test              Run deterministic tests\n"
        << "  --help                   Show this help\n";
}

bool parseArguments(int argc, char** argv, Options& options) {
    auto requireValue = [argc, argv](int& index, const std::string& option) {
        if (index + 1 >= argc) {
            throw std::runtime_error("Missing value for " + option);
        }
        ++index;
        return std::string(argv[index]);
    };

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--image1") {
                options.image1_path = requireValue(i, arg);
            } else if (arg == "--image2") {
                options.image2_path = requireValue(i, arg);
            } else if (arg == "--output-dir") {
                options.output_dir = requireValue(i, arg);
            } else if (arg == "--max-features") {
                options.max_features = std::stoi(requireValue(i, arg));
            } else if (arg == "--ratio") {
                options.ratio_threshold = std::stod(requireValue(i, arg));
            } else if (arg == "--clip") {
                options.descriptor_clip = std::stod(requireValue(i, arg));
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
    if (options.image1_path.empty() ||
        options.image2_path.empty() ||
        options.output_dir.empty()) {
        std::cerr << "Error: --image1, --image2 and --output-dir are required.\n";
        return false;
    }
    if (options.max_features <= 0) {
        std::cerr << "Error: --max-features must be > 0.\n";
        return false;
    }
    if (options.ratio_threshold <= 0.0 || options.ratio_threshold >= 1.0) {
        std::cerr << "Error: --ratio must be in (0,1).\n";
        return false;
    }
    if (options.descriptor_clip <= 0.0) {
        std::cerr << "Error: --clip must be > 0.\n";
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

float l2Norm(const std::vector<float>& values) {
    double sum = 0.0;
    for (const float value : values) {
        sum += static_cast<double>(value) * value;
    }
    return static_cast<float>(std::sqrt(sum));
}

void normalizeDescriptor(std::vector<float>& descriptor,
                         double clip_value) {
    float norm = l2Norm(descriptor);
    if (norm <= 1e-12f) {
        std::fill(descriptor.begin(), descriptor.end(), 0.0f);
        return;
    }

    for (float& value : descriptor) {
        value /= norm;
        value = std::min(value, static_cast<float>(clip_value));
    }

    norm = l2Norm(descriptor);
    if (norm <= 1e-12f) {
        return;
    }
    for (float& value : descriptor) {
        value /= norm;
    }
}

void addTrilinearVote(std::vector<float>& descriptor,
                      float cell_x,
                      float cell_y,
                      float orientation_bin,
                      float vote) {
    const int x0 = static_cast<int>(std::floor(cell_x));
    const int y0 = static_cast<int>(std::floor(cell_y));
    const int o0 = static_cast<int>(std::floor(orientation_bin));

    const float fx = cell_x - x0;
    const float fy = cell_y - y0;
    const float fo = orientation_bin - std::floor(orientation_bin);

    for (int dx = 0; dx <= 1; ++dx) {
        const int cx = x0 + dx;
        if (cx < 0 || cx >= kSpatialCells) {
            continue;
        }
        const float wx = dx == 0 ? 1.0f - fx : fx;

        for (int dy = 0; dy <= 1; ++dy) {
            const int cy = y0 + dy;
            if (cy < 0 || cy >= kSpatialCells) {
                continue;
            }
            const float wy = dy == 0 ? 1.0f - fy : fy;

            for (int db = 0; db <= 1; ++db) {
                int bin = o0 + db;
                bin %= kOrientationBins;
                if (bin < 0) {
                    bin += kOrientationBins;
                }
                const float wo = db == 0 ? 1.0f - fo : fo;

                const int index =
                    (cy * kSpatialCells + cx) * kOrientationBins + bin;
                descriptor[static_cast<std::size_t>(index)] +=
                    vote * wx * wy * wo;
            }
        }
    }
}

std::vector<float> computeDescriptor(const cv::Mat& image,
                                     const cv::KeyPoint& keypoint,
                                     double clip_value) {
    CV_Assert(image.type() == CV_32F);

    std::vector<float> descriptor(
        static_cast<std::size_t>(kDescriptorLength), 0.0f);

    const float sigma =
        std::max(1.0f, keypoint.size * 0.5f);
    const float hist_width = 3.0f * sigma;
    const float descriptor_window =
        static_cast<float>(kSpatialCells) * hist_width;
    const int radius = std::max(
        1,
        cvRound(0.5 * std::sqrt(2.0) *
                descriptor_window));

    const double angle_rad =
        static_cast<double>(keypoint.angle) * CV_PI / 180.0;
    const float cos_angle = static_cast<float>(std::cos(angle_rad));
    const float sin_angle = static_cast<float>(std::sin(angle_rad));

    const float weight_sigma =
        0.5f * descriptor_window;
    const float two_weight_sigma_sq =
        2.0f * weight_sigma * weight_sigma;

    for (int dy = -radius; dy <= radius; ++dy) {
        const int row = cvRound(keypoint.pt.y) + dy;
        if (row <= 0 || row + 1 >= image.rows) {
            continue;
        }

        for (int dx = -radius; dx <= radius; ++dx) {
            const int col = cvRound(keypoint.pt.x) + dx;
            if (col <= 0 || col + 1 >= image.cols) {
                continue;
            }

            const float rotated_x =
                ( cos_angle * dx + sin_angle * dy) / hist_width;
            const float rotated_y =
                (-sin_angle * dx + cos_angle * dy) / hist_width;

            const float cell_x =
                rotated_x + static_cast<float>(kSpatialCells) / 2.0f - 0.5f;
            const float cell_y =
                rotated_y + static_cast<float>(kSpatialCells) / 2.0f - 0.5f;

            if (cell_x <= -1.0f || cell_x >= kSpatialCells ||
                cell_y <= -1.0f || cell_y >= kSpatialCells) {
                continue;
            }

            const float gx =
                image.at<float>(row, col + 1) -
                image.at<float>(row, col - 1);
            const float gy =
                image.at<float>(row + 1, col) -
                image.at<float>(row - 1, col);
            const float magnitude = std::sqrt(gx * gx + gy * gy);
            if (magnitude <= 0.0f) {
                continue;
            }

            double gradient_angle =
                std::atan2(static_cast<double>(gy),
                           static_cast<double>(gx)) *
                180.0 / CV_PI;
            if (gradient_angle < 0.0) {
                gradient_angle += 360.0;
            }

            double relative_angle =
                gradient_angle - keypoint.angle;
            while (relative_angle < 0.0) {
                relative_angle += 360.0;
            }
            while (relative_angle >= 360.0) {
                relative_angle -= 360.0;
            }

            const float orientation_bin =
                static_cast<float>(
                    relative_angle * kOrientationBins / 360.0);

            const float spatial_weight =
                std::exp(
                    -static_cast<float>(dx * dx + dy * dy) /
                    two_weight_sigma_sq);

            addTrilinearVote(
                descriptor,
                cell_x,
                cell_y,
                orientation_bin,
                spatial_weight * magnitude);
        }
    }

    normalizeDescriptor(descriptor, clip_value);
    return descriptor;
}

float descriptorDistance(const std::vector<float>& a,
                         const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Descriptor sizes do not match.");
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double delta =
            static_cast<double>(a[i]) - b[i];
        sum += delta * delta;
    }
    return static_cast<float>(std::sqrt(sum));
}

std::vector<DescriptorRecord> buildDescriptors(
    const cv::Mat& image,
    const std::vector<cv::KeyPoint>& keypoints,
    double clip_value) {
    std::vector<DescriptorRecord> records;
    records.reserve(keypoints.size());

    for (const auto& keypoint : keypoints) {
        if (keypoint.angle < 0.0f) {
            continue;
        }
        std::vector<float> descriptor =
            computeDescriptor(image, keypoint, clip_value);
        if (l2Norm(descriptor) > 0.0f) {
            records.push_back({keypoint, std::move(descriptor)});
        }
    }
    return records;
}

std::vector<Match> matchDescriptors(
    const std::vector<DescriptorRecord>& query,
    const std::vector<DescriptorRecord>& train,
    double ratio_threshold) {
    std::vector<Match> matches;
    if (train.size() < 2) {
        return matches;
    }

    for (std::size_t query_index = 0;
         query_index < query.size();
         ++query_index) {
        float best = std::numeric_limits<float>::infinity();
        float second = std::numeric_limits<float>::infinity();
        int best_index = -1;

        for (std::size_t train_index = 0;
             train_index < train.size();
             ++train_index) {
            const float distance =
                descriptorDistance(
                    query[query_index].descriptor,
                    train[train_index].descriptor);

            if (distance < best) {
                second = best;
                best = distance;
                best_index = static_cast<int>(train_index);
            } else if (distance < second) {
                second = distance;
            }
        }

        if (best_index >= 0 &&
            std::isfinite(second) &&
            best < static_cast<float>(ratio_threshold) * second) {
            matches.push_back({
                static_cast<int>(query_index),
                best_index,
                best,
                second
            });
        }
    }
    return matches;
}

std::vector<cv::KeyPoint> detectOrientedKeypoints(
    const cv::Mat& image_u8,
    int max_features) {
    auto sift = cv::SIFT::create(max_features);
    std::vector<cv::KeyPoint> keypoints;
    sift->detect(image_u8, keypoints);

    std::sort(
        keypoints.begin(),
        keypoints.end(),
        [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
            return a.response > b.response;
        });

    if (static_cast<int>(keypoints.size()) > max_features) {
        keypoints.resize(static_cast<std::size_t>(max_features));
    }
    return keypoints;
}

cv::Mat drawManualMatches(
    const cv::Mat& image1,
    const cv::Mat& image2,
    const std::vector<DescriptorRecord>& descriptors1,
    const std::vector<DescriptorRecord>& descriptors2,
    const std::vector<Match>& matches) {
    cv::Mat left;
    cv::Mat right;
    cv::cvtColor(image1, left, cv::COLOR_GRAY2BGR);
    cv::cvtColor(image2, right, cv::COLOR_GRAY2BGR);

    const int output_rows = std::max(left.rows, right.rows);
    const int output_cols = left.cols + right.cols;
    cv::Mat output(
        output_rows,
        output_cols,
        CV_8UC3,
        cv::Scalar(0, 0, 0));

    left.copyTo(output(cv::Rect(0, 0, left.cols, left.rows)));
    right.copyTo(output(cv::Rect(
        left.cols, 0, right.cols, right.rows)));

    for (const auto& match : matches) {
        const cv::Point p1 = descriptors1[
            static_cast<std::size_t>(match.query_index)].keypoint.pt;
        cv::Point p2 = descriptors2[
            static_cast<std::size_t>(match.train_index)].keypoint.pt;
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

std::vector<float> unitDescriptor(int index) {
    std::vector<float> descriptor(
        static_cast<std::size_t>(kDescriptorLength), 0.0f);
    descriptor[static_cast<std::size_t>(index)] = 1.0f;
    return descriptor;
}

bool runSelfTest() {
    bool passed = true;
    auto check = [&passed](bool condition, const std::string& name) {
        if (!condition) {
            std::cerr << "Self-test failed: " << name << '\n';
            passed = false;
        }
    };

    cv::Mat ramp_x(81, 81, CV_32F);
    cv::Mat ramp_x_scaled(81, 81, CV_32F);
    for (int row = 0; row < ramp_x.rows; ++row) {
        for (int col = 0; col < ramp_x.cols; ++col) {
            const float value =
                static_cast<float>(20.0 + 0.5 * col + 0.1 * row);
            ramp_x.at<float>(row, col) = value;
            ramp_x_scaled.at<float>(row, col) = value * 2.5f;
        }
    }

    cv::KeyPoint keypoint(
        cv::Point2f(40.0f, 40.0f),
        8.0f,
        0.0f);

    const auto descriptor =
        computeDescriptor(ramp_x, keypoint, 0.2);
    check(descriptor.size() ==
              static_cast<std::size_t>(kDescriptorLength),
          "SIFT descriptor has 128 elements");
    check(std::abs(l2Norm(descriptor) - 1.0f) <= 1e-5f,
          "nonzero descriptor is L2 normalized");

    const auto scaled_descriptor =
        computeDescriptor(ramp_x_scaled, keypoint, 0.2);
    check(descriptorDistance(descriptor, scaled_descriptor) <= 1e-4f,
          "descriptor normalization suppresses multiplicative contrast scaling");

    cv::Mat vertical(81, 81, CV_32F);
    for (int row = 0; row < vertical.rows; ++row) {
        for (int col = 0; col < vertical.cols; ++col) {
            vertical.at<float>(row, col) =
                static_cast<float>(20.0 + 0.5 * row + 0.1 * col);
        }
    }

    cv::KeyPoint vertical_keypoint(
        cv::Point2f(40.0f, 40.0f),
        8.0f,
        90.0f);
    const auto vertical_descriptor =
        computeDescriptor(vertical, vertical_keypoint, 0.2);
    check(descriptorDistance(
              descriptor,
              vertical_descriptor) < 0.20f,
          "orientation alignment makes rotated gradient structure similar");

    std::vector<float> clipping_test(
        static_cast<std::size_t>(kDescriptorLength), 0.0f);
    clipping_test[0] = 100.0f;
    clipping_test[1] = 1.0f;
    clipping_test[2] = 1.0f;
    normalizeDescriptor(clipping_test, 0.2);
    check(std::abs(l2Norm(clipping_test) - 1.0f) <= 1e-5f,
          "clipped descriptor is renormalized");
    check(clipping_test[0] < 1.0f,
          "dominant component is limited by clipping before renormalization");

    check(descriptorDistance(descriptor, descriptor) <= 1e-7f,
          "identical descriptors have zero distance");

    std::vector<DescriptorRecord> query = {
        {cv::KeyPoint(), unitDescriptor(0)},
        {cv::KeyPoint(), unitDescriptor(10)}
    };

    auto close_to_zero = unitDescriptor(0);
    close_to_zero[0] = 0.98f;
    close_to_zero[1] = 0.20f;
    normalizeDescriptor(close_to_zero, 1.0);

    auto ambiguous_a = unitDescriptor(10);
    ambiguous_a[11] = 0.20f;
    normalizeDescriptor(ambiguous_a, 1.0);

    auto ambiguous_b = unitDescriptor(10);
    ambiguous_b[12] = 0.22f;
    normalizeDescriptor(ambiguous_b, 1.0);

    std::vector<DescriptorRecord> train = {
        {cv::KeyPoint(), close_to_zero},
        {cv::KeyPoint(), unitDescriptor(20)},
        {cv::KeyPoint(), ambiguous_a},
        {cv::KeyPoint(), ambiguous_b}
    };

    const auto matches =
        matchDescriptors(query, train, 0.75);
    check(matches.size() == 1,
          "ratio test accepts distinctive match and rejects ambiguous match");
    if (!matches.empty()) {
        check(matches.front().query_index == 0,
              "distinctive query survives ratio test");
        check(matches.front().train_index == 0,
              "distinctive query selects nearest descriptor");
        check(matches.front().best_distance <
                  0.75f * matches.front().second_distance,
              "accepted match satisfies Lowe ratio inequality");
    }

    cv::Mat flat(81, 81, CV_32F, cv::Scalar(100.0f));
    const auto flat_descriptor =
        computeDescriptor(flat, keypoint, 0.2);
    check(l2Norm(flat_descriptor) <= 1e-7f,
          "flat patch produces zero descriptor");

    if (passed) {
        std::cout << "SIFT descriptor/matching self-test passed.\n";
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

    const cv::Mat image1_u8 =
        cv::imread(options.image1_path, cv::IMREAD_GRAYSCALE);
    const cv::Mat image2_u8 =
        cv::imread(options.image2_path, cv::IMREAD_GRAYSCALE);

    if (image1_u8.empty() || image2_u8.empty()) {
        std::cerr << "Error: could not read both input images.\n";
        return 1;
    }

    const cv::Mat image1 = toFloatGray(image1_u8);
    const cv::Mat image2 = toFloatGray(image2_u8);

    const auto keypoints1 =
        detectOrientedKeypoints(image1_u8, options.max_features);
    const auto keypoints2 =
        detectOrientedKeypoints(image2_u8, options.max_features);

    const auto descriptors1 =
        buildDescriptors(
            image1,
            keypoints1,
            options.descriptor_clip);
    const auto descriptors2 =
        buildDescriptors(
            image2,
            keypoints2,
            options.descriptor_clip);

    const auto matches =
        matchDescriptors(
            descriptors1,
            descriptors2,
            options.ratio_threshold);

    const cv::Mat visualization =
        drawManualMatches(
            image1_u8,
            image2_u8,
            descriptors1,
            descriptors2,
            matches);

    const fs::path output_dir(options.output_dir);
    std::error_code error;
    fs::create_directories(output_dir, error);
    if (error) {
        std::cerr << "Error: could not create output directory "
                  << output_dir << ": " << error.message() << '\n';
        return 1;
    }

    if (!writeImage(
            output_dir / "sift_manual_descriptor_matches.png",
            visualization)) {
        return 1;
    }

    std::cout
        << "Image 1 oriented keypoints: " << keypoints1.size() << '\n'
        << "Image 2 oriented keypoints: " << keypoints2.size() << '\n'
        << "Image 1 manual descriptors: " << descriptors1.size() << '\n'
        << "Image 2 manual descriptors: " << descriptors2.size() << '\n'
        << "Descriptor length: " << kDescriptorLength << '\n'
        << "Ratio threshold: " << options.ratio_threshold << '\n'
        << "Accepted matches: " << matches.size() << '\n'
        << "Output: "
        << (output_dir / "sift_manual_descriptor_matches.png") << '\n';

    return 0;
}
