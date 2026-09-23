#include <opencv2/opencv.hpp>

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
    int octaves = 4;
    int intervals = 3;
    double sigma0 = 1.6;
    double candidate_threshold = 1.0;
    double contrast_threshold = 2.0;
    double edge_ratio = 10.0;
    int orientation_bins = 36;
    double orientation_peak_ratio = 0.8;
    bool self_test = false;
    bool help = false;
};

struct ScaleExtremum {
    int octave = 0;
    int dog_layer = 0;
    cv::Point point;
    float response = 0.0f;
};

struct RefinedKeypoint {
    int octave = 0;
    int dog_layer = 0;
    cv::Point2f point;
    float layer_offset = 0.0f;
    float contrast = 0.0f;
    float sigma_octave = 0.0f;
};

struct OrientedKeypoint {
    RefinedKeypoint keypoint;
    float orientation_deg = 0.0f;
};

enum class RefineStatus {
    Accepted,
    OutOfBounds,
    SingularHessian,
    DidNotConverge,
    LowContrast,
    EdgeLike
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
        << "  --input <path>                    Input image\n"
        << "  --output-dir <path>               Output directory\n"
        << "  --octaves <N>                     Number of octaves (default: 4)\n"
        << "  --intervals <N>                   Scale intervals per octave (default: 3)\n"
        << "  --sigma <value>                   Base sigma (default: 1.6)\n"
        << "  --candidate-threshold <value>     Raw DoG pre-threshold (default: 1.0)\n"
        << "  --contrast-threshold <value>      Localized DoG threshold (default: 2.0)\n"
        << "  --edge-ratio <value>              Max principal-curvature ratio (default: 10)\n"
        << "  --orientation-bins <N>            Orientation histogram bins (default: 36)\n"
        << "  --orientation-peak-ratio <0..1>   Secondary peak ratio (default: 0.8)\n"
        << "  --self-test                       Run deterministic tests\n"
        << "  --help                            Show this help\n";
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
            } else if (arg == "--candidate-threshold") {
                options.candidate_threshold = std::stod(requireValue(i, arg));
            } else if (arg == "--contrast-threshold") {
                options.contrast_threshold = std::stod(requireValue(i, arg));
            } else if (arg == "--edge-ratio") {
                options.edge_ratio = std::stod(requireValue(i, arg));
            } else if (arg == "--orientation-bins") {
                options.orientation_bins = std::stoi(requireValue(i, arg));
            } else if (arg == "--orientation-peak-ratio") {
                options.orientation_peak_ratio = std::stod(requireValue(i, arg));
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
    if (options.octaves <= 0 || options.intervals <= 0 || options.sigma0 <= 0.0) {
        std::cerr << "Error: octaves/intervals must be > 0 and sigma must be > 0.\n";
        return false;
    }
    if (options.candidate_threshold < 0.0 ||
        options.contrast_threshold < 0.0 ||
        options.edge_ratio <= 0.0) {
        std::cerr << "Error: thresholds must be non-negative and edge-ratio > 0.\n";
        return false;
    }
    if (options.orientation_bins < 4 ||
        options.orientation_peak_ratio <= 0.0 ||
        options.orientation_peak_ratio > 1.0) {
        std::cerr << "Error: orientation bins >= 4 and peak ratio in (0,1].\n";
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

std::vector<double> incrementalSigmas(int intervals, double sigma0) {
    const int layers = intervals + 3;
    const double k = scaleStep(intervals);

    std::vector<double> absolute(static_cast<std::size_t>(layers));
    for (int layer = 0; layer < layers; ++layer) {
        absolute[static_cast<std::size_t>(layer)] =
            sigma0 * std::pow(k, layer);
    }

    std::vector<double> incremental(absolute.size(), 0.0);
    incremental[0] = sigma0;
    for (std::size_t layer = 1; layer < absolute.size(); ++layer) {
        incremental[layer] =
            std::sqrt(std::max(
                0.0,
                absolute[layer] * absolute[layer] -
                absolute[layer - 1] * absolute[layer - 1]));
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
    const auto incremental = incrementalSigmas(intervals, sigma0);
    Pyramid pyramid;
    cv::Mat octave_base = input.clone();

    for (int octave_index = 0; octave_index < octaves; ++octave_index) {
        if (octave_base.rows < 8 || octave_base.cols < 8) {
            break;
        }

        Octave octave;
        cv::Mat first;
        if (octave_index == 0) {
            cv::GaussianBlur(octave_base,
                             first,
                             cv::Size(0, 0),
                             incremental[0],
                             incremental[0],
                             cv::BORDER_REFLECT_101);
        } else {
            first = octave_base.clone();
        }
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
        for (std::size_t layer = 0; layer + 1 < octave.size(); ++layer) {
            cv::Mat difference;
            cv::subtract(octave[layer + 1],
                         octave[layer],
                         difference,
                         cv::noArray(),
                         CV_32F);
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
    const float center =
        dog_octave.at(static_cast<std::size_t>(layer)).at<float>(row, col);

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
    double candidate_threshold) {
    std::vector<ScaleExtremum> extrema;

    for (std::size_t octave_index = 0; octave_index < dog.size(); ++octave_index) {
        const Octave& octave = dog[octave_index];
        if (octave.size() < 3) {
            continue;
        }

        for (std::size_t layer = 1; layer + 1 < octave.size(); ++layer) {
            const cv::Mat& image = octave[layer];
            for (int row = 1; row < image.rows - 1; ++row) {
                for (int col = 1; col < image.cols - 1; ++col) {
                    const float response = image.at<float>(row, col);
                    if (std::abs(static_cast<double>(response)) <
                        candidate_threshold) {
                        continue;
                    }
                    if (isScaleSpaceExtremum(
                            octave,
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

bool inRefinementBounds(const Octave& octave,
                        int layer,
                        int row,
                        int col) {
    if (layer <= 0 || layer + 1 >= static_cast<int>(octave.size())) {
        return false;
    }
    const cv::Mat& image = octave[static_cast<std::size_t>(layer)];
    return row > 0 && row + 1 < image.rows &&
           col > 0 && col + 1 < image.cols;
}

bool derivativeModel(const Octave& octave,
                     int layer,
                     int row,
                     int col,
                     cv::Vec3d& gradient,
                     cv::Matx33d& hessian) {
    if (!inRefinementBounds(octave, layer, row, col)) {
        return false;
    }

    const cv::Mat& current = octave[static_cast<std::size_t>(layer)];
    const cv::Mat& previous = octave[static_cast<std::size_t>(layer - 1)];
    const cv::Mat& next = octave[static_cast<std::size_t>(layer + 1)];

    const double center = current.at<float>(row, col);

    const double dx =
        0.5 * (current.at<float>(row, col + 1) -
               current.at<float>(row, col - 1));
    const double dy =
        0.5 * (current.at<float>(row + 1, col) -
               current.at<float>(row - 1, col));
    const double ds =
        0.5 * (next.at<float>(row, col) -
               previous.at<float>(row, col));

    const double dxx =
        current.at<float>(row, col + 1) +
        current.at<float>(row, col - 1) -
        2.0 * center;
    const double dyy =
        current.at<float>(row + 1, col) +
        current.at<float>(row - 1, col) -
        2.0 * center;
    const double dss =
        next.at<float>(row, col) +
        previous.at<float>(row, col) -
        2.0 * center;

    const double dxy =
        0.25 * (
            current.at<float>(row + 1, col + 1) -
            current.at<float>(row + 1, col - 1) -
            current.at<float>(row - 1, col + 1) +
            current.at<float>(row - 1, col - 1));

    const double dxs =
        0.25 * (
            next.at<float>(row, col + 1) -
            next.at<float>(row, col - 1) -
            previous.at<float>(row, col + 1) +
            previous.at<float>(row, col - 1));

    const double dys =
        0.25 * (
            next.at<float>(row + 1, col) -
            next.at<float>(row - 1, col) -
            previous.at<float>(row + 1, col) +
            previous.at<float>(row - 1, col));

    gradient = cv::Vec3d(dx, dy, ds);
    hessian = cv::Matx33d(
        dxx, dxy, dxs,
        dxy, dyy, dys,
        dxs, dys, dss);
    return true;
}

bool solveOffset(const cv::Vec3d& gradient,
                 const cv::Matx33d& hessian,
                 cv::Vec3d& offset) {
    const double determinant = cv::determinant(cv::Mat(hessian));
    if (std::abs(determinant) < 1e-10) {
        return false;
    }

    const cv::Vec3d rhs(-gradient[0], -gradient[1], -gradient[2]);
    offset = hessian.inv(cv::DECOMP_LU) * rhs;
    return std::isfinite(offset[0]) &&
           std::isfinite(offset[1]) &&
           std::isfinite(offset[2]);
}

bool passesEdgeResponse(const cv::Mat& dog_layer,
                        int row,
                        int col,
                        double edge_ratio) {
    if (row <= 0 || row + 1 >= dog_layer.rows ||
        col <= 0 || col + 1 >= dog_layer.cols) {
        return false;
    }

    const double center = dog_layer.at<float>(row, col);
    const double dxx =
        dog_layer.at<float>(row, col + 1) +
        dog_layer.at<float>(row, col - 1) -
        2.0 * center;
    const double dyy =
        dog_layer.at<float>(row + 1, col) +
        dog_layer.at<float>(row - 1, col) -
        2.0 * center;
    const double dxy =
        0.25 * (
            dog_layer.at<float>(row + 1, col + 1) -
            dog_layer.at<float>(row + 1, col - 1) -
            dog_layer.at<float>(row - 1, col + 1) +
            dog_layer.at<float>(row - 1, col - 1));

    const double determinant = dxx * dyy - dxy * dxy;
    if (determinant <= 0.0) {
        return false;
    }

    const double trace = dxx + dyy;
    const double curvature_measure = (trace * trace) / determinant;
    const double allowed =
        ((edge_ratio + 1.0) * (edge_ratio + 1.0)) / edge_ratio;
    return curvature_measure < allowed;
}

float sigmaAtLayer(double sigma0,
                   int intervals,
                   int dog_layer,
                   float layer_offset) {
    return static_cast<float>(
        sigma0 *
        std::pow(scaleStep(intervals),
                 static_cast<double>(dog_layer) + layer_offset));
}

RefineStatus refineCandidate(const Octave& dog_octave,
                             int octave_index,
                             const ScaleExtremum& candidate,
                             int intervals,
                             double sigma0,
                             double contrast_threshold,
                             double edge_ratio,
                             RefinedKeypoint& refined) {
    int col = candidate.point.x;
    int row = candidate.point.y;
    int layer = candidate.dog_layer;

    cv::Vec3d offset(0.0, 0.0, 0.0);
    cv::Vec3d gradient;
    cv::Matx33d hessian;
    bool converged = false;

    constexpr int kMaxIterations = 5;

    for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
        if (!derivativeModel(dog_octave,
                             layer,
                             row,
                             col,
                             gradient,
                             hessian)) {
            return RefineStatus::OutOfBounds;
        }

        if (!solveOffset(gradient, hessian, offset)) {
            return RefineStatus::SingularHessian;
        }

        if (std::abs(offset[0]) < 0.5 &&
            std::abs(offset[1]) < 0.5 &&
            std::abs(offset[2]) < 0.5) {
            converged = true;
            break;
        }

        col += cvRound(offset[0]);
        row += cvRound(offset[1]);
        layer += cvRound(offset[2]);
    }

    if (!converged) {
        return RefineStatus::DidNotConverge;
    }

    if (!derivativeModel(dog_octave,
                         layer,
                         row,
                         col,
                         gradient,
                         hessian) ||
        !solveOffset(gradient, hessian, offset)) {
        return RefineStatus::SingularHessian;
    }

    const double center =
        dog_octave[static_cast<std::size_t>(layer)].at<float>(row, col);
    const double interpolated_contrast =
        center + 0.5 * gradient.dot(offset);

    if (std::abs(interpolated_contrast) < contrast_threshold) {
        return RefineStatus::LowContrast;
    }

    if (!passesEdgeResponse(
            dog_octave[static_cast<std::size_t>(layer)],
            row,
            col,
            edge_ratio)) {
        return RefineStatus::EdgeLike;
    }

    refined.octave = octave_index;
    refined.dog_layer = layer;
    refined.point = cv::Point2f(
        static_cast<float>(col + offset[0]),
        static_cast<float>(row + offset[1]));
    refined.layer_offset = static_cast<float>(offset[2]);
    refined.contrast = static_cast<float>(interpolated_contrast);
    refined.sigma_octave =
        sigmaAtLayer(sigma0, intervals, layer, refined.layer_offset);

    return RefineStatus::Accepted;
}

std::vector<float> buildOrientationHistogram(const cv::Mat& gaussian,
                                             const RefinedKeypoint& keypoint,
                                             int bins) {
    std::vector<float> histogram(static_cast<std::size_t>(bins), 0.0f);

    const double window_sigma =
        1.5 * static_cast<double>(keypoint.sigma_octave);
    const int radius =
        std::max(1, cvRound(3.0 * window_sigma));
    const double two_sigma_sq =
        2.0 * window_sigma * window_sigma;

    const int center_x = cvRound(keypoint.point.x);
    const int center_y = cvRound(keypoint.point.y);

    for (int dy = -radius; dy <= radius; ++dy) {
        const int row = center_y + dy;
        if (row <= 0 || row + 1 >= gaussian.rows) {
            continue;
        }

        for (int dx = -radius; dx <= radius; ++dx) {
            const int col = center_x + dx;
            if (col <= 0 || col + 1 >= gaussian.cols) {
                continue;
            }

            const float gx =
                gaussian.at<float>(row, col + 1) -
                gaussian.at<float>(row, col - 1);
            const float gy =
                gaussian.at<float>(row + 1, col) -
                gaussian.at<float>(row - 1, col);
            const float magnitude = std::sqrt(gx * gx + gy * gy);
            if (magnitude <= 0.0f) {
                continue;
            }

            double angle =
                std::atan2(static_cast<double>(gy),
                           static_cast<double>(gx)) *
                180.0 / CV_PI;
            if (angle < 0.0) {
                angle += 360.0;
            }

            const double weight =
                std::exp(-(static_cast<double>(dx * dx + dy * dy)) /
                         two_sigma_sq);
            const double bin_position =
                angle * static_cast<double>(bins) / 360.0;
            const int lower =
                static_cast<int>(std::floor(bin_position)) % bins;
            const int upper = (lower + 1) % bins;
            const float fraction =
                static_cast<float>(bin_position - std::floor(bin_position));

            histogram[static_cast<std::size_t>(lower)] +=
                static_cast<float>(weight) * magnitude * (1.0f - fraction);
            histogram[static_cast<std::size_t>(upper)] +=
                static_cast<float>(weight) * magnitude * fraction;
        }
    }

    for (int pass = 0; pass < 6; ++pass) {
        std::vector<float> smoothed(histogram.size(), 0.0f);
        for (int bin = 0; bin < bins; ++bin) {
            const int previous = (bin - 1 + bins) % bins;
            const int next = (bin + 1) % bins;
            smoothed[static_cast<std::size_t>(bin)] =
                0.25f * histogram[static_cast<std::size_t>(previous)] +
                0.5f * histogram[static_cast<std::size_t>(bin)] +
                0.25f * histogram[static_cast<std::size_t>(next)];
        }
        histogram.swap(smoothed);
    }

    return histogram;
}

std::vector<float> orientationPeaks(const std::vector<float>& histogram,
                                    double peak_ratio) {
    std::vector<float> orientations;
    if (histogram.empty()) {
        return orientations;
    }

    const float maximum =
        *std::max_element(histogram.begin(), histogram.end());
    if (maximum <= 0.0f) {
        return orientations;
    }

    const int bins = static_cast<int>(histogram.size());
    const float threshold =
        static_cast<float>(peak_ratio) * maximum;

    for (int bin = 0; bin < bins; ++bin) {
        const int previous = (bin - 1 + bins) % bins;
        const int next = (bin + 1) % bins;

        const float left = histogram[static_cast<std::size_t>(previous)];
        const float center = histogram[static_cast<std::size_t>(bin)];
        const float right = histogram[static_cast<std::size_t>(next)];

        if (center < threshold ||
            center <= left ||
            center <= right) {
            continue;
        }

        const float denominator =
            left - 2.0f * center + right;
        float offset = 0.0f;
        if (std::abs(denominator) > 1e-12f) {
            offset = 0.5f * (left - right) / denominator;
            offset = std::clamp(offset, -0.5f, 0.5f);
        }

        double angle =
            360.0 *
            (static_cast<double>(bin) + offset) /
            static_cast<double>(bins);

        while (angle < 0.0) {
            angle += 360.0;
        }
        while (angle >= 360.0) {
            angle -= 360.0;
        }

        orientations.push_back(static_cast<float>(angle));
    }

    return orientations;
}

std::vector<OrientedKeypoint> assignOrientations(
    const Pyramid& gaussian,
    const std::vector<RefinedKeypoint>& refined,
    int bins,
    double peak_ratio) {
    std::vector<OrientedKeypoint> oriented;

    for (const auto& keypoint : refined) {
        if (keypoint.octave < 0 ||
            keypoint.octave >= static_cast<int>(gaussian.size())) {
            continue;
        }

        const Octave& octave =
            gaussian[static_cast<std::size_t>(keypoint.octave)];
        const int gaussian_layer = std::clamp(
            cvRound(static_cast<double>(keypoint.dog_layer) +
                    keypoint.layer_offset),
            0,
            static_cast<int>(octave.size()) - 1);

        const auto histogram =
            buildOrientationHistogram(
                octave[static_cast<std::size_t>(gaussian_layer)],
                keypoint,
                bins);
        const auto orientations =
            orientationPeaks(histogram, peak_ratio);

        for (const float orientation : orientations) {
            oriented.push_back({keypoint, orientation});
        }
    }

    return oriented;
}

cv::Mat toByteImage(const cv::Mat& input) {
    cv::Mat output;
    input.convertTo(output, CV_8U);
    return output;
}

cv::Point originalPoint(const RefinedKeypoint& keypoint) {
    const int scale = 1 << keypoint.octave;
    return cv::Point(
        cvRound(keypoint.point.x * scale),
        cvRound(keypoint.point.y * scale));
}

cv::Mat drawRawExtrema(const cv::Mat& input,
                       const std::vector<ScaleExtremum>& extrema) {
    cv::Mat output;
    cv::cvtColor(toByteImage(input), output, cv::COLOR_GRAY2BGR);

    for (const auto& extremum : extrema) {
        const int scale = 1 << extremum.octave;
        cv::circle(output,
                   cv::Point(extremum.point.x * scale,
                             extremum.point.y * scale),
                   std::max(2, scale),
                   cv::Scalar(0, 0, 255),
                   1,
                   cv::LINE_AA);
    }
    return output;
}

cv::Mat drawRefined(const cv::Mat& input,
                    const std::vector<RefinedKeypoint>& keypoints) {
    cv::Mat output;
    cv::cvtColor(toByteImage(input), output, cv::COLOR_GRAY2BGR);

    for (const auto& keypoint : keypoints) {
        const int scale = 1 << keypoint.octave;
        const int radius =
            std::max(2,
                     cvRound(2.0 * keypoint.sigma_octave * scale));
        cv::circle(output,
                   originalPoint(keypoint),
                   radius,
                   cv::Scalar(0, 0, 255),
                   1,
                   cv::LINE_AA);
    }
    return output;
}

cv::Mat drawOriented(const cv::Mat& input,
                     const std::vector<OrientedKeypoint>& keypoints) {
    cv::Mat output;
    cv::cvtColor(toByteImage(input), output, cv::COLOR_GRAY2BGR);

    for (const auto& oriented : keypoints) {
        const RefinedKeypoint& keypoint = oriented.keypoint;
        const int scale = 1 << keypoint.octave;
        const int radius =
            std::max(3,
                     cvRound(2.5 * keypoint.sigma_octave * scale));
        const cv::Point center = originalPoint(keypoint);

        cv::circle(output,
                   center,
                   radius,
                   cv::Scalar(0, 0, 255),
                   1,
                   cv::LINE_AA);

        const double angle_rad =
            static_cast<double>(oriented.orientation_deg) *
            CV_PI / 180.0;
        const cv::Point end(
            center.x + cvRound(radius * std::cos(angle_rad)),
            center.y + cvRound(radius * std::sin(angle_rad)));
        cv::line(output,
                 center,
                 end,
                 cv::Scalar(0, 255, 0),
                 1,
                 cv::LINE_AA);
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

double angleDistance(double a, double b) {
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

    Octave quadratic;
    const double target_x = 4.25;
    const double target_y = 3.75;
    const double target_s = 1.20;

    for (int layer = 0; layer < 3; ++layer) {
        cv::Mat image(9, 9, CV_32F);
        for (int row = 0; row < image.rows; ++row) {
            for (int col = 0; col < image.cols; ++col) {
                const double dx = col - target_x;
                const double dy = row - target_y;
                const double ds = layer - target_s;
                image.at<float>(row, col) =
                    static_cast<float>(
                        100.0 -
                        2.0 * dx * dx -
                        3.0 * dy * dy -
                        4.0 * ds * ds);
            }
        }
        quadratic.push_back(image);
    }

    ScaleExtremum candidate;
    candidate.octave = 0;
    candidate.dog_layer = 1;
    candidate.point = cv::Point(4, 4);
    candidate.response = quadratic[1].at<float>(4, 4);

    RefinedKeypoint refined;
    const RefineStatus status =
        refineCandidate(
            quadratic,
            0,
            candidate,
            3,
            1.6,
            1.0,
            10.0,
            refined);

    check(status == RefineStatus::Accepted,
          "quadratic extremum passes localization");
    check(std::abs(refined.point.x - target_x) <= 1e-4,
          "Taylor localization recovers fractional x");
    check(std::abs(refined.point.y - target_y) <= 1e-4,
          "Taylor localization recovers fractional y");
    check(std::abs(refined.layer_offset -
                   static_cast<float>(target_s - 1.0)) <= 1e-4,
          "Taylor localization recovers fractional scale");

    RefinedKeypoint low_contrast;
    Octave weak_quadratic;
    for (const auto& layer : quadratic) {
        weak_quadratic.push_back(layer * 0.005f);
    }
    const RefineStatus weak_status =
        refineCandidate(
            weak_quadratic,
            0,
            candidate,
            3,
            1.6,
            1.0,
            10.0,
            low_contrast);
    check(weak_status == RefineStatus::LowContrast,
          "localized low-contrast extremum is rejected");

    cv::Mat isotropic(9, 9, CV_32F);
    cv::Mat ridge(9, 9, CV_32F);
    for (int row = 0; row < 9; ++row) {
        for (int col = 0; col < 9; ++col) {
            const double x = col - 4.0;
            const double y = row - 4.0;
            isotropic.at<float>(row, col) =
                static_cast<float>(100.0 - x * x - y * y);
            ridge.at<float>(row, col) =
                static_cast<float>(100.0 - 100.0 * x * x - y * y);
        }
    }
    check(passesEdgeResponse(isotropic, 4, 4, 10.0),
          "isotropic blob passes principal-curvature test");
    check(!passesEdgeResponse(ridge, 4, 4, 10.0),
          "edge-like ridge fails principal-curvature test");

    RefinedKeypoint orientation_keypoint;
    orientation_keypoint.point = cv::Point2f(20.0f, 20.0f);
    orientation_keypoint.sigma_octave = 1.6f;

    cv::Mat x_ramp(41, 41, CV_32F);
    cv::Mat y_ramp(41, 41, CV_32F);
    for (int row = 0; row < 41; ++row) {
        for (int col = 0; col < 41; ++col) {
            x_ramp.at<float>(row, col) = static_cast<float>(col);
            y_ramp.at<float>(row, col) = static_cast<float>(row);
        }
    }

    const auto x_hist =
        buildOrientationHistogram(x_ramp, orientation_keypoint, 36);
    const auto x_orientations =
        orientationPeaks(x_hist, 0.8);
    check(!x_orientations.empty(),
          "x ramp produces an orientation peak");
    if (!x_orientations.empty()) {
        check(angleDistance(x_orientations.front(), 0.0) <= 5.0,
              "x ramp dominant orientation is near 0 degrees");
    }

    const auto y_hist =
        buildOrientationHistogram(y_ramp, orientation_keypoint, 36);
    const auto y_orientations =
        orientationPeaks(y_hist, 0.8);
    check(!y_orientations.empty(),
          "y ramp produces an orientation peak");
    if (!y_orientations.empty()) {
        check(angleDistance(y_orientations.front(), 90.0) <= 5.0,
              "y ramp dominant orientation is near 90 degrees");
    }

    cv::Mat flat(41, 41, CV_32F, cv::Scalar(25.0f));
    const auto flat_hist =
        buildOrientationHistogram(flat, orientation_keypoint, 36);
    check(orientationPeaks(flat_hist, 0.8).empty(),
          "flat patch produces no orientation");

    check(std::abs(
              sigmaAtLayer(1.6, 3, 3, 0.0f) -
              3.2f) <= 1e-5f,
          "three intervals double sigma within an octave");

    if (passed) {
        std::cout << "SIFT localization/orientation self-test passed.\n";
    }
    return passed;
}

const char* statusName(RefineStatus status) {
    switch (status) {
        case RefineStatus::Accepted: return "accepted";
        case RefineStatus::OutOfBounds: return "out_of_bounds";
        case RefineStatus::SingularHessian: return "singular_hessian";
        case RefineStatus::DidNotConverge: return "did_not_converge";
        case RefineStatus::LowContrast: return "low_contrast";
        case RefineStatus::EdgeLike: return "edge_like";
    }
    return "unknown";
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
    const Pyramid gaussian =
        buildGaussianPyramid(
            input,
            options.octaves,
            options.intervals,
            options.sigma0);
    const Pyramid dog = buildDoGPyramid(gaussian);

    const auto raw_extrema =
        detectScaleSpaceExtrema(
            dog,
            options.candidate_threshold);

    std::vector<RefinedKeypoint> refined;
    int rejected_out_of_bounds = 0;
    int rejected_singular = 0;
    int rejected_convergence = 0;
    int rejected_contrast = 0;
    int rejected_edge = 0;

    for (const auto& candidate_item : raw_extrema) {
        if (candidate_item.octave < 0 ||
            candidate_item.octave >= static_cast<int>(dog.size())) {
            continue;
        }

        RefinedKeypoint keypoint;
        const RefineStatus status =
            refineCandidate(
                dog[static_cast<std::size_t>(candidate_item.octave)],
                candidate_item.octave,
                candidate_item,
                options.intervals,
                options.sigma0,
                options.contrast_threshold,
                options.edge_ratio,
                keypoint);

        if (status == RefineStatus::Accepted) {
            refined.push_back(keypoint);
        } else if (status == RefineStatus::OutOfBounds) {
            ++rejected_out_of_bounds;
        } else if (status == RefineStatus::SingularHessian) {
            ++rejected_singular;
        } else if (status == RefineStatus::DidNotConverge) {
            ++rejected_convergence;
        } else if (status == RefineStatus::LowContrast) {
            ++rejected_contrast;
        } else if (status == RefineStatus::EdgeLike) {
            ++rejected_edge;
        }
    }

    const auto oriented =
        assignOrientations(
            gaussian,
            refined,
            options.orientation_bins,
            options.orientation_peak_ratio);

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
        output_dir / "01_raw_scale_space_extrema.png",
        drawRawExtrema(input, raw_extrema));
    ok &= writeImage(
        output_dir / "02_localized_keypoints.png",
        drawRefined(input, refined));
    ok &= writeImage(
        output_dir / "03_oriented_keypoints.png",
        drawOriented(input, oriented));

    if (!ok) {
        return 1;
    }

    std::cout << "Input: " << options.input_path << '\n'
              << "Raw extrema: " << raw_extrema.size() << '\n'
              << "Localized + filtered keypoints: " << refined.size() << '\n'
              << "Oriented keypoint instances: " << oriented.size() << '\n'
              << "Rejected out-of-bounds: " << rejected_out_of_bounds << '\n'
              << "Rejected singular Hessian: " << rejected_singular << '\n'
              << "Rejected non-convergence: " << rejected_convergence << '\n'
              << "Rejected low contrast: " << rejected_contrast << '\n'
              << "Rejected edge response: " << rejected_edge << '\n'
              << "Candidate threshold: " << options.candidate_threshold << '\n'
              << "Localized contrast threshold: "
              << options.contrast_threshold << '\n'
              << "Edge ratio: " << options.edge_ratio << '\n'
              << "Orientation bins: " << options.orientation_bins << '\n'
              << "Orientation peak ratio: "
              << options.orientation_peak_ratio << '\n'
              << "Outputs written to: " << output_dir << '\n';

    (void)statusName;
    return 0;
}
