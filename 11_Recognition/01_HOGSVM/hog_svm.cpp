#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kWindowWidth = 64;
constexpr int kWindowHeight = 128;
constexpr int kCellSize = 8;
constexpr int kBlockCells = 2;
constexpr int kBins = 9;
constexpr double kL2HysClip = 0.20;

struct Options {
    std::string input_path;
    std::string output_dir;
    int positives = 80;
    int negatives = 80;
    int test_samples = 20;
    int stride = 8;
    double scale_step = 1.20;
    int max_scales = 5;
    unsigned int seed = 42;
    bool self_test = false;
    bool help = false;
};

struct HogResult {
    std::vector<float> descriptor;
    std::vector<std::array<float, kBins>> cell_histograms;
    int cells_x = 0;
    int cells_y = 0;
};

struct Detection {
    cv::Rect box;
    float raw_score = 0.0f;
    int scale_index = 0;
};

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " --output-dir <dir> [options]\n"
        << "  " << program << " --input <image> --output-dir <dir> [options]\n"
        << "  " << program << " --self-test\n\n"
        << "Options:\n"
        << "  --input <path>          Optional detection image\n"
        << "  --output-dir <path>     Output directory\n"
        << "  --positives <N>         Synthetic training positives (default: 80)\n"
        << "  --negatives <N>         Synthetic training negatives (default: 80)\n"
        << "  --test-samples <N>      Holdout positives + negatives (default: 20 each)\n"
        << "  --stride <pixels>       Sliding-window stride (default: 8)\n"
        << "  --scale-step <value>    Image pyramid step > 1 (default: 1.20)\n"
        << "  --max-scales <N>        Maximum detection pyramid levels (default: 5)\n"
        << "  --seed <N>              Deterministic dataset seed (default: 42)\n"
        << "  --self-test             Run deterministic tests\n"
        << "  --help                  Show this help\n";
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

            if (arg == "--input") {
                options.input_path = value(i, arg);
            } else if (arg == "--output-dir") {
                options.output_dir = value(i, arg);
            } else if (arg == "--positives") {
                options.positives = std::stoi(value(i, arg));
            } else if (arg == "--negatives") {
                options.negatives = std::stoi(value(i, arg));
            } else if (arg == "--test-samples") {
                options.test_samples = std::stoi(value(i, arg));
            } else if (arg == "--stride") {
                options.stride = std::stoi(value(i, arg));
            } else if (arg == "--scale-step") {
                options.scale_step = std::stod(value(i, arg));
            } else if (arg == "--max-scales") {
                options.max_scales = std::stoi(value(i, arg));
            } else if (arg == "--seed") {
                options.seed =
                    static_cast<unsigned int>(
                        std::stoul(value(i, arg)));
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

    if (options.output_dir.empty()) {
        std::cerr << "Error: --output-dir is required.\n";
        return false;
    }

    if (options.positives < 10 ||
        options.negatives < 10 ||
        options.test_samples < 1 ||
        options.stride <= 0 ||
        options.scale_step <= 1.0 ||
        options.max_scales < 1) {
        std::cerr << "Error: invalid training/detection configuration.\n";
        return false;
    }

    return true;
}

cv::Mat toGray(const cv::Mat& input) {
    if (input.empty()) {
        return cv::Mat();
    }

    if (input.channels() == 1) {
        return input.clone();
    }

    cv::Mat gray;
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

int descriptorLength() {
    const int cells_x =
        kWindowWidth / kCellSize;
    const int cells_y =
        kWindowHeight / kCellSize;
    const int blocks_x =
        cells_x - kBlockCells + 1;
    const int blocks_y =
        cells_y - kBlockCells + 1;

    return blocks_x *
           blocks_y *
           kBlockCells *
           kBlockCells *
           kBins;
}

void addOrientationVote(
    std::array<float, kBins>& histogram,
    float angle_degrees,
    float magnitude) {
    float angle =
        std::fmod(angle_degrees, 180.0f);

    if (angle < 0.0f) {
        angle += 180.0f;
    }

    const float bin_width =
        180.0f /
        static_cast<float>(kBins);

    const float position =
        angle / bin_width;

    const int lower =
        static_cast<int>(
            std::floor(position)) %
        kBins;

    const int upper =
        (lower + 1) %
        kBins;

    const float upper_weight =
        position -
        std::floor(position);

    const float lower_weight =
        1.0f -
        upper_weight;

    histogram[static_cast<std::size_t>(lower)] +=
        lower_weight *
        magnitude;

    histogram[static_cast<std::size_t>(upper)] +=
        upper_weight *
        magnitude;
}

std::vector<float> normalizeBlockL2Hys(
    const std::vector<float>& values) {
    constexpr double epsilon = 1e-6;

    double norm2 = 0.0;

    for (const float value : values) {
        norm2 +=
            static_cast<double>(value) *
            value;
    }

    const double inverse_norm =
        1.0 /
        std::sqrt(
            norm2 +
            epsilon * epsilon);

    std::vector<float> normalized;
    normalized.reserve(values.size());

    for (const float value : values) {
        normalized.push_back(
            static_cast<float>(
                std::min(
                    kL2HysClip,
                    value *
                    inverse_norm)));
    }

    norm2 = 0.0;

    for (const float value : normalized) {
        norm2 +=
            static_cast<double>(value) *
            value;
    }

    const double inverse_norm_after_clip =
        1.0 /
        std::sqrt(
            norm2 +
            epsilon * epsilon);

    for (float& value : normalized) {
        value =
            static_cast<float>(
                value *
                inverse_norm_after_clip);
    }

    return normalized;
}

HogResult computeHog(const cv::Mat& input) {
    const cv::Mat gray =
        toGray(input);

    if (gray.size() !=
        cv::Size(
            kWindowWidth,
            kWindowHeight)) {
        throw std::runtime_error(
            "HOG input must be exactly 64x128.");
    }

    cv::Mat float_image;
    gray.convertTo(
        float_image,
        CV_32F,
        1.0 / 255.0);

    cv::Mat gx;
    cv::Mat gy;

    cv::Sobel(
        float_image,
        gx,
        CV_32F,
        1,
        0,
        1,
        1.0,
        0.0,
        cv::BORDER_REFLECT101);

    cv::Sobel(
        float_image,
        gy,
        CV_32F,
        0,
        1,
        1,
        1.0,
        0.0,
        cv::BORDER_REFLECT101);

    cv::Mat magnitude;
    cv::Mat angle;

    cv::cartToPolar(
        gx,
        gy,
        magnitude,
        angle,
        true);

    HogResult result;
    result.cells_x =
        kWindowWidth /
        kCellSize;
    result.cells_y =
        kWindowHeight /
        kCellSize;

    result.cell_histograms.resize(
        static_cast<std::size_t>(
            result.cells_x *
            result.cells_y));

    for (int row = 0;
         row < gray.rows;
         ++row) {
        for (int col = 0;
             col < gray.cols;
             ++col) {
            const int cell_x =
                col /
                kCellSize;
            const int cell_y =
                row /
                kCellSize;

            auto& histogram =
                result.cell_histograms[
                    static_cast<std::size_t>(
                        cell_y *
                            result.cells_x +
                        cell_x)];

            addOrientationVote(
                histogram,
                angle.at<float>(
                    row,
                    col),
                magnitude.at<float>(
                    row,
                    col));
        }
    }

    for (int block_y = 0;
         block_y <=
             result.cells_y -
             kBlockCells;
         ++block_y) {
        for (int block_x = 0;
             block_x <=
                 result.cells_x -
                 kBlockCells;
             ++block_x) {
            std::vector<float> block;
            block.reserve(
                kBlockCells *
                kBlockCells *
                kBins);

            for (int dy = 0;
                 dy < kBlockCells;
                 ++dy) {
                for (int dx = 0;
                     dx < kBlockCells;
                     ++dx) {
                    const auto& histogram =
                        result.cell_histograms[
                            static_cast<std::size_t>(
                                (block_y + dy) *
                                    result.cells_x +
                                (block_x + dx))];

                    block.insert(
                        block.end(),
                        histogram.begin(),
                        histogram.end());
                }
            }

            const auto normalized =
                normalizeBlockL2Hys(
                    block);

            result.descriptor.insert(
                result.descriptor.end(),
                normalized.begin(),
                normalized.end());
        }
    }

    return result;
}

cv::Mat makePositiveSample(
    std::mt19937& rng,
    bool jitter) {
    cv::Mat image(
        kWindowHeight,
        kWindowWidth,
        CV_8U,
        cv::Scalar(20));

    int dx = 0;
    int dy = 0;
    int thickness_delta = 0;

    if (jitter) {
        std::uniform_int_distribution<int>
            shift(-3, 3);
        std::uniform_int_distribution<int>
            thickness(-2, 2);

        dx = shift(rng);
        dy = shift(rng);
        thickness_delta =
            thickness(rng);
    }

    const cv::Point center(
        kWindowWidth / 2 + dx,
        20 + dy);

    cv::circle(
        image,
        center,
        8,
        cv::Scalar(235),
        cv::FILLED,
        cv::LINE_AA);

    const int torso_half =
        std::max(
            8,
            11 + thickness_delta);

    std::vector<cv::Point> torso = {
        cv::Point(
            center.x - torso_half,
            31 + dy),
        cv::Point(
            center.x + torso_half,
            31 + dy),
        cv::Point(
            center.x + torso_half + 3,
            76 + dy),
        cv::Point(
            center.x - torso_half - 3,
            76 + dy)
    };

    cv::fillConvexPoly(
        image,
        torso,
        cv::Scalar(220),
        cv::LINE_AA);

    cv::line(
        image,
        cv::Point(
            center.x - torso_half,
            42 + dy),
        cv::Point(
            center.x - 22,
            66 + dy),
        cv::Scalar(230),
        5,
        cv::LINE_AA);

    cv::line(
        image,
        cv::Point(
            center.x + torso_half,
            42 + dy),
        cv::Point(
            center.x + 22,
            66 + dy),
        cv::Scalar(230),
        5,
        cv::LINE_AA);

    cv::line(
        image,
        cv::Point(
            center.x - 7,
            75 + dy),
        cv::Point(
            center.x - 12,
            117 + dy),
        cv::Scalar(235),
        7,
        cv::LINE_AA);

    cv::line(
        image,
        cv::Point(
            center.x + 7,
            75 + dy),
        cv::Point(
            center.x + 12,
            117 + dy),
        cv::Scalar(235),
        7,
        cv::LINE_AA);

    cv::GaussianBlur(
        image,
        image,
        cv::Size(3, 3),
        0.7);

    return image;
}

cv::Mat makeNegativeSample(
    std::mt19937& rng) {
    cv::Mat image(
        kWindowHeight,
        kWindowWidth,
        CV_8U,
        cv::Scalar(20));

    std::uniform_int_distribution<int>
        type_dist(0, 4);

    const int type =
        type_dist(rng);

    std::uniform_int_distribution<int>
        gray_dist(120, 240);

    const int intensity =
        gray_dist(rng);

    if (type == 0) {
        cv::rectangle(
            image,
            cv::Rect(8, 46, 48, 30),
            cv::Scalar(intensity),
            cv::FILLED,
            cv::LINE_AA);
    } else if (type == 1) {
        cv::circle(
            image,
            cv::Point(32, 64),
            23,
            cv::Scalar(intensity),
            cv::FILLED,
            cv::LINE_AA);
    } else if (type == 2) {
        for (int y = 12;
             y < 120;
             y += 18) {
            cv::line(
                image,
                cv::Point(5, y),
                cv::Point(59, y),
                cv::Scalar(intensity),
                4,
                cv::LINE_AA);
        }
    } else if (type == 3) {
        for (int x = 8;
             x < 60;
             x += 14) {
            cv::line(
                image,
                cv::Point(x, 14),
                cv::Point(x, 114),
                cv::Scalar(intensity),
                4,
                cv::LINE_AA);
        }
    } else {
        std::uniform_int_distribution<int>
            x_dist(4, 48);
        std::uniform_int_distribution<int>
            y_dist(5, 105);
        std::uniform_int_distribution<int>
            size_dist(8, 24);

        for (int i = 0;
             i < 5;
             ++i) {
            const int x =
                x_dist(rng);
            const int y =
                y_dist(rng);
            const int size =
                size_dist(rng);

            cv::rectangle(
                image,
                cv::Rect(
                    x,
                    y,
                    std::min(
                        size,
                        kWindowWidth - x),
                    std::min(
                        size,
                        kWindowHeight - y)),
                cv::Scalar(
                    gray_dist(rng)),
                cv::FILLED,
                cv::LINE_AA);
        }
    }

    cv::GaussianBlur(
        image,
        image,
        cv::Size(3, 3),
        0.8);

    return image;
}

cv::Mat descriptorRow(
    const cv::Mat& sample) {
    const HogResult hog =
        computeHog(sample);

    cv::Mat row(
        1,
        static_cast<int>(
            hog.descriptor.size()),
        CV_32F);

    for (int col = 0;
         col < row.cols;
         ++col) {
        row.at<float>(0, col) =
            hog.descriptor[
                static_cast<std::size_t>(
                    col)];
    }

    return row;
}

void appendTrainingRow(
    cv::Mat& samples,
    cv::Mat& labels,
    const cv::Mat& sample,
    int label) {
    samples.push_back(
        descriptorRow(sample));

    cv::Mat label_row(
        1,
        1,
        CV_32S);

    label_row.at<int>(0, 0) =
        label;

    labels.push_back(
        label_row);
}

cv::Ptr<cv::ml::SVM> trainLinearSvm(
    int positive_count,
    int negative_count,
    unsigned int seed) {
    std::mt19937 rng(seed);

    cv::Mat samples;
    cv::Mat labels;

    for (int i = 0;
         i < positive_count;
         ++i) {
        appendTrainingRow(
            samples,
            labels,
            makePositiveSample(
                rng,
                true),
            +1);
    }

    for (int i = 0;
         i < negative_count;
         ++i) {
        appendTrainingRow(
            samples,
            labels,
            makeNegativeSample(
                rng),
            -1);
    }

    auto svm =
        cv::ml::SVM::create();

    svm->setType(
        cv::ml::SVM::C_SVC);

    svm->setKernel(
        cv::ml::SVM::LINEAR);

    svm->setC(0.5);

    svm->setTermCriteria(
        cv::TermCriteria(
            cv::TermCriteria::COUNT |
            cv::TermCriteria::EPS,
            500,
            1e-7));

    if (!svm->train(
            samples,
            cv::ml::ROW_SAMPLE,
            labels)) {
        throw std::runtime_error(
            "Linear SVM training failed.");
    }

    return svm;
}

int predictLabel(
    const cv::Ptr<cv::ml::SVM>& svm,
    const cv::Mat& window) {
    return cvRound(
        svm->predict(
            descriptorRow(window)));
}

float rawDecision(
    const cv::Ptr<cv::ml::SVM>& svm,
    const cv::Mat& window) {
    return svm->predict(
        descriptorRow(window),
        cv::noArray(),
        cv::ml::StatModel::RAW_OUTPUT);
}

double evaluateAccuracy(
    const cv::Ptr<cv::ml::SVM>& svm,
    int count_per_class,
    unsigned int seed) {
    std::mt19937 rng(seed);

    int correct = 0;
    int total = 0;

    for (int i = 0;
         i < count_per_class;
         ++i) {
        const cv::Mat positive =
            makePositiveSample(
                rng,
                true);

        if (predictLabel(
                svm,
                positive) == +1) {
            ++correct;
        }
        ++total;

        const cv::Mat negative =
            makeNegativeSample(rng);

        if (predictLabel(
                svm,
                negative) == -1) {
            ++correct;
        }
        ++total;
    }

    return total == 0
        ? 0.0
        : static_cast<double>(
              correct) /
          total;
}

cv::Mat visualizeHog(
    const cv::Mat& sample,
    const HogResult& hog,
    int scale = 4) {
    cv::Mat enlarged;

    cv::resize(
        sample,
        enlarged,
        cv::Size(),
        scale,
        scale,
        cv::INTER_NEAREST);

    cv::cvtColor(
        enlarged,
        enlarged,
        cv::COLOR_GRAY2BGR);

    const float bin_width =
        180.0f /
        static_cast<float>(kBins);

    for (int cell_y = 0;
         cell_y < hog.cells_y;
         ++cell_y) {
        for (int cell_x = 0;
             cell_x < hog.cells_x;
             ++cell_x) {
            const auto& histogram =
                hog.cell_histograms[
                    static_cast<std::size_t>(
                        cell_y *
                            hog.cells_x +
                        cell_x)];

            const float max_value =
                *std::max_element(
                    histogram.begin(),
                    histogram.end());

            if (max_value <= 1e-6f) {
                continue;
            }

            const cv::Point2f center(
                static_cast<float>(
                    (cell_x *
                         kCellSize +
                     kCellSize * 0.5) *
                    scale),
                static_cast<float>(
                    (cell_y *
                         kCellSize +
                     kCellSize * 0.5) *
                    scale));

            const float half_length =
                0.42f *
                kCellSize *
                scale;

            for (int bin = 0;
                 bin < kBins;
                 ++bin) {
                const float strength =
                    histogram[
                        static_cast<std::size_t>(
                            bin)] /
                    max_value;

                if (strength < 0.10f) {
                    continue;
                }

                const float angle_deg =
                    (bin + 0.5f) *
                    bin_width;

                const float angle_rad =
                    angle_deg *
                    static_cast<float>(
                        CV_PI /
                        180.0);

                const cv::Point2f direction(
                    std::cos(angle_rad),
                    std::sin(angle_rad));

                const cv::Point2f delta =
                    direction *
                    (half_length *
                     strength);

                cv::line(
                    enlarged,
                    center - delta,
                    center + delta,
                    cv::Scalar(
                        0,
                        255,
                        255),
                    1,
                    cv::LINE_AA);
            }
        }
    }

    return enlarged;
}

cv::Mat createDetectionScene(
    std::mt19937& rng,
    cv::Rect& ground_truth) {
    cv::Mat scene(
        280,
        360,
        CV_8U,
        cv::Scalar(30));

    for (int y = 20;
         y < scene.rows;
         y += 40) {
        cv::line(
            scene,
            cv::Point(0, y),
            cv::Point(
                scene.cols - 1,
                y),
            cv::Scalar(45),
            1,
            cv::LINE_AA);
    }

    const cv::Mat positive =
        makePositiveSample(
            rng,
            true);

    ground_truth =
        cv::Rect(
            144,
            80,
            kWindowWidth,
            kWindowHeight);

    positive.copyTo(
        scene(
            ground_truth));

    cv::rectangle(
        scene,
        cv::Rect(
            20,
            35,
            70,
            28),
        cv::Scalar(180),
        cv::FILLED,
        cv::LINE_AA);

    cv::circle(
        scene,
        cv::Point(300, 190),
        28,
        cv::Scalar(180),
        cv::FILLED,
        cv::LINE_AA);

    return scene;
}

float intersectionOverUnion(
    const cv::Rect& a,
    const cv::Rect& b) {
    const cv::Rect intersection =
        a & b;

    const int intersection_area =
        intersection.area();

    const int union_area =
        a.area() +
        b.area() -
        intersection_area;

    return union_area <= 0
        ? 0.0f
        : static_cast<float>(
              intersection_area) /
          static_cast<float>(
              union_area);
}

std::vector<Detection> nonMaximumSuppress(
    std::vector<Detection> detections,
    float iou_threshold) {
    if (detections.empty()) {
        return {};
    }

    // Raw-output sign is implementation/label-order dependent. We therefore
    // rank only among windows that SVM::predict already classified positive,
    // using absolute distance from the hyperplane as confidence magnitude.
    std::sort(
        detections.begin(),
        detections.end(),
        [](const Detection& a,
           const Detection& b) {
            return std::abs(a.raw_score) >
                   std::abs(b.raw_score);
        });

    std::vector<Detection> kept;
    std::vector<bool> suppressed(
        detections.size(),
        false);

    for (std::size_t i = 0;
         i < detections.size();
         ++i) {
        if (suppressed[i]) {
            continue;
        }

        kept.push_back(
            detections[i]);

        for (std::size_t j = i + 1;
             j < detections.size();
             ++j) {
            if (suppressed[j]) {
                continue;
            }

            if (intersectionOverUnion(
                    detections[i].box,
                    detections[j].box) >
                iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return kept;
}

std::vector<Detection> detectMultiScale(
    const cv::Mat& input,
    const cv::Ptr<cv::ml::SVM>& svm,
    int stride,
    double scale_step,
    int max_scales) {
    const cv::Mat gray =
        toGray(input);

    std::vector<Detection> candidates;

    double scale = 1.0;

    for (int level = 0;
         level < max_scales;
         ++level) {
        const int scaled_width =
            cvRound(
                gray.cols /
                scale);

        const int scaled_height =
            cvRound(
                gray.rows /
                scale);

        if (scaled_width <
                kWindowWidth ||
            scaled_height <
                kWindowHeight) {
            break;
        }

        cv::Mat scaled;

        cv::resize(
            gray,
            scaled,
            cv::Size(
                scaled_width,
                scaled_height),
            0.0,
            0.0,
            cv::INTER_LINEAR);

        for (int y = 0;
             y <=
                 scaled.rows -
                 kWindowHeight;
             y += stride) {
            for (int x = 0;
                 x <=
                     scaled.cols -
                     kWindowWidth;
                 x += stride) {
                const cv::Rect roi(
                    x,
                    y,
                    kWindowWidth,
                    kWindowHeight);

                const cv::Mat window =
                    scaled(roi);

                if (predictLabel(
                        svm,
                        window) != +1) {
                    continue;
                }

                Detection detection;
                detection.box =
                    cv::Rect(
                        cvRound(
                            x *
                            scale),
                        cvRound(
                            y *
                            scale),
                        cvRound(
                            kWindowWidth *
                            scale),
                        cvRound(
                            kWindowHeight *
                            scale));

                detection.box &=
                    cv::Rect(
                        0,
                        0,
                        gray.cols,
                        gray.rows);

                detection.raw_score =
                    rawDecision(
                        svm,
                        window);

                detection.scale_index =
                    level;

                candidates.push_back(
                    detection);
            }
        }

        scale *=
            scale_step;
    }

    return nonMaximumSuppress(
        std::move(candidates),
        0.35f);
}

cv::Mat drawDetections(
    const cv::Mat& image,
    const std::vector<Detection>& detections) {
    cv::Mat output;

    if (image.channels() == 1) {
        cv::cvtColor(
            image,
            output,
            cv::COLOR_GRAY2BGR);
    } else {
        output =
            image.clone();
    }

    for (const auto& detection :
         detections) {
        cv::rectangle(
            output,
            detection.box,
            cv::Scalar(
                0,
                255,
                0),
            2,
            cv::LINE_AA);

        cv::putText(
            output,
            "HOG+SVM",
            detection.box.tl() +
                cv::Point(0, -5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.45,
            cv::Scalar(
                0,
                255,
                0),
            1,
            cv::LINE_AA);
    }

    return output;
}

bool writeImage(
    const fs::path& path,
    const cv::Mat& image) {
    if (!cv::imwrite(
            path.string(),
            image)) {
        std::cerr
            << "Error: failed to write "
            << path
            << '\n';
        return false;
    }

    return true;
}

bool runSelfTest() {
    bool passed = true;

    auto check =
        [&passed](
            bool condition,
            const std::string& name) {
            if (!condition) {
                std::cerr
                    << "Self-test failed: "
                    << name
                    << '\n';
                passed = false;
            }
        };

    check(
        descriptorLength() == 3780,
        "64x128 / 8x8 / 2x2 / 9-bin HOG has 3780 dimensions");

    cv::Mat constant(
        kWindowHeight,
        kWindowWidth,
        CV_8U,
        cv::Scalar(128));

    const HogResult constant_hog =
        computeHog(
            constant);

    double constant_norm2 = 0.0;

    for (const float value :
         constant_hog.descriptor) {
        constant_norm2 +=
            static_cast<double>(value) *
            value;
    }

    check(
        constant_hog.descriptor.size() ==
            static_cast<std::size_t>(
                descriptorLength()),
        "manual HOG descriptor has expected length");

    check(
        constant_norm2 < 1e-12,
        "constant image produces zero HOG descriptor");

    std::array<float, kBins> histogram{};

    addOrientationVote(
        histogram,
        0.0f,
        1.0f);

    check(
        std::abs(
            histogram[0] -
            1.0f) < 1e-6f,
        "0-degree gradient votes into first unsigned bin");

    histogram.fill(0.0f);

    addOrientationVote(
        histogram,
        90.0f,
        1.0f);

    const int ninety_bin =
        4;

    check(
        histogram[
            static_cast<std::size_t>(
                ninety_bin)] > 0.4f &&
        histogram[
            static_cast<std::size_t>(
                ninety_bin + 1)] > 0.4f,
        "90-degree gradient interpolates between neighboring bins");

    const std::vector<float> raw_block = {
        1.0f,
        2.0f,
        3.0f,
        4.0f
    };

    const auto normalized =
        normalizeBlockL2Hys(
            raw_block);

    double norm2 = 0.0;

    for (const float value :
         normalized) {
        norm2 +=
            static_cast<double>(value) *
            value;
    }

    check(
        std::abs(
            std::sqrt(norm2) -
            1.0) < 1e-5,
        "L2-Hys block is renormalized to unit length");

    const auto svm =
        trainLinearSvm(
            80,
            80,
            19);

    check(
        !svm.empty() &&
        svm->isTrained(),
        "linear SVM trains successfully");

    const double accuracy =
        evaluateAccuracy(
            svm,
            30,
            1234);

    check(
        accuracy >= 0.95,
        "synthetic holdout accuracy is at least 95 percent");

    std::mt19937 rng(2026);

    const cv::Mat positive =
        makePositiveSample(
            rng,
            true);

    const cv::Mat negative =
        makeNegativeSample(
            rng);

    check(
        predictLabel(
            svm,
            positive) == +1,
        "held-out positive sample is classified positive");

    check(
        predictLabel(
            svm,
            negative) == -1,
        "held-out negative sample is classified negative");

    cv::Rect ground_truth;

    const cv::Mat scene =
        createDetectionScene(
            rng,
            ground_truth);

    const auto detections =
        detectMultiScale(
            scene,
            svm,
            8,
            1.20,
            4);

    float best_iou = 0.0f;

    for (const auto& detection :
         detections) {
        best_iou =
            std::max(
                best_iou,
                intersectionOverUnion(
                    detection.box,
                    ground_truth));
    }

    check(
        best_iou >= 0.45f,
        "sliding-window detector localizes synthetic positive region");

    if (passed) {
        std::cout
            << "HOG + SVM self-test passed.\n";
    }

    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;

    if (!parseArguments(
            argc,
            argv,
            options)) {
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

    const auto svm =
        trainLinearSvm(
            options.positives,
            options.negatives,
            options.seed);

    const double holdout_accuracy =
        evaluateAccuracy(
            svm,
            options.test_samples,
            options.seed + 1000U);

    std::mt19937 rng(
        options.seed + 2000U);

    const cv::Mat positive =
        makePositiveSample(
            rng,
            false);

    const cv::Mat negative =
        makeNegativeSample(
            rng);

    const HogResult positive_hog =
        computeHog(
            positive);

    cv::Mat detection_image;
    cv::Rect synthetic_ground_truth;
    bool synthetic_scene = false;

    if (options.input_path.empty()) {
        detection_image =
            createDetectionScene(
                rng,
                synthetic_ground_truth);

        synthetic_scene = true;
    } else {
        detection_image =
            cv::imread(
                options.input_path,
                cv::IMREAD_GRAYSCALE);

        if (detection_image.empty()) {
            std::cerr
                << "Error: could not read input image: "
                << options.input_path
                << '\n';
            return 1;
        }
    }

    const auto detections =
        detectMultiScale(
            detection_image,
            svm,
            options.stride,
            options.scale_step,
            options.max_scales);

    const fs::path output_dir(
        options.output_dir);

    std::error_code error;
    fs::create_directories(
        output_dir,
        error);

    if (error) {
        std::cerr
            << "Error: could not create output directory "
            << output_dir
            << ": "
            << error.message()
            << '\n';
        return 1;
    }

    bool ok = true;

    ok &= writeImage(
        output_dir /
            "01_positive_sample.png",
        positive);

    ok &= writeImage(
        output_dir /
            "02_negative_sample.png",
        negative);

    ok &= writeImage(
        output_dir /
            "03_hog_cells.png",
        visualizeHog(
            positive,
            positive_hog));

    ok &= writeImage(
        output_dir /
            "04_detection_input.png",
        detection_image);

    ok &= writeImage(
        output_dir /
            "05_hog_svm_detections.png",
        drawDetections(
            detection_image,
            detections));

    if (!ok) {
        return 1;
    }

    std::cout
        << std::fixed
        << std::setprecision(4)
        << "HOG configuration\n"
        << "  window: "
        << kWindowWidth
        << "x"
        << kWindowHeight
        << '\n'
        << "  cell: "
        << kCellSize
        << "x"
        << kCellSize
        << '\n'
        << "  block: "
        << kBlockCells
        << "x"
        << kBlockCells
        << " cells\n"
        << "  orientation bins: "
        << kBins
        << " unsigned [0,180)\n"
        << "  descriptor length: "
        << positive_hog.descriptor.size()
        << '\n'
        << "SVM configuration\n"
        << "  kernel: linear\n"
        << "  training positives: "
        << options.positives
        << '\n'
        << "  training negatives: "
        << options.negatives
        << '\n'
        << "  holdout accuracy: "
        << holdout_accuracy
        << '\n'
        << "Detection\n"
        << "  retained detections after NMS: "
        << detections.size()
        << '\n';

    if (synthetic_scene) {
        float best_iou = 0.0f;

        for (const auto& detection :
             detections) {
            best_iou =
                std::max(
                    best_iou,
                    intersectionOverUnion(
                        detection.box,
                        synthetic_ground_truth));
        }

        std::cout
            << "  synthetic best IoU: "
            << best_iou
            << '\n';
    }

    std::cout
        << "Outputs written to: "
        << output_dir
        << '\n';

    return 0;
}
