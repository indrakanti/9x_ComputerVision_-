#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kImageSize = 96;
constexpr int kFeatureCount = 7;
constexpr int kClassCount = 3;

enum class ShapeClass : int {
    Circle = 0,
    Rectangle = 1,
    Triangle = 2
};

struct Options {
    std::string output_dir;
    int train_per_class = 90;
    int test_per_class = 30;
    int knn_k = 5;
    unsigned int seed = 42;
    bool self_test = false;
    bool help = false;
};

struct FeatureVector {
    std::array<float, kFeatureCount> values{};
};

struct StandardScaler {
    std::array<double, kFeatureCount> mean{};
    std::array<double, kFeatureCount> stddev{};
};

struct Dataset {
    cv::Mat features;
    cv::Mat labels_int;
    cv::Mat labels_float;
    std::vector<cv::Mat> examples;
};

struct Metrics {
    cv::Mat confusion;
    double accuracy = 0.0;
    std::array<double, kClassCount> precision{};
    std::array<double, kClassCount> recall{};
};

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " --output-dir <dir> [options]\n"
        << "  " << program << " --self-test\n\n"
        << "Options:\n"
        << "  --output-dir <path>      Output directory\n"
        << "  --train-per-class <N>    Training samples per class (default: 90)\n"
        << "  --test-per-class <N>     Test samples per class (default: 30)\n"
        << "  --knn-k <N>              k for k-NN (default: 5)\n"
        << "  --seed <N>               Deterministic random seed (default: 42)\n"
        << "  --self-test              Run deterministic tests\n"
        << "  --help                   Show this help\n";
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

            if (arg == "--output-dir") {
                options.output_dir = value(i, arg);
            } else if (arg == "--train-per-class") {
                options.train_per_class = std::stoi(value(i, arg));
            } else if (arg == "--test-per-class") {
                options.test_per_class = std::stoi(value(i, arg));
            } else if (arg == "--knn-k") {
                options.knn_k = std::stoi(value(i, arg));
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
        std::cerr
            << "Argument error: "
            << error.what()
            << '\n';
        return false;
    }

    return true;
}

bool validateOptions(const Options& options) {
    if (options.help || options.self_test) {
        return true;
    }

    if (options.output_dir.empty()) {
        std::cerr
            << "Error: --output-dir is required.\n";
        return false;
    }

    if (options.train_per_class < 10 ||
        options.test_per_class < 5 ||
        options.knn_k < 1 ||
        options.knn_k >
            options.train_per_class * kClassCount) {
        std::cerr
            << "Error: invalid dataset/k-NN configuration.\n";
        return false;
    }

    return true;
}

std::string className(int label) {
    switch (
        static_cast<ShapeClass>(
            label)) {
        case ShapeClass::Circle:
            return "circle";
        case ShapeClass::Rectangle:
            return "rectangle";
        case ShapeClass::Triangle:
            return "triangle";
    }

    return "unknown";
}

cv::Mat generateShape(
    ShapeClass shape,
    std::mt19937& rng) {
    cv::Mat image(
        kImageSize,
        kImageSize,
        CV_8U,
        cv::Scalar(0));

    std::uniform_int_distribution<int>
        shift(-5, 5);
    std::uniform_int_distribution<int>
        intensity(190, 255);
    std::uniform_int_distribution<int>
        size_jitter(-4, 4);

    const int cx =
        kImageSize / 2 +
        shift(rng);

    const int cy =
        kImageSize / 2 +
        shift(rng);

    const int value =
        intensity(rng);

    if (shape ==
        ShapeClass::Circle) {
        const int radius =
            23 +
            size_jitter(rng);

        cv::circle(
            image,
            cv::Point(cx, cy),
            radius,
            cv::Scalar(value),
            cv::FILLED,
            cv::LINE_AA);
    } else if (
        shape ==
        ShapeClass::Rectangle) {
        const int width =
            48 +
            size_jitter(rng);

        const int height =
            30 +
            size_jitter(rng);

        const cv::Rect box(
            std::max(
                1,
                cx - width / 2),
            std::max(
                1,
                cy - height / 2),
            std::min(
                width,
                kImageSize -
                    std::max(
                        1,
                        cx - width / 2) -
                    1),
            std::min(
                height,
                kImageSize -
                    std::max(
                        1,
                        cy - height / 2) -
                    1));

        cv::rectangle(
            image,
            box,
            cv::Scalar(value),
            cv::FILLED,
            cv::LINE_AA);
    } else {
        const int half_width =
            27 +
            size_jitter(rng);

        const int half_height =
            27 +
            size_jitter(rng);

        std::vector<cv::Point> triangle = {
            cv::Point(
                cx,
                cy - half_height),
            cv::Point(
                cx - half_width,
                cy + half_height),
            cv::Point(
                cx + half_width,
                cy + half_height)
        };

        cv::fillConvexPoly(
            image,
            triangle,
            cv::Scalar(value),
            cv::LINE_AA);
    }

    cv::GaussianBlur(
        image,
        image,
        cv::Size(3, 3),
        0.6);

    cv::threshold(
        image,
        image,
        96,
        255,
        cv::THRESH_BINARY);

    return image;
}

std::vector<cv::Point> largestContour(
    const cv::Mat& binary) {
    std::vector<std::vector<cv::Point>>
        contours;

    cv::findContours(
        binary.clone(),
        contours,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        return {};
    }

    return *std::max_element(
        contours.begin(),
        contours.end(),
        [](const auto& a,
           const auto& b) {
            return std::abs(
                       cv::contourArea(a)) <
                   std::abs(
                       cv::contourArea(b));
        });
}

FeatureVector extractFeatures(
    const cv::Mat& binary) {
    const auto contour =
        largestContour(binary);

    if (contour.empty()) {
        throw std::runtime_error(
            "No foreground contour found.");
    }

    const double area =
        std::abs(
            cv::contourArea(
                contour));

    const double perimeter =
        cv::arcLength(
            contour,
            true);

    const cv::Rect bbox =
        cv::boundingRect(
            contour);

    const double circularity =
        perimeter > 1e-12
            ? 4.0 *
                  CV_PI *
                  area /
                  (perimeter *
                   perimeter)
            : 0.0;

    const double aspect_ratio =
        bbox.height > 0
            ? static_cast<double>(
                  bbox.width) /
                  bbox.height
            : 0.0;

    const double bbox_area =
        static_cast<double>(
            bbox.area());

    const double extent =
        bbox_area > 0.0
            ? area /
                  bbox_area
            : 0.0;

    std::vector<cv::Point> hull;

    cv::convexHull(
        contour,
        hull);

    const double hull_area =
        std::abs(
            cv::contourArea(
                hull));

    const double solidity =
        hull_area > 1e-12
            ? area /
                  hull_area
            : 0.0;

    std::vector<cv::Point> polygon;

    cv::approxPolyDP(
        contour,
        polygon,
        0.02 *
            perimeter,
        true);

    const cv::Moments moments =
        cv::moments(contour);

    double hu[7]{};
    cv::HuMoments(
        moments,
        hu);

    const double hu1 =
        std::abs(hu[0]) < 1e-20
            ? 0.0
            : -std::copysign(
                  std::log10(
                      std::abs(
                          hu[0])),
                  hu[0]);

    FeatureVector features;

    features.values = {
        static_cast<float>(
            area /
            (kImageSize *
             kImageSize)),
        static_cast<float>(
            perimeter /
            (2.0 *
             (kImageSize +
              kImageSize))),
        static_cast<float>(
            circularity),
        static_cast<float>(
            aspect_ratio),
        static_cast<float>(
            extent),
        static_cast<float>(
            solidity),
        static_cast<float>(
            hu1)
    };

    return features;
}

Dataset generateDataset(
    int samples_per_class,
    unsigned int seed,
    bool save_examples) {
    std::mt19937 rng(seed);

    Dataset dataset;

    for (int label = 0;
         label < kClassCount;
         ++label) {
        for (int i = 0;
             i < samples_per_class;
             ++i) {
            const cv::Mat image =
                generateShape(
                    static_cast<ShapeClass>(
                        label),
                    rng);

            const auto features =
                extractFeatures(
                    image);

            cv::Mat row(
                1,
                kFeatureCount,
                CV_32F);

            for (int col = 0;
                 col < kFeatureCount;
                 ++col) {
                row.at<float>(
                    0,
                    col) =
                    features.values[
                        static_cast<std::size_t>(
                            col)];
            }

            dataset.features.push_back(
                row);

            cv::Mat label_int(
                1,
                1,
                CV_32S);

            label_int.at<int>(
                0,
                0) =
                label;

            dataset.labels_int.push_back(
                label_int);

            cv::Mat label_float(
                1,
                1,
                CV_32F);

            label_float.at<float>(
                0,
                0) =
                static_cast<float>(
                    label);

            dataset.labels_float.push_back(
                label_float);

            if (save_examples &&
                i == 0) {
                dataset.examples.push_back(
                    image.clone());
            }
        }
    }

    return dataset;
}

StandardScaler fitScaler(
    const cv::Mat& features) {
    StandardScaler scaler;

    for (int col = 0;
         col < features.cols;
         ++col) {
        double sum = 0.0;

        for (int row = 0;
             row < features.rows;
             ++row) {
            sum +=
                features.at<float>(
                    row,
                    col);
        }

        const double mean =
            sum /
            features.rows;

        double variance = 0.0;

        for (int row = 0;
             row < features.rows;
             ++row) {
            const double delta =
                features.at<float>(
                    row,
                    col) -
                mean;

            variance +=
                delta *
                delta;
        }

        variance /=
            std::max(
                1,
                features.rows);

        scaler.mean[
            static_cast<std::size_t>(
                col)] =
            mean;

        scaler.stddev[
            static_cast<std::size_t>(
                col)] =
            std::max(
                1e-8,
                std::sqrt(
                    variance));
    }

    return scaler;
}

cv::Mat applyScaler(
    const cv::Mat& features,
    const StandardScaler& scaler) {
    cv::Mat output(
        features.size(),
        CV_32F);

    for (int row = 0;
         row < features.rows;
         ++row) {
        for (int col = 0;
             col < features.cols;
             ++col) {
            output.at<float>(
                row,
                col) =
                static_cast<float>(
                    (features.at<float>(
                         row,
                         col) -
                     scaler.mean[
                         static_cast<std::size_t>(
                             col)]) /
                    scaler.stddev[
                        static_cast<std::size_t>(
                            col)]);
        }
    }

    return output;
}

cv::Ptr<cv::ml::KNearest> trainKnn(
    const cv::Mat& features,
    const cv::Mat& labels,
    int k) {
    auto knn =
        cv::ml::KNearest::create();

    knn->setDefaultK(k);
    knn->setIsClassifier(true);
    knn->setAlgorithmType(
        cv::ml::KNearest::BRUTE_FORCE);

    if (!knn->train(
            features,
            cv::ml::ROW_SAMPLE,
            labels)) {
        throw std::runtime_error(
            "k-NN training failed.");
    }

    return knn;
}

cv::Ptr<cv::ml::SVM> trainSvm(
    const cv::Mat& features,
    const cv::Mat& labels) {
    auto svm =
        cv::ml::SVM::create();

    svm->setType(
        cv::ml::SVM::C_SVC);

    svm->setKernel(
        cv::ml::SVM::RBF);

    svm->setC(4.0);
    svm->setGamma(0.5);

    svm->setTermCriteria(
        cv::TermCriteria(
            cv::TermCriteria::COUNT |
            cv::TermCriteria::EPS,
            1000,
            1e-7));

    if (!svm->train(
            features,
            cv::ml::ROW_SAMPLE,
            labels)) {
        throw std::runtime_error(
            "SVM training failed.");
    }

    return svm;
}

Metrics computeMetrics(
    const cv::Mat& truth,
    const std::vector<int>& predicted) {
    Metrics metrics;

    metrics.confusion =
        cv::Mat::zeros(
            kClassCount,
            kClassCount,
            CV_32S);

    int correct = 0;

    for (int row = 0;
         row < truth.rows &&
         row <
             static_cast<int>(
                 predicted.size());
         ++row) {
        const int actual =
            truth.at<int>(
                row,
                0);

        const int estimate =
            predicted[
                static_cast<std::size_t>(
                    row)];

        if (actual >= 0 &&
            actual < kClassCount &&
            estimate >= 0 &&
            estimate < kClassCount) {
            metrics.confusion.at<int>(
                actual,
                estimate) += 1;
        }

        if (actual ==
            estimate) {
            ++correct;
        }
    }

    metrics.accuracy =
        truth.rows == 0
            ? 0.0
            : static_cast<double>(
                  correct) /
              truth.rows;

    for (int cls = 0;
         cls < kClassCount;
         ++cls) {
        const double tp =
            metrics.confusion.at<int>(
                cls,
                cls);

        double predicted_as_class = 0.0;
        double actual_class = 0.0;

        for (int i = 0;
             i < kClassCount;
             ++i) {
            predicted_as_class +=
                metrics.confusion.at<int>(
                    i,
                    cls);

            actual_class +=
                metrics.confusion.at<int>(
                    cls,
                    i);
        }

        metrics.precision[
            static_cast<std::size_t>(
                cls)] =
            predicted_as_class > 0.0
                ? tp /
                      predicted_as_class
                : 0.0;

        metrics.recall[
            static_cast<std::size_t>(
                cls)] =
            actual_class > 0.0
                ? tp /
                      actual_class
                : 0.0;
    }

    return metrics;
}

std::vector<int> predictKnn(
    const cv::Ptr<cv::ml::KNearest>& knn,
    const cv::Mat& features,
    int k) {
    std::vector<int> predicted;
    predicted.reserve(
        static_cast<std::size_t>(
            features.rows));

    for (int row = 0;
         row < features.rows;
         ++row) {
        cv::Mat results;
        cv::Mat neighbor_responses;
        cv::Mat distances;

        const float response =
            knn->findNearest(
                features.row(row),
                k,
                results,
                neighbor_responses,
                distances);

        predicted.push_back(
            cvRound(response));
    }

    return predicted;
}

std::vector<int> predictSvm(
    const cv::Ptr<cv::ml::SVM>& svm,
    const cv::Mat& features) {
    std::vector<int> predicted;
    predicted.reserve(
        static_cast<std::size_t>(
            features.rows));

    for (int row = 0;
         row < features.rows;
         ++row) {
        predicted.push_back(
            cvRound(
                svm->predict(
                    features.row(row))));
    }

    return predicted;
}

cv::Mat drawFeatureScatter(
    const cv::Mat& standardized,
    const cv::Mat& labels) {
    constexpr int width = 720;
    constexpr int height = 520;
    constexpr int margin = 60;

    cv::Mat image(
        height,
        width,
        CV_8UC3,
        cv::Scalar(
            245,
            245,
            245));

    // Plot standardized circularity (feature 2)
    // against standardized aspect ratio (feature 3).
    const int x_feature = 2;
    const int y_feature = 3;

    double max_abs = 1.0;

    for (int row = 0;
         row < standardized.rows;
         ++row) {
        max_abs =
            std::max(
                max_abs,
                std::abs(
                    static_cast<double>(
                        standardized.at<float>(
                            row,
                            x_feature))));

        max_abs =
            std::max(
                max_abs,
                std::abs(
                    static_cast<double>(
                        standardized.at<float>(
                            row,
                            y_feature))));
    }

    const auto mapX =
        [max_abs](double x) {
            return cvRound(
                margin +
                (x + max_abs) /
                    (2.0 * max_abs) *
                    (width -
                     2 * margin));
        };

    const auto mapY =
        [max_abs](double y) {
            return cvRound(
                height -
                margin -
                (y + max_abs) /
                    (2.0 * max_abs) *
                    (height -
                     2 * margin));
        };

    cv::line(
        image,
        cv::Point(
            mapX(-max_abs),
            mapY(0.0)),
        cv::Point(
            mapX(max_abs),
            mapY(0.0)),
        cv::Scalar(
            150,
            150,
            150),
        1,
        cv::LINE_AA);

    cv::line(
        image,
        cv::Point(
            mapX(0.0),
            mapY(-max_abs)),
        cv::Point(
            mapX(0.0),
            mapY(max_abs)),
        cv::Scalar(
            150,
            150,
            150),
        1,
        cv::LINE_AA);

    const std::array<cv::Scalar, kClassCount>
        colors = {
            cv::Scalar(50, 120, 240),
            cv::Scalar(70, 180, 70),
            cv::Scalar(220, 90, 90)
        };

    for (int row = 0;
         row < standardized.rows;
         ++row) {
        const int label =
            labels.at<int>(
                row,
                0);

        cv::circle(
            image,
            cv::Point(
                mapX(
                    standardized.at<float>(
                        row,
                        x_feature)),
                mapY(
                    standardized.at<float>(
                        row,
                        y_feature))),
            4,
            colors[
                static_cast<std::size_t>(
                    std::clamp(
                        label,
                        0,
                        kClassCount - 1))],
            cv::FILLED,
            cv::LINE_AA);
    }

    cv::putText(
        image,
        "standardized circularity",
        cv::Point(
            width / 2 - 100,
            height - 18),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(30, 30, 30),
        1,
        cv::LINE_AA);

    cv::putText(
        image,
        "standardized aspect ratio",
        cv::Point(12, 28),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(30, 30, 30),
        1,
        cv::LINE_AA);

    return image;
}

cv::Mat drawConfusion(
    const Metrics& metrics,
    const std::string& title) {
    constexpr int cell = 120;
    constexpr int left = 130;
    constexpr int top = 85;

    cv::Mat image(
        top +
            kClassCount * cell +
            70,
        left +
            kClassCount * cell +
            30,
        CV_8UC3,
        cv::Scalar(
            250,
            250,
            250));

    cv::putText(
        image,
        title,
        cv::Point(20, 32),
        cv::FONT_HERSHEY_SIMPLEX,
        0.75,
        cv::Scalar(20, 20, 20),
        2,
        cv::LINE_AA);

    for (int cls = 0;
         cls < kClassCount;
         ++cls) {
        cv::putText(
            image,
            className(cls),
            cv::Point(
                left +
                    cls * cell +
                    10,
                top - 16),
            cv::FONT_HERSHEY_SIMPLEX,
            0.48,
            cv::Scalar(30, 30, 30),
            1,
            cv::LINE_AA);

        cv::putText(
            image,
            className(cls),
            cv::Point(
                10,
                top +
                    cls * cell +
                    cell / 2),
            cv::FONT_HERSHEY_SIMPLEX,
            0.48,
            cv::Scalar(30, 30, 30),
            1,
            cv::LINE_AA);
    }

    for (int actual = 0;
         actual < kClassCount;
         ++actual) {
        for (int predicted = 0;
             predicted < kClassCount;
             ++predicted) {
            const cv::Rect box(
                left +
                    predicted * cell,
                top +
                    actual * cell,
                cell,
                cell);

            const int count =
                metrics.confusion.at<int>(
                    actual,
                    predicted);

            const bool diagonal =
                actual ==
                predicted;

            cv::rectangle(
                image,
                box,
                diagonal
                    ? cv::Scalar(
                          220,
                          245,
                          220)
                    : cv::Scalar(
                          235,
                          235,
                          245),
                cv::FILLED);

            cv::rectangle(
                image,
                box,
                cv::Scalar(
                    120,
                    120,
                    120),
                1);

            cv::putText(
                image,
                std::to_string(
                    count),
                cv::Point(
                    box.x +
                        cell / 2 -
                        12,
                    box.y +
                        cell / 2 +
                        8),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(20, 20, 20),
                2,
                cv::LINE_AA);
        }
    }

    cv::putText(
        image,
        "rows=true, cols=predicted  accuracy=" +
            cv::format(
                "%.3f",
                metrics.accuracy),
        cv::Point(
            20,
            image.rows - 20),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(20, 20, 20),
        1,
        cv::LINE_AA);

    return image;
}

bool writeCsv(
    const fs::path& path,
    const cv::Mat& original,
    const cv::Mat& standardized,
    const cv::Mat& labels,
    const std::vector<int>& knn_pred,
    const std::vector<int>& svm_pred) {
    std::ofstream output(path);

    if (!output) {
        return false;
    }

    output
        << "label,knn_prediction,svm_prediction,"
        << "area_norm,perimeter_norm,circularity,aspect_ratio,"
        << "extent,solidity,hu1,"
        << "z_area,z_perimeter,z_circularity,z_aspect,"
        << "z_extent,z_solidity,z_hu1\n";

    output
        << std::fixed
        << std::setprecision(7);

    for (int row = 0;
         row < original.rows;
         ++row) {
        output
            << labels.at<int>(
                   row,
                   0)
            << ","
            << knn_pred[
                   static_cast<std::size_t>(
                       row)]
            << ","
            << svm_pred[
                   static_cast<std::size_t>(
                       row)];

        for (int col = 0;
             col < original.cols;
             ++col) {
            output
                << ","
                << original.at<float>(
                       row,
                       col);
        }

        for (int col = 0;
             col < standardized.cols;
             ++col) {
            output
                << ","
                << standardized.at<float>(
                       row,
                       col);
        }

        output << "\n";
    }

    return true;
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

cv::Mat makeExampleStrip(
    const std::vector<cv::Mat>& examples) {
    cv::Mat strip(
        kImageSize + 50,
        kClassCount *
            (kImageSize + 20),
        CV_8UC3,
        cv::Scalar(
            245,
            245,
            245));

    for (int cls = 0;
         cls < kClassCount &&
         cls <
             static_cast<int>(
                 examples.size());
         ++cls) {
        cv::Mat bgr;
        cv::cvtColor(
            examples[
                static_cast<std::size_t>(
                    cls)],
            bgr,
            cv::COLOR_GRAY2BGR);

        const int x =
            cls *
            (kImageSize + 20);

        bgr.copyTo(
            strip(
                cv::Rect(
                    x,
                    0,
                    kImageSize,
                    kImageSize)));

        cv::putText(
            strip,
            className(cls),
            cv::Point(
                x,
                kImageSize + 28),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            cv::Scalar(20, 20, 20),
            1,
            cv::LINE_AA);
    }

    return strip;
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

    const Dataset train =
        generateDataset(
            90,
            101,
            false);

    const Dataset test =
        generateDataset(
            30,
            202,
            false);

    check(
        train.features.rows ==
            270 &&
        train.features.cols ==
            kFeatureCount,
        "training dataset has expected shape");

    check(
        test.features.rows ==
            90,
        "test dataset has expected sample count");

    const StandardScaler scaler =
        fitScaler(
            train.features);

    const cv::Mat train_scaled =
        applyScaler(
            train.features,
            scaler);

    const cv::Mat test_scaled =
        applyScaler(
            test.features,
            scaler);

    for (int col = 0;
         col < train_scaled.cols;
         ++col) {
        double mean = 0.0;

        for (int row = 0;
             row < train_scaled.rows;
             ++row) {
            mean +=
                train_scaled.at<float>(
                    row,
                    col);
        }

        mean /=
            train_scaled.rows;

        check(
            std::abs(mean) <
                1e-4,
            "standardized training feature mean is near zero");
    }

    const auto knn =
        trainKnn(
            train_scaled,
            train.labels_float,
            5);

    const auto svm =
        trainSvm(
            train_scaled,
            train.labels_int);

    check(
        !knn.empty() &&
        knn->isTrained(),
        "k-NN trains successfully");

    check(
        !svm.empty() &&
        svm->isTrained(),
        "SVM trains successfully");

    const auto knn_predictions =
        predictKnn(
            knn,
            test_scaled,
            5);

    const auto svm_predictions =
        predictSvm(
            svm,
            test_scaled);

    const Metrics knn_metrics =
        computeMetrics(
            test.labels_int,
            knn_predictions);

    const Metrics svm_metrics =
        computeMetrics(
            test.labels_int,
            svm_predictions);

    check(
        knn_metrics.accuracy >=
            0.95,
        "k-NN test accuracy is at least 95 percent");

    check(
        svm_metrics.accuracy >=
            0.95,
        "SVM test accuracy is at least 95 percent");

    int knn_total = 0;
    int svm_total = 0;

    for (int row = 0;
         row < kClassCount;
         ++row) {
        for (int col = 0;
             col < kClassCount;
             ++col) {
            knn_total +=
                knn_metrics.confusion.at<int>(
                    row,
                    col);

            svm_total +=
                svm_metrics.confusion.at<int>(
                    row,
                    col);
        }
    }

    check(
        knn_total ==
            test.features.rows,
        "k-NN confusion matrix accounts for every test sample");

    check(
        svm_total ==
            test.features.rows,
        "SVM confusion matrix accounts for every test sample");

    for (int cls = 0;
         cls < kClassCount;
         ++cls) {
        check(
            knn_metrics.recall[
                static_cast<std::size_t>(
                    cls)] >=
                0.90,
            "k-NN class recall is at least 90 percent");

        check(
            svm_metrics.recall[
                static_cast<std::size_t>(
                    cls)] >=
                0.90,
            "SVM class recall is at least 90 percent");
    }

    if (passed) {
        std::cout
            << "Classical classification self-test passed.\n";
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
        return runSelfTest()
            ? 0
            : 1;
    }

    const Dataset train =
        generateDataset(
            options.train_per_class,
            options.seed,
            true);

    const Dataset test =
        generateDataset(
            options.test_per_class,
            options.seed + 1000U,
            false);

    const StandardScaler scaler =
        fitScaler(
            train.features);

    const cv::Mat train_scaled =
        applyScaler(
            train.features,
            scaler);

    const cv::Mat test_scaled =
        applyScaler(
            test.features,
            scaler);

    const auto knn =
        trainKnn(
            train_scaled,
            train.labels_float,
            options.knn_k);

    const auto svm =
        trainSvm(
            train_scaled,
            train.labels_int);

    const auto knn_predictions =
        predictKnn(
            knn,
            test_scaled,
            options.knn_k);

    const auto svm_predictions =
        predictSvm(
            svm,
            test_scaled);

    const Metrics knn_metrics =
        computeMetrics(
            test.labels_int,
            knn_predictions);

    const Metrics svm_metrics =
        computeMetrics(
            test.labels_int,
            svm_predictions);

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
            "01_class_examples.png",
        makeExampleStrip(
            train.examples));

    ok &= writeImage(
        output_dir /
            "02_feature_scatter.png",
        drawFeatureScatter(
            train_scaled,
            train.labels_int));

    ok &= writeImage(
        output_dir /
            "03_knn_confusion.png",
        drawConfusion(
            knn_metrics,
            "k-NN confusion matrix"));

    ok &= writeImage(
        output_dir /
            "04_svm_confusion.png",
        drawConfusion(
            svm_metrics,
            "RBF-SVM confusion matrix"));

    ok &= writeCsv(
        output_dir /
            "test_predictions.csv",
        test.features,
        test_scaled,
        test.labels_int,
        knn_predictions,
        svm_predictions);

    if (!ok) {
        std::cerr
            << "Error: failed to write one or more outputs.\n";
        return 1;
    }

    std::cout
        << std::fixed
        << std::setprecision(4)
        << "Feature pipeline\n"
        << "  feature count: "
        << kFeatureCount
        << '\n'
        << "  features: area_norm, perimeter_norm, circularity, "
        << "aspect_ratio, extent, solidity, Hu1\n"
        << "  normalization: training-set z-score\n"
        << "Dataset\n"
        << "  train per class: "
        << options.train_per_class
        << '\n'
        << "  test per class: "
        << options.test_per_class
        << '\n'
        << "k-NN\n"
        << "  k: "
        << options.knn_k
        << '\n'
        << "  accuracy: "
        << knn_metrics.accuracy
        << '\n'
        << "RBF SVM\n"
        << "  C: 4.0\n"
        << "  gamma: 0.5\n"
        << "  accuracy: "
        << svm_metrics.accuracy
        << '\n';

    for (int cls = 0;
         cls < kClassCount;
         ++cls) {
        std::cout
            << "Class "
            << className(cls)
            << "\n"
            << "  kNN precision="
            << knn_metrics.precision[
                   static_cast<std::size_t>(
                       cls)]
            << " recall="
            << knn_metrics.recall[
                   static_cast<std::size_t>(
                       cls)]
            << '\n'
            << "  SVM precision="
            << svm_metrics.precision[
                   static_cast<std::size_t>(
                       cls)]
            << " recall="
            << svm_metrics.recall[
                   static_cast<std::size_t>(
                       cls)]
            << '\n';
    }

    std::cout
        << "Outputs written to: "
        << output_dir
        << '\n';

    return 0;
}
