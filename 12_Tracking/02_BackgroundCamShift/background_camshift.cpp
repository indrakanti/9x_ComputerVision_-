#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Detection {
    cv::Rect box;
    double score = 1.0;
    int class_id = 0;
};

struct Options {
    std::string video_path;
    std::string output_dir;
    cv::Rect initial_roi;
    bool has_roi = false;
    int max_frames = 240;
    int history = 80;
    double var_threshold = 16.0;
    int min_area = 250;
    int morph_kernel = 3;
    double learning_rate = -1.0;
    bool detect_shadows = true;
    bool self_test = false;
    bool help = false;
};

struct SyntheticDetectionSequence {
    std::vector<cv::Mat> frames;
    std::vector<std::vector<cv::Rect>> truth;
    int warmup_frames = 15;
};

struct SyntheticColorSequence {
    std::vector<cv::Mat> frames;
    std::vector<cv::Rect> truth;
    cv::Rect initial_roi;
};

struct Mog2FrameResult {
    cv::Mat raw_mask;
    cv::Mat clean_mask;
    std::vector<Detection> detections;
};

struct TrackingMetrics {
    double mean_center_error = 0.0;
    double mean_iou = 0.0;
    double final_iou = 0.0;
};

std::vector<std::string> split(const std::string& text, char delimiter) {
    std::vector<std::string> values;
    std::string current;

    for (char ch : text) {
        if (ch == delimiter) {
            values.push_back(current);
            current.clear();
        } else {
            current.push_back(ch);
        }
    }

    values.push_back(current);
    return values;
}

cv::Rect parseRoi(const std::string& text) {
    const auto parts = split(text, ',');

    if (parts.size() != 4U) {
        throw std::runtime_error("ROI must be x,y,w,h");
    }

    return cv::Rect(
        std::stoi(parts[0]),
        std::stoi(parts[1]),
        std::stoi(parts[2]),
        std::stoi(parts[3]));
}

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " --output-dir <dir> [options]\n"
        << "  " << program << " --video <path> --output-dir <dir> [options]\n"
        << "  " << program << " --self-test\n\n"
        << "Options:\n"
        << "  --video <path>            Optional input video\n"
        << "  --roi x,y,w,h             Optional initial ROI for MeanShift/CamShift\n"
        << "  --output-dir <path>       Output directory\n"
        << "  --max-frames <N>          Maximum frames to process (default: 240)\n"
        << "  --history <N>             MOG2 history length (default: 80)\n"
        << "  --var-threshold <value>   MOG2 variance threshold (default: 16)\n"
        << "  --min-area <pixels>       Minimum connected-component area (default: 250)\n"
        << "  --morph-kernel <odd N>    Morphology kernel size (default: 3)\n"
        << "  --learning-rate <value>   MOG2 learning rate; -1 = automatic\n"
        << "  --no-shadows              Disable MOG2 shadow detection\n"
        << "  --self-test               Run deterministic tests\n"
        << "  --help                    Show this help\n";
}

bool parseArguments(int argc, char** argv, Options& options) {
    auto value = [argc, argv](int& i, const std::string& name) {
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value for " + name);
        }
        return std::string(argv[++i]);
    };

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];

            if (arg == "--video") {
                options.video_path = value(i, arg);
            } else if (arg == "--roi") {
                options.initial_roi = parseRoi(value(i, arg));
                options.has_roi = true;
            } else if (arg == "--output-dir") {
                options.output_dir = value(i, arg);
            } else if (arg == "--max-frames") {
                options.max_frames = std::stoi(value(i, arg));
            } else if (arg == "--history") {
                options.history = std::stoi(value(i, arg));
            } else if (arg == "--var-threshold") {
                options.var_threshold = std::stod(value(i, arg));
            } else if (arg == "--min-area") {
                options.min_area = std::stoi(value(i, arg));
            } else if (arg == "--morph-kernel") {
                options.morph_kernel = std::stoi(value(i, arg));
            } else if (arg == "--learning-rate") {
                options.learning_rate = std::stod(value(i, arg));
            } else if (arg == "--no-shadows") {
                options.detect_shadows = false;
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

    if (options.max_frames < 1 ||
        options.history < 1 ||
        options.var_threshold <= 0.0 ||
        options.min_area < 1 ||
        options.morph_kernel < 1 ||
        options.morph_kernel % 2 == 0 ||
        options.learning_rate < -1.0 ||
        options.learning_rate > 1.0) {
        std::cerr << "Error: invalid MOG2/blob configuration.\n";
        return false;
    }

    if (options.has_roi &&
        (options.initial_roi.width < 2 ||
         options.initial_roi.height < 2)) {
        std::cerr << "Error: ROI must have positive size.\n";
        return false;
    }

    return true;
}

cv::Rect clipRect(const cv::Rect& box, cv::Size size) {
    return box & cv::Rect(0, 0, size.width, size.height);
}

double intersectionOverUnion(const cv::Rect& a, const cv::Rect& b) {
    const cv::Rect intersection = a & b;
    const double intersection_area =
        static_cast<double>(
            std::max(0, intersection.width) *
            std::max(0, intersection.height));

    const double union_area =
        static_cast<double>(a.area()) +
        static_cast<double>(b.area()) -
        intersection_area;

    if (union_area <= 0.0) {
        return 0.0;
    }

    return intersection_area / union_area;
}

double centerError(const cv::Rect& a, const cv::Rect& b) {
    const cv::Point2d ca(
        a.x + 0.5 * a.width,
        a.y + 0.5 * a.height);

    const cv::Point2d cb(
        b.x + 0.5 * b.width,
        b.y + 0.5 * b.height);

    return cv::norm(ca - cb);
}

cv::Mat cleanForegroundMask(const cv::Mat& raw_mask, int kernel_size) {
    cv::Mat binary;

    // MOG2 shadow pixels are commonly 127. Keep only strong foreground.
    cv::threshold(
        raw_mask,
        binary,
        200,
        255,
        cv::THRESH_BINARY);

    const cv::Mat kernel =
        cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(kernel_size, kernel_size));

    cv::morphologyEx(
        binary,
        binary,
        cv::MORPH_OPEN,
        kernel);

    cv::morphologyEx(
        binary,
        binary,
        cv::MORPH_CLOSE,
        kernel);

    return binary;
}

std::vector<Detection> extractBlobs(
    const cv::Mat& clean_mask,
    int min_area) {
    CV_Assert(clean_mask.type() == CV_8UC1);

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;

    const int component_count =
        cv::connectedComponentsWithStats(
            clean_mask,
            labels,
            stats,
            centroids,
            8,
            CV_32S);

    std::vector<Detection> detections;

    for (int label = 1;
         label < component_count;
         ++label) {
        const int area =
            stats.at<int>(
                label,
                cv::CC_STAT_AREA);

        if (area < min_area) {
            continue;
        }

        const int x =
            stats.at<int>(
                label,
                cv::CC_STAT_LEFT);

        const int y =
            stats.at<int>(
                label,
                cv::CC_STAT_TOP);

        const int width =
            stats.at<int>(
                label,
                cv::CC_STAT_WIDTH);

        const int height =
            stats.at<int>(
                label,
                cv::CC_STAT_HEIGHT);

        if (width < 3 || height < 3) {
            continue;
        }

        Detection detection;
        detection.box =
            cv::Rect(
                x,
                y,
                width,
                height);

        const double box_area =
            static_cast<double>(
                width * height);

        detection.score =
            box_area > 0.0
                ? static_cast<double>(area) /
                      box_area
                : 0.0;

        detections.push_back(
            detection);
    }

    std::sort(
        detections.begin(),
        detections.end(),
        [](const Detection& a, const Detection& b) {
            if (a.box.x != b.box.x) {
                return a.box.x < b.box.x;
            }
            return a.box.y < b.box.y;
        });

    return detections;
}

Mog2FrameResult processMog2Frame(
    const cv::Mat& frame,
    cv::Ptr<cv::BackgroundSubtractorMOG2>& mog2,
    int morph_kernel,
    int min_area,
    double learning_rate) {
    Mog2FrameResult result;

    mog2->apply(
        frame,
        result.raw_mask,
        learning_rate);

    result.clean_mask =
        cleanForegroundMask(
            result.raw_mask,
            morph_kernel);

    result.detections =
        extractBlobs(
            result.clean_mask,
            min_area);

    return result;
}

SyntheticDetectionSequence makeSyntheticDetectionSequence(
    int frame_count,
    int warmup_frames) {
    SyntheticDetectionSequence sequence;
    sequence.warmup_frames = warmup_frames;

    const cv::Size frame_size(
        480,
        320);

    cv::Mat background(
        frame_size,
        CV_8UC3,
        cv::Scalar(25, 28, 32));

    for (int x = 0; x < frame_size.width; x += 40) {
        cv::line(
            background,
            cv::Point(x, 0),
            cv::Point(x, frame_size.height - 1),
            cv::Scalar(31, 34, 38),
            1,
            cv::LINE_8);
    }

    for (int y = 0; y < frame_size.height; y += 40) {
        cv::line(
            background,
            cv::Point(0, y),
            cv::Point(frame_size.width - 1, y),
            cv::Scalar(31, 34, 38),
            1,
            cv::LINE_8);
    }

    for (int frame_index = 0;
         frame_index < frame_count;
         ++frame_index) {
        cv::Mat frame =
            background.clone();

        std::vector<cv::Rect> truth;

        if (frame_index >= warmup_frames) {
            const int t =
                frame_index - warmup_frames;

            const cv::Rect object_a(
                35 + 3 * t,
                75 + t / 3,
                42,
                34);

            const cv::Rect object_b(
                390 - 2 * t,
                200 - t / 4,
                36,
                44);

            if (clipRect(
                    object_a,
                    frame.size()) ==
                object_a) {
                cv::rectangle(
                    frame,
                    object_a,
                    cv::Scalar(40, 220, 240),
                    cv::FILLED,
                    cv::LINE_8);

                truth.push_back(
                    object_a);
            }

            if (clipRect(
                    object_b,
                    frame.size()) ==
                object_b) {
                cv::rectangle(
                    frame,
                    object_b,
                    cv::Scalar(220, 80, 60),
                    cv::FILLED,
                    cv::LINE_8);

                truth.push_back(
                    object_b);
            }
        }

        sequence.frames.push_back(
            frame);

        sequence.truth.push_back(
            truth);
    }

    return sequence;
}

SyntheticColorSequence makeSyntheticColorSequence(
    int frame_count) {
    SyntheticColorSequence sequence;

    const cv::Size frame_size(
        360,
        240);

    for (int frame_index = 0;
         frame_index < frame_count;
         ++frame_index) {
        cv::Mat frame(
            frame_size,
            CV_8UC3,
            cv::Scalar(18, 18, 18));

        const double scale =
            1.0 +
            0.012 *
                static_cast<double>(
                    frame_index);

        const int width =
            cvRound(42.0 * scale);

        const int height =
            cvRound(30.0 * scale);

        const cv::Rect truth(
            45 + 3 * frame_index,
            85 + frame_index / 2,
            width,
            height);

        cv::rectangle(
            frame,
            truth,
            cv::Scalar(40, 220, 40),
            cv::FILLED,
            cv::LINE_8);

        // Distractor of different hue.
        cv::circle(
            frame,
            cv::Point(
                280,
                45 + frame_index / 4),
            18,
            cv::Scalar(220, 80, 60),
            cv::FILLED,
            cv::LINE_8);

        sequence.frames.push_back(
            frame);

        sequence.truth.push_back(
            truth);
    }

    sequence.initial_roi =
        sequence.truth.front();

    return sequence;
}

cv::Mat hueHistogram(
    const cv::Mat& frame,
    const cv::Rect& roi) {
    CV_Assert(
        clipRect(
            roi,
            frame.size()) ==
        roi);

    cv::Mat hsv;
    cv::cvtColor(
        frame,
        hsv,
        cv::COLOR_BGR2HSV);

    const cv::Mat hsv_roi =
        hsv(roi);

    cv::Mat mask;

    cv::inRange(
        hsv_roi,
        cv::Scalar(0, 60, 32),
        cv::Scalar(180, 255, 255),
        mask);

    int channels[] = {0};
    int hist_size[] = {180};
    float hue_range[] = {
        0.0F,
        180.0F
    };
    const float* ranges[] = {
        hue_range
    };

    cv::Mat histogram;

    cv::calcHist(
        &hsv_roi,
        1,
        channels,
        mask,
        histogram,
        1,
        hist_size,
        ranges,
        true,
        false);

    cv::normalize(
        histogram,
        histogram,
        0.0,
        255.0,
        cv::NORM_MINMAX);

    return histogram;
}

cv::Mat hueBackProjection(
    const cv::Mat& frame,
    const cv::Mat& histogram) {
    cv::Mat hsv;

    cv::cvtColor(
        frame,
        hsv,
        cv::COLOR_BGR2HSV);

    int channels[] = {0};
    float hue_range[] = {
        0.0F,
        180.0F
    };
    const float* ranges[] = {
        hue_range
    };

    cv::Mat back_projection;

    cv::calcBackProject(
        &hsv,
        1,
        channels,
        histogram,
        back_projection,
        ranges,
        1.0,
        true);

    cv::Mat saturation_mask;

    cv::inRange(
        hsv,
        cv::Scalar(0, 60, 32),
        cv::Scalar(180, 255, 255),
        saturation_mask);

    cv::bitwise_and(
        back_projection,
        saturation_mask,
        back_projection);

    return back_projection;
}

std::vector<cv::Rect> trackMeanShift(
    const std::vector<cv::Mat>& frames,
    const cv::Rect& initial_roi,
    const cv::Mat& histogram) {
    std::vector<cv::Rect> boxes;

    if (frames.empty()) {
        return boxes;
    }

    cv::Rect window =
        initial_roi;

    boxes.push_back(
        window);

    const cv::TermCriteria criteria(
        cv::TermCriteria::EPS |
            cv::TermCriteria::COUNT,
        20,
        0.5);

    for (std::size_t i = 1;
         i < frames.size();
         ++i) {
        const cv::Mat back_projection =
            hueBackProjection(
                frames[i],
                histogram);

        cv::meanShift(
            back_projection,
            window,
            criteria);

        window =
            clipRect(
                window,
                frames[i].size());

        boxes.push_back(
            window);
    }

    return boxes;
}

std::vector<cv::Rect> trackCamShift(
    const std::vector<cv::Mat>& frames,
    const cv::Rect& initial_roi,
    const cv::Mat& histogram) {
    std::vector<cv::Rect> boxes;

    if (frames.empty()) {
        return boxes;
    }

    cv::Rect window =
        initial_roi;

    boxes.push_back(
        window);

    const cv::TermCriteria criteria(
        cv::TermCriteria::EPS |
            cv::TermCriteria::COUNT,
        20,
        0.5);

    for (std::size_t i = 1;
         i < frames.size();
         ++i) {
        const cv::Mat back_projection =
            hueBackProjection(
                frames[i],
                histogram);

        const cv::RotatedRect rotated =
            cv::CamShift(
                back_projection,
                window,
                criteria);

        cv::Rect next =
            rotated.boundingRect();

        next =
            clipRect(
                next,
                frames[i].size());

        if (next.width >= 2 &&
            next.height >= 2) {
            window = next;
        }

        boxes.push_back(
            window);
    }

    return boxes;
}

TrackingMetrics computeTrackingMetrics(
    const std::vector<cv::Rect>& predicted,
    const std::vector<cv::Rect>& truth) {
    TrackingMetrics metrics;

    const std::size_t count =
        std::min(
            predicted.size(),
            truth.size());

    if (count == 0U) {
        return metrics;
    }

    double center_sum = 0.0;
    double iou_sum = 0.0;

    for (std::size_t i = 0;
         i < count;
         ++i) {
        center_sum +=
            centerError(
                predicted[i],
                truth[i]);

        iou_sum +=
            intersectionOverUnion(
                predicted[i],
                truth[i]);
    }

    metrics.mean_center_error =
        center_sum /
        static_cast<double>(count);

    metrics.mean_iou =
        iou_sum /
        static_cast<double>(count);

    metrics.final_iou =
        intersectionOverUnion(
            predicted[count - 1],
            truth[count - 1]);

    return metrics;
}

bool writeDetectionsCsv(
    const fs::path& path,
    const std::vector<std::vector<Detection>>& detections) {
    std::ofstream output(path);

    if (!output) {
        return false;
    }

    output
        << "frame,x,y,w,h,score,class_id\n"
        << std::fixed
        << std::setprecision(6);

    for (std::size_t frame = 0;
         frame < detections.size();
         ++frame) {
        for (const auto& detection :
             detections[frame]) {
            output
                << frame << ","
                << detection.box.x << ","
                << detection.box.y << ","
                << detection.box.width << ","
                << detection.box.height << ","
                << detection.score << ","
                << detection.class_id << "\n";
        }
    }

    return true;
}

bool writeTrackingMetricsCsv(
    const fs::path& path,
    const TrackingMetrics& mean_shift,
    const TrackingMetrics& cam_shift) {
    std::ofstream output(path);

    if (!output) {
        return false;
    }

    output
        << "tracker,mean_center_error,mean_iou,final_iou\n"
        << std::fixed
        << std::setprecision(6)
        << "meanshift,"
        << mean_shift.mean_center_error << ","
        << mean_shift.mean_iou << ","
        << mean_shift.final_iou << "\n"
        << "camshift,"
        << cam_shift.mean_center_error << ","
        << cam_shift.mean_iou << ","
        << cam_shift.final_iou << "\n";

    return true;
}

cv::Mat drawBlobOverlay(
    const cv::Mat& frame,
    const std::vector<Detection>& detections) {
    cv::Mat overlay =
        frame.clone();

    for (const auto& detection :
         detections) {
        cv::rectangle(
            overlay,
            detection.box,
            cv::Scalar(0, 255, 255),
            2,
            cv::LINE_AA);
    }

    return overlay;
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

    {
        cv::Mat raw(
            80,
            100,
            CV_8UC1,
            cv::Scalar(0));

        cv::rectangle(
            raw,
            cv::Rect(20, 20, 30, 25),
            cv::Scalar(255),
            cv::FILLED);

        raw.at<unsigned char>(3, 3) = 255;
        raw.at<unsigned char>(70, 90) = 255;

        const cv::Mat clean =
            cleanForegroundMask(
                raw,
                3);

        const auto blobs =
            extractBlobs(
                clean,
                100);

        check(
            blobs.size() == 1U,
            "morphology/blob filtering rejects isolated speckles");

        if (!blobs.empty()) {
            check(
                intersectionOverUnion(
                    blobs.front().box,
                    cv::Rect(
                        20,
                        20,
                        30,
                        25)) >=
                    0.90,
                "blob extraction recovers the expected foreground box");
        }
    }

    {
        const auto sequence =
            makeSyntheticDetectionSequence(
                30,
                15);

        auto mog2 =
            cv::createBackgroundSubtractorMOG2(
                50,
                16.0,
                false);

        Mog2FrameResult last;

        for (int i = 0; i < 15; ++i) {
            last =
                processMog2Frame(
                    sequence.frames[
                        static_cast<std::size_t>(i)],
                    mog2,
                    3,
                    200,
                    0.5);
        }

        last =
            processMog2Frame(
                sequence.frames[15],
                mog2,
                3,
                200,
                0.0);

        check(
            last.detections.size() >= 2U,
            "MOG2 detects both newly appearing moving objects after background warmup");

        double best_iou = 0.0;

        for (const auto& detection :
             last.detections) {
            best_iou =
                std::max(
                    best_iou,
                    intersectionOverUnion(
                        detection.box,
                        sequence.truth[15][0]));
        }

        check(
            best_iou >= 0.80,
            "MOG2 blob localization has high overlap with synthetic truth");
    }

    const auto color_sequence =
        makeSyntheticColorSequence(36);

    const cv::Mat histogram =
        hueHistogram(
            color_sequence.frames.front(),
            color_sequence.initial_roi);

    check(
        !histogram.empty() &&
        histogram.rows == 180,
        "HSV hue histogram has the expected 180 bins");

    const cv::Mat back_projection =
        hueBackProjection(
            color_sequence.frames.front(),
            histogram);

    const double inside_mean =
        cv::mean(
            back_projection(
                color_sequence.initial_roi))[0];

    const cv::Rect background_roi(
        5,
        5,
        30,
        30);

    const double background_mean =
        cv::mean(
            back_projection(
                background_roi))[0];

    check(
        inside_mean >
            background_mean +
                100.0,
        "histogram back-projection strongly separates target hue from background");

    const auto mean_shift_boxes =
        trackMeanShift(
            color_sequence.frames,
            color_sequence.initial_roi,
            histogram);

    const auto cam_shift_boxes =
        trackCamShift(
            color_sequence.frames,
            color_sequence.initial_roi,
            histogram);

    const TrackingMetrics mean_metrics =
        computeTrackingMetrics(
            mean_shift_boxes,
            color_sequence.truth);

    const TrackingMetrics cam_metrics =
        computeTrackingMetrics(
            cam_shift_boxes,
            color_sequence.truth);

    check(
        mean_metrics.mean_center_error <=
            4.0,
        "MeanShift follows the translating target with bounded center error");

    check(
        cam_metrics.mean_center_error <=
            4.0,
        "CamShift follows the translating target with bounded center error");

    check(
        cam_metrics.mean_iou >=
            0.65,
        "CamShift maintains useful overlap while target scale changes");

    check(
        cam_shift_boxes.back().width >
            cam_shift_boxes.front().width,
        "CamShift adapts its search window as target scale grows");

    if (passed) {
        std::cout
            << "MOG2 / MeanShift / CamShift self-test passed.\n"
            << std::fixed
            << std::setprecision(4)
            << "MeanShift center error="
            << mean_metrics.mean_center_error
            << " meanIoU="
            << mean_metrics.mean_iou
            << "\nCamShift center error="
            << cam_metrics.mean_center_error
            << " meanIoU="
            << cam_metrics.mean_iou
            << " finalIoU="
            << cam_metrics.final_iou
            << '\n';
    }

    return passed;
}

int runSyntheticDemo(
    const Options& options,
    const fs::path& output_dir) {
    const auto detection_sequence =
        makeSyntheticDetectionSequence(
            90,
            15);

    auto mog2 =
        cv::createBackgroundSubtractorMOG2(
            options.history,
            options.var_threshold,
            options.detect_shadows);

    std::vector<std::vector<Detection>>
        all_detections(
            detection_sequence.frames.size());

    cv::VideoWriter blob_writer(
        (output_dir /
         "mog2_blob_overlay.avi")
            .string(),
        cv::VideoWriter::fourcc(
            'M',
            'J',
            'P',
            'G'),
        20.0,
        detection_sequence.frames.front().size(),
        true);

    for (std::size_t frame = 0;
         frame <
             detection_sequence.frames.size();
         ++frame) {
        const double learning_rate =
            frame <
                    static_cast<std::size_t>(
                        detection_sequence.warmup_frames)
                ? 0.5
                : 0.0;

        Mog2FrameResult result =
            processMog2Frame(
                detection_sequence.frames[frame],
                mog2,
                options.morph_kernel,
                options.min_area,
                learning_rate);

        all_detections[frame] =
            result.detections;

        if (frame ==
            static_cast<std::size_t>(
                detection_sequence.warmup_frames)) {
            cv::imwrite(
                (output_dir /
                 "01_background_model_input.png")
                    .string(),
                detection_sequence.frames[
                    frame - 1]);

            cv::imwrite(
                (output_dir /
                 "02_foreground_raw.png")
                    .string(),
                result.raw_mask);

            cv::imwrite(
                (output_dir /
                 "03_foreground_clean.png")
                    .string(),
                result.clean_mask);

            cv::imwrite(
                (output_dir /
                 "04_blob_overlay.png")
                    .string(),
                drawBlobOverlay(
                    detection_sequence.frames[frame],
                    result.detections));
        }

        if (blob_writer.isOpened()) {
            blob_writer.write(
                drawBlobOverlay(
                    detection_sequence.frames[frame],
                    result.detections));
        }
    }

    if (!writeDetectionsCsv(
            output_dir /
                "mog2_detections.csv",
            all_detections)) {
        std::cerr
            << "Error: failed to write MOG2 detections CSV.\n";
        return 1;
    }

    const auto color_sequence =
        makeSyntheticColorSequence(55);

    const cv::Mat histogram =
        hueHistogram(
            color_sequence.frames.front(),
            color_sequence.initial_roi);

    const auto mean_shift_boxes =
        trackMeanShift(
            color_sequence.frames,
            color_sequence.initial_roi,
            histogram);

    const auto cam_shift_boxes =
        trackCamShift(
            color_sequence.frames,
            color_sequence.initial_roi,
            histogram);

    const TrackingMetrics mean_metrics =
        computeTrackingMetrics(
            mean_shift_boxes,
            color_sequence.truth);

    const TrackingMetrics cam_metrics =
        computeTrackingMetrics(
            cam_shift_boxes,
            color_sequence.truth);

    writeTrackingMetricsCsv(
        output_dir /
            "appearance_tracking_metrics.csv",
        mean_metrics,
        cam_metrics);

    cv::VideoWriter appearance_writer(
        (output_dir /
         "meanshift_camshift_overlay.avi")
            .string(),
        cv::VideoWriter::fourcc(
            'M',
            'J',
            'P',
            'G'),
        20.0,
        color_sequence.frames.front().size(),
        true);

    for (std::size_t frame = 0;
         frame < color_sequence.frames.size();
         ++frame) {
        cv::Mat overlay =
            color_sequence.frames[frame].clone();

        cv::rectangle(
            overlay,
            color_sequence.truth[frame],
            cv::Scalar(140, 140, 140),
            1,
            cv::LINE_AA);

        cv::rectangle(
            overlay,
            mean_shift_boxes[frame],
            cv::Scalar(0, 220, 255),
            2,
            cv::LINE_AA);

        cv::rectangle(
            overlay,
            cam_shift_boxes[frame],
            cv::Scalar(255, 120, 0),
            2,
            cv::LINE_AA);

        cv::putText(
            overlay,
            "gray=truth yellow=MeanShift blue=CamShift",
            cv::Point(8, 20),
            cv::FONT_HERSHEY_SIMPLEX,
            0.45,
            cv::Scalar(235, 235, 235),
            1,
            cv::LINE_AA);

        if (appearance_writer.isOpened()) {
            appearance_writer.write(
                overlay);
        }
    }

    cv::imwrite(
        (output_dir /
         "05_hue_backprojection.png")
            .string(),
        hueBackProjection(
            color_sequence.frames.front(),
            histogram));

    std::cout
        << "Synthetic detect-then-track demo complete.\n"
        << "MOG2 detections: "
        << (output_dir /
            "mog2_detections.csv")
        << '\n'
        << "Feed into PR #28 tracker with:\n"
        << "  ./build/cv9x_multi_object_tracking"
        << " --detections "
        << (output_dir /
            "mog2_detections.csv")
        << " --output-dir "
        << (output_dir /
            "sort_from_mog2")
        << '\n'
        << std::fixed
        << std::setprecision(4)
        << "MeanShift center error: "
        << mean_metrics.mean_center_error
        << " px\n"
        << "CamShift center error: "
        << cam_metrics.mean_center_error
        << " px\n"
        << "CamShift mean IoU: "
        << cam_metrics.mean_iou
        << '\n';

    return 0;
}

int runVideoDemo(
    const Options& options,
    const fs::path& output_dir) {
    cv::VideoCapture capture(
        options.video_path);

    if (!capture.isOpened()) {
        std::cerr
            << "Error: could not open video: "
            << options.video_path
            << '\n';
        return 1;
    }

    const double fps =
        std::max(
            1.0,
            capture.get(
                cv::CAP_PROP_FPS));

    const cv::Size frame_size(
        static_cast<int>(
            capture.get(
                cv::CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(
            capture.get(
                cv::CAP_PROP_FRAME_HEIGHT)));

    auto mog2 =
        cv::createBackgroundSubtractorMOG2(
            options.history,
            options.var_threshold,
            options.detect_shadows);

    std::vector<std::vector<Detection>>
        all_detections;

    cv::VideoWriter writer(
        (output_dir /
         "mog2_blob_overlay.avi")
            .string(),
        cv::VideoWriter::fourcc(
            'M',
            'J',
            'P',
            'G'),
        fps,
        frame_size,
        true);

    cv::Mat frame;

    cv::Mat histogram;
    cv::Rect mean_window;
    cv::Rect cam_window;
    bool appearance_initialized = false;

    int frame_index = 0;

    while (frame_index <
               options.max_frames &&
           capture.read(frame)) {
        Mog2FrameResult result =
            processMog2Frame(
                frame,
                mog2,
                options.morph_kernel,
                options.min_area,
                options.learning_rate);

        all_detections.push_back(
            result.detections);

        cv::Mat overlay =
            drawBlobOverlay(
                frame,
                result.detections);

        if (options.has_roi &&
            !appearance_initialized) {
            const cv::Rect roi =
                clipRect(
                    options.initial_roi,
                    frame.size());

            if (roi !=
                options.initial_roi) {
                std::cerr
                    << "Error: ROI is outside the first video frame.\n";
                return 1;
            }

            histogram =
                hueHistogram(
                    frame,
                    roi);

            mean_window = roi;
            cam_window = roi;
            appearance_initialized = true;
        } else if (
            appearance_initialized) {
            const cv::Mat back_projection =
                hueBackProjection(
                    frame,
                    histogram);

            const cv::TermCriteria criteria(
                cv::TermCriteria::EPS |
                    cv::TermCriteria::COUNT,
                20,
                0.5);

            cv::meanShift(
                back_projection,
                mean_window,
                criteria);

            const cv::RotatedRect cam_box =
                cv::CamShift(
                    back_projection,
                    cam_window,
                    criteria);

            cam_window =
                clipRect(
                    cam_box.boundingRect(),
                    frame.size());

            mean_window =
                clipRect(
                    mean_window,
                    frame.size());

            cv::rectangle(
                overlay,
                mean_window,
                cv::Scalar(0, 220, 255),
                2,
                cv::LINE_AA);

            cv::rectangle(
                overlay,
                cam_window,
                cv::Scalar(255, 120, 0),
                2,
                cv::LINE_AA);
        }

        if (writer.isOpened()) {
            writer.write(
                overlay);
        }

        ++frame_index;
    }

    if (!writeDetectionsCsv(
            output_dir /
                "mog2_detections.csv",
            all_detections)) {
        std::cerr
            << "Error: failed to write detection CSV.\n";
        return 1;
    }

    std::cout
        << "Processed "
        << all_detections.size()
        << " video frames.\n"
        << "Detections: "
        << (output_dir /
            "mog2_detections.csv")
        << '\n';

    return 0;
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

    const fs::path output_dir(
        options.output_dir);

    std::error_code error;

    fs::create_directories(
        output_dir,
        error);

    if (error) {
        std::cerr
            << "Error: could not create output directory: "
            << error.message()
            << '\n';
        return 1;
    }

    if (options.video_path.empty()) {
        return runSyntheticDemo(
            options,
            output_dir);
    }

    return runVideoDemo(
        options,
        output_dir);
}
