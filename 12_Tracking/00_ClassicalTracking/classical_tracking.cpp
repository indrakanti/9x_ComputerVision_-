#include <opencv2/opencv.hpp>
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
#include <vector>

namespace fs = std::filesystem;

namespace {

enum class TrackState {
    Initializing,
    Tracking,
    Coasting,
    Lost
};

struct Options {
    std::string video_path;
    std::string output_dir;
    cv::Rect initial_roi;
    bool has_roi = false;
    int search_radius = 28;
    int max_missed = 5;
    int max_frames = 180;
    double ncc_threshold = 0.72;
    double template_alpha = 0.0;
    bool self_test = false;
    bool help = false;
};

struct MatchResult {
    bool found = false;
    cv::Rect box;
    double score = -1.0;
};

struct TrackRecord {
    int frame_index = 0;
    TrackState state = TrackState::Initializing;
    cv::Rect box;
    cv::Point2d predicted_center;
    cv::Point2d estimated_center;
    double match_score = -1.0;
    int missed = 0;
    bool measurement_used = false;
};

struct SyntheticSequence {
    std::vector<cv::Mat> frames;
    std::vector<cv::Rect> truth;
    std::vector<bool> visible;
    cv::Rect initial_roi;
};

std::string stateName(TrackState state) {
    switch (state) {
        case TrackState::Initializing:
            return "initializing";
        case TrackState::Tracking:
            return "tracking";
        case TrackState::Coasting:
            return "coasting";
        case TrackState::Lost:
            return "lost";
    }

    return "unknown";
}

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " --output-dir <dir> [synthetic options]\n"
        << "  " << program << " --video <path> --roi x,y,w,h --output-dir <dir> [options]\n"
        << "  " << program << " --self-test\n\n"
        << "Options:\n"
        << "  --video <path>            Optional input video\n"
        << "  --roi x,y,w,h             Initial target ROI for video mode\n"
        << "  --output-dir <path>       Output directory\n"
        << "  --search-radius <px>      Correlation search radius (default: 28)\n"
        << "  --max-missed <N>          Misses before Lost (default: 5)\n"
        << "  --max-frames <N>          Max frames to process (default: 180)\n"
        << "  --ncc-threshold <value>   Accept measurement at/above score (default: 0.72)\n"
        << "  --template-alpha <0..1>   Optional template update rate (default: 0)\n"
        << "  --self-test               Run deterministic tests\n"
        << "  --help                    Show this help\n";
}

cv::Rect parseRoi(const std::string& text) {
    std::vector<int> values;
    std::string token;

    for (std::size_t i = 0;
         i <= text.size();
         ++i) {
        const bool separator =
            i == text.size() ||
            text[i] == ',';

        if (!separator) {
            token.push_back(text[i]);
            continue;
        }

        if (token.empty()) {
            throw std::runtime_error(
                "ROI must be x,y,w,h.");
        }

        values.push_back(
            std::stoi(token));

        token.clear();
    }

    if (values.size() != 4U) {
        throw std::runtime_error(
            "ROI must contain four integers: x,y,w,h.");
    }

    return cv::Rect(
        values[0],
        values[1],
        values[2],
        values[3]);
}

bool parseArguments(int argc, char** argv, Options& options) {
    auto value = [argc, argv](int& i, const std::string& name) {
        if (i + 1 >= argc) {
            throw std::runtime_error(
                "Missing value for " + name);
        }

        ++i;
        return std::string(argv[i]);
    };

    try {
        for (int i = 1;
             i < argc;
             ++i) {
            const std::string arg =
                argv[i];

            if (arg == "--video") {
                options.video_path =
                    value(i, arg);
            } else if (arg == "--roi") {
                options.initial_roi =
                    parseRoi(
                        value(i, arg));
                options.has_roi = true;
            } else if (arg == "--output-dir") {
                options.output_dir =
                    value(i, arg);
            } else if (arg == "--search-radius") {
                options.search_radius =
                    std::stoi(
                        value(i, arg));
            } else if (arg == "--max-missed") {
                options.max_missed =
                    std::stoi(
                        value(i, arg));
            } else if (arg == "--max-frames") {
                options.max_frames =
                    std::stoi(
                        value(i, arg));
            } else if (arg == "--ncc-threshold") {
                options.ncc_threshold =
                    std::stod(
                        value(i, arg));
            } else if (arg == "--template-alpha") {
                options.template_alpha =
                    std::stod(
                        value(i, arg));
            } else if (arg == "--self-test") {
                options.self_test = true;
            } else if (
                arg == "--help" ||
                arg == "-h") {
                options.help = true;
            } else {
                throw std::runtime_error(
                    "Unknown option: " + arg);
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
    if (options.help ||
        options.self_test) {
        return true;
    }

    if (options.output_dir.empty()) {
        std::cerr
            << "Error: --output-dir is required.\n";
        return false;
    }

    if (!options.video_path.empty() &&
        !options.has_roi) {
        std::cerr
            << "Error: video mode requires --roi x,y,w,h.\n";
        return false;
    }

    if (options.search_radius < 1 ||
        options.max_missed < 0 ||
        options.max_frames < 1 ||
        options.ncc_threshold < -1.0 ||
        options.ncc_threshold > 1.0 ||
        options.template_alpha < 0.0 ||
        options.template_alpha > 1.0) {
        std::cerr
            << "Error: invalid tracker configuration.\n";
        return false;
    }

    if (options.has_roi &&
        (options.initial_roi.width <= 1 ||
         options.initial_roi.height <= 1)) {
        std::cerr
            << "Error: ROI must have positive width and height.\n";
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

    cv::cvtColor(
        input,
        gray,
        cv::COLOR_BGR2GRAY);

    return gray;
}

cv::Mat toBgr(const cv::Mat& input) {
    if (input.channels() == 3) {
        return input.clone();
    }

    cv::Mat bgr;

    cv::cvtColor(
        input,
        bgr,
        cv::COLOR_GRAY2BGR);

    return bgr;
}

cv::Point2d rectCenter(
    const cv::Rect& box) {
    return cv::Point2d(
        box.x +
            0.5 *
            box.width,
        box.y +
            0.5 *
            box.height);
}

cv::Rect centeredBox(
    const cv::Point2d& center,
    cv::Size size) {
    return cv::Rect(
        cvRound(
            center.x -
            0.5 *
            size.width),
        cvRound(
            center.y -
            0.5 *
            size.height),
        size.width,
        size.height);
}

cv::Rect clipRect(
    const cv::Rect& box,
    cv::Size image_size) {
    return box &
           cv::Rect(
               0,
               0,
               image_size.width,
               image_size.height);
}

double normalizedCrossCorrelation(
    const cv::Mat& template_gray,
    const cv::Mat& patch_gray) {
    CV_Assert(
        template_gray.type() ==
            CV_8U &&
        patch_gray.type() ==
            CV_8U &&
        template_gray.size() ==
            patch_gray.size());

    const int count =
        template_gray.rows *
        template_gray.cols;

    if (count <= 1) {
        return -1.0;
    }

    double mean_template = 0.0;
    double mean_patch = 0.0;

    for (int row = 0;
         row < template_gray.rows;
         ++row) {
        for (int col = 0;
             col < template_gray.cols;
             ++col) {
            mean_template +=
                template_gray.at<unsigned char>(
                    row,
                    col);

            mean_patch +=
                patch_gray.at<unsigned char>(
                    row,
                    col);
        }
    }

    mean_template /=
        count;

    mean_patch /=
        count;

    double numerator = 0.0;
    double template_energy = 0.0;
    double patch_energy = 0.0;

    for (int row = 0;
         row < template_gray.rows;
         ++row) {
        for (int col = 0;
             col < template_gray.cols;
             ++col) {
            const double a =
                template_gray.at<unsigned char>(
                    row,
                    col) -
                mean_template;

            const double b =
                patch_gray.at<unsigned char>(
                    row,
                    col) -
                mean_patch;

            numerator +=
                a * b;

            template_energy +=
                a * a;

            patch_energy +=
                b * b;
        }
    }

    const double denominator =
        std::sqrt(
            template_energy *
            patch_energy);

    if (denominator <= 1e-12) {
        return -1.0;
    }

    return numerator /
           denominator;
}

MatchResult searchTemplateNcc(
    const cv::Mat& gray,
    const cv::Mat& template_gray,
    const cv::Point2d& predicted_center,
    int search_radius) {
    MatchResult best;

    const int template_width =
        template_gray.cols;

    const int template_height =
        template_gray.rows;

    const int predicted_x =
        cvRound(
            predicted_center.x -
            0.5 *
            template_width);

    const int predicted_y =
        cvRound(
            predicted_center.y -
            0.5 *
            template_height);

    const int min_x =
        std::max(
            0,
            predicted_x -
                search_radius);

    const int max_x =
        std::min(
            gray.cols -
                template_width,
            predicted_x +
                search_radius);

    const int min_y =
        std::max(
            0,
            predicted_y -
                search_radius);

    const int max_y =
        std::min(
            gray.rows -
                template_height,
            predicted_y +
                search_radius);

    if (max_x < min_x ||
        max_y < min_y) {
        return best;
    }

    for (int y = min_y;
         y <= max_y;
         ++y) {
        for (int x = min_x;
             x <= max_x;
             ++x) {
            const cv::Rect candidate(
                x,
                y,
                template_width,
                template_height);

            const double score =
                normalizedCrossCorrelation(
                    template_gray,
                    gray(candidate));

            if (!best.found ||
                score >
                    best.score) {
                best.found = true;
                best.box =
                    candidate;
                best.score =
                    score;
            }
        }
    }

    return best;
}

cv::KalmanFilter makeKalman(
    const cv::Point2d& center,
    const cv::Point2d& velocity =
        cv::Point2d(0.0, 0.0)) {
    cv::KalmanFilter filter(
        4,
        2,
        0,
        CV_64F);

    filter.transitionMatrix =
        (cv::Mat_<double>(4, 4) <<
            1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0);

    filter.measurementMatrix =
        (cv::Mat_<double>(2, 4) <<
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0);

    cv::setIdentity(
        filter.processNoiseCov,
        cv::Scalar::all(
            1e-2));

    cv::setIdentity(
        filter.measurementNoiseCov,
        cv::Scalar::all(
            1e-1));

    cv::setIdentity(
        filter.errorCovPost,
        cv::Scalar::all(
            1.0));

    filter.statePost.at<double>(
        0,
        0) =
        center.x;

    filter.statePost.at<double>(
        1,
        0) =
        center.y;

    filter.statePost.at<double>(
        2,
        0) =
        velocity.x;

    filter.statePost.at<double>(
        3,
        0) =
        velocity.y;

    return filter;
}

cv::Point2d kalmanPredict(
    cv::KalmanFilter& filter) {
    const cv::Mat prediction =
        filter.predict();

    return cv::Point2d(
        prediction.at<double>(
            0,
            0),
        prediction.at<double>(
            1,
            0));
}

cv::Point2d kalmanCorrect(
    cv::KalmanFilter& filter,
    const cv::Point2d& measurement) {
    cv::Mat measurement_vector(
        2,
        1,
        CV_64F);

    measurement_vector.at<double>(
        0,
        0) =
        measurement.x;

    measurement_vector.at<double>(
        1,
        0) =
        measurement.y;

    const cv::Mat corrected =
        filter.correct(
            measurement_vector);

    return cv::Point2d(
        corrected.at<double>(
            0,
            0),
        corrected.at<double>(
            1,
            0));
}

void maybeUpdateTemplate(
    cv::Mat& template_gray,
    const cv::Mat& gray,
    const cv::Rect& box,
    double alpha) {
    if (alpha <= 0.0 ||
        box.size() !=
            template_gray.size()) {
        return;
    }

    cv::Mat current =
        gray(box);

    cv::addWeighted(
        template_gray,
        1.0 - alpha,
        current,
        alpha,
        0.0,
        template_gray);
}

SyntheticSequence createSyntheticSequence(
    int frame_count,
    int occlusion_begin,
    int occlusion_end) {
    SyntheticSequence sequence;

    const cv::Size frame_size(
        480,
        320);

    const cv::Size target_size(
        34,
        42);

    cv::Mat target(
        target_size,
        CV_8U,
        cv::Scalar(55));

    for (int y = 3;
         y < target.rows - 3;
         y += 6) {
        cv::line(
            target,
            cv::Point(2, y),
            cv::Point(
                target.cols - 3,
                y),
            cv::Scalar(220),
            2,
            cv::LINE_8);
    }

    cv::rectangle(
        target,
        cv::Rect(
            6,
            8,
            9,
            22),
        cv::Scalar(245),
        cv::FILLED,
        cv::LINE_8);

    cv::circle(
        target,
        cv::Point(
            25,
            27),
        6,
        cv::Scalar(15),
        cv::FILLED,
        cv::LINE_8);

    cv::GaussianBlur(
        target,
        target,
        cv::Size(3, 3),
        0.6);

    const cv::Point2d start(
        70.0,
        70.0);

    const cv::Point2d velocity(
        3.0,
        2.0);

    cv::RNG rng(
        0x1234ABCD);

    for (int frame_index = 0;
         frame_index <
             frame_count;
         ++frame_index) {
        cv::Mat frame(
            frame_size,
            CV_8U,
            cv::Scalar(25));

        cv::Mat noise(
            frame_size,
            CV_8U);

        rng.fill(
            noise,
            cv::RNG::UNIFORM,
            0,
            30);

        cv::add(
            frame,
            noise,
            frame);

        for (int x = 40;
             x < frame.cols;
             x += 90) {
            cv::line(
                frame,
                cv::Point(x, 0),
                cv::Point(
                    x,
                    frame.rows - 1),
                cv::Scalar(45),
                1,
                cv::LINE_8);
        }

        const cv::Point2d top_left =
            start +
            velocity *
                static_cast<double>(
                    frame_index);

        const cv::Rect box(
            cvRound(
                top_left.x),
            cvRound(
                top_left.y),
            target.cols,
            target.rows);

        const bool visible =
            frame_index <
                occlusion_begin ||
            frame_index >
                occlusion_end;

        if (visible &&
            clipRect(
                box,
                frame.size()) ==
                box) {
            target.copyTo(
                frame(box));
        }

        sequence.frames.push_back(
            frame);

        sequence.truth.push_back(
            box);

        sequence.visible.push_back(
            visible);
    }

    sequence.initial_roi =
        sequence.truth.front();

    return sequence;
}

std::vector<TrackRecord> trackSequence(
    const std::vector<cv::Mat>& frames,
    const cv::Rect& initial_roi,
    int search_radius,
    int max_missed,
    double ncc_threshold,
    double template_alpha,
    cv::Mat* final_template = nullptr) {
    std::vector<TrackRecord> records;

    if (frames.empty()) {
        return records;
    }

    const cv::Mat first =
        toGray(
            frames.front());

    if (clipRect(
            initial_roi,
            first.size()) !=
        initial_roi) {
        throw std::runtime_error(
            "Initial ROI is outside the first frame.");
    }

    cv::Mat template_gray =
        first(initial_roi).clone();

    cv::KalmanFilter filter =
        makeKalman(
            rectCenter(
                initial_roi));

    cv::Rect current_box =
        initial_roi;

    int missed = 0;

    TrackRecord initial;
    initial.frame_index = 0;
    initial.state =
        TrackState::Initializing;
    initial.box =
        initial_roi;
    initial.predicted_center =
        rectCenter(
            initial_roi);
    initial.estimated_center =
        rectCenter(
            initial_roi);
    initial.match_score = 1.0;
    initial.measurement_used = true;

    records.push_back(
        initial);

    for (std::size_t index = 1;
         index < frames.size();
         ++index) {
        const cv::Mat gray =
            toGray(
                frames[index]);

        const cv::Point2d predicted =
            kalmanPredict(
                filter);

        const MatchResult match =
            searchTemplateNcc(
                gray,
                template_gray,
                predicted,
                search_radius);

        const bool accepted =
            match.found &&
            match.score >=
                ncc_threshold;

        TrackRecord record;
        record.frame_index =
            static_cast<int>(
                index);
        record.predicted_center =
            predicted;
        record.match_score =
            match.found
                ? match.score
                : -1.0;

        if (accepted) {
            const cv::Point2d measurement =
                rectCenter(
                    match.box);

            const cv::Point2d corrected =
                kalmanCorrect(
                    filter,
                    measurement);

            current_box =
                centeredBox(
                    corrected,
                    initial_roi.size());

            current_box =
                clipRect(
                    current_box,
                    gray.size());

            if (current_box.size() !=
                initial_roi.size()) {
                current_box =
                    match.box;
            }

            missed = 0;

            record.state =
                TrackState::Tracking;
            record.measurement_used =
                true;
            record.estimated_center =
                corrected;
            record.box =
                current_box;

            maybeUpdateTemplate(
                template_gray,
                gray,
                match.box,
                template_alpha);
        } else {
            ++missed;

            current_box =
                centeredBox(
                    predicted,
                    initial_roi.size());

            current_box =
                clipRect(
                    current_box,
                    gray.size());

            record.measurement_used =
                false;
            record.estimated_center =
                predicted;
            record.box =
                current_box;
            record.state =
                missed >
                        max_missed
                    ? TrackState::Lost
                    : TrackState::Coasting;
        }

        record.missed =
            missed;

        records.push_back(
            record);
    }

    if (final_template !=
        nullptr) {
        *final_template =
            template_gray.clone();
    }

    return records;
}

cv::Mat drawRecord(
    const cv::Mat& frame,
    const TrackRecord& record,
    const cv::Rect* truth = nullptr) {
    cv::Mat output =
        toBgr(frame);

    const cv::Scalar state_color =
        record.state ==
                TrackState::Tracking ||
            record.state ==
                TrackState::Initializing
            ? cv::Scalar(
                  0,
                  255,
                  0)
            : record.state ==
                      TrackState::Coasting
                  ? cv::Scalar(
                        0,
                        200,
                        255)
                  : cv::Scalar(
                        0,
                        0,
                        255);

    if (record.box.width > 0 &&
        record.box.height > 0) {
        cv::rectangle(
            output,
            record.box,
            state_color,
            2,
            cv::LINE_AA);
    }

    cv::circle(
        output,
        cv::Point(
            cvRound(
                record.predicted_center.x),
            cvRound(
                record.predicted_center.y)),
        4,
        cv::Scalar(
            255,
            0,
            255),
        1,
        cv::LINE_AA);

    cv::circle(
        output,
        cv::Point(
            cvRound(
                record.estimated_center.x),
            cvRound(
                record.estimated_center.y)),
        4,
        state_color,
        cv::FILLED,
        cv::LINE_AA);

    if (truth != nullptr) {
        cv::rectangle(
            output,
            *truth,
            cv::Scalar(
                255,
                160,
                0),
            1,
            cv::LINE_AA);
    }

    cv::putText(
        output,
        "state=" +
            stateName(
                record.state) +
            " score=" +
            std::to_string(
                record.match_score),
        cv::Point(
            15,
            28),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(
            255,
            255,
            255),
        1,
        cv::LINE_AA);

    cv::putText(
        output,
        "missed=" +
            std::to_string(
                record.missed),
        cv::Point(
            15,
            52),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(
            255,
            255,
            255),
        1,
        cv::LINE_AA);

    return output;
}

cv::Mat drawTrajectory(
    const std::vector<TrackRecord>& records,
    const std::vector<cv::Rect>* truth,
    cv::Size size) {
    cv::Mat image(
        size,
        CV_8UC3,
        cv::Scalar(
            25,
            25,
            25));

    for (std::size_t i = 1;
         i < records.size();
         ++i) {
        cv::line(
            image,
            cv::Point(
                cvRound(
                    records[i - 1]
                        .estimated_center.x),
                cvRound(
                    records[i - 1]
                        .estimated_center.y)),
            cv::Point(
                cvRound(
                    records[i]
                        .estimated_center.x),
                cvRound(
                    records[i]
                        .estimated_center.y)),
            cv::Scalar(
                0,
                255,
                0),
            2,
            cv::LINE_AA);
    }

    if (truth != nullptr &&
        truth->size() ==
            records.size()) {
        for (std::size_t i = 1;
             i < truth->size();
             ++i) {
            cv::line(
                image,
                cv::Point(
                    cvRound(
                        rectCenter(
                            (*truth)[i - 1])
                            .x),
                    cvRound(
                        rectCenter(
                            (*truth)[i - 1])
                            .y)),
                cv::Point(
                    cvRound(
                        rectCenter(
                            (*truth)[i])
                            .x),
                    cvRound(
                        rectCenter(
                            (*truth)[i])
                            .y)),
                cv::Scalar(
                    255,
                    160,
                    0),
                1,
                cv::LINE_AA);
        }
    }

    cv::putText(
        image,
        "green=estimate, orange=ground truth",
        cv::Point(
            15,
            25),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(
            255,
            255,
            255),
        1,
        cv::LINE_AA);

    return image;
}

bool writeCsv(
    const fs::path& path,
    const std::vector<TrackRecord>& records,
    const std::vector<cv::Rect>* truth = nullptr,
    const std::vector<bool>* visible = nullptr) {
    std::ofstream output(path);

    if (!output) {
        return false;
    }

    output
        << "frame,state,box_x,box_y,box_w,box_h,"
        << "predicted_x,predicted_y,estimated_x,estimated_y,"
        << "score,missed,measurement_used";

    if (truth != nullptr) {
        output
            << ",truth_x,truth_y,truth_w,truth_h";
    }

    if (visible != nullptr) {
        output
            << ",truth_visible";
    }

    output << "\n";

    output
        << std::fixed
        << std::setprecision(6);

    for (std::size_t i = 0;
         i < records.size();
         ++i) {
        const auto& record =
            records[i];

        output
            << record.frame_index << ","
            << stateName(
                   record.state)
            << ","
            << record.box.x << ","
            << record.box.y << ","
            << record.box.width << ","
            << record.box.height << ","
            << record.predicted_center.x << ","
            << record.predicted_center.y << ","
            << record.estimated_center.x << ","
            << record.estimated_center.y << ","
            << record.match_score << ","
            << record.missed << ","
            << (record.measurement_used
                    ? 1
                    : 0);

        if (truth != nullptr &&
            i < truth->size()) {
            const cv::Rect box =
                (*truth)[i];

            output
                << ","
                << box.x
                << ","
                << box.y
                << ","
                << box.width
                << ","
                << box.height;
        }

        if (visible != nullptr &&
            i < visible->size()) {
            output
                << ","
                << ((*visible)[i]
                        ? 1
                        : 0);
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

double centerError(
    const TrackRecord& record,
    const cv::Rect& truth) {
    const cv::Point2d expected =
        rectCenter(
            truth);

    const double dx =
        record.estimated_center.x -
        expected.x;

    const double dy =
        record.estimated_center.y -
        expected.y;

    return std::sqrt(
        dx * dx +
        dy * dy);
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

    cv::Mat template_image(
        24,
        20,
        CV_8U,
        cv::Scalar(20));

    cv::rectangle(
        template_image,
        cv::Rect(
            3,
            4,
            7,
            12),
        cv::Scalar(220),
        cv::FILLED);

    cv::circle(
        template_image,
        cv::Point(
            15,
            16),
        3,
        cv::Scalar(70),
        cv::FILLED);

    const double self_ncc =
        normalizedCrossCorrelation(
            template_image,
            template_image);

    check(
        std::abs(
            self_ncc -
            1.0) <
            1e-12,
        "manual NCC of identical patches is one");

    cv::Mat shifted_scene(
        100,
        140,
        CV_8U,
        cv::Scalar(15));

    const cv::Rect expected_box(
        64,
        37,
        template_image.cols,
        template_image.rows);

    template_image.copyTo(
        shifted_scene(
            expected_box));

    const MatchResult match =
        searchTemplateNcc(
            shifted_scene,
            template_image,
            rectCenter(
                cv::Rect(
                    58,
                    34,
                    template_image.cols,
                    template_image.rows)),
            16);

    check(
        match.found &&
        match.box ==
            expected_box,
        "manual NCC search localizes the exact template");

    cv::Mat opencv_response;

    cv::matchTemplate(
        template_image,
        template_image,
        opencv_response,
        cv::TM_CCOEFF_NORMED);

    check(
        std::abs(
            self_ncc -
            opencv_response.at<float>(
                0,
                0)) <
            1e-6,
        "manual NCC matches OpenCV normalized correlation reference");

    cv::KalmanFilter filter =
        makeKalman(
            cv::Point2d(
                10.0,
                20.0),
            cv::Point2d(
                3.0,
                -2.0));

    const cv::Point2d predicted =
        kalmanPredict(
            filter);

    check(
        std::abs(
            predicted.x -
            13.0) <
            1e-10 &&
        std::abs(
            predicted.y -
            18.0) <
            1e-10,
        "constant-velocity Kalman prediction advances by velocity");

    const SyntheticSequence sequence =
        createSyntheticSequence(
            52,
            20,
            22);

    const auto records =
        trackSequence(
            sequence.frames,
            sequence.initial_roi,
            24,
            4,
            0.78,
            0.0);

    check(
        records.size() ==
            sequence.frames.size(),
        "tracker emits one record per frame");

    double visible_error_sum = 0.0;
    int visible_count = 0;
    bool coasted_during_occlusion =
        true;
    bool lost_during_short_occlusion =
        false;
    bool reacquired_after_occlusion =
        false;

    for (std::size_t i = 1;
         i < records.size();
         ++i) {
        if (sequence.visible[i]) {
            visible_error_sum +=
                centerError(
                    records[i],
                    sequence.truth[i]);

            ++visible_count;
        }

        if (!sequence.visible[i]) {
            coasted_during_occlusion &=
                records[i].state ==
                    TrackState::Coasting;

            lost_during_short_occlusion |=
                records[i].state ==
                    TrackState::Lost;
        }

        if (i > 22U &&
            sequence.visible[i] &&
            records[i].state ==
                TrackState::Tracking) {
            reacquired_after_occlusion =
                true;
        }
    }

    const double mean_visible_error =
        visible_count == 0
            ? std::numeric_limits<double>::infinity()
            : visible_error_sum /
                  visible_count;

    check(
        mean_visible_error <
            1.5,
        "synthetic visible-frame mean center error is below 1.5 px");

    check(
        coasted_during_occlusion &&
        !lost_during_short_occlusion,
        "short occlusion uses Kalman coasting without declaring Lost");

    check(
        reacquired_after_occlusion,
        "correlation measurement reacquires after short occlusion");

    const SyntheticSequence long_occlusion =
        createSyntheticSequence(
            35,
            10,
            22);

    const auto long_records =
        trackSequence(
            long_occlusion.frames,
            long_occlusion.initial_roi,
            24,
            3,
            0.78,
            0.0);

    bool observed_lost = false;

    for (const auto& record :
         long_records) {
        observed_lost |=
            record.state ==
                TrackState::Lost;
    }

    check(
        observed_lost,
        "long measurement outage transitions tracker to Lost");

    if (passed) {
        std::cout
            << "Classical tracking self-test passed.\n";
    }

    return passed;
}

bool runSynthetic(
    const Options& options,
    const fs::path& output_dir) {
    const SyntheticSequence sequence =
        createSyntheticSequence(
            std::min(
                options.max_frames,
                90),
            28,
            31);

    cv::Mat final_template;

    const auto records =
        trackSequence(
            sequence.frames,
            sequence.initial_roi,
            options.search_radius,
            options.max_missed,
            options.ncc_threshold,
            options.template_alpha,
            &final_template);

    const fs::path video_path =
        output_dir /
        "tracking_overlay.avi";

    cv::VideoWriter writer(
        video_path.string(),
        cv::VideoWriter::fourcc(
            'M',
            'J',
            'P',
            'G'),
        20.0,
        sequence.frames.front().size(),
        true);

    if (!writer.isOpened()) {
        std::cerr
            << "Error: could not open synthetic output video.\n";
        return false;
    }

    for (std::size_t i = 0;
         i < records.size();
         ++i) {
        writer.write(
            drawRecord(
                sequence.frames[i],
                records[i],
                &sequence.truth[i]));
    }

    writer.release();

    bool ok = true;

    ok &= writeImage(
        output_dir /
            "01_initial_template.png",
        sequence.frames.front()(
            sequence.initial_roi));

    ok &= writeImage(
        output_dir /
            "02_final_template.png",
        final_template);

    ok &= writeImage(
        output_dir /
            "03_trajectory.png",
        drawTrajectory(
            records,
            &sequence.truth,
            sequence.frames.front().size()));

    ok &= writeCsv(
        output_dir /
            "tracking_metrics.csv",
        records,
        &sequence.truth,
        &sequence.visible);

    double error_sum = 0.0;
    int error_count = 0;
    int tracking_frames = 0;
    int coasting_frames = 0;
    int lost_frames = 0;

    for (std::size_t i = 0;
         i < records.size();
         ++i) {
        if (sequence.visible[i]) {
            error_sum +=
                centerError(
                    records[i],
                    sequence.truth[i]);
            ++error_count;
        }

        tracking_frames +=
            records[i].state ==
                    TrackState::Tracking
                ? 1
                : 0;

        coasting_frames +=
            records[i].state ==
                    TrackState::Coasting
                ? 1
                : 0;

        lost_frames +=
            records[i].state ==
                    TrackState::Lost
                ? 1
                : 0;
    }

    std::cout
        << std::fixed
        << std::setprecision(4)
        << "Synthetic tracker summary\n"
        << "  frames: "
        << records.size()
        << '\n'
        << "  tracking frames: "
        << tracking_frames
        << '\n'
        << "  coasting frames: "
        << coasting_frames
        << '\n'
        << "  lost frames: "
        << lost_frames
        << '\n'
        << "  mean visible center error: "
        << (
            error_count == 0
                ? 0.0
                : error_sum /
                      error_count)
        << " px\n"
        << "  overlay video: "
        << video_path
        << '\n';

    return ok;
}

bool runVideo(
    const Options& options,
    const fs::path& output_dir) {
    cv::VideoCapture capture(
        options.video_path);

    if (!capture.isOpened()) {
        std::cerr
            << "Error: could not open video: "
            << options.video_path
            << '\n';
        return false;
    }

    cv::Mat first_bgr;

    if (!capture.read(
            first_bgr) ||
        first_bgr.empty()) {
        std::cerr
            << "Error: could not read first video frame.\n";
        return false;
    }

    const cv::Mat first_gray =
        toGray(
            first_bgr);

    if (clipRect(
            options.initial_roi,
            first_gray.size()) !=
        options.initial_roi) {
        std::cerr
            << "Error: initial ROI is outside the frame.\n";
        return false;
    }

    std::vector<cv::Mat> frames;
    frames.push_back(
        first_gray);

    for (int i = 1;
         i < options.max_frames;
         ++i) {
        cv::Mat frame;

        if (!capture.read(frame) ||
            frame.empty()) {
            break;
        }

        frames.push_back(
            toGray(frame));
    }

    if (frames.size() <
        2U) {
        std::cerr
            << "Error: not enough video frames.\n";
        return false;
    }

    cv::Mat final_template;

    const auto records =
        trackSequence(
            frames,
            options.initial_roi,
            options.search_radius,
            options.max_missed,
            options.ncc_threshold,
            options.template_alpha,
            &final_template);

    double fps =
        capture.get(
            cv::CAP_PROP_FPS);

    if (!std::isfinite(fps) ||
        fps <= 1.0) {
        fps = 30.0;
    }

    const fs::path video_path =
        output_dir /
        "tracking_overlay.avi";

    cv::VideoWriter writer(
        video_path.string(),
        cv::VideoWriter::fourcc(
            'M',
            'J',
            'P',
            'G'),
        fps,
        frames.front().size(),
        true);

    if (!writer.isOpened()) {
        std::cerr
            << "Error: could not open output video writer.\n";
        return false;
    }

    for (std::size_t i = 0;
         i < records.size();
         ++i) {
        writer.write(
            drawRecord(
                frames[i],
                records[i]));
    }

    writer.release();

    bool ok = true;

    ok &= writeImage(
        output_dir /
            "01_initial_template.png",
        frames.front()(
            options.initial_roi));

    ok &= writeImage(
        output_dir /
            "02_final_template.png",
        final_template);

    ok &= writeImage(
        output_dir /
            "03_trajectory.png",
        drawTrajectory(
            records,
            nullptr,
            frames.front().size()));

    ok &= writeCsv(
        output_dir /
            "tracking_metrics.csv",
        records);

    std::cout
        << "Video frames tracked: "
        << records.size()
        << '\n'
        << "Tracking overlay: "
        << video_path
        << '\n';

    return ok;
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
            << "Error: could not create output directory "
            << output_dir
            << ": "
            << error.message()
            << '\n';
        return 1;
    }

    const bool ok =
        options.video_path.empty()
            ? runSynthetic(
                  options,
                  output_dir)
            : runVideo(
                  options,
                  output_dir);

    if (!ok) {
        return 1;
    }

    std::cout
        << "Tracker configuration\n"
        << "  NCC threshold: "
        << options.ncc_threshold
        << '\n'
        << "  search radius: "
        << options.search_radius
        << " px\n"
        << "  max missed: "
        << options.max_missed
        << '\n'
        << "  template alpha: "
        << options.template_alpha
        << '\n'
        << "Outputs written to: "
        << output_dir
        << '\n';

    return 0;
}
