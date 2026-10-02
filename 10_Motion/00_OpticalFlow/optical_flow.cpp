#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string frame1_path;
    std::string frame2_path;
    std::string video_path;
    std::string output_dir;
    int max_corners = 250;
    double quality_level = 0.01;
    double min_distance = 10.0;
    int window_size = 21;
    int max_level = 3;
    double fb_threshold = 1.0;
    int max_frames = 300;
    int redetect_below = 80;
    int trajectory_length = 30;
    int synthetic_dx = 8;
    int synthetic_dy = 5;
    bool self_test = false;
    bool help = false;
};

struct FlowTrack {
    cv::Point2f start;
    cv::Point2f end;
    float forward_error = 0.0f;
    float backward_error = 0.0f;
    bool valid = false;
};

struct PersistentTrack {
    int id = -1;
    cv::Point2f point;
    std::deque<cv::Point2f> history;
};

struct SyntheticFrames {
    cv::Mat first;
    cv::Mat second;
    cv::Point2f translation;
};

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " --output-dir <dir> [pair options]\n"
        << "  " << program << " --video <path> --output-dir <dir> [options]\n"
        << "  " << program << " --self-test\n\n"
        << "Pair input:\n"
        << "  --frame1 <path>           First grayscale/color frame\n"
        << "  --frame2 <path>           Second grayscale/color frame\n"
        << "  If neither is supplied, a deterministic translated pair is generated.\n\n"
        << "Tracking options:\n"
        << "  --max-corners <N>         Feature budget (default: 250)\n"
        << "  --quality <value>         goodFeaturesToTrack quality (default: 0.01)\n"
        << "  --min-distance <px>       Minimum feature spacing (default: 10)\n"
        << "  --window <odd N>          LK window size (default: 21)\n"
        << "  --levels <N>              Pyramid max level (default: 3)\n"
        << "  --fb-threshold <px>       Forward/backward threshold (default: 1.0)\n"
        << "  --max-frames <N>          Video frames to process (default: 300)\n"
        << "  --redetect-below <N>      Replenish tracks below this count (default: 80)\n"
        << "  --trajectory-length <N>   History length per track (default: 30)\n"
        << "  --synthetic-dx <px>       Synthetic horizontal motion (default: 8)\n"
        << "  --synthetic-dy <px>       Synthetic vertical motion (default: 5)\n"
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

            if (arg == "--frame1") {
                options.frame1_path = value(i, arg);
            } else if (arg == "--frame2") {
                options.frame2_path = value(i, arg);
            } else if (arg == "--video") {
                options.video_path = value(i, arg);
            } else if (arg == "--output-dir") {
                options.output_dir = value(i, arg);
            } else if (arg == "--max-corners") {
                options.max_corners = std::stoi(value(i, arg));
            } else if (arg == "--quality") {
                options.quality_level = std::stod(value(i, arg));
            } else if (arg == "--min-distance") {
                options.min_distance = std::stod(value(i, arg));
            } else if (arg == "--window") {
                options.window_size = std::stoi(value(i, arg));
            } else if (arg == "--levels") {
                options.max_level = std::stoi(value(i, arg));
            } else if (arg == "--fb-threshold") {
                options.fb_threshold = std::stod(value(i, arg));
            } else if (arg == "--max-frames") {
                options.max_frames = std::stoi(value(i, arg));
            } else if (arg == "--redetect-below") {
                options.redetect_below = std::stoi(value(i, arg));
            } else if (arg == "--trajectory-length") {
                options.trajectory_length = std::stoi(value(i, arg));
            } else if (arg == "--synthetic-dx") {
                options.synthetic_dx = std::stoi(value(i, arg));
            } else if (arg == "--synthetic-dy") {
                options.synthetic_dy = std::stoi(value(i, arg));
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

    const bool one_frame =
        options.frame1_path.empty() != options.frame2_path.empty();

    if (one_frame) {
        std::cerr << "Error: provide both --frame1 and --frame2, or neither.\n";
        return false;
    }

    if (!options.video_path.empty() &&
        (!options.frame1_path.empty() || !options.frame2_path.empty())) {
        std::cerr << "Error: choose either --video or --frame1/--frame2.\n";
        return false;
    }

    if (options.max_corners <= 0 ||
        options.quality_level <= 0.0 ||
        options.quality_level >= 1.0 ||
        options.min_distance <= 0.0 ||
        options.window_size < 3 ||
        options.window_size % 2 == 0 ||
        options.max_level < 0 ||
        options.fb_threshold <= 0.0 ||
        options.max_frames <= 0 ||
        options.redetect_below < 0 ||
        options.trajectory_length < 2) {
        std::cerr << "Error: invalid feature/tracking configuration.\n";
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

cv::Mat toBgr(const cv::Mat& input) {
    if (input.channels() == 3) {
        return input.clone();
    }

    cv::Mat bgr;
    cv::cvtColor(input, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

std::vector<cv::Point2f> detectFeatures(
    const cv::Mat& gray,
    int max_corners,
    double quality,
    double min_distance,
    const cv::Mat& mask = cv::Mat()) {
    std::vector<cv::Point2f> points;

    cv::goodFeaturesToTrack(
        gray,
        points,
        max_corners,
        quality,
        min_distance,
        mask,
        3,
        false,
        0.04);

    return points;
}

bool solveLucasKanadeNormalEquations(
    const std::vector<cv::Vec3d>& gradient_time_samples,
    cv::Point2d& flow,
    double min_eigenvalue = 1e-9) {
    double gxx = 0.0;
    double gxy = 0.0;
    double gyy = 0.0;
    double b1 = 0.0;
    double b2 = 0.0;

    for (const auto& sample : gradient_time_samples) {
        const double ix = sample[0];
        const double iy = sample[1];
        const double it = sample[2];

        gxx += ix * ix;
        gxy += ix * iy;
        gyy += iy * iy;

        b1 -= ix * it;
        b2 -= iy * it;
    }

    const double trace = gxx + gyy;
    const double determinant =
        gxx * gyy - gxy * gxy;

    const double discriminant =
        std::max(
            0.0,
            trace * trace -
            4.0 * determinant);

    const double min_lambda =
        0.5 *
        (trace - std::sqrt(discriminant));

    if (min_lambda <= min_eigenvalue ||
        std::abs(determinant) <= 1e-12) {
        return false;
    }

    flow.x =
        (gyy * b1 - gxy * b2) /
        determinant;

    flow.y =
        (-gxy * b1 + gxx * b2) /
        determinant;

    return std::isfinite(flow.x) &&
           std::isfinite(flow.y);
}

bool manualLucasKanadeAtPoint(
    const cv::Mat& first,
    const cv::Mat& second,
    const cv::Point2f& point,
    int radius,
    cv::Point2d& flow) {
    CV_Assert(
        first.type() == CV_8U &&
        second.type() == CV_8U &&
        first.size() == second.size());

    cv::Mat first64;
    cv::Mat second64;
    first.convertTo(first64, CV_64F, 1.0 / 255.0);
    second.convertTo(second64, CV_64F, 1.0 / 255.0);

    cv::Mat ix;
    cv::Mat iy;

    cv::Sobel(
        first64,
        ix,
        CV_64F,
        1,
        0,
        3,
        0.5,
        0.0,
        cv::BORDER_REFLECT101);

    cv::Sobel(
        first64,
        iy,
        CV_64F,
        0,
        1,
        3,
        0.5,
        0.0,
        cv::BORDER_REFLECT101);

    const cv::Mat it =
        second64 - first64;

    const int cx = cvRound(point.x);
    const int cy = cvRound(point.y);

    if (cx - radius < 1 ||
        cy - radius < 1 ||
        cx + radius >= first.cols - 1 ||
        cy + radius >= first.rows - 1) {
        return false;
    }

    std::vector<cv::Vec3d> samples;
    samples.reserve(
        static_cast<std::size_t>(
            (2 * radius + 1) *
            (2 * radius + 1)));

    for (int row = cy - radius;
         row <= cy + radius;
         ++row) {
        for (int col = cx - radius;
             col <= cx + radius;
             ++col) {
            samples.emplace_back(
                ix.at<double>(row, col),
                iy.at<double>(row, col),
                it.at<double>(row, col));
        }
    }

    return solveLucasKanadeNormalEquations(
        samples,
        flow,
        1e-5);
}

std::vector<FlowTrack> trackForwardBackward(
    const cv::Mat& first,
    const cv::Mat& second,
    const std::vector<cv::Point2f>& points,
    int window_size,
    int max_level,
    double fb_threshold) {
    std::vector<FlowTrack> result;

    if (points.empty()) {
        return result;
    }

    std::vector<cv::Point2f> forward;
    std::vector<unsigned char> status_forward;
    std::vector<float> error_forward;

    cv::calcOpticalFlowPyrLK(
        first,
        second,
        points,
        forward,
        status_forward,
        error_forward,
        cv::Size(window_size, window_size),
        max_level,
        cv::TermCriteria(
            cv::TermCriteria::COUNT |
            cv::TermCriteria::EPS,
            30,
            0.01));

    std::vector<cv::Point2f> backward;
    std::vector<unsigned char> status_backward;
    std::vector<float> error_backward;

    cv::calcOpticalFlowPyrLK(
        second,
        first,
        forward,
        backward,
        status_backward,
        error_backward,
        cv::Size(window_size, window_size),
        max_level,
        cv::TermCriteria(
            cv::TermCriteria::COUNT |
            cv::TermCriteria::EPS,
            30,
            0.01));

    result.reserve(points.size());

    for (std::size_t i = 0;
         i < points.size();
         ++i) {
        FlowTrack track;
        track.start = points[i];
        track.end =
            i < forward.size()
                ? forward[i]
                : points[i];

        if (i < error_forward.size()) {
            track.forward_error =
                error_forward[i];
        }

        if (i >= status_forward.size() ||
            i >= status_backward.size() ||
            i >= backward.size() ||
            status_forward[i] == 0U ||
            status_backward[i] == 0U) {
            result.push_back(track);
            continue;
        }

        const double dx =
            static_cast<double>(
                points[i].x -
                backward[i].x);

        const double dy =
            static_cast<double>(
                points[i].y -
                backward[i].y);

        track.backward_error =
            static_cast<float>(
                std::sqrt(
                    dx * dx +
                    dy * dy));

        track.valid =
            track.backward_error <=
                fb_threshold &&
            track.end.x >= 0.0f &&
            track.end.y >= 0.0f &&
            track.end.x < second.cols &&
            track.end.y < second.rows;

        result.push_back(track);
    }

    return result;
}

std::vector<cv::Point2f> validEndPoints(
    const std::vector<FlowTrack>& tracks) {
    std::vector<cv::Point2f> points;

    for (const auto& track : tracks) {
        if (track.valid) {
            points.push_back(track.end);
        }
    }

    return points;
}

double median(
    std::vector<double> values) {
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const std::size_t middle =
        values.size() / 2;

    std::nth_element(
        values.begin(),
        values.begin() + middle,
        values.end());

    if (values.size() % 2 == 1) {
        return values[middle];
    }

    const double upper = values[middle];

    std::nth_element(
        values.begin(),
        values.begin() + middle - 1,
        values.end());

    return 0.5 *
           (values[middle - 1] + upper);
}

cv::Point2d medianMotion(
    const std::vector<FlowTrack>& tracks) {
    std::vector<double> dx;
    std::vector<double> dy;

    for (const auto& track : tracks) {
        if (!track.valid) {
            continue;
        }

        dx.push_back(
            track.end.x -
            track.start.x);

        dy.push_back(
            track.end.y -
            track.start.y);
    }

    return cv::Point2d(
        median(dx),
        median(dy));
}

double medianForwardBackwardError(
    const std::vector<FlowTrack>& tracks) {
    std::vector<double> errors;

    for (const auto& track : tracks) {
        if (track.valid) {
            errors.push_back(
                track.backward_error);
        }
    }

    return median(errors);
}

SyntheticFrames createSyntheticFrames(
    int width,
    int height,
    int dx,
    int dy) {
    SyntheticFrames frames;

    cv::Mat base(
        height,
        width,
        CV_8U,
        cv::Scalar(25));

    cv::RNG rng(0xC0FFEE);

    for (int i = 0; i < 80; ++i) {
        const int x =
            rng.uniform(30, width - 30);
        const int y =
            rng.uniform(30, height - 30);
        const int radius =
            rng.uniform(3, 9);
        const int intensity =
            rng.uniform(80, 245);

        cv::circle(
            base,
            cv::Point(x, y),
            radius,
            cv::Scalar(intensity),
            cv::FILLED,
            cv::LINE_AA);
    }

    for (int y = 50;
         y < height - 30;
         y += 70) {
        cv::line(
            base,
            cv::Point(30, y),
            cv::Point(width - 30, y),
            cv::Scalar(180),
            2,
            cv::LINE_AA);
    }

    for (int x = 60;
         x < width - 30;
         x += 90) {
        cv::rectangle(
            base,
            cv::Rect(
                x - 12,
                height / 2 - 20,
                24,
                40),
            cv::Scalar(225),
            2,
            cv::LINE_AA);
    }

    cv::putText(
        base,
        "9X FLOW",
        cv::Point(
            width / 4,
            height / 2 + 75),
        cv::FONT_HERSHEY_SIMPLEX,
        1.2,
        cv::Scalar(245),
        2,
        cv::LINE_AA);

    const cv::Mat transform =
        (cv::Mat_<double>(2, 3) <<
            1.0, 0.0, dx,
            0.0, 1.0, dy);

    cv::Mat shifted;

    cv::warpAffine(
        base,
        shifted,
        transform,
        base.size(),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(25));

    frames.first = base;
    frames.second = shifted;
    frames.translation =
        cv::Point2f(
            static_cast<float>(dx),
            static_cast<float>(dy));

    return frames;
}

cv::Mat drawFeatures(
    const cv::Mat& frame,
    const std::vector<cv::Point2f>& points) {
    cv::Mat output =
        toBgr(frame);

    for (const auto& point : points) {
        cv::circle(
            output,
            point,
            3,
            cv::Scalar(0, 255, 255),
            cv::FILLED,
            cv::LINE_AA);
    }

    return output;
}

cv::Mat drawTracks(
    const cv::Mat& second,
    const std::vector<FlowTrack>& tracks,
    bool valid_only) {
    cv::Mat output =
        toBgr(second);

    for (const auto& track : tracks) {
        if (valid_only &&
            !track.valid) {
            continue;
        }

        const cv::Scalar color =
            track.valid
                ? cv::Scalar(0, 255, 0)
                : cv::Scalar(0, 0, 255);

        cv::line(
            output,
            track.start,
            track.end,
            color,
            1,
            cv::LINE_AA);

        cv::circle(
            output,
            track.end,
            3,
            color,
            cv::FILLED,
            cv::LINE_AA);
    }

    return output;
}

cv::Mat drawManualSmallMotion(
    const cv::Mat& first,
    const cv::Mat& second,
    const std::vector<cv::Point2f>& points) {
    cv::Mat output =
        toBgr(first);

    const std::size_t limit =
        std::min<std::size_t>(
            points.size(),
            60);

    for (std::size_t i = 0;
         i < limit;
         ++i) {
        cv::Point2d flow;

        if (!manualLucasKanadeAtPoint(
                first,
                second,
                points[i],
                5,
                flow)) {
            continue;
        }

        const cv::Point2f end(
            points[i].x +
                static_cast<float>(flow.x),
            points[i].y +
                static_cast<float>(flow.y));

        cv::line(
            output,
            points[i],
            end,
            cv::Scalar(255, 255, 0),
            1,
            cv::LINE_AA);

        cv::circle(
            output,
            points[i],
            2,
            cv::Scalar(0, 255, 255),
            cv::FILLED,
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

void drawPersistentTracks(
    cv::Mat& frame,
    const std::vector<PersistentTrack>& tracks) {
    for (const auto& track : tracks) {
        for (std::size_t i = 1;
             i < track.history.size();
             ++i) {
            cv::line(
                frame,
                track.history[i - 1],
                track.history[i],
                cv::Scalar(0, 255, 0),
                1,
                cv::LINE_AA);
        }

        cv::circle(
            frame,
            track.point,
            3,
            cv::Scalar(0, 255, 255),
            cv::FILLED,
            cv::LINE_AA);

        cv::putText(
            frame,
            std::to_string(track.id),
            track.point + cv::Point2f(4.0f, -4.0f),
            cv::FONT_HERSHEY_SIMPLEX,
            0.35,
            cv::Scalar(255, 255, 255),
            1,
            cv::LINE_AA);
    }
}

void addNewTracks(
    const cv::Mat& gray,
    std::vector<PersistentTrack>& tracks,
    int& next_id,
    const Options& options) {
    cv::Mat mask(
        gray.size(),
        CV_8U,
        cv::Scalar(255));

    for (const auto& track : tracks) {
        cv::circle(
            mask,
            track.point,
            cvRound(options.min_distance),
            cv::Scalar(0),
            cv::FILLED);
    }

    const int available =
        std::max(
            0,
            options.max_corners -
            static_cast<int>(tracks.size()));

    if (available == 0) {
        return;
    }

    const auto new_points =
        detectFeatures(
            gray,
            available,
            options.quality_level,
            options.min_distance,
            mask);

    for (const auto& point :
         new_points) {
        PersistentTrack track;
        track.id = next_id++;
        track.point = point;
        track.history.push_back(point);
        tracks.push_back(track);
    }
}

std::vector<PersistentTrack> updatePersistentTracks(
    const cv::Mat& previous,
    const cv::Mat& current,
    const std::vector<PersistentTrack>& tracks,
    const Options& options) {
    std::vector<cv::Point2f> points;
    points.reserve(tracks.size());

    for (const auto& track :
         tracks) {
        points.push_back(track.point);
    }

    const auto flow =
        trackForwardBackward(
            previous,
            current,
            points,
            options.window_size,
            options.max_level,
            options.fb_threshold);

    std::vector<PersistentTrack> updated;
    updated.reserve(tracks.size());

    for (std::size_t i = 0;
         i < tracks.size() &&
         i < flow.size();
         ++i) {
        if (!flow[i].valid) {
            continue;
        }

        PersistentTrack track =
            tracks[i];

        track.point =
            flow[i].end;

        track.history.push_back(
            track.point);

        while (
            static_cast<int>(
                track.history.size()) >
            options.trajectory_length) {
            track.history.pop_front();
        }

        updated.push_back(
            std::move(track));
    }

    return updated;
}

bool processVideo(
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

    cv::Mat frame;
    if (!capture.read(frame) ||
        frame.empty()) {
        std::cerr
            << "Error: could not read first video frame.\n";
        return false;
    }

    cv::Mat previous =
        toGray(frame);

    double fps =
        capture.get(cv::CAP_PROP_FPS);

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
            'M', 'J', 'P', 'G'),
        fps,
        frame.size(),
        true);

    if (!writer.isOpened()) {
        std::cerr
            << "Error: could not open output video writer.\n";
        return false;
    }

    std::vector<PersistentTrack> tracks;
    int next_id = 0;

    addNewTracks(
        previous,
        tracks,
        next_id,
        options);

    int processed = 0;

    while (processed <
           options.max_frames) {
        cv::Mat current_bgr;

        if (!capture.read(current_bgr) ||
            current_bgr.empty()) {
            break;
        }

        const cv::Mat current =
            toGray(current_bgr);

        tracks =
            updatePersistentTracks(
                previous,
                current,
                tracks,
                options);

        if (static_cast<int>(
                tracks.size()) <
            options.redetect_below) {
            addNewTracks(
                current,
                tracks,
                next_id,
                options);
        }

        cv::Mat overlay =
            toBgr(current_bgr);

        drawPersistentTracks(
            overlay,
            tracks);

        cv::putText(
            overlay,
            "tracks=" +
                std::to_string(
                    tracks.size()),
            cv::Point(20, 30),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(255, 255, 255),
            2,
            cv::LINE_AA);

        writer.write(overlay);

        previous =
            current.clone();

        ++processed;
    }

    writer.release();

    std::cout
        << "Video frames processed: "
        << processed
        << '\n'
        << "Final active tracks: "
        << tracks.size()
        << '\n'
        << "Tracking overlay: "
        << video_path
        << '\n';

    return processed > 0;
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

    const double true_u = 1.25;
    const double true_v = -0.75;

    std::vector<cv::Vec3d> exact_samples;

    const std::vector<cv::Point2d> gradients = {
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0},
        {2.0, -1.0},
        {-1.0, 2.0}
    };

    for (const auto& gradient :
         gradients) {
        const double it =
            -(gradient.x * true_u +
              gradient.y * true_v);

        exact_samples.emplace_back(
            gradient.x,
            gradient.y,
            it);
    }

    cv::Point2d solved_flow;

    check(
        solveLucasKanadeNormalEquations(
            exact_samples,
            solved_flow),
        "Lucas-Kanade normal equations solve a 2-D textured patch");

    check(
        std::abs(
            solved_flow.x -
            true_u) < 1e-12 &&
        std::abs(
            solved_flow.y -
            true_v) < 1e-12,
        "Lucas-Kanade normal equations recover known flow exactly");

    const std::vector<cv::Vec3d> edge_only = {
        {1.0, 0.0, -1.0},
        {2.0, 0.0, -2.0},
        {-1.0, 0.0, 1.0},
        {3.0, 0.0, -3.0}
    };

    cv::Point2d aperture_flow;

    check(
        !solveLucasKanadeNormalEquations(
            edge_only,
            aperture_flow),
        "aperture-problem patch is rejected as singular");

    const SyntheticFrames synthetic =
        createSyntheticFrames(
            640,
            420,
            7,
            4);

    const auto points =
        detectFeatures(
            synthetic.first,
            250,
            0.01,
            10.0);

    check(
        points.size() >= 40,
        "synthetic scene provides many trackable corners");

    const auto tracks =
        trackForwardBackward(
            synthetic.first,
            synthetic.second,
            points,
            21,
            3,
            0.75);

    int valid_count = 0;

    for (const auto& track :
         tracks) {
        if (track.valid) {
            ++valid_count;
        }
    }

    check(
        valid_count >= 30,
        "pyramidal LK retains many translated synthetic features");

    const cv::Point2d motion =
        medianMotion(tracks);

    check(
        std::isfinite(motion.x) &&
        std::isfinite(motion.y),
        "median synthetic motion is finite");

    check(
        std::abs(
            motion.x -
            synthetic.translation.x) < 0.20 &&
        std::abs(
            motion.y -
            synthetic.translation.y) < 0.20,
        "pyramidal LK recovers known frame translation");

    const double median_fb =
        medianForwardBackwardError(
            tracks);

    check(
        std::isfinite(median_fb) &&
        median_fb < 0.20,
        "forward/backward consistency error is small");

    if (passed) {
        std::cout
            << "Optical-flow self-test passed.\n";
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

    if (!options.video_path.empty()) {
        return processVideo(
                   options,
                   output_dir)
            ? 0
            : 1;
    }

    cv::Mat first;
    cv::Mat second;
    bool synthetic_mode = false;
    cv::Point2f synthetic_translation;

    if (!options.frame1_path.empty()) {
        first =
            cv::imread(
                options.frame1_path,
                cv::IMREAD_GRAYSCALE);

        second =
            cv::imread(
                options.frame2_path,
                cv::IMREAD_GRAYSCALE);

        if (first.empty() ||
            second.empty()) {
            std::cerr
                << "Error: could not read both frames.\n";
            return 1;
        }

        if (first.size() !=
            second.size()) {
            std::cerr
                << "Error: frame sizes must match.\n";
            return 1;
        }
    } else {
        const SyntheticFrames synthetic =
            createSyntheticFrames(
                960,
                540,
                options.synthetic_dx,
                options.synthetic_dy);

        first =
            synthetic.first;

        second =
            synthetic.second;

        synthetic_translation =
            synthetic.translation;

        synthetic_mode = true;
    }

    const auto points =
        detectFeatures(
            first,
            options.max_corners,
            options.quality_level,
            options.min_distance);

    if (points.empty()) {
        std::cerr
            << "Error: no trackable features were detected.\n";
        return 1;
    }

    const auto tracks =
        trackForwardBackward(
            first,
            second,
            points,
            options.window_size,
            options.max_level,
            options.fb_threshold);

    int valid_count = 0;

    for (const auto& track :
         tracks) {
        if (track.valid) {
            ++valid_count;
        }
    }

    const cv::Point2d motion =
        medianMotion(
            tracks);

    const double median_fb =
        medianForwardBackwardError(
            tracks);

    bool ok = true;

    ok &= writeImage(
        output_dir /
            "01_detected_features.png",
        drawFeatures(
            first,
            points));

    ok &= writeImage(
        output_dir /
            "02_all_forward_tracks.png",
        drawTracks(
            second,
            tracks,
            false));

    ok &= writeImage(
        output_dir /
            "03_fb_validated_tracks.png",
        drawTracks(
            second,
            tracks,
            true));

    ok &= writeImage(
        output_dir /
            "04_manual_single_scale_lk.png",
        drawManualSmallMotion(
            first,
            second,
            points));

    if (!ok) {
        return 1;
    }

    std::cout
        << std::fixed
        << std::setprecision(4)
        << "Detected features: "
        << points.size()
        << '\n'
        << "Forward/backward-valid tracks: "
        << valid_count
        << '\n'
        << "Retention ratio: "
        << (
            points.empty()
                ? 0.0
                : static_cast<double>(
                      valid_count) /
                  points.size())
        << '\n'
        << "Median motion: ("
        << motion.x
        << ", "
        << motion.y
        << ") px\n"
        << "Median forward/backward error: "
        << median_fb
        << " px\n"
        << "Pyramid max level: "
        << options.max_level
        << '\n'
        << "LK window: "
        << options.window_size
        << "x"
        << options.window_size
        << '\n';

    if (synthetic_mode) {
        std::cout
            << "Synthetic ground-truth motion: ("
            << synthetic_translation.x
            << ", "
            << synthetic_translation.y
            << ") px\n";
    }

    std::cout
        << "Outputs written to: "
        << output_dir
        << '\n';

    return 0;
}
