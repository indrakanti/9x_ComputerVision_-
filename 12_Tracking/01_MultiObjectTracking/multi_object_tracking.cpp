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
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr double kForbiddenCost = 1.0e6;

struct Detection {
    cv::Rect2d box;
    double score = 1.0;
    int class_id = 0;
};

struct GroundTruth {
    int id = -1;
    cv::Rect2d box;
};

enum class TrackState {
    Tentative,
    Confirmed,
    Coasting
};

struct TrackSnapshot {
    int id = -1;
    TrackState state = TrackState::Tentative;
    cv::Rect2d box;
    int age = 0;
    int hits = 0;
    int missed = 0;
};

struct Track {
    int id = -1;
    TrackState state = TrackState::Tentative;
    std::unique_ptr<cv::KalmanFilter> filter;
    cv::Size2d size;
    cv::Rect2d predicted_box;
    int age = 0;
    int hits = 0;
    int missed = 0;
};

struct Options {
    std::string output_dir;
    std::string detections_csv;
    std::string truth_csv;
    std::string video_path;
    int max_missed = 3;
    int min_hits = 2;
    double iou_gate = 0.05;
    double mahalanobis_gate = 9.21;
    double evaluation_iou = 0.30;
    bool self_test = false;
    bool help = false;
};

struct EvaluationSummary {
    int gt_instances = 0;
    int predicted_instances = 0;
    int matches = 0;
    int false_positives = 0;
    int false_negatives = 0;
    int id_switches = 0;
    double mean_iou = 0.0;
    double mean_center_error = 0.0;
    double mota = 0.0;
    double idf1 = 0.0;
    int idtp = 0;
    int idfp = 0;
    int idfn = 0;
};

std::string stateName(TrackState state) {
    switch (state) {
        case TrackState::Tentative:
            return "tentative";
        case TrackState::Confirmed:
            return "confirmed";
        case TrackState::Coasting:
            return "coasting";
    }
    return "unknown";
}

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " --output-dir <dir> [options]\n"
        << "  " << program << " --detections <csv> --output-dir <dir> [options]\n"
        << "  " << program << " --self-test\n\n"
        << "Options:\n"
        << "  --output-dir <path>        Output directory\n"
        << "  --detections <csv>         frame,x,y,w,h,score,class_id\n"
        << "  --truth <csv>              frame,id,x,y,w,h\n"
        << "  --video <path>             Optional video for overlay\n"
        << "  --max-missed <N>           Delete after N missed frames (default: 3)\n"
        << "  --min-hits <N>             Hits before confirmation (default: 2)\n"
        << "  --iou-gate <value>         Minimum IoU for association (default: 0.05)\n"
        << "  --mahal-gate <value>       Max squared Mahalanobis distance (default: 9.21)\n"
        << "  --eval-iou <value>         Evaluation match IoU (default: 0.30)\n"
        << "  --self-test                Run deterministic tests\n"
        << "  --help                     Show this help\n";
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
            if (arg == "--output-dir") {
                options.output_dir = value(i, arg);
            } else if (arg == "--detections") {
                options.detections_csv = value(i, arg);
            } else if (arg == "--truth") {
                options.truth_csv = value(i, arg);
            } else if (arg == "--video") {
                options.video_path = value(i, arg);
            } else if (arg == "--max-missed") {
                options.max_missed = std::stoi(value(i, arg));
            } else if (arg == "--min-hits") {
                options.min_hits = std::stoi(value(i, arg));
            } else if (arg == "--iou-gate") {
                options.iou_gate = std::stod(value(i, arg));
            } else if (arg == "--mahal-gate") {
                options.mahalanobis_gate = std::stod(value(i, arg));
            } else if (arg == "--eval-iou") {
                options.evaluation_iou = std::stod(value(i, arg));
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

    if (options.max_missed < 0 || options.min_hits < 1 ||
        options.iou_gate < 0.0 || options.iou_gate > 1.0 ||
        options.mahalanobis_gate <= 0.0 ||
        options.evaluation_iou < 0.0 || options.evaluation_iou > 1.0) {
        std::cerr << "Error: invalid tracker configuration.\n";
        return false;
    }

    return true;
}

cv::Point2d centerOf(const cv::Rect2d& box) {
    return cv::Point2d(
        box.x + 0.5 * box.width,
        box.y + 0.5 * box.height);
}

double centerDistance(const cv::Rect2d& a, const cv::Rect2d& b) {
    const cv::Point2d ca = centerOf(a);
    const cv::Point2d cb = centerOf(b);
    const double dx = ca.x - cb.x;
    const double dy = ca.y - cb.y;
    return std::sqrt(dx * dx + dy * dy);
}

double intersectionOverUnion(const cv::Rect2d& a, const cv::Rect2d& b) {
    const cv::Rect2d intersection = a & b;
    const double intersection_area =
        std::max(0.0, intersection.width) *
        std::max(0.0, intersection.height);

    const double union_area =
        a.area() + b.area() - intersection_area;

    if (union_area <= 0.0) {
        return 0.0;
    }

    return intersection_area / union_area;
}

std::unique_ptr<cv::KalmanFilter> makeFilter(const cv::Rect2d& box) {
    auto filter =
        std::make_unique<cv::KalmanFilter>(
            4,
            2,
            0,
            CV_64F);

    filter->transitionMatrix =
        (cv::Mat_<double>(4, 4) <<
            1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0);

    filter->measurementMatrix =
        (cv::Mat_<double>(2, 4) <<
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0);

    cv::setIdentity(
        filter->processNoiseCov,
        cv::Scalar::all(0.05));

    cv::setIdentity(
        filter->measurementNoiseCov,
        cv::Scalar::all(1.0));

    cv::setIdentity(
        filter->errorCovPost,
        cv::Scalar::all(10.0));

    const cv::Point2d center = centerOf(box);

    filter->statePost.at<double>(0, 0) = center.x;
    filter->statePost.at<double>(1, 0) = center.y;
    filter->statePost.at<double>(2, 0) = 0.0;
    filter->statePost.at<double>(3, 0) = 0.0;

    return filter;
}

cv::Rect2d boxFromState(
    const cv::Mat& state,
    const cv::Size2d& size) {
    const double cx = state.at<double>(0, 0);
    const double cy = state.at<double>(1, 0);

    return cv::Rect2d(
        cx - 0.5 * size.width,
        cy - 0.5 * size.height,
        size.width,
        size.height);
}

double mahalanobisSquared(
    const Track& track,
    const Detection& detection) {
    const cv::Point2d predicted_center =
        centerOf(track.predicted_box);

    const cv::Point2d measured_center =
        centerOf(detection.box);

    cv::Mat innovation =
        (cv::Mat_<double>(2, 1) <<
            measured_center.x - predicted_center.x,
            measured_center.y - predicted_center.y);

    cv::Mat covariance =
        track.filter->errorCovPre(
            cv::Rect(0, 0, 2, 2)).clone() +
        track.filter->measurementNoiseCov;

    cv::Mat inverse;
    if (cv::invert(
            covariance,
            inverse,
            cv::DECOMP_SVD) == 0.0) {
        return std::numeric_limits<double>::infinity();
    }

    const cv::Mat distance =
        innovation.t() *
        inverse *
        innovation;

    return distance.at<double>(0, 0);
}

std::vector<int> hungarianAssignment(
    const std::vector<std::vector<double>>& input_costs) {
    if (input_costs.empty()) {
        return {};
    }

    const int rows =
        static_cast<int>(
            input_costs.size());

    const int cols =
        input_costs.front().empty()
            ? 0
            : static_cast<int>(
                  input_costs.front().size());

    if (cols == 0) {
        return std::vector<int>(
            static_cast<std::size_t>(rows),
            -1);
    }

    const int n = std::max(rows, cols);

    std::vector<std::vector<double>> cost(
        static_cast<std::size_t>(n + 1),
        std::vector<double>(
            static_cast<std::size_t>(n + 1),
            kForbiddenCost));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cost[
                static_cast<std::size_t>(i + 1)]
                [static_cast<std::size_t>(j + 1)] =
                input_costs[
                    static_cast<std::size_t>(i)]
                    [static_cast<std::size_t>(j)];
        }
    }

    std::vector<double> u(
        static_cast<std::size_t>(n + 1),
        0.0);
    std::vector<double> v(
        static_cast<std::size_t>(n + 1),
        0.0);
    std::vector<int> p(
        static_cast<std::size_t>(n + 1),
        0);
    std::vector<int> way(
        static_cast<std::size_t>(n + 1),
        0);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;

        std::vector<double> minv(
            static_cast<std::size_t>(n + 1),
            std::numeric_limits<double>::infinity());

        std::vector<bool> used(
            static_cast<std::size_t>(n + 1),
            false);

        do {
            used[static_cast<std::size_t>(j0)] = true;
            const int i0 =
                p[static_cast<std::size_t>(j0)];

            double delta =
                std::numeric_limits<double>::infinity();
            int j1 = 0;

            for (int j = 1; j <= n; ++j) {
                if (used[static_cast<std::size_t>(j)]) {
                    continue;
                }

                const double cur =
                    cost[
                        static_cast<std::size_t>(i0)]
                        [static_cast<std::size_t>(j)] -
                    u[static_cast<std::size_t>(i0)] -
                    v[static_cast<std::size_t>(j)];

                if (cur <
                    minv[static_cast<std::size_t>(j)]) {
                    minv[static_cast<std::size_t>(j)] =
                        cur;
                    way[static_cast<std::size_t>(j)] =
                        j0;
                }

                if (minv[static_cast<std::size_t>(j)] <
                    delta) {
                    delta =
                        minv[static_cast<std::size_t>(j)];
                    j1 = j;
                }
            }

            for (int j = 0; j <= n; ++j) {
                if (used[static_cast<std::size_t>(j)]) {
                    u[
                        static_cast<std::size_t>(
                            p[static_cast<std::size_t>(j)])] +=
                        delta;
                    v[static_cast<std::size_t>(j)] -=
                        delta;
                } else {
                    minv[static_cast<std::size_t>(j)] -=
                        delta;
                }
            }

            j0 = j1;
        } while (
            p[static_cast<std::size_t>(j0)] != 0);

        do {
            const int j1 =
                way[static_cast<std::size_t>(j0)];

            p[static_cast<std::size_t>(j0)] =
                p[static_cast<std::size_t>(j1)];

            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<int> assignment(
        static_cast<std::size_t>(rows),
        -1);

    for (int j = 1; j <= n; ++j) {
        const int i =
            p[static_cast<std::size_t>(j)];

        if (i >= 1 &&
            i <= rows &&
            j <= cols) {
            const double original_cost =
                input_costs[
                    static_cast<std::size_t>(i - 1)]
                    [static_cast<std::size_t>(j - 1)];

            if (original_cost <
                0.5 * kForbiddenCost) {
                assignment[
                    static_cast<std::size_t>(i - 1)] =
                    j - 1;
            }
        }
    }

    return assignment;
}

class MultiObjectTracker {
public:
    MultiObjectTracker(
        int max_missed,
        int min_hits,
        double iou_gate,
        double mahalanobis_gate)
        : max_missed_(max_missed),
          min_hits_(min_hits),
          iou_gate_(iou_gate),
          mahalanobis_gate_(mahalanobis_gate) {}

    std::vector<TrackSnapshot> update(
        const std::vector<Detection>& detections) {
        predictTracks();

        std::vector<std::vector<double>> costs(
            tracks_.size(),
            std::vector<double>(
                detections.size(),
                kForbiddenCost));

        for (std::size_t i = 0;
             i < tracks_.size();
             ++i) {
            for (std::size_t j = 0;
                 j < detections.size();
                 ++j) {
                const double iou =
                    intersectionOverUnion(
                        tracks_[i].predicted_box,
                        detections[j].box);

                const double mahal =
                    mahalanobisSquared(
                        tracks_[i],
                        detections[j]);

                if (iou >= iou_gate_ &&
                    mahal <= mahalanobis_gate_) {
                    costs[i][j] =
                        1.0 - iou;
                }
            }
        }

        const std::vector<int> assignment =
            hungarianAssignment(costs);

        std::vector<bool> detection_used(
            detections.size(),
            false);

        for (std::size_t i = 0;
             i < tracks_.size();
             ++i) {
            const int detection_index =
                i < assignment.size()
                    ? assignment[i]
                    : -1;

            const bool matched =
                detection_index >= 0 &&
                detection_index <
                    static_cast<int>(
                        detections.size()) &&
                costs[i][
                    static_cast<std::size_t>(
                        detection_index)] <
                    0.5 * kForbiddenCost;

            if (matched) {
                correctTrack(
                    tracks_[i],
                    detections[
                        static_cast<std::size_t>(
                            detection_index)]);

                detection_used[
                    static_cast<std::size_t>(
                        detection_index)] =
                    true;
            } else {
                missTrack(
                    tracks_[i]);
            }
        }

        eraseDeadTracks();

        for (std::size_t i = 0;
             i < detections.size();
             ++i) {
            if (!detection_used[i]) {
                createTrack(
                    detections[i]);
            }
        }

        return snapshots();
    }

    std::size_t activeTrackCount() const {
        return tracks_.size();
    }

private:
    void predictTracks() {
        for (auto& track : tracks_) {
            const cv::Mat prediction =
                track.filter->predict();

            track.predicted_box =
                boxFromState(
                    prediction,
                    track.size);

            ++track.age;
        }
    }

    void correctTrack(
        Track& track,
        const Detection& detection) {
        const cv::Point2d center =
            centerOf(detection.box);

        cv::Mat measurement =
            (cv::Mat_<double>(2, 1) <<
                center.x,
                center.y);

        const cv::Mat corrected =
            track.filter->correct(
                measurement);

        track.size = cv::Size2d(
            detection.box.width,
            detection.box.height);

        track.predicted_box =
            boxFromState(
                corrected,
                track.size);

        ++track.hits;
        track.missed = 0;

        if (track.hits >= min_hits_) {
            track.state =
                TrackState::Confirmed;
        } else {
            track.state =
                TrackState::Tentative;
        }
    }

    void missTrack(Track& track) {
        ++track.missed;

        if (track.state ==
                TrackState::Confirmed ||
            track.state ==
                TrackState::Coasting) {
            track.state =
                TrackState::Coasting;
        }
    }

    void eraseDeadTracks() {
        tracks_.erase(
            std::remove_if(
                tracks_.begin(),
                tracks_.end(),
                [this](const Track& track) {
                    if (track.state ==
                            TrackState::Tentative &&
                        track.missed > 0) {
                        return true;
                    }

                    return track.missed >
                           max_missed_;
                }),
            tracks_.end());
    }

    void createTrack(
        const Detection& detection) {
        Track track;
        track.id = next_id_++;
        track.state =
            min_hits_ <= 1
                ? TrackState::Confirmed
                : TrackState::Tentative;
        track.filter =
            makeFilter(
                detection.box);
        track.size =
            detection.box.size();
        track.predicted_box =
            detection.box;
        track.age = 1;
        track.hits = 1;
        track.missed = 0;

        tracks_.push_back(
            std::move(track));
    }

    std::vector<TrackSnapshot> snapshots() const {
        std::vector<TrackSnapshot> output;

        for (const auto& track : tracks_) {
            if (track.state ==
                TrackState::Tentative) {
                continue;
            }

            TrackSnapshot snapshot;
            snapshot.id =
                track.id;
            snapshot.state =
                track.state;
            snapshot.box =
                track.predicted_box;
            snapshot.age =
                track.age;
            snapshot.hits =
                track.hits;
            snapshot.missed =
                track.missed;

            output.push_back(
                snapshot);
        }

        return output;
    }

    int max_missed_ = 3;
    int min_hits_ = 2;
    double iou_gate_ = 0.05;
    double mahalanobis_gate_ = 9.21;
    int next_id_ = 1;
    std::vector<Track> tracks_;
};

std::vector<std::string> splitCsvLine(
    const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;

    while (std::getline(
        stream,
        field,
        ',')) {
        fields.push_back(field);
    }

    return fields;
}

std::vector<std::vector<Detection>>
loadDetectionsCsv(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
            "Could not open detections CSV: " +
            path);
    }

    std::vector<std::vector<Detection>>
        frames;

    std::string line;

    while (std::getline(input, line)) {
        if (line.empty() ||
            line[0] == '#') {
            continue;
        }

        const auto fields =
            splitCsvLine(line);

        if (fields.size() < 5U ||
            !std::isdigit(
                static_cast<unsigned char>(
                    fields[0][0]))) {
            continue;
        }

        const int frame =
            std::stoi(fields[0]);

        if (frame < 0) {
            continue;
        }

        if (frames.size() <=
            static_cast<std::size_t>(frame)) {
            frames.resize(
                static_cast<std::size_t>(
                    frame + 1));
        }

        Detection detection;
        detection.box =
            cv::Rect2d(
                std::stod(fields[1]),
                std::stod(fields[2]),
                std::stod(fields[3]),
                std::stod(fields[4]));

        if (fields.size() > 5U) {
            detection.score =
                std::stod(fields[5]);
        }

        if (fields.size() > 6U) {
            detection.class_id =
                std::stoi(fields[6]);
        }

        frames[
            static_cast<std::size_t>(frame)]
            .push_back(
                detection);
    }

    return frames;
}

std::vector<std::vector<GroundTruth>>
loadTruthCsv(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
            "Could not open truth CSV: " +
            path);
    }

    std::vector<std::vector<GroundTruth>>
        frames;

    std::string line;

    while (std::getline(input, line)) {
        if (line.empty() ||
            line[0] == '#') {
            continue;
        }

        const auto fields =
            splitCsvLine(line);

        if (fields.size() < 6U ||
            !std::isdigit(
                static_cast<unsigned char>(
                    fields[0][0]))) {
            continue;
        }

        const int frame =
            std::stoi(fields[0]);

        if (frame < 0) {
            continue;
        }

        if (frames.size() <=
            static_cast<std::size_t>(frame)) {
            frames.resize(
                static_cast<std::size_t>(
                    frame + 1));
        }

        GroundTruth truth;
        truth.id =
            std::stoi(fields[1]);

        truth.box =
            cv::Rect2d(
                std::stod(fields[2]),
                std::stod(fields[3]),
                std::stod(fields[4]),
                std::stod(fields[5]));

        frames[
            static_cast<std::size_t>(frame)]
            .push_back(
                truth);
    }

    return frames;
}

struct SyntheticData {
    std::vector<std::vector<Detection>>
        detections;
    std::vector<std::vector<GroundTruth>>
        truth;
    cv::Size frame_size =
        cv::Size(480, 320);
};

SyntheticData makeSyntheticData(
    int frame_count) {
    SyntheticData data;

    data.detections.resize(
        static_cast<std::size_t>(
            frame_count));

    data.truth.resize(
        static_cast<std::size_t>(
            frame_count));

    std::mt19937 rng(42);
    std::normal_distribution<double>
        position_noise(0.0, 0.55);
    std::normal_distribution<double>
        size_noise(0.0, 0.30);

    struct ObjectModel {
        int id;
        cv::Point2d start;
        cv::Point2d velocity;
        cv::Size2d size;
    };

    const std::vector<ObjectModel> objects = {
        {1, cv::Point2d(30.0, 55.0),
         cv::Point2d(3.0, 1.0),
         cv::Size2d(36.0, 28.0)},
        {2, cv::Point2d(330.0, 65.0),
         cv::Point2d(-3.0, 1.0),
         cv::Size2d(36.0, 28.0)},
        {3, cv::Point2d(90.0, 245.0),
         cv::Point2d(2.0, -1.2),
         cv::Size2d(30.0, 40.0)}
    };

    for (int frame = 0;
         frame < frame_count;
         ++frame) {
        for (const auto& object :
             objects) {
            const cv::Point2d top_left =
                object.start +
                object.velocity *
                    static_cast<double>(
                        frame);

            GroundTruth truth;
            truth.id =
                object.id;
            truth.box =
                cv::Rect2d(
                    top_left.x,
                    top_left.y,
                    object.size.width,
                    object.size.height);

            data.truth[
                static_cast<std::size_t>(frame)]
                .push_back(
                    truth);

            const bool intentionally_missed =
                (object.id == 1 &&
                 (frame == 28 ||
                  frame == 29)) ||
                (object.id == 2 &&
                 frame == 42) ||
                (object.id == 3 &&
                 frame >= 55 &&
                 frame <= 57);

            if (intentionally_missed) {
                continue;
            }

            Detection detection;
            detection.box =
                cv::Rect2d(
                    truth.box.x +
                        position_noise(rng),
                    truth.box.y +
                        position_noise(rng),
                    truth.box.width +
                        size_noise(rng),
                    truth.box.height +
                        size_noise(rng));
            detection.score = 0.95;
            detection.class_id = 1;

            data.detections[
                static_cast<std::size_t>(frame)]
                .push_back(
                    detection);
        }

        if (frame == 20 ||
            frame == 60) {
            Detection false_positive;
            false_positive.box =
                cv::Rect2d(
                    410.0,
                    250.0,
                    24.0,
                    24.0);
            false_positive.score = 0.55;
            false_positive.class_id = 1;

            data.detections[
                static_cast<std::size_t>(frame)]
                .push_back(
                    false_positive);
        }

        std::shuffle(
            data.detections[
                static_cast<std::size_t>(frame)]
                .begin(),
            data.detections[
                static_cast<std::size_t>(frame)]
                .end(),
            rng);
    }

    return data;
}

std::vector<int> matchBoxesByIou(
    const std::vector<cv::Rect2d>& a,
    const std::vector<cv::Rect2d>& b,
    double threshold) {
    std::vector<std::vector<double>> costs(
        a.size(),
        std::vector<double>(
            b.size(),
            kForbiddenCost));

    for (std::size_t i = 0;
         i < a.size();
         ++i) {
        for (std::size_t j = 0;
             j < b.size();
             ++j) {
            const double iou =
                intersectionOverUnion(
                    a[i],
                    b[j]);

            if (iou >= threshold) {
                costs[i][j] =
                    1.0 - iou;
            }
        }
    }

    return hungarianAssignment(costs);
}

EvaluationSummary evaluateTracking(
    const std::vector<std::vector<TrackSnapshot>>& tracks,
    const std::vector<std::vector<GroundTruth>>& truth,
    double iou_threshold) {
    EvaluationSummary summary;

    const std::size_t frame_count =
        std::max(
            tracks.size(),
            truth.size());

    std::unordered_map<int, int>
        last_track_for_gt;

    std::map<std::pair<int, int>, int>
        identity_pair_counts;

    std::set<int> all_gt_ids;
    std::set<int> all_track_ids;

    double iou_sum = 0.0;
    double center_error_sum = 0.0;

    for (std::size_t frame = 0;
         frame < frame_count;
         ++frame) {
        const std::vector<TrackSnapshot>
            empty_tracks;

        const std::vector<GroundTruth>
            empty_truth;

        const auto& frame_tracks =
            frame < tracks.size()
                ? tracks[frame]
                : empty_tracks;

        const auto& frame_truth =
            frame < truth.size()
                ? truth[frame]
                : empty_truth;

        summary.predicted_instances +=
            static_cast<int>(
                frame_tracks.size());

        summary.gt_instances +=
            static_cast<int>(
                frame_truth.size());

        std::vector<cv::Rect2d>
            predicted_boxes;

        std::vector<cv::Rect2d>
            truth_boxes;

        for (const auto& track :
             frame_tracks) {
            predicted_boxes.push_back(
                track.box);
            all_track_ids.insert(
                track.id);
        }

        for (const auto& item :
             frame_truth) {
            truth_boxes.push_back(
                item.box);
            all_gt_ids.insert(
                item.id);
        }

        const auto assignment =
            matchBoxesByIou(
                truth_boxes,
                predicted_boxes,
                iou_threshold);

        std::vector<bool> track_used(
            frame_tracks.size(),
            false);

        int frame_matches = 0;

        for (std::size_t gt_index = 0;
             gt_index < frame_truth.size();
             ++gt_index) {
            const int track_index =
                gt_index <
                        assignment.size()
                    ? assignment[gt_index]
                    : -1;

            if (track_index < 0 ||
                track_index >=
                    static_cast<int>(
                        frame_tracks.size())) {
                continue;
            }

            const double iou =
                intersectionOverUnion(
                    frame_truth[gt_index].box,
                    frame_tracks[
                        static_cast<std::size_t>(
                            track_index)]
                        .box);

            if (iou < iou_threshold) {
                continue;
            }

            ++frame_matches;
            ++summary.matches;

            track_used[
                static_cast<std::size_t>(
                    track_index)] =
                true;

            iou_sum += iou;
            center_error_sum +=
                centerDistance(
                    frame_truth[gt_index].box,
                    frame_tracks[
                        static_cast<std::size_t>(
                            track_index)]
                        .box);

            const int gt_id =
                frame_truth[gt_index].id;

            const int track_id =
                frame_tracks[
                    static_cast<std::size_t>(
                        track_index)]
                    .id;

            const auto previous =
                last_track_for_gt.find(
                    gt_id);

            if (previous !=
                    last_track_for_gt.end() &&
                previous->second !=
                    track_id) {
                ++summary.id_switches;
            }

            last_track_for_gt[gt_id] =
                track_id;

            ++identity_pair_counts[
                std::make_pair(
                    gt_id,
                    track_id)];
        }

        summary.false_negatives +=
            static_cast<int>(
                frame_truth.size()) -
            frame_matches;

        summary.false_positives +=
            static_cast<int>(
                frame_tracks.size()) -
            frame_matches;
    }

    if (summary.matches > 0) {
        summary.mean_iou =
            iou_sum /
            summary.matches;

        summary.mean_center_error =
            center_error_sum /
            summary.matches;
    }

    if (summary.gt_instances > 0) {
        summary.mota =
            1.0 -
            static_cast<double>(
                summary.false_negatives +
                summary.false_positives +
                summary.id_switches) /
            summary.gt_instances;
    }

    const std::vector<int> gt_ids(
        all_gt_ids.begin(),
        all_gt_ids.end());

    const std::vector<int> track_ids(
        all_track_ids.begin(),
        all_track_ids.end());

    if (!gt_ids.empty() &&
        !track_ids.empty()) {
        int max_pair_count = 0;

        for (const auto& entry :
             identity_pair_counts) {
            max_pair_count =
                std::max(
                    max_pair_count,
                    entry.second);
        }

        std::vector<std::vector<double>>
            id_costs(
                gt_ids.size(),
                std::vector<double>(
                    track_ids.size(),
                    static_cast<double>(
                        max_pair_count)));

        for (std::size_t i = 0;
             i < gt_ids.size();
             ++i) {
            for (std::size_t j = 0;
                 j < track_ids.size();
                 ++j) {
                const auto found =
                    identity_pair_counts.find(
                        std::make_pair(
                            gt_ids[i],
                            track_ids[j]));

                const int count =
                    found ==
                            identity_pair_counts.end()
                        ? 0
                        : found->second;

                id_costs[i][j] =
                    static_cast<double>(
                        max_pair_count -
                        count);
            }
        }

        const auto id_assignment =
            hungarianAssignment(
                id_costs);

        for (std::size_t i = 0;
             i < id_assignment.size();
             ++i) {
            const int j =
                id_assignment[i];

            if (j < 0 ||
                j >= static_cast<int>(
                    track_ids.size())) {
                continue;
            }

            const auto found =
                identity_pair_counts.find(
                    std::make_pair(
                        gt_ids[i],
                        track_ids[
                            static_cast<std::size_t>(
                                j)]));

            if (found !=
                identity_pair_counts.end()) {
                summary.idtp +=
                    found->second;
            }
        }
    }

    summary.idfp =
        summary.predicted_instances -
        summary.idtp;

    summary.idfn =
        summary.gt_instances -
        summary.idtp;

    const int denominator =
        2 * summary.idtp +
        summary.idfp +
        summary.idfn;

    if (denominator > 0) {
        summary.idf1 =
            static_cast<double>(
                2 * summary.idtp) /
            denominator;
    }

    return summary;
}

cv::Scalar colorForId(int id) {
    static const std::vector<cv::Scalar>
        colors = {
            cv::Scalar(60, 220, 60),
            cv::Scalar(220, 100, 60),
            cv::Scalar(80, 160, 255),
            cv::Scalar(220, 80, 180),
            cv::Scalar(255, 220, 80)
        };

    return colors[
        static_cast<std::size_t>(
            std::abs(id) %
            static_cast<int>(
                colors.size()))];
}

cv::Mat renderSyntheticFrame(
    cv::Size frame_size,
    const std::vector<GroundTruth>& truth,
    const std::vector<Detection>& detections,
    const std::vector<TrackSnapshot>& tracks) {
    cv::Mat image(
        frame_size,
        CV_8UC3,
        cv::Scalar(24, 24, 24));

    for (const auto& item :
         truth) {
        cv::rectangle(
            image,
            item.box,
            cv::Scalar(110, 110, 110),
            1,
            cv::LINE_AA);
    }

    for (const auto& detection :
         detections) {
        cv::rectangle(
            image,
            detection.box,
            cv::Scalar(0, 220, 220),
            1,
            cv::LINE_AA);
    }

    for (const auto& track :
         tracks) {
        const cv::Scalar color =
            colorForId(
                track.id);

        cv::rectangle(
            image,
            track.box,
            color,
            2,
            cv::LINE_AA);

        cv::putText(
            image,
            "ID " +
                std::to_string(
                    track.id) +
                " " +
                stateName(
                    track.state),
            cv::Point(
                cvRound(track.box.x),
                std::max(
                    18,
                    cvRound(
                        track.box.y) -
                        5)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv::LINE_AA);
    }

    cv::putText(
        image,
        "gray=truth yellow=detection color=track",
        cv::Point(10, 20),
        cv::FONT_HERSHEY_SIMPLEX,
        0.45,
        cv::Scalar(240, 240, 240),
        1,
        cv::LINE_AA);

    return image;
}

bool writeTrackCsv(
    const fs::path& path,
    const std::vector<std::vector<TrackSnapshot>>& all_tracks) {
    std::ofstream output(path);
    if (!output) {
        return false;
    }

    output
        << "frame,track_id,state,x,y,w,h,age,hits,missed\n"
        << std::fixed
        << std::setprecision(4);

    for (std::size_t frame = 0;
         frame < all_tracks.size();
         ++frame) {
        for (const auto& track :
             all_tracks[frame]) {
            output
                << frame << ","
                << track.id << ","
                << stateName(track.state) << ","
                << track.box.x << ","
                << track.box.y << ","
                << track.box.width << ","
                << track.box.height << ","
                << track.age << ","
                << track.hits << ","
                << track.missed << "\n";
        }
    }

    return true;
}

bool writeSummary(
    const fs::path& path,
    const EvaluationSummary& summary) {
    std::ofstream output(path);
    if (!output) {
        return false;
    }

    output
        << std::fixed
        << std::setprecision(6)
        << "gt_instances=" << summary.gt_instances << "\n"
        << "predicted_instances=" << summary.predicted_instances << "\n"
        << "matches=" << summary.matches << "\n"
        << "false_positives=" << summary.false_positives << "\n"
        << "false_negatives=" << summary.false_negatives << "\n"
        << "id_switches=" << summary.id_switches << "\n"
        << "mean_iou=" << summary.mean_iou << "\n"
        << "mean_center_error=" << summary.mean_center_error << "\n"
        << "mota=" << summary.mota << "\n"
        << "idtp=" << summary.idtp << "\n"
        << "idfp=" << summary.idfp << "\n"
        << "idfn=" << summary.idfn << "\n"
        << "idf1=" << summary.idf1 << "\n";

    return true;
}

std::vector<std::vector<TrackSnapshot>>
runTracker(
    const std::vector<std::vector<Detection>>& detections,
    const Options& options) {
    MultiObjectTracker tracker(
        options.max_missed,
        options.min_hits,
        options.iou_gate,
        options.mahalanobis_gate);

    std::vector<std::vector<TrackSnapshot>>
        all_tracks;

    all_tracks.reserve(
        detections.size());

    for (const auto& frame_detections :
         detections) {
        all_tracks.push_back(
            tracker.update(
                frame_detections));
    }

    return all_tracks;
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
        std::abs(
            intersectionOverUnion(
                cv::Rect2d(0, 0, 10, 10),
                cv::Rect2d(0, 0, 10, 10)) -
            1.0) <
            1e-12,
        "identical boxes have IoU 1");

    check(
        intersectionOverUnion(
            cv::Rect2d(0, 0, 10, 10),
            cv::Rect2d(20, 20, 10, 10)) ==
            0.0,
        "separated boxes have IoU 0");

    {
        const std::vector<std::vector<double>>
            costs = {
                {4.0, 1.0, 3.0},
                {2.0, 0.0, 5.0},
                {3.0, 2.0, 2.0}
            };

        const auto assignment =
            hungarianAssignment(
                costs);

        check(
            assignment.size() == 3U &&
            assignment[0] == 1 &&
            assignment[1] == 0 &&
            assignment[2] == 2,
            "Hungarian assignment finds known minimum-cost solution");
    }

    {
        MultiObjectTracker tracker(
            2,
            2,
            0.01,
            25.0);

        std::vector<Detection> detections = {
            {cv::Rect2d(10, 10, 20, 20),
             1.0,
             0}
        };

        const auto first =
            tracker.update(
                detections);

        check(
            first.empty(),
            "new track remains tentative before minimum hits");

        detections[0].box.x += 1.0;

        const auto second =
            tracker.update(
                detections);

        check(
            second.size() == 1U &&
            second[0].id == 1 &&
            second[0].state ==
                TrackState::Confirmed,
            "track confirms after configured hit count");

        tracker.update({});
        tracker.update({});

        const auto after_third_miss =
            tracker.update({});

        check(
            after_third_miss.empty() &&
            tracker.activeTrackCount() == 0U,
            "track is deleted after exceeding miss budget");
    }

    const SyntheticData synthetic =
        makeSyntheticData(80);

    Options options;
    options.max_missed = 3;
    options.min_hits = 2;
    options.iou_gate = 0.02;
    options.mahalanobis_gate = 16.0;
    options.evaluation_iou = 0.30;

    const auto tracks =
        runTracker(
            synthetic.detections,
            options);

    const EvaluationSummary summary =
        evaluateTracking(
            tracks,
            synthetic.truth,
            options.evaluation_iou);

    check(
        summary.mean_iou >= 0.80,
        "SORT-style tracker mean IoU is at least 0.80");

    check(
        summary.mean_center_error <= 3.0,
        "SORT-style tracker mean center error is at most 3 px");

    check(
        summary.mota >= 0.90,
        "SORT-style tracker MOTA is at least 0.90");

    check(
        summary.idf1 >= 0.90,
        "SORT-style tracker IDF1 is at least 0.90");

    check(
        summary.id_switches <= 1,
        "crossing sequence preserves identities with at most one switch");

    if (passed) {
        std::cout
            << "Multi-object tracking self-test passed.\n"
            << std::fixed
            << std::setprecision(4)
            << "MOTA=" << summary.mota
            << " IDF1=" << summary.idf1
            << " meanIoU=" << summary.mean_iou
            << " centerError=" << summary.mean_center_error
            << " idSwitches=" << summary.id_switches
            << '\n';
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

    fs::path output_dir(
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

    std::vector<std::vector<Detection>>
        detections;

    std::vector<std::vector<GroundTruth>>
        truth;

    cv::Size synthetic_frame_size(
        480,
        320);

    bool synthetic_mode =
        options.detections_csv.empty();

    if (synthetic_mode) {
        const SyntheticData synthetic =
            makeSyntheticData(80);

        detections =
            synthetic.detections;

        truth =
            synthetic.truth;

        synthetic_frame_size =
            synthetic.frame_size;
    } else {
        try {
            detections =
                loadDetectionsCsv(
                    options.detections_csv);

            if (!options.truth_csv.empty()) {
                truth =
                    loadTruthCsv(
                        options.truth_csv);
            }
        } catch (const std::exception& exception) {
            std::cerr
                << "Input error: "
                << exception.what()
                << '\n';
            return 1;
        }
    }

    const auto tracks =
        runTracker(
            detections,
            options);

    if (!writeTrackCsv(
            output_dir /
                "mot_tracks.csv",
            tracks)) {
        std::cerr
            << "Error: failed to write mot_tracks.csv\n";
        return 1;
    }

    EvaluationSummary summary;

    if (!truth.empty()) {
        summary =
            evaluateTracking(
                tracks,
                truth,
                options.evaluation_iou);

        if (!writeSummary(
                output_dir /
                    "tracking_summary.txt",
                summary)) {
            std::cerr
                << "Error: failed to write tracking summary\n";
            return 1;
        }
    }

    if (synthetic_mode) {
        const fs::path video_path =
            output_dir /
            "multi_object_tracking.avi";

        cv::VideoWriter writer(
            video_path.string(),
            cv::VideoWriter::fourcc(
                'M',
                'J',
                'P',
                'G'),
            20.0,
            synthetic_frame_size,
            true);

        if (writer.isOpened()) {
            for (std::size_t frame = 0;
                 frame < tracks.size();
                 ++frame) {
                writer.write(
                    renderSyntheticFrame(
                        synthetic_frame_size,
                        truth[frame],
                        detections[frame],
                        tracks[frame]));
            }

            writer.release();
        }
    }

    if (!options.video_path.empty()) {
        cv::VideoCapture capture(
            options.video_path);

        if (!capture.isOpened()) {
            std::cerr
                << "Warning: could not open overlay video: "
                << options.video_path
                << '\n';
        } else {
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

            cv::VideoWriter writer(
                (output_dir /
                 "tracking_overlay.avi")
                    .string(),
                cv::VideoWriter::fourcc(
                    'M',
                    'J',
                    'P',
                    'G'),
                fps,
                frame_size,
                true);

            cv::Mat frame_image;
            std::size_t frame = 0;

            while (
                frame < tracks.size() &&
                capture.read(frame_image)) {
                for (const auto& track :
                     tracks[frame]) {
                    const cv::Scalar color =
                        colorForId(
                            track.id);

                    cv::rectangle(
                        frame_image,
                        track.box,
                        color,
                        2,
                        cv::LINE_AA);

                    cv::putText(
                        frame_image,
                        "ID " +
                            std::to_string(
                                track.id),
                        cv::Point(
                            cvRound(
                                track.box.x),
                            std::max(
                                18,
                                cvRound(
                                    track.box.y) -
                                    5)),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv::LINE_AA);
                }

                if (writer.isOpened()) {
                    writer.write(
                        frame_image);
                }

                ++frame;
            }
        }
    }

    std::cout
        << "Multi-object tracker\n"
        << "  frames: "
        << detections.size()
        << '\n'
        << "  max missed: "
        << options.max_missed
        << '\n'
        << "  min hits: "
        << options.min_hits
        << '\n'
        << "  IoU gate: "
        << options.iou_gate
        << '\n'
        << "  Mahalanobis gate: "
        << options.mahalanobis_gate
        << '\n';

    if (!truth.empty()) {
        std::cout
            << std::fixed
            << std::setprecision(4)
            << "  MOTA: "
            << summary.mota
            << '\n'
            << "  IDF1: "
            << summary.idf1
            << '\n'
            << "  mean IoU: "
            << summary.mean_iou
            << '\n'
            << "  mean center error: "
            << summary.mean_center_error
            << " px\n"
            << "  ID switches: "
            << summary.id_switches
            << '\n';
    }

    std::cout
        << "Outputs: "
        << output_dir
        << '\n';

    return 0;
}
