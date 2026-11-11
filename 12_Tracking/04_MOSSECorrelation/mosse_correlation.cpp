#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#ifdef CV9X_HAVE_OPENCV_TRACKING
#include <opencv2/tracking.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string output_dir;
    std::string video_path;
    std::string ground_truth_path;
    int max_frames = 0;
    int frames = 140;
    int width = 640;
    int height = 360;
    double learning_rate = 0.125;
    double psr_threshold = 7.0;
    bool self_test = false;
    bool help = false;
};

struct FrameSample {
    cv::Mat image;
    cv::Rect2d truth;
};

struct TrackerRecord {
    int frame = 0;
    std::string tracker;
    cv::Rect2d truth;
    cv::Rect2d estimate;
    double iou = 0.0;
    double center_error = 0.0;
    double psr = -1.0;
    double latency_ms = 0.0;
    bool ok = false;
};

struct TrackerSummary {
    std::string name;
    double mean_iou = 0.0;
    double precision_20px = 0.0;
    double success_50 = 0.0;
    double mean_center_error = 0.0;
    double mean_latency_ms = 0.0;
    int valid_frames = 0;
};

double rectIoU(const cv::Rect2d& a, const cv::Rect2d& b) {
    const cv::Rect2d intersection = a & b;
    const double inter_area = std::max(0.0, intersection.area());
    const double union_area = std::max(0.0, a.area()) + std::max(0.0, b.area()) - inter_area;
    return union_area > 0.0 ? inter_area / union_area : 0.0;
}

double centerError(const cv::Rect2d& a, const cv::Rect2d& b) {
    const cv::Point2d ca(a.x + 0.5 * a.width, a.y + 0.5 * a.height);
    const cv::Point2d cb(b.x + 0.5 * b.width, b.y + 0.5 * b.height);
    return cv::norm(ca - cb);
}

cv::Mat makeGaussianResponse(const cv::Size& size, double sigma) {
    cv::Mat response(size, CV_32F);
    const float cx = 0.5F * static_cast<float>(size.width - 1);
    const float cy = 0.5F * static_cast<float>(size.height - 1);
    const float denom = static_cast<float>(2.0 * sigma * sigma);

    for (int y = 0; y < size.height; ++y) {
        float* row = response.ptr<float>(y);
        for (int x = 0; x < size.width; ++x) {
            const float dx = static_cast<float>(x) - cx;
            const float dy = static_cast<float>(y) - cy;
            row[x] = std::exp(-(dx * dx + dy * dy) / denom);
        }
    }
    return response;
}

cv::Mat complexDivide(const cv::Mat& numerator, const cv::Mat& denominator, float epsilon) {
    CV_Assert(numerator.type() == CV_32FC2);
    CV_Assert(denominator.type() == CV_32FC2);
    CV_Assert(numerator.size() == denominator.size());

    std::vector<cv::Mat> num_channels;
    std::vector<cv::Mat> den_channels;
    cv::split(numerator, num_channels);
    cv::split(denominator, den_channels);

    cv::Mat den_power = den_channels[0].mul(den_channels[0]) +
                        den_channels[1].mul(den_channels[1]) + epsilon;

    cv::Mat real = (num_channels[0].mul(den_channels[0]) +
                    num_channels[1].mul(den_channels[1])) / den_power;
    cv::Mat imag = (num_channels[1].mul(den_channels[0]) -
                    num_channels[0].mul(den_channels[1])) / den_power;

    cv::Mat result;
    cv::merge(std::vector<cv::Mat>{real, imag}, result);
    return result;
}

cv::Mat preprocessPatch(const cv::Mat& gray, const cv::Mat& hann) {
    cv::Mat patch;
    gray.convertTo(patch, CV_32F);
    patch += 1.0F;
    cv::log(patch, patch);

    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(patch, mean, stddev);
    patch = (patch - static_cast<float>(mean[0])) /
            static_cast<float>(stddev[0] + 1e-5);

    return patch.mul(hann);
}

cv::Mat extractPatch(const cv::Mat& frame, const cv::Point2f& center, const cv::Size& size) {
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame;
    }

    cv::Mat patch;
    cv::getRectSubPix(gray, size, center, patch);
    return patch;
}

class MosseTracker {
public:
    MosseTracker(double learning_rate, double psr_threshold)
        : learning_rate_(learning_rate), psr_threshold_(psr_threshold) {}

    void init(const cv::Mat& frame, const cv::Rect2d& box) {
        if (box.width < 8.0 || box.height < 8.0) {
            throw std::runtime_error("MOSSE initialization box is too small.");
        }

        size_ = cv::Size(
            std::max(8, static_cast<int>(std::round(box.width))),
            std::max(8, static_cast<int>(std::round(box.height))));
        center_ = cv::Point2f(
            static_cast<float>(box.x + 0.5 * box.width),
            static_cast<float>(box.y + 0.5 * box.height));

        cv::createHanningWindow(hann_, size_, CV_32F);

        const double sigma =
            0.125 * std::sqrt(static_cast<double>(size_.width * size_.height));
        const cv::Mat gaussian = makeGaussianResponse(size_, sigma);
        cv::dft(gaussian, desired_frequency_, cv::DFT_COMPLEX_OUTPUT);

        numerator_ = cv::Mat::zeros(size_, CV_32FC2);
        denominator_ = cv::Mat::zeros(size_, CV_32FC2);

        std::mt19937 generator(0x31A4U);
        std::uniform_real_distribution<double> angle(-0.06, 0.06);
        std::uniform_real_distribution<double> scale(0.97, 1.03);
        std::uniform_real_distribution<double> offset(-2.0, 2.0);

        constexpr int kInitialSamples = 12;
        for (int i = 0; i < kInitialSamples; ++i) {
            cv::Mat patch = extractPatch(frame, center_, size_);
            if (i > 0) {
                const cv::Point2f c(
                    0.5F * static_cast<float>(size_.width - 1),
                    0.5F * static_cast<float>(size_.height - 1));
                cv::Mat warp = cv::getRotationMatrix2D(
                    c, angle(generator) * 180.0 / CV_PI, scale(generator));
                warp.at<double>(0, 2) += offset(generator);
                warp.at<double>(1, 2) += offset(generator);
                cv::warpAffine(
                    patch, patch, warp, size_, cv::INTER_LINEAR,
                    cv::BORDER_REFLECT_101);
            }
            accumulateTraining(patch, 1.0);
        }

        numerator_ /= static_cast<float>(kInitialSamples);
        denominator_ /= static_cast<float>(kInitialSamples);
        recomputeFilter();
        initialized_ = true;
    }

    bool update(const cv::Mat& frame, cv::Rect2d& box) {
        if (!initialized_) {
            throw std::runtime_error("MOSSE update called before init.");
        }

        const cv::Mat patch = extractPatch(frame, center_, size_);
        const cv::Mat processed = preprocessPatch(patch, hann_);

        cv::Mat frequency;
        cv::dft(processed, frequency, cv::DFT_COMPLEX_OUTPUT);

        cv::Mat response_frequency;
        cv::mulSpectrums(filter_, frequency, response_frequency, 0, false);

        cv::Mat response;
        cv::idft(
            response_frequency, response,
            cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

        double min_value = 0.0;
        double max_value = 0.0;
        cv::Point max_location;
        cv::minMaxLoc(response, &min_value, &max_value, nullptr, &max_location);
        (void)min_value;

        last_psr_ = computePsr(response, max_location, max_value);

        const cv::Point2f expected(
            0.5F * static_cast<float>(size_.width - 1),
            0.5F * static_cast<float>(size_.height - 1));
        cv::Point2f displacement(
            static_cast<float>(max_location.x) - expected.x,
            static_cast<float>(max_location.y) - expected.y);

        center_ += displacement;

        const bool confident =
            std::isfinite(last_psr_) && last_psr_ >= psr_threshold_;

        if (confident) {
            const cv::Mat aligned = extractPatch(frame, center_, size_);
            accumulateTraining(aligned, learning_rate_);
            recomputeFilter();
        }

        box = currentBox();
        return confident;
    }

    double lastPsr() const {
        return last_psr_;
    }

    cv::Rect2d currentBox() const {
        return cv::Rect2d(
            center_.x - 0.5 * static_cast<double>(size_.width),
            center_.y - 0.5 * static_cast<double>(size_.height),
            static_cast<double>(size_.width),
            static_cast<double>(size_.height));
    }

private:
    void accumulateTraining(const cv::Mat& patch, double weight) {
        const cv::Mat processed = preprocessPatch(patch, hann_);

        cv::Mat frequency;
        cv::dft(processed, frequency, cv::DFT_COMPLEX_OUTPUT);

        cv::Mat a;
        cv::mulSpectrums(
            desired_frequency_, frequency, a, 0, true);

        cv::Mat b;
        cv::mulSpectrums(
            frequency, frequency, b, 0, true);

        if (weight >= 0.999) {
            numerator_ += a;
            denominator_ += b;
        } else {
            numerator_ =
                (1.0 - weight) * numerator_ + weight * a;
            denominator_ =
                (1.0 - weight) * denominator_ + weight * b;
        }
    }

    void recomputeFilter() {
        filter_ = complexDivide(numerator_, denominator_, 1e-5F);
    }

    static double computePsr(
        const cv::Mat& response,
        const cv::Point& peak,
        double peak_value) {
        cv::Mat mask(response.size(), CV_8U, cv::Scalar(255));

        const int radius = 5;
        const cv::Rect exclusion(
            std::max(0, peak.x - radius),
            std::max(0, peak.y - radius),
            std::min(response.cols, peak.x + radius + 1) -
                std::max(0, peak.x - radius),
            std::min(response.rows, peak.y + radius + 1) -
                std::max(0, peak.y - radius));
        mask(exclusion).setTo(0);

        cv::Scalar mean;
        cv::Scalar stddev;
        cv::meanStdDev(response, mean, stddev, mask);
        return (peak_value - mean[0]) / (stddev[0] + 1e-6);
    }

    double learning_rate_ = 0.125;
    double psr_threshold_ = 7.0;
    bool initialized_ = false;
    cv::Size size_;
    cv::Point2f center_;
    cv::Mat hann_;
    cv::Mat desired_frequency_;
    cv::Mat numerator_;
    cv::Mat denominator_;
    cv::Mat filter_;
    double last_psr_ = 0.0;
};

cv::Mat makeTargetTexture(const cv::Size& size) {
    cv::Mat target(size, CV_8UC3, cv::Scalar(35, 35, 35));

    for (int y = 0; y < size.height; ++y) {
        for (int x = 0; x < size.width; ++x) {
            const int checker = ((x / 8) + (y / 8)) % 2;
            cv::Vec3b& pixel = target.at<cv::Vec3b>(y, x);
            if (checker == 0) {
                pixel = cv::Vec3b(220, 170, 60);
            } else {
                pixel = cv::Vec3b(70, 190, 230);
            }
        }
    }

    cv::circle(
        target,
        cv::Point(size.width / 2, size.height / 2),
        std::min(size.width, size.height) / 5,
        cv::Scalar(250, 250, 250),
        2,
        cv::LINE_AA);
    cv::line(
        target,
        cv::Point(5, 5),
        cv::Point(size.width - 6, size.height - 6),
        cv::Scalar(20, 20, 20),
        2,
        cv::LINE_AA);
    cv::line(
        target,
        cv::Point(size.width - 6, 5),
        cv::Point(5, size.height - 6),
        cv::Scalar(20, 20, 20),
        2,
        cv::LINE_AA);

    return target;
}

std::vector<FrameSample> makeSyntheticSequence(
    int frames,
    int width,
    int height) {
    const cv::Size target_size(72, 54);
    const cv::Mat target = makeTargetTexture(target_size);

    std::mt19937 generator(0x9C0FFEEU);
    std::normal_distribution<double> noise(0.0, 7.0);

    std::vector<FrameSample> sequence;
    sequence.reserve(static_cast<std::size_t>(frames));

    for (int frame_index = 0; frame_index < frames; ++frame_index) {
        cv::Mat image(height, width, CV_8UC3, cv::Scalar(55, 60, 65));

        for (int y = 0; y < height; y += 24) {
            cv::line(
                image,
                cv::Point(0, y),
                cv::Point(width - 1, y),
                cv::Scalar(70, 72, 74),
                1);
        }
        for (int x = 0; x < width; x += 31) {
            cv::line(
                image,
                cv::Point(x, 0),
                cv::Point(x, height - 1),
                cv::Scalar(45, 48, 52),
                1);
        }

        const double t = static_cast<double>(frame_index);
        const double x = 70.0 + 2.7 * t + 10.0 * std::sin(0.055 * t);
        const double y = 95.0 + 0.85 * t + 15.0 * std::sin(0.09 * t);

        const cv::Rect roi(
            static_cast<int>(std::round(x)),
            static_cast<int>(std::round(y)),
            target_size.width,
            target_size.height);

        target.copyTo(image(roi));

        for (int yy = 0; yy < image.rows; ++yy) {
            cv::Vec3b* row = image.ptr<cv::Vec3b>(yy);
            for (int xx = 0; xx < image.cols; ++xx) {
                for (int c = 0; c < 3; ++c) {
                    row[xx][c] = cv::saturate_cast<uchar>(
                        static_cast<double>(row[xx][c]) + noise(generator));
                }
            }
        }

        sequence.push_back({
            image,
            cv::Rect2d(
                static_cast<double>(roi.x),
                static_cast<double>(roi.y),
                static_cast<double>(roi.width),
                static_cast<double>(roi.height))});
    }

    return sequence;
}


std::vector<std::pair<int, cv::Rect2d>> loadGroundTruthCsv(
    const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Could not open ground-truth CSV: " + path);
    }

    std::vector<std::pair<int, cv::Rect2d>> truth;
    std::string line;
    int line_number = 0;

    while (std::getline(file, line)) {
        ++line_number;
        if (line.empty()) {
            continue;
        }

        std::stringstream stream(line);
        std::string token;
        std::vector<std::string> fields;

        while (std::getline(stream, token, ',')) {
            fields.push_back(token);
        }

        if (fields.size() != 5) {
            if (line_number == 1) {
                continue;
            }
            throw std::runtime_error(
                "Ground-truth CSV must contain frame,x,y,w,h at line " +
                std::to_string(line_number));
        }

        try {
            const int frame = std::stoi(fields[0]);
            const double x = std::stod(fields[1]);
            const double y = std::stod(fields[2]);
            const double width = std::stod(fields[3]);
            const double height = std::stod(fields[4]);

            if (frame < 0 || width <= 0.0 || height <= 0.0) {
                throw std::runtime_error("invalid geometry");
            }

            truth.emplace_back(
                frame,
                cv::Rect2d(x, y, width, height));
        } catch (const std::exception&) {
            if (line_number == 1) {
                continue;
            }
            throw std::runtime_error(
                "Invalid ground-truth row at line " +
                std::to_string(line_number));
        }
    }

    if (truth.empty()) {
        throw std::runtime_error("Ground-truth CSV contains no boxes.");
    }

    std::sort(
        truth.begin(),
        truth.end(),
        [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

    for (std::size_t i = 1; i < truth.size(); ++i) {
        if (truth[i].first != truth[i - 1].first + 1) {
            throw std::runtime_error(
                "Ground-truth frames must be contiguous for single-object tracking.");
        }
    }

    return truth;
}

std::vector<FrameSample> loadRealSequence(
    const std::string& video_path,
    const std::string& ground_truth_path,
    int max_frames) {
    const auto truth = loadGroundTruthCsv(ground_truth_path);

    cv::VideoCapture capture(video_path);
    if (!capture.isOpened()) {
        throw std::runtime_error("Could not open video: " + video_path);
    }

    const int first_frame = truth.front().first;
    const int last_frame = truth.back().first;
    const int requested_count =
        max_frames > 0
            ? std::min(max_frames, static_cast<int>(truth.size()))
            : static_cast<int>(truth.size());

    std::vector<FrameSample> sequence;
    sequence.reserve(static_cast<std::size_t>(requested_count));

    cv::Mat frame;
    int decoded_index = -1;
    std::size_t truth_index = 0;

    while (capture.read(frame)) {
        ++decoded_index;

        if (decoded_index < first_frame) {
            continue;
        }
        if (decoded_index > last_frame ||
            static_cast<int>(sequence.size()) >= requested_count) {
            break;
        }

        if (truth_index >= truth.size()) {
            break;
        }

        if (decoded_index == truth[truth_index].first) {
            const cv::Rect2d& box = truth[truth_index].second;

            if (box.x < 0.0 ||
                box.y < 0.0 ||
                box.x + box.width > static_cast<double>(frame.cols) ||
                box.y + box.height > static_cast<double>(frame.rows)) {
                throw std::runtime_error(
                    "Ground-truth box lies outside video frame at index " +
                    std::to_string(decoded_index));
            }

            sequence.push_back({frame.clone(), box});
            ++truth_index;
        }
    }

    if (sequence.size() < 2) {
        throw std::runtime_error(
            "Need at least two annotated video frames for tracking.");
    }

    if (static_cast<int>(sequence.size()) != requested_count) {
        throw std::runtime_error(
            "Video ended before all requested ground-truth frames were decoded.");
    }

    return sequence;
}

std::vector<TrackerRecord> evaluateMosse(
    const std::vector<FrameSample>& sequence,
    double learning_rate,
    double psr_threshold) {
    MosseTracker tracker(learning_rate, psr_threshold);
    tracker.init(sequence.front().image, sequence.front().truth);

    std::vector<TrackerRecord> records;
    records.reserve(sequence.size());

    for (std::size_t i = 0; i < sequence.size(); ++i) {
        cv::Rect2d estimate = sequence.front().truth;
        bool ok = true;
        double psr = 0.0;
        double latency_ms = 0.0;

        if (i > 0) {
            const auto start = std::chrono::steady_clock::now();
            ok = tracker.update(sequence[i].image, estimate);
            const auto end = std::chrono::steady_clock::now();
            latency_ms =
                std::chrono::duration<double, std::milli>(end - start).count();
            psr = tracker.lastPsr();
        }

        records.push_back({
            static_cast<int>(i),
            "MOSSE",
            sequence[i].truth,
            estimate,
            rectIoU(sequence[i].truth, estimate),
            centerError(sequence[i].truth, estimate),
            psr,
            latency_ms,
            ok});
    }

    return records;
}

#ifdef CV9X_HAVE_OPENCV_TRACKING
template <typename TrackerFactory>
std::vector<TrackerRecord> evaluateOpenCvTracker(
    const std::vector<FrameSample>& sequence,
    const std::string& name,
    TrackerFactory factory) {
    cv::Ptr<cv::Tracker> tracker = factory();
    cv::Rect initial(
        static_cast<int>(std::round(sequence.front().truth.x)),
        static_cast<int>(std::round(sequence.front().truth.y)),
        static_cast<int>(std::round(sequence.front().truth.width)),
        static_cast<int>(std::round(sequence.front().truth.height)));
    tracker->init(sequence.front().image, initial);

    cv::Rect estimate = initial;
    std::vector<TrackerRecord> records;
    records.reserve(sequence.size());

    for (std::size_t i = 0; i < sequence.size(); ++i) {
        bool ok = true;
        double latency_ms = 0.0;

        if (i > 0) {
            const auto start = std::chrono::steady_clock::now();
            ok = tracker->update(sequence[i].image, estimate);
            const auto end = std::chrono::steady_clock::now();
            latency_ms =
                std::chrono::duration<double, std::milli>(end - start).count();
        }

        const cv::Rect2d estimate_d(
            estimate.x, estimate.y, estimate.width, estimate.height);

        records.push_back({
            static_cast<int>(i),
            name,
            sequence[i].truth,
            estimate_d,
            rectIoU(sequence[i].truth, estimate_d),
            centerError(sequence[i].truth, estimate_d),
            -1.0,
            latency_ms,
            ok});
    }

    return records;
}
#endif

TrackerSummary summarize(
    const std::string& name,
    const std::vector<TrackerRecord>& records) {
    TrackerSummary summary;
    summary.name = name;

    if (records.empty()) {
        return summary;
    }

    double iou_sum = 0.0;
    double center_sum = 0.0;
    double latency_sum = 0.0;
    int precision_hits = 0;
    int success_hits = 0;
    int latency_count = 0;

    for (const auto& record : records) {
        iou_sum += record.iou;
        center_sum += record.center_error;
        precision_hits += record.center_error <= 20.0 ? 1 : 0;
        success_hits += record.iou >= 0.5 ? 1 : 0;
        if (record.frame > 0) {
            latency_sum += record.latency_ms;
            ++latency_count;
        }
        if (record.ok) {
            ++summary.valid_frames;
        }
    }

    const double count = static_cast<double>(records.size());
    summary.mean_iou = iou_sum / count;
    summary.precision_20px = static_cast<double>(precision_hits) / count;
    summary.success_50 = static_cast<double>(success_hits) / count;
    summary.mean_center_error = center_sum / count;
    summary.mean_latency_ms =
        latency_count > 0 ? latency_sum / static_cast<double>(latency_count) : 0.0;
    return summary;
}

bool writeRecords(
    const fs::path& path,
    const std::vector<std::vector<TrackerRecord>>& all_records) {
    std::ofstream file(path);
    if (!file) {
        return false;
    }

    file
        << "tracker,frame,truth_x,truth_y,truth_w,truth_h,"
        << "estimate_x,estimate_y,estimate_w,estimate_h,"
        << "iou,center_error_px,psr,latency_ms,ok\n";

    file << std::fixed << std::setprecision(6);
    for (const auto& records : all_records) {
        for (const auto& record : records) {
            file
                << record.tracker << ','
                << record.frame << ','
                << record.truth.x << ','
                << record.truth.y << ','
                << record.truth.width << ','
                << record.truth.height << ','
                << record.estimate.x << ','
                << record.estimate.y << ','
                << record.estimate.width << ','
                << record.estimate.height << ','
                << record.iou << ','
                << record.center_error << ','
                << record.psr << ','
                << record.latency_ms << ','
                << (record.ok ? 1 : 0)
                << '\n';
        }
    }
    return true;
}

bool writeSummary(
    const fs::path& path,
    const std::vector<TrackerSummary>& summaries) {
    std::ofstream file(path);
    if (!file) {
        return false;
    }

    file
        << "tracker,mean_iou,precision_20px,success_iou_0_5,"
        << "mean_center_error_px,mean_latency_ms,valid_frames\n";
    file << std::fixed << std::setprecision(6);

    for (const auto& summary : summaries) {
        file
            << summary.name << ','
            << summary.mean_iou << ','
            << summary.precision_20px << ','
            << summary.success_50 << ','
            << summary.mean_center_error << ','
            << summary.mean_latency_ms << ','
            << summary.valid_frames << '\n';
    }
    return true;
}

cv::Mat renderComparison(
    const std::vector<FrameSample>& sequence,
    const std::vector<std::vector<TrackerRecord>>& all_records) {
    cv::Mat canvas = sequence.back().image.clone();

    cv::rectangle(
        canvas,
        sequence.back().truth,
        cv::Scalar(255, 255, 255),
        2,
        cv::LINE_AA);
    cv::putText(
        canvas,
        "truth",
        cv::Point(
            static_cast<int>(sequence.back().truth.x),
            std::max(18, static_cast<int>(sequence.back().truth.y) - 8)),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(255, 255, 255),
        1,
        cv::LINE_AA);

    const std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 255, 0),
        cv::Scalar(0, 180, 255),
        cv::Scalar(255, 100, 80)};

    for (std::size_t i = 0; i < all_records.size(); ++i) {
        if (all_records[i].empty()) {
            continue;
        }
        const auto& record = all_records[i].back();
        const cv::Scalar color = colors[i % colors.size()];
        cv::rectangle(canvas, record.estimate, color, 2, cv::LINE_AA);
        cv::putText(
            canvas,
            record.tracker,
            cv::Point(
                static_cast<int>(record.estimate.x),
                std::min(
                    canvas.rows - 8,
                    static_cast<int>(record.estimate.y + record.estimate.height + 18))),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv::LINE_AA);
    }

    return canvas;
}

cv::Mat renderSuccessPlot(
    const std::vector<std::vector<TrackerRecord>>& all_records) {
    const int width = 760;
    const int height = 440;
    const int left = 70;
    const int right = 30;
    const int top = 35;
    const int bottom = 60;
    const int plot_w = width - left - right;
    const int plot_h = height - top - bottom;

    cv::Mat image(height, width, CV_8UC3, cv::Scalar(245, 245, 245));

    cv::line(
        image,
        cv::Point(left, top + plot_h),
        cv::Point(left + plot_w, top + plot_h),
        cv::Scalar(30, 30, 30),
        2);
    cv::line(
        image,
        cv::Point(left, top),
        cv::Point(left, top + plot_h),
        cv::Scalar(30, 30, 30),
        2);

    const std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 150, 0),
        cv::Scalar(0, 120, 220),
        cv::Scalar(220, 80, 40)};

    for (int tick = 0; tick <= 10; ++tick) {
        const int x = left + tick * plot_w / 10;
        const int y = top + plot_h - tick * plot_h / 10;
        cv::line(
            image,
            cv::Point(x, top + plot_h),
            cv::Point(x, top + plot_h + 5),
            cv::Scalar(30, 30, 30),
            1);
        cv::line(
            image,
            cv::Point(left - 5, y),
            cv::Point(left, y),
            cv::Scalar(30, 30, 30),
            1);
    }

    for (std::size_t tracker_index = 0;
         tracker_index < all_records.size();
         ++tracker_index) {
        const auto& records = all_records[tracker_index];
        if (records.empty()) {
            continue;
        }

        std::vector<cv::Point> points;
        for (int step = 0; step <= 100; ++step) {
            const double threshold = static_cast<double>(step) / 100.0;
            int hits = 0;
            for (const auto& record : records) {
                hits += record.iou >= threshold ? 1 : 0;
            }
            const double success =
                static_cast<double>(hits) / static_cast<double>(records.size());

            points.emplace_back(
                left + static_cast<int>(std::round(threshold * plot_w)),
                top + plot_h - static_cast<int>(std::round(success * plot_h)));
        }

        cv::polylines(
            image,
            points,
            false,
            colors[tracker_index % colors.size()],
            2,
            cv::LINE_AA);

        cv::putText(
            image,
            records.front().tracker,
            cv::Point(left + 20, top + 25 + static_cast<int>(tracker_index) * 24),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            colors[tracker_index % colors.size()],
            2,
            cv::LINE_AA);
    }

    cv::putText(
        image,
        "IoU threshold",
        cv::Point(width / 2 - 55, height - 18),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(30, 30, 30),
        1,
        cv::LINE_AA);

    cv::putText(
        image,
        "success rate",
        cv::Point(8, 24),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(30, 30, 30),
        1,
        cv::LINE_AA);

    return image;
}

cv::Mat renderPrecisionPlot(
    const std::vector<std::vector<TrackerRecord>>& all_records) {
    const int width = 760;
    const int height = 440;
    const int left = 70;
    const int right = 30;
    const int top = 35;
    const int bottom = 60;
    const int plot_w = width - left - right;
    const int plot_h = height - top - bottom;
    constexpr double kMaxThreshold = 50.0;

    cv::Mat image(height, width, CV_8UC3, cv::Scalar(245, 245, 245));

    cv::line(
        image,
        cv::Point(left, top + plot_h),
        cv::Point(left + plot_w, top + plot_h),
        cv::Scalar(30, 30, 30),
        2);
    cv::line(
        image,
        cv::Point(left, top),
        cv::Point(left, top + plot_h),
        cv::Scalar(30, 30, 30),
        2);

    const std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 150, 0),
        cv::Scalar(0, 120, 220),
        cv::Scalar(220, 80, 40)};

    for (std::size_t tracker_index = 0;
         tracker_index < all_records.size();
         ++tracker_index) {
        const auto& records = all_records[tracker_index];
        if (records.empty()) {
            continue;
        }

        std::vector<cv::Point> points;
        for (int step = 0; step <= 100; ++step) {
            const double threshold =
                kMaxThreshold * static_cast<double>(step) / 100.0;
            int hits = 0;
            for (const auto& record : records) {
                hits += record.center_error <= threshold ? 1 : 0;
            }

            const double precision =
                static_cast<double>(hits) /
                static_cast<double>(records.size());

            points.emplace_back(
                left + step * plot_w / 100,
                top + plot_h -
                    static_cast<int>(std::round(precision * plot_h)));
        }

        cv::polylines(
            image,
            points,
            false,
            colors[tracker_index % colors.size()],
            2,
            cv::LINE_AA);

        cv::putText(
            image,
            records.front().tracker,
            cv::Point(left + 20, top + 25 + static_cast<int>(tracker_index) * 24),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            colors[tracker_index % colors.size()],
            2,
            cv::LINE_AA);
    }

    cv::putText(
        image,
        "center-error threshold (px)",
        cv::Point(width / 2 - 95, height - 18),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(30, 30, 30),
        1,
        cv::LINE_AA);

    cv::putText(
        image,
        "precision",
        cv::Point(8, 24),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(30, 30, 30),
        1,
        cv::LINE_AA);

    return image;
}

bool runSelfTest() {
    bool passed = true;

    auto check =
        [&passed](bool condition, const std::string& name) {
            if (!condition) {
                std::cerr << "Self-test failed: " << name << '\n';
                passed = false;
            }
        };

    const cv::Mat gaussian = makeGaussianResponse(cv::Size(48, 36), 4.0);
    double max_value = 0.0;
    cv::Point max_location;
    cv::minMaxLoc(gaussian, nullptr, &max_value, nullptr, &max_location);
    check(
        std::abs(max_location.x - 23) <= 1 &&
        std::abs(max_location.y - 17) <= 1 &&
        max_value > 0.9,
        "desired Gaussian response peaks near the patch center");

    const auto sequence = makeSyntheticSequence(100, 520, 320);
    const auto mosse = evaluateMosse(sequence, 0.125, 6.0);
    const TrackerSummary mosse_summary = summarize("MOSSE", mosse);

    check(
        mosse.size() == sequence.size(),
        "MOSSE returns one record per frame");
    check(
        mosse_summary.mean_iou > 0.72,
        "MOSSE maintains high mean IoU on deterministic translation");
    check(
        mosse_summary.precision_20px > 0.92,
        "MOSSE center precision is high on deterministic sequence");
    check(
        mosse_summary.success_50 > 0.90,
        "MOSSE succeeds above IoU 0.5 on most frames");

    bool finite_psr = true;
    int high_psr_frames = 0;
    for (std::size_t i = 1; i < mosse.size(); ++i) {
        finite_psr = finite_psr && std::isfinite(mosse[i].psr);
        high_psr_frames += mosse[i].psr >= 6.0 ? 1 : 0;
    }
    check(finite_psr, "MOSSE PSR remains finite");
    check(
        high_psr_frames > static_cast<int>(0.50 * (mosse.size() - 1)),
        "MOSSE PSR accepts a majority of clean synthetic frames");

#ifdef CV9X_HAVE_OPENCV_TRACKING
    const auto kcf = evaluateOpenCvTracker(
        sequence,
        "KCF",
        []() { return cv::TrackerKCF::create(); });
    const auto csrt = evaluateOpenCvTracker(
        sequence,
        "CSRT",
        []() { return cv::TrackerCSRT::create(); });

    const TrackerSummary kcf_summary = summarize("KCF", kcf);
    const TrackerSummary csrt_summary = summarize("CSRT", csrt);

    check(kcf_summary.mean_iou > 0.45, "OpenCV KCF comparison remains functional");
    check(csrt_summary.mean_iou > 0.55, "OpenCV CSRT comparison remains functional");
#endif

    if (passed) {
        std::cout
            << "MOSSE self-test passed.\n"
            << std::fixed
            << std::setprecision(4)
            << "MOSSE mean IoU="
            << mosse_summary.mean_iou
            << " precision@20px="
            << mosse_summary.precision_20px
            << " success@0.5="
            << mosse_summary.success_50
            << '\n';
#ifdef CV9X_HAVE_OPENCV_TRACKING
        std::cout << "OpenCV contrib tracking comparison: enabled\n";
#else
        std::cout << "OpenCV contrib tracking comparison: unavailable; MOSSE baseline still tested\n";
#endif
    }

    return passed;
}

void printUsage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --output-dir <path>       Output directory\n"
        << "  --frames <N>              Synthetic sequence length (default: 140)\n"
        << "  --width <px>              Frame width (default: 640)\n"
        << "  --height <px>             Frame height (default: 360)\n"
        << "  --learning-rate <value>   MOSSE adaptation rate (default: 0.125)\n"
        << "  --psr-threshold <value>   MOSSE confidence threshold (default: 7.0)\n"
        << "  --self-test               Run deterministic correctness checks\n"
        << "  --help                    Show this help\n";
}

bool parseArguments(int argc, char** argv, Options& options) {
    auto value =
        [argc, argv](int& index, const std::string& name) {
            if (index + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return std::string(argv[++index]);
        };

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--output-dir") {
                options.output_dir = value(i, arg);
            } else if (arg == "--video") {
                options.video_path = value(i, arg);
            } else if (arg == "--ground-truth") {
                options.ground_truth_path = value(i, arg);
            } else if (arg == "--max-frames") {
                options.max_frames = std::stoi(value(i, arg));
            } else if (arg == "--frames") {
                options.frames = std::stoi(value(i, arg));
            } else if (arg == "--width") {
                options.width = std::stoi(value(i, arg));
            } else if (arg == "--height") {
                options.height = std::stoi(value(i, arg));
            } else if (arg == "--learning-rate") {
                options.learning_rate = std::stod(value(i, arg));
            } else if (arg == "--psr-threshold") {
                options.psr_threshold = std::stod(value(i, arg));
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

    const bool has_video = !options.video_path.empty();
    const bool has_truth = !options.ground_truth_path.empty();

    if (has_video != has_truth) {
        std::cerr
            << "Error: --video and --ground-truth must be provided together.\n";
        return false;
    }

    if (options.max_frames < 0 ||
        options.learning_rate <= 0.0 ||
        options.learning_rate > 1.0 ||
        options.psr_threshold <= 0.0) {
        std::cerr << "Error: invalid tracker experiment configuration.\n";
        return false;
    }

    if (!has_video &&
        (options.frames < 20 ||
         options.width < 320 ||
         options.height < 220)) {
        std::cerr << "Error: invalid synthetic experiment configuration.\n";
        return false;
    }

    return true;
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

    const fs::path output_dir(options.output_dir);
    std::error_code error;
    fs::create_directories(output_dir, error);
    if (error) {
        std::cerr
            << "Error: could not create output directory: "
            << error.message()
            << '\n';
        return 1;
    }

    std::vector<FrameSample> sequence;

    try {
        if (!options.video_path.empty()) {
            sequence = loadRealSequence(
                options.video_path,
                options.ground_truth_path,
                options.max_frames);
        } else {
            sequence =
                makeSyntheticSequence(
                    options.frames,
                    options.width,
                    options.height);
        }
    } catch (const std::exception& error) {
        std::cerr
            << "Error: "
            << error.what()
            << '\n';
        return 1;
    }

    std::vector<std::vector<TrackerRecord>> all_records;
    all_records.push_back(
        evaluateMosse(sequence, options.learning_rate, options.psr_threshold));

#ifdef CV9X_HAVE_OPENCV_TRACKING
    all_records.push_back(
        evaluateOpenCvTracker(
            sequence, "KCF",
            []() { return cv::TrackerKCF::create(); }));
    all_records.push_back(
        evaluateOpenCvTracker(
            sequence, "CSRT",
            []() { return cv::TrackerCSRT::create(); }));
#endif

    std::vector<TrackerSummary> summaries;
    summaries.reserve(all_records.size());
    for (const auto& records : all_records) {
        if (!records.empty()) {
            summaries.push_back(summarize(records.front().tracker, records));
        }
    }

    if (!writeRecords(output_dir / "tracker_frames.csv", all_records)) {
        std::cerr << "Error: failed to write tracker_frames.csv\n";
        return 1;
    }
    if (!writeSummary(output_dir / "tracker_summary.csv", summaries)) {
        std::cerr << "Error: failed to write tracker_summary.csv\n";
        return 1;
    }

    const cv::Mat overlay = renderComparison(sequence, all_records);
    const cv::Mat success_plot = renderSuccessPlot(all_records);
    const cv::Mat precision_plot = renderPrecisionPlot(all_records);

    if (!cv::imwrite(
            (output_dir / "tracker_comparison_last_frame.png").string(),
            overlay) ||
        !cv::imwrite(
            (output_dir / "success_plot.png").string(),
            success_plot) ||
        !cv::imwrite(
            (output_dir / "precision_plot.png").string(),
            precision_plot)) {
        std::cerr << "Error: failed to write comparison images\n";
        return 1;
    }

    std::cout
        << "Advanced Tracking A4/A5 - MOSSE / KCF / CSRT benchmark\n"
        << std::fixed
        << std::setprecision(4);

    for (const auto& summary : summaries) {
        std::cout
            << "  " << summary.name
            << ": mean IoU=" << summary.mean_iou
            << " precision@20px=" << summary.precision_20px
            << " success@0.5=" << summary.success_50
            << " center error=" << summary.mean_center_error << " px"
            << " latency=" << summary.mean_latency_ms << " ms\n";
    }

#ifdef CV9X_HAVE_OPENCV_TRACKING
    std::cout << "  OpenCV contrib tracking comparison: enabled\n";
#else
    std::cout
        << "  OpenCV contrib tracking comparison: not available in this OpenCV build\n"
        << "  MOSSE remains fully available and tested from first principles.\n";
#endif

    std::cout << "Outputs: " << output_dir << '\n';
    return 0;
}
