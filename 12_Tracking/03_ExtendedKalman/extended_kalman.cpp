#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kStateDim = 5;
constexpr int kMeasurementDim = 2;
constexpr double kTurnEpsilon = 1e-8;

struct Options {
    std::string output_dir;
    int frames = 180;
    double dt = 0.15;
    double measurement_sigma = 1.5;
    bool self_test = false;
    bool help = false;
};

struct StepRecord {
    int frame = 0;
    double time = 0.0;
    cv::Mat truth;
    cv::Point2d measurement;
    cv::Mat ekf_state;
    cv::Mat ekf_covariance;
    cv::Point2d cv_estimate;
    double nis = 0.0;
    double nees = 0.0;
};

struct Summary {
    double ekf_position_rmse = 0.0;
    double cv_position_rmse = 0.0;
    double mean_nis = 0.0;
    double mean_nees = 0.0;
    double nis_95_fraction = 0.0;
    double nees_95_fraction = 0.0;
};

double normalizeAngle(double angle) {
    while (angle > CV_PI) {
        angle -= 2.0 * CV_PI;
    }
    while (angle < -CV_PI) {
        angle += 2.0 * CV_PI;
    }
    return angle;
}

cv::Mat makeState(
    double px,
    double py,
    double speed,
    double heading,
    double yaw_rate) {
    return (cv::Mat_<double>(kStateDim, 1) <<
        px,
        py,
        speed,
        heading,
        yaw_rate);
}

cv::Mat ctrvPredictState(
    const cv::Mat& state,
    double dt) {
    CV_Assert(
        state.rows == kStateDim &&
        state.cols == 1 &&
        state.type() == CV_64F);

    const double px =
        state.at<double>(0, 0);
    const double py =
        state.at<double>(1, 0);
    const double speed =
        state.at<double>(2, 0);
    const double heading =
        state.at<double>(3, 0);
    const double yaw_rate =
        state.at<double>(4, 0);

    double next_x = px;
    double next_y = py;

    if (std::abs(yaw_rate) >
        kTurnEpsilon) {
        const double next_heading =
            heading +
            yaw_rate * dt;

        next_x +=
            speed / yaw_rate *
            (std::sin(next_heading) -
             std::sin(heading));

        next_y +=
            speed / yaw_rate *
            (-std::cos(next_heading) +
             std::cos(heading));
    } else {
        next_x +=
            speed *
            std::cos(heading) *
            dt;

        next_y +=
            speed *
            std::sin(heading) *
            dt;
    }

    return makeState(
        next_x,
        next_y,
        speed,
        normalizeAngle(
            heading +
            yaw_rate * dt),
        yaw_rate);
}

cv::Mat ctrvJacobian(
    const cv::Mat& state,
    double dt) {
    CV_Assert(
        state.rows == kStateDim &&
        state.cols == 1 &&
        state.type() == CV_64F);

    const double speed =
        state.at<double>(2, 0);
    const double heading =
        state.at<double>(3, 0);
    const double yaw_rate =
        state.at<double>(4, 0);

    cv::Mat jacobian =
        cv::Mat::eye(
            kStateDim,
            kStateDim,
            CV_64F);

    jacobian.at<double>(3, 4) =
        dt;

    if (std::abs(yaw_rate) >
        kTurnEpsilon) {
        const double next_heading =
            heading +
            yaw_rate * dt;

        const double sin_heading =
            std::sin(heading);

        const double cos_heading =
            std::cos(heading);

        const double sin_next =
            std::sin(next_heading);

        const double cos_next =
            std::cos(next_heading);

        const double yaw_squared =
            yaw_rate * yaw_rate;

        jacobian.at<double>(0, 2) =
            (sin_next -
             sin_heading) /
            yaw_rate;

        jacobian.at<double>(0, 3) =
            speed /
            yaw_rate *
            (cos_next -
             cos_heading);

        jacobian.at<double>(0, 4) =
            speed *
            (yaw_rate *
                 dt *
                 cos_next -
             (sin_next -
              sin_heading)) /
            yaw_squared;

        jacobian.at<double>(1, 2) =
            (-cos_next +
             cos_heading) /
            yaw_rate;

        jacobian.at<double>(1, 3) =
            speed /
            yaw_rate *
            (sin_next -
             sin_heading);

        jacobian.at<double>(1, 4) =
            speed *
            (yaw_rate *
                 dt *
                 sin_next -
             (cos_heading -
              cos_next)) /
            yaw_squared;
    } else {
        const double sin_heading =
            std::sin(heading);

        const double cos_heading =
            std::cos(heading);

        jacobian.at<double>(0, 2) =
            cos_heading * dt;

        jacobian.at<double>(0, 3) =
            -speed *
            sin_heading *
            dt;

        jacobian.at<double>(0, 4) =
            -0.5 *
            speed *
            sin_heading *
            dt *
            dt;

        jacobian.at<double>(1, 2) =
            sin_heading * dt;

        jacobian.at<double>(1, 3) =
            speed *
            cos_heading *
            dt;

        jacobian.at<double>(1, 4) =
            0.5 *
            speed *
            cos_heading *
            dt *
            dt;
    }

    return jacobian;
}

cv::Mat numericalJacobian(
    const cv::Mat& state,
    double dt,
    double epsilon = 1e-6) {
    cv::Mat jacobian(
        kStateDim,
        kStateDim,
        CV_64F,
        cv::Scalar(0));

    for (int column = 0;
         column < kStateDim;
         ++column) {
        cv::Mat plus =
            state.clone();

        cv::Mat minus =
            state.clone();

        plus.at<double>(
            column,
            0) += epsilon;

        minus.at<double>(
            column,
            0) -= epsilon;

        cv::Mat predicted_plus =
            ctrvPredictState(
                plus,
                dt);

        cv::Mat predicted_minus =
            ctrvPredictState(
                minus,
                dt);

        cv::Mat difference =
            (predicted_plus -
             predicted_minus) /
            (2.0 * epsilon);

        difference.at<double>(
            3,
            0) =
            normalizeAngle(
                predicted_plus.at<double>(
                    3,
                    0) -
                predicted_minus.at<double>(
                    3,
                    0)) /
            (2.0 * epsilon);

        difference.copyTo(
            jacobian.col(column));
    }

    return jacobian;
}

class ExtendedKalmanFilter {
public:
    ExtendedKalmanFilter(
        const cv::Mat& initial_state,
        const cv::Mat& initial_covariance,
        double measurement_sigma)
        : state_(initial_state.clone()),
          covariance_(
              initial_covariance.clone()) {
        measurement_matrix_ =
            cv::Mat::zeros(
                kMeasurementDim,
                kStateDim,
                CV_64F);

        measurement_matrix_.at<double>(
            0,
            0) = 1.0;

        measurement_matrix_.at<double>(
            1,
            1) = 1.0;

        measurement_noise_ =
            cv::Mat::eye(
                kMeasurementDim,
                kMeasurementDim,
                CV_64F) *
            (measurement_sigma *
             measurement_sigma);

        process_noise_ =
            cv::Mat::zeros(
                kStateDim,
                kStateDim,
                CV_64F);

        process_noise_.at<double>(0, 0) =
            0.01;

        process_noise_.at<double>(1, 1) =
            0.01;

        process_noise_.at<double>(2, 2) =
            0.03;

        process_noise_.at<double>(3, 3) =
            3e-4;

        process_noise_.at<double>(4, 4) =
            2e-4;
    }

    void predict(double dt) {
        const cv::Mat jacobian =
            ctrvJacobian(
                state_,
                dt);

        state_ =
            ctrvPredictState(
                state_,
                dt);

        covariance_ =
            jacobian *
            covariance_ *
            jacobian.t() +
            process_noise_;

        symmetrizeCovariance();
    }

    double update(
        const cv::Point2d& measurement) {
        cv::Mat z =
            (cv::Mat_<double>(
                kMeasurementDim,
                1) <<
                measurement.x,
                measurement.y);

        cv::Mat innovation =
            z -
            measurement_matrix_ *
                state_;

        cv::Mat innovation_covariance =
            measurement_matrix_ *
                covariance_ *
                measurement_matrix_.t() +
            measurement_noise_;

        cv::Mat innovation_inverse;

        if (cv::invert(
                innovation_covariance,
                innovation_inverse,
                cv::DECOMP_SVD) == 0.0) {
            throw std::runtime_error(
                "Innovation covariance is singular.");
        }

        const cv::Mat nis_matrix =
            innovation.t() *
            innovation_inverse *
            innovation;

        const double nis =
            nis_matrix.at<double>(
                0,
                0);

        const cv::Mat gain =
            covariance_ *
            measurement_matrix_.t() *
            innovation_inverse;

        state_ =
            state_ +
            gain *
            innovation;

        state_.at<double>(
            3,
            0) =
            normalizeAngle(
                state_.at<double>(
                    3,
                    0));

        const cv::Mat identity =
            cv::Mat::eye(
                kStateDim,
                kStateDim,
                CV_64F);

        const cv::Mat correction =
            identity -
            gain *
            measurement_matrix_;

        covariance_ =
            correction *
                covariance_ *
                correction.t() +
            gain *
                measurement_noise_ *
                gain.t();

        symmetrizeCovariance();

        return nis;
    }

    double nees(
        const cv::Mat& truth) const {
        cv::Mat error =
            state_ -
            truth;

        error.at<double>(
            3,
            0) =
            normalizeAngle(
                error.at<double>(
                    3,
                    0));

        cv::Mat inverse;

        if (cv::invert(
                covariance_,
                inverse,
                cv::DECOMP_SVD) == 0.0) {
            return std::numeric_limits<double>::
                infinity();
        }

        const cv::Mat nees_matrix =
            error.t() *
            inverse *
            error;

        return nees_matrix.at<double>(
            0,
            0);
    }

    const cv::Mat& state() const {
        return state_;
    }

    const cv::Mat& covariance() const {
        return covariance_;
    }

private:
    void symmetrizeCovariance() {
        covariance_ =
            0.5 *
            (covariance_ +
             covariance_.t());
    }

    cv::Mat state_;
    cv::Mat covariance_;
    cv::Mat measurement_matrix_;
    cv::Mat measurement_noise_;
    cv::Mat process_noise_;
};

class ConstantVelocityFilter {
public:
    ConstantVelocityFilter(
        const cv::Point2d& initial_position,
        double measurement_sigma,
        double dt)
        : filter_(
              4,
              2,
              0,
              CV_64F) {
        filter_.transitionMatrix =
            (cv::Mat_<double>(4, 4) <<
                1.0, 0.0, dt, 0.0,
                0.0, 1.0, 0.0, dt,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0);

        filter_.measurementMatrix =
            (cv::Mat_<double>(2, 4) <<
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0);

        cv::setIdentity(
            filter_.processNoiseCov,
            cv::Scalar::all(0.01));

        cv::setIdentity(
            filter_.measurementNoiseCov,
            cv::Scalar::all(
                measurement_sigma *
                measurement_sigma));

        cv::setIdentity(
            filter_.errorCovPost,
            cv::Scalar::all(10.0));

        filter_.statePost.at<double>(
            0,
            0) =
            initial_position.x;

        filter_.statePost.at<double>(
            1,
            0) =
            initial_position.y;
    }

    cv::Point2d step(
        const cv::Point2d& measurement) {
        const cv::Mat prediction =
            filter_.predict();

        cv::Mat z =
            (cv::Mat_<double>(2, 1) <<
                measurement.x,
                measurement.y);

        const cv::Mat corrected =
            filter_.correct(z);

        (void)prediction;

        return cv::Point2d(
            corrected.at<double>(
                0,
                0),
            corrected.at<double>(
                1,
                0));
    }

private:
    cv::KalmanFilter filter_;
};

std::vector<cv::Mat> generateTruth(
    int frame_count,
    double dt) {
    std::vector<cv::Mat> truth;

    truth.reserve(
        static_cast<std::size_t>(
            frame_count));

    cv::Mat state =
        makeState(
            80.0,
            80.0,
            15.0,
            0.25,
            0.32);

    for (int frame = 0;
         frame < frame_count;
         ++frame) {
        truth.push_back(
            state.clone());

        state =
            ctrvPredictState(
                state,
                dt);
    }

    return truth;
}

std::vector<cv::Point2d> generateMeasurements(
    const std::vector<cv::Mat>& truth,
    double sigma) {
    std::mt19937 generator(0x9A30U);

    std::normal_distribution<double>
        noise(
            0.0,
            sigma);

    std::vector<cv::Point2d>
        measurements;

    measurements.reserve(
        truth.size());

    for (const cv::Mat& state :
         truth) {
        measurements.emplace_back(
            state.at<double>(0, 0) +
                noise(generator),
            state.at<double>(1, 0) +
                noise(generator));
    }

    return measurements;
}

std::vector<StepRecord> runExperiment(
    const Options& options) {
    const auto truth =
        generateTruth(
            options.frames,
            options.dt);

    const auto measurements =
        generateMeasurements(
            truth,
            options.measurement_sigma);

    cv::Mat initial_state =
        makeState(
            measurements.front().x,
            measurements.front().y,
            13.5,
            0.35,
            0.24);

    cv::Mat initial_covariance =
        cv::Mat::zeros(
            kStateDim,
            kStateDim,
            CV_64F);

    initial_covariance.at<double>(0, 0) =
        4.0;

    initial_covariance.at<double>(1, 1) =
        4.0;

    initial_covariance.at<double>(2, 2) =
        6.0;

    initial_covariance.at<double>(3, 3) =
        0.12;

    initial_covariance.at<double>(4, 4) =
        0.06;

    ExtendedKalmanFilter ekf(
        initial_state,
        initial_covariance,
        options.measurement_sigma);

    ConstantVelocityFilter cv_filter(
        measurements.front(),
        options.measurement_sigma,
        options.dt);

    std::vector<StepRecord> records;

    records.reserve(
        truth.size());

    for (std::size_t frame = 0;
         frame < truth.size();
         ++frame) {
        double nis = 0.0;

        if (frame > 0U) {
            ekf.predict(
                options.dt);

            nis =
                ekf.update(
                    measurements[frame]);
        } else {
            nis =
                ekf.update(
                    measurements[frame]);
        }

        const cv::Point2d cv_estimate =
            cv_filter.step(
                measurements[frame]);

        StepRecord record;
        record.frame =
            static_cast<int>(
                frame);

        record.time =
            static_cast<double>(
                frame) *
            options.dt;

        record.truth =
            truth[frame].clone();

        record.measurement =
            measurements[frame];

        record.ekf_state =
            ekf.state().clone();

        record.ekf_covariance =
            ekf.covariance().clone();

        record.cv_estimate =
            cv_estimate;

        record.nis =
            nis;

        record.nees =
            ekf.nees(
                truth[frame]);

        records.push_back(
            record);
    }

    return records;
}

Summary summarize(
    const std::vector<StepRecord>& records,
    int warmup_frames = 20) {
    Summary summary;

    if (records.empty()) {
        return summary;
    }

    double ekf_squared_error = 0.0;
    double cv_squared_error = 0.0;
    double nis_sum = 0.0;
    double nees_sum = 0.0;

    int sample_count = 0;
    int nis_inside = 0;
    int nees_inside = 0;

    // 95% two-sided chi-square intervals.
    constexpr double nis_lower =
        0.0506;
    constexpr double nis_upper =
        7.3778;

    constexpr double nees_lower =
        0.8312;
    constexpr double nees_upper =
        12.8325;

    for (std::size_t index =
             static_cast<std::size_t>(
                 std::max(
                     0,
                     warmup_frames));
         index < records.size();
         ++index) {
        const auto& record =
            records[index];

        const double truth_x =
            record.truth.at<double>(
                0,
                0);

        const double truth_y =
            record.truth.at<double>(
                1,
                0);

        const double ekf_dx =
            record.ekf_state.at<double>(
                0,
                0) -
            truth_x;

        const double ekf_dy =
            record.ekf_state.at<double>(
                1,
                0) -
            truth_y;

        const double cv_dx =
            record.cv_estimate.x -
            truth_x;

        const double cv_dy =
            record.cv_estimate.y -
            truth_y;

        ekf_squared_error +=
            ekf_dx * ekf_dx +
            ekf_dy * ekf_dy;

        cv_squared_error +=
            cv_dx * cv_dx +
            cv_dy * cv_dy;

        if (std::isfinite(
                record.nis) &&
            std::isfinite(
                record.nees)) {
            nis_sum +=
                record.nis;

            nees_sum +=
                record.nees;

            if (record.nis >=
                    nis_lower &&
                record.nis <=
                    nis_upper) {
                ++nis_inside;
            }

            if (record.nees >=
                    nees_lower &&
                record.nees <=
                    nees_upper) {
                ++nees_inside;
            }

            ++sample_count;
        }
    }

    if (sample_count == 0) {
        return summary;
    }

    summary.ekf_position_rmse =
        std::sqrt(
            ekf_squared_error /
            sample_count);

    summary.cv_position_rmse =
        std::sqrt(
            cv_squared_error /
            sample_count);

    summary.mean_nis =
        nis_sum /
        sample_count;

    summary.mean_nees =
        nees_sum /
        sample_count;

    summary.nis_95_fraction =
        static_cast<double>(
            nis_inside) /
        sample_count;

    summary.nees_95_fraction =
        static_cast<double>(
            nees_inside) /
        sample_count;

    return summary;
}

bool covarianceLooksValid(
    const cv::Mat& covariance) {
    if (covariance.empty() ||
        covariance.rows !=
            kStateDim ||
        covariance.cols !=
            kStateDim) {
        return false;
    }

    const double symmetry_error =
        cv::norm(
            covariance -
            covariance.t(),
            cv::NORM_INF);

    if (!std::isfinite(
            symmetry_error) ||
        symmetry_error >
            1e-9) {
        return false;
    }

    for (int i = 0;
         i < kStateDim;
         ++i) {
        const double diagonal =
            covariance.at<double>(
                i,
                i);

        if (!std::isfinite(
                diagonal) ||
            diagonal <= 0.0) {
            return false;
        }
    }

    cv::Mat eigenvalues;

    if (!cv::eigen(
            covariance,
            eigenvalues)) {
        return false;
    }

    for (int i = 0;
         i < eigenvalues.rows;
         ++i) {
        if (eigenvalues.at<double>(
                i,
                0) <
            -1e-10) {
            return false;
        }
    }

    return true;
}

bool writeCsv(
    const fs::path& path,
    const std::vector<StepRecord>& records) {
    std::ofstream output(path);

    if (!output) {
        return false;
    }

    output
        << "frame,time,"
        << "truth_x,truth_y,truth_speed,truth_heading,truth_yaw_rate,"
        << "meas_x,meas_y,"
        << "ekf_x,ekf_y,ekf_speed,ekf_heading,ekf_yaw_rate,"
        << "cv_x,cv_y,nis,nees\n"
        << std::fixed
        << std::setprecision(8);

    for (const auto& record :
         records) {
        output
            << record.frame << ","
            << record.time << ","
            << record.truth.at<double>(0, 0) << ","
            << record.truth.at<double>(1, 0) << ","
            << record.truth.at<double>(2, 0) << ","
            << record.truth.at<double>(3, 0) << ","
            << record.truth.at<double>(4, 0) << ","
            << record.measurement.x << ","
            << record.measurement.y << ","
            << record.ekf_state.at<double>(0, 0) << ","
            << record.ekf_state.at<double>(1, 0) << ","
            << record.ekf_state.at<double>(2, 0) << ","
            << record.ekf_state.at<double>(3, 0) << ","
            << record.ekf_state.at<double>(4, 0) << ","
            << record.cv_estimate.x << ","
            << record.cv_estimate.y << ","
            << record.nis << ","
            << record.nees << "\n";
    }

    return true;
}

cv::Point mapPoint(
    double x,
    double y,
    double min_x,
    double min_y,
    double scale,
    int canvas_height,
    int margin) {
    return cv::Point(
        margin +
            cvRound(
                (x - min_x) *
                scale),
        canvas_height -
            margin -
            cvRound(
                (y - min_y) *
                scale));
}

cv::Mat renderTrajectory(
    const std::vector<StepRecord>& records) {
    constexpr int width = 1000;
    constexpr int height = 760;
    constexpr int margin = 60;

    double min_x =
        std::numeric_limits<double>::
            infinity();

    double min_y =
        std::numeric_limits<double>::
            infinity();

    double max_x =
        -std::numeric_limits<double>::
            infinity();

    double max_y =
        -std::numeric_limits<double>::
            infinity();

    for (const auto& record :
         records) {
        const std::vector<cv::Point2d>
            points = {
                cv::Point2d(
                    record.truth.at<double>(
                        0,
                        0),
                    record.truth.at<double>(
                        1,
                        0)),
                record.measurement,
                cv::Point2d(
                    record.ekf_state.at<double>(
                        0,
                        0),
                    record.ekf_state.at<double>(
                        1,
                        0)),
                record.cv_estimate
            };

        for (const auto& point :
             points) {
            min_x =
                std::min(
                    min_x,
                    point.x);

            min_y =
                std::min(
                    min_y,
                    point.y);

            max_x =
                std::max(
                    max_x,
                    point.x);

            max_y =
                std::max(
                    max_y,
                    point.y);
        }
    }

    const double range_x =
        std::max(
            1.0,
            max_x - min_x);

    const double range_y =
        std::max(
            1.0,
            max_y - min_y);

    const double scale =
        std::min(
            static_cast<double>(
                width -
                2 * margin) /
                range_x,
            static_cast<double>(
                height -
                2 * margin) /
                range_y);

    cv::Mat canvas(
        height,
        width,
        CV_8UC3,
        cv::Scalar(
            245,
            245,
            245));

    auto drawPath =
        [&](auto point_getter,
            const cv::Scalar& color,
            int thickness) {
            for (std::size_t i = 1;
                 i < records.size();
                 ++i) {
                const cv::Point2d a =
                    point_getter(
                        records[i - 1]);

                const cv::Point2d b =
                    point_getter(
                        records[i]);

                cv::line(
                    canvas,
                    mapPoint(
                        a.x,
                        a.y,
                        min_x,
                        min_y,
                        scale,
                        height,
                        margin),
                    mapPoint(
                        b.x,
                        b.y,
                        min_x,
                        min_y,
                        scale,
                        height,
                        margin),
                    color,
                    thickness,
                    cv::LINE_AA);
            }
        };

    drawPath(
        [](const StepRecord& record) {
            return cv::Point2d(
                record.truth.at<double>(
                    0,
                    0),
                record.truth.at<double>(
                    1,
                    0));
        },
        cv::Scalar(
            40,
            40,
            40),
        3);

    drawPath(
        [](const StepRecord& record) {
            return cv::Point2d(
                record.ekf_state.at<double>(
                    0,
                    0),
                record.ekf_state.at<double>(
                    1,
                    0));
        },
        cv::Scalar(
            30,
            170,
            30),
        2);

    drawPath(
        [](const StepRecord& record) {
            return record.cv_estimate;
        },
        cv::Scalar(
            220,
            100,
            30),
        2);

    for (std::size_t i = 0;
         i < records.size();
         i += 6U) {
        cv::circle(
            canvas,
            mapPoint(
                records[i].measurement.x,
                records[i].measurement.y,
                min_x,
                min_y,
                scale,
                height,
                margin),
            2,
            cv::Scalar(
                120,
                120,
                120),
            cv::FILLED,
            cv::LINE_AA);
    }

    cv::putText(
        canvas,
        "black=truth green=CTRV EKF orange=constant-velocity KF gray=measurements",
        cv::Point(
            25,
            32),
        cv::FONT_HERSHEY_SIMPLEX,
        0.58,
        cv::Scalar(
            40,
            40,
            40),
        1,
        cv::LINE_AA);

    return canvas;
}

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program
        << " --output-dir <dir> [options]\n"
        << "  " << program
        << " --self-test\n\n"
        << "Options:\n"
        << "  --output-dir <path>       Output directory\n"
        << "  --frames <N>              Number of synthetic frames (default: 180)\n"
        << "  --dt <seconds>            Sample period (default: 0.15)\n"
        << "  --measurement-sigma <px>  Position measurement sigma (default: 1.5)\n"
        << "  --self-test               Run deterministic tests\n"
        << "  --help                    Show this help\n";
}

bool parseArguments(
    int argc,
    char** argv,
    Options& options) {
    auto value =
        [argc, argv](
            int& index,
            const std::string& name) {
            if (index + 1 >= argc) {
                throw std::runtime_error(
                    "Missing value for " +
                    name);
            }

            return std::string(
                argv[++index]);
        };

    try {
        for (int i = 1;
             i < argc;
             ++i) {
            const std::string arg =
                argv[i];

            if (arg ==
                "--output-dir") {
                options.output_dir =
                    value(
                        i,
                        arg);
            } else if (
                arg ==
                "--frames") {
                options.frames =
                    std::stoi(
                        value(
                            i,
                            arg));
            } else if (
                arg ==
                "--dt") {
                options.dt =
                    std::stod(
                        value(
                            i,
                            arg));
            } else if (
                arg ==
                "--measurement-sigma") {
                options.measurement_sigma =
                    std::stod(
                        value(
                            i,
                            arg));
            } else if (
                arg ==
                "--self-test") {
                options.self_test =
                    true;
            } else if (
                arg ==
                    "--help" ||
                arg ==
                    "-h") {
                options.help =
                    true;
            } else {
                throw std::runtime_error(
                    "Unknown option: " +
                    arg);
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

bool validateOptions(
    const Options& options) {
    if (options.help ||
        options.self_test) {
        return true;
    }

    if (options.output_dir.empty()) {
        std::cerr
            << "Error: --output-dir is required.\n";

        return false;
    }

    if (options.frames < 40 ||
        options.dt <= 0.0 ||
        options.measurement_sigma <=
            0.0) {
        std::cerr
            << "Error: invalid EKF experiment configuration.\n";

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

    {
        const cv::Mat state =
            makeState(
                12.0,
                -4.0,
                8.0,
                0.7,
                0.23);

        const cv::Mat analytic =
            ctrvJacobian(
                state,
                0.15);

        const cv::Mat numeric =
            numericalJacobian(
                state,
                0.15);

        const double error =
            cv::norm(
                analytic -
                numeric,
                cv::NORM_INF);

        check(
            error <
                2e-5,
            "analytic CTRV Jacobian agrees with finite differences");
    }

    {
        const cv::Mat state =
            makeState(
                10.0,
                20.0,
                6.0,
                0.4,
                0.0);

        const cv::Mat predicted =
            ctrvPredictState(
                state,
                0.2);

        const double expected_x =
            10.0 +
            6.0 *
                std::cos(0.4) *
                0.2;

        const double expected_y =
            20.0 +
            6.0 *
                std::sin(0.4) *
                0.2;

        check(
            std::abs(
                predicted.at<double>(
                    0,
                    0) -
                expected_x) <
                1e-10 &&
            std::abs(
                predicted.at<double>(
                    1,
                    0) -
                expected_y) <
                1e-10,
            "zero-turn CTRV limit matches straight-line motion");
    }

    Options options;
    options.frames = 180;
    options.dt = 0.15;
    options.measurement_sigma = 1.5;

    const auto records =
        runExperiment(
            options);

    check(
        records.size() ==
            static_cast<std::size_t>(
                options.frames),
        "experiment returns one record per frame");

    bool covariance_valid = true;
    bool statistics_finite = true;

    for (const auto& record :
         records) {
        covariance_valid =
            covariance_valid &&
            covarianceLooksValid(
                record.ekf_covariance);

        statistics_finite =
            statistics_finite &&
            std::isfinite(
                record.nis) &&
            std::isfinite(
                record.nees) &&
            record.nis >= 0.0 &&
            record.nees >= 0.0;
    }

    check(
        covariance_valid,
        "EKF covariance remains symmetric positive semidefinite");

    check(
        statistics_finite,
        "NIS and NEES remain finite and non-negative");

    const Summary summary =
        summarize(
            records);

    check(
        summary.ekf_position_rmse <
            1.5,
        "CTRV EKF position RMSE remains bounded");

    check(
        summary.ekf_position_rmse <
            summary.cv_position_rmse,
        "CTRV EKF outperforms constant-velocity KF on turning motion");

    check(
        summary.mean_nis >
                0.2 &&
            summary.mean_nis <
                8.0,
        "mean NIS remains in a plausible range");

    check(
        summary.mean_nees >
                0.2 &&
            summary.mean_nees <
                25.0,
        "mean NEES remains in a plausible range");

    check(
        summary.nis_95_fraction >
            0.55,
        "majority of NIS samples lie inside 95% chi-square interval");

    check(
        summary.nees_95_fraction >
            0.45,
        "substantial fraction of NEES samples lie inside 95% chi-square interval");

    if (passed) {
        std::cout
            << "Extended Kalman filter self-test passed.\n"
            << std::fixed
            << std::setprecision(4)
            << "EKF RMSE="
            << summary.ekf_position_rmse
            << " CV RMSE="
            << summary.cv_position_rmse
            << " meanNIS="
            << summary.mean_nis
            << " meanNEES="
            << summary.mean_nees
            << " NIS95="
            << summary.nis_95_fraction
            << " NEES95="
            << summary.nees_95_fraction
            << '\n';
    }

    return passed;
}

}  // namespace

int main(
    int argc,
    char** argv) {
    Options options;

    if (!parseArguments(
            argc,
            argv,
            options)) {
        printUsage(
            argv[0]);

        return 2;
    }

    if (options.help) {
        printUsage(
            argv[0]);

        return 0;
    }

    if (!validateOptions(
            options)) {
        printUsage(
            argv[0]);

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

    const auto records =
        runExperiment(
            options);

    const Summary summary =
        summarize(
            records);

    if (!writeCsv(
            output_dir /
                "ekf_consistency.csv",
            records)) {
        std::cerr
            << "Error: failed to write EKF CSV.\n";

        return 1;
    }

    const cv::Mat trajectory =
        renderTrajectory(
            records);

    if (!cv::imwrite(
            (output_dir /
             "ekf_vs_cv_trajectory.png")
                .string(),
            trajectory)) {
        std::cerr
            << "Error: failed to write trajectory image.\n";

        return 1;
    }

    std::ofstream report(
        output_dir /
        "ekf_summary.txt");

    if (!report) {
        std::cerr
            << "Error: failed to write summary.\n";

        return 1;
    }

    report
        << std::fixed
        << std::setprecision(6)
        << "ekf_position_rmse="
        << summary.ekf_position_rmse
        << "\ncv_position_rmse="
        << summary.cv_position_rmse
        << "\nmean_nis="
        << summary.mean_nis
        << "\nmean_nees="
        << summary.mean_nees
        << "\nnis_95_fraction="
        << summary.nis_95_fraction
        << "\nnees_95_fraction="
        << summary.nees_95_fraction
        << "\n";

    std::cout
        << "CTRV Extended Kalman Filter experiment\n"
        << std::fixed
        << std::setprecision(4)
        << "  EKF position RMSE: "
        << summary.ekf_position_rmse
        << " px\n"
        << "  constant-velocity KF RMSE: "
        << summary.cv_position_rmse
        << " px\n"
        << "  mean NIS: "
        << summary.mean_nis
        << '\n'
        << "  mean NEES: "
        << summary.mean_nees
        << '\n'
        << "  NIS inside 95% interval: "
        << summary.nis_95_fraction *
               100.0
        << "%\n"
        << "  NEES inside 95% interval: "
        << summary.nees_95_fraction *
               100.0
        << "%\n"
        << "Outputs: "
        << output_dir
        << '\n';

    return 0;
}
