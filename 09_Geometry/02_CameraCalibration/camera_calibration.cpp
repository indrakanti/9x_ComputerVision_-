#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string output_dir;
    int width = 1280;
    int height = 720;
    int board_cols = 9;
    int board_rows = 6;
    double square_size = 0.04;
    int views = 12;
    double noise_sigma_px = 0.15;
    unsigned int seed = 42;
    bool self_test = false;
    bool help = false;
};

struct GroundTruthCamera {
    cv::Mat camera_matrix;
    cv::Mat distortion;
};

struct CalibrationData {
    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;
    std::vector<cv::Mat> true_rvecs;
    std::vector<cv::Mat> true_tvecs;
};

struct CalibrationResult {
    cv::Mat camera_matrix;
    cv::Mat distortion;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    double optimizer_rms = 0.0;
    std::vector<double> per_view_rmse;
    double global_rmse = 0.0;
    double max_residual = 0.0;
};

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " --output-dir <dir> [options]\n"
        << "  " << program << " --self-test\n\n"
        << "Options:\n"
        << "  --output-dir <path>      Output directory\n"
        << "  --width <pixels>         Image width (default: 1280)\n"
        << "  --height <pixels>        Image height (default: 720)\n"
        << "  --board-cols <N>         Inner corners per row (default: 9)\n"
        << "  --board-rows <N>         Inner corners per column (default: 6)\n"
        << "  --square-size <meters>   Target square spacing (default: 0.04)\n"
        << "  --views <N>              Synthetic calibration views (default: 12)\n"
        << "  --noise <pixels>         Gaussian corner noise sigma (default: 0.15)\n"
        << "  --seed <N>               Deterministic noise seed (default: 42)\n"
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
            } else if (arg == "--width") {
                options.width = std::stoi(value(i, arg));
            } else if (arg == "--height") {
                options.height = std::stoi(value(i, arg));
            } else if (arg == "--board-cols") {
                options.board_cols = std::stoi(value(i, arg));
            } else if (arg == "--board-rows") {
                options.board_rows = std::stoi(value(i, arg));
            } else if (arg == "--square-size") {
                options.square_size = std::stod(value(i, arg));
            } else if (arg == "--views") {
                options.views = std::stoi(value(i, arg));
            } else if (arg == "--noise") {
                options.noise_sigma_px = std::stod(value(i, arg));
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

    if (options.width <= 0 ||
        options.height <= 0 ||
        options.board_cols < 3 ||
        options.board_rows < 3 ||
        options.square_size <= 0.0 ||
        options.views < 5 ||
        options.noise_sigma_px < 0.0) {
        std::cerr
            << "Error: invalid image/board/view/noise configuration.\n";
        return false;
    }

    return true;
}

GroundTruthCamera makeGroundTruthCamera(
    int width,
    int height) {
    GroundTruthCamera camera;

    camera.camera_matrix =
        (cv::Mat_<double>(3, 3) <<
            910.0, 0.0, width * 0.5 + 4.0,
            0.0, 895.0, height * 0.5 - 3.0,
            0.0, 0.0, 1.0);

    camera.distortion =
        (cv::Mat_<double>(1, 5) <<
            -0.10,
             0.020,
             0.0012,
            -0.0008,
             0.0);

    return camera;
}

std::vector<cv::Point3f> createBoardPoints(
    int cols,
    int rows,
    double square_size) {
    std::vector<cv::Point3f> points;
    points.reserve(
        static_cast<std::size_t>(cols * rows));

    const double center_x =
        0.5 * static_cast<double>(cols - 1);
    const double center_y =
        0.5 * static_cast<double>(rows - 1);

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            points.emplace_back(
                static_cast<float>(
                    (col - center_x) * square_size),
                static_cast<float>(
                    (row - center_y) * square_size),
                0.0f);
        }
    }

    return points;
}

cv::Vec3d viewEulerDegrees(int index) {
    const double roll =
        -8.0 + 4.0 * static_cast<double>(index % 5);
    const double pitch =
        -13.0 + 5.5 * static_cast<double>((index * 2) % 6);
    const double yaw =
        -12.0 + 6.0 * static_cast<double>((index * 3) % 5);

    return cv::Vec3d(roll, pitch, yaw);
}

cv::Matx33d rotationFromEulerDegrees(
    const cv::Vec3d& euler_deg) {
    const double roll =
        euler_deg[0] * CV_PI / 180.0;
    const double pitch =
        euler_deg[1] * CV_PI / 180.0;
    const double yaw =
        euler_deg[2] * CV_PI / 180.0;

    const double cr = std::cos(roll);
    const double sr = std::sin(roll);
    const double cp = std::cos(pitch);
    const double sp = std::sin(pitch);
    const double cy = std::cos(yaw);
    const double sy = std::sin(yaw);

    const cv::Matx33d rx(
        1.0, 0.0, 0.0,
        0.0, cr, -sr,
        0.0, sr, cr);

    const cv::Matx33d ry(
        cp, 0.0, sp,
        0.0, 1.0, 0.0,
        -sp, 0.0, cp);

    const cv::Matx33d rz(
        cy, -sy, 0.0,
        sy, cy, 0.0,
        0.0, 0.0, 1.0);

    return rz * ry * rx;
}

cv::Mat rvecFromEulerDegrees(
    const cv::Vec3d& euler_deg) {
    const cv::Mat rotation(
        rotationFromEulerDegrees(euler_deg));

    cv::Mat rvec;
    cv::Rodrigues(rotation, rvec);
    return rvec;
}

cv::Mat makeViewTranslation(int index) {
    const double tx =
        -0.10 + 0.04 * static_cast<double>(index % 6);
    const double ty =
        -0.06 + 0.03 * static_cast<double>((index * 2) % 5);
    const double tz =
        0.90 + 0.07 * static_cast<double>(index % 7);

    return (cv::Mat_<double>(3, 1) <<
        tx,
        ty,
        tz);
}

bool allPointsInside(
    const std::vector<cv::Point2f>& points,
    cv::Size image_size,
    double margin = 10.0) {
    for (const auto& point : points) {
        if (point.x < margin ||
            point.y < margin ||
            point.x >= image_size.width - margin ||
            point.y >= image_size.height - margin) {
            return false;
        }
    }
    return true;
}

CalibrationData generateCalibrationData(
    cv::Size image_size,
    int board_cols,
    int board_rows,
    double square_size,
    int views,
    const GroundTruthCamera& camera,
    double noise_sigma_px,
    unsigned int seed) {
    CalibrationData data;

    const auto board =
        createBoardPoints(
            board_cols,
            board_rows,
            square_size);

    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(
        0.0,
        noise_sigma_px);

    int candidate_index = 0;

    while (
        static_cast<int>(data.object_points.size()) < views &&
        candidate_index < views * 20) {
        const cv::Mat rvec =
            rvecFromEulerDegrees(
                viewEulerDegrees(candidate_index));

        cv::Mat tvec =
            makeViewTranslation(candidate_index);

        if (candidate_index >= views) {
            tvec.at<double>(2, 0) +=
                0.015 * static_cast<double>(candidate_index - views);
        }

        std::vector<cv::Point2f> projected;

        cv::projectPoints(
            board,
            rvec,
            tvec,
            camera.camera_matrix,
            camera.distortion,
            projected);

        if (!allPointsInside(
                projected,
                image_size)) {
            ++candidate_index;
            continue;
        }

        for (auto& point : projected) {
            point.x += static_cast<float>(noise(rng));
            point.y += static_cast<float>(noise(rng));
        }

        data.object_points.push_back(board);
        data.image_points.push_back(projected);
        data.true_rvecs.push_back(rvec);
        data.true_tvecs.push_back(tvec);

        ++candidate_index;
    }

    if (static_cast<int>(data.object_points.size()) != views) {
        throw std::runtime_error(
            "Could not generate requested number of visible calibration views.");
    }

    return data;
}

CalibrationResult calibrate(
    const CalibrationData& data,
    cv::Size image_size) {
    CalibrationResult result;

    result.camera_matrix =
        cv::Mat::eye(3, 3, CV_64F);

    result.distortion =
        cv::Mat::zeros(1, 5, CV_64F);

    const int flags =
        cv::CALIB_FIX_K3;

    result.optimizer_rms =
        cv::calibrateCamera(
            data.object_points,
            data.image_points,
            image_size,
            result.camera_matrix,
            result.distortion,
            result.rvecs,
            result.tvecs,
            flags,
            cv::TermCriteria(
                cv::TermCriteria::COUNT |
                cv::TermCriteria::EPS,
                100,
                1e-12));

    return result;
}

void computeResiduals(
    const CalibrationData& data,
    CalibrationResult& result) {
    result.per_view_rmse.clear();
    result.per_view_rmse.reserve(
        data.object_points.size());

    double total_squared_error = 0.0;
    std::size_t total_points = 0;
    result.max_residual = 0.0;

    for (std::size_t view = 0;
         view < data.object_points.size();
         ++view) {
        std::vector<cv::Point2f> projected;

        cv::projectPoints(
            data.object_points[view],
            result.rvecs[view],
            result.tvecs[view],
            result.camera_matrix,
            result.distortion,
            projected);

        double view_squared_error = 0.0;

        for (std::size_t i = 0;
             i < projected.size();
             ++i) {
            const double dx =
                static_cast<double>(projected[i].x) -
                data.image_points[view][i].x;

            const double dy =
                static_cast<double>(projected[i].y) -
                data.image_points[view][i].y;

            const double squared =
                dx * dx + dy * dy;

            const double residual =
                std::sqrt(squared);

            view_squared_error += squared;
            total_squared_error += squared;
            ++total_points;

            result.max_residual =
                std::max(
                    result.max_residual,
                    residual);
        }

        result.per_view_rmse.push_back(
            projected.empty()
                ? 0.0
                : std::sqrt(
                    view_squared_error /
                    projected.size()));
    }

    result.global_rmse =
        total_points == 0
            ? 0.0
            : std::sqrt(
                total_squared_error /
                static_cast<double>(total_points));
}

double relativeError(
    double estimated,
    double truth) {
    return std::abs(estimated - truth) /
           std::max(1e-12, std::abs(truth));
}

double coefficient(
    const cv::Mat& distortion,
    int index) {
    cv::Mat flat =
        distortion.reshape(1, 1);

    if (index < 0 ||
        index >= flat.cols) {
        return 0.0;
    }

    return flat.at<double>(0, index);
}

cv::Mat drawCoverage(
    const CalibrationData& data,
    cv::Size image_size) {
    cv::Mat image(
        image_size,
        CV_8UC3,
        cv::Scalar(20, 20, 20));

    for (std::size_t view = 0;
         view < data.image_points.size();
         ++view) {
        const int intensity =
            70 +
            static_cast<int>(
                (150 * view) /
                std::max<std::size_t>(
                    1,
                    data.image_points.size() - 1));

        const cv::Scalar color(
            intensity,
            255 - intensity / 2,
            120 + intensity / 3);

        for (const auto& point :
             data.image_points[view]) {
            cv::circle(
                image,
                cv::Point(
                    cvRound(point.x),
                    cvRound(point.y)),
                2,
                color,
                cv::FILLED,
                cv::LINE_AA);
        }
    }

    return image;
}

cv::Mat drawResidualsForView(
    const CalibrationData& data,
    const CalibrationResult& result,
    cv::Size image_size,
    std::size_t view_index,
    double vector_scale = 25.0) {
    cv::Mat image(
        image_size,
        CV_8UC3,
        cv::Scalar(20, 20, 20));

    if (view_index >=
        data.object_points.size()) {
        return image;
    }

    std::vector<cv::Point2f> projected;

    cv::projectPoints(
        data.object_points[view_index],
        result.rvecs[view_index],
        result.tvecs[view_index],
        result.camera_matrix,
        result.distortion,
        projected);

    for (std::size_t i = 0;
         i < projected.size();
         ++i) {
        const cv::Point2f observed =
            data.image_points[view_index][i];

        const cv::Point2f estimated =
            projected[i];

        const cv::Point observed_px(
            cvRound(observed.x),
            cvRound(observed.y));

        const cv::Point residual_tip(
            cvRound(
                observed.x +
                vector_scale *
                (estimated.x - observed.x)),
            cvRound(
                observed.y +
                vector_scale *
                (estimated.y - observed.y)));

        cv::circle(
            image,
            observed_px,
            3,
            cv::Scalar(0, 255, 0),
            cv::FILLED,
            cv::LINE_AA);

        cv::line(
            image,
            observed_px,
            residual_tip,
            cv::Scalar(0, 0, 255),
            1,
            cv::LINE_AA);
    }

    cv::putText(
        image,
        "green=observed, red=residual x25",
        cv::Point(20, 30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(255, 255, 255),
        1,
        cv::LINE_AA);

    return image;
}

bool writeReport(
    const fs::path& path,
    const Options& options,
    const GroundTruthCamera& truth,
    const CalibrationResult& result) {
    cv::FileStorage storage(
        path.string(),
        cv::FileStorage::WRITE);

    if (!storage.isOpened()) {
        return false;
    }

    storage << "image_width" << options.width;
    storage << "image_height" << options.height;
    storage << "board_cols" << options.board_cols;
    storage << "board_rows" << options.board_rows;
    storage << "square_size" << options.square_size;
    storage << "views" << options.views;
    storage << "noise_sigma_px" << options.noise_sigma_px;
    storage << "seed" << static_cast<int>(options.seed);

    storage << "ground_truth_camera_matrix"
            << truth.camera_matrix;
    storage << "estimated_camera_matrix"
            << result.camera_matrix;
    storage << "ground_truth_distortion"
            << truth.distortion;
    storage << "estimated_distortion"
            << result.distortion;

    storage << "optimizer_rms"
            << result.optimizer_rms;
    storage << "global_reprojection_rmse"
            << result.global_rmse;
    storage << "max_residual_px"
            << result.max_residual;

    storage << "per_view_rmse" << "[";
    for (const double value :
         result.per_view_rmse) {
        storage << value;
    }
    storage << "]";

    storage.release();
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

bool runCalibrationCase(
    cv::Size image_size,
    int board_cols,
    int board_rows,
    double square_size,
    int views,
    double noise_sigma_px,
    unsigned int seed,
    GroundTruthCamera& truth,
    CalibrationData& data,
    CalibrationResult& result) {
    truth =
        makeGroundTruthCamera(
            image_size.width,
            image_size.height);

    data =
        generateCalibrationData(
            image_size,
            board_cols,
            board_rows,
            square_size,
            views,
            truth,
            noise_sigma_px,
            seed);

    result =
        calibrate(
            data,
            image_size);

    computeResiduals(
        data,
        result);

    return !result.camera_matrix.empty() &&
           result.rvecs.size() ==
               data.object_points.size();
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

    GroundTruthCamera truth;
    CalibrationData data;
    CalibrationResult result;

    check(
        runCalibrationCase(
            cv::Size(1280, 720),
            9,
            6,
            0.04,
            14,
            0.0,
            7,
            truth,
            data,
            result),
        "zero-noise synthetic calibration completes");

    if (!result.camera_matrix.empty()) {
        check(
            result.optimizer_rms < 1e-3,
            "zero-noise optimizer RMS is near zero");

        check(
            result.global_rmse < 1e-3,
            "independent zero-noise reprojection RMSE is near zero");

        check(
            relativeError(
                result.camera_matrix.at<double>(0, 0),
                truth.camera_matrix.at<double>(0, 0)) < 0.005,
            "zero-noise fx is recovered within 0.5 percent");

        check(
            relativeError(
                result.camera_matrix.at<double>(1, 1),
                truth.camera_matrix.at<double>(1, 1)) < 0.005,
            "zero-noise fy is recovered within 0.5 percent");

        check(
            std::abs(
                result.camera_matrix.at<double>(0, 2) -
                truth.camera_matrix.at<double>(0, 2)) < 1.0,
            "zero-noise cx is recovered within one pixel");

        check(
            std::abs(
                result.camera_matrix.at<double>(1, 2) -
                truth.camera_matrix.at<double>(1, 2)) < 1.0,
            "zero-noise cy is recovered within one pixel");

        check(
            std::abs(
                coefficient(result.distortion, 0) -
                coefficient(truth.distortion, 0)) < 0.01,
            "zero-noise k1 is recovered");

        check(
            std::abs(
                coefficient(result.distortion, 1) -
                coefficient(truth.distortion, 1)) < 0.03,
            "zero-noise k2 is recovered");

        check(
            std::abs(
                coefficient(result.distortion, 2) -
                coefficient(truth.distortion, 2)) < 0.002,
            "zero-noise p1 is recovered");

        check(
            std::abs(
                coefficient(result.distortion, 3) -
                coefficient(truth.distortion, 3)) < 0.002,
            "zero-noise p2 is recovered");
    }

    GroundTruthCamera noisy_truth;
    CalibrationData noisy_data;
    CalibrationResult noisy_result;

    check(
        runCalibrationCase(
            cv::Size(1280, 720),
            9,
            6,
            0.04,
            14,
            0.20,
            123,
            noisy_truth,
            noisy_data,
            noisy_result),
        "noisy synthetic calibration completes");

    if (!noisy_result.camera_matrix.empty()) {
        check(
            noisy_result.global_rmse < 0.5,
            "0.20-pixel noise produces subpixel reprojection RMSE");

        check(
            noisy_result.max_residual < 1.5,
            "0.20-pixel noise keeps maximum residual bounded");

        check(
            relativeError(
                noisy_result.camera_matrix.at<double>(0, 0),
                noisy_truth.camera_matrix.at<double>(0, 0)) < 0.03,
            "noisy fx remains within 3 percent");

        check(
            relativeError(
                noisy_result.camera_matrix.at<double>(1, 1),
                noisy_truth.camera_matrix.at<double>(1, 1)) < 0.03,
            "noisy fy remains within 3 percent");

        check(
            noisy_result.per_view_rmse.size() ==
                noisy_data.object_points.size(),
            "one reprojection RMSE is reported per calibration view");
    }

    const auto centered_board =
        createBoardPoints(
            9,
            6,
            0.04);

    check(
        centered_board.size() == 54,
        "9x6 calibration target creates 54 object points");

    double mean_x = 0.0;
    double mean_y = 0.0;

    for (const auto& point :
         centered_board) {
        mean_x += point.x;
        mean_y += point.y;
    }

    mean_x /= centered_board.size();
    mean_y /= centered_board.size();

    check(
        std::abs(mean_x) < 1e-7 &&
        std::abs(mean_y) < 1e-7,
        "synthetic target is centered around board origin");

    if (passed) {
        std::cout
            << "Camera calibration self-test passed.\n";
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

    GroundTruthCamera truth;
    CalibrationData data;
    CalibrationResult result;

    try {
        if (!runCalibrationCase(
                cv::Size(
                    options.width,
                    options.height),
                options.board_cols,
                options.board_rows,
                options.square_size,
                options.views,
                options.noise_sigma_px,
                options.seed,
                truth,
                data,
                result)) {
            std::cerr
                << "Error: calibration did not produce a complete result.\n";
            return 1;
        }
    } catch (const std::exception& error) {
        std::cerr
            << "Error: "
            << error.what()
            << '\n';
        return 1;
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

    bool ok = true;

    ok &= writeImage(
        output_dir /
            "01_calibration_view_coverage.png",
        drawCoverage(
            data,
            cv::Size(
                options.width,
                options.height)));

    ok &= writeImage(
        output_dir /
            "02_reprojection_residuals_view0.png",
        drawResidualsForView(
            data,
            result,
            cv::Size(
                options.width,
                options.height),
            0));

    ok &= writeReport(
        output_dir /
            "calibration_report.yml",
        options,
        truth,
        result);

    if (!ok) {
        std::cerr
            << "Error: failed to write one or more outputs.\n";
        return 1;
    }

    std::cout
        << std::fixed
        << std::setprecision(6)
        << "Synthetic calibration configuration\n"
        << "  image: "
        << options.width
        << "x"
        << options.height
        << '\n'
        << "  board: "
        << options.board_cols
        << "x"
        << options.board_rows
        << " inner corners\n"
        << "  square size: "
        << options.square_size
        << '\n'
        << "  views: "
        << options.views
        << '\n'
        << "  observation noise sigma: "
        << options.noise_sigma_px
        << " px\n\n"
        << "Ground truth K:\n"
        << truth.camera_matrix
        << "\n\nEstimated K:\n"
        << result.camera_matrix
        << "\n\nGround truth distortion [k1 k2 p1 p2 k3]:\n"
        << truth.distortion
        << "\n\nEstimated distortion:\n"
        << result.distortion
        << "\n\nOptimizer-reported RMS: "
        << result.optimizer_rms
        << " px\n"
        << "Independently recomputed global RMSE: "
        << result.global_rmse
        << " px\n"
        << "Maximum residual: "
        << result.max_residual
        << " px\n"
        << "Per-view RMSE:\n";

    for (std::size_t view = 0;
         view < result.per_view_rmse.size();
         ++view) {
        std::cout
            << "  view "
            << view
            << ": "
            << result.per_view_rmse[view]
            << " px\n";
    }

    std::cout
        << "Outputs written to: "
        << output_dir
        << '\n';

    return 0;
}
