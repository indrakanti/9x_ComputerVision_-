#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Intrinsics {
    double fx = 800.0;
    double fy = 800.0;
    double cx = 640.0;
    double cy = 360.0;
    double skew = 0.0;
};

struct Distortion {
    double k1 = 0.0;
    double k2 = 0.0;
    double p1 = 0.0;
    double p2 = 0.0;
    double k3 = 0.0;
};

struct Extrinsics {
    double roll_deg = 0.0;
    double pitch_deg = 0.0;
    double yaw_deg = 0.0;
    cv::Vec3d translation{0.0, 0.0, 0.0};
};

struct Options {
    std::string output_dir;
    int width = 1280;
    int height = 720;
    Intrinsics intrinsics;
    Distortion distortion;
    Extrinsics extrinsics;
    bool self_test = false;
    bool help = false;
};

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " --output-dir <dir> [options]\n"
        << "  " << program << " --self-test\n\n"
        << "Options:\n"
        << "  --output-dir <path>     Output directory\n"
        << "  --width <pixels>        Image width (default: 1280)\n"
        << "  --height <pixels>       Image height (default: 720)\n"
        << "  --fx <value>            Focal length fx in pixels\n"
        << "  --fy <value>            Focal length fy in pixels\n"
        << "  --cx <value>            Principal point cx\n"
        << "  --cy <value>            Principal point cy\n"
        << "  --skew <value>          Pixel-axis skew (default: 0)\n"
        << "  --roll <deg>            Camera-frame roll convention\n"
        << "  --pitch <deg>           Camera-frame pitch convention\n"
        << "  --yaw <deg>             Camera-frame yaw convention\n"
        << "  --tx <value>            Translation x\n"
        << "  --ty <value>            Translation y\n"
        << "  --tz <value>            Translation z\n"
        << "  --k1 <value>            Radial distortion k1\n"
        << "  --k2 <value>            Radial distortion k2\n"
        << "  --p1 <value>            Tangential distortion p1\n"
        << "  --p2 <value>            Tangential distortion p2\n"
        << "  --k3 <value>            Radial distortion k3\n"
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

            if (arg == "--output-dir") {
                options.output_dir = value(i, arg);
            } else if (arg == "--width") {
                options.width = std::stoi(value(i, arg));
            } else if (arg == "--height") {
                options.height = std::stoi(value(i, arg));
            } else if (arg == "--fx") {
                options.intrinsics.fx = std::stod(value(i, arg));
            } else if (arg == "--fy") {
                options.intrinsics.fy = std::stod(value(i, arg));
            } else if (arg == "--cx") {
                options.intrinsics.cx = std::stod(value(i, arg));
            } else if (arg == "--cy") {
                options.intrinsics.cy = std::stod(value(i, arg));
            } else if (arg == "--skew") {
                options.intrinsics.skew = std::stod(value(i, arg));
            } else if (arg == "--roll") {
                options.extrinsics.roll_deg = std::stod(value(i, arg));
            } else if (arg == "--pitch") {
                options.extrinsics.pitch_deg = std::stod(value(i, arg));
            } else if (arg == "--yaw") {
                options.extrinsics.yaw_deg = std::stod(value(i, arg));
            } else if (arg == "--tx") {
                options.extrinsics.translation[0] = std::stod(value(i, arg));
            } else if (arg == "--ty") {
                options.extrinsics.translation[1] = std::stod(value(i, arg));
            } else if (arg == "--tz") {
                options.extrinsics.translation[2] = std::stod(value(i, arg));
            } else if (arg == "--k1") {
                options.distortion.k1 = std::stod(value(i, arg));
            } else if (arg == "--k2") {
                options.distortion.k2 = std::stod(value(i, arg));
            } else if (arg == "--p1") {
                options.distortion.p1 = std::stod(value(i, arg));
            } else if (arg == "--p2") {
                options.distortion.p2 = std::stod(value(i, arg));
            } else if (arg == "--k3") {
                options.distortion.k3 = std::stod(value(i, arg));
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
        options.intrinsics.fx <= 0.0 ||
        options.intrinsics.fy <= 0.0) {
        std::cerr
            << "Error: width/height/fx/fy must be > 0.\n";
        return false;
    }

    return true;
}

double degToRad(double degrees) {
    return degrees * CV_PI / 180.0;
}

cv::Matx33d rotationMatrix(const Extrinsics& extrinsics) {
    const double roll = degToRad(extrinsics.roll_deg);
    const double pitch = degToRad(extrinsics.pitch_deg);
    const double yaw = degToRad(extrinsics.yaw_deg);

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

cv::Matx33d intrinsicMatrix(const Intrinsics& intrinsics) {
    return cv::Matx33d(
        intrinsics.fx, intrinsics.skew, intrinsics.cx,
        0.0, intrinsics.fy, intrinsics.cy,
        0.0, 0.0, 1.0);
}

cv::Vec3d worldToCamera(
    const cv::Point3d& world,
    const cv::Matx33d& rotation,
    const cv::Vec3d& translation) {
    return rotation *
               cv::Vec3d(world.x, world.y, world.z) +
           translation;
}

cv::Point2d distortNormalized(
    const cv::Point2d& normalized,
    const Distortion& distortion) {
    const double x = normalized.x;
    const double y = normalized.y;

    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;

    const double radial =
        1.0 +
        distortion.k1 * r2 +
        distortion.k2 * r4 +
        distortion.k3 * r6;

    const double x_tangential =
        2.0 * distortion.p1 * x * y +
        distortion.p2 * (r2 + 2.0 * x * x);

    const double y_tangential =
        distortion.p1 * (r2 + 2.0 * y * y) +
        2.0 * distortion.p2 * x * y;

    return cv::Point2d(
        x * radial + x_tangential,
        y * radial + y_tangential);
}

bool projectPointManual(
    const cv::Point3d& world,
    const Intrinsics& intrinsics,
    const Distortion& distortion,
    const cv::Matx33d& rotation,
    const cv::Vec3d& translation,
    cv::Point2d& pixel,
    cv::Vec3d* camera_point = nullptr) {
    const cv::Vec3d camera =
        worldToCamera(world, rotation, translation);

    if (camera_point != nullptr) {
        *camera_point = camera;
    }

    if (camera[2] <= 1e-12) {
        return false;
    }

    const cv::Point2d normalized(
        camera[0] / camera[2],
        camera[1] / camera[2]);

    const cv::Point2d distorted =
        distortNormalized(normalized, distortion);

    pixel.x =
        intrinsics.fx * distorted.x +
        intrinsics.skew * distorted.y +
        intrinsics.cx;

    pixel.y =
        intrinsics.fy * distorted.y +
        intrinsics.cy;

    return std::isfinite(pixel.x) &&
           std::isfinite(pixel.y);
}

std::vector<cv::Point3d> createScenePoints() {
    std::vector<cv::Point3d> points;

    for (int z_index = 0; z_index < 4; ++z_index) {
        const double z = 4.0 + z_index * 1.5;

        for (int y_index = -2; y_index <= 2; ++y_index) {
            for (int x_index = -3; x_index <= 3; ++x_index) {
                points.emplace_back(
                    x_index * 0.45,
                    y_index * 0.35,
                    z);
            }
        }
    }

    return points;
}

cv::Mat renderProjection(
    int width,
    int height,
    const std::vector<cv::Point3d>& points,
    const Intrinsics& intrinsics,
    const Distortion& distortion,
    const cv::Matx33d& rotation,
    const cv::Vec3d& translation) {
    cv::Mat image(
        height,
        width,
        CV_8UC3,
        cv::Scalar(20, 20, 20));

    cv::line(
        image,
        cv::Point(0, cvRound(intrinsics.cy)),
        cv::Point(width - 1, cvRound(intrinsics.cy)),
        cv::Scalar(60, 60, 60),
        1,
        cv::LINE_AA);

    cv::line(
        image,
        cv::Point(cvRound(intrinsics.cx), 0),
        cv::Point(cvRound(intrinsics.cx), height - 1),
        cv::Scalar(60, 60, 60),
        1,
        cv::LINE_AA);

    for (const auto& point : points) {
        cv::Point2d pixel;
        cv::Vec3d camera;

        if (!projectPointManual(
                point,
                intrinsics,
                distortion,
                rotation,
                translation,
                pixel,
                &camera)) {
            continue;
        }

        if (pixel.x < 0.0 ||
            pixel.y < 0.0 ||
            pixel.x >= width ||
            pixel.y >= height) {
            continue;
        }

        const int radius =
            std::clamp(
                cvRound(7.0 / camera[2]),
                2,
                5);

        cv::circle(
            image,
            cv::Point(
                cvRound(pixel.x),
                cvRound(pixel.y)),
            radius,
            cv::Scalar(0, 220, 255),
            cv::FILLED,
            cv::LINE_AA);
    }

    cv::circle(
        image,
        cv::Point(
            cvRound(intrinsics.cx),
            cvRound(intrinsics.cy)),
        5,
        cv::Scalar(0, 0, 255),
        cv::FILLED,
        cv::LINE_AA);

    return image;
}

cv::Mat distortionVector(const Distortion& distortion) {
    return (cv::Mat_<double>(1, 5) <<
        distortion.k1,
        distortion.k2,
        distortion.p1,
        distortion.p2,
        distortion.k3);
}

std::vector<cv::Point2d> projectOpenCvReference(
    const std::vector<cv::Point3d>& points,
    const Intrinsics& intrinsics,
    const Distortion& distortion,
    const cv::Matx33d& rotation,
    const cv::Vec3d& translation) {
    cv::Mat rotation_matrix(rotation);
    cv::Mat rvec;
    cv::Rodrigues(rotation_matrix, rvec);

    const cv::Mat camera_matrix(
        intrinsicMatrix(intrinsics));

    std::vector<cv::Point2d> projected;

    cv::projectPoints(
        points,
        rvec,
        cv::Mat(translation),
        camera_matrix,
        distortionVector(distortion),
        projected);

    return projected;
}

double maxProjectionError(
    const std::vector<cv::Point3d>& points,
    const Intrinsics& intrinsics,
    const Distortion& distortion,
    const cv::Matx33d& rotation,
    const cv::Vec3d& translation) {
    const auto reference =
        projectOpenCvReference(
            points,
            intrinsics,
            distortion,
            rotation,
            translation);

    double max_error = 0.0;

    for (std::size_t i = 0; i < points.size(); ++i) {
        cv::Point2d manual;

        if (!projectPointManual(
                points[i],
                intrinsics,
                distortion,
                rotation,
                translation,
                manual)) {
            continue;
        }

        const double dx =
            manual.x - reference[i].x;
        const double dy =
            manual.y - reference[i].y;

        max_error =
            std::max(
                max_error,
                std::sqrt(dx * dx + dy * dy));
    }

    return max_error;
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

    Intrinsics intrinsics;
    intrinsics.fx = 100.0;
    intrinsics.fy = 100.0;
    intrinsics.cx = 320.0;
    intrinsics.cy = 240.0;
    intrinsics.skew = 0.0;

    Distortion no_distortion;
    Extrinsics identity_extrinsics;
    const cv::Matx33d identity_rotation =
        rotationMatrix(identity_extrinsics);

    cv::Point2d pixel;

    check(
        projectPointManual(
            cv::Point3d(0.0, 0.0, 10.0),
            intrinsics,
            no_distortion,
            identity_rotation,
            identity_extrinsics.translation,
            pixel),
        "optical-axis point projects successfully");

    check(
        std::abs(pixel.x - 320.0) < 1e-12 &&
        std::abs(pixel.y - 240.0) < 1e-12,
        "optical-axis point lands at principal point");

    check(
        projectPointManual(
            cv::Point3d(1.0, 2.0, 10.0),
            intrinsics,
            no_distortion,
            identity_rotation,
            identity_extrinsics.translation,
            pixel),
        "simple pinhole point projects successfully");

    check(
        std::abs(pixel.x - 330.0) < 1e-12 &&
        std::abs(pixel.y - 260.0) < 1e-12,
        "pinhole projection matches hand calculation");

    Extrinsics translated;
    translated.translation =
        cv::Vec3d(1.0, 0.0, 0.0);

    check(
        projectPointManual(
            cv::Point3d(0.0, 0.0, 10.0),
            intrinsics,
            no_distortion,
            rotationMatrix(translated),
            translated.translation,
            pixel),
        "translated camera-coordinate point projects");

    check(
        std::abs(pixel.x - 330.0) < 1e-12,
        "positive camera-coordinate translation shifts pixel right");

    cv::Point2d distorted;
    Distortion radial;
    radial.k1 = 0.20;

    distorted =
        distortNormalized(
            cv::Point2d(0.5, 0.0),
            radial);

    check(
        distorted.x > 0.5,
        "positive k1 pushes off-axis normalized point outward");

    Distortion tangential;
    tangential.p1 = 0.01;
    tangential.p2 = -0.02;

    const cv::Point2d tangential_result =
        distortNormalized(
            cv::Point2d(0.2, -0.1),
            tangential);

    check(
        std::abs(tangential_result.x - 0.1975) < 1e-12 &&
        std::abs(tangential_result.y + 0.09875) < 1e-12,
        "tangential distortion matches explicit formula");

    cv::Point2d behind_pixel;
    check(
        !projectPointManual(
            cv::Point3d(0.0, 0.0, -1.0),
            intrinsics,
            no_distortion,
            identity_rotation,
            identity_extrinsics.translation,
            behind_pixel),
        "point behind camera is rejected");

    Extrinsics rotated;
    rotated.roll_deg = 7.0;
    rotated.pitch_deg = -11.0;
    rotated.yaw_deg = 14.0;
    rotated.translation =
        cv::Vec3d(0.15, -0.08, 0.4);

    Intrinsics reference_intrinsics;
    reference_intrinsics.fx = 910.0;
    reference_intrinsics.fy = 895.0;
    reference_intrinsics.cx = 641.0;
    reference_intrinsics.cy = 357.0;
    reference_intrinsics.skew = 0.0;

    Distortion reference_distortion;
    reference_distortion.k1 = -0.12;
    reference_distortion.k2 = 0.025;
    reference_distortion.p1 = 0.0015;
    reference_distortion.p2 = -0.0009;
    reference_distortion.k3 = -0.003;

    const std::vector<cv::Point3d> points = {
        {-0.8, -0.5, 4.0},
        {-0.2,  0.4, 5.0},
        { 0.5, -0.3, 6.0},
        { 0.9,  0.7, 7.0},
        { 0.0,  0.0, 8.0}
    };

    const double reference_error =
        maxProjectionError(
            points,
            reference_intrinsics,
            reference_distortion,
            rotationMatrix(rotated),
            rotated.translation);

    check(
        reference_error < 1e-8,
        "manual camera model matches OpenCV projectPoints");

    const cv::Matx33d r =
        rotationMatrix(rotated);

    const cv::Matx33d should_be_identity =
        r * r.t();

    double max_orthogonality_error = 0.0;

    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            const double expected =
                row == col ? 1.0 : 0.0;

            max_orthogonality_error =
                std::max(
                    max_orthogonality_error,
                    std::abs(
                        should_be_identity(row, col) -
                        expected));
        }
    }

    check(
        max_orthogonality_error < 1e-12,
        "Euler-composed rotation matrix is orthonormal");

    check(
        std::abs(cv::determinant(cv::Mat(r)) - 1.0) < 1e-12,
        "rotation matrix determinant is +1");

    if (passed) {
        std::cout
            << "Camera model self-test passed.\n";
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

    const cv::Matx33d rotation =
        rotationMatrix(options.extrinsics);

    const auto scene_points =
        createScenePoints();

    Distortion no_distortion;

    const cv::Mat pinhole =
        renderProjection(
            options.width,
            options.height,
            scene_points,
            options.intrinsics,
            no_distortion,
            rotation,
            options.extrinsics.translation);

    const cv::Mat distorted =
        renderProjection(
            options.width,
            options.height,
            scene_points,
            options.intrinsics,
            options.distortion,
            rotation,
            options.extrinsics.translation);

    const double reference_error =
        maxProjectionError(
            scene_points,
            options.intrinsics,
            options.distortion,
            rotation,
            options.extrinsics.translation);

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
        output_dir / "01_pinhole_projection.png",
        pinhole);

    ok &= writeImage(
        output_dir / "02_distorted_projection.png",
        distorted);

    if (!ok) {
        return 1;
    }

    std::cout
        << "Camera intrinsic matrix K:\n"
        << cv::Mat(intrinsicMatrix(options.intrinsics))
        << '\n'
        << "World-to-camera rotation R (Rz * Ry * Rx):\n"
        << cv::Mat(rotation)
        << '\n'
        << "World-to-camera translation t: "
        << options.extrinsics.translation
        << '\n'
        << "Distortion [k1 k2 p1 p2 k3]: ["
        << options.distortion.k1 << ", "
        << options.distortion.k2 << ", "
        << options.distortion.p1 << ", "
        << options.distortion.p2 << ", "
        << options.distortion.k3 << "]\n"
        << "Manual vs OpenCV max projection error: "
        << reference_error
        << " px\n"
        << "Projected synthetic 3-D points: "
        << scene_points.size()
        << '\n'
        << "Outputs written to: "
        << output_dir
        << '\n';

    return 0;
}
