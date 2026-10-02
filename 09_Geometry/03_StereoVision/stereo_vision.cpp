#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string left_path;
    std::string right_path;
    std::string output_dir;
    int width = 960;
    int height = 540;
    double fx = 800.0;
    double fy = 800.0;
    double baseline_m = 0.20;
    double synthetic_depth_m = 5.0;
    int num_disparities = 96;
    int block_size = 5;
    bool self_test = false;
    bool help = false;
};

struct StereoRig {
    cv::Matx33d k;
    cv::Matx33d rotation_right_from_left;
    cv::Vec3d translation_right_from_left;
};

struct Rectification {
    cv::Mat r1;
    cv::Mat r2;
    cv::Mat p1;
    cv::Mat p2;
    cv::Mat q;
};

struct SyntheticPair {
    cv::Mat left;
    cv::Mat right;
    int disparity_px = 0;
    cv::Rect evaluation_roi;
};

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " --output-dir <dir> [options]\n"
        << "  " << program << " --self-test\n\n"
        << "Options:\n"
        << "  --left <path>             Optional already-rectified left image\n"
        << "  --right <path>            Optional already-rectified right image\n"
        << "  --output-dir <path>       Output directory\n"
        << "  --width <pixels>          Synthetic image width (default: 960)\n"
        << "  --height <pixels>         Synthetic image height (default: 540)\n"
        << "  --fx <pixels>             Rectified focal length (default: 800)\n"
        << "  --fy <pixels>             Rectified focal length (default: 800)\n"
        << "  --baseline <meters>       Stereo baseline (default: 0.20)\n"
        << "  --depth <meters>          Synthetic plane depth (default: 5.0)\n"
        << "  --num-disparities <N>     SGBM range, multiple of 16 (default: 96)\n"
        << "  --block-size <odd N>      SGBM block size (default: 5)\n"
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

            if (arg == "--left") {
                options.left_path = value(i, arg);
            } else if (arg == "--right") {
                options.right_path = value(i, arg);
            } else if (arg == "--output-dir") {
                options.output_dir = value(i, arg);
            } else if (arg == "--width") {
                options.width = std::stoi(value(i, arg));
            } else if (arg == "--height") {
                options.height = std::stoi(value(i, arg));
            } else if (arg == "--fx") {
                options.fx = std::stod(value(i, arg));
            } else if (arg == "--fy") {
                options.fy = std::stod(value(i, arg));
            } else if (arg == "--baseline") {
                options.baseline_m = std::stod(value(i, arg));
            } else if (arg == "--depth") {
                options.synthetic_depth_m = std::stod(value(i, arg));
            } else if (arg == "--num-disparities") {
                options.num_disparities = std::stoi(value(i, arg));
            } else if (arg == "--block-size") {
                options.block_size = std::stoi(value(i, arg));
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

    const bool one_real_image =
        options.left_path.empty() != options.right_path.empty();

    if (one_real_image) {
        std::cerr
            << "Error: provide both --left and --right, or neither.\n";
        return false;
    }

    if (options.width <= 0 ||
        options.height <= 0 ||
        options.fx <= 0.0 ||
        options.fy <= 0.0 ||
        options.baseline_m <= 0.0 ||
        options.synthetic_depth_m <= 0.0) {
        std::cerr
            << "Error: image size, focal length, baseline and depth must be > 0.\n";
        return false;
    }

    if (options.num_disparities <= 0 ||
        options.num_disparities % 16 != 0) {
        std::cerr
            << "Error: --num-disparities must be a positive multiple of 16.\n";
        return false;
    }

    if (options.block_size < 3 ||
        options.block_size % 2 == 0) {
        std::cerr
            << "Error: --block-size must be odd and >= 3.\n";
        return false;
    }

    return true;
}

cv::Matx33d makeK(
    double fx,
    double fy,
    double cx,
    double cy) {
    return cv::Matx33d(
        fx, 0.0, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0);
}

cv::Matx33d skewSymmetric(const cv::Vec3d& t) {
    return cv::Matx33d(
         0.0, -t[2],  t[1],
         t[2],  0.0, -t[0],
        -t[1],  t[0],  0.0);
}

cv::Matx33d essentialMatrix(const StereoRig& rig) {
    return skewSymmetric(
               rig.translation_right_from_left) *
           rig.rotation_right_from_left;
}

cv::Matx33d fundamentalMatrix(const StereoRig& rig) {
    const cv::Matx33d k_inv =
        rig.k.inv();

    return k_inv.t() *
           essentialMatrix(rig) *
           k_inv;
}

bool projectPinhole(
    const cv::Matx33d& k,
    const cv::Point3d& point_camera,
    cv::Point2d& pixel) {
    if (point_camera.z <= 1e-12) {
        return false;
    }

    const cv::Vec3d normalized(
        point_camera.x / point_camera.z,
        point_camera.y / point_camera.z,
        1.0);

    const cv::Vec3d homogeneous =
        k * normalized;

    pixel.x = homogeneous[0] / homogeneous[2];
    pixel.y = homogeneous[1] / homogeneous[2];

    return std::isfinite(pixel.x) &&
           std::isfinite(pixel.y);
}

cv::Point3d transformLeftToRight(
    const StereoRig& rig,
    const cv::Point3d& point_left) {
    const cv::Vec3d p =
        rig.rotation_right_from_left *
            cv::Vec3d(
                point_left.x,
                point_left.y,
                point_left.z) +
        rig.translation_right_from_left;

    return cv::Point3d(
        p[0],
        p[1],
        p[2]);
}

double epipolarResidual(
    const cv::Point2d& left,
    const cv::Point2d& right,
    const cv::Matx33d& f) {
    const cv::Vec3d x1(
        left.x,
        left.y,
        1.0);

    const cv::Vec3d x2(
        right.x,
        right.y,
        1.0);

    return x2.dot(f * x1);
}

double pointLineDistance(
    const cv::Point2d& point,
    const cv::Vec3d& line) {
    const double numerator =
        std::abs(
            line[0] * point.x +
            line[1] * point.y +
            line[2]);

    const double denominator =
        std::sqrt(
            line[0] * line[0] +
            line[1] * line[1]);

    return denominator <= 1e-12
        ? std::numeric_limits<double>::infinity()
        : numerator / denominator;
}

Rectification rectifyRig(
    const StereoRig& rig,
    cv::Size image_size) {
    Rectification result;

    const cv::Mat k(rig.k);
    const cv::Mat distortion =
        cv::Mat::zeros(1, 5, CV_64F);
    const cv::Mat r(
        rig.rotation_right_from_left);
    const cv::Mat t =
        (cv::Mat_<double>(3, 1) <<
            rig.translation_right_from_left[0],
            rig.translation_right_from_left[1],
            rig.translation_right_from_left[2]);

    cv::Rect valid1;
    cv::Rect valid2;

    cv::stereoRectify(
        k,
        distortion,
        k,
        distortion,
        image_size,
        r,
        t,
        result.r1,
        result.r2,
        result.p1,
        result.p2,
        result.q,
        cv::CALIB_ZERO_DISPARITY,
        0.0,
        image_size,
        &valid1,
        &valid2);

    return result;
}

std::vector<cv::Point2f> rectifyPoints(
    const std::vector<cv::Point2f>& points,
    const cv::Matx33d& k,
    const cv::Mat& rectification_rotation,
    const cv::Mat& projection) {
    std::vector<cv::Point2f> rectified;

    cv::undistortPoints(
        points,
        rectified,
        cv::Mat(k),
        cv::Mat(),
        rectification_rotation,
        projection);

    return rectified;
}

double depthFromDisparity(
    double focal_px,
    double baseline_m,
    double disparity_px) {
    if (disparity_px <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return focal_px *
           baseline_m /
           disparity_px;
}

cv::Point3d pointFromDisparity(
    double u,
    double v,
    double disparity_px,
    double fx,
    double fy,
    double cx,
    double cy,
    double baseline_m) {
    const double z =
        depthFromDisparity(
            fx,
            baseline_m,
            disparity_px);

    if (!std::isfinite(z)) {
        const double nan =
            std::numeric_limits<double>::quiet_NaN();
        return cv::Point3d(nan, nan, nan);
    }

    return cv::Point3d(
        (u - cx) * z / fx,
        (v - cy) * z / fy,
        z);
}

cv::Mat projectionMatrix(
    const cv::Matx33d& k,
    const cv::Matx33d& r,
    const cv::Vec3d& t) {
    cv::Mat rt =
        cv::Mat::zeros(3, 4, CV_64F);

    cv::Mat(cv::Matx33d(r)).copyTo(
        rt(cv::Rect(0, 0, 3, 3)));

    rt.at<double>(0, 3) = t[0];
    rt.at<double>(1, 3) = t[1];
    rt.at<double>(2, 3) = t[2];

    return cv::Mat(k) * rt;
}

cv::Point3d triangulateOne(
    const cv::Mat& p1,
    const cv::Mat& p2,
    const cv::Point2d& left,
    const cv::Point2d& right) {
    std::vector<cv::Point2f> left_points = {
        cv::Point2f(
            static_cast<float>(left.x),
            static_cast<float>(left.y))
    };

    std::vector<cv::Point2f> right_points = {
        cv::Point2f(
            static_cast<float>(right.x),
            static_cast<float>(right.y))
    };

    cv::Mat homogeneous;

    cv::triangulatePoints(
        p1,
        p2,
        left_points,
        right_points,
        homogeneous);

    cv::Mat homogeneous64;
    homogeneous.convertTo(
        homogeneous64,
        CV_64F);

    const double w =
        homogeneous64.at<double>(3, 0);

    if (std::abs(w) < 1e-12) {
        const double nan =
            std::numeric_limits<double>::quiet_NaN();
        return cv::Point3d(nan, nan, nan);
    }

    return cv::Point3d(
        homogeneous64.at<double>(0, 0) / w,
        homogeneous64.at<double>(1, 0) / w,
        homogeneous64.at<double>(2, 0) / w);
}

SyntheticPair makeSyntheticRectifiedPair(
    int width,
    int height,
    double fx,
    double baseline_m,
    double depth_m) {
    SyntheticPair pair;

    pair.disparity_px =
        std::max(
            1,
            cvRound(
                fx *
                baseline_m /
                depth_m));

    cv::Mat texture(
        height,
        width,
        CV_8U);

    cv::RNG rng(0x5EED1234);
    rng.fill(
        texture,
        cv::RNG::UNIFORM,
        25,
        230);

    cv::GaussianBlur(
        texture,
        texture,
        cv::Size(3, 3),
        0.8);

    for (int y = 40;
         y < height;
         y += 70) {
        cv::line(
            texture,
            cv::Point(0, y),
            cv::Point(width - 1, y),
            cv::Scalar(245),
            1,
            cv::LINE_AA);
    }

    for (int x = 50;
         x < width;
         x += 90) {
        cv::circle(
            texture,
            cv::Point(
                x,
                height / 2),
            16,
            cv::Scalar(10),
            2,
            cv::LINE_AA);
    }

    cv::putText(
        texture,
        "9X STEREO",
        cv::Point(
            width / 4,
            height / 2 + 60),
        cv::FONT_HERSHEY_SIMPLEX,
        1.2,
        cv::Scalar(250),
        2,
        cv::LINE_AA);

    pair.left =
        texture.clone();

    pair.right =
        cv::Mat(
            height,
            width,
            CV_8U,
            cv::Scalar(0));

    const int d =
        pair.disparity_px;

    if (d < width) {
        texture(
            cv::Rect(
                d,
                0,
                width - d,
                height))
            .copyTo(
                pair.right(
                    cv::Rect(
                        0,
                        0,
                        width - d,
                        height)));
    }

    pair.evaluation_roi =
        cv::Rect(
            std::min(
                width - 1,
                d + 32),
            32,
            std::max(
                1,
                width - d - 64),
            std::max(
                1,
                height - 64));

    return pair;
}

cv::Mat computeSgbm(
    const cv::Mat& left,
    const cv::Mat& right,
    int num_disparities,
    int block_size) {
    const int channels = 1;

    auto matcher =
        cv::StereoSGBM::create(
            0,
            num_disparities,
            block_size);

    matcher->setP1(
        8 *
        channels *
        block_size *
        block_size);

    matcher->setP2(
        32 *
        channels *
        block_size *
        block_size);

    matcher->setPreFilterCap(31);
    matcher->setUniquenessRatio(8);
    matcher->setSpeckleWindowSize(50);
    matcher->setSpeckleRange(2);
    matcher->setDisp12MaxDiff(1);
    matcher->setMode(
        cv::StereoSGBM::MODE_SGBM_3WAY);

    cv::Mat disparity16;
    matcher->compute(
        left,
        right,
        disparity16);

    cv::Mat disparity32;
    disparity16.convertTo(
        disparity32,
        CV_32F,
        1.0 / 16.0);

    return disparity32;
}

cv::Mat disparityVisualization(
    const cv::Mat& disparity,
    int num_disparities) {
    cv::Mat normalized;

    disparity.convertTo(
        normalized,
        CV_8U,
        255.0 /
            std::max(
                1,
                num_disparities));

    cv::threshold(
        normalized,
        normalized,
        0,
        0,
        cv::THRESH_TOZERO);

    return normalized;
}

cv::Mat depthMap(
    const cv::Mat& disparity,
    double focal_px,
    double baseline_m) {
    cv::Mat depth(
        disparity.size(),
        CV_32F,
        cv::Scalar(
            std::numeric_limits<float>::quiet_NaN()));

    for (int row = 0;
         row < disparity.rows;
         ++row) {
        for (int col = 0;
             col < disparity.cols;
             ++col) {
            const float d =
                disparity.at<float>(
                    row,
                    col);

            if (d > 0.0f) {
                depth.at<float>(
                    row,
                    col) =
                    static_cast<float>(
                        focal_px *
                        baseline_m /
                        d);
            }
        }
    }

    return depth;
}

cv::Mat depthVisualization(
    const cv::Mat& depth,
    double expected_depth_m) {
    cv::Mat visualization(
        depth.size(),
        CV_8U,
        cv::Scalar(0));

    const double max_depth =
        std::max(
            1.0,
            expected_depth_m * 2.0);

    for (int row = 0;
         row < depth.rows;
         ++row) {
        for (int col = 0;
             col < depth.cols;
             ++col) {
            const float z =
                depth.at<float>(
                    row,
                    col);

            if (!std::isfinite(z) ||
                z <= 0.0f) {
                continue;
            }

            const double clamped =
                std::clamp(
                    static_cast<double>(z),
                    0.0,
                    max_depth);

            visualization.at<unsigned char>(
                row,
                col) =
                cv::saturate_cast<unsigned char>(
                    255.0 *
                    (1.0 -
                     clamped /
                     max_depth));
        }
    }

    return visualization;
}

double medianValid(
    const cv::Mat& values,
    const cv::Rect& roi,
    double min_value) {
    const cv::Rect image_rect(
        0,
        0,
        values.cols,
        values.rows);

    const cv::Rect clipped =
        roi & image_rect;

    std::vector<float> samples;

    for (int row = clipped.y;
         row < clipped.y + clipped.height;
         ++row) {
        for (int col = clipped.x;
             col < clipped.x + clipped.width;
             ++col) {
            const float value =
                values.at<float>(
                    row,
                    col);

            if (std::isfinite(value) &&
                value > min_value) {
                samples.push_back(value);
            }
        }
    }

    if (samples.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const std::size_t middle =
        samples.size() / 2;

    std::nth_element(
        samples.begin(),
        samples.begin() + middle,
        samples.end());

    return samples[middle];
}

cv::Mat drawEpipolarExample(
    int width,
    int height,
    const cv::Point2d& left,
    const cv::Point2d& right,
    const cv::Matx33d& f) {
    cv::Mat canvas(
        height,
        width * 2,
        CV_8UC3,
        cv::Scalar(25, 25, 25));

    const cv::Vec3d x1(
        left.x,
        left.y,
        1.0);

    const cv::Vec3d line2 =
        f * x1;

    if (std::abs(line2[1]) > 1e-12) {
        const double y0 =
            -line2[2] /
            line2[1];

        const double y1 =
            -(line2[0] *
                  (width - 1) +
              line2[2]) /
            line2[1];

        cv::line(
            canvas,
            cv::Point(
                width,
                cvRound(y0)),
            cv::Point(
                2 * width - 1,
                cvRound(y1)),
            cv::Scalar(0, 255, 255),
            1,
            cv::LINE_AA);
    }

    cv::circle(
        canvas,
        cv::Point(
            cvRound(left.x),
            cvRound(left.y)),
        5,
        cv::Scalar(0, 255, 0),
        cv::FILLED,
        cv::LINE_AA);

    cv::circle(
        canvas,
        cv::Point(
            width + cvRound(right.x),
            cvRound(right.y)),
        5,
        cv::Scalar(0, 0, 255),
        cv::FILLED,
        cv::LINE_AA);

    cv::putText(
        canvas,
        "left point",
        cv::Point(20, 30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(220, 220, 220),
        1,
        cv::LINE_AA);

    cv::putText(
        canvas,
        "right point + epipolar line",
        cv::Point(width + 20, 30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(220, 220, 220),
        1,
        cv::LINE_AA);

    return canvas;
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

double pointDistance3d(
    const cv::Point3d& a,
    const cv::Point3d& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dz = a.z - b.z;

    return std::sqrt(
        dx * dx +
        dy * dy +
        dz * dz);
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

    const double fx = 800.0;
    const double fy = 805.0;
    const double cx = 640.0;
    const double cy = 360.0;
    const double baseline = 0.20;

    StereoRig rectified_rig;
    rectified_rig.k =
        makeK(
            fx,
            fy,
            cx,
            cy);

    rectified_rig.rotation_right_from_left =
        cv::Matx33d::eye();

    rectified_rig.translation_right_from_left =
        cv::Vec3d(
            -baseline,
            0.0,
            0.0);

    const cv::Point3d point_left(
        0.35,
        -0.12,
        5.0);

    const cv::Point3d point_right =
        transformLeftToRight(
            rectified_rig,
            point_left);

    cv::Point2d pixel_left;
    cv::Point2d pixel_right;

    check(
        projectPinhole(
            rectified_rig.k,
            point_left,
            pixel_left),
        "left point projects");

    check(
        projectPinhole(
            rectified_rig.k,
            point_right,
            pixel_right),
        "right point projects");

    const double expected_disparity =
        fx *
        baseline /
        point_left.z;

    const double actual_disparity =
        pixel_left.x -
        pixel_right.x;

    check(
        std::abs(
            actual_disparity -
            expected_disparity) < 1e-10,
        "rectified disparity equals fB/Z");

    check(
        std::abs(
            pixel_left.y -
            pixel_right.y) < 1e-10,
        "rectified correspondence has equal row coordinate");

    check(
        std::abs(
            depthFromDisparity(
                fx,
                baseline,
                actual_disparity) -
            point_left.z) < 1e-10,
        "depth formula recovers metric Z");

    const cv::Point3d recovered_point =
        pointFromDisparity(
            pixel_left.x,
            pixel_left.y,
            actual_disparity,
            fx,
            fy,
            cx,
            cy,
            baseline);

    check(
        pointDistance3d(
            recovered_point,
            point_left) < 1e-10,
        "disparity back-projection recovers 3-D point");

    check(
        std::isnan(
            depthFromDisparity(
                fx,
                baseline,
                0.0)),
        "zero disparity is rejected");

    const cv::Matx33d f =
        fundamentalMatrix(
            rectified_rig);

    check(
        std::abs(
            epipolarResidual(
                pixel_left,
                pixel_right,
                f)) < 1e-10,
        "rectified correspondence satisfies epipolar constraint");

    const cv::Vec3d line_right =
        f *
        cv::Vec3d(
            pixel_left.x,
            pixel_left.y,
            1.0);

    check(
        pointLineDistance(
            pixel_right,
            line_right) < 1e-10,
        "right point lies on epipolar line");

    const cv::Mat p1 =
        projectionMatrix(
            rectified_rig.k,
            cv::Matx33d::eye(),
            cv::Vec3d(0.0, 0.0, 0.0));

    const cv::Mat p2 =
        projectionMatrix(
            rectified_rig.k,
            rectified_rig.rotation_right_from_left,
            rectified_rig.translation_right_from_left);

    const cv::Point3d triangulated =
        triangulateOne(
            p1,
            p2,
            pixel_left,
            pixel_right);

    check(
        pointDistance3d(
            triangulated,
            point_left) < 1e-4,
        "triangulation recovers the original 3-D point");

    StereoRig unrectified_rig =
        rectified_rig;

    const double yaw =
        4.0 * CV_PI / 180.0;

    unrectified_rig.rotation_right_from_left =
        cv::Matx33d(
            std::cos(yaw), 0.0, std::sin(yaw),
            0.0, 1.0, 0.0,
            -std::sin(yaw), 0.0, std::cos(yaw));

    unrectified_rig.translation_right_from_left =
        cv::Vec3d(
            -baseline,
            0.01,
            0.005);

    std::vector<cv::Point3d> geometry_points = {
        {-0.6, -0.3, 4.0},
        {-0.2,  0.4, 5.0},
        { 0.4, -0.1, 6.5},
        { 0.8,  0.5, 8.0}
    };

    const cv::Matx33d f_general =
        fundamentalMatrix(
            unrectified_rig);

    double max_epipolar_residual = 0.0;

    std::vector<cv::Point2f> left_pixels;
    std::vector<cv::Point2f> right_pixels;

    for (const auto& point :
         geometry_points) {
        const cv::Point3d right_point =
            transformLeftToRight(
                unrectified_rig,
                point);

        cv::Point2d left_pixel;
        cv::Point2d right_pixel;

        check(
            projectPinhole(
                unrectified_rig.k,
                point,
                left_pixel),
            "general left projection succeeds");

        check(
            projectPinhole(
                unrectified_rig.k,
                right_point,
                right_pixel),
            "general right projection succeeds");

        max_epipolar_residual =
            std::max(
                max_epipolar_residual,
                std::abs(
                    epipolarResidual(
                        left_pixel,
                        right_pixel,
                        f_general)));

        left_pixels.emplace_back(
            static_cast<float>(left_pixel.x),
            static_cast<float>(left_pixel.y));

        right_pixels.emplace_back(
            static_cast<float>(right_pixel.x),
            static_cast<float>(right_pixel.y));
    }

    check(
        max_epipolar_residual < 1e-8,
        "general stereo correspondences satisfy x2^T F x1 = 0");

    const Rectification rectification =
        rectifyRig(
            unrectified_rig,
            cv::Size(1280, 720));

    const auto left_rectified =
        rectifyPoints(
            left_pixels,
            unrectified_rig.k,
            rectification.r1,
            rectification.p1);

    const auto right_rectified =
        rectifyPoints(
            right_pixels,
            unrectified_rig.k,
            rectification.r2,
            rectification.p2);

    double max_vertical_error = 0.0;

    for (std::size_t i = 0;
         i < left_rectified.size();
         ++i) {
        max_vertical_error =
            std::max(
                max_vertical_error,
                std::abs(
                    static_cast<double>(
                        left_rectified[i].y -
                        right_rectified[i].y)));
    }

    check(
        max_vertical_error < 1e-3,
        "stereoRectify aligns corresponding points onto the same rows");

    if (passed) {
        std::cout
            << "Stereo vision self-test passed.\n";
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

    cv::Mat left;
    cv::Mat right;
    int known_disparity = 0;
    cv::Rect evaluation_roi;

    const bool use_real_images =
        !options.left_path.empty();

    if (use_real_images) {
        left =
            cv::imread(
                options.left_path,
                cv::IMREAD_GRAYSCALE);

        right =
            cv::imread(
                options.right_path,
                cv::IMREAD_GRAYSCALE);

        if (left.empty() ||
            right.empty()) {
            std::cerr
                << "Error: could not read both stereo images.\n";
            return 1;
        }

        if (left.size() != right.size()) {
            std::cerr
                << "Error: left/right images must have the same size.\n";
            return 1;
        }

        evaluation_roi =
            cv::Rect(
                options.num_disparities + 16,
                16,
                std::max(
                    1,
                    left.cols -
                    options.num_disparities -
                    32),
                std::max(
                    1,
                    left.rows - 32));
    } else {
        const SyntheticPair pair =
            makeSyntheticRectifiedPair(
                options.width,
                options.height,
                options.fx,
                options.baseline_m,
                options.synthetic_depth_m);

        left = pair.left;
        right = pair.right;
        known_disparity =
            pair.disparity_px;
        evaluation_roi =
            pair.evaluation_roi;
    }

    const cv::Mat disparity =
        computeSgbm(
            left,
            right,
            options.num_disparities,
            options.block_size);

    const cv::Mat depth =
        depthMap(
            disparity,
            options.fx,
            options.baseline_m);

    const double median_disparity =
        medianValid(
            disparity,
            evaluation_roi,
            0.0);

    const double median_depth =
        medianValid(
            depth,
            evaluation_roi,
            0.0);

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
        output_dir / "01_left.png",
        left);

    ok &= writeImage(
        output_dir / "02_right.png",
        right);

    ok &= writeImage(
        output_dir / "03_disparity.png",
        disparityVisualization(
            disparity,
            options.num_disparities));

    ok &= writeImage(
        output_dir / "04_depth_inverse_visualization.png",
        depthVisualization(
            depth,
            options.synthetic_depth_m));

    StereoRig demo_rig;
    demo_rig.k =
        makeK(
            options.fx,
            options.fy,
            left.cols * 0.5,
            left.rows * 0.5);

    demo_rig.rotation_right_from_left =
        cv::Matx33d::eye();

    demo_rig.translation_right_from_left =
        cv::Vec3d(
            -options.baseline_m,
            0.0,
            0.0);

    const cv::Point3d demo_point(
        0.25,
        0.0,
        options.synthetic_depth_m);

    const cv::Point3d demo_point_right =
        transformLeftToRight(
            demo_rig,
            demo_point);

    cv::Point2d demo_left;
    cv::Point2d demo_right;

    if (projectPinhole(
            demo_rig.k,
            demo_point,
            demo_left) &&
        projectPinhole(
            demo_rig.k,
            demo_point_right,
            demo_right)) {
        ok &= writeImage(
            output_dir /
                "05_epipolar_geometry.png",
            drawEpipolarExample(
                left.cols,
                left.rows,
                demo_left,
                demo_right,
                fundamentalMatrix(
                    demo_rig)));
    }

    if (!ok) {
        return 1;
    }

    std::cout
        << std::fixed
        << std::setprecision(4)
        << "Stereo configuration\n"
        << "  image size: "
        << left.cols
        << "x"
        << left.rows
        << '\n'
        << "  focal fx: "
        << options.fx
        << " px\n"
        << "  baseline: "
        << options.baseline_m
        << " m\n"
        << "  disparity range: "
        << options.num_disparities
        << " px\n"
        << "  block size: "
        << options.block_size
        << '\n'
        << "  median valid disparity: "
        << median_disparity
        << " px\n"
        << "  median valid depth: "
        << median_depth
        << " m\n";

    if (!use_real_images) {
        std::cout
            << "  synthetic ground-truth disparity: "
            << known_disparity
            << " px\n"
            << "  synthetic ground-truth depth: "
            << options.synthetic_depth_m
            << " m\n";
    }

    std::cout
        << "Outputs written to: "
        << output_dir
        << '\n';

    return 0;
}
