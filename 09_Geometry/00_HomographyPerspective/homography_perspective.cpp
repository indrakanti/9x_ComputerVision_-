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

struct Options {
    std::string input_path;
    std::string output_dir;
    std::array<cv::Point2f, 4> source_points{};
    bool has_source_points = false;
    int output_width = 0;
    int output_height = 0;
    bool self_test = false;
    bool help = false;
};

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program
        << " --input <image> --output-dir <dir>"
        << " --src x0 y0 x1 y1 x2 y2 x3 y3 [options]\n"
        << "  " << program << " --self-test\n\n"
        << "Source-point order:\n"
        << "  top-left, top-right, bottom-right, bottom-left\n\n"
        << "Options:\n"
        << "  --input <path>         Input image\n"
        << "  --output-dir <path>    Output directory\n"
        << "  --src <8 values>       Source quadrilateral coordinates\n"
        << "  --width <pixels>       Rectified width; 0 = infer from source quad\n"
        << "  --height <pixels>      Rectified height; 0 = infer from source quad\n"
        << "  --self-test            Run deterministic tests\n"
        << "  --help                 Show this help\n";
}

bool parseArguments(int argc, char** argv, Options& options) {
    auto requireValue = [argc, argv](int& index, const std::string& option) {
        if (index + 1 >= argc) {
            throw std::runtime_error("Missing value for " + option);
        }
        ++index;
        return std::string(argv[index]);
    };

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];

            if (arg == "--input") {
                options.input_path = requireValue(i, arg);
            } else if (arg == "--output-dir") {
                options.output_dir = requireValue(i, arg);
            } else if (arg == "--src") {
                if (i + 8 >= argc) {
                    throw std::runtime_error(
                        "--src requires eight numeric values");
                }

                for (int point = 0; point < 4; ++point) {
                    const float x = std::stof(argv[++i]);
                    const float y = std::stof(argv[++i]);
                    options.source_points[static_cast<std::size_t>(point)] =
                        cv::Point2f(x, y);
                }
                options.has_source_points = true;
            } else if (arg == "--width") {
                options.output_width =
                    std::stoi(requireValue(i, arg));
            } else if (arg == "--height") {
                options.output_height =
                    std::stoi(requireValue(i, arg));
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

    if (options.input_path.empty() ||
        options.output_dir.empty() ||
        !options.has_source_points) {
        std::cerr
            << "Error: --input, --output-dir and --src are required.\n";
        return false;
    }

    if (options.output_width < 0 ||
        options.output_height < 0) {
        std::cerr
            << "Error: width/height must be >= 0.\n";
        return false;
    }

    return true;
}

double quadrilateralArea(
    const std::array<cv::Point2f, 4>& points) {
    double twice_area = 0.0;

    for (std::size_t i = 0; i < points.size(); ++i) {
        const cv::Point2f& a = points[i];
        const cv::Point2f& b =
            points[(i + 1) % points.size()];

        twice_area +=
            static_cast<double>(a.x) * b.y -
            static_cast<double>(b.x) * a.y;
    }

    return 0.5 * std::abs(twice_area);
}

bool solveHomography4(
    const std::array<cv::Point2f, 4>& source,
    const std::array<cv::Point2f, 4>& destination,
    cv::Mat& homography) {
    if (quadrilateralArea(source) < 1e-6 ||
        quadrilateralArea(destination) < 1e-6) {
        return false;
    }

    cv::Mat a = cv::Mat::zeros(8, 8, CV_64F);
    cv::Mat b = cv::Mat::zeros(8, 1, CV_64F);

    for (int i = 0; i < 4; ++i) {
        const double x =
            source[static_cast<std::size_t>(i)].x;
        const double y =
            source[static_cast<std::size_t>(i)].y;
        const double u =
            destination[static_cast<std::size_t>(i)].x;
        const double v =
            destination[static_cast<std::size_t>(i)].y;

        const int row_u = 2 * i;
        const int row_v = row_u + 1;

        a.at<double>(row_u, 0) = x;
        a.at<double>(row_u, 1) = y;
        a.at<double>(row_u, 2) = 1.0;
        a.at<double>(row_u, 6) = -u * x;
        a.at<double>(row_u, 7) = -u * y;
        b.at<double>(row_u, 0) = u;

        a.at<double>(row_v, 3) = x;
        a.at<double>(row_v, 4) = y;
        a.at<double>(row_v, 5) = 1.0;
        a.at<double>(row_v, 6) = -v * x;
        a.at<double>(row_v, 7) = -v * y;
        b.at<double>(row_v, 0) = v;
    }

    cv::Mat h;
    if (!cv::solve(a, b, h, cv::DECOMP_LU)) {
        return false;
    }

    homography =
        (cv::Mat_<double>(3, 3) <<
            h.at<double>(0, 0),
            h.at<double>(1, 0),
            h.at<double>(2, 0),
            h.at<double>(3, 0),
            h.at<double>(4, 0),
            h.at<double>(5, 0),
            h.at<double>(6, 0),
            h.at<double>(7, 0),
            1.0);

    return cv::checkRange(homography);
}

bool projectPoint(const cv::Mat& homography,
                  const cv::Point2f& source,
                  cv::Point2f& destination) {
    if (homography.empty() ||
        homography.rows != 3 ||
        homography.cols != 3) {
        return false;
    }

    cv::Mat h64;
    homography.convertTo(h64, CV_64F);

    const double x = source.x;
    const double y = source.y;

    const double w =
        h64.at<double>(2, 0) * x +
        h64.at<double>(2, 1) * y +
        h64.at<double>(2, 2);

    if (std::abs(w) < 1e-12) {
        return false;
    }

    destination.x =
        static_cast<float>(
            (h64.at<double>(0, 0) * x +
             h64.at<double>(0, 1) * y +
             h64.at<double>(0, 2)) / w);

    destination.y =
        static_cast<float>(
            (h64.at<double>(1, 0) * x +
             h64.at<double>(1, 1) * y +
             h64.at<double>(1, 2)) / w);

    return std::isfinite(destination.x) &&
           std::isfinite(destination.y);
}

double pointDistance(const cv::Point2f& a,
                     const cv::Point2f& b) {
    const double dx =
        static_cast<double>(a.x) - b.x;
    const double dy =
        static_cast<double>(a.y) - b.y;

    return std::sqrt(dx * dx + dy * dy);
}

unsigned char bilinearSample(
    const cv::Mat& image,
    double x,
    double y) {
    if (x < 0.0 ||
        y < 0.0 ||
        x > image.cols - 1 ||
        y > image.rows - 1) {
        return 0;
    }

    const int x0 =
        std::clamp(
            static_cast<int>(std::floor(x)),
            0,
            image.cols - 1);

    const int y0 =
        std::clamp(
            static_cast<int>(std::floor(y)),
            0,
            image.rows - 1);

    const int x1 =
        std::min(x0 + 1, image.cols - 1);

    const int y1 =
        std::min(y0 + 1, image.rows - 1);

    const double fx = x - x0;
    const double fy = y - y0;

    const double top =
        (1.0 - fx) *
            image.at<unsigned char>(y0, x0) +
        fx *
            image.at<unsigned char>(y0, x1);

    const double bottom =
        (1.0 - fx) *
            image.at<unsigned char>(y1, x0) +
        fx *
            image.at<unsigned char>(y1, x1);

    const double value =
        (1.0 - fy) * top +
        fy * bottom;

    return cv::saturate_cast<unsigned char>(
        std::round(value));
}

cv::Mat manualWarpPerspective(
    const cv::Mat& source,
    const cv::Mat& source_to_destination,
    cv::Size destination_size) {
    CV_Assert(source.type() == CV_8U);

    cv::Mat inverse;
    if (cv::invert(
            source_to_destination,
            inverse,
            cv::DECOMP_LU) == 0.0) {
        throw std::runtime_error(
            "Homography is not invertible.");
    }

    cv::Mat destination(
        destination_size,
        CV_8U,
        cv::Scalar(0));

    for (int row = 0;
         row < destination.rows;
         ++row) {
        for (int col = 0;
             col < destination.cols;
             ++col) {
            cv::Point2f source_point;

            if (!projectPoint(
                    inverse,
                    cv::Point2f(
                        static_cast<float>(col),
                        static_cast<float>(row)),
                    source_point)) {
                continue;
            }

            destination.at<unsigned char>(
                row,
                col) =
                bilinearSample(
                    source,
                    source_point.x,
                    source_point.y);
        }
    }

    return destination;
}

int inferWidth(
    const std::array<cv::Point2f, 4>& source) {
    const double top =
        pointDistance(source[0], source[1]);
    const double bottom =
        pointDistance(source[3], source[2]);

    return std::max(
        2,
        cvRound(std::max(top, bottom)));
}

int inferHeight(
    const std::array<cv::Point2f, 4>& source) {
    const double left =
        pointDistance(source[0], source[3]);
    const double right =
        pointDistance(source[1], source[2]);

    return std::max(
        2,
        cvRound(std::max(left, right)));
}

std::array<cv::Point2f, 4> destinationRectangle(
    int width,
    int height) {
    return {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(
            static_cast<float>(width - 1),
            0.0f),
        cv::Point2f(
            static_cast<float>(width - 1),
            static_cast<float>(height - 1)),
        cv::Point2f(
            0.0f,
            static_cast<float>(height - 1))
    };
}

cv::Mat annotateSource(
    const cv::Mat& source,
    const std::array<cv::Point2f, 4>& points) {
    cv::Mat output;
    cv::cvtColor(
        source,
        output,
        cv::COLOR_GRAY2BGR);

    for (std::size_t i = 0;
         i < points.size();
         ++i) {
        const cv::Point a(
            cvRound(points[i].x),
            cvRound(points[i].y));

        const cv::Point2f& next =
            points[(i + 1) % points.size()];

        const cv::Point b(
            cvRound(next.x),
            cvRound(next.y));

        cv::circle(
            output,
            a,
            4,
            cv::Scalar(0, 0, 255),
            cv::FILLED,
            cv::LINE_AA);

        cv::line(
            output,
            a,
            b,
            cv::Scalar(0, 255, 0),
            2,
            cv::LINE_AA);

        cv::putText(
            output,
            std::to_string(i),
            a + cv::Point(5, -5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(255, 255, 255),
            1,
            cv::LINE_AA);
    }

    return output;
}

bool writeImage(const fs::path& path,
                const cv::Mat& image) {
    if (!cv::imwrite(
            path.string(),
            image)) {
        std::cerr
            << "Error: failed to write "
            << path << '\n';
        return false;
    }

    return true;
}

double maxAbsDifference(
    const cv::Mat& a,
    const cv::Mat& b) {
    cv::Mat difference;
    cv::absdiff(a, b, difference);

    double max_value = 0.0;
    cv::minMaxLoc(
        difference,
        nullptr,
        &max_value);

    return max_value;
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
                    << name << '\n';
                passed = false;
            }
        };

    const std::array<cv::Point2f, 4> square = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(100.0f, 0.0f),
        cv::Point2f(100.0f, 80.0f),
        cv::Point2f(0.0f, 80.0f)
    };

    cv::Mat identity_h;
    check(
        solveHomography4(
            square,
            square,
            identity_h),
        "identity homography can be solved");

    if (!identity_h.empty()) {
        const std::vector<cv::Point2f> probes = {
            {0.0f, 0.0f},
            {20.0f, 15.0f},
            {75.0f, 60.0f},
            {100.0f, 80.0f}
        };

        for (const auto& probe : probes) {
            cv::Point2f projected;
            check(
                projectPoint(
                    identity_h,
                    probe,
                    projected),
                "identity projection is valid");

            check(
                pointDistance(
                    probe,
                    projected) < 1e-6,
                "identity homography preserves points");
        }
    }

    const cv::Mat known_h =
        (cv::Mat_<double>(3, 3) <<
            1.10, 0.08, 15.0,
           -0.04, 1.06, 10.0,
            0.0007, -0.0003, 1.0);

    std::array<cv::Point2f, 4> warped_square{};

    for (std::size_t i = 0;
         i < square.size();
         ++i) {
        check(
            projectPoint(
                known_h,
                square[i],
                warped_square[i]),
            "known homography projects control point");
    }

    cv::Mat recovered_h;
    check(
        solveHomography4(
            square,
            warped_square,
            recovered_h),
        "manual 8-DoF solve recovers projective mapping");

    if (!recovered_h.empty()) {
        const std::vector<cv::Point2f> probes = {
            {12.0f, 9.0f},
            {50.0f, 40.0f},
            {90.0f, 70.0f}
        };

        for (const auto& probe : probes) {
            cv::Point2f expected;
            cv::Point2f actual;

            check(
                projectPoint(
                    known_h,
                    probe,
                    expected),
                "ground-truth probe projection succeeds");

            check(
                projectPoint(
                    recovered_h,
                    probe,
                    actual),
                "recovered probe projection succeeds");

            check(
                pointDistance(
                    expected,
                    actual) < 1e-4,
                "recovered homography matches ground truth on unseen point");
        }

        cv::Mat inverse;
        check(
            cv::invert(
                recovered_h,
                inverse,
                cv::DECOMP_LU) != 0.0,
            "recovered homography is invertible");

        cv::Point2f mapped;
        cv::Point2f round_trip;

        check(
            projectPoint(
                recovered_h,
                cv::Point2f(37.0f, 29.0f),
                mapped),
            "forward project succeeds");

        check(
            projectPoint(
                inverse,
                mapped,
                round_trip),
            "inverse project succeeds");

        check(
            pointDistance(
                round_trip,
                cv::Point2f(37.0f, 29.0f)) < 1e-4,
            "homography inverse round-trip recovers point");
    }

    const std::array<cv::Point2f, 4> collinear = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(10.0f, 0.0f),
        cv::Point2f(20.0f, 0.0f),
        cv::Point2f(30.0f, 0.0f)
    };

    cv::Mat degenerate_h;
    check(
        !solveHomography4(
            collinear,
            square,
            degenerate_h),
        "collinear source quadrilateral is rejected");

    cv::Mat synthetic(
        48,
        64,
        CV_8U);

    for (int row = 0;
         row < synthetic.rows;
         ++row) {
        for (int col = 0;
             col < synthetic.cols;
             ++col) {
            synthetic.at<unsigned char>(
                row,
                col) =
                static_cast<unsigned char>(
                    (3 * col + 5 * row) % 256);
        }
    }

    const std::array<cv::Point2f, 4> image_corners = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(
            static_cast<float>(synthetic.cols - 1),
            0.0f),
        cv::Point2f(
            static_cast<float>(synthetic.cols - 1),
            static_cast<float>(synthetic.rows - 1)),
        cv::Point2f(
            0.0f,
            static_cast<float>(synthetic.rows - 1))
    };

    cv::Mat image_identity;
    check(
        solveHomography4(
            image_corners,
            image_corners,
            image_identity),
        "image identity homography is solvable");

    if (!image_identity.empty()) {
        const cv::Mat manual =
            manualWarpPerspective(
                synthetic,
                image_identity,
                synthetic.size());

        check(
            maxAbsDifference(
                synthetic,
                manual) == 0.0,
            "manual inverse warp preserves identity image exactly");
    }

    if (passed) {
        std::cout
            << "Homography/perspective self-test passed.\n";
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

    const cv::Mat input =
        cv::imread(
            options.input_path,
            cv::IMREAD_GRAYSCALE);

    if (input.empty()) {
        std::cerr
            << "Error: could not read input image: "
            << options.input_path
            << '\n';
        return 1;
    }

    const int width =
        options.output_width > 0
            ? options.output_width
            : inferWidth(options.source_points);

    const int height =
        options.output_height > 0
            ? options.output_height
            : inferHeight(options.source_points);

    const auto destination =
        destinationRectangle(
            width,
            height);

    cv::Mat homography;
    if (!solveHomography4(
            options.source_points,
            destination,
            homography)) {
        std::cerr
            << "Error: source/destination quadrilateral is degenerate "
            << "or the homography solve failed.\n";
        return 1;
    }

    const cv::Mat manual =
        manualWarpPerspective(
            input,
            homography,
            cv::Size(width, height));

    cv::Mat opencv_reference;
    cv::warpPerspective(
        input,
        opencv_reference,
        homography,
        cv::Size(width, height),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(0));

    cv::Mat difference;
    cv::absdiff(
        manual,
        opencv_reference,
        difference);

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
            "01_source_quadrilateral.png",
        annotateSource(
            input,
            options.source_points));

    ok &= writeImage(
        output_dir /
            "02_manual_rectified.png",
        manual);

    ok &= writeImage(
        output_dir /
            "03_opencv_rectified_reference.png",
        opencv_reference);

    ok &= writeImage(
        output_dir /
            "04_manual_vs_opencv_difference.png",
        difference);

    if (!ok) {
        return 1;
    }

    double mean_difference =
        cv::mean(difference)[0];

    double max_difference = 0.0;
    cv::minMaxLoc(
        difference,
        nullptr,
        &max_difference);

    std::cout
        << "Input: "
        << options.input_path
        << '\n'
        << "Rectified size: "
        << width
        << "x"
        << height
        << '\n'
        << "Source quadrilateral area: "
        << quadrilateralArea(
            options.source_points)
        << " px^2\n"
        << "Manual/OpenCV mean abs difference: "
        << mean_difference
        << '\n'
        << "Manual/OpenCV max abs difference: "
        << max_difference
        << '\n'
        << "Homography (source -> rectified):\n"
        << homography
        << '\n'
        << "Outputs written to: "
        << output_dir
        << '\n';

    return 0;
}
