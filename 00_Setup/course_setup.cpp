#include <opencv2/opencv.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string output_dir;
    bool self_test = false;
    bool help = false;
};

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " --output-dir <dir>\n"
        << "  " << program << " --self-test\n\n"
        << "Options:\n"
        << "  --output-dir <path>   Write sanity image + environment report\n"
        << "  --self-test           Run deterministic environment checks\n"
        << "  --help                Show this help\n";
}

bool parseArguments(
    int argc,
    char** argv,
    Options& options) {
    try {
        for (int i = 1;
             i < argc;
             ++i) {
            const std::string arg =
                argv[i];

            if (arg == "--output-dir") {
                if (i + 1 >= argc) {
                    throw std::runtime_error(
                        "Missing value for --output-dir");
                }

                options.output_dir =
                    argv[++i];
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

cv::Mat makeSanityImage() {
    constexpr int width = 320;
    constexpr int height = 240;

    cv::Mat image(
        height,
        width,
        CV_8UC3,
        cv::Scalar(20, 20, 20));

    // Deterministic grid.
    for (int y = 0;
         y < height;
         y += 40) {
        cv::line(
            image,
            cv::Point(0, y),
            cv::Point(
                width - 1,
                y),
            cv::Scalar(60, 60, 60),
            1,
            cv::LINE_8);
    }

    for (int x = 0;
         x < width;
         x += 40) {
        cv::line(
            image,
            cv::Point(x, 0),
            cv::Point(
                x,
                height - 1),
            cv::Scalar(60, 60, 60),
            1,
            cv::LINE_8);
    }

    cv::rectangle(
        image,
        cv::Rect(
            35,
            45,
            90,
            70),
        cv::Scalar(40, 190, 80),
        cv::FILLED,
        cv::LINE_8);

    cv::circle(
        image,
        cv::Point(
            220,
            105),
        42,
        cv::Scalar(220, 110, 40),
        cv::FILLED,
        cv::LINE_8);

    cv::putText(
        image,
        "9x CV",
        cv::Point(
            95,
            200),
        cv::FONT_HERSHEY_SIMPLEX,
        1.0,
        cv::Scalar(240, 240, 240),
        2,
        cv::LINE_AA);

    return image;
}

std::uint64_t checksum(
    const cv::Mat& image) {
    CV_Assert(
        image.type() ==
        CV_8UC1);

    std::uint64_t total = 0;

    for (int row = 0;
         row < image.rows;
         ++row) {
        const auto* ptr =
            image.ptr<unsigned char>(
                row);

        for (int col = 0;
             col < image.cols;
             ++col) {
            total +=
                ptr[col];
        }
    }

    return total;
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
        __cplusplus >= 201703L,
        "compiler reports C++17 or newer");

    check(
        CV_VERSION_MAJOR >= 4,
        "OpenCV major version is 4 or newer");

    const cv::Mat image =
        makeSanityImage();

    check(
        image.rows == 240 &&
        image.cols == 320 &&
        image.type() ==
            CV_8UC3,
        "sanity image has expected shape and type");

    cv::Mat gray;

    cv::cvtColor(
        image,
        gray,
        cv::COLOR_BGR2GRAY);

    check(
        gray.rows ==
            image.rows &&
        gray.cols ==
            image.cols &&
        gray.type() ==
            CV_8UC1,
        "OpenCV imgproc converts BGR to grayscale");

    cv::Mat blurred;

    cv::GaussianBlur(
        gray,
        blurred,
        cv::Size(5, 5),
        1.0);

    check(
        !blurred.empty() &&
        blurred.size() ==
            gray.size() &&
        blurred.type() ==
            CV_8UC1,
        "OpenCV Gaussian blur executes successfully");

    const std::uint64_t gray_checksum =
        checksum(gray);

    check(
        gray_checksum ==
            3425516ULL,
        "deterministic grayscale checksum matches expected value");

    check(
        cv::getTickFrequency() >
            0.0,
        "OpenCV high-resolution timing is available");

    if (passed) {
        std::cout
            << "Course setup self-test passed.\n"
            << "C++ language level: "
            << __cplusplus
            << '\n'
            << "OpenCV version: "
            << CV_VERSION
            << '\n';
    }

    return passed;
}

bool writeReport(
    const fs::path& path,
    const cv::Mat& image,
    const cv::Mat& gray) {
    std::ofstream output(path);

    if (!output) {
        return false;
    }

    output
        << "9x Computer Vision environment sanity report\n"
        << "============================================\n"
        << "C++ language level: "
        << __cplusplus
        << "\n"
        << "OpenCV version: "
        << CV_VERSION
        << "\n"
        << "OpenCV major/minor/patch: "
        << CV_VERSION_MAJOR
        << "."
        << CV_VERSION_MINOR
        << "."
        << CV_VERSION_REVISION
        << "\n"
        << "OpenCV optimized code paths enabled: "
        << (cv::useOptimized()
                ? "yes"
                : "no")
        << "\n"
        << "OpenCV thread count: "
        << cv::getNumThreads()
        << "\n"
        << "Tick frequency: "
        << std::fixed
        << std::setprecision(0)
        << cv::getTickFrequency()
        << " ticks/s\n"
        << "Image size: "
        << image.cols
        << "x"
        << image.rows
        << "\n"
        << "Image type: CV_8UC3\n"
        << "Grayscale checksum: "
        << checksum(gray)
        << "\n";

    return true;
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
        printUsage(argv[0]);
        return 2;
    }

    if (options.help) {
        printUsage(argv[0]);
        return 0;
    }

    if (options.self_test) {
        return runSelfTest()
            ? 0
            : 1;
    }

    if (options.output_dir.empty()) {
        std::cerr
            << "Error: --output-dir is required unless --self-test is used.\n";
        printUsage(argv[0]);
        return 2;
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

    const cv::Mat image =
        makeSanityImage();

    cv::Mat gray;

    cv::cvtColor(
        image,
        gray,
        cv::COLOR_BGR2GRAY);

    const fs::path image_path =
        output_dir /
        "setup_sanity.png";

    const fs::path report_path =
        output_dir /
        "setup_report.txt";

    if (!cv::imwrite(
            image_path.string(),
            image)) {
        std::cerr
            << "Error: failed to write "
            << image_path
            << '\n';
        return 1;
    }

    if (!writeReport(
            report_path,
            image,
            gray)) {
        std::cerr
            << "Error: failed to write "
            << report_path
            << '\n';
        return 1;
    }

    std::cout
        << "9x Computer Vision setup is working.\n"
        << "C++ language level: "
        << __cplusplus
        << '\n'
        << "OpenCV version: "
        << CV_VERSION
        << '\n'
        << "Sanity image: "
        << image_path
        << '\n'
        << "Environment report: "
        << report_path
        << '\n';

    return 0;
}
