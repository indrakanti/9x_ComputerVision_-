#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string output_dir;
    int sample_factor = 4;
    int quant_bits = 3;
    bool self_test = false;
    bool help = false;
};

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " --output-dir <dir> [options]\n"
        << "  " << program << " --self-test\n\n"
        << "Options:\n"
        << "  --output-dir <path>    Output directory\n"
        << "  --sample-factor <N>    Integer spatial downsample factor (default: 4)\n"
        << "  --quant-bits <1..8>    Output intensity bit depth (default: 3)\n"
        << "  --self-test            Run deterministic tests\n"
        << "  --help                 Show this help\n";
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
            } else if (arg == "--sample-factor") {
                options.sample_factor = std::stoi(value(i, arg));
            } else if (arg == "--quant-bits") {
                options.quant_bits = std::stoi(value(i, arg));
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

    if (options.sample_factor < 2 || options.sample_factor > 16) {
        std::cerr << "Error: --sample-factor must be in [2,16].\n";
        return false;
    }

    if (options.quant_bits < 1 || options.quant_bits > 8) {
        std::cerr << "Error: --quant-bits must be in [1,8].\n";
        return false;
    }

    return true;
}

cv::Mat createSyntheticScene() {
    constexpr int width = 640;
    constexpr int height = 360;

    cv::Mat image(
        height,
        width,
        CV_8UC3,
        cv::Scalar(18, 18, 18));

    // Smooth horizontal grayscale ramp.
    for (int x = 0; x < width; ++x) {
        const unsigned char value =
            static_cast<unsigned char>(
                std::lround(
                    255.0 *
                    x /
                    (width - 1)));

        for (int y = 0; y < 90; ++y) {
            image.at<cv::Vec3b>(y, x) =
                cv::Vec3b(value, value, value);
        }
    }

    // High-frequency black/white stripes that intentionally alias.
    for (int x = 0; x < width; ++x) {
        const unsigned char value =
            ((x / 2) % 2 == 0)
                ? 20
                : 235;

        cv::rectangle(
            image,
            cv::Rect(x, 105, 1, 75),
            cv::Scalar(value, value, value),
            cv::FILLED);
    }

    // Color patches in OpenCV's BGR channel order.
    cv::rectangle(
        image,
        cv::Rect(35, 220, 135, 100),
        cv::Scalar(0, 0, 255),
        cv::FILLED);

    cv::rectangle(
        image,
        cv::Rect(205, 220, 135, 100),
        cv::Scalar(0, 255, 0),
        cv::FILLED);

    cv::rectangle(
        image,
        cv::Rect(375, 220, 135, 100),
        cv::Scalar(255, 0, 0),
        cv::FILLED);

    cv::putText(
        image,
        "R",
        cv::Point(80, 285),
        cv::FONT_HERSHEY_SIMPLEX,
        1.6,
        cv::Scalar(255, 255, 255),
        3,
        cv::LINE_AA);

    cv::putText(
        image,
        "G",
        cv::Point(250, 285),
        cv::FONT_HERSHEY_SIMPLEX,
        1.6,
        cv::Scalar(255, 255, 255),
        3,
        cv::LINE_AA);

    cv::putText(
        image,
        "B",
        cv::Point(420, 285),
        cv::FONT_HERSHEY_SIMPLEX,
        1.6,
        cv::Scalar(255, 255, 255),
        3,
        cv::LINE_AA);

    return image;
}

cv::Mat nearestSample(const cv::Mat& input, int factor) {
    const int width =
        input.cols / factor;
    const int height =
        input.rows / factor;

    cv::Mat output(
        height,
        width,
        input.type());

    for (int y = 0; y < height; ++y) {
        const int src_y =
            y * factor;

        for (int x = 0; x < width; ++x) {
            const int src_x =
                x * factor;

            if (input.type() == CV_8UC1) {
                output.at<unsigned char>(y, x) =
                    input.at<unsigned char>(src_y, src_x);
            } else if (input.type() == CV_8UC3) {
                output.at<cv::Vec3b>(y, x) =
                    input.at<cv::Vec3b>(src_y, src_x);
            } else {
                throw std::runtime_error(
                    "nearestSample supports CV_8UC1 and CV_8UC3.");
            }
        }
    }

    return output;
}

cv::Mat antiAliasedSample(const cv::Mat& input, int factor) {
    cv::Mat filtered;

    int kernel =
        std::max(
            3,
            2 * factor + 1);

    if (kernel % 2 == 0) {
        ++kernel;
    }

    cv::GaussianBlur(
        input,
        filtered,
        cv::Size(kernel, kernel),
        0.5 * factor,
        0.5 * factor,
        cv::BORDER_REFLECT101);

    return nearestSample(
        filtered,
        factor);
}

unsigned char quantizeValue(
    unsigned char value,
    int bits) {
    const int levels =
        1 << bits;

    if (levels >= 256) {
        return value;
    }

    const double normalized =
        static_cast<double>(value) /
        255.0;

    const int index =
        static_cast<int>(
            std::lround(
                normalized *
                (levels - 1)));

    return static_cast<unsigned char>(
        std::lround(
            255.0 *
            index /
            (levels - 1)));
}

cv::Mat quantizeImage(
    const cv::Mat& gray,
    int bits) {
    CV_Assert(
        gray.type() ==
        CV_8UC1);

    cv::Mat output(
        gray.size(),
        CV_8UC1);

    for (int y = 0; y < gray.rows; ++y) {
        for (int x = 0; x < gray.cols; ++x) {
            output.at<unsigned char>(y, x) =
                quantizeValue(
                    gray.at<unsigned char>(y, x),
                    bits);
        }
    }

    return output;
}

unsigned char manualBgrToGray(
    const cv::Vec3b& bgr) {
    const double b =
        bgr[0];
    const double g =
        bgr[1];
    const double r =
        bgr[2];

    return cv::saturate_cast<unsigned char>(
        0.114 * b +
        0.587 * g +
        0.299 * r);
}

cv::Mat makeQuantizationStrip(
    const cv::Mat& gray) {
    const std::array<int, 4> bits = {
        8,
        4,
        2,
        1
    };

    cv::Mat strip(
        gray.rows,
        gray.cols *
            static_cast<int>(bits.size()),
        CV_8UC1);

    for (std::size_t i = 0; i < bits.size(); ++i) {
        const cv::Mat q =
            quantizeImage(
                gray,
                bits[i]);

        q.copyTo(
            strip(
                cv::Rect(
                    static_cast<int>(i) *
                        gray.cols,
                    0,
                    gray.cols,
                    gray.rows)));
    }

    return strip;
}

cv::Mat visualizeChannels(
    const cv::Mat& bgr) {
    std::vector<cv::Mat> channels;

    cv::split(
        bgr,
        channels);

    cv::Mat strip(
        bgr.rows,
        bgr.cols * 3,
        CV_8UC1);

    for (int i = 0; i < 3; ++i) {
        channels[
            static_cast<std::size_t>(i)]
            .copyTo(
                strip(
                    cv::Rect(
                        i * bgr.cols,
                        0,
                        bgr.cols,
                        bgr.rows)));
    }

    return strip;
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

    cv::Mat grid(
        8,
        8,
        CV_8UC1);

    for (int y = 0; y < grid.rows; ++y) {
        for (int x = 0; x < grid.cols; ++x) {
            grid.at<unsigned char>(y, x) =
                static_cast<unsigned char>(
                    y * 10 + x);
        }
    }

    const cv::Mat sampled =
        nearestSample(
            grid,
            2);

    check(
        sampled.rows == 4 &&
        sampled.cols == 4,
        "2x spatial sampling halves each image dimension");

    check(
        sampled.at<unsigned char>(2, 3) ==
            grid.at<unsigned char>(4, 6),
        "nearest sampler uses the expected source lattice point");

    cv::Mat stripes(
        16,
        16,
        CV_8UC1);

    for (int y = 0; y < stripes.rows; ++y) {
        for (int x = 0; x < stripes.cols; ++x) {
            stripes.at<unsigned char>(y, x) =
                (x % 2 == 0)
                    ? 0
                    : 255;
        }
    }

    const cv::Mat aliased =
        nearestSample(
            stripes,
            2);

    check(
        cv::countNonZero(
            aliased) == 0,
        "undersampling a Nyquist-violating stripe pattern aliases to constant black");

    check(
        quantizeValue(0, 2) == 0 &&
        quantizeValue(85, 2) == 85 &&
        quantizeValue(170, 2) == 170 &&
        quantizeValue(255, 2) == 255,
        "2-bit quantization exposes four equally spaced output levels");

    check(
        quantizeValue(128, 1) == 255,
        "1-bit midpoint rounds to the high level");

    check(
        quantizeValue(173, 8) == 173,
        "8-bit quantization preserves an 8-bit input value");

    const cv::Vec3b pure_red(
        0,
        0,
        255);

    cv::Mat red(
        1,
        1,
        CV_8UC3,
        pure_red);

    cv::Mat gray;

    cv::cvtColor(
        red,
        gray,
        cv::COLOR_BGR2GRAY);

    const int manual =
        manualBgrToGray(
            pure_red);

    const int opencv =
        gray.at<unsigned char>(
            0,
            0);

    check(
        std::abs(
            manual -
            opencv) <= 1,
        "manual luma-style BGR-to-gray conversion agrees with OpenCV");

    cv::Mat hsv;
    cv::Mat yuv;

    cv::cvtColor(
        red,
        hsv,
        cv::COLOR_BGR2HSV);

    cv::cvtColor(
        red,
        yuv,
        cv::COLOR_BGR2YUV);

    check(
        hsv.type() ==
            CV_8UC3 &&
        yuv.type() ==
            CV_8UC3,
        "OpenCV color-space transforms preserve three-channel image shape");

    const cv::Mat scene =
        createSyntheticScene();

    check(
        scene.rows == 360 &&
        scene.cols == 640 &&
        scene.type() ==
            CV_8UC3,
        "synthetic image-formation scene has expected dimensions/type");

    const cv::Mat roi =
        scene(
            cv::Rect(
                10,
                10,
                100,
                100));

    check(
        roi.data ==
            scene.ptr<unsigned char>(
                10) +
            10 *
                static_cast<int>(
                    scene.elemSize()),
        "cv::Mat ROI is a view into parent image memory");

    if (passed) {
        std::cout
            << "Image formation self-test passed.\n";
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

    const cv::Mat scene =
        createSyntheticScene();

    cv::Mat gray;

    cv::cvtColor(
        scene,
        gray,
        cv::COLOR_BGR2GRAY);

    const cv::Mat sampled =
        nearestSample(
            scene,
            options.sample_factor);

    const cv::Mat anti_aliased =
        antiAliasedSample(
            scene,
            options.sample_factor);

    const cv::Mat quantized =
        quantizeImage(
            gray,
            options.quant_bits);

    const cv::Mat quant_strip =
        makeQuantizationStrip(
            gray);

    cv::Mat hsv;
    cv::Mat yuv;

    cv::cvtColor(
        scene,
        hsv,
        cv::COLOR_BGR2HSV);

    cv::cvtColor(
        scene,
        yuv,
        cv::COLOR_BGR2YUV);

    bool ok = true;

    ok &= writeImage(
        output_dir /
            "01_sensor_scene_bgr.png",
        scene);

    ok &= writeImage(
        output_dir /
            "02_naive_spatial_sampling.png",
        sampled);

    ok &= writeImage(
        output_dir /
            "03_antialiased_sampling.png",
        anti_aliased);

    ok &= writeImage(
        output_dir /
            "04_grayscale.png",
        gray);

    ok &= writeImage(
        output_dir /
            "05_quantized.png",
        quantized);

    ok &= writeImage(
        output_dir /
            "06_quantization_8_4_2_1_bit.png",
        quant_strip);

    ok &= writeImage(
        output_dir /
            "07_bgr_channels.png",
        visualizeChannels(
            scene));

    ok &= writeImage(
        output_dir /
            "08_hsv_encoded.png",
        hsv);

    ok &= writeImage(
        output_dir /
            "09_yuv_encoded.png",
        yuv);

    if (!ok) {
        return 1;
    }

    const int levels =
        1 <<
        options.quant_bits;

    std::cout
        << "Image formation demo\n"
        << "  source: "
        << scene.cols
        << "x"
        << scene.rows
        << " CV_8UC3 (OpenCV BGR order)\n"
        << "  spatial sample factor: "
        << options.sample_factor
        << " -> "
        << sampled.cols
        << "x"
        << sampled.rows
        << '\n'
        << "  quantization bits: "
        << options.quant_bits
        << " -> "
        << levels
        << " representable intensity levels\n"
        << "  grayscale storage: "
        << gray.total() *
               gray.elemSize()
        << " bytes\n"
        << "  BGR storage: "
        << scene.total() *
               scene.elemSize()
        << " bytes\n"
        << "Outputs written to: "
        << output_dir
        << '\n';

    return 0;
}
