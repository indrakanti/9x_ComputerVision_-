#include <opencv2/opencv.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string input_path;
    std::string output_dir;
    int kernel_size = 5;
    double sigma = 1.2;
    int iterations = 50;
    std::string border = "reflect101";
    bool self_test = false;
    bool help = false;
};

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " --input <image> --output-dir <directory> [options]\n"
        << "  " << program << " --self-test\n\n"
        << "Options:\n"
        << "  --kernel-size <odd>  Odd kernel size >= 3 (default: 5)\n"
        << "  --sigma <double>     Gaussian sigma > 0 (default: 1.2)\n"
        << "  --border <mode>      reflect101 or replicate (default: reflect101)\n"
        << "  --iterations <int>   Benchmark iterations; 0 disables timing (default: 50)\n"
        << "  --self-test          Run deterministic in-memory validation\n"
        << "  --help               Show this help text\n";
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
            } else if (arg == "--kernel-size") {
                options.kernel_size = std::stoi(requireValue(i, arg));
            } else if (arg == "--sigma") {
                options.sigma = std::stod(requireValue(i, arg));
            } else if (arg == "--border") {
                options.border = requireValue(i, arg);
            } else if (arg == "--iterations") {
                options.iterations = std::stoi(requireValue(i, arg));
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
    if (options.input_path.empty()) {
        std::cerr << "Error: --input is required.\n";
        return false;
    }
    if (options.output_dir.empty()) {
        std::cerr << "Error: --output-dir is required.\n";
        return false;
    }
    if (options.kernel_size < 3 || options.kernel_size % 2 == 0) {
        std::cerr << "Error: --kernel-size must be an odd integer >= 3.\n";
        return false;
    }
    if (options.sigma <= 0.0) {
        std::cerr << "Error: --sigma must be > 0.\n";
        return false;
    }
    if (options.iterations < 0 || options.iterations > 10000) {
        std::cerr << "Error: --iterations must be in the range 0..10000.\n";
        return false;
    }
    if (options.border != "reflect101" && options.border != "replicate") {
        std::cerr << "Error: --border must be reflect101 or replicate.\n";
        return false;
    }
    return true;
}

int borderType(const std::string& border) {
    return border == "replicate" ? cv::BORDER_REPLICATE : cv::BORDER_REFLECT_101;
}

cv::Mat makeBoxKernel(int kernel_size) {
    return cv::Mat::ones(kernel_size, kernel_size, CV_32F) /
           static_cast<float>(kernel_size * kernel_size);
}

cv::Mat makeGaussian1D(int kernel_size, double sigma) {
    return cv::getGaussianKernel(kernel_size, sigma, CV_32F);
}

cv::Mat makeGaussian2D(const cv::Mat& gaussian_1d) {
    return gaussian_1d * gaussian_1d.t();
}

double maxAbsDiff(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat difference;
    cv::absdiff(a, b, difference);
    double max_value = 0.0;
    cv::minMaxLoc(difference, nullptr, &max_value);
    return max_value;
}

bool writeFloatImage(const fs::path& path, const cv::Mat& image) {
    cv::Mat output;
    image.convertTo(output, CV_8U);
    if (!cv::imwrite(path.string(), output)) {
        std::cerr << "Error: failed to write " << path << '\n';
        return false;
    }
    return true;
}

bool writeDifferenceImage(const fs::path& path, const cv::Mat& a, const cv::Mat& b) {
    cv::Mat difference;
    cv::absdiff(a, b, difference);

    double max_value = 0.0;
    cv::minMaxLoc(difference, nullptr, &max_value);

    cv::Mat output;
    if (max_value <= 1e-12) {
        output = cv::Mat::zeros(difference.size(), CV_8U);
    } else {
        cv::normalize(difference, output, 0, 255, cv::NORM_MINMAX, CV_8U);
    }

    if (!cv::imwrite(path.string(), output)) {
        std::cerr << "Error: failed to write " << path << '\n';
        return false;
    }
    return true;
}

template <typename Function>
double benchmarkMilliseconds(int iterations, Function&& function) {
    function();  // warm-up
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        function();
    }
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    return elapsed / static_cast<double>(iterations);
}

bool runSelfTestForBorder(int border_type) {
    const int kernel_size = 5;
    const double sigma = 1.2;

    const cv::Mat input = (cv::Mat_<float>(5, 5) <<
        0, 10, 20, 30, 40,
        50, 60, 70, 80, 90,
        100, 110, 120, 130, 140,
        150, 160, 170, 180, 190,
        200, 210, 220, 230, 240);

    const cv::Mat box_kernel = makeBoxKernel(kernel_size);
    const cv::Mat gaussian_1d = makeGaussian1D(kernel_size, sigma);
    const cv::Mat gaussian_2d = makeGaussian2D(gaussian_1d);

    cv::Mat box_filter2d;
    cv::Mat box_reference;
    cv::filter2D(input, box_filter2d, CV_32F, box_kernel, cv::Point(-1, -1), 0.0, border_type);
    cv::boxFilter(input, box_reference, CV_32F, cv::Size(kernel_size, kernel_size),
                  cv::Point(-1, -1), true, border_type);

    cv::Mat gaussian_full;
    cv::Mat gaussian_separable;
    cv::Mat gaussian_reference;
    cv::filter2D(input, gaussian_full, CV_32F, gaussian_2d,
                 cv::Point(-1, -1), 0.0, border_type);
    cv::sepFilter2D(input, gaussian_separable, CV_32F, gaussian_1d, gaussian_1d,
                    cv::Point(-1, -1), 0.0, border_type);
    cv::GaussianBlur(input, gaussian_reference, cv::Size(kernel_size, kernel_size),
                     sigma, sigma, border_type);

    bool passed = true;
    const auto check = [&passed](const std::string& name, double value, double tolerance) {
        if (value > tolerance) {
            std::cerr << "Self-test failed: " << name << " = " << value
                      << " (tolerance " << tolerance << ")\n";
            passed = false;
        }
    };

    check("box kernel sum", std::abs(cv::sum(box_kernel)[0] - 1.0), 1e-6);
    check("Gaussian kernel sum", std::abs(cv::sum(gaussian_2d)[0] - 1.0), 1e-5);
    check("box filter2D vs boxFilter", maxAbsDiff(box_filter2d, box_reference), 1e-4);
    check("Gaussian 2D vs separable", maxAbsDiff(gaussian_full, gaussian_separable), 1e-3);
    check("Gaussian separable vs GaussianBlur", maxAbsDiff(gaussian_separable, gaussian_reference), 1e-3);

    const cv::Mat a = (cv::Mat_<float>(3, 3) <<
        1, 2, 3,
        4, 5, 6,
        7, 8, 9);
    const cv::Mat b = (cv::Mat_<float>(3, 3) <<
        9, 8, 7,
        6, 5, 4,
        3, 2, 1);
    const cv::Mat small_kernel = makeBoxKernel(3);

    cv::Mat filtered_a;
    cv::Mat filtered_b;
    cv::Mat filtered_combination;
    cv::filter2D(a, filtered_a, CV_32F, small_kernel, cv::Point(-1, -1), 0.0, border_type);
    cv::filter2D(b, filtered_b, CV_32F, small_kernel, cv::Point(-1, -1), 0.0, border_type);
    cv::filter2D(0.5F * a + 2.0F * b, filtered_combination, CV_32F, small_kernel,
                 cv::Point(-1, -1), 0.0, border_type);
    check("linearity", maxAbsDiff(filtered_combination, 0.5F * filtered_a + 2.0F * filtered_b), 1e-4);

    return passed;
}

bool runSelfTest() {
    bool passed = true;
    passed &= runSelfTestForBorder(cv::BORDER_REFLECT_101);
    passed &= runSelfTestForBorder(cv::BORDER_REPLICATE);

    if (passed) {
        std::cout << "Linear-filtering self-test passed.\n";
    }
    return passed;
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

    const cv::Mat input_u8 = cv::imread(options.input_path, cv::IMREAD_GRAYSCALE);
    if (input_u8.empty()) {
        std::cerr << "Error: could not read input image: " << options.input_path << '\n';
        return 1;
    }

    cv::Mat input;
    input_u8.convertTo(input, CV_32F);

    const int border = borderType(options.border);
    const cv::Mat box_kernel = makeBoxKernel(options.kernel_size);
    const cv::Mat gaussian_1d = makeGaussian1D(options.kernel_size, options.sigma);
    const cv::Mat gaussian_2d = makeGaussian2D(gaussian_1d);

    cv::Mat box_filter2d;
    cv::Mat box_reference;
    cv::Mat gaussian_full;
    cv::Mat gaussian_separable;
    cv::Mat gaussian_reference;

    cv::filter2D(input, box_filter2d, CV_32F, box_kernel,
                 cv::Point(-1, -1), 0.0, border);
    cv::boxFilter(input, box_reference, CV_32F,
                  cv::Size(options.kernel_size, options.kernel_size),
                  cv::Point(-1, -1), true, border);
    cv::filter2D(input, gaussian_full, CV_32F, gaussian_2d,
                 cv::Point(-1, -1), 0.0, border);
    cv::sepFilter2D(input, gaussian_separable, CV_32F,
                    gaussian_1d, gaussian_1d, cv::Point(-1, -1), 0.0, border);
    cv::GaussianBlur(input, gaussian_reference,
                     cv::Size(options.kernel_size, options.kernel_size),
                     options.sigma, options.sigma, border);

    const fs::path output_dir(options.output_dir);
    std::error_code error;
    fs::create_directories(output_dir, error);
    if (error) {
        std::cerr << "Error: could not create output directory " << output_dir
                  << ": " << error.message() << '\n';
        return 1;
    }

    bool ok = true;
    ok &= writeFloatImage(output_dir / "box.png", box_reference);
    ok &= writeFloatImage(output_dir / "gaussian_2d.png", gaussian_full);
    ok &= writeFloatImage(output_dir / "gaussian_separable.png", gaussian_separable);
    ok &= writeFloatImage(output_dir / "gaussian_opencv.png", gaussian_reference);
    ok &= writeDifferenceImage(output_dir / "box_vs_gaussian_difference.png",
                               box_reference, gaussian_reference);
    ok &= writeDifferenceImage(output_dir / "gaussian_2d_vs_separable_difference.png",
                               gaussian_full, gaussian_separable);
    if (!ok) {
        return 1;
    }

    std::cout << std::fixed << std::setprecision(6)
              << "Input: " << options.input_path << '\n'
              << "Image: " << input.cols << "x" << input.rows << " pixels\n"
              << "Kernel size: " << options.kernel_size << "x" << options.kernel_size << '\n'
              << "Sigma: " << options.sigma << '\n'
              << "Border: " << options.border << '\n'
              << "Box kernel sum: " << cv::sum(box_kernel)[0] << '\n'
              << "Gaussian kernel sum: " << cv::sum(gaussian_2d)[0] << '\n'
              << "Max diff, box filter2D vs boxFilter: "
              << maxAbsDiff(box_filter2d, box_reference) << '\n'
              << "Max diff, Gaussian 2D vs separable: "
              << maxAbsDiff(gaussian_full, gaussian_separable) << '\n'
              << "Max diff, separable vs GaussianBlur: "
              << maxAbsDiff(gaussian_separable, gaussian_reference) << '\n';

    if (options.kernel_size <= 9) {
        std::cout << "\nBox kernel:\n" << box_kernel
                  << "\n\nGaussian 1D kernel:\n" << gaussian_1d
                  << "\n\nGaussian 2D kernel:\n" << gaussian_2d << "\n";
    }

    if (options.iterations > 0) {
        cv::Mat benchmark_output;
        const double box_ms = benchmarkMilliseconds(options.iterations, [&] {
            cv::boxFilter(input, benchmark_output, CV_32F,
                          cv::Size(options.kernel_size, options.kernel_size),
                          cv::Point(-1, -1), true, border);
        });
        const double gaussian_2d_ms = benchmarkMilliseconds(options.iterations, [&] {
            cv::filter2D(input, benchmark_output, CV_32F, gaussian_2d,
                         cv::Point(-1, -1), 0.0, border);
        });
        const double gaussian_sep_ms = benchmarkMilliseconds(options.iterations, [&] {
            cv::sepFilter2D(input, benchmark_output, CV_32F, gaussian_1d, gaussian_1d,
                            cv::Point(-1, -1), 0.0, border);
        });
        const double gaussian_api_ms = benchmarkMilliseconds(options.iterations, [&] {
            cv::GaussianBlur(input, benchmark_output,
                             cv::Size(options.kernel_size, options.kernel_size),
                             options.sigma, options.sigma, border);
        });

        std::cout << "\nAverage runtime over " << options.iterations << " iterations:\n"
                  << "  boxFilter:            " << box_ms << " ms\n"
                  << "  Gaussian 2D filter2D: " << gaussian_2d_ms << " ms\n"
                  << "  Gaussian sepFilter2D: " << gaussian_sep_ms << " ms\n"
                  << "  GaussianBlur:         " << gaussian_api_ms << " ms\n"
                  << "Timing is illustrative; measure on the deployment target before making design decisions.\n";
    }

    std::cout << "\nOutputs written to: " << output_dir << '\n';
    return 0;
}
