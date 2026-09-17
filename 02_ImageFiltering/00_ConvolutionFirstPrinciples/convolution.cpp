#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string input_path;
    std::string output_dir;
    std::string kernel_name = "box";
    std::string operation = "correlation";
    std::string border = "replicate";
    bool self_test = false;
    bool help = false;
};

cv::Mat kernelFromName(const std::string& name) {
    if (name == "box") {
        return (cv::Mat_<float>(3, 3) <<
                1.0F / 9.0F, 1.0F / 9.0F, 1.0F / 9.0F,
                1.0F / 9.0F, 1.0F / 9.0F, 1.0F / 9.0F,
                1.0F / 9.0F, 1.0F / 9.0F, 1.0F / 9.0F);
    }
    if (name == "sharpen") {
        return (cv::Mat_<float>(3, 3) <<
                0, -1, 0,
               -1,  5, -1,
                0, -1, 0);
    }
    if (name == "edge-x") {
        return (cv::Mat_<float>(3, 3) <<
               -1, 0, 1,
               -2, 0, 2,
               -1, 0, 1);
    }
    throw std::runtime_error("Unknown kernel: " + name);
}

int borderType(const std::string& border) {
    if (border == "constant") {
        return cv::BORDER_CONSTANT;
    }
    if (border == "replicate") {
        return cv::BORDER_REPLICATE;
    }
    throw std::runtime_error("Unknown border: " + border);
}

cv::Mat flipKernel(const cv::Mat& kernel) {
    cv::Mat flipped;
    cv::flip(kernel, flipped, -1);
    return flipped;
}

cv::Mat manualCorrelation(const cv::Mat& input_u8,
                          const cv::Mat& kernel,
                          int border_type) {
    if (input_u8.type() != CV_8UC1 || kernel.type() != CV_32FC1) {
        throw std::runtime_error("manualCorrelation expects CV_8UC1 input and CV_32FC1 kernel");
    }
    if (kernel.rows % 2 == 0 || kernel.cols % 2 == 0) {
        throw std::runtime_error("Kernel dimensions must be odd");
    }

    const int pad_y = kernel.rows / 2;
    const int pad_x = kernel.cols / 2;
    cv::Mat padded;
    cv::copyMakeBorder(input_u8, padded, pad_y, pad_y, pad_x, pad_x,
                       border_type, cv::Scalar(0));

    cv::Mat output(input_u8.size(), CV_32FC1, cv::Scalar(0));

    for (int y = 0; y < input_u8.rows; ++y) {
        for (int x = 0; x < input_u8.cols; ++x) {
            float sum = 0.0F;
            for (int ky = 0; ky < kernel.rows; ++ky) {
                for (int kx = 0; kx < kernel.cols; ++kx) {
                    const float pixel = static_cast<float>(padded.at<unsigned char>(y + ky, x + kx));
                    sum += pixel * kernel.at<float>(ky, kx);
                }
            }
            output.at<float>(y, x) = sum;
        }
    }

    return output;
}

cv::Mat manualFilter(const cv::Mat& input_u8,
                     const cv::Mat& kernel,
                     const std::string& operation,
                     int border_type) {
    if (operation == "correlation") {
        return manualCorrelation(input_u8, kernel, border_type);
    }
    if (operation == "convolution") {
        return manualCorrelation(input_u8, flipKernel(kernel), border_type);
    }
    throw std::runtime_error("Unknown operation: " + operation);
}

cv::Mat openCvReference(const cv::Mat& input_u8,
                        const cv::Mat& kernel,
                        const std::string& operation,
                        int border_type) {
    cv::Mat reference;
    const cv::Mat effective_kernel =
        operation == "convolution" ? flipKernel(kernel) : kernel;
    cv::filter2D(input_u8, reference, CV_32F, effective_kernel,
                 cv::Point(-1, -1), 0.0, border_type);
    return reference;
}

cv::Mat normalizeForDisplay(const cv::Mat& image_f32) {
    cv::Mat display;
    double min_value = 0.0;
    double max_value = 0.0;
    cv::minMaxLoc(image_f32, &min_value, &max_value);

    if (min_value >= 0.0 && max_value <= 255.0) {
        image_f32.convertTo(display, CV_8U);
        return display;
    }

    if (max_value == min_value) {
        return cv::Mat(image_f32.size(), CV_8U, cv::Scalar(0));
    }

    cv::normalize(image_f32, display, 0, 255, cv::NORM_MINMAX, CV_8U);
    return display;
}

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " --input <image> --output-dir <dir> [options]\n"
        << "  " << program << " --self-test\n\n"
        << "Options:\n"
        << "  --kernel <box|sharpen|edge-x>            default: box\n"
        << "  --operation <correlation|convolution>    default: correlation\n"
        << "  --border <constant|replicate>            default: replicate\n"
        << "  --self-test\n"
        << "  --help\n";
}

bool parseArguments(int argc, char** argv, Options& options) {
    auto value = [argc, argv](int& index, const std::string& option) {
        if (index + 1 >= argc) {
            throw std::runtime_error("Missing value for " + option);
        }
        return std::string(argv[++index]);
    };

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--input") {
                options.input_path = value(i, arg);
            } else if (arg == "--output-dir") {
                options.output_dir = value(i, arg);
            } else if (arg == "--kernel") {
                options.kernel_name = value(i, arg);
            } else if (arg == "--operation") {
                options.operation = value(i, arg);
            } else if (arg == "--border") {
                options.border = value(i, arg);
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

bool runSelfTest() {
    const cv::Mat input = (cv::Mat_<unsigned char>(3, 3) <<
        10, 20, 30,
        40, 50, 60,
        70, 80, 90);

    const cv::Mat asymmetric = (cv::Mat_<float>(3, 3) <<
        1, 2, 3,
        0, 0, 0,
       -1,-2,-3);

    bool passed = true;
    for (const std::string& operation : {"correlation", "convolution"}) {
        for (const std::string& border : {"constant", "replicate"}) {
            const int cv_border = borderType(border);
            const cv::Mat manual = manualFilter(input, asymmetric, operation, cv_border);
            const cv::Mat reference = openCvReference(input, asymmetric, operation, cv_border);
            const double error = cv::norm(manual, reference, cv::NORM_INF);
            if (error > 1e-5) {
                std::cerr << "Self-test failed for " << operation << "/" << border
                          << ", max error=" << error << '\n';
                passed = false;
            }
        }
    }

    if (passed) {
        std::cout << "Convolution lesson self-test passed.\n";
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
    if (options.self_test) {
        return runSelfTest() ? 0 : 1;
    }
    if (options.input_path.empty() || options.output_dir.empty()) {
        std::cerr << "Error: --input and --output-dir are required.\n";
        printUsage(argv[0]);
        return 2;
    }

    try {
        const cv::Mat kernel = kernelFromName(options.kernel_name);
        const int cv_border = borderType(options.border);

        const cv::Mat input = cv::imread(options.input_path, cv::IMREAD_GRAYSCALE);
        if (input.empty()) {
            std::cerr << "Error: could not read input image: " << options.input_path << '\n';
            return 1;
        }

        const cv::Mat manual = manualFilter(input, kernel, options.operation, cv_border);
        const cv::Mat reference = openCvReference(input, kernel, options.operation, cv_border);
        const double max_error = cv::norm(manual, reference, cv::NORM_INF);

        const fs::path output_dir(options.output_dir);
        std::error_code error;
        fs::create_directories(output_dir, error);
        if (error) {
            std::cerr << "Error creating output directory: " << error.message() << '\n';
            return 1;
        }

        cv::imwrite((output_dir / "manual.png").string(), normalizeForDisplay(manual));
        cv::imwrite((output_dir / "opencv_reference.png").string(), normalizeForDisplay(reference));

        cv::Mat difference;
        cv::absdiff(manual, reference, difference);
        cv::imwrite((output_dir / "difference.png").string(), normalizeForDisplay(difference));

        std::cout << "Kernel: " << options.kernel_name << '\n'
                  << "Operation: " << options.operation << '\n'
                  << "Border: " << options.border << '\n'
                  << "Manual vs OpenCV max absolute error: " << max_error << '\n'
                  << "Outputs: " << output_dir << '\n';

        return max_error <= 1e-5 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 2;
    }
}
