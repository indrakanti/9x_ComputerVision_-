#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string input_path;
    std::string output_dir;
    int threshold_value = 127;
    double min_area = 100.0;
    double approx_epsilon_fraction = 0.02;
    bool invert = false;
    bool self_test = false;
    bool help = false;
};

struct ComponentInfo {
    int label = -1;
    int area = 0;
    cv::Rect bbox;
    cv::Point2d centroid;
};

struct ShapeDescriptor {
    int contour_index = -1;
    bool is_hole = false;
    double area = 0.0;
    double perimeter = 0.0;
    cv::Rect bbox;
    cv::Point2d centroid;
    double circularity = 0.0;
    double aspect_ratio = 0.0;
    double extent = 0.0;
    double solidity = 0.0;
    int polygon_vertices = 0;
    std::array<double, 7> hu{};
    std::string shape_class;
};

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " --output-dir <dir> [options]\n"
        << "  " << program << " --input <image> --output-dir <dir> [options]\n"
        << "  " << program << " --self-test\n\n"
        << "Options:\n"
        << "  --input <path>           Optional grayscale/color image\n"
        << "  --output-dir <path>      Output directory\n"
        << "  --threshold <0..255>     Binary threshold (default: 127)\n"
        << "  --min-area <pixels>      Ignore tiny outer contours (default: 100)\n"
        << "  --epsilon <fraction>     approxPolyDP fraction of perimeter (default: 0.02)\n"
        << "  --invert                 Invert threshold polarity\n"
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

            if (arg == "--input") {
                options.input_path = value(i, arg);
            } else if (arg == "--output-dir") {
                options.output_dir = value(i, arg);
            } else if (arg == "--threshold") {
                options.threshold_value = std::stoi(value(i, arg));
            } else if (arg == "--min-area") {
                options.min_area = std::stod(value(i, arg));
            } else if (arg == "--epsilon") {
                options.approx_epsilon_fraction = std::stod(value(i, arg));
            } else if (arg == "--invert") {
                options.invert = true;
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

    if (options.threshold_value < 0 ||
        options.threshold_value > 255 ||
        options.min_area < 0.0 ||
        options.approx_epsilon_fraction <= 0.0 ||
        options.approx_epsilon_fraction >= 0.25) {
        std::cerr << "Error: invalid threshold/area/epsilon configuration.\n";
        return false;
    }

    return true;
}

cv::Mat toGray(const cv::Mat& input) {
    if (input.empty()) {
        return cv::Mat();
    }

    if (input.channels() == 1) {
        return input.clone();
    }

    cv::Mat gray;
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

cv::Mat createSyntheticScene() {
    cv::Mat image(
        520,
        820,
        CV_8U,
        cv::Scalar(0));

    // Circle.
    cv::circle(
        image,
        cv::Point(120, 120),
        48,
        cv::Scalar(255),
        cv::FILLED,
        cv::LINE_8);

    // Axis-aligned rectangle.
    cv::rectangle(
        image,
        cv::Rect(260, 65, 125, 100),
        cv::Scalar(255),
        cv::FILLED,
        cv::LINE_8);

    // Triangle.
    const std::vector<cv::Point> triangle = {
        cv::Point(500, 175),
        cv::Point(445, 65),
        cv::Point(570, 75)
    };
    cv::fillConvexPoly(
        image,
        triangle,
        cv::Scalar(255),
        cv::LINE_8);

    // Donut: one connected foreground component with one hole.
    cv::circle(
        image,
        cv::Point(690, 125),
        58,
        cv::Scalar(255),
        cv::FILLED,
        cv::LINE_8);
    cv::circle(
        image,
        cv::Point(690, 125),
        25,
        cv::Scalar(0),
        cv::FILLED,
        cv::LINE_8);

    // A second row of small diagnostic shapes, intentionally below the
    // default min-area filter but still present in the binary image.
    cv::circle(
        image,
        cv::Point(150, 360),
        5,
        cv::Scalar(255),
        cv::FILLED,
        cv::LINE_8);

    cv::rectangle(
        image,
        cv::Rect(300, 350, 8, 8),
        cv::Scalar(255),
        cv::FILLED,
        cv::LINE_8);

    return image;
}

cv::Mat makeBinary(
    const cv::Mat& gray,
    int threshold_value,
    bool invert) {
    cv::Mat binary;

    const int type =
        invert
            ? cv::THRESH_BINARY_INV
            : cv::THRESH_BINARY;

    cv::threshold(
        gray,
        binary,
        threshold_value,
        255,
        type);

    return binary;
}

std::vector<ComponentInfo> connectedComponents(
    const cv::Mat& binary,
    int& total_labels,
    cv::Mat& labels_out) {
    cv::Mat stats;
    cv::Mat centroids;

    total_labels =
        cv::connectedComponentsWithStats(
            binary,
            labels_out,
            stats,
            centroids,
            8,
            CV_32S);

    std::vector<ComponentInfo> components;

    for (int label = 1;
         label < total_labels;
         ++label) {
        ComponentInfo info;
        info.label = label;
        info.area =
            stats.at<int>(
                label,
                cv::CC_STAT_AREA);

        info.bbox =
            cv::Rect(
                stats.at<int>(
                    label,
                    cv::CC_STAT_LEFT),
                stats.at<int>(
                    label,
                    cv::CC_STAT_TOP),
                stats.at<int>(
                    label,
                    cv::CC_STAT_WIDTH),
                stats.at<int>(
                    label,
                    cv::CC_STAT_HEIGHT));

        info.centroid =
            cv::Point2d(
                centroids.at<double>(
                    label,
                    0),
                centroids.at<double>(
                    label,
                    1));

        components.push_back(info);
    }

    return components;
}

cv::Scalar labelColor(int label) {
    const int r =
        70 + (label * 67) % 180;
    const int g =
        70 + (label * 101) % 180;
    const int b =
        70 + (label * 43) % 180;

    return cv::Scalar(b, g, r);
}

cv::Mat visualizeComponents(
    const cv::Mat& labels,
    const std::vector<ComponentInfo>& components) {
    cv::Mat output(
        labels.size(),
        CV_8UC3,
        cv::Scalar(0, 0, 0));

    for (const auto& component : components) {
        cv::Mat mask =
            labels == component.label;

        output.setTo(
            labelColor(component.label),
            mask);

        cv::rectangle(
            output,
            component.bbox,
            cv::Scalar(255, 255, 255),
            1,
            cv::LINE_AA);

        cv::circle(
            output,
            component.centroid,
            3,
            cv::Scalar(255, 255, 255),
            cv::FILLED,
            cv::LINE_AA);

        cv::putText(
            output,
            "L" + std::to_string(component.label) +
                " A=" + std::to_string(component.area),
            component.bbox.tl() + cv::Point(0, -5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.42,
            cv::Scalar(255, 255, 255),
            1,
            cv::LINE_AA);
    }

    return output;
}

std::array<double, 7> huMomentsLog(
    const cv::Moments& moments) {
    double raw[7]{};
    cv::HuMoments(
        moments,
        raw);

    std::array<double, 7> transformed{};

    for (int i = 0; i < 7; ++i) {
        const double value = raw[i];

        if (std::abs(value) < 1e-30) {
            transformed[static_cast<std::size_t>(i)] = 0.0;
            continue;
        }

        transformed[static_cast<std::size_t>(i)] =
            -std::copysign(
                std::log10(
                    std::abs(value)),
                value);
    }

    return transformed;
}

std::string classifyShape(
    int vertices,
    double circularity,
    double aspect_ratio,
    double solidity) {
    if (vertices == 3) {
        return "triangle";
    }

    if (vertices == 4) {
        if (aspect_ratio > 0.90 &&
            aspect_ratio < 1.10) {
            return "square-like";
        }
        return "rectangle";
    }

    if (circularity > 0.82 &&
        solidity > 0.95) {
        return "circle-like";
    }

    return "other";
}

ShapeDescriptor describeContour(
    const std::vector<cv::Point>& contour,
    int contour_index,
    bool is_hole,
    double epsilon_fraction) {
    ShapeDescriptor descriptor;
    descriptor.contour_index =
        contour_index;
    descriptor.is_hole =
        is_hole;

    descriptor.area =
        std::abs(
            cv::contourArea(contour));

    descriptor.perimeter =
        cv::arcLength(
            contour,
            true);

    descriptor.bbox =
        cv::boundingRect(
            contour);

    const cv::Moments moments =
        cv::moments(contour);

    if (std::abs(moments.m00) > 1e-12) {
        descriptor.centroid =
            cv::Point2d(
                moments.m10 / moments.m00,
                moments.m01 / moments.m00);
    }

    if (descriptor.perimeter > 1e-12) {
        descriptor.circularity =
            4.0 *
            CV_PI *
            descriptor.area /
            (descriptor.perimeter *
             descriptor.perimeter);
    }

    if (descriptor.bbox.height > 0) {
        descriptor.aspect_ratio =
            static_cast<double>(
                descriptor.bbox.width) /
            descriptor.bbox.height;
    }

    const double bbox_area =
        static_cast<double>(
            descriptor.bbox.area());

    if (bbox_area > 0.0) {
        descriptor.extent =
            descriptor.area /
            bbox_area;
    }

    std::vector<cv::Point> hull;

    cv::convexHull(
        contour,
        hull);

    const double hull_area =
        std::abs(
            cv::contourArea(hull));

    if (hull_area > 1e-12) {
        descriptor.solidity =
            descriptor.area /
            hull_area;
    }

    std::vector<cv::Point> polygon;

    cv::approxPolyDP(
        contour,
        polygon,
        epsilon_fraction *
            descriptor.perimeter,
        true);

    descriptor.polygon_vertices =
        static_cast<int>(
            polygon.size());

    descriptor.hu =
        huMomentsLog(moments);

    descriptor.shape_class =
        classifyShape(
            descriptor.polygon_vertices,
            descriptor.circularity,
            descriptor.aspect_ratio,
            descriptor.solidity);

    return descriptor;
}

std::vector<ShapeDescriptor> analyzeContours(
    const cv::Mat& binary,
    double min_area,
    double epsilon_fraction,
    std::vector<std::vector<cv::Point>>& contours_out,
    std::vector<cv::Vec4i>& hierarchy_out) {
    cv::Mat work =
        binary.clone();

    cv::findContours(
        work,
        contours_out,
        hierarchy_out,
        cv::RETR_CCOMP,
        cv::CHAIN_APPROX_SIMPLE);

    std::vector<ShapeDescriptor> descriptors;

    for (std::size_t i = 0;
         i < contours_out.size();
         ++i) {
        const bool is_hole =
            !hierarchy_out.empty() &&
            hierarchy_out[i][3] >= 0;

        const double area =
            std::abs(
                cv::contourArea(
                    contours_out[i]));

        if (!is_hole &&
            area < min_area) {
            continue;
        }

        descriptors.push_back(
            describeContour(
                contours_out[i],
                static_cast<int>(i),
                is_hole,
                epsilon_fraction));
    }

    return descriptors;
}

cv::Mat drawContourAnalysis(
    const cv::Mat& gray,
    const std::vector<std::vector<cv::Point>>& contours,
    const std::vector<ShapeDescriptor>& descriptors) {
    cv::Mat output;
    cv::cvtColor(
        gray,
        output,
        cv::COLOR_GRAY2BGR);

    for (const auto& descriptor : descriptors) {
        const cv::Scalar color =
            descriptor.is_hole
                ? cv::Scalar(0, 0, 255)
                : cv::Scalar(0, 255, 0);

        cv::drawContours(
            output,
            contours,
            descriptor.contour_index,
            color,
            2,
            cv::LINE_AA);

        cv::rectangle(
            output,
            descriptor.bbox,
            cv::Scalar(255, 180, 0),
            1,
            cv::LINE_AA);

        cv::circle(
            output,
            descriptor.centroid,
            3,
            cv::Scalar(255, 255, 255),
            cv::FILLED,
            cv::LINE_AA);

        const std::string label =
            descriptor.is_hole
                ? "hole"
                : descriptor.shape_class;

        cv::putText(
            output,
            label,
            descriptor.bbox.tl() +
                cv::Point(0, -6),
            cv::FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv::LINE_AA);
    }

    return output;
}

bool writeCsv(
    const fs::path& path,
    const std::vector<ShapeDescriptor>& descriptors) {
    std::ofstream output(path);

    if (!output) {
        return false;
    }

    output
        << "contour_index,is_hole,shape_class,area,perimeter,"
        << "centroid_x,centroid_y,bbox_x,bbox_y,bbox_w,bbox_h,"
        << "circularity,aspect_ratio,extent,solidity,vertices,"
        << "hu1,hu2,hu3,hu4,hu5,hu6,hu7\n";

    output
        << std::fixed
        << std::setprecision(8);

    for (const auto& descriptor : descriptors) {
        output
            << descriptor.contour_index << ","
            << (descriptor.is_hole ? 1 : 0) << ","
            << descriptor.shape_class << ","
            << descriptor.area << ","
            << descriptor.perimeter << ","
            << descriptor.centroid.x << ","
            << descriptor.centroid.y << ","
            << descriptor.bbox.x << ","
            << descriptor.bbox.y << ","
            << descriptor.bbox.width << ","
            << descriptor.bbox.height << ","
            << descriptor.circularity << ","
            << descriptor.aspect_ratio << ","
            << descriptor.extent << ","
            << descriptor.solidity << ","
            << descriptor.polygon_vertices;

        for (const double value :
             descriptor.hu) {
            output << "," << value;
        }

        output << "\n";
    }

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

const ShapeDescriptor* nearestOuterShape(
    const std::vector<ShapeDescriptor>& descriptors,
    const cv::Point2d& target) {
    const ShapeDescriptor* best = nullptr;
    double best_distance =
        std::numeric_limits<double>::infinity();

    for (const auto& descriptor : descriptors) {
        if (descriptor.is_hole) {
            continue;
        }

        const double dx =
            descriptor.centroid.x -
            target.x;

        const double dy =
            descriptor.centroid.y -
            target.y;

        const double distance =
            std::sqrt(
                dx * dx +
                dy * dy);

        if (distance < best_distance) {
            best_distance = distance;
            best = &descriptor;
        }
    }

    return best;
}

std::array<double, 7> huForTranslatedRectangle(
    int x,
    int y) {
    cv::Mat image(
        220,
        300,
        CV_8U,
        cv::Scalar(0));

    cv::rectangle(
        image,
        cv::Rect(x, y, 80, 50),
        cv::Scalar(255),
        cv::FILLED,
        cv::LINE_8);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(
        image,
        contours,
        hierarchy,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        return {};
    }

    return huMomentsLog(
        cv::moments(contours.front()));
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

    const cv::Mat binary =
        createSyntheticScene();

    int total_labels = 0;
    cv::Mat labels;

    const auto components =
        connectedComponents(
            binary,
            total_labels,
            labels);

    check(
        total_labels == 7,
        "background plus six foreground components are labeled");

    check(
        components.size() == 6,
        "six foreground connected components are reported");

    int large_components = 0;

    for (const auto& component :
         components) {
        if (component.area >= 100) {
            ++large_components;
        }
    }

    check(
        large_components == 4,
        "four large semantic shapes remain after area filtering");

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    const auto descriptors =
        analyzeContours(
            binary,
            100.0,
            0.02,
            contours,
            hierarchy);

    int outer_count = 0;
    int hole_count = 0;

    for (const auto& descriptor :
         descriptors) {
        if (descriptor.is_hole) {
            ++hole_count;
        } else {
            ++outer_count;
        }
    }

    check(
        outer_count == 4,
        "four large outer contours are analyzed");

    check(
        hole_count == 1,
        "donut contributes exactly one retained hole contour");

    const ShapeDescriptor* circle =
        nearestOuterShape(
            descriptors,
            cv::Point2d(120.0, 120.0));

    const ShapeDescriptor* rectangle =
        nearestOuterShape(
            descriptors,
            cv::Point2d(322.0, 114.0));

    const ShapeDescriptor* triangle =
        nearestOuterShape(
            descriptors,
            cv::Point2d(505.0, 105.0));

    const ShapeDescriptor* donut =
        nearestOuterShape(
            descriptors,
            cv::Point2d(690.0, 125.0));

    check(
        circle != nullptr &&
        circle->shape_class == "circle-like",
        "circle is classified as circle-like");

    check(
        circle != nullptr &&
        circle->circularity > 0.85,
        "circle has high circularity");

    check(
        rectangle != nullptr &&
        rectangle->shape_class == "rectangle",
        "rectangle is classified from polygon vertices/aspect ratio");

    check(
        rectangle != nullptr &&
        rectangle->polygon_vertices == 4,
        "rectangle approximates to four vertices");

    check(
        triangle != nullptr &&
        triangle->shape_class == "triangle",
        "triangle is classified from three vertices");

    check(
        triangle != nullptr &&
        triangle->polygon_vertices == 3,
        "triangle approximates to three vertices");

    check(
        donut != nullptr &&
        donut->shape_class == "circle-like",
        "donut outer boundary is circle-like");

    check(
        circle != nullptr &&
        std::abs(circle->centroid.x - 120.0) < 0.5 &&
        std::abs(circle->centroid.y - 120.0) < 0.5,
        "circle centroid is recovered");

    const auto hu_a =
        huForTranslatedRectangle(
            25,
            30);

    const auto hu_b =
        huForTranslatedRectangle(
            145,
            125);

    double max_hu_difference = 0.0;

    for (std::size_t i = 0;
         i < hu_a.size();
         ++i) {
        max_hu_difference =
            std::max(
                max_hu_difference,
                std::abs(
                    hu_a[i] -
                    hu_b[i]));
    }

    check(
        max_hu_difference < 1e-10,
        "Hu moments are translation invariant for identical rectangles");

    if (passed) {
        std::cout
            << "Contours/components/shapes self-test passed.\n";
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

    cv::Mat gray;

    if (options.input_path.empty()) {
        gray =
            createSyntheticScene();
    } else {
        gray =
            toGray(
                cv::imread(
                    options.input_path,
                    cv::IMREAD_UNCHANGED));

        if (gray.empty()) {
            std::cerr
                << "Error: could not read input image: "
                << options.input_path
                << '\n';
            return 1;
        }
    }

    const cv::Mat binary =
        makeBinary(
            gray,
            options.threshold_value,
            options.invert);

    int total_labels = 0;
    cv::Mat labels;

    const auto components =
        connectedComponents(
            binary,
            total_labels,
            labels);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    const auto descriptors =
        analyzeContours(
            binary,
            options.min_area,
            options.approx_epsilon_fraction,
            contours,
            hierarchy);

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
            "01_binary.png",
        binary);

    ok &= writeImage(
        output_dir /
            "02_connected_components.png",
        visualizeComponents(
            labels,
            components));

    ok &= writeImage(
        output_dir /
            "03_contours_and_shapes.png",
        drawContourAnalysis(
            gray,
            contours,
            descriptors));

    ok &= writeCsv(
        output_dir /
            "shape_descriptors.csv",
        descriptors);

    if (!ok) {
        std::cerr
            << "Error: failed to write one or more outputs.\n";
        return 1;
    }

    int retained_outer = 0;
    int retained_holes = 0;

    std::cout
        << std::fixed
        << std::setprecision(4)
        << "Connected-component foreground count: "
        << components.size()
        << '\n'
        << "Total contour count before filtering: "
        << contours.size()
        << '\n'
        << "Retained contour descriptors:\n";

    for (const auto& descriptor :
         descriptors) {
        if (descriptor.is_hole) {
            ++retained_holes;
        } else {
            ++retained_outer;
        }

        std::cout
            << "  contour="
            << descriptor.contour_index
            << " "
            << (descriptor.is_hole
                    ? "hole"
                    : descriptor.shape_class)
            << " area="
            << descriptor.area
            << " perimeter="
            << descriptor.perimeter
            << " centroid=("
            << descriptor.centroid.x
            << ","
            << descriptor.centroid.y
            << ") circularity="
            << descriptor.circularity
            << " aspect="
            << descriptor.aspect_ratio
            << " extent="
            << descriptor.extent
            << " solidity="
            << descriptor.solidity
            << " vertices="
            << descriptor.polygon_vertices
            << '\n';
    }

    std::cout
        << "Retained outer contours: "
        << retained_outer
        << '\n'
        << "Retained holes: "
        << retained_holes
        << '\n'
        << "Outputs written to: "
        << output_dir
        << '\n';

    return 0;
}
