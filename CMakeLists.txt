cmake_minimum_required(VERSION 3.16)

project(9x_computer_vision
    VERSION 0.2.0
    DESCRIPTION "Computer Vision from First Principles to Production C++ on Linux"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CV9X_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS
    core
    imgproc
    imgcodecs
    highgui
    features2d)

include(CTest)

function(add_cv9x_example target source)
    add_executable(${target} "${source}")
    target_include_directories(${target} PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE ${OpenCV_LIBS})

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wpedantic)
        if(CV9X_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
endfunction()

# -----------------------------------------------------------------------------
# Classical vision lessons currently present in the repository.
#
# Target names are intentionally stable and independent of the legacy directory
# spelling. Directory renames can therefore happen later without changing the
# commands students use to build individual examples.
# -----------------------------------------------------------------------------

# 02 - Image filtering
add_cv9x_example(cv9x_point_filter
    "02_ImageFiltering/01_PointFiltering/03_TransformationInCpp/pointfiltering.cpp")

if(BUILD_TESTING)
    add_test(NAME cv9x_point_filter_self_test
        COMMAND cv9x_point_filter --self-test)
endif()

add_cv9x_example(cv9x_box_filter
    "02_ImageFiltering/02_LinearFiltering/02_BoxFilter/boxfilter.cpp")
add_cv9x_example(cv9x_gaussian_2d
    "02_ImageFiltering/02_LinearFiltering/03_GaussianFilter/gaussian2d.cpp")
add_cv9x_example(cv9x_gaussian_blend_outputs
    "02_ImageFiltering/02_LinearFiltering/03_GaussianFilter/blendgaussianOutputs.cpp")
add_cv9x_example(cv9x_gaussian_pyramid
    "02_ImageFiltering/02_LinearFiltering/03_GaussianFilter/gaussianPyramid.cpp")
add_cv9x_example(cv9x_laplacian_residuals
    "02_ImageFiltering/02_LinearFiltering/04_Laplacian Pyramid and Image Residuals/lpimageresiduals.cpp")

# 03 - Image gradients
add_cv9x_example(cv9x_sobel
    "03_ImageGradients/02_sobel/sobel.cpp")
add_cv9x_example(cv9x_laplacian
    "03_ImageGradients/03_Laplacian/laplacian.cpp")

# 04 - Image blending
add_cv9x_example(cv9x_laplacian_blending
    "04_ImageBlending/02_LaplacianBlending/lpblending.cpp")

# 05 - Derivatives, edges, and Harris corner concepts
add_cv9x_example(cv9x_edge_partial_derivatives
    "05_DerivativesEdgeDetection/02_EdgeDetectionPartialDerivates/edgedetection.cpp")
add_cv9x_example(cv9x_harris_corner
    "05_DerivativesEdgeDetection/03_HarrisCorner/HarrisCorner.cpp")
add_cv9x_example(cv9x_harris_covariance
    "05_DerivativesEdgeDetection/03_HarrisCorner/HarrisCorner_CovarianceMatrix.cpp")
add_cv9x_example(cv9x_harris_error_coefficient
    "05_DerivativesEdgeDetection/03_HarrisCorner/Hc_covarianceWithErrorCoeff.cpp")
add_cv9x_example(cv9x_harris_taylor_plane
    "05_DerivativesEdgeDetection/04_HarrisCornerTaylorseriers/taylorsplane.cpp")
add_cv9x_example(cv9x_harris_eigen
    "05_DerivativesEdgeDetection/05_HarrisEigen/HarrisEigen.cpp")
add_cv9x_example(cv9x_classify_image_regions
    "05_DerivativesEdgeDetection/06_OptionalClassifyImageRegions/EigenCLassifyImages.cpp")
add_cv9x_example(cv9x_harris_response
    "05_DerivativesEdgeDetection/07_HarrisResponseFinal/HarrisResponse.cpp")

# 06 - Feature detection
add_cv9x_example(cv9x_feature_detection
    "06_FeatureDetection/featuredetection.cpp")

# 07 - SIFT learning exercises. Some legacy paths/files use SHIFT in their
# spelling; the CMake target names expose the correct SIFT terminology.
add_cv9x_example(cv9x_sift_config_match
    "07_SIFT/01_SHIFTConfigMatch/shiftconfmatch.cpp")
add_cv9x_example(cv9x_sift_keypoint_orientation
    "07_SIFT/02_SHIFTKeypointLOcalization/shiftkeypointorientation.cpp")
add_cv9x_example(cv9x_sift_descriptor_math
    "07_SIFT/03_SHIFTKeypointDesciptor/shiftdescriptormath.cpp")

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "9x CV examples will be written to: ${CMAKE_BINARY_DIR}")
