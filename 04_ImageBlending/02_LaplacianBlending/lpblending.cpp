#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void buildGaussianPyramid(const Mat& img, vector<Mat>& pyramid, int levels) {
    pyramid.clear();
    Mat current = img.clone();
    pyramid.push_back(current);
    for (int i = 1; i < levels; ++i) {
        pyrDown(current, current);
        pyramid.push_back(current);
    }
}

void buildLaplacianPyramid(const vector<Mat>& gaussianPyramid, vector<Mat>& laplacianPyramid) {
    laplacianPyramid.clear();
    for (size_t i = 0; i < gaussianPyramid.size() - 1; ++i) {
        Mat up;
        pyrUp(gaussianPyramid[i + 1], up, gaussianPyramid[i].size());
        Mat lap;
        subtract(gaussianPyramid[i], up, lap);
        laplacianPyramid.push_back(lap);
    }
    laplacianPyramid.push_back(gaussianPyramid.back());
}

Mat reconstructFromLaplacianPyramid(const vector<Mat>& laplacianPyramid) {
    Mat current = laplacianPyramid.back();
    for (int i = laplacianPyramid.size() - 2; i >= 0; --i) {
        pyrUp(current, current, laplacianPyramid[i].size());
        add(current, laplacianPyramid[i], current);
    }
    return current;
}

int main() {
    Mat img1 = imread("../01_Images/sunset.jpg");
    Mat img2 = imread("../01_Images/cityscape.jpg");

    if (img1.empty() || img2.empty()) {
        cout << "Failed to load images!" << endl;
        return -1;
    }

    // Resize to same dimensions
    Size sz(512, 512);
    resize(img1, img1, sz);
    resize(img2, img2, sz);

    // Split into left and right halves
    Mat left = img1(Rect(0, 0, sz.width / 2, sz.height));
    Mat right = img2(Rect(sz.width / 2, 0, sz.width / 2, sz.height));

    Mat combined(sz, CV_8UC3);
    left.copyTo(combined(Rect(0, 0, sz.width / 2, sz.height)));
    right.copyTo(combined(Rect(sz.width / 2, 0, sz.width / 2, sz.height)));

    imwrite("simple_combined.jpg", combined); // Simple hard blend

    // Laplacian Pyramid Blend
    vector<Mat> gp_left, gp_right;
    vector<Mat> lp_left, lp_right;
    int levels = 6;

    buildGaussianPyramid(left, gp_left, levels);
    buildGaussianPyramid(right, gp_right, levels);
    buildLaplacianPyramid(gp_left, lp_left);
    buildLaplacianPyramid(gp_right, lp_right);

    // Blend pyramids
    vector<Mat> blendedPyramid;
    for (int i = 0; i < levels; ++i) {
        Mat lap;
        hconcat(lp_left[i], lp_right[i], lap);  // Concatenate horizontally
        blendedPyramid.push_back(lap);
    }

    Mat final_blend = reconstructFromLaplacianPyramid(blendedPyramid);
    imwrite("laplacian_blend.jpg", final_blend);

    cout << "Blended image saved as 'laplacian_blend.jpg'" << endl;
    return 0;
}
