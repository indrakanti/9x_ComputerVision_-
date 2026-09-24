# Module 09 — Feature Matching Done Correctly: Ratio Test + Homography + RANSAC

A descriptor match is only a **similarity hypothesis**.

It is not yet proof that two image points correspond to the same physical scene point.

This lesson completes the classical feature-matching pipeline:

~~~text
SIFT keypoints + descriptors
        |
        v
nearest / second-nearest matching
        |
        v
Lowe ratio test
        |
        v
candidate correspondences
        |
        v
homography RANSAC
        |
        v
geometric inliers + outliers
~~~

The main engineering distinction is:

> **Descriptor match != geometric inlier.**

## Learning objectives

By the end of the lesson you should be able to:

- explain what the ratio test does and does not prove
- explain when a homography is an appropriate geometry model
- define homogeneous point mapping
- define reprojection error
- explain the RANSAC sample / score / consensus loop
- understand why four correspondences are the minimum for a general homography
- distinguish model estimation from inlier classification
- interpret inlier count, inlier ratio, and reprojection RMSE
- understand insufficient and degenerate match cases
- explain why RANSAC depends on threshold, iteration budget, and data

## Prerequisites

Complete:

- SIFT Parts 1–3
- ORB vs SIFT
- ratio-test matching

This lesson uses SIFT for the real-image demo so the geometry stage remains the focus.

## 1. Descriptor similarity is local

A SIFT descriptor summarizes local gradient structure.

Two descriptors can be similar because of:

- a true correspondence
- repeated texture
- repeated windows/bricks/tiles
- accidental local similarity
- motion blur
- descriptor noise
- viewpoint or illumination effects

The ratio test removes some ambiguous matches, but it does not use image-wide geometry.

## 2. Ratio test recap

For each descriptor in image 1:

- nearest distance = \(d_1\)
- second-nearest distance = \(d_2\)

Accept when:

$$
d_1 < t d_2
$$

with a common starting point such as:

~~~text
t = 0.75
~~~

This says:

> The best descriptor match should be meaningfully more distinctive than the next alternative.

It does **not** say that the match fits a consistent camera/image transformation.

## 3. Homography model

A homography maps points on one projective plane to another:

$$
\mathbf{x}' \sim H\mathbf{x}
$$

where:

$$
H =
\begin{bmatrix}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
h_{31} & h_{32} & h_{33}
\end{bmatrix}
$$

and image points use homogeneous coordinates:

$$
\mathbf{x} =
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
$$

After multiplication, divide by the third homogeneous coordinate.

## 4. When a homography is appropriate

A single homography is a useful model when:

- the observed scene is approximately planar
- or the camera undergoes pure rotation around its optical center
- or the task is explicitly a perspective mapping between planes

Examples:

- document scanning
- posters/signs
- planar markers
- image stitching under suitable camera motion

A general 3-D scene under camera translation may require fundamental/essential-matrix geometry instead.

## 5. Reprojection error

Given a candidate homography \(H\) and source point \(p_i\):

1. project the source point using \(H\)
2. compare the projected location with the observed destination point

The lesson uses Euclidean pixel error:

$$
e_i =
\left\|
\hat{p}'_i - p'_i
\right\|_2
$$

A correspondence is an inlier when:

$$
e_i \le \tau
$$

where \(\tau\) is the reprojection threshold.

CLI default:

~~~text
3.0 pixels
~~~

## 6. Why RANSAC is needed

If we estimate a homography from all descriptor matches at once, outliers can corrupt the model.

RANSAC repeatedly asks:

> What if this small subset is correct?

For a homography:

1. randomly sample four correspondences
2. estimate a homography
3. project every correspondence
4. count points with reprojection error below threshold
5. keep the strongest consensus
6. repeat
7. refine the model using all inliers

The implementation performs this loop explicitly.

## 7. Why four points?

A general homography has eight independent degrees of freedom because its overall scale is arbitrary.

Each point correspondence supplies two equations.

Therefore the minimal non-degenerate sample is four correspondences.

Four collinear or otherwise degenerate samples may still fail to produce a useful model.

## 8. RANSAC scoring

This lesson prefers a model when it has:

1. more inliers
2. for equal inlier count, smaller total squared reprojection error

After the best hypothesis is selected, all current inliers are used to refine the homography.

Then the inlier mask is recomputed.

## 9. Deterministic randomness

The CLI exposes:

~~~text
--seed
~~~

A fixed seed makes the teaching demo and self-test reproducible.

Production systems may choose a different policy, but reproducibility is valuable for debugging and CI.

## 10. Inlier ratio

The lesson reports:

$$
\text{inlier ratio}
=
\frac{\text{RANSAC inliers}}
     {\text{ratio-test matches}}
$$

This is more informative than raw match count alone.

Example:

~~~text
100 ratio matches
30 geometric inliers
70 geometric outliers
inlier ratio = 0.30
~~~

A high raw match count can still represent a poor correspondence set.

## 11. Inlier reprojection RMSE

For accepted inliers:

$$
RMSE =
\sqrt{
\frac{1}{N}
\sum_i e_i^2
}
$$

This gives a pixel-space measure of how well the final homography explains the inlier correspondences.

Low RMSE is useful, but it is not a complete quality metric by itself.

A model can still be poor if:

- inliers occupy a tiny region
- points are nearly collinear
- the wrong physical plane dominates
- there are too few inliers
- the scene is not homographic

## 12. What the three output images mean

### 01_ratio_test_matches.png

Descriptor-level candidates after the ratio test.

These are **not yet trusted correspondences**.

### 02_ransac_inliers.png

Only matches that agree with the final homography.

These are geometric inliers under the selected model and threshold.

### 03_projected_image_bounds.png

Projects the four corners of image 1 into image 2 using the estimated homography.

This makes a wrong model visually obvious even when numeric counts look plausible.

## 13. Implementation boundary

The lesson manually implements:

- L2 descriptor distance
- nearest + second-nearest search
- ratio test
- RANSAC sampling loop
- reprojection scoring
- consensus selection
- inlier-mask construction
- final statistics

OpenCV is used for:

- SIFT feature extraction
- four-point perspective solve
- least-squares homography refinement on selected inliers

That keeps the lesson focused on the geometry/robust-estimation logic while reusing trusted linear algebra primitives.

## Build

The lesson adds OpenCV calib3d to the repository build because homography refinement uses that module.

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_feature_matching_ransac
~~~

## Run

Use two views of the same approximately planar object/scene:

~~~bash
./build/cv9x_feature_matching_ransac \
  --image1 path/to/image1.png \
  --image2 path/to/image2.png \
  --output-dir build/feature_matching_ransac \
  --max-features 800 \
  --ratio 0.75 \
  --ransac-iterations 1000 \
  --reprojection-threshold 3.0 \
  --seed 42
~~~

## Outputs

~~~text
01_ratio_test_matches.png
02_ransac_inliers.png
03_projected_image_bounds.png
~~~

The console reports:

- SIFT keypoint counts
- ratio-test matches
- RANSAC inliers
- RANSAC outliers
- inlier ratio
- inlier reprojection RMSE
- RANSAC iteration budget
- reprojection threshold
- random seed

## Deterministic self-test

~~~bash
./build/cv9x_feature_matching_ransac --self-test
ctest --test-dir build --output-on-failure
~~~

The self-test creates:

- a known projective homography
- 20 synthetic inlier correspondences
- small deterministic subpixel noise
- 4 large geometric outliers

It verifies that manual RANSAC:

- recovers a homography
- keeps almost all true inliers
- rejects the injected outliers
- achieves low inlier reprojection RMSE
- projects unseen probe points close to the known ground-truth homography
- rejects the insufficient-correspondence case
- preserves expected ratio-test behavior on synthetic descriptors

## Common failure cases

### Treating every ratio-test match as valid geometry

Descriptor distinctiveness is not geometric consistency.

### Using homography for arbitrary 3-D scenes

A single homography may be the wrong model when parallax is present.

### Choosing a reprojection threshold without units

This lesson uses pixels. Resolution and localization accuracy matter.

### Using too few RANSAC iterations

A low inlier ratio can require many samples before a good four-point subset is drawn.

### Increasing the threshold until everything becomes an inlier

That defeats robust estimation.

### Trusting only the inlier count

Inspect spatial distribution, reprojection error, and projected geometry.

## Engineering notes

A production matching interface should expose at least:

~~~text
raw descriptor matches
ratio-test matches
geometric inlier mask
model type
model parameters
reprojection threshold
inlier count
inlier ratio
reprojection error
estimation status / failure reason
~~~

That evidence makes failures diagnosable.

For safety- or reliability-sensitive perception, silently returning a transformation from weak geometry is often worse than returning an explicit failure.

## Exercises

1. Sweep the ratio threshold and plot ratio-match count vs geometric inlier count.
2. Sweep reprojection threshold and plot inlier ratio vs RMSE.
3. Reduce/increase RANSAC iterations and measure stability.
4. Add symmetric/mutual descriptor matching before RANSAC.
5. Add a minimum spatial-coverage check for accepted inliers.
6. Compare manual RANSAC with OpenCV findHomography using RANSAC.
7. Replace the homography with fundamental-matrix RANSAC for a non-planar stereo pair.

## Next lesson

Episode 15 dives deeper into **homography and perspective transforms**:

- projective coordinates
- degrees of freedom
- point/line mappings
- perspective warping
- document rectification
- failure/degeneracy cases

See VIDEO.md.
