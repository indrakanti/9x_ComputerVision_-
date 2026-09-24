# Video: Feature Matching Done Correctly — Ratio Test + Homography + RANSAC

YouTube: TBD

Suggested title:

**Feature Matching Done Correctly — Ratio Test + Homography + RANSAC in C++**

Suggested thumbnail:

**MATCH != INLIER**

Target duration: **28–34 minutes**

## Learning objectives

Viewers should understand:

- descriptor match vs geometric inlier
- ratio-test limitations
- homography geometry
- reprojection error
- the RANSAC loop
- four-point minimal sampling
- inlier ratio and reprojection RMSE
- why visual projected geometry matters

## Chapters

~~~text
00:00 A good descriptor match can still be wrong
02:00 Ratio test recap
04:30 Descriptor match vs geometric inlier
06:30 Homography model
10:00 Reprojection error
12:30 Why four points?
14:30 RANSAC from first principles
19:00 Consensus scoring
21:30 Refining on all inliers
23:00 C++ implementation
27:00 Deterministic synthetic test
29:00 Ratio matches vs RANSAC inliers
31:00 Failure modes
33:00 Next: homography deep dive
~~~

## Script outline

### 00:00 — A descriptor can lie

Show 01_ratio_test_matches.png.

Point to visually plausible and implausible lines.

Narration:

> The ratio test tells us the best descriptor is distinctive. It does not tell us that all matches agree with one camera/image geometry.

### 02:00 — Ratio test

Write:

$$
d_1 < t d_2
$$

Explain nearest vs second-nearest distinctiveness.

### 04:30 — Match vs inlier

Use two labels:

~~~text
descriptor match = local appearance hypothesis
geometric inlier = match consistent with model
~~~

This is the main lesson.

### 06:30 — Homography

Write:

$$
x' \sim Hx
$$

Draw a planar poster/document under perspective change.

Show the 3×3 matrix.

Explain planar-scene/pure-rotation assumptions.

### 10:00 — Reprojection error

Take one source point.

Project it through H.

Draw the observed destination point.

Measure pixel distance.

Write:

$$
e_i = ||\hat p'_i-p'_i||_2
$$

### 12:30 — Four-point minimum

Explain:

- homography: 8 independent DoF
- correspondence: 2 equations
- minimum: 4 non-degenerate point pairs

### 14:30 — Manual RANSAC loop

Animate:

~~~text
sample 4
estimate H
score all matches
count inliers
repeat
keep best
~~~

Stress that one sampled outlier can ruin a hypothesis.

### 19:00 — Scoring

Explain:

1. maximize inlier count
2. tie-break with reprojection error

Then show threshold circles around projected points.

### 21:30 — Refinement

After consensus:

~~~text
best hypothesis
 -> collect all inliers
 -> re-estimate H
 -> recompute inliers
~~~

Explain why a 4-point model should not be the final estimate when dozens of inliers are available.

### 23:00 — C++ functions

Walk through:

- ratioMatch()
- sampleFourUnique()
- estimateFromFour()
- reprojectionError()
- classifyInliers()
- runHomographyRansac()

Show deterministic seed.

### 27:00 — Synthetic test

Show the known homography.

Generate:

~~~text
20 true correspondences
4 injected outliers
small deterministic noise
~~~

Run:

~~~bash
./build/cv9x_feature_matching_ransac --self-test
~~~

Explain why known synthetic geometry is stronger than visual-only testing.

### 29:00 — Real outputs

Show side-by-side:

~~~text
01_ratio_test_matches.png
02_ransac_inliers.png
03_projected_image_bounds.png
~~~

Print:

- ratio matches
- inliers
- inlier ratio
- RMSE

### 31:00 — Failure modes

Discuss:

- wrong model for 3-D scene
- too-loose threshold
- too-few iterations
- clustered/collinear matches
- repeated texture
- too few inliers

### 33:00 — Next

Preview homography deep dive and perspective warping.

## Commands demonstrated

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_feature_matching_ransac

./build/cv9x_feature_matching_ransac \
  --image1 path/to/image1.png \
  --image2 path/to/image2.png \
  --output-dir build/feature_matching_ransac \
  --ratio 0.75 \
  --ransac-iterations 1000 \
  --reprojection-threshold 3.0 \
  --seed 42

./build/cv9x_feature_matching_ransac --self-test
ctest --test-dir build --output-on-failure
~~~

## Short idea

**A Feature Match Is NOT an Inlier**

30–45 seconds:

1. show ratio-test matches
2. say "these only agree in descriptor space"
3. fit a homography
4. remove inconsistent matches
5. show RANSAC inliers
6. close with "appearance gives candidates; geometry gives evidence"

## YouTube description

In this lesson we complete the classical feature-matching pipeline.

We start with SIFT descriptor matches, apply the ratio test, then implement the RANSAC loop explicitly to estimate a homography, classify geometric inliers, refine the model, and measure reprojection error.

Source code + lesson notes:
https://github.com/indrakanti/9x_ComputerVision_-/tree/main/08_FeatureMatching/01_RatioRansac

Full course:
https://github.com/indrakanti/9x_ComputerVision_-

What you will learn:
- ratio test
- homography geometry
- reprojection error
- RANSAC
- geometric inliers/outliers
- deterministic robust-estimation tests

#ComputerVision #OpenCV #CPP #SIFT #RANSAC #Homography #Linux

## Recording checklist

- [ ] show wrong descriptor matches
- [ ] define match vs inlier
- [ ] derive homography point mapping
- [ ] explain reprojection error
- [ ] explain four-point minimum
- [ ] animate manual RANSAC
- [ ] explain consensus/refinement
- [ ] run synthetic test
- [ ] show ratio matches vs inliers
- [ ] show projected image bounds
- [ ] discuss wrong-model failure
- [ ] preview homography deep dive
