# Module 08 — ORB vs SIFT: Binary Descriptors, Hamming Distance & Engineering Tradeoffs

SIFT and ORB can both produce repeatable local features, but they make very different engineering choices.

This lesson does **not** declare one universally better.

Instead, it asks:

> What are we paying for—in computation, memory, descriptor richness, and matching cost—and what do we get in return?

The executable runs SIFT and ORB on the same image pair, uses the correct distance metric for each, applies the same nearest/second-nearest ratio concept, reports descriptor memory, measures runtime, and saves both match visualizations.

## Learning objectives

By the end of the lesson you should be able to:

- summarize the SIFT detector/descriptor pipeline
- explain ORB's relationship to FAST and BRIEF
- distinguish floating-point and binary descriptors
- explain why SIFT uses L2 distance
- explain why ORB uses Hamming distance
- compute Hamming distance from XOR + bit count
- compare descriptor memory per feature
- interpret runtime measurements without treating one machine as universal truth
- understand where SIFT's scale-space design differs from ORB's image-pyramid approach
- choose metrics based on system constraints rather than popularity

## 1. SIFT recap

The preceding three lessons built SIFT as:

```text
Gaussian scale space
 -> DoG extrema
 -> subpixel/subscale localization
 -> edge rejection
 -> orientation
 -> 128-D float descriptor
 -> L2 matching
```

The descriptor in this repository is conceptually:

```text
128 float values
```

At 4 bytes per float:

$$
128 \times 4 = 512\text{ bytes}
$$

per descriptor.

## 2. What ORB combines

ORB stands for **Oriented FAST and Rotated BRIEF**.

At a high level:

```text
image pyramid
 -> FAST-like corner detection
 -> keypoint scoring / selection
 -> orientation estimation
 -> rotated BRIEF binary descriptor
 -> Hamming matching
```

The OpenCV configuration in this lesson uses:

- 8 pyramid levels
- scale factor 1.2
- FAST threshold 20
- Harris score for ranking
- 256-bit ORB descriptor

## 3. ORB descriptor representation

OpenCV ORB produces:

```text
32 bytes
```

per descriptor.

That is:

$$
32 \times 8 = 256\text{ bits}
$$

Compared with the 512-byte SIFT float descriptor:

$$
\frac{512}{32}=16
$$

So the descriptor payload per feature is 16× smaller for this standard ORB-vs-SIFT representation.

That statement is about **descriptor storage only**. Keypoint metadata, allocator overhead, indexes, and other pipeline data are additional.

## 4. BRIEF intuition

A binary descriptor can be built from many local intensity comparisons.

Conceptually one bit asks:

$$
I(p_a) < I(p_b)?
$$

If true, write one bit value; otherwise write the other.

Many such tests create a bit string.

ORB rotates its BRIEF sampling pattern according to keypoint orientation, which improves rotational robustness compared with an un-oriented binary test pattern.

## 5. L2 vs Hamming

SIFT descriptors are floating-point vectors.

The natural metric is Euclidean distance:

$$
d_{L2}(a,b)
=
\sqrt{
\sum_i(a_i-b_i)^2
}
$$

ORB descriptors are bit strings.

For two bytes/vectors:

```text
XOR corresponding bits
count the 1s
sum across the descriptor
```

Mathematically:

$$
d_H(a,b)=\operatorname{popcount}(a\oplus b)
$$

The lesson implements both metrics manually and validates them against OpenCV's `NORM_L2` and `NORM_HAMMING`.

## 6. Same ratio-test concept, different distance metric

For each query descriptor:

1. find best distance $d_1$
2. find second-best distance $d_2$
3. accept only if

$$
d_1 < t d_2
$$

This lesson applies the same default:

```text
t = 0.75
```

to both methods for comparison.

That does **not** mean 0.75 is universally optimal for every descriptor/data distribution. It keeps the experiment controlled.

## 7. Scale handling

### SIFT

SIFT creates a Gaussian scale space and explicitly searches extrema across scale.

### ORB

ORB detects features across a discrete image pyramid.

Both are multi-scale ideas, but their scale-space construction, detector behavior, descriptor design, and invariance properties are not identical.

## 8. Orientation

### SIFT

Uses a local gradient orientation histogram.

### ORB

Uses an oriented keypoint estimate and rotates the BRIEF sampling pattern.

Both aim to reduce sensitivity to image rotation, but they arrive there differently.

## 9. Memory comparison

Typical standard OpenCV descriptor payloads:

| Method | Descriptor | Storage type | Bytes / feature |
|---|---:|---|---:|
| SIFT | 128 scalars | 32-bit float | 512 |
| ORB | 256 bits | 8-bit bytes | 32 |

For 1000 features:

```text
SIFT descriptor payload: 512,000 bytes
ORB descriptor payload:   32,000 bytes
```

approximately:

```text
SIFT: 500 KiB
ORB:   31.25 KiB
```

before keypoint metadata and container/index overhead.

For embedded and high-rate systems, that memory/bandwidth difference can matter.

## 10. Runtime comparison

The executable measures:

- detect + describe time on image 1
- detect + describe time on image 2
- manual brute-force matching time

for both SIFT and ORB.

It repeats each measurement multiple times and prints the average.

Important:

> These timings describe the machine, image pair, OpenCV build, parameters, cache state, and run conditions you actually used.

They are **not** universal performance claims.

CI intentionally has no timing pass/fail assertions.

## 11. What to compare besides speed

Feature pipelines should be evaluated on more than milliseconds.

Consider:

- repeatability
- number of useful matches
- scale change
- rotation change
- viewpoint change
- illumination change
- blur/noise
- descriptor memory
- matching memory bandwidth
- CPU vectorization
- accelerator support
- downstream geometric inlier rate

This lesson reports counts and latency, but a production decision needs task-specific data.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_orb_vs_sift
```

## Run

```bash
./build/cv9x_orb_vs_sift \
  --image1 path/to/image1.png \
  --image2 path/to/image2.png \
  --output-dir build/orb_vs_sift \
  --max-features 500 \
  --ratio 0.75 \
  --iterations 20
```

Outputs:

```text
sift_matches.png
orb_matches.png
```

The console reports descriptor types, dimensions, bytes, feature counts, accepted matches, and average timings.

## Deterministic self-test

```bash
./build/cv9x_orb_vs_sift --self-test
ctest --test-dir build --output-on-failure
```

The self-test verifies:

- manual Hamming distance matches OpenCV
- manual L2 distance matches OpenCV
- binary ratio matching finds a distinctive neighbor
- SIFT produces descriptors on deterministic synthetic texture
- SIFT descriptors are `CV_32F`
- SIFT descriptors have 128 scalar columns
- SIFT standard descriptor payload is 512 bytes
- ORB produces descriptors on the same deterministic texture
- ORB descriptors are `CV_8U`
- ORB descriptors contain 32 bytes / 256 bits
- ORB descriptor payload per feature is smaller than SIFT's

No self-test asserts that one method is faster or produces more matches.

## Common failure cases

### Matching ORB with L2

ORB is a binary descriptor. Use Hamming distance.

### Matching SIFT with Hamming

SIFT is a floating-point histogram descriptor. Use L2-style distance.

### Comparing only feature count

More keypoints do not automatically mean more useful geometric correspondences.

### Treating benchmark timing as a universal ranking

Measurements are platform and workload dependent.

### Calling ORB simply "FAST + BRIEF"

ORB adds orientation, a multi-scale pyramid, keypoint ranking, and a learned/selected binary test design around BRIEF-style comparisons.

### Ignoring descriptor memory

At high feature counts and frame rates, descriptor storage and matching bandwidth can be as important as detector runtime.

## Engineering decision framework

Instead of asking "Which one is best?", ask:

```text
What scale/rotation/viewpoint changes do I expect?
What feature repeatability do I need?
What latency budget do I have?
What memory/bandwidth budget do I have?
What CPU/accelerator is available?
How many features per frame?
What geometric inlier rate do I actually get?
```

For an embedded system, ORB's compact binary representation can be attractive.

For applications requiring stronger scale-space distinctiveness, SIFT's richer floating-point representation may be valuable.

The correct answer is application-dependent and should be measured.

## Exercises

1. Rotate image 2 by 15°, 45°, and 90° and compare match counts.
2. Downscale image 2 and compare repeatability.
3. Add Gaussian blur/noise and compare accepted matches.
4. Plot runtime vs feature budget for 100, 250, 500, 1000 features.
5. Plot descriptor memory vs feature budget.
6. Add geometric verification and compare inlier ratios instead of raw match counts.
7. Compare ORB `HARRIS_SCORE` vs `FAST_SCORE`.

## Next lesson

Episode 14 builds the complete **feature matching + RANSAC geometric verification** pipeline.

It will distinguish:

```text
descriptor match
        vs
geometrically consistent inlier
```

using a homography and RANSAC.

See [VIDEO.md](VIDEO.md).
