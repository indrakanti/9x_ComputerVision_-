# 9x Computer Vision Lesson Guide

New and modernized lessons should follow the same learning contract so GitHub and YouTube remain aligned.

## Directory naming

Use:

```text
NN_TopicName/
```

Guidelines:

- use the course/module number as a two-digit prefix
- avoid spaces in new directory names
- use the correct algorithm name in paths and targets (`SIFT`, not `SHIFT`)
- do not rename a legacy directory only for cosmetics if it would break linked assets; migrate it deliberately in a dedicated change

## Minimum lesson contents

A completed lesson should contain:

```text
NN_TopicName/
├── README.md
├── src/
│   └── example.cpp
├── data/                 # only small redistributable inputs
└── expected/             # optional reference outputs
```

The top-level `CMakeLists.txt` remains the canonical build entry point.

## README structure

Each lesson README should answer these sections in order:

1. **Goal** — what the learner will understand or build.
2. **Prerequisites** — concepts and earlier lessons required.
3. **Theory** — equations, assumptions, and intuition.
4. **Algorithm** — ordered processing steps or pseudocode.
5. **C++ implementation** — important code decisions, not only an API listing.
6. **Build and run** — exact commands from repository root.
7. **Expected result** — image/output and what should be observed.
8. **Failure cases** — where the method stops working or becomes unstable.
9. **Engineering notes** — memory, latency, precision, determinism, or deployment implications when relevant.
10. **Exercises** — at least two experiments that force parameter or implementation changes.
11. **YouTube lesson** — published link, or `TBD` before release.

## C++ conventions

- C++17 baseline.
- Prefer explicit command-line inputs over hard-coded absolute paths.
- Return a non-zero status on invalid input or failed image/model loading.
- Avoid hidden global state.
- Keep algorithmic code separate from visualization when a lesson becomes non-trivial.
- Name measurements with units (`latency_ms`, `distance_px`, `sigma_px`).
- Explain intentional copies of `cv::Mat`; avoid accidental deep copies.
- Use `cv::imwrite` for reproducible outputs when GUI display is not essential.

## Build target naming

Use the prefix:

```text
cv9x_<topic>
```

Examples:

```text
cv9x_sobel
cv9x_camera_calibration
cv9x_stereo_depth
cv9x_yolo_onnx
```

The target name should remain stable even if the lesson's internal directory layout changes.

## YouTube mapping

A video should point to one primary lesson directory. The lesson should link back to the video through `VIDEO_SERIES.md` and its own README.

Recommended video flow:

```text
problem / visual hook
        ↓
intuition
        ↓
math
        ↓
implementation
        ↓
visual result
        ↓
engineering consequence
        ↓
exercise / next lesson
```

This keeps the series differentiated from API-only OpenCV tutorials.

## Definition of done

A lesson is considered complete when:

- it builds in CI
- commands are reproducible from a clean checkout
- inputs and expected outputs are documented
- mathematical notation is defined
- at least one failure/edge case is discussed
- the lesson has exercises
- the GitHub/YouTube mapping is present

Runtime regression tests will be added incrementally as the legacy lessons are standardized.
