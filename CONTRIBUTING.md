# Contributing to 9x Computer Vision

Thank you for helping improve the 9x Computer Vision course.

This repository is the completed **Part 1: Classical Computer Vision** learning track. Contributions should preserve the course's first-principles, reproducible, engineering-oriented approach.

## What belongs here

Good contributions include:

- correcting technical explanations or mathematical derivations
- fixing bugs in existing examples
- improving tests and deterministic self-tests
- improving portability, build reliability, or diagnostics
- adding useful visualizations for existing lessons
- improving documentation, examples, or exercises
- modernizing legacy examples without breaking published course links

New deep-learning / learned-perception material belongs in the separate **9x ML Computer Vision** repository rather than in this classical-CV repository.

## Development baseline

The canonical environment is:

- Linux
- C++17
- OpenCV 4
- CMake
- CTest

Configure and build from the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

See [BUILDING.md](BUILDING.md) for details.

## Course design rules

Changes should preserve these principles:

1. Explain the mathematics or physical intuition before hiding it behind an API.
2. Prefer deterministic, reproducible examples.
3. Keep algorithm correctness separate from benchmark/performance evidence.
4. Avoid requiring optional OpenCV contrib modules for the baseline build.
5. Keep existing published lesson URLs working when practical.
6. Prefer canonical new names over destructive renames of legacy paths.
7. Do not commit generated build products, large downloaded datasets, or third-party assets without a clear redistribution right.

## Pull requests

Keep pull requests focused. A good PR should:

- describe the problem and intended learning outcome
- identify affected lessons or modules
- include or update tests when behavior changes
- build through the top-level CMake project
- pass CTest
- document new runtime dependencies or external assets
- avoid unrelated formatting churn

For lesson structure and definition of done, see [LESSON_GUIDE.md](LESSON_GUIDE.md).

## Third-party data and media

The Apache-2.0 license in this repository applies to repository-owned source code and documentation unless a file states otherwise.

It does **not** automatically grant rights to third-party:

- images
- videos
- datasets
- model files
- annotations
- papers or figures
- fonts
- logos or trademarks

External assets must retain their own attribution, license, usage restrictions, and download source. Prefer scripts or documented download instructions over redistributing third-party data in the repository.

If licensing or redistribution terms are unclear, do not commit the asset.

## Contribution licensing

Unless explicitly stated otherwise, contributions intentionally submitted for inclusion in this repository are provided under the repository's Apache License 2.0, consistent with Section 5 of that license.

By opening a pull request, you represent that you have the right to submit the contribution under those terms.
