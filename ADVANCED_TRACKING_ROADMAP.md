# Advanced Tracking & Repository Hardening Roadmap

The numbered classical course (Modules 00–19 / Episodes 00–23) is complete.

These are post-v1 extensions. They improve technical depth and repository quality without renumbering or destabilizing the completed course.

## Advanced tracking

| Extension | Scope | Planned PR |
|---|---|---:|
| A1 | SORT-style multi-object tracking: Hungarian assignment, IoU/Mahalanobis gating, track IDs, birth/death, MOTA/IDF1 | 28 |
| A2 | Background subtraction (MOG2), blob extraction, MeanShift and CamShift, detect-then-track pipeline | 29 |
| A3 | Extended Kalman filter with nonlinear/turning motion and NEES consistency checks | 30 |
| A4 | MOSSE correlation filter from first principles; KCF/CSRT comparison where OpenCV contrib is available | 31 |
| A5 | Real captured test video, reproducible download/attribution, success/precision plots, tracker comparison | 32 |

## Repository hardening

| Extension | Scope |
|---|---|
| H1 | License selected by repository owner |
| H2 | Canonical naming + compatibility redirects for legacy misspelled paths |
| H3 | Legacy-example inventory and test/data modernization |
| H4 | Canonical numbering/index mapping across folders, modules and episodes |
| H5 | clang-format, clang-tidy, CONTRIBUTING.md and pre-commit guidance |
| H6 | Broader platform/version CI, including additional Linux/compiler coverage and OpenCV-version experiments |

## Design rules

- Keep existing published lesson URLs working when practical.
- Prefer canonical new paths over destructive mass renames.
- Do not silently change the license.
- Treat real-video assets as licensed/attributed dependencies.
- Keep deterministic synthetic tests even after real-video tests are added.
- Separate algorithm correctness from benchmark/performance evidence.
- Keep optional OpenCV-contrib dependencies isolated from the baseline build.
