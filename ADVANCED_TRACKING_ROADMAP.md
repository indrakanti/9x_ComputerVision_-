# Advanced Tracking & Repository Hardening Roadmap

The numbered classical course (Modules 00–19 / Episodes 00–23) is complete.

These are post-v1 extensions. They improve technical depth and repository quality without renumbering or destabilizing the completed course.

## Advanced tracking

| Extension | Scope | Planned PR |
|---|---|---:|
| A1 | SORT-style multi-object tracking: Hungarian assignment, IoU/Mahalanobis gating, track IDs, birth/death, MOTA/IDF1 | **Implemented in PR 28** |
| A2 | Background subtraction (MOG2), blob extraction, MeanShift and CamShift, detect-then-track pipeline | **Implemented in PR 29** |
| A3 | Extended Kalman filter with nonlinear/turning motion and NIS/NEES consistency checks | **Implemented in PR 30** |
| A4 | MOSSE correlation filter from first principles; KCF/CSRT comparison where OpenCV contrib is available | **Implemented in PR 31** |
| A5 | Real captured test video, reproducible download/attribution, success/precision plots, tracker comparison | **Implemented in PR 32** |

## Repository hardening

| Extension | Scope | Status |
|---|---|---|
| H1 | License selected by repository owner | **Implemented in PR 33 — Apache-2.0** |
| H2 | Canonical naming + compatibility redirects for legacy misspelled paths | **Implemented in PR 34** |
| H3 | Legacy-example inventory and test/data modernization | **Baseline complete; deeper legacy modernization deferred maintenance** |
| H4 | Canonical numbering/index mapping across folders, modules and episodes | **Implemented in PR 36** |
| H5 | clang-format, clang-tidy, CONTRIBUTING.md and PR guidance | **Implemented in PRs 33 and 35** |
| H6 | Broader platform/version CI, including additional Linux/compiler coverage and OpenCV-version experiments | **Deferred maintenance; not a v1.0 blocker** |

## Design rules

- Keep existing published lesson URLs working when practical.
- Prefer canonical new paths over destructive mass renames.
- Do not silently change the license.
- Treat real-video assets as licensed/attributed dependencies.
- Keep deterministic synthetic tests even after real-video tests are added.
- Separate algorithm correctness from benchmark/performance evidence.
- Keep optional OpenCV-contrib dependencies isolated from the baseline build.

## Part 1 closure

Part 1 will be tagged as **v1.0.0 — Classical Computer Vision Complete** after PR #36 is merged and CI is green. New learned/deep Computer Vision work continues in [9x ML Computer Vision](https://github.com/indrakanti/9x_ML_ComputerVision).
