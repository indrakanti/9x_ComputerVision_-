# Naming and Compatibility Map

The repository contains a small number of historical path/name inconsistencies from the earliest lessons. Part 1 keeps those paths available when changing them could break old links, while exposing canonical names in course documentation and CMake targets.

## Root README

Canonical:

- `README.md`

Legacy:

- `Readme.md` — replaced by the canonical filename in PR #34.

## Fundamentals

Canonical spelling:

- `Fundamentals/`

Historical path retained for compatibility:

- `Fundametals/`

The historical directory contains the original optics/light notes and assets. The completed numbered course uses `01_ImageFormation/` as the course-facing image-formation module.

## SIFT

Canonical terminology is **SIFT**.

Canonical lesson directories:

- `07_SIFT/00_ScaleSpaceDoG/`
- `07_SIFT/01_LocalizationOrientation/`
- `07_SIFT/02_DescriptorMatching/`

Historical exercise directories retained for compatibility:

| Legacy path | Canonical terminology / role |
|---|---|
| `07_SIFT/01_SHIFTConfigMatch/` | SIFT configuration/matching exercise |
| `07_SIFT/02_SHIFTKeypointLOcalization/` | SIFT keypoint localization/orientation exercise |
| `07_SIFT/03_SHIFTKeypointDesciptor/` | SIFT descriptor mathematics exercise |

The top-level CMake target names intentionally use correct SIFT terminology:

- `cv9x_sift_config_match`
- `cv9x_sift_keypoint_orientation`
- `cv9x_sift_descriptor_math`

Do not introduce new paths or symbols using `SHIFT` for SIFT material.

## Compatibility policy

- New files and lessons use canonical spelling.
- Existing published paths remain available when practical.
- CMake target names and course-facing documentation use canonical terminology.
- Destructive renames are avoided when they would break external lesson/video links.
- Legacy names should be documented rather than copied into new APIs or examples.
