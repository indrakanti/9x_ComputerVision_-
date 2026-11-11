# Advanced Tracking A4 — MOSSE from First Principles + KCF/CSRT Comparison

This lesson implements Minimum Output Sum of Squared Error (MOSSE) correlation-filter tracking directly from the Fourier-domain equations, then compares it with OpenCV KCF and CSRT when the optional tracking component is available.

## Learning objectives

- derive and implement MOSSE in the Fourier domain
- apply log/contrast normalization and a Hann window
- learn a desired Gaussian response
- localize translation from the inverse-DFT response peak
- compute Peak-to-Sidelobe Ratio (PSR) as tracker confidence
- gate online adaptation using confidence
- evaluate IoU, center precision, success rate, and latency
- compare MOSSE, KCF, and CSRT without making contrib a baseline dependency

## MOSSE objective

For training patch f_i, desired output g_i, and filter h, MOSSE minimizes the squared difference between the desired and actual correlation responses. In the frequency domain:

~~~text
A = sum G_i * conj(F_i)
B = sum F_i * conj(F_i)
H = A / (B + epsilon)
~~~

The implementation stores the complex numerator and denominator explicitly and performs complex division to form H.

## Detection

At each frame:

~~~text
extract patch at previous center
-> preprocess
-> DFT
-> multiply H * F_t
-> inverse DFT
-> response peak
-> dx,dy
-> new target center
~~~

The target size is fixed in this first-principles implementation, so this lesson focuses on translation rather than scale estimation.

## Preprocessing

~~~text
grayscale
-> float
-> log(I + 1)
-> subtract mean
-> divide by standard deviation
-> multiply by Hann window
~~~

The Hann window reduces artificial high-frequency energy caused by the finite patch boundary.

## Desired Gaussian response

The training target is a smooth Gaussian centered in the patch. Nearby translations therefore correspond to nearby response peaks instead of a brittle one-pixel impulse objective.

## Peak-to-Sidelobe Ratio

Every response map has a maximum, including a bad response. MOSSE therefore needs confidence:

~~~text
PSR = (peak - sidelobe_mean) / (sidelobe_stddev + epsilon)
~~~

An 11x11 neighborhood around the response peak is excluded before computing the sidelobe statistics. Low PSR prevents model adaptation, reducing the chance that one bad localization immediately corrupts the tracker.

## Online adaptation

After a confident localization:

~~~text
A_t = (1-eta) A_(t-1) + eta A_new
B_t = (1-eta) B_(t-1) + eta B_new
~~~

A small eta adapts slowly but resists drift. A large eta follows appearance change faster but can learn the wrong appearance after a bad localization.

## Synthetic benchmark

The executable creates a deterministic textured target moving over a noisy structured background. Every frame records:

- truth and estimated rectangles
- IoU
- center error
- MOSSE PSR
- update latency
- tracker success flag

PR #32 will add a real captured-video benchmark and reproducible attribution/download path. Keeping a deterministic synthetic benchmark remains useful for regression testing.

## KCF and CSRT

When OpenCV provides the optional tracking component, the same sequence is also evaluated with cv::TrackerKCF and cv::TrackerCSRT. If it is absent, MOSSE still builds and tests normally.

| Tracker | Core idea | Strength | Tradeoff |
|---|---|---|---|
| MOSSE | adaptive single-channel correlation filter | lightweight and transparent | fixed-scale, translation-focused |
| KCF | kernelized correlation filtering | strong speed/accuracy balance | more complex feature/model assumptions |
| CSRT | channel and spatial reliability | robust localization | heavier runtime |

These are conceptual differences, not a universal ranking.

## Metrics

- mean IoU
- precision@20 px: fraction of frames with center error <= 20 px
- success@0.5: fraction of frames with IoU >= 0.5
- full IoU-threshold success curve
- mean update latency

Latency is local engineering evidence, not a portable benchmark; compiler flags, CPU, OpenCV build options, power state, and threading all matter.

## Build

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_mosse_correlation
./build/cv9x_mosse_correlation --self-test
./build/cv9x_mosse_correlation --output-dir build/mosse --frames 140 --learning-rate 0.125 --psr-threshold 7.0
ctest --test-dir build --output-on-failure
~~~

## Outputs

~~~text
tracker_frames.csv
tracker_summary.csv
tracker_comparison_last_frame.png
success_plot.png
~~~

## Deterministic self-test

The self-test verifies the centered Gaussian target, one output per frame, strong mean IoU, center precision, IoU success rate, finite PSR, and confidence acceptance. When OpenCV tracking is available, KCF and CSRT also run smoke/accuracy checks.

## Failure modes

- scale change: fixed template size eventually reduces overlap
- rotation/deformation: small training perturbations are not a full deformation model
- occlusion: PSR detects ambiguity but does not re-detect a lost target
- distractors: a visually similar region can produce a competing response peak
- drift: wrong updates can teach the tracker the wrong appearance
- DFT boundaries: Hann tapering reduces but does not remove all boundary effects

## Engineering takeaways

- FFT-domain closed forms can make an apparently expensive optimization compact.
- Confidence belongs in the tracker interface.
- Online adaptation should be gated by confidence.
- Accuracy and runtime metrics answer different questions.
- Optional comparison implementations should not destabilize the baseline build.
- Deterministic synthetic tests remain valuable after real-video evaluation is added.

## Exercises

- sweep learning rate and PSR threshold
- add scale search at 0.95x / 1.0x / 1.05x
- add temporary occlusion and a distractor
- compute AUC of the IoU success curve
- compare Release and Debug latency

## Next

A5 / PR #32: real captured test video, reproducible download and attribution, precision/success plots, and comparison on non-synthetic data.
