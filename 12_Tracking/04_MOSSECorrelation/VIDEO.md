# Video Guide — MOSSE from First Principles + KCF/CSRT Comparison

Target length: 30–38 minutes.

## 1. Why correlation-filter tracking? — 2 min
Start from frame-to-frame localization and contrast raw template matching, state-estimator-only tracking, and discriminative correlation filtering.

## 2. MOSSE least-squares objective — 5 min
Derive A = sum G*conj(F), B = sum F*conj(F), H = A/(B+epsilon). Explain why the DFT turns correlation/filter learning into element-wise frequency-domain operations.

## 3. Preprocessing — 4 min
Show grayscale -> log(I+1) -> zero mean -> unit variance -> Hann window. Explain illumination compression and boundary tapering.

## 4. Perturbed initialization — 3 min
Show deterministic small affine perturbations and explain why one exact crop is brittle.

## 5. Detection — 5 min
Walk through crop -> preprocess -> DFT -> H*F_t -> inverse DFT -> peak -> dx,dy update.

## 6. PSR confidence — 4 min
Explain why a maximum always exists and why Peak-to-Sidelobe Ratio is a better confidence signal. Show the 11x11 exclusion region.

## 7. Online adaptation — 3 min
Discuss the learning-rate tradeoff and confidence-gated updates.

## 8. Evaluation — 4 min
Explain IoU, center error, precision@20 px, success@0.5, success curve, and update latency.

## 9. KCF and CSRT — 4 min
If contrib tracking is present, run all three on the same sequence. Describe MOSSE as simplest/lightest, KCF as kernelized correlation filtering, and CSRT as reliability-aware but heavier.

## 10. Failure cases — 3 min
Cover scale, occlusion, deformation, distractors, drift, and DFT boundary effects.

## 11. Engineering close — 2 min
A practical tracker is a model, confidence estimate, adaptation policy, and evaluation contract—not only a bounding-box update.

## Demo

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_mosse_correlation
./build/cv9x_mosse_correlation --self-test
./build/cv9x_mosse_correlation --output-dir build/mosse --frames 140 --learning-rate 0.125 --psr-threshold 7.0
~~~

Open tracker_comparison_last_frame.png, success_plot.png, tracker_summary.csv, and tracker_frames.csv.

## Suggested thumbnail
MOSSE — TRACKING FROM FFTs, with a response-map peak and small 'vs KCF / vs CSRT' labels.

## Short idea
Why every correlation response has a peak—and why PSR tells you whether to trust it.
