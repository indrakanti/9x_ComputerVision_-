# Advanced Tracking A3 — Extended Kalman Filter, CTRV, NIS & NEES

Suggested title:

**Extended Kalman Filter in C++ — CTRV Motion, Jacobians, NIS & NEES**

Suggested thumbnail:

**TURNING MOTION → EKF**

Target duration: 42–50 minutes.

## Chapters

```text
00:00 Why linear Kalman filters fail on turns
04:00 CTRV state
08:00 nonlinear prediction
13:00 zero-yaw-rate limit
17:00 EKF Jacobian
23:00 finite-difference Jacobian test
27:00 covariance propagation
31:00 NIS
35:00 NEES
40:00 chi-square consistency
44:00 EKF vs constant-velocity KF
48:00 next: MOSSE/KCF/CSRT
```

## Demo

```bash
cmake --build build --target cv9x_extended_kalman
./build/cv9x_extended_kalman --output-dir build/ekf
./build/cv9x_extended_kalman --self-test
```

Open:

```text
ekf_vs_cv_trajectory.png
ekf_summary.txt
ekf_consistency.csv
```

Central point:

> A smooth estimate is not enough. You need both accuracy and credible uncertainty.
