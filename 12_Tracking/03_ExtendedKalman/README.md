# Advanced Tracking A3 — Extended Kalman Filter, CTRV, NIS & NEES

This lesson extends the linear constant-velocity tracker into a nonlinear **Constant Turn Rate and Velocity (CTRV)** Extended Kalman Filter.

## Core state

```text
x = [px, py, speed, heading, yaw_rate]^T
measurement = [px, py]^T
```

For non-zero yaw rate:

```text
px' = px + v/w * (sin(heading + w*dt) - sin(heading))
py' = py + v/w * (-cos(heading + w*dt) + cos(heading))
heading' = heading + w*dt
```

Near zero yaw rate, the implementation uses the straight-line limit instead of dividing by a tiny value.

## Why EKF

CTRV is nonlinear, so the state transition is written as:

```text
x(k) = f(x(k-1)) + process noise
```

The EKF linearizes f around the current estimate with the Jacobian:

```text
F = df/dx
P- = F P F^T + Q
```

The lesson implements the analytic Jacobian and verifies it against a numerical finite-difference Jacobian.

## Measurement update

Position-only measurements use:

```text
innovation = z - H x-
S = H P- H^T + R
K = P- H^T S^-1
x+ = x- + K innovation
```

Covariance is updated with the Joseph form for better numerical behavior.

## NIS

Normalized Innovation Squared:

```text
NIS = innovation^T S^-1 innovation
```

NIS does not require ground truth. It tests whether measurement residuals are consistent with the predicted measurement uncertainty.

For a 2-D measurement, this lesson also reports the fraction of samples inside the approximate 95% chi-square interval.

## NEES

Normalized Estimation Error Squared:

```text
NEES = error^T P^-1 error
```

NEES requires ground truth and tests whether actual state error agrees with the covariance claimed by the filter.

For this 5-state EKF, the lesson reports the fraction of samples inside the approximate 95% chi-square interval.

## Accuracy is not consistency

A filter can have low position error and still be overconfident.

Likewise, a filter can be accurate but claim uncertainty that is far too large.

That is why this lesson reports both:

- position RMSE
- mean NIS
- mean NEES
- NIS 95% in-band fraction
- NEES 95% in-band fraction

## Constant-velocity baseline

A conventional Cartesian constant-velocity KF runs in parallel.

The synthetic truth follows sustained turning motion, so the comparison demonstrates when a nonlinear motion model becomes useful.

## Outputs

```text
ekf_consistency.csv
ekf_summary.txt
ekf_vs_cv_trajectory.png
```

The trajectory image uses:

```text
black  ground truth
green  CTRV EKF
orange constant-velocity KF
gray   noisy measurements
```

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_extended_kalman

./build/cv9x_extended_kalman --output-dir build/ekf
./build/cv9x_extended_kalman --self-test
```

## Self-test coverage

The deterministic test verifies:

1. analytic CTRV Jacobian agrees with finite differences
2. zero-turn limit matches straight-line motion
3. covariance remains symmetric and positive semidefinite
4. NIS/NEES remain finite and non-negative
5. EKF position RMSE stays bounded
6. CTRV EKF outperforms the constant-velocity baseline on turning motion
7. NIS remains statistically plausible
8. NEES remains statistically plausible
9. meaningful fractions of NIS/NEES samples fall inside their 95% intervals

## Connection to A1

A1 uses a linear constant-velocity Kalman filter inside each track.

A future integration can replace that estimator with this CTRV EKF while keeping the association layer unchanged:

```text
predict nonlinear track state
-> propagate covariance
-> Mahalanobis gate
-> associate detection
-> EKF correction
```

This also shows why covariance consistency matters for data association: bad covariance produces bad gates.

## Exercises

- change measurement noise and update R correctly
- change measurement noise without changing R and inspect NIS
- reduce process noise and inspect NEES
- increase yaw rate
- test near-zero yaw rate
- add acceleration and observe model mismatch
- use NIS for measurement gating
- integrate this EKF into the A1 multi-object tracker
- run Monte Carlo consistency experiments across many random seeds

## Next

A4: MOSSE correlation filter from first principles, followed by KCF/CSRT comparison when OpenCV contrib support is available.
