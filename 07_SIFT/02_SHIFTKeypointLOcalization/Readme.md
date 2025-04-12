# 🧠 SIFT — Stage 2: Keypoint Localization and Orientation Assignment

---

## 📌 Goal

After detecting scale-space extrema (candidate keypoints) in Stage 1, we now:
1. **Precisely localize** each keypoint in space and scale
2. **Eliminate unstable keypoints** (e.g., low contrast or edge responses)
3. Assign a **dominant orientation** to each keypoint so the descriptor can be **rotation-invariant**

---

## 🔎 Step-by-Step Breakdown

### 1️⃣ Keypoint Localization Using Taylor Expansion

Each keypoint is refined to **sub-pixel and sub-scale accuracy** by fitting a 3D quadratic function (Taylor expansion) to the DoG scale-space:

\[
D(x) = D + \frac{\partial D}{\partial x}^T x + \frac{1}{2} x^T \frac{\partial^2 D}{\partial x^2} x
\]

Where:
- \( D \) is the DoG value at the keypoint
- \( \partial D / \partial x \) and \( \partial^2 D / \partial x^2 \) are the gradient and Hessian at the keypoint
- \( x \) is the offset from the original keypoint

This improves accuracy and allows rejection of low-contrast points.

---

### 2️⃣ Eliminate Edge Responses

Some keypoints lie on edges and are poorly localized. These are rejected using the **ratio of principal curvatures** of the Hessian matrix.

If the ratio of the eigenvalues is too high (elongated structure), the keypoint is discarded.

A simple edge score is computed from the Hessian \( H \):

\[
\text{Edge Score} = \frac{(\text{trace}(H))^2}{\text{det}(H)}
\]

- Reject keypoints where the edge score exceeds a threshold (e.g., 10).

---

### 3️⃣ Orientation Assignment

For every refined keypoint:
- Compute the gradient magnitude and orientation in a region around the keypoint at the corresponding scale:

\[
m(x, y) = \sqrt{(L_x)^2 + (L_y)^2}
\]
\[
\theta(x, y) = \tan^{-1}(L_y / L_x)
\]

- Create an **orientation histogram** (usually 36 bins, 10° each) within a window around the keypoint.
- Assign the **peak orientation** as the keypoint’s canonical direction.
- If there are other strong peaks (>80% of the max), create **duplicate keypoints** with those orientations.

This allows the SIFT descriptor to be **rotation-invariant**.

---

## 🎯 Output of This Stage

- A set of refined, well-localized keypoints
- Each keypoint includes:
  - Subpixel location
  - Assigned scale (σ)
  - Dominant orientation (θ)

These keypoints are now ready for descriptor construction in Stage 3.

---

## 🧠 Why This Matters

| Stage             | Purpose                          |
|------------------|----------------------------------|
| Localization     | Improves precision & stability   |
| Edge rejection   | Removes poorly-defined features  |
| Orientation      | Enables rotation-invariant descriptors |

---

## ⏭️ Next: Descriptor Construction

We now build a **feature vector** for each keypoint using local gradient orientation histograms — the heart of SIFT matching!

