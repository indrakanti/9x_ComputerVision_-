# 9x Computer Vision — Continuous Perception Evolution Plan

The long-term goal of this course is not to stop at still-image algorithms.

It should mature from:

~~~text
pixels
 -> local features
 -> geometry
 -> frame-to-frame motion
 -> feature tracks
 -> object detection
 -> object tracks
 -> 3-D scene understanding
 -> temporal memory
 -> multimodal perception
 -> world-aware / embodied perception
~~~

The course should teach both **how the ideas work** and **how to engineer them into live systems**.

## Guiding Principle

A mature perception system should not merely answer:

> What is in this image?

It should increasingly answer:

> What is here, where is it, how is it moving, what changed, what persists over time, what is uncertain, and what should the system attend to next?

That progression requires temporal reasoning, state, geometry, learned representations, uncertainty handling, and production engineering.

## Stage A — Still-Image Foundations

Goal: understand the building blocks.

Topics:

- pixels and sampling
- filtering
- gradients
- edges
- corners
- SIFT / ORB
- descriptor matching
- geometric verification
- homography

Outcome:

> Reliable correspondences and geometry between isolated images.

## Stage B — Camera Geometry and 3-D Foundations

Topics:

- coordinate systems
- intrinsic/extrinsic calibration
- lens distortion
- PnP
- epipolar geometry
- stereo
- triangulation
- depth
- pose estimation

Outcome:

> Move from 2-D image coordinates toward metric scene understanding.

## Stage C — Video and Feature Tracking

This is the first major transition from **images to continuous time**.

Topics:

- video capture and timestamps
- frame-rate and latency
- frame differencing
- optical flow
- Lucas-Kanade
- pyramidal Lucas-Kanade
- KLT feature tracks
- forward/backward consistency
- track birth/death
- feature re-detection
- homography tracking across frames
- motion compensation
- image stabilization

Outcome:

> Track image features robustly across a live video stream.

## Stage D — Object Tracking

Topics:

- template tracking
- correlation filters
- MOSSE / KCF / CSRT concepts
- Kalman filter
- state prediction
- measurement update
- bounding-box state
- association costs
- IoU association
- Hungarian assignment
- track lifecycle
- multi-object tracking
- appearance embeddings
- modern tracking-by-detection pipelines

Outcome:

> Maintain stable object identities over time.

## Stage E — Detection, Segmentation, and Pose

Topics:

- CNN fundamentals
- one-stage / two-stage detection concepts
- modern YOLO-style detection
- DETR-style set prediction
- semantic segmentation
- instance segmentation
- keypoints / pose
- real-time inference
- quantization
- ONNX / accelerator deployment

Outcome:

> Detect and understand objects instead of only tracking low-level features.

## Stage F — Video Understanding

Topics:

- temporal feature aggregation
- tubelets / tracklets
- video transformers
- temporal attention
- action recognition
- video object segmentation
- video instance segmentation
- long-range temporal consistency
- event detection
- streaming inference

Outcome:

> Understand not only objects but behavior and temporal context.

## Stage G — 3-D, Multi-Camera, and Spatial Perception

Topics:

- multi-camera calibration
- visual odometry
- SLAM fundamentals
- sparse/dense mapping
- depth networks
- point clouds
- occupancy
- bird's-eye-view representations
- multi-camera fusion
- camera + radar/lidar fusion concepts
- scene flow
- 4-D spatiotemporal perception

Outcome:

> Maintain a coherent spatial model of the environment over time.

## Stage H — Modern Representation Learning

Topics:

- vision transformers
- self-supervised visual learning
- contrastive image/text learning
- open-vocabulary detection
- promptable segmentation
- foundation visual encoders
- embedding spaces
- few-shot / zero-shot transfer
- retrieval and visual search

Outcome:

> Use reusable learned representations instead of task-specific features only.

## Stage I — Multimodal and Vision-Language Systems

Topics:

- image-text alignment
- visual question answering
- grounded language
- referring expressions
- visual reasoning
- multimodal retrieval
- vision-language models
- tool-using perception
- structured outputs from visual models

Outcome:

> Connect visual perception to semantic reasoning and language.

## Stage J — World Models and Embodied Perception

Topics:

- temporal memory
- object permanence
- scene state
- uncertainty
- prediction
- future-state estimation
- learned dynamics
- active perception
- sensor selection
- embodied agents
- visual navigation
- closed-loop perception/action
- continual adaptation
- online learning safeguards

Outcome:

> Move toward systems that perceive continuously, retain state, predict, and act.

The goal is not to claim biological equivalence to the human brain. The useful engineering direction is to build systems with progressively richer perception, memory, prediction, reasoning, and interaction.

## Production Track — Runs Through Every Stage

Production engineering is not a final appendix.

Every major module should eventually address:

- latency
- throughput
- jitter
- bounded memory
- zero-copy paths
- thread scheduling
- backpressure
- dropped frames
- timestamps
- deterministic replay
- observability
- failure handling
- hardware acceleration
- CPU/GPU/NPU partitioning
- model versioning
- calibration/version compatibility
- test evidence
- safety and uncertainty

## Live-Perception Milestones

### Milestone 1 — Live feature tracker

Input:

~~~text
camera / video file
~~~

Output:

~~~text
persistent feature IDs
feature trajectories
track age
tracking confidence
frame latency
~~~

### Milestone 2 — Planar object tracker

Use:

- feature detection
- descriptor matching
- RANSAC
- homography
- frame-to-frame update

Output:

- tracked planar target
- projected polygon
- confidence
- loss/reacquisition behavior

### Milestone 3 — Real-time object detector

Input:

- live video

Output:

- object boxes
- class
- confidence
- inference latency

### Milestone 4 — Multi-object tracker

Output:

~~~text
track ID
class
position
velocity
age
confidence
history
~~~

### Milestone 5 — Temporal perception pipeline

Combine:

- detector
- tracker
- optical flow
- pose / segmentation
- temporal smoothing
- metrics
- replay tests

### Milestone 6 — 3-D / world-aware perception

Combine:

- multi-camera geometry
- depth
- tracks
- ego motion
- occupancy / BEV
- temporal memory

### Milestone 7 — Multimodal perception agent

Combine:

- live video
- visual embeddings
- object/track memory
- language grounding
- temporal reasoning
- tools / actions

## Periodic Modernization Policy

The course should remain a living engineering curriculum.

### Every quarter

Review:

- important new algorithms
- new model families
- major deployment runtimes
- accelerator support
- meaningful OpenCV / ONNX / inference changes
- tracking / detection / segmentation advances
- benchmark methodology

Add or revise lessons only when the technology is mature enough to teach with clear engineering value.

### Every major course release

Audit:

- terminology
- deprecated APIs
- build/toolchain versions
- benchmark assumptions
- model export paths
- reproducibility
- datasets/assets
- hardware notes
- production failure modes

### Keep fundamentals stable

Do not remove first-principles material just because a newer model exists.

Convolution, geometry, optical flow, estimation, tracking, timing, memory, and uncertainty remain useful for understanding newer systems.

## Future Capstone Direction

The eventual capstone should become a **live perception system**, not a single-image demo.

Target architecture:

~~~text
camera / recorded video
        |
        v
capture + timestamps
        |
        v
preprocessing
        |
        +----------------------+
        |                      |
        v                      v
feature / flow            detector / segmenter
        |                      |
        +----------+-----------+
                   |
                   v
             temporal fusion
                   |
                   v
            object/feature tracks
                   |
                   v
         geometry / depth / motion
                   |
                   v
             world-state memory
                   |
                   v
        metrics + visualization + logs
~~~

Later versions can extend this with multimodal reasoning and active perception.

## Course Success Criterion

A learner who completes the mature course should be able to:

- reason from raw pixels to modern learned representations
- build algorithms from first principles
- debug visual failure modes
- process live video
- track features and objects over time
- estimate geometry and motion
- understand 3-D perception
- deploy optimized C++ pipelines
- integrate modern foundation vision components
- evaluate uncertainty and system limitations
- evolve a perception stack as technology changes
