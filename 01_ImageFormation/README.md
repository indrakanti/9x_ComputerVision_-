# Module 01 — Image Formation: From Light to Digital Pixels

Computer vision starts before the first OpenCV call.

A real imaging pipeline converts physical light into a finite digital array:

~~~text
scene radiance
   |
lens / aperture
   |
sensor exposure
   |
spatial sampling
   |
analog response
   |
quantization
   |
color processing
   |
digital image matrix
~~~

This lesson connects the existing optics notes in the legacy Fundametals directory to the digital representation used by every later course module.

## Learning objectives

By the end of this lesson you should be able to:

- explain how a camera maps light onto a finite sensor grid
- distinguish a continuous scene from discrete pixel samples
- explain focal length, aperture, exposure, and sensor response conceptually
- explain spatial sampling and Nyquist intuition
- recognize aliasing
- explain why low-pass filtering should precede downsampling
- explain quantization and bit depth
- distinguish bit depth from effective dynamic range
- explain grayscale conversion
- distinguish RGB semantics from OpenCV BGR storage order
- explain HSV and YUV as alternative color coordinate systems
- understand basic cv::Mat size, channels, element size, and ROI behavior

## 1. Light is the input signal

A camera measures energy arriving at a sensor over finite space and finite time.

A pixel value is not simply an object's "true color." It depends on:

- illumination
- material reflectance
- viewing geometry
- optics
- exposure
- sensor spectral response
- gain
- quantization
- image-signal processing

This matters whenever a later algorithm assumes brightness constancy, stable color, or calibrated intensity.

## 2. Pinhole and lens intuition

The legacy fundamentals material introduces the pinhole camera and lenses.

An ideal pinhole maps a 3-D scene point through one optical center onto an image plane. Later geometry lessons formalize this as:

$$
u=f_x\frac{X}{Z}+c_x
$$

$$
v=f_y\frac{Y}{Z}+c_y
$$

Real cameras use lenses because an infinitesimal pinhole would admit too little light.

Focal length affects field of view and projection scale. Aperture affects light collection and depth of field. Exposure time controls how long the sensor integrates light.

Long exposure can improve signal but increase motion blur. Short exposure reduces blur but can require more gain and therefore more noise.

## 3. Sensor grid and spatial sampling

A continuous image irradiance field can be written conceptually as:

$$
I(x,y)
$$

A digital sensor measures it only at finite locations:

$$
I[m,n]
$$

That conversion is spatial sampling.

Resolution tells us how many samples exist. It does not guarantee equivalent real-world detail if optics, focus, noise, or motion limit the signal first.

## 4. Manual sampling

The executable implements a deliberately simple integer-lattice sampler.

For factor s:

$$
I_d[m,n]=I[sm,sn]
$$

Output dimensions become approximately:

$$
W_d=W/s
$$

$$
H_d=H/s
$$

This is intentionally primitive so aliasing is visible.

## 5. Aliasing

If the source contains spatial frequencies above what the new sampling grid can represent, multiple source patterns can map to the same digital samples.

That ambiguity is aliasing.

The self-test creates alternating black/white vertical stripes. Keeping every second sample at one phase makes the result uniformly black even though the original contains strong high-frequency structure.

The sampled image therefore contains a false pattern.

## 6. Anti-alias filtering

Before reducing sampling density, suppress frequencies the lower-resolution grid cannot represent.

The demonstration uses:

~~~text
Gaussian low-pass filter
then
lower-rate sampling
~~~

This same signal-processing principle reappears in Gaussian image pyramids.

Downsampling is not merely deleting pixels.

## 7. Quantization

After physical sensing, an analog response must be represented using finite digital codes.

For b bits:

$$
L=2^b
$$

representable levels exist.

Examples:

~~~text
1 bit  -> 2 levels
2 bits -> 4 levels
4 bits -> 16 levels
8 bits -> 256 levels
12 bits -> 4096 levels
~~~

The executable quantizes an 8-bit grayscale image into fewer levels so banding becomes visible.

## 8. Quantization equation

For input x in [0,255]:

$$
q=
round
\left(
\frac{x}{255}(L-1)
\right)
$$

and the displayed reconstruction is:

$$
\hat x=
round
\left(
255\frac{q}{L-1}
\right)
$$

The self-test verifies exact 2-bit reconstruction levels 0, 85, 170, and 255.

## 9. Bit depth is not dynamic range

Bit depth tells us how many digital codes can be stored.

Dynamic range describes the useful ratio between the largest non-saturated signal and the smallest distinguishable signal above noise.

A 12-bit container does not automatically mean 12 bits of useful sensor information.

Noise, full-well capacity, gain, ADC performance, clipping, and image processing all affect the effective signal.

## 10. Grayscale

A common luminance-style approximation is:

$$
Y\approx0.299R+0.587G+0.114B
$$

The weights are unequal because brightness perception/encoding is not a simple mean of three channels.

The self-test implements this manually and checks agreement with OpenCV BGR-to-gray conversion to within one code value.

## 11. RGB semantics vs OpenCV BGR

We usually describe color semantically as:

~~~text
R G B
~~~

OpenCV conventionally stores an 8-bit color image as:

~~~text
B G R
~~~

For a cv::Vec3b pixel:

~~~text
index 0 = blue
index 1 = green
index 2 = red
~~~

Confusing RGB terminology with BGR memory ordering is a common source of incorrect color interpretation.

## 12. HSV

HSV reorganizes color into:

- hue
- saturation
- value

It can be useful when chromatic information should be treated separately from intensity-like value.

HSV does not make color invariant to lighting. Hue also becomes unstable for weakly saturated pixels.

## 13. YUV

YUV-family representations separate luma-related information from chroma-related information.

This matters in image/video pipelines because luma and chroma are often sampled or compressed differently.

Embedded and automotive camera pipelines commonly encounter YUV-family formats before conversion into BGR/RGB.

## 14. Bayer and the image-signal processor

Many color sensors do not measure full R, G, and B at every photosite.

A color filter array can place different spectral filters on neighboring photosites. Demosaicing reconstructs missing components.

A camera ISP can also perform:

- black-level correction
- demosaicing
- white balance
- denoising
- color correction
- tone mapping
- sharpening
- gamma encoding

A final BGR pixel may therefore already be the result of substantial processing.

## 15. Gamma and nonlinear encoding

Stored display-oriented RGB values are often nonlinear with physical scene radiance.

That matters because averaging, blending, thresholding, and brightness changes can have different physical meaning in linear-light versus encoded space.

Module 02 explores gamma as a point operation.

## 16. Image coordinates

OpenCV typically treats:

~~~text
x = column
y = row
~~~

Element access is therefore conceptually:

~~~text
image.at(y, x)
~~~

This convention carries through filtering, features, calibration, stereo, optical flow, and tracking.

## 17. cv::Mat basics

A cv::Mat contains a matrix header describing data plus metadata such as:

- rows
- columns
- type
- channel count
- element size
- row step
- data pointer

A matrix header can reference shared data. Copying a cv::Mat header is not automatically a deep byte-for-byte image copy.

## 18. ROI behavior

A region of interest normally creates a view into parent storage.

The self-test checks that the ROI data pointer corresponds to the expected offset in the parent image.

This is efficient, but it also means writable ROI changes can affect the parent matrix.

## 19. Synthetic demonstration

The executable creates one deterministic scene containing:

- a smooth grayscale ramp
- high-frequency black/white stripes
- red, green, and blue patches

The same input demonstrates:

- quantization banding
- aliasing
- anti-alias filtering
- grayscale conversion
- BGR channel separation
- HSV encoding
- YUV encoding

## 20. Outputs

Run:

~~~bash
./build/cv9x_image_formation \
  --output-dir build/image_formation
~~~

Outputs:

~~~text
01_sensor_scene_bgr.png
02_naive_spatial_sampling.png
03_antialiased_sampling.png
04_grayscale.png
05_quantized.png
06_quantization_8_4_2_1_bit.png
07_bgr_channels.png
08_hsv_encoded.png
09_yuv_encoded.png
~~~

HSV/YUV matrices are written directly for channel inspection. Convert them back to BGR before treating them as normal display-color images.

## 21. Deterministic self-test

~~~bash
./build/cv9x_image_formation --self-test
ctest --test-dir build --output-on-failure
~~~

The self-test verifies:

- downsampled dimensions
- expected source lattice selection
- deterministic stripe aliasing
- 2-bit reconstruction levels
- 1-bit midpoint behavior
- 8-bit identity quantization
- manual grayscale conversion vs OpenCV
- HSV/YUV matrix shape/type
- synthetic-scene dimensions/type
- ROI memory-view semantics

## 22. Why image formation matters later

### Filtering

Sampling theory explains low-pass filtering before decimation.

### Gradients

Noise and quantization strongly affect derivatives.

### Features

Blur and resolution change corners and descriptors.

### Calibration

Optics and image coordinates become geometric parameters.

### Stereo

Subpixel disparity precision controls depth precision.

### Optical flow

Exposure, blur, and brightness consistency affect tracking.

### Recognition

Color space and preprocessing define model/classifier input distributions.

### Production systems

Pixel format, bit depth, row stride, timestamp, exposure, gain, and calibration become interface contracts.

## Common misconceptions

### More megapixels always mean more detail

Not if optics, noise, focus, or motion limit the signal first.

### Downsampling means deleting pixels

Deleting without appropriate low-pass filtering can alias.

### 8-bit storage means 8 bits of effective sensor dynamic range

It does not.

### RGB and BGR are interchangeable channel orders

They are not.

### HSV creates illumination invariance

It does not.

### A pixel is the object's true color

A pixel is the result of the whole imaging chain.

## Engineering notes

A real camera-frame interface should make metadata explicit:

~~~text
width
height
pixel format
bit depth
row stride
timestamp
exposure time
analog/digital gain
white-balance state
sensor/camera ID
calibration version
frame sequence number
~~~

Without that context, identical byte buffers can be interpreted incorrectly.

## Build

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_image_formation
~~~

## Demo

~~~bash
./build/cv9x_image_formation \
  --output-dir build/image_formation \
  --sample-factor 4 \
  --quant-bits 3
~~~

## Exercises

1. Change stripe frequency and sample factor until the alias pattern changes.
2. Compare nearest sampling with area interpolation.
3. Disable the Gaussian prefilter before downsampling.
4. Quantize a smooth ramp at 8, 6, 4, 3, 2, and 1 bits.
5. Compute quantization error statistics.
6. Modify HSV saturation and convert back to BGR.
7. Inspect Y, U, and V separately.
8. Create a synthetic Bayer mosaic and implement simple demosaicing.
9. Compare payload size for several cv::Mat element types.
10. Inspect ROI continuity and row stride.

## Legacy fundamentals material

The original optics/light notes and assets remain under the Fundametals directory.

They are intentionally preserved so historical links are not broken.

This canonical Module 01 supplies the course-facing sequence and the executable sampling/quantization/color demonstration.

## Classical course completion

With Modules 00 and 01 now canonical, this repository has a continuous classical-computer-vision path from environment setup and physical image formation through filtering, features, geometry, stereo, motion, recognition, classification, and tracking.

Learned/deep computer vision will continue in a separate ML Computer Vision repository rather than extending this classical roadmap.
