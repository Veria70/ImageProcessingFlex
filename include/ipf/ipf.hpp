// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniel. Part of ImageProcessingFlex (ipf); see LICENSE.
// ImageProcessingFlex (ipf) - a header-only, dependency-free C++17 image processing library.
//
// Include this single header to get everything:
//
//   #include <ipf/ipf.hpp>
//
//   ipf::ImageU8 img = ipf::loadImage("photo.jpg");
//   ipf::ImageU8 small = ipf::resize(img, img.width() / 2, img.height() / 2, ipf::Interpolation::Lanczos);
//   ipf::ImageU8 edges = ipf::canny(ipf::toGray(small), 50, 150);
//   ipf::saveImage(edges, "edges.png");
//
// Modules (each can also be included on its own):
//   core.hpp        Image<T> container, Point/Rect/Size, borders, conversions, statistics
//   io.hpp          load/save PNG/APNG, JPEG (incl. 12-bit, lossless, arithmetic), BMP, TGA, PNM, PFM
//                   with metadata (ICC, EXIF orientation, DPI, text)
//   icc.hpp         ICC color management (matrix/TRC, gray and LUT/CMYK profiles)
//   color.hpp       Gray, HSV, HSL, YCbCr, XYZ, Lab conversions, adjustments, colormaps
//   pixel_ops.hpp   arithmetic, blending, masks, thresholds, histograms, equalization, CLAHE
//   filter.hpp      convolution, box/Gaussian/median/bilateral, Sobel/Scharr/Laplacian, sharpening
//   geometry.hpp    flip/rotate, resize, affine/perspective warps, remap, pyramids
//   morphology.hpp  erode/dilate/open/close/..., thinning, hole filling
//   analysis.hpp    connected components, contours, shape descriptors, moments, distance transform
//   features.hpp    Canny, Harris/Shi-Tomasi, Hough lines/segments/circles, template matching, LBP, NMS
//   fft.hpp         FFT (any size), DFT/IDFT, FFT filtering, phase correlation
//   parallel.hpp    setNumThreads / parallelFor (multithreading used by all heavy algorithms)
//   draw.hpp        lines, shapes, polygons, markers, text
//   metrics.hpp     MSE, PSNR, SSIM, entropy, sharpness
#pragma once

#include "analysis.hpp"
#include "color.hpp"
#include "core.hpp"
#include "draw.hpp"
#include "features.hpp"
#include "fft.hpp"
#include "filter.hpp"
#include "geometry.hpp"
#include "icc.hpp"
#include "io.hpp"
#include "metrics.hpp"
#include "morphology.hpp"
#include "parallel.hpp"
#include "pixel_ops.hpp"
