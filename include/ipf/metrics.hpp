// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniel. Part of ImageProcessingFlex (ipf); see LICENSE.
// ipf metrics: image quality and similarity measures (MSE, PSNR, MAE, SSIM),
// entropy and a sharpness (focus) measure.
#pragma once

#include <array>
#include <limits>

#include "core.hpp"
#include "filter.hpp"
#include "pixel_ops.hpp"

namespace ipf {

/// Mean squared error over all samples.
template <class T>
double mse(const Image<T>& a, const Image<T>& b) {
    detail::require(a.sameShape(b) && !a.empty(), "mse: images must be non-empty and have the same shape");
    double s = 0;
    for (std::size_t i = 0; i < a.sampleCount(); ++i) {
        const double d = static_cast<double>(a.data()[i]) - static_cast<double>(b.data()[i]);
        s += d * d;
    }
    return s / static_cast<double>(a.sampleCount());
}

/// Mean absolute error over all samples.
template <class T>
double mae(const Image<T>& a, const Image<T>& b) {
    detail::require(a.sameShape(b) && !a.empty(), "mae: images must be non-empty and have the same shape");
    double s = 0;
    for (std::size_t i = 0; i < a.sampleCount(); ++i)
        s += std::fabs(static_cast<double>(a.data()[i]) - static_cast<double>(b.data()[i]));
    return s / static_cast<double>(a.sampleCount());
}

/// Peak signal-to-noise ratio in dB (peak defaults to the type's maximum: 255 for uint8,
/// 1 for float). Identical images give +infinity.
template <class T>
double psnr(const Image<T>& a, const Image<T>& b, double peak = PixelTraits<T>::max()) {
    const double m = mse(a, b);
    return m == 0 ? std::numeric_limits<double>::infinity() : 10.0 * std::log10(peak * peak / m);
}

/// Structural similarity (Wang et al. 2004): 11x11 Gaussian window (sigma 1.5), K1 = 0.01,
/// K2 = 0.03, dynamic range = the type's maximum. Multi-channel images return the mean over
/// channels. The mean is taken over the "valid" region (windows fully inside the image);
/// if `ssimMap` is given it receives the full-size per-pixel SSIM map (one channel per input channel).
template <class T>
double ssim(const Image<T>& a, const Image<T>& b, ImageF* ssimMap = nullptr) {
    detail::require(a.sameShape(b) && !a.empty(), "ssim: images must be non-empty and have the same shape");
    const double L = PixelTraits<T>::max(), C1 = (0.01 * L) * (0.01 * L), C2 = (0.03 * L) * (0.03 * L);
    const ImageD x = a.template convertTo<double>(), y = b.template convertTo<double>();
    ImageD xx(x.width(), x.height(), x.channels()), yy = xx, xy = xx;
    for (std::size_t i = 0; i < x.sampleCount(); ++i) {
        xx.data()[i] = x.data()[i] * x.data()[i];
        yy.data()[i] = y.data()[i] * y.data()[i];
        xy.data()[i] = x.data()[i] * y.data()[i];
    }
    auto g = [](const ImageD& im) { return gaussianBlur(im, 1.5, 1.5, 11, BorderMode::Reflect101); };
    const ImageD mx = g(x), my = g(y), sxx = g(xx), syy = g(yy), sxy = g(xy);
    ImageF map(x.width(), x.height(), x.channels());
    for (std::size_t i = 0; i < x.sampleCount(); ++i) {
        const double ux = mx.data()[i], uy = my.data()[i];
        const double vx = sxx.data()[i] - ux * ux, vy = syy.data()[i] - uy * uy, cxy = sxy.data()[i] - ux * uy;
        map.data()[i] = static_cast<float>(((2 * ux * uy + C1) * (2 * cxy + C2)) / ((ux * ux + uy * uy + C1) * (vx + vy + C2)));
    }
    const int r = 5;
    const bool valid = x.width() > 2 * r && x.height() > 2 * r;
    const Rect roi = valid ? Rect(r, r, x.width() - 2 * r, x.height() - 2 * r) : x.rect();
    double s = 0;
    for (int yy0 = roi.y; yy0 < roi.bottom(); ++yy0)
        for (int xx0 = roi.x; xx0 < roi.right(); ++xx0)
            for (int c = 0; c < x.channels(); ++c) s += map(xx0, yy0, c);
    if (ssimMap) *ssimMap = std::move(map);
    return s / (static_cast<double>(roi.area()) * x.channels());
}

/// Shannon entropy (bits) of the 256-bin histogram of a uint8 image (all channels pooled).
inline double entropy(const ImageU8& img) {
    std::array<std::uint64_t, 256> h{};
    for (std::uint8_t v : img) ++h[v];
    double e = 0;
    const double n = static_cast<double>(img.sampleCount());
    for (auto c : h)
        if (c) {
            const double p = static_cast<double>(c) / n;
            e -= p * std::log2(p);
        }
    return e;
}

/// Sharpness / focus measure: variance of the Laplacian of the grayscale image
/// (low values indicate a blurry image). Multi-channel input uses the channel mean.
template <class T>
double sharpness(const Image<T>& img) {
    ImageF gray(img.width(), img.height(), 1);
    const int cn = img.channels();
    for (std::size_t i = 0; i < gray.pixelCount(); ++i) {
        double s = 0;
        for (int c = 0; c < cn; ++c) s += static_cast<double>(img.data()[i * cn + c]);
        gray.data()[i] = static_cast<float>(s / cn);
    }
    const double sd = stddev(laplacian(gray))[0];
    return sd * sd;
}

}  // namespace ipf
