// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniel. Part of ImageProcessingFlex (ipf); see LICENSE.
// ipf geometry: flips/rotations, high-quality resizing, affine and perspective warps,
// remapping, transform estimation and Gaussian pyramids.
//
// Coordinate convention for warps: pixel (x, y) has its center at integer coordinates
// (OpenCV convention). Transform matrices map SOURCE coordinates to DESTINATION
// coordinates unless stated otherwise.
#pragma once

#include <array>

#include "core.hpp"
#include "filter.hpp"

namespace ipf {

// ---------------------------------------------------------------------------
// Exact transforms
// ---------------------------------------------------------------------------

/// Mirrors the image left-right.
template <class T>
Image<T> flipHorizontal(const Image<T>& src) {
    Image<T> out(src.width(), src.height(), src.channels());
    const int W = src.width(), cn = src.channels();
    for (int y = 0; y < src.height(); ++y)
        for (int x = 0; x < W; ++x) std::memcpy(out.ptr(W - 1 - x, y), src.ptr(x, y), sizeof(T) * cn);
    return out;
}

/// Mirrors the image top-bottom.
template <class T>
Image<T> flipVertical(const Image<T>& src) {
    Image<T> out(src.width(), src.height(), src.channels());
    for (int y = 0; y < src.height(); ++y)
        std::memcpy(out.row(src.height() - 1 - y), src.row(y), sizeof(T) * src.stride());
    return out;
}

/// Swaps rows and columns.
template <class T>
Image<T> transpose(const Image<T>& src) {
    Image<T> out(src.height(), src.width(), src.channels());
    const int cn = src.channels();
    for (int y = 0; y < src.height(); ++y)
        for (int x = 0; x < src.width(); ++x) std::memcpy(out.ptr(y, x), src.ptr(x, y), sizeof(T) * cn);
    return out;
}

/// Rotates by k * 90 degrees counter-clockwise (k may be negative).
template <class T>
Image<T> rotate90(const Image<T>& src, int k = 1) {
    k = ((k % 4) + 4) % 4;
    switch (k) {
        case 0: return src;
        case 1: return flipVertical(transpose(src));    // counter-clockwise
        case 2: return flipVertical(flipHorizontal(src));
        default: return flipHorizontal(transpose(src));  // clockwise
    }
}

/// Pads all four sides by n pixels.
template <class T>
Image<T> pad(const Image<T>& src, int n, BorderMode mode = BorderMode::Constant, T value = T{}) {
    return copyMakeBorder(src, n, n, n, n, mode, value);
}

// ---------------------------------------------------------------------------
// Resizing (separable convolution resampling, antialiased when downscaling)
// ---------------------------------------------------------------------------

enum class Interpolation {
    Nearest,   ///< nearest neighbour
    Bilinear,  ///< triangle filter (bilinear interpolation when upscaling)
    Bicubic,   ///< cubic convolution (a = -0.5 for resize, -0.75 for warps)
    Lanczos,   ///< Lanczos-3 windowed sinc (sharpest, best for downscaling photos)
    Area,      ///< box filter: pixel-area averaging when downscaling
    Hamming    ///< Hamming-windowed sinc with bilinear support
};

namespace detail {

inline double resampleSupport(Interpolation f) {
    switch (f) {
        case Interpolation::Area: return 0.5;
        case Interpolation::Bilinear: return 1.0;
        case Interpolation::Hamming: return 1.0;
        case Interpolation::Bicubic: return 2.0;
        case Interpolation::Lanczos: return 3.0;
        default: return 0.0;
    }
}

inline double sinc(double x) {
    if (x == 0.0) return 1.0;
    x *= kPi;
    return std::sin(x) / x;
}

inline double resampleFilter(Interpolation f, double x) {
    switch (f) {
        case Interpolation::Area: return (x > -0.5 && x <= 0.5) ? 1.0 : 0.0;
        case Interpolation::Bilinear: x = std::fabs(x); return x < 1.0 ? 1.0 - x : 0.0;
        case Interpolation::Hamming:
            x = std::fabs(x);
            if (x == 0.0) return 1.0;
            if (x >= 1.0) return 0.0;
            return std::sin(x * kPi) / (x * kPi) * (0.54 + 0.46 * std::cos(x * kPi));
        case Interpolation::Bicubic: {
            const double a = -0.5;
            x = std::fabs(x);
            if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
            if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
            return 0.0;
        }
        case Interpolation::Lanczos: return (x > -3.0 && x < 3.0) ? sinc(x) * sinc(x / 3.0) : 0.0;
        default: return 0.0;
    }
}

struct ResampleTaps {
    std::vector<int> start, count;
    std::vector<double> weights;  // count.size() * maxTaps
    int maxTaps = 0;
};

/// Per-output-pixel filter taps for resizing inSize -> outSize (Pillow's algorithm).
inline ResampleTaps resampleTaps(int inSize, int outSize, Interpolation f) {
    ResampleTaps t;
    const double scale = static_cast<double>(inSize) / outSize;
    const double filterscale = std::max(scale, 1.0);
    const double support = resampleSupport(f) * filterscale;
    t.maxTaps = static_cast<int>(std::ceil(support)) * 2 + 1;
    t.start.resize(static_cast<std::size_t>(outSize));
    t.count.resize(static_cast<std::size_t>(outSize));
    t.weights.assign(static_cast<std::size_t>(outSize) * t.maxTaps, 0.0);
    for (int o = 0; o < outSize; ++o) {
        const double center = (o + 0.5) * scale;
        const int xmin = std::max(static_cast<int>(center - support + 0.5), 0);
        const int xmax = std::min(static_cast<int>(center + support + 0.5), inSize);
        const int n = std::min(xmax - xmin, t.maxTaps);
        double* w = &t.weights[static_cast<std::size_t>(o) * t.maxTaps];
        double total = 0;
        for (int i = 0; i < n; ++i) {
            w[i] = resampleFilter(f, (i + xmin - center + 0.5) / filterscale);
            total += w[i];
        }
        if (total != 0)
            for (int i = 0; i < n; ++i) w[i] /= total;
        t.start[o] = xmin;
        t.count[o] = n;
    }
    return t;
}

}  // namespace detail

/// Resizes to newWidth x newHeight. Filters are evaluated on the source grid with
/// pixel-center alignment and widened when downscaling (antialiasing), matching
/// Pillow's Image.resize. Nearest picks floor((x + 0.5) * scale).
template <class T>
Image<T> resize(const Image<T>& src, int newWidth, int newHeight, Interpolation interp = Interpolation::Bilinear) {
    detail::require(newWidth > 0 && newHeight > 0, "resize: target size must be positive");
    detail::require(!src.empty(), "resize: empty source image");
    const int W = src.width(), H = src.height(), cn = src.channels();
    if (interp == Interpolation::Nearest) {
        Image<T> out(newWidth, newHeight, cn);
        std::vector<int> xs(static_cast<std::size_t>(newWidth));
        for (int x = 0; x < newWidth; ++x)
            xs[x] = std::min(W - 1, static_cast<int>((x + 0.5) * W / newWidth));
        parallelFor(0, newHeight, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y) {
                const int sy = std::min(H - 1, static_cast<int>((y + 0.5) * H / newHeight));
                for (int x = 0; x < newWidth; ++x) std::memcpy(out.ptr(x, y), src.ptr(xs[x], sy), sizeof(T) * cn);
            }
        }, static_cast<double>(newWidth) * cn);
        return out;
    }
    // Horizontal pass into a double buffer, then vertical pass.
    const bool doX = newWidth != W, doY = newHeight != H;
    std::vector<double> tmp(static_cast<std::size_t>(newWidth) * H * cn);
    if (doX) {
        const detail::ResampleTaps tx = detail::resampleTaps(W, newWidth, interp);
        parallelFor(0, H, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y) {
                const T* s = src.row(y);
                double* d = &tmp[static_cast<std::size_t>(y) * newWidth * cn];
                for (int x = 0; x < newWidth; ++x) {
                    const double* w = &tx.weights[static_cast<std::size_t>(x) * tx.maxTaps];
                    for (int c = 0; c < cn; ++c) {
                        double acc = 0;
                        for (int i = 0; i < tx.count[x]; ++i)
                            acc += w[i] * static_cast<double>(s[static_cast<std::size_t>(tx.start[x] + i) * cn + c]);
                        d[static_cast<std::size_t>(x) * cn + c] = acc;
                    }
                }
            }
        }, static_cast<double>(newWidth) * cn * tx.maxTaps);
    } else {
        for (std::size_t i = 0; i < src.sampleCount(); ++i) tmp[i] = static_cast<double>(src.data()[i]);
    }
    Image<T> out(newWidth, newHeight, cn);
    const std::size_t rowLen = static_cast<std::size_t>(newWidth) * cn;
    if (doY) {
        const detail::ResampleTaps ty = detail::resampleTaps(H, newHeight, interp);
        parallelFor(0, newHeight, [&](int y0, int y1) {
            const std::size_t len = rowLen;  // local copy keeps the inner loop vectorizable
            std::vector<double> accBuf(len);
            double* acc = accBuf.data();
            for (int y = y0; y < y1; ++y) {
                std::fill(acc, acc + len, 0.0);
                const double* w = &ty.weights[static_cast<std::size_t>(y) * ty.maxTaps];
                for (int i = 0; i < ty.count[y]; ++i) {
                    const double wi = w[i];
                    const double* r = &tmp[static_cast<std::size_t>(ty.start[y] + i) * len];
                    for (std::size_t k = 0; k < len; ++k) acc[k] += wi * r[k];
                }
                T* d = out.row(y);
                for (std::size_t k = 0; k < len; ++k) d[k] = saturate_cast<T>(acc[k]);
            }
        }, static_cast<double>(rowLen) * ty.maxTaps);
    } else {
        for (std::size_t i = 0; i < out.sampleCount(); ++i) out.data()[i] = saturate_cast<T>(tmp[i]);
    }
    return out;
}

/// Resizes by a scale factor (e.g. 0.5 halves both dimensions).
template <class T>
Image<T> rescale(const Image<T>& src, double factor, Interpolation interp = Interpolation::Bilinear) {
    detail::require(factor > 0, "rescale: factor must be positive");
    return resize(src, std::max(1, static_cast<int>(std::lround(src.width() * factor))),
                  std::max(1, static_cast<int>(std::lround(src.height() * factor))), interp);
}

/// Shrinks/enlarges the image to fit inside maxWidth x maxHeight while keeping the aspect ratio.
template <class T>
Image<T> resizeToFit(const Image<T>& src, int maxWidth, int maxHeight, Interpolation interp = Interpolation::Lanczos) {
    detail::require(maxWidth > 0 && maxHeight > 0, "resizeToFit: bounds must be positive");
    const double f = std::min(static_cast<double>(maxWidth) / src.width(), static_cast<double>(maxHeight) / src.height());
    const int w = clamp(static_cast<int>(std::lround(src.width() * f)), 1, maxWidth);
    const int h = clamp(static_cast<int>(std::lround(src.height() * f)), 1, maxHeight);
    return resize(src, w, h, interp);
}

// ---------------------------------------------------------------------------
// Point sampling (used by warps and remap)
// ---------------------------------------------------------------------------

namespace detail {

template <class T>
inline double fetch(const Image<T>& img, int x, int y, int c, BorderMode b, double bv) {
    const int xx = borderIndex(x, img.width(), b), yy = borderIndex(y, img.height(), b);
    return (xx < 0 || yy < 0) ? bv : static_cast<double>(img(xx, yy, c));
}

inline void cubicWeights(double t, double* w) {  // Keys cubic, a = -0.75 (OpenCV)
    const double a = -0.75;
    auto k = [a](double x) {
        x = std::fabs(x);
        if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
        if (x < 2) return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
        return 0.0;
    };
    w[0] = k(t + 1);
    w[1] = k(t);
    w[2] = k(1 - t);
    w[3] = k(2 - t);
}

}  // namespace detail

/// Samples channel c at a sub-pixel position (pixel centers at integer coordinates).
/// Supported interpolations: Nearest, Bilinear, Bicubic (others fall back to Bilinear).
template <class T>
double sample(const Image<T>& img, double x, double y, int c, Interpolation interp = Interpolation::Bilinear,
              BorderMode border = BorderMode::Constant, double borderValue = 0.0) {
    if (interp == Interpolation::Nearest) {
        return detail::fetch(img, static_cast<int>(std::floor(x + 0.5)), static_cast<int>(std::floor(y + 0.5)), c, border,
                             borderValue);
    }
    const double fx = std::floor(x), fy = std::floor(y);
    const int x0 = static_cast<int>(fx), y0 = static_cast<int>(fy);
    const double tx = x - fx, ty = y - fy;
    if (interp == Interpolation::Bicubic) {
        double wx[4], wy[4];
        detail::cubicWeights(tx, wx);
        detail::cubicWeights(ty, wy);
        double acc = 0;
        for (int j = 0; j < 4; ++j) {
            double row = 0;
            for (int i = 0; i < 4; ++i) row += wx[i] * detail::fetch(img, x0 - 1 + i, y0 - 1 + j, c, border, borderValue);
            acc += wy[j] * row;
        }
        return acc;
    }
    const double v00 = detail::fetch(img, x0, y0, c, border, borderValue);
    const double v10 = detail::fetch(img, x0 + 1, y0, c, border, borderValue);
    const double v01 = detail::fetch(img, x0, y0 + 1, c, border, borderValue);
    const double v11 = detail::fetch(img, x0 + 1, y0 + 1, c, border, borderValue);
    return (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty;
}

// ---------------------------------------------------------------------------
// Transform matrices
// ---------------------------------------------------------------------------

/// 2x3 affine matrix, row-major: [a b c; d e f] maps (x, y) -> (a x + b y + c, d x + e y + f).
using AffineMatrix = std::array<double, 6>;
/// 3x3 homography, row-major.
using Homography = std::array<double, 9>;

inline PointF applyAffine(const AffineMatrix& m, PointF p) {
    return {m[0] * p.x + m[1] * p.y + m[2], m[3] * p.x + m[4] * p.y + m[5]};
}
inline PointF applyHomography(const Homography& h, PointF p) {
    const double w = h[6] * p.x + h[7] * p.y + h[8];
    detail::require(w != 0, "applyHomography: point maps to infinity");
    return {(h[0] * p.x + h[1] * p.y + h[2]) / w, (h[3] * p.x + h[4] * p.y + h[5]) / w};
}

inline AffineMatrix invertAffine(const AffineMatrix& m) {
    const double det = m[0] * m[4] - m[1] * m[3];
    detail::require(std::fabs(det) > 1e-14, "invertAffine: matrix is singular");
    const double a = m[4] / det, b = -m[1] / det, d = -m[3] / det, e = m[0] / det;
    return {a, b, -(a * m[2] + b * m[5]), d, e, -(d * m[2] + e * m[5])};
}

inline Homography invertHomography(const Homography& h) {
    const double a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7], i = h[8];
    const double A = e * i - f * hh, B = -(d * i - f * g), C = d * hh - e * g;
    const double det = a * A + b * B + c * C;
    detail::require(std::fabs(det) > 1e-14, "invertHomography: matrix is singular");
    return {A / det, -(b * i - c * hh) / det, (b * f - c * e) / det,
            B / det, (a * i - c * g) / det,   -(a * f - c * d) / det,
            C / det, -(a * hh - b * g) / det, (a * e - b * d) / det};
}

/// Composes two affine maps: result(p) = a(b(p)).
inline AffineMatrix composeAffine(const AffineMatrix& a, const AffineMatrix& b) {
    return {a[0] * b[0] + a[1] * b[3], a[0] * b[1] + a[1] * b[4], a[0] * b[2] + a[1] * b[5] + a[2],
            a[3] * b[0] + a[4] * b[3], a[3] * b[1] + a[4] * b[4], a[3] * b[2] + a[4] * b[5] + a[5]};
}

/// Rotation by `angleDegrees` (counter-clockwise as displayed) about `center`, with uniform scale
/// (identical to OpenCV's getRotationMatrix2D).
inline AffineMatrix getRotationMatrix2D(PointF center, double angleDegrees, double scale = 1.0) {
    const double a = angleDegrees * detail::kPi / 180.0;
    const double alpha = scale * std::cos(a), beta = scale * std::sin(a);
    return {alpha, beta, (1 - alpha) * center.x - beta * center.y, -beta, alpha, beta * center.x + (1 - alpha) * center.y};
}

namespace detail {
/// Solves A x = b (n x n, row-major) by Gaussian elimination with partial pivoting.
inline std::vector<double> solveLinear(std::vector<double> A, std::vector<double> b) {
    const int n = static_cast<int>(b.size());
    for (int col = 0; col < n; ++col) {
        int piv = col;
        for (int r = col + 1; r < n; ++r)
            if (std::fabs(A[r * n + col]) > std::fabs(A[piv * n + col])) piv = r;
        if (std::fabs(A[piv * n + col]) < 1e-12) fail("solveLinear: degenerate point configuration");
        if (piv != col) {
            for (int k = 0; k < n; ++k) std::swap(A[col * n + k], A[piv * n + k]);
            std::swap(b[col], b[piv]);
        }
        for (int r = col + 1; r < n; ++r) {
            const double f = A[r * n + col] / A[col * n + col];
            for (int k = col; k < n; ++k) A[r * n + k] -= f * A[col * n + k];
            b[r] -= f * b[col];
        }
    }
    std::vector<double> x(static_cast<std::size_t>(n));
    for (int r = n - 1; r >= 0; --r) {
        double s = b[r];
        for (int k = r + 1; k < n; ++k) s -= A[r * n + k] * x[k];
        x[r] = s / A[r * n + r];
    }
    return x;
}
}  // namespace detail

/// Affine transform mapping three source points onto three destination points.
inline AffineMatrix getAffineTransform(const std::array<PointF, 3>& src, const std::array<PointF, 3>& dst) {
    std::vector<double> A(36, 0.0), b(6);
    for (int i = 0; i < 3; ++i) {
        const double row1[6] = {src[i].x, src[i].y, 1, 0, 0, 0}, row2[6] = {0, 0, 0, src[i].x, src[i].y, 1};
        std::copy(row1, row1 + 6, A.begin() + i * 12);
        std::copy(row2, row2 + 6, A.begin() + i * 12 + 6);
        b[i * 2] = dst[i].x;
        b[i * 2 + 1] = dst[i].y;
    }
    const std::vector<double> x = detail::solveLinear(A, b);
    return {x[0], x[1], x[2], x[3], x[4], x[5]};
}

/// Homography mapping four source points onto four destination points.
inline Homography getPerspectiveTransform(const std::array<PointF, 4>& src, const std::array<PointF, 4>& dst) {
    std::vector<double> A(64, 0.0), b(8);
    for (int i = 0; i < 4; ++i) {
        const double x = src[i].x, y = src[i].y, u = dst[i].x, v = dst[i].y;
        const double r1[8] = {x, y, 1, 0, 0, 0, -x * u, -y * u}, r2[8] = {0, 0, 0, x, y, 1, -x * v, -y * v};
        std::copy(r1, r1 + 8, A.begin() + i * 16);
        std::copy(r2, r2 + 8, A.begin() + i * 16 + 8);
        b[i * 2] = u;
        b[i * 2 + 1] = v;
    }
    const std::vector<double> h = detail::solveLinear(A, b);
    return {h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0};
}

// ---------------------------------------------------------------------------
// Warps
// ---------------------------------------------------------------------------

/// Remaps: out(x, y) = src(mapX(x, y), mapY(x, y)). Maps are single-channel float images
/// defining the output size.
template <class T>
Image<T> remap(const Image<T>& src, const ImageF& mapX, const ImageF& mapY, Interpolation interp = Interpolation::Bilinear,
               BorderMode border = BorderMode::Constant, double borderValue = 0.0) {
    detail::require(mapX.sameShape(mapY) && mapX.channels() == 1, "remap: maps must be equal-size single-channel images");
    const int cn = src.channels();
    Image<T> out(mapX.width(), mapX.height(), cn);
    parallelFor(0, out.height(), [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < out.width(); ++x)
                for (int c = 0; c < cn; ++c)
                    out(x, y, c) = saturate_cast<T>(sample(src, mapX(x, y), mapY(x, y), c, interp, border, borderValue));
    }, static_cast<double>(out.width()) * cn * 16);
    return out;
}

/// Affine warp. `m` maps source to destination coordinates (set inverseMap = true if it maps
/// destination to source). Output size defaults to the source size.
template <class T>
Image<T> warpAffine(const Image<T>& src, const AffineMatrix& m, Size dsize = Size(), Interpolation interp = Interpolation::Bilinear,
                    BorderMode border = BorderMode::Constant, double borderValue = 0.0, bool inverseMap = false) {
    if (dsize.width <= 0 || dsize.height <= 0) dsize = src.size();
    const AffineMatrix inv = inverseMap ? m : invertAffine(m);
    const int cn = src.channels();
    Image<T> out(dsize.width, dsize.height, cn);
    parallelFor(0, dsize.height, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < dsize.width; ++x) {
                const double sx = inv[0] * x + inv[1] * y + inv[2], sy = inv[3] * x + inv[4] * y + inv[5];
                for (int c = 0; c < cn; ++c)
                    out(x, y, c) = saturate_cast<T>(sample(src, sx, sy, c, interp, border, borderValue));
            }
    }, static_cast<double>(dsize.width) * cn * 16);
    return out;
}

/// Perspective warp. `h` maps source to destination coordinates (inverseMap = true if it maps
/// destination to source). Output size defaults to the source size.
template <class T>
Image<T> warpPerspective(const Image<T>& src, const Homography& h, Size dsize = Size(),
                         Interpolation interp = Interpolation::Bilinear, BorderMode border = BorderMode::Constant,
                         double borderValue = 0.0, bool inverseMap = false) {
    if (dsize.width <= 0 || dsize.height <= 0) dsize = src.size();
    const Homography inv = inverseMap ? h : invertHomography(h);
    const int cn = src.channels();
    Image<T> out(dsize.width, dsize.height, cn);
    parallelFor(0, dsize.height, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < dsize.width; ++x) {
                const double w = inv[6] * x + inv[7] * y + inv[8];
                const bool valid = std::fabs(w) > 1e-12;
                const double sx = valid ? (inv[0] * x + inv[1] * y + inv[2]) / w : -1e9;
                const double sy = valid ? (inv[3] * x + inv[4] * y + inv[5]) / w : -1e9;
                for (int c = 0; c < cn; ++c)
                    out(x, y, c) = saturate_cast<T>(sample(src, sx, sy, c, interp, border, borderValue));
            }
    }, static_cast<double>(dsize.width) * cn * 16);
    return out;
}

/// Rotates the image by `angleDegrees` counter-clockwise about its center. With expand = true
/// the canvas grows so that the whole rotated image is visible.
template <class T>
Image<T> rotate(const Image<T>& src, double angleDegrees, Interpolation interp = Interpolation::Bilinear,
                bool expand = false, BorderMode border = BorderMode::Constant, double borderValue = 0.0) {
    const PointF c((src.width() - 1) / 2.0, (src.height() - 1) / 2.0);
    AffineMatrix m = getRotationMatrix2D(c, angleDegrees);
    Size dsize = src.size();
    if (expand) {
        const double a = angleDegrees * detail::kPi / 180.0;
        const double cs = std::fabs(std::cos(a)), sn = std::fabs(std::sin(a));
        dsize.width = static_cast<int>(std::ceil(src.width() * cs + src.height() * sn - 1e-9));
        dsize.height = static_cast<int>(std::ceil(src.width() * sn + src.height() * cs - 1e-9));
        m[2] += (dsize.width - 1) / 2.0 - c.x;
        m[5] += (dsize.height - 1) / 2.0 - c.y;
    }
    return warpAffine(src, m, dsize, interp, border, borderValue);
}

// ---------------------------------------------------------------------------
// Gaussian pyramids
// ---------------------------------------------------------------------------

/// Blurs with the 5-tap binomial kernel and drops every other row/column.
/// Output size ((W + 1) / 2, (H + 1) / 2), as OpenCV's pyrDown.
template <class T>
Image<T> pyrDown(const Image<T>& src) {
    const std::vector<double> k = {1 / 16.0, 4 / 16.0, 6 / 16.0, 4 / 16.0, 1 / 16.0};
    const ImageD blurred = sepFilter2D<double>(src, k, k, BorderMode::Reflect101);
    const int W = (src.width() + 1) / 2, H = (src.height() + 1) / 2, cn = src.channels();
    Image<T> out(W, H, cn);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < cn; ++c) out(x, y, c) = saturate_cast<T>(blurred(2 * x, 2 * y, c));
    return out;
}

/// Doubles the size: zero-insertion followed by the binomial kernel scaled by 4.
template <class T>
Image<T> pyrUp(const Image<T>& src) {
    const int W = src.width() * 2, H = src.height() * 2, cn = src.channels();
    ImageD up(W, H, cn, 0.0);
    for (int y = 0; y < src.height(); ++y)
        for (int x = 0; x < src.width(); ++x)
            for (int c = 0; c < cn; ++c) up(2 * x, 2 * y, c) = static_cast<double>(src(x, y, c));
    const std::vector<double> k = {1 / 8.0, 4 / 8.0, 6 / 8.0, 4 / 8.0, 1 / 8.0};
    return sepFilter2D<T>(up, k, k, BorderMode::Reflect101);
}

/// Builds a Gaussian pyramid with `levels` images (level 0 = the source).
template <class T>
std::vector<Image<T>> buildPyramid(const Image<T>& src, int levels) {
    detail::require(levels >= 1, "buildPyramid: levels must be >= 1");
    std::vector<Image<T>> pyr{src};
    while (static_cast<int>(pyr.size()) < levels && pyr.back().width() > 1 && pyr.back().height() > 1)
        pyr.push_back(pyrDown(pyr.back()));
    return pyr;
}

}  // namespace ipf
